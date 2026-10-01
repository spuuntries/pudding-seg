from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class NCAParams(NamedTuple):
    w1: jax.Array  # (hidden_dim, in_channels * 3, 1, 1)
    b1: jax.Array  # (hidden_dim, 1, 1)
    w2: jax.Array  # (channels, hidden_dim, 1, 1)
    b2: jax.Array  # (channels, 1, 1)


def make_sobel_filters(channels: int, dtype=jnp.float32) -> jax.Array:
    """Fixed depthwise 3x3 Sobel-x, Sobel-y, Identity perception filters."""
    sobel_x = jnp.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=dtype) / 8.0
    sobel_y = jnp.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=dtype) / 8.0
    ident = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=dtype)

    # Stack: (3, 1, 3, 3)
    filters = jnp.stack([ident, sobel_x, sobel_y], axis=0)[:, None, :, :]
    # Tile across channels: (3 * channels, 1, 3, 3)
    # Using depthwise conv with feature_group_count = channels
    filters = jnp.repeat(filters, channels, axis=1)  # (3, channels, 3, 3)
    # Reshape so for each channel we have ident, sobel_x, sobel_y
    filters = jnp.concatenate([filters[0], filters[1], filters[2]], axis=0)[:, None, :, :]
    return filters


def perceive(z: jax.Array) -> jax.Array:
    """Depthwise 2D convolution extracting (identity, sobel_x, sobel_y) per channel."""
    channels = z.shape[1]
    # Simple manual depthwise roll for pure speed and zero conv padding quirks
    # z: (B, C, H, W)
    z_pad = jnp.pad(z, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="wrap")

    # 3x3 neighborhood slices
    tl = z_pad[:, :, :-2, :-2]
    tc = z_pad[:, :, :-2, 1:-1]
    tr = z_pad[:, :, :-2, 2:]

    ml = z_pad[:, :, 1:-1, :-2]
    mc = z[:, :, :, :]
    mr = z_pad[:, :, 1:-1, 2:]

    bl = z_pad[:, :, 2:, :-2]
    bc = z_pad[:, :, 2:, 1:-1]
    br = z_pad[:, :, 2:, 2:]

    dx = (-tl + tr - 2.0 * ml + 2.0 * mr - bl + br) / 8.0
    dy = (-tl - 2.0 * tc - tr + bl + 2.0 * bc + br) / 8.0

    return jnp.concatenate([mc, dx, dy], axis=1)  # (B, 3*C, H, W)


def init_nca_params(
    key: jax.Array,
    *,
    channels: int = 16,
    hidden_dim: int = 64,
    dtype=jnp.float32,
) -> NCAParams:
    k1, k2 = jax.random.split(key)
    in_dim = channels * 3

    # LeCun / He init for layer 1
    std1 = 1.0 / math.sqrt(in_dim)
    w1 = jax.random.normal(k1, (hidden_dim, in_dim, 1, 1), dtype=dtype) * std1
    b1 = jnp.zeros((hidden_dim, 1, 1), dtype=dtype)

    # Zero init for layer 2 so initial cell update is a gentle perturbation
    w2 = jnp.zeros((channels, hidden_dim, 1, 1), dtype=dtype)
    b2 = jnp.zeros((channels, 1, 1), dtype=dtype)

    return NCAParams(w1=w1, b1=b1, w2=w2, b2=b2)


def nca_step(z: jax.Array, params: NCAParams, step_size: float = 1.0) -> jax.Array:
    """Single NCA recurrence: z_{t+1} = z_t + step_size * Delta_z."""
    p = perceive(z)  # (B, 3*C, H, W)
    # 1x1 conv layer 1: tensordot over channel dimension
    # p: (B, 3C, H, W), w1: (H_dim, 3C, 1, 1)
    h = jnp.tensordot(p, params.w1[:, :, 0, 0], axes=([1], [1]))  # (B, H, W, H_dim)
    h = jnp.transpose(h, (0, 3, 1, 2)) + params.b1
    h = jax.nn.relu(h)

    # 1x1 conv layer 2
    delta = jnp.tensordot(h, params.w2[:, :, 0, 0], axes=([1], [1]))
    delta = jnp.transpose(delta, (0, 3, 1, 2)) + params.b2

    return z + step_size * delta


def nca_delta(z: jax.Array, params: NCAParams) -> jax.Array:
    """Computes Delta_z directly."""
    p = perceive(z)
    h = jnp.tensordot(p, params.w1[:, :, 0, 0], axes=([1], [1]))
    h = jnp.transpose(h, (0, 3, 1, 2)) + params.b1
    h = jax.nn.relu(h)

    delta = jnp.tensordot(h, params.w2[:, :, 0, 0], axes=([1], [1]))
    delta = jnp.transpose(delta, (0, 3, 1, 2)) + params.b2
    return delta


# =========================================================================
# Deep Equilibrium (DEQ) PC-ALM formulation
# Stationary fixed point condition: Delta_z(z*) = 0
# Objective: min 0.5 * || z*[:, :C_out] - y ||^2  s.t.  Delta_z(z*) = 0
# =========================================================================

def deq_al_energy(
    params: NCAParams,
    z_eq: jax.Array,
    dual_eq: jax.Array,
    y: jax.Array,
    rho: float = 1.0,
    out_channels: int = 1,
) -> jax.Array:
    batch_size = z_eq.shape[0]
    # Supervised loss: first out_channels match target
    pred_y = z_eq[:, :out_channels]
    loss_sup = 0.5 * jnp.sum((pred_y - y) ** 2) / batch_size

    # Equilibrium constraint: Delta_z = 0
    delta = nca_delta(z_eq, params)
    shifted = delta + dual_eq / rho
    loss_constraint = 0.5 * rho * jnp.sum(shifted * shifted) / batch_size

    return loss_sup + loss_constraint


def run_deq_pcalm_settle(
    params: NCAParams,
    z_init: jax.Array,
    y: jax.Array,
    *,
    steps: int = 10,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    out_channels: int = 1,
) -> tuple[jax.Array, jax.Array]:
    """Primal-dual settling directly at the DEQ equilibrium point."""
    z = z_init
    duals = jnp.zeros_like(z)

    def inner_step(z_curr, dual_curr):
        def energy(zc):
            return deq_al_energy(params, zc, dual_curr, y, rho=rho, out_channels=out_channels)

        grad_fn = jax.grad(energy)

        def step(zc, _):
            return zc - state_lr * grad_fn(zc), None

        z_next, _ = jax.lax.scan(step, z_curr, xs=None, length=inner_steps)
        return z_next

    def outer(carry, _):
        zc, dualc = carry
        zc = inner_step(zc, dualc)
        r = nca_delta(zc, params)
        dual_next = dualc + alpha * r
        return (zc, dual_next), None

    if steps > 1:
        (z, duals), _ = jax.lax.scan(outer, (z, duals), xs=None, length=steps - 1)

    z = inner_step(z, duals)
    return z, duals


def compute_nca_grads(
    params: NCAParams,
    z_init: jax.Array,
    y: jax.Array,
    *,
    steps: int = 8,
    inner_steps: int = 2,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    out_channels: int = 1,
) -> tuple[NCAParams, float, jax.Array]:
    """Computes parameter gradients via DEQ PC-ALM equilibrium and returns (grads, loss, z_eq)."""
    batch_size = y.shape[0]

    # Primal-dual solve for DEQ fixed point
    z_eq, dual_eq = run_deq_pcalm_settle(
        params,
        z_init,
        y,
        steps=steps,
        inner_steps=inner_steps,
        state_lr=state_lr,
        rho=rho,
        alpha=alpha,
        out_channels=out_channels,
    )

    loss = 0.5 * jnp.sum((z_eq[:, :out_channels] - y) ** 2) / batch_size

    # Detach states at equilibrium
    z_eq_stop = jax.lax.stop_gradient(z_eq)
    dual_eq_stop = jax.lax.stop_gradient(dual_eq)

    # Parameter gradient on the equilibrium augmented Lagrangian
    def p_loss(p):
        return deq_al_energy(p, z_eq_stop, dual_eq_stop, y, rho=rho, out_channels=out_channels)

    grads = jax.grad(p_loss)(params)
    return grads, loss, z_eq


# =========================================================================
# Trajectory NCA PC-ALM formulation (Time-unrolled cellular dynamics)
# z_0 = seed -> z_1 -> ... -> z_T
# Supervised loss only at final state z_T
# PC-ALM duals lambda_t carry temporal credit without unrolling autograd
# =========================================================================

def forward_trajectory(z_0: jax.Array, params: NCAParams, steps: int) -> list[jax.Array]:
    """Autonomous feedforward recurrence of the NCA from seed z_0."""
    states: list[jax.Array] = []
    z_curr = z_0
    for _ in range(steps):
        z_curr = nca_step(z_curr, params)
        states.append(z_curr)
    return states


def trajectory_al_energy(
    params: NCAParams,
    z_0: jax.Array,
    y: jax.Array,
    states: list[jax.Array],
    duals: list[jax.Array],
    rho: float = 1.0,
    out_channels: int = 1,
) -> jax.Array:
    batch_size = z_0.shape[0]
    residuals: list[jax.Array] = []
    z_prev = z_0
    for z_curr in states:
        pred = nca_step(z_prev, params)
        residuals.append(z_curr - pred)
        z_prev = z_curr

    # Supervised loss ONLY on final state
    pred_y = states[-1][:, :out_channels]
    loss_sup = 0.5 * jnp.sum((pred_y - y) ** 2) / batch_size

    # Constraints across all time steps
    shifted = [r + lam / rho for lam, r in zip(duals, residuals)]
    loss_constraints = 0.5 * rho * sum(jnp.sum(s * s) / batch_size for s in shifted)
    return loss_sup + loss_constraints


def run_trajectory_pcalm(
    params: NCAParams,
    z_0: jax.Array,
    y: jax.Array,
    *,
    steps: int = 16,
    budget: int = 6,
    inner_steps: int = 2,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    out_channels: int = 1,
) -> tuple[list[jax.Array], list[jax.Array]]:
    states = forward_trajectory(z_0, params, steps)
    duals = [jnp.zeros_like(s) for s in states]

    def solve_inner(states_curr, duals_curr):
        def energy(s):
            return trajectory_al_energy(params, z_0, y, s, duals_curr, rho=rho, out_channels=out_channels)

        grad_fn = jax.grad(energy)

        def step(s, _):
            grads = grad_fn(s)
            s_next = [si - state_lr * gi for si, gi in zip(s, grads)]
            return s_next, None

        states_next, _ = jax.lax.scan(step, states_curr, xs=None, length=inner_steps)
        return states_next

    def outer(carry, _):
        s_c, duals_c = carry
        s_c = solve_inner(s_c, duals_c)
        res = []
        z_prev = z_0
        for z_curr in s_c:
            pred = nca_step(z_prev, params)
            res.append(z_curr - pred)
            z_prev = z_curr
        duals_next = [lam + alpha * r for lam, r in zip(duals_c, res)]
        return (s_c, duals_next), None

    if budget > 1:
        (states, duals), _ = jax.lax.scan(outer, (states, duals), xs=None, length=budget - 1)

    states = solve_inner(states, duals)
    return states, duals


def compute_trajectory_grads(
    params: NCAParams,
    z_0: jax.Array,
    y: jax.Array,
    *,
    steps: int = 16,
    budget: int = 6,
    inner_steps: int = 2,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    out_channels: int = 1,
) -> tuple[NCAParams, jax.Array, jax.Array]:
    batch_size = z_0.shape[0]
    states, duals = run_trajectory_pcalm(
        params, z_0, y,
        steps=steps, budget=budget, inner_steps=inner_steps,
        state_lr=state_lr, rho=rho, alpha=alpha, out_channels=out_channels,
    )

    loss = 0.5 * jnp.sum((states[-1][:, :out_channels] - y) ** 2) / batch_size

    states_stop = jax.tree_util.tree_map(jax.lax.stop_gradient, states)
    duals_stop = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)

    def p_loss(p):
        return trajectory_al_energy(p, z_0, y, states_stop, duals_stop, rho=rho, out_channels=out_channels)

    grads = jax.grad(p_loss)(params)
    return grads, loss, states[-1]

