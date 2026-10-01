from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class ConvLayer(NamedTuple):
    W: jax.Array  # (C_out, C_in, K, K)
    b: jax.Array  # (C_out, 1, 1)


Params = list[ConvLayer]


def conv2d(x: jax.Array, W: jax.Array) -> jax.Array:
    """Standard 2D convolution with SAME padding, NCHW layout."""
    return jax.lax.conv_general_dilated(
        x,
        W,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )


def init_conv_params(
    key: jax.Array,
    *,
    depth: int,
    channels: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    dtype=jnp.float32,
) -> Params:
    keys = jax.random.split(key, depth)
    params: Params = []
    k = kernel_size

    for layer_ix in range(depth):
        c_in = in_channels if layer_ix == 0 else channels
        c_out = out_channels if layer_ix == depth - 1 else channels
        k1, k2 = jax.random.split(keys[layer_ix])

        # Standard He / LeCun init scale
        fan_in = c_in * k * k
        if layer_ix == 0:
            std = 1.0 / math.sqrt(fan_in)
        elif layer_ix == depth - 1:
            std = 1.0 / math.sqrt(fan_in)
        else:
            # ResNet / deep scale to keep variance 1 across depth
            std = 1.0 / math.sqrt(fan_in * depth)

        W = jax.random.normal(k1, (c_out, c_in, k, k), dtype=dtype) * std
        b = jnp.zeros((c_out, 1, 1), dtype=dtype)
        params.append(ConvLayer(W=W, b=b))

    return params


def skip_mask(depth: int) -> tuple[bool, ...]:
    """Skip connections on all hidden layers."""
    if depth <= 2:
        return tuple([False] * depth)
    return tuple([False] + [True] * (depth - 2) + [False])


def block_pred(
    layer: ConvLayer,
    skip: bool,
    z_prev: jax.Array,
    phi: Callable[[jax.Array], jax.Array],
    *,
    is_first: bool,
) -> jax.Array:
    inp = z_prev if is_first else phi(z_prev)
    pred = conv2d(inp, layer.W) + layer.b
    if skip:
        pred = pred + z_prev
    return pred


def forward(
    params: Params,
    skips: tuple[bool, ...],
    x: jax.Array,
    phi: Callable[[jax.Array], jax.Array],
) -> tuple[list[jax.Array], jax.Array]:
    """Feedforward pass returning (all hidden states, final output)."""
    states: list[jax.Array] = []
    z_prev = x
    depth = len(params)
    for i, (layer, skip) in enumerate(zip(params, skips)):
        z_curr = block_pred(layer, skip, z_prev, phi, is_first=(i == 0))
        if i < depth - 1:
            states.append(z_curr)
        z_prev = z_curr
    return states, z_prev


def constraint_residuals(
    params: Params,
    skips: tuple[bool, ...],
    x: jax.Array,
    states: list[jax.Array],
    phi: Callable[[jax.Array], jax.Array],
) -> list[jax.Array]:
    residuals: list[jax.Array] = []
    z_prev = x
    for i, (layer, skip, z_curr) in enumerate(zip(params[:-1], skips[:-1], states)):
        pred = block_pred(layer, skip, z_prev, phi, is_first=(i == 0))
        residuals.append(z_curr - pred)
        z_prev = z_curr
    return residuals


def al_energy(
    params: Params,
    skips: tuple[bool, ...],
    x: jax.Array,
    y: jax.Array,
    states: list[jax.Array],
    duals: list[jax.Array],
    rho: float,
    phi: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    batch_size = x.shape[0]
    residuals = constraint_residuals(params, skips, x, states, phi)

    # Supervised loss at top layer
    y_pred = block_pred(params[-1], skips[-1], states[-1], phi, is_first=False)
    loss_sup = 0.5 * jnp.sum((y_pred - y) ** 2) / batch_size

    # Augmented Lagrangian constraints: lambda^T r + (rho/2) * ||r||^2 = (rho/2) * ||r + lambda/rho||^2 - ||lambda||^2 / (2*rho)
    shifted = [r + lam / rho for lam, r in zip(duals, residuals)]
    loss_constraints = 0.5 * rho * sum(jnp.sum(s * s) / batch_size for s in shifted)

    return loss_sup + loss_constraints


def run_pcalm_inference(
    params: Params,
    skips: tuple[bool, ...],
    x: jax.Array,
    y: jax.Array,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    budget: int,
    inner_steps: int,
    phi: Callable[[jax.Array], jax.Array],
) -> tuple[list[jax.Array], list[jax.Array]]:
    """Primal-dual settling on activations and multipliers."""
    # Warm start: forward pass activations
    states, _ = forward(params, skips, x, phi)
    duals = [jnp.zeros_like(s) for s in states]

    def solve_inner(states_curr, duals_curr):
        def inner_energy(s):
            return al_energy(params, skips, x, y, s, duals_curr, rho, phi)

        grad_fn = jax.grad(inner_energy)

        def step(s, _):
            grads = grad_fn(s)
            s_next = [si - state_lr * gi for si, gi in zip(s, grads)]
            return s_next, None

        states_next, _ = jax.lax.scan(step, states_curr, xs=None, length=inner_steps)
        return states_next

    def outer(carry, _):
        s_c, duals_c = carry
        s_c = solve_inner(s_c, duals_c)
        r = constraint_residuals(params, skips, x, s_c, phi)
        duals_next = [lam + alpha * ri for lam, ri in zip(duals_c, r)]
        return (s_c, duals_next), None

    if budget > 1:
        (states, duals_before), _ = jax.lax.scan(outer, (states, duals), xs=None, length=budget - 1)
    else:
        duals_before = duals

    states = solve_inner(states, duals_before)
    return states, duals_before


def run_pc_inference(
    params: Params,
    skips: tuple[bool, ...],
    x: jax.Array,
    y: jax.Array,
    *,
    state_lr: float,
    rho: float,
    steps: int,
    phi: Callable[[jax.Array], jax.Array],
) -> list[jax.Array]:
    """Standard PC (no duals, alpha=0, pure energy relaxation)."""
    states, _ = forward(params, skips, x, phi)
    zero_duals = [jnp.zeros_like(s) for s in states]

    def energy(s):
        return al_energy(params, skips, x, y, s, zero_duals, rho, phi)

    grad_fn = jax.grad(energy)

    def step(s, _):
        grads = grad_fn(s)
        s_next = [si - state_lr * gi for si, gi in zip(s, grads)]
        return s_next, None

    states, _ = jax.lax.scan(step, states, xs=None, length=steps)
    return states


def compute_grads(
    method: str,  # "bp", "pc", "pcalm"
    params: Params,
    skips: tuple[bool, ...],
    x: jax.Array,
    y: jax.Array,
    *,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    budget: int = 5,
    inner_steps: int = 2,
    phi: Callable[[jax.Array], jax.Array] = jax.nn.relu,
) -> tuple[Params, float]:
    batch_size = x.shape[0]
    _, y_pred = forward(params, skips, x, phi)
    loss = 0.5 * jnp.sum((y_pred - y) ** 2) / batch_size

    if method == "bp":
        def bp_loss(p):
            _, pred = forward(p, skips, x, phi)
            return 0.5 * jnp.sum((pred - y) ** 2) / batch_size

        grads = jax.grad(bp_loss)(params)
        return grads, loss

    if method == "pc":
        states = run_pc_inference(
            params, skips, x, y, state_lr=state_lr, rho=rho, steps=budget * inner_steps, phi=phi
        )
        states = jax.tree_util.tree_map(jax.lax.stop_gradient, states)
        zero_duals = [jnp.zeros_like(s) for s in states]
        grads = jax.grad(lambda p: al_energy(p, skips, x, y, states, zero_duals, rho, phi))(params)
        return grads, loss

    if method == "pcalm":
        states, duals = run_pcalm_inference(
            params, skips, x, y,
            state_lr=state_lr, rho=rho, alpha=alpha,
            budget=budget, inner_steps=inner_steps, phi=phi
        )
        states = jax.tree_util.tree_map(jax.lax.stop_gradient, states)
        duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
        grads = jax.grad(lambda p: al_energy(p, skips, x, y, states, duals, rho, phi))(params)
        return grads, loss

    raise ValueError(f"Unknown method: {method}")
