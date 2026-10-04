from __future__ import annotations

import math
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .deq_conditioned import load_target_image
from .dip_experiment import adam_apply, adam_init, psnr
from .nca_experiment import extract_discrete_segmentation, extract_segmentation_pca
from .pool_regenerative_deq import create_damage_mask


def downsample_2x(x: jax.Array) -> jax.Array:
    """Exact area-average 2x downsampling."""
    b, c, h, w = x.shape
    return jnp.mean(x.reshape(b, c, h // 2, 2, w // 2, 2), axis=(3, 5))


def upsample_2x(x: jax.Array) -> jax.Array:
    """2x upsampling with nearest-neighbor repeat (preserves step edges)."""
    return jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)


def perceive_standard(z: jax.Array) -> jax.Array:
    """Standard 3x3 Sobel perception with edge replicate padding (no torus wrap!)."""
    zp = jnp.pad(z, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    tl = zp[:, :, :-2, :-2]
    tc = zp[:, :, :-2, 1:-1]
    tr = zp[:, :, :-2, 2:]
    ml = zp[:, :, 1:-1, :-2]
    mr = zp[:, :, 1:-1, 2:]
    bl = zp[:, :, 2:, :-2]
    bc = zp[:, :, 2:, 1:-1]
    br = zp[:, :, 2:, 2:]
    dx = (-tl + tr - 2.0 * ml + 2.0 * mr - bl + br) / 8.0
    dy = (-tl - 2.0 * tc - tr + bl + 2.0 * bc + br) / 8.0
    return jnp.concatenate([z, dx, dy], axis=1)


def init_pure_pyramid_deq(
    key: jax.Array,
    channels: int = 16,
    hidden_dim: int = 64,
    out_channels: int = 1,
) -> dict:
    """Initializes 100% coordinate-free 3-level Pyramid NCA parameters (48x48 -> 24x24 -> 12x12)."""
    k0, k1, k2, k_out = jax.random.split(key, 4)

    # Scale 0 (Fine, 48x48): in = 3*C (perception) + C (top-down from mid) = 64
    in0 = 3 * channels + channels
    std0 = 1.0 / math.sqrt(in0)
    w1_0 = jax.random.normal(k0, (hidden_dim, in0, 1, 1)) * std0
    b1_0 = jnp.zeros((hidden_dim, 1, 1))
    w2_0 = jnp.zeros((channels, hidden_dim, 1, 1))
    b2_0 = jnp.zeros((channels, 1, 1))

    # Scale 1 (Mid, 24x24): in = 3*C (perception) + C (bottom-up from fine) + C (top-down from coarse) = 80
    in1 = 3 * channels + 2 * channels
    std1 = 1.0 / math.sqrt(in1)
    w1_1 = jax.random.normal(k1, (hidden_dim, in1, 1, 1)) * std1
    b1_1 = jnp.zeros((hidden_dim, 1, 1))
    w2_1 = jnp.zeros((channels, hidden_dim, 1, 1))
    b2_1 = jnp.zeros((channels, 1, 1))

    # Scale 2 (Coarse, 12x12): in = 3*C (perception) + C (bottom-up from mid) = 64
    in2 = 3 * channels + channels
    std2 = 1.0 / math.sqrt(in2)
    w1_2 = jax.random.normal(k2, (hidden_dim, in2, 1, 1)) * std2
    b1_2 = jnp.zeros((hidden_dim, 1, 1))
    w2_2 = jnp.zeros((channels, hidden_dim, 1, 1))
    b2_2 = jnp.zeros((channels, 1, 1))

    # Shared readout head
    std_out = 1.0 / math.sqrt(channels)
    w_out = jax.random.normal(k_out, (out_channels, channels, 1, 1)) * std_out
    b_out = jnp.zeros((out_channels, 1, 1))

    return {
        "w1_0": w1_0, "b1_0": b1_0, "w2_0": w2_0, "b2_0": b2_0,
        "w1_1": w1_1, "b1_1": b1_1, "w2_1": w2_1, "b2_1": b2_1,
        "w1_2": w1_2, "b1_2": b1_2, "w2_2": w2_2, "b2_2": b2_2,
        "w_out": w_out, "b_out": b_out,
    }


def pure_pyramid_delta(
    z_pyr: tuple[jax.Array, jax.Array, jax.Array],
    params: dict,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes pure coordinate-free updates across fine, mid, and coarse scales."""
    z0, z1, z2 = z_pyr

    # Inter-scale communication
    z1_to_0 = upsample_2x(z1)
    z0_to_1 = downsample_2x(z0)
    z2_to_1 = upsample_2x(z2)
    z1_to_2 = downsample_2x(z1)

    # Scale 0 (Fine, 48x48)
    p0 = perceive_standard(z0)
    in0 = jnp.concatenate([p0, z1_to_0], axis=1)
    h0 = jnp.tensordot(in0, params["w1_0"][:, :, 0, 0], axes=([1], [1]))
    h0 = jnp.transpose(h0, (0, 3, 1, 2)) + params["b1_0"]
    h0 = jax.nn.relu(h0)
    d0 = jnp.tensordot(h0, params["w2_0"][:, :, 0, 0], axes=([1], [1]))
    d0 = jnp.transpose(d0, (0, 3, 1, 2)) + params["b2_0"]

    # Scale 1 (Mid, 24x24)
    p1 = perceive_standard(z1)
    in1 = jnp.concatenate([p1, z0_to_1, z2_to_1], axis=1)
    h1 = jnp.tensordot(in1, params["w1_1"][:, :, 0, 0], axes=([1], [1]))
    h1 = jnp.transpose(h1, (0, 3, 1, 2)) + params["b1_1"]
    h1 = jax.nn.relu(h1)
    d1 = jnp.tensordot(h1, params["w2_1"][:, :, 0, 0], axes=([1], [1]))
    d1 = jnp.transpose(d1, (0, 3, 1, 2)) + params["b2_1"]

    # Scale 2 (Coarse, 12x12)
    p2 = perceive_standard(z2)
    in2 = jnp.concatenate([p2, z1_to_2], axis=1)
    h2 = jnp.tensordot(in2, params["w1_2"][:, :, 0, 0], axes=([1], [1]))
    h2 = jnp.transpose(h2, (0, 3, 1, 2)) + params["b1_2"]
    h2 = jax.nn.relu(h2)
    d2 = jnp.tensordot(h2, params["w2_2"][:, :, 0, 0], axes=([1], [1]))
    d2 = jnp.transpose(d2, (0, 3, 1, 2)) + params["b2_2"]

    return (d0, d1, d2)


def readout(z: jax.Array, params: dict) -> jax.Array:
    """Projects hidden state to image channels."""
    out = jnp.tensordot(z, params["w_out"][:, :, 0, 0], axes=([1], [1]))
    return jnp.transpose(out, (0, 3, 1, 2)) + params["b_out"]


def pure_pyramid_deq_energy(
    params: dict,
    z_pyr: tuple[jax.Array, jax.Array, jax.Array],
    dual_pyr: tuple[jax.Array, jax.Array, jax.Array],
    y_pyr: tuple[jax.Array, jax.Array, jax.Array],
    rho: float = 1.0,
    tv_weight: float = 0.005,
) -> jax.Array:
    """Augmented Lagrangian energy on 3-scale pure coordinate-free pyramid."""
    z0, z1, z2 = z_pyr
    dual0, dual1, dual2 = dual_pyr
    y0, y1, y2 = y_pyr
    b = z0.shape[0]

    pred0 = readout(z0, params)
    pred1 = readout(z1, params)
    pred2 = readout(z2, params)

    loss_sup0 = (0.5 * jnp.sum((pred0 - y0) ** 2) + 0.5 * jnp.sum(jnp.sqrt((pred0 - y0) ** 2 + 1e-6))) / b
    loss_sup1 = (0.5 * jnp.sum((pred1 - y1) ** 2) + 0.5 * jnp.sum(jnp.sqrt((pred1 - y1) ** 2 + 1e-6))) / b
    loss_sup2 = (0.5 * jnp.sum((pred2 - y2) ** 2) + 0.5 * jnp.sum(jnp.sqrt((pred2 - y2) ** 2 + 1e-6))) / b
    loss_sup = loss_sup0 + 0.5 * loss_sup1 + 0.25 * loss_sup2

    d0, d1, d2 = pure_pyramid_delta(z_pyr, params)
    shifted0 = d0 + dual0 / rho
    shifted1 = d1 + dual1 / rho
    shifted2 = d2 + dual2 / rho
    loss_eq = 0.5 * rho * (jnp.sum(shifted0 ** 2) + jnp.sum(shifted1 ** 2) + jnp.sum(shifted2 ** 2)) / b

    diff_x0 = z0[:, :, :, 1:] - z0[:, :, :, :-1]
    diff_y0 = z0[:, :, 1:, :] - z0[:, :, :-1, :]
    loss_tv0 = tv_weight * (jnp.sum(jnp.sqrt(diff_x0 ** 2 + 1e-6)) + jnp.sum(jnp.sqrt(diff_y0 ** 2 + 1e-6))) / b

    return loss_sup + loss_eq + loss_tv0


def settle_pure_pyramid_deq_pcalm(
    params: dict,
    z_init: tuple[jax.Array, jax.Array, jax.Array],
    y_pyr: tuple[jax.Array, jax.Array, jax.Array],
    *,
    steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    tv_weight: float = 0.005,
) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """Settles 3-level pyramid states using PC-ALM."""
    z0, z1, z2 = z_init
    duals = (jnp.zeros_like(z0), jnp.zeros_like(z1), jnp.zeros_like(z2))

    def inner_step(z_curr, dual_curr):
        def energy(zc):
            return pure_pyramid_deq_energy(params, zc, dual_curr, y_pyr, rho=rho, tv_weight=tv_weight)

        grad_fn = jax.grad(energy)

        def step(zc, _):
            g0, g1, g2 = grad_fn(zc)
            return (zc[0] - state_lr * g0, zc[1] - state_lr * g1, zc[2] - state_lr * g2), None

        z_next, _ = jax.lax.scan(step, z_curr, xs=None, length=inner_steps)
        return z_next

    def outer(carry, _):
        zc, dualc = carry
        zc = inner_step(zc, dualc)
        r0, r1, r2 = pure_pyramid_delta(zc, params)
        dual_next = (dualc[0] + alpha * r0, dualc[1] + alpha * r1, dualc[2] + alpha * r2)
        return (zc, dual_next), None

    if steps > 1:
        (z_final, dual_final), _ = jax.lax.scan(outer, (z_init, duals), xs=None, length=steps - 1)
    else:
        z_final, dual_final = z_init, duals

    z_final = inner_step(z_final, dual_final)
    return z_final, dual_final


def compute_pure_pyramid_deq_grads(
    params: dict,
    z_init: tuple[jax.Array, jax.Array, jax.Array],
    y_pyr: tuple[jax.Array, jax.Array, jax.Array],
    *,
    steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    tv_weight: float = 0.005,
) -> tuple[dict, jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    z_eq, dual_eq = settle_pure_pyramid_deq_pcalm(
        params, z_init, y_pyr,
        steps=steps, inner_steps=inner_steps, state_lr=state_lr, rho=rho, alpha=alpha, tv_weight=tv_weight,
    )

    pred0 = readout(z_eq[0], params)
    batch_size = pred0.shape[0]
    loss = (0.5 * jnp.sum((pred0 - y_pyr[0]) ** 2) + 0.5 * jnp.sum(jnp.sqrt((pred0 - y_pyr[0]) ** 2 + 1e-6))) / batch_size

    z_eq_stop = (jax.lax.stop_gradient(z_eq[0]), jax.lax.stop_gradient(z_eq[1]), jax.lax.stop_gradient(z_eq[2]))
    dual_eq_stop = (jax.lax.stop_gradient(dual_eq[0]), jax.lax.stop_gradient(dual_eq[1]), jax.lax.stop_gradient(dual_eq[2]))

    def p_loss(p):
        return pure_pyramid_deq_energy(p, z_eq_stop, dual_eq_stop, y_pyr, rho=rho, tv_weight=tv_weight)

    grads = jax.grad(p_loss)(params)
    return grads, loss, z_eq


def compute_pure_pyramid_spectral_radius(
    params: dict,
    z_eq: tuple[jax.Array, jax.Array, jax.Array],
    n_iter: int = 15,
) -> float:
    k0, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
    v0 = jax.random.normal(k0, z_eq[0].shape)
    v1 = jax.random.normal(k1, z_eq[1].shape)
    v2 = jax.random.normal(k2, z_eq[2].shape)
    norm = math.sqrt(float(jnp.sum(v0 ** 2) + jnp.sum(v1 ** 2) + jnp.sum(v2 ** 2))) + 1e-8
    v = (v0 / norm, v1 / norm, v2 / norm)

    def op(z):
        d0, d1, d2 = pure_pyramid_delta(z, params)
        return (z[0] + 0.5 * d0, z[1] + 0.5 * d1, z[2] + 0.5 * d2)

    for _ in range(n_iter):
        _, jv = jax.jvp(op, (z_eq,), (v,))
        norm = math.sqrt(float(jnp.sum(jv[0] ** 2) + jnp.sum(jv[1] ** 2) + jnp.sum(jv[2] ** 2))) + 1e-8
        v = (jv[0] / norm, jv[1] / norm, jv[2] / norm)

    _, jv = jax.jvp(op, (z_eq,), (v,))
    return math.sqrt(float(jnp.sum(jv[0] ** 2) + jnp.sum(jv[1] ** 2) + jnp.sum(jv[2] ** 2)))


def run_pure_pyramid_experiment(
    image_name: str = "camera",
    pool_size: int = 32,
    batch_size: int = 8,
    steps: int = 400,
    size: int = 48,
    channels: int = 16,
    hidden_dim: int = 64,
    lr: float = 3e-3,
    deq_steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    tv_weight: float = 0.005,
    seed: int = 42,
    save_dir: Path | None = None,
) -> dict:
    """Runs 100% Coordinate-Free 3-Level Pyramid NCA DEQ."""
    key = jax.random.PRNGKey(seed)
    k_net, k_pool0, k_pool1, k_pool2, k_loop = jax.random.split(key, 5)

    clean_np, out_channels = load_target_image(image_name, size=size)
    y_target0 = jnp.asarray(clean_np[None, ...], dtype=jnp.float32)
    y_batch0 = jnp.broadcast_to(y_target0, (batch_size, out_channels, size, size))
    y_batch1 = downsample_2x(y_batch0)
    y_batch2 = downsample_2x(y_batch1)
    y_pyr = (y_batch0, y_batch1, y_batch2)

    params = init_pure_pyramid_deq(k_net, channels=channels, hidden_dim=hidden_dim, out_channels=out_channels)
    opt_state = adam_init(params)

    pool0 = jax.random.normal(k_pool0, (pool_size, channels, size, size)) * 0.05
    pool1 = jax.random.normal(k_pool1, (pool_size, channels, size // 2, size // 2)) * 0.05
    pool2 = jax.random.normal(k_pool2, (pool_size, channels, size // 4, size // 4)) * 0.05

    @jax.jit
    def pool_train_step(p, opt_s, p0_s, p1_s, p2_s, k):
        k_idx, k_dam, k_seed, k_n, k_pert = jax.random.split(k, 5)
        idx = jax.random.choice(k_idx, pool_size, (batch_size,), replace=False)
        batch_z0 = p0_s[idx]
        batch_z1 = p1_s[idx]
        batch_z2 = p2_s[idx]

        preds = readout(batch_z0, p)
        sample_losses = jnp.mean((preds - y_batch0) ** 2, axis=(1, 2, 3))
        worst_idx = jnp.argmax(sample_losses)
        fresh0 = jax.random.normal(k_seed, (channels, size, size)) * 0.05
        fresh1 = downsample_2x(fresh0[None, ...])[0]
        fresh2 = downsample_2x(fresh1[None, ...])[0]
        batch_z0 = batch_z0.at[worst_idx].set(fresh0)
        batch_z1 = batch_z1.at[worst_idx].set(fresh1)
        batch_z2 = batch_z2.at[worst_idx].set(fresh2)

        k_dam_split = jax.random.split(k_dam, batch_size)
        def dam_fn(k_d, i):
            d_type = jax.random.randint(k_d, (), 0, 3)
            d_type = jnp.where(i == 0, 0, d_type)
            m0 = create_damage_mask(k_d, size, size, d_type)[None, :, :]
            return jnp.where(i < 4, m0, jnp.ones_like(m0))

        masks0 = jax.vmap(dam_fn)(k_dam_split, jnp.arange(batch_size))
        masks1 = downsample_2x(masks0)
        masks2 = downsample_2x(masks1)

        batch_z0_dam = batch_z0 * masks0
        batch_z1_dam = batch_z1 * masks1
        batch_z2_dam = batch_z2 * masks2

        deq_grads, total_loss, z_eq = compute_pure_pyramid_deq_grads(
            p, (batch_z0_dam, batch_z1_dam, batch_z2_dam), y_pyr,
            steps=deq_steps, inner_steps=inner_steps, state_lr=state_lr, rho=rho, alpha=alpha, tv_weight=tv_weight,
        )

        def flow_loss_fn(p_curr):
            n_steps = jax.random.randint(k_n, (), 35, 61)
            z0_target_stop = jax.lax.stop_gradient(z_eq[0][:4])
            z1_target_stop = jax.lax.stop_gradient(z_eq[1][:4])
            z2_target_stop = jax.lax.stop_gradient(z_eq[2][:4])

            m0_target = masks0[:4]
            m1_target = masks1[:4]
            m2_target = masks2[:4]
            dam_mask0 = 1.0 - m0_target
            dam_mask1 = 1.0 - m1_target
            dam_mask2 = 1.0 - m2_target
            dam_pixels0 = jnp.sum(dam_mask0) + 1e-6

            def flow_step(zc, _):
                d0, d1, d2 = pure_pyramid_delta(zc, p_curr)
                z0_next = jnp.where(m0_target > 0.5, z0_target_stop, zc[0] + 0.5 * d0)
                z1_next = jnp.where(m1_target > 0.5, z1_target_stop, zc[1] + 0.5 * d1)
                z2_next = jnp.where(m2_target > 0.5, z2_target_stop, zc[2] + 0.5 * d2)
                return (z0_next, z1_next, z2_next), (z0_next, z1_next, z2_next)

            _, (z0_traj, z1_traj, z2_traj) = jax.lax.scan(flow_step, (batch_z0_dam[:4], batch_z1_dam[:4], batch_z2_dam[:4]), xs=None, length=60)
            z0_final = z0_traj[n_steps - 1]
            z1_final = z1_traj[n_steps - 1]
            z2_final = z2_traj[n_steps - 1]

            l_flow0 = jnp.sum(dam_mask0 * (z0_final - z0_target_stop) ** 2) / (dam_pixels0 * channels)
            l_flow1 = jnp.sum(dam_mask1 * (z1_final - z1_target_stop) ** 2) / (jnp.sum(dam_mask1) * channels + 1e-6)
            l_flow2 = jnp.sum(dam_mask2 * (z2_final - z2_target_stop) ** 2) / (jnp.sum(dam_mask2) * channels + 1e-6)
            l_flow = l_flow0 + 0.5 * l_flow1 + 0.25 * l_flow2

            pred_flow = readout(z0_final, p_curr)
            y_sub = y_batch0[:4]
            l_recon_l2 = jnp.sum(dam_mask0 * (pred_flow - y_sub) ** 2) / dam_pixels0
            l_recon_l1 = jnp.sum(dam_mask0 * jnp.sqrt((pred_flow - y_sub) ** 2 + 1e-6)) / dam_pixels0

            gx_pred = pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1]
            gy_pred = pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :]
            gx_y = y_sub[:, :, :, 1:] - y_sub[:, :, :, :-1]
            gy_y = y_sub[:, :, 1:, :] - y_sub[:, :, :-1, :]
            mask_gx = dam_mask0[:, :, :, 1:] * dam_mask0[:, :, :, :-1]
            mask_gy = dam_mask0[:, :, 1:, :] * dam_mask0[:, :, :-1, :]
            l_edge = (
                jnp.sum(mask_gx * jnp.sqrt((gx_pred - gx_y) ** 2 + 1e-6)) / (jnp.sum(mask_gx) + 1e-6)
                + jnp.sum(mask_gy * jnp.sqrt((gy_pred - gy_y) ** 2 + 1e-6)) / (jnp.sum(mask_gy) + 1e-6)
            )

            dam_px_sample = jnp.sum(dam_mask0, axis=(2, 3), keepdims=True) + 1e-6
            mu_pred = jnp.sum(dam_mask0 * pred_flow, axis=(2, 3), keepdims=True) / dam_px_sample
            var_pred = jnp.sum(dam_mask0 * (pred_flow - mu_pred) ** 2, axis=(2, 3), keepdims=True) / dam_px_sample
            std_pred = jnp.sqrt(var_pred + 1e-6)
            mu_y = jnp.sum(dam_mask0 * y_sub, axis=(2, 3), keepdims=True) / dam_px_sample
            var_y = jnp.sum(dam_mask0 * (y_sub - mu_y) ** 2, axis=(2, 3), keepdims=True) / dam_px_sample
            std_y = jnp.sqrt(var_y + 1e-6)
            l_contrast = jnp.mean(jnp.abs(std_pred - std_y))
            l_mean_match = jnp.mean(jnp.abs(mu_pred - mu_y))

            l_recon = 0.5 * l_recon_l2 + 1.5 * l_recon_l1 + 2.5 * l_edge + 1.5 * l_contrast + 0.5 * l_mean_match

            d0_target, d1_target, d2_target = pure_pyramid_delta((z0_target_stop, z1_target_stop, z2_target_stop), p_curr)
            l_target_drift = jnp.mean(d0_target ** 2) + 0.5 * jnp.mean(d1_target ** 2) + 0.25 * jnp.mean(d2_target ** 2)

            noise0 = jax.random.normal(k_pert, z0_target_stop.shape) * 0.20
            noise1 = downsample_2x(noise0)
            noise2 = downsample_2x(noise1)
            z0_pert = z0_target_stop + noise0
            z1_pert = z1_target_stop + noise1
            z2_pert = z2_target_stop + noise2
            d0_p, d1_p, d2_p = pure_pyramid_delta((z0_pert, z1_pert, z2_pert), p_curr)
            z0_res = z0_pert + 0.5 * d0_p
            z1_res = z1_pert + 0.5 * d1_p
            z2_res = z2_pert + 0.5 * d2_p
            l_spring = (
                jnp.mean((z0_res - z0_target_stop) ** 2)
                + 0.5 * jnp.mean((z1_res - z1_target_stop) ** 2)
                + 0.25 * jnp.mean((z2_res - z2_target_stop) ** 2)
            )

            l_reg = 1e-4 * (jnp.sum(p_curr["w2_0"] ** 2) + jnp.sum(p_curr["w2_1"] ** 2) + jnp.sum(p_curr["w2_2"] ** 2))
            loss = l_flow + 1.2 * l_recon + 1.0 * l_target_drift + 3.0 * l_spring + l_reg
            return loss, (z0_final, z1_final, z2_final)

        (flow_val, z_final_stepped), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(p)
        combined_grads = jax.tree_util.tree_map(lambda g1, g2: g1 + 1.0 * g2, deq_grads, flow_grads)

        p, opt_s = adam_apply(p, combined_grads, opt_s, lr=lr)

        batch_z0_up = batch_z0.at[:4].set(jax.lax.stop_gradient(z_final_stepped[0]))
        batch_z0_up = batch_z0_up.at[4:].set(jax.lax.stop_gradient(z_eq[0][4:]))
        new_p0 = p0_s.at[idx].set(batch_z0_up)

        batch_z1_up = batch_z1.at[:4].set(jax.lax.stop_gradient(z_final_stepped[1]))
        batch_z1_up = batch_z1_up.at[4:].set(jax.lax.stop_gradient(z_eq[1][4:]))
        new_p1 = p1_s.at[idx].set(batch_z1_up)

        batch_z2_up = batch_z2.at[:4].set(jax.lax.stop_gradient(z_final_stepped[2]))
        batch_z2_up = batch_z2_up.at[4:].set(jax.lax.stop_gradient(z_eq[2][4:]))
        new_p2 = p2_s.at[idx].set(batch_z2_up)

        return p, opt_s, new_p0, new_p1, new_p2, total_loss

    print(f"=== Starting 100% Coordinate-Free 3-Level Pyramid NCA on '{image_name}' (48->24->12, steps={steps}) ===")
    history = {"step": [], "loss": [], "psnr": []}
    t0 = time.time()
    k_steps = jax.random.split(k_loop, steps)

    for s in range(1, steps + 1):
        params, opt_state, pool0, pool1, pool2, loss_val = pool_train_step(params, opt_state, pool0, pool1, pool2, k_steps[s - 1])

        if s % 20 == 0 or s == 1 or s == steps:
            preds_all = readout(pool0, params)
            mses = [float(np.mean((np.asarray(preds_all[i]) - clean_np) ** 2)) for i in range(pool_size)]
            best_idx = int(np.argmin(mses))
            best_p = psnr(np.asarray(preds_all[best_idx]), clean_np)
            history["step"].append(s)
            history["loss"].append(float(loss_val))
            history["psnr"].append(best_p)
            print(f"[PURE-PYRAMID] Step {s:3d}/{steps} | Loss: {float(loss_val):.5f} | Best Fine PSNR: {best_p:.2f} dB (idx {best_idx})")

    elapsed = time.time() - t0
    print(f"[PURE-PYRAMID] Training finished in {elapsed:.2f}s.")

    preds_all = readout(pool0, params)
    mses = [float(np.mean((np.asarray(preds_all[i]) - clean_np) ** 2)) for i in range(pool_size)]
    best_idx = int(np.argmin(mses))
    z_eq = (pool0[best_idx:best_idx + 1], pool1[best_idx:best_idx + 1], pool2[best_idx:best_idx + 1])

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        pred_clean0 = readout(z_eq[0], params)
        pred_np0 = np.asarray(pred_clean0[0])

        if out_channels == 1:
            Image.fromarray((clean_np[0] * 255.0).astype(np.uint8)).save(save_dir / "target.png")
            Image.fromarray(np.clip(pred_np0[0] * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_recon.png")
        else:
            Image.fromarray(np.clip(np.transpose(clean_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "target.png")
            Image.fromarray(np.clip(np.transpose(pred_np0, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_recon.png")

        hidden_np = np.asarray(z_eq[0][0])
        seg_pca = extract_segmentation_pca(hidden_np)
        Image.fromarray(seg_pca).save(save_dir / "deq_segmentation_pca.png")

        labels_2d, discrete_rgb, _ = extract_discrete_segmentation(hidden_np, n_clusters=4, seed=seed)
        Image.fromarray(discrete_rgb).save(save_dir / "deq_discrete_seg.png")

        run_pure_pyramid_decimation_battery(params, z_eq, clean_np, out_channels, size, save_dir)

    return history


def run_pure_pyramid_decimation_battery(
    params: dict,
    z_eq: tuple[jax.Array, jax.Array, jax.Array],
    clean_np: np.ndarray,
    out_channels: int,
    size: int,
    save_dir: Path,
):
    """Executes decimation tests on 100% coordinate-free 3-level Pyramid."""
    print("=== Running Pure Coordinate-Free Pyramid Decimation Battery ===")

    lam_max = compute_pure_pyramid_spectral_radius(params, z_eq)
    print(f"[THEORY] Spectral radius of pure pyramid relaxation at z_eq: |lambda_max| = {lam_max:.4f}")

    def forward_pure_relax(z_start_pyr, keep_mask0, n_steps=60, step_size=0.5):
        zc0, zc1, zc2 = z_start_pyr
        keep_mask1 = downsample_2x(keep_mask0)
        keep_mask2 = downsample_2x(keep_mask1)
        dam_mask0 = 1.0 - keep_mask0

        m_padded = jnp.pad(keep_mask0, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
        up = m_padded[:, :, :size, 1:size + 1]
        down = m_padded[:, :, 2:, 1:size + 1]
        left = m_padded[:, :, 1:size + 1, :size]
        right = m_padded[:, :, 1:size + 1, 2:]
        seam_mask = ((up + down + left + right) > 0.5) & (keep_mask0 < 0.5)
        seam_weight = float(jnp.sum(seam_mask)) + 1e-6

        history0 = [zc0]
        seam_errors = []
        residuals = []

        for s in range(1, n_steps + 1):
            d0, d1, d2 = pure_pyramid_delta((zc0, zc1, zc2), params)
            res = float(jnp.mean((dam_mask0 * d0) ** 2))
            residuals.append(res)

            zc0 = jnp.where(keep_mask0 > 0.5, z_eq[0], zc0 + step_size * d0)
            zc1 = jnp.where(keep_mask1 > 0.5, z_eq[1], zc1 + step_size * d1)
            zc2 = jnp.where(keep_mask2 > 0.5, z_eq[2], zc2 + step_size * d2)
            history0.append(zc0)

            seam_err = float(jnp.sum(jnp.abs(zc0 - z_eq[0]) * seam_mask) / seam_weight)
            seam_errors.append(seam_err)

            if s >= 45 and res < 1e-6:
                break

        t_final = len(history0) - 1
        indices = np.linspace(0, t_final, 5, dtype=int)
        print(f"[PURE-RELAX] Ran {t_final} steps | Final seam_err: {seam_errors[-1]:.5f}, res: {residuals[-1]:.6f} | linspace frames: {indices.tolist()}")

        snapshots = [history0[idx] for idx in indices]
        return snapshots

    def to_img(z_state):
        pred = readout(z_state, params)
        p_np = np.asarray(pred[0])
        if out_channels == 1:
            return np.clip(p_np[0] * 255.0, 0, 255).astype(np.uint8)
        else:
            return np.clip(np.transpose(p_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)

    # --- Test 1: Half-Wipe ---
    mask_half0 = jnp.ones((1, 1, size, size), dtype=jnp.float32)
    mask_half0 = mask_half0.at[:, :, :, size // 2:].set(0.0)
    z_half0 = z_eq[0] * mask_half0
    z_half1 = z_eq[1] * downsample_2x(mask_half0)
    z_half2 = z_eq[2] * downsample_2x(downsample_2x(mask_half0))

    snaps_half = forward_pure_relax((z_half0, z_half1, z_half2), mask_half0, n_steps=60, step_size=0.5)
    imgs_half = [to_img(s) for s in snaps_half]
    if out_channels == 1:
        imgs_half = [np.stack([im, im, im], axis=-1) for im in imgs_half]
    strip_half = np.concatenate(imgs_half, axis=1)
    Image.fromarray(strip_half).save(save_dir / "regen_half_wipe_strip.png")

    # --- Test 2: Center Crater ---
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = size / 2.0, size / 2.0
    r = size / 3.0
    mask_circle0 = ((yy - cy) ** 2 + (xx - cx) ** 2 >= r ** 2).astype(np.float32)[None, None, ...]
    z_circle0 = z_eq[0] * jnp.asarray(mask_circle0)
    z_circle1 = z_eq[1] * downsample_2x(jnp.asarray(mask_circle0))
    z_circle2 = z_eq[2] * downsample_2x(downsample_2x(jnp.asarray(mask_circle0)))

    snaps_circle = forward_pure_relax((z_circle0, z_circle1, z_circle2), jnp.asarray(mask_circle0), n_steps=60, step_size=0.5)
    imgs_circle = [to_img(s) for s in snaps_circle]
    if out_channels == 1:
        imgs_circle = [np.stack([im, im, im], axis=-1) for im in imgs_circle]
    strip_circle = np.concatenate(imgs_circle, axis=1)
    Image.fromarray(strip_circle).save(save_dir / "regen_crater_strip.png")

    # --- Test 3: Pepper Decimation ---
    np.random.seed(42)
    mask_pepper0 = (np.random.rand(1, 1, size, size) > 0.5).astype(np.float32)
    z_pepper0 = z_eq[0] * jnp.asarray(mask_pepper0)
    z_pepper1 = z_eq[1] * downsample_2x(jnp.asarray(mask_pepper0))
    z_pepper2 = z_eq[2] * downsample_2x(downsample_2x(jnp.asarray(mask_pepper0)))

    snaps_pepper = forward_pure_relax((z_pepper0, z_pepper1, z_pepper2), jnp.asarray(mask_pepper0), n_steps=60, step_size=0.5)
    imgs_pepper = [to_img(s) for s in snaps_pepper]
    if out_channels == 1:
        imgs_pepper = [np.stack([im, im, im], axis=-1) for im in imgs_pepper]
    strip_pepper = np.concatenate(imgs_pepper, axis=1)
    Image.fromarray(strip_pepper).save(save_dir / "regen_pepper_strip.png")

    for strip_name, strip_arr in [
        ("regen_half_wipe_strip", strip_half),
        ("regen_crater_strip", strip_circle),
        ("regen_pepper_strip", strip_pepper),
    ]:
        im = Image.fromarray(strip_arr)
        im_large = im.resize((im.width * 6, im.height * 6), Image.Resampling.NEAREST)
        im_large.save(save_dir / f"{strip_name}_large.png")

    print(f"Saved Pure Pyramid decimation strips to {save_dir}")
