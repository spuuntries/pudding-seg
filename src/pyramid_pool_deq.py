from __future__ import annotations

import math
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .deq_conditioned import load_target_image, make_fourier_coords
from .dip_experiment import adam_apply, adam_init, psnr
from .nca_experiment import extract_discrete_segmentation, extract_segmentation_pca
from .pool_regenerative_deq import create_damage_mask


def downsample_2x(x: jax.Array) -> jax.Array:
    """Exact area-average 2x downsampling."""
    b, c, h, w = x.shape
    return jnp.mean(x.reshape(b, c, h // 2, 2, w // 2, 2), axis=(3, 5))


def upsample_2x(x: jax.Array) -> jax.Array:
    """2x upsampling with nearest-neighbor repeat (preserves crisp step edges without bilinear blur)."""
    return jnp.repeat(jnp.repeat(x, 2, axis=2), 2, axis=3)


def perceive_standard(z: jax.Array) -> jax.Array:
    """Multi-scale perception: state + Sobel gradients + direct 1-pixel directional diffs + 5-point Laplacian."""
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
    d_r = mr - z
    d_l = ml - z
    d_d = bc - z
    d_u = tc - z
    lap = (tc + bc + ml + mr - 4.0 * z) / 4.0
    return jnp.concatenate([z, dx, dy, d_r, d_l, d_d, d_u, lap], axis=1)


def init_pyramid_deq(
    key: jax.Array,
    channels: int = 16,
    hidden_dim: int = 64,
    in_cond_dim: int = 18,
    out_channels: int = 1,
) -> dict:
    """Initializes 2-level Pyramid NCA DEQ parameters."""
    k0, k1, k_out = jax.random.split(key, 3)

    # Input: 8*C (local perception) + C (inter-scale context) + in_cond_dim (spatial coordinates)
    total_in = 8 * channels + channels + in_cond_dim
    std_in = 1.0 / math.sqrt(total_in)

    # Scale 0 (Fine)
    k0_1, k0_2 = jax.random.split(k0)
    w1_0 = jax.random.normal(k0_1, (hidden_dim, total_in, 1, 1)) * std_in
    b1_0 = jnp.zeros((hidden_dim, 1, 1))
    w2_0 = jnp.zeros((channels, hidden_dim, 1, 1))
    b2_0 = jnp.zeros((channels, 1, 1))

    # Scale 1 (Coarse)
    k1_1, k1_2 = jax.random.split(k1)
    w1_1 = jax.random.normal(k1_1, (hidden_dim, total_in, 1, 1)) * std_in
    b1_1 = jnp.zeros((hidden_dim, 1, 1))
    w2_1 = jnp.zeros((channels, hidden_dim, 1, 1))
    b2_1 = jnp.zeros((channels, 1, 1))

    # Shared Readout head: channels -> out_channels
    std_out = 1.0 / math.sqrt(channels)
    w_out = jax.random.normal(k_out, (out_channels, channels, 1, 1)) * std_out
    b_out = jnp.zeros((out_channels, 1, 1))

    return {
        "w1_0": w1_0, "b1_0": b1_0, "w2_0": w2_0, "b2_0": b2_0,
        "w1_1": w1_1, "b1_1": b1_1, "w2_1": w2_1, "b2_1": b2_1,
        "w_out": w_out, "b_out": b_out,
    }


def pyramid_delta(
    z_pyr: tuple[jax.Array, jax.Array],
    cond_pyr: tuple[jax.Array, jax.Array],
    params: dict,
) -> tuple[jax.Array, jax.Array]:
    """Computes joint velocity updates across fine and coarse scales."""
    z0, z1 = z_pyr
    cond0, cond1 = cond_pyr

    # Inter-scale communication
    z1_to_0 = upsample_2x(z1)
    z0_to_1 = downsample_2x(z0)

    # Fine level (scale 0)
    p0 = perceive_standard(z0)
    if cond0.shape[0] != p0.shape[0]:
        c0 = jnp.broadcast_to(cond0, (p0.shape[0], *cond0.shape[1:]))
    else:
        c0 = cond0
    in0 = jnp.concatenate([p0, z1_to_0, c0], axis=1)
    h0 = jnp.tensordot(in0, params["w1_0"][:, :, 0, 0], axes=([1], [1]))
    h0 = jnp.transpose(h0, (0, 3, 1, 2)) + params["b1_0"]
    h0 = jax.nn.relu(h0)
    d0 = jnp.tensordot(h0, params["w2_0"][:, :, 0, 0], axes=([1], [1]))
    d0 = jnp.transpose(d0, (0, 3, 1, 2)) + params["b2_0"]

    # Coarse level (scale 1)
    p1 = perceive_standard(z1)
    if cond1.shape[0] != p1.shape[0]:
        c1 = jnp.broadcast_to(cond1, (p1.shape[0], *cond1.shape[1:]))
    else:
        c1 = cond1
    in1 = jnp.concatenate([p1, z0_to_1, c1], axis=1)
    h1 = jnp.tensordot(in1, params["w1_1"][:, :, 0, 0], axes=([1], [1]))
    h1 = jnp.transpose(h1, (0, 3, 1, 2)) + params["b1_1"]
    h1 = jax.nn.relu(h1)
    d1 = jnp.tensordot(h1, params["w2_1"][:, :, 0, 0], axes=([1], [1]))
    d1 = jnp.transpose(d1, (0, 3, 1, 2)) + params["b2_1"]

    return (d0, d1)


def readout(z: jax.Array, params: dict) -> jax.Array:
    """Projects hidden state to image channels."""
    out = jnp.tensordot(z, params["w_out"][:, :, 0, 0], axes=([1], [1]))
    return jnp.transpose(out, (0, 3, 1, 2)) + params["b_out"]


def pyramid_deq_energy(
    params: dict,
    z_pyr: tuple[jax.Array, jax.Array],
    dual_pyr: tuple[jax.Array, jax.Array],
    cond_pyr: tuple[jax.Array, jax.Array],
    y_pyr: tuple[jax.Array, jax.Array],
    rho: float = 1.0,
    tv_weight: float = 0.005,
) -> jax.Array:
    """Augmented Lagrangian energy on 2-scale pyramid."""
    z0, z1 = z_pyr
    dual0, dual1 = dual_pyr
    y0, y1 = y_pyr
    b = z0.shape[0]

    # Supervision on both fine and coarse scales
    pred0 = readout(z0, params)
    pred1 = readout(z1, params)
    loss_sup0 = (0.5 * jnp.sum((pred0 - y0) ** 2) + 0.5 * jnp.sum(jnp.sqrt((pred0 - y0) ** 2 + 1e-6))) / b
    loss_sup1 = (0.5 * jnp.sum((pred1 - y1) ** 2) + 0.5 * jnp.sum(jnp.sqrt((pred1 - y1) ** 2 + 1e-6))) / b
    loss_sup = loss_sup0 + 0.5 * loss_sup1

    # Fixed-point equilibrium constraint: Delta(Z*) = 0 across both scales
    d0, d1 = pyramid_delta(z_pyr, cond_pyr, params)
    shifted0 = d0 + dual0 / rho
    shifted1 = d1 + dual1 / rho
    loss_eq0 = 0.5 * rho * jnp.sum(shifted0 ** 2) / b
    loss_eq1 = 0.5 * rho * jnp.sum(shifted1 ** 2) / b
    loss_eq = loss_eq0 + loss_eq1

    # Edge-preserving TV regularizer
    diff_x0 = z0[:, :, :, 1:] - z0[:, :, :, :-1]
    diff_y0 = z0[:, :, 1:, :] - z0[:, :, :-1, :]
    loss_tv0 = tv_weight * (jnp.sum(jnp.sqrt(diff_x0 ** 2 + 1e-6)) + jnp.sum(jnp.sqrt(diff_y0 ** 2 + 1e-6))) / b

    diff_x1 = z1[:, :, :, 1:] - z1[:, :, :, :-1]
    diff_y1 = z1[:, :, 1:, :] - z1[:, :, :-1, :]
    loss_tv1 = tv_weight * (jnp.sum(jnp.sqrt(diff_x1 ** 2 + 1e-6)) + jnp.sum(jnp.sqrt(diff_y1 ** 2 + 1e-6))) / b

    return loss_sup + loss_eq + (loss_tv0 + 0.5 * loss_tv1)


def settle_pyramid_deq_pcalm(
    params: dict,
    z_init: tuple[jax.Array, jax.Array],
    cond_pyr: tuple[jax.Array, jax.Array],
    y_pyr: tuple[jax.Array, jax.Array],
    *,
    steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    tv_weight: float = 0.005,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Settles pyramid hidden states using PC-ALM Augmented Lagrangian."""
    z0, z1 = z_init
    duals = (jnp.zeros_like(z0), jnp.zeros_like(z1))

    def inner_step(z_curr, dual_curr):
        def energy(zc):
            return pyramid_deq_energy(params, zc, dual_curr, cond_pyr, y_pyr, rho=rho, tv_weight=tv_weight)

        grad_fn = jax.grad(energy)

        def step(zc, _):
            g0, g1 = grad_fn(zc)
            z0_next = zc[0] - state_lr * g0
            z1_next = zc[1] - state_lr * g1
            return (z0_next, z1_next), None

        z_next, _ = jax.lax.scan(step, z_curr, xs=None, length=inner_steps)
        return z_next

    def outer(carry, _):
        zc, dualc = carry
        zc = inner_step(zc, dualc)
        r0, r1 = pyramid_delta(zc, cond_pyr, params)
        dual_next = (dualc[0] + alpha * r0, dualc[1] + alpha * r1)
        return (zc, dual_next), None

    if steps > 1:
        (z_final, dual_final), _ = jax.lax.scan(outer, (z_init, duals), xs=None, length=steps - 1)
    else:
        z_final, dual_final = z_init, duals

    z_final = inner_step(z_final, dual_final)
    return z_final, dual_final


def compute_pyramid_deq_grads(
    params: dict,
    z_init: tuple[jax.Array, jax.Array],
    cond_pyr: tuple[jax.Array, jax.Array],
    y_pyr: tuple[jax.Array, jax.Array],
    *,
    steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    tv_weight: float = 0.005,
) -> tuple[dict, jax.Array, tuple[jax.Array, jax.Array]]:
    """Solves PC-ALM stationary equilibrium and computes analytical DEQ gradients."""
    z_eq, dual_eq = settle_pyramid_deq_pcalm(
        params, z_init, cond_pyr, y_pyr,
        steps=steps, inner_steps=inner_steps, state_lr=state_lr, rho=rho, alpha=alpha, tv_weight=tv_weight,
    )

    pred0 = readout(z_eq[0], params)
    batch_size = pred0.shape[0]
    loss = (0.5 * jnp.sum((pred0 - y_pyr[0]) ** 2) + 0.5 * jnp.sum(jnp.sqrt((pred0 - y_pyr[0]) ** 2 + 1e-6))) / batch_size

    z_eq_stop = (jax.lax.stop_gradient(z_eq[0]), jax.lax.stop_gradient(z_eq[1]))
    dual_eq_stop = (jax.lax.stop_gradient(dual_eq[0]), jax.lax.stop_gradient(dual_eq[1]))

    def p_loss(p):
        return pyramid_deq_energy(p, z_eq_stop, dual_eq_stop, cond_pyr, y_pyr, rho=rho, tv_weight=tv_weight)

    grads = jax.grad(p_loss)(params)
    return grads, loss, z_eq


def compute_pyramid_spectral_radius(
    params: dict,
    z_eq: tuple[jax.Array, jax.Array],
    cond_pyr: tuple[jax.Array, jax.Array],
    n_iter: int = 15,
) -> float:
    """Computes joint spectral radius |lambda_max| of pyramid relaxation operator."""
    k0, k1 = jax.random.split(jax.random.PRNGKey(0))
    v0 = jax.random.normal(k0, z_eq[0].shape)
    v1 = jax.random.normal(k1, z_eq[1].shape)
    norm = math.sqrt(float(jnp.sum(v0 ** 2) + jnp.sum(v1 ** 2))) + 1e-8
    v = (v0 / norm, v1 / norm)

    def op(z):
        d0, d1 = pyramid_delta(z, cond_pyr, params)
        return (z[0] + 0.5 * d0, z[1] + 0.5 * d1)

    for _ in range(n_iter):
        _, jv = jax.jvp(op, (z_eq,), (v,))
        norm = math.sqrt(float(jnp.sum(jv[0] ** 2) + jnp.sum(jv[1] ** 2))) + 1e-8
        v = (jv[0] / norm, jv[1] / norm)

    _, jv = jax.jvp(op, (z_eq,), (v,))
    return math.sqrt(float(jnp.sum(jv[0] ** 2) + jnp.sum(jv[1] ** 2)))


def run_pyramid_experiment(
    image_name: str = "camera",
    pool_size: int = 32,
    batch_size: int = 8,
    steps: int = 350,
    size: int = 48,
    channels: int = 16,
    hidden_dim: int = 96,
    lr: float = 3e-3,
    deq_steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    tv_weight: float = 0.005,
    octaves: int = 7,
    seed: int = 42,
    save_dir: Path | None = None,
) -> dict:
    """Runs Pyramid NCA DEQ with multi-scale distillation pool and forward decimation battery."""
    key = jax.random.PRNGKey(seed)
    k_net, k_pool0, k_pool1, k_loop = jax.random.split(key, 4)

    clean_np, out_channels = load_target_image(image_name, size=size)
    y_target0 = jnp.asarray(clean_np[None, ...], dtype=jnp.float32)
    y_batch0 = jnp.broadcast_to(y_target0, (batch_size, out_channels, size, size))
    y_batch1 = downsample_2x(y_batch0)
    y_pyr = (y_batch0, y_batch1)

    # Condition: Fourier coordinates generated cleanly on both fine and coarse scales
    cond_np0 = make_fourier_coords(size, octaves=octaves)
    cond_np1 = make_fourier_coords(size // 2, octaves=octaves)
    cond0 = jnp.asarray(cond_np0)
    cond1 = jnp.asarray(cond_np1)
    cond_pyr = (cond0, cond1)
    in_cond_dim = cond0.shape[1]

    params = init_pyramid_deq(k_net, channels=channels, hidden_dim=hidden_dim, in_cond_dim=in_cond_dim, out_channels=out_channels)
    opt_state = adam_init(params)

    # Persistent pools for both fine (48x48) and coarse (24x24) scales
    pool0 = jax.random.normal(k_pool0, (pool_size, channels, size, size)) * 0.05
    pool1 = jax.random.normal(k_pool1, (pool_size, channels, size // 2, size // 2)) * 0.05

    @jax.jit
    def pool_train_step(p, opt_s, p0_state, p1_state, k, lr_cur):
        k_idx, k_dam, k_seed, k_n, k_pert = jax.random.split(k, 5)
        idx = jax.random.choice(k_idx, pool_size, (batch_size,), replace=False)
        batch_z0 = p0_state[idx]
        batch_z1 = p1_state[idx]

        # Reseed worst sample in batch
        preds = readout(batch_z0, p)
        sample_losses = jnp.mean((preds - y_batch0) ** 2, axis=(1, 2, 3))
        worst_idx = jnp.argmax(sample_losses)
        fresh0 = jax.random.normal(k_seed, (channels, size, size)) * 0.05
        fresh1 = downsample_2x(fresh0[None, ...])[0]
        batch_z0 = batch_z0.at[worst_idx].set(fresh0)
        batch_z1 = batch_z1.at[worst_idx].set(fresh1)

        # Apply damage mask to first 4 samples
        k_dam_split = jax.random.split(k_dam, batch_size)
        def dam_fn(k_d, i):
            d_type = jax.random.randint(k_d, (), 0, 3)
            d_type = jnp.where(i == 0, 0, d_type)  # Sample 0 gets dedicated half-wipe
            m0 = create_damage_mask(k_d, size, size, d_type)[None, :, :]
            keep0 = jnp.where(i < 4, m0, jnp.ones_like(m0))
            return keep0

        masks0 = jax.vmap(dam_fn)(k_dam_split, jnp.arange(batch_size))  # (B, 1, H, W)
        masks1 = downsample_2x(masks0)  # (B, 1, H/2, W/2)
        batch_z0_dam = batch_z0 * masks0
        batch_z1_dam = batch_z1 * masks1

        # DEQ PC-ALM stationary equilibrium solve
        deq_grads, total_loss, z_eq = compute_pyramid_deq_grads(
            p, (batch_z0_dam, batch_z1_dam), cond_pyr, y_pyr,
            steps=deq_steps, inner_steps=inner_steps, state_lr=state_lr, rho=rho, alpha=alpha, tv_weight=tv_weight,
        )

        # Multi-scale matched flow loss with PINNED intact pixels: N ~ Uniform(20, 40)
        def flow_loss_fn(p_curr):
            n_steps = jax.random.randint(k_n, (), 20, 41)
            z0_target_stop = jax.lax.stop_gradient(z_eq[0][:4])
            z1_target_stop = jax.lax.stop_gradient(z_eq[1][:4])
            m0_target = masks0[:4]
            m1_target = masks1[:4]
            dam_mask0 = 1.0 - m0_target
            dam_mask1 = 1.0 - m1_target
            dam_pixels0 = jnp.sum(dam_mask0) + 1e-6

            def flow_step(zc, _):
                d0, d1 = pyramid_delta(zc, cond_pyr, p_curr)
                z0_next = zc[0] + 0.5 * d0
                z1_next = zc[1] + 0.5 * d1
                z0_next = jnp.where(m0_target > 0.5, z0_target_stop, z0_next)
                z1_next = jnp.where(m1_target > 0.5, z1_target_stop, z1_next)
                return (z0_next, z1_next), (z0_next, z1_next)

            _, (z0_traj, z1_traj) = jax.lax.scan(flow_step, (batch_z0_dam[:4], batch_z1_dam[:4]), xs=None, length=40)
            z0_final = z0_traj[n_steps - 1]
            z1_final = z1_traj[n_steps - 1]

            # Void flow loss across both scales using Charbonnier Smooth L1 (no blurry L2 averaging!)
            l_flow0 = jnp.sum(dam_mask0 * jnp.sqrt((z0_final - z0_target_stop) ** 2 + 1e-6)) / (dam_pixels0 * channels)
            l_flow1 = jnp.sum(dam_mask1 * jnp.sqrt((z1_final - z1_target_stop) ** 2 + 1e-6)) / (jnp.sum(dam_mask1) * channels + 1e-6)
            l_flow = l_flow0 + 0.5 * l_flow1

            # High-frequency sharpening and contrast matching on fine void
            pred_flow = readout(z0_final, p_curr)
            y_sub = y_batch0[:4]
            l_recon_l1 = jnp.sum(dam_mask0 * jnp.sqrt((pred_flow - y_sub) ** 2 + 1e-6)) / dam_pixels0

            gx_pred = pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1]
            gy_pred = pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :]
            gx_y = y_sub[:, :, :, 1:] - y_sub[:, :, :, :-1]
            gy_y = y_sub[:, :, 1:] - y_sub[:, :, :-1, :]
            mask_gx = ((dam_mask0[:, :, :, 1:] + dam_mask0[:, :, :, :-1]) > 0.5).astype(jnp.float32)
            mask_gy = ((dam_mask0[:, :, 1:, :] + dam_mask0[:, :, :-1, :]) > 0.5).astype(jnp.float32)
            l_edge = (
                jnp.sum(mask_gx * jnp.sqrt((gx_pred - gx_y) ** 2 + 1e-6)) / (jnp.sum(mask_gx) + 1e-6)
                + jnp.sum(mask_gy * jnp.sqrt((gy_pred - gy_y) ** 2 + 1e-6)) / (jnp.sum(mask_gy) + 1e-6)
            )

            # 2nd-order Laplacian curvature (gentle sharpness guidance without contrast blowout)
            p_pad = jnp.pad(pred_flow, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
            y_pad = jnp.pad(y_sub, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
            lap_pred = p_pad[:, :, :-2, 1:-1] + p_pad[:, :, 2:, 1:-1] + p_pad[:, :, 1:-1, :-2] + p_pad[:, :, 1:-1, 2:] - 4.0 * pred_flow
            lap_y = y_pad[:, :, :-2, 1:-1] + y_pad[:, :, 2:, 1:-1] + y_pad[:, :, 1:-1, :-2] + y_pad[:, :, 1:-1, 2:] - 4.0 * y_sub
            l_lap = jnp.sum(dam_mask0 * jnp.sqrt((lap_pred - lap_y) ** 2 + 1e-6)) / dam_pixels0

            dam_px_sample = jnp.sum(dam_mask0, axis=(2, 3), keepdims=True) + 1e-6
            mu_pred = jnp.sum(dam_mask0 * pred_flow, axis=(2, 3), keepdims=True) / dam_px_sample
            var_pred = jnp.sum(dam_mask0 * (pred_flow - mu_pred) ** 2, axis=(2, 3), keepdims=True) / dam_px_sample
            std_pred = jnp.sqrt(var_pred + 1e-6)
            mu_y = jnp.sum(dam_mask0 * y_sub, axis=(2, 3), keepdims=True) / dam_px_sample
            var_y = jnp.sum(dam_mask0 * (y_sub - mu_y) ** 2, axis=(2, 3), keepdims=True) / dam_px_sample
            std_y = jnp.sqrt(var_y + 1e-6)
            l_contrast = jnp.mean(jnp.abs(std_pred - std_y))
            l_mean_match = jnp.mean(jnp.abs(mu_pred - mu_y))

            l_recon = 2.0 * l_recon_l1 + 2.5 * l_edge + 0.5 * l_lap + 0.5 * l_contrast + 4.0 * l_mean_match

            # Stationary constraints: Delta at target must be 0
            d0_target, d1_target = pyramid_delta((z0_target_stop, z1_target_stop), cond_pyr, p_curr)
            l_target_drift = jnp.mean(d0_target ** 2) + 0.5 * jnp.mean(d1_target ** 2)

            # Contractive restoring spring on perturbed target states
            noise0 = jax.random.normal(k_pert, z0_target_stop.shape) * 0.20
            noise1 = downsample_2x(noise0)
            z0_pert = z0_target_stop + noise0
            z1_pert = z1_target_stop + noise1
            d0_pert, d1_pert = pyramid_delta((z0_pert, z1_pert), cond_pyr, p_curr)
            z0_restored = z0_pert + 0.5 * d0_pert
            z1_restored = z1_pert + 0.5 * d1_pert
            l_spring = jnp.mean((z0_restored - z0_target_stop) ** 2) + 0.5 * jnp.mean((z1_restored - z1_target_stop) ** 2)

            loss = 2.0 * l_flow + 1.2 * l_recon + 1.0 * l_target_drift + 3.0 * l_spring
            return loss, (z0_final, z1_final)

        (flow_val, z_final_stepped), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(p)
        combined_grads = jax.tree_util.tree_map(lambda g1, g2: g1 + 1.0 * g2, deq_grads, flow_grads)

        p, opt_s = adam_apply(p, combined_grads, opt_s, lr=lr_cur)

        # Distill persistent pool: put stepped states back
        batch_z0_up = batch_z0.at[:4].set(jax.lax.stop_gradient(z_final_stepped[0]))
        batch_z0_up = batch_z0_up.at[4:].set(jax.lax.stop_gradient(z_eq[0][4:]))
        new_p0 = p0_state.at[idx].set(batch_z0_up)

        batch_z1_up = batch_z1.at[:4].set(jax.lax.stop_gradient(z_final_stepped[1]))
        batch_z1_up = batch_z1_up.at[4:].set(jax.lax.stop_gradient(z_eq[1][4:]))
        new_p1 = p1_state.at[idx].set(batch_z1_up)

        return p, opt_s, new_p0, new_p1, total_loss

    print(f"=== Starting Pyramid Multiscale DEQ on '{image_name}' (fine={size}x{size}, coarse={size//2}x{size//2}, steps={steps}) ===")
    history = {"step": [], "loss": [], "psnr": []}
    t0 = time.time()
    k_steps = jax.random.split(k_loop, steps)

    for s in range(1, steps + 1):
        if s <= 20:
            lr_s = lr * (s / 20.0)
        else:
            prog = (s - 20) / max(1, steps - 20)
            lr_s = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        params, opt_state, pool0, pool1, loss_val = pool_train_step(params, opt_state, pool0, pool1, k_steps[s - 1], lr_s)

        if s % 20 == 0 or s == 1 or s == steps:
            preds_all = readout(pool0, params)
            mses = [float(np.mean((np.asarray(preds_all[i]) - clean_np) ** 2)) for i in range(pool_size)]
            best_idx = int(np.argmin(mses))
            best_p = psnr(np.asarray(preds_all[best_idx]), clean_np)
            history["step"].append(s)
            history["loss"].append(float(loss_val))
            history["psnr"].append(best_p)
            print(f"[PYRAMID-DEQ] Step {s:3d}/{steps} | Loss: {float(loss_val):.5f} | Best Fine PSNR: {best_p:.2f} dB (idx {best_idx})")

    elapsed = time.time() - t0
    print(f"[PYRAMID-DEQ] Training finished in {elapsed:.2f}s.")

    # Find best equilibrium states in pool balancing clean PSNR and stationary drift
    preds_all = readout(pool0, params)
    mses = [float(np.mean((np.asarray(preds_all[i]) - clean_np) ** 2)) for i in range(pool_size)]
    psnrs_all = [psnr(np.asarray(preds_all[i]), clean_np) for i in range(pool_size)]
    d0_all, d1_all = pyramid_delta((pool0, pool1), cond_pyr, params)
    drifts = [float(np.mean(np.asarray(d0_all[i]) ** 2) + 0.5 * np.mean(np.asarray(d1_all[i]) ** 2)) for i in range(pool_size)]

    for i in range(pool_size):
        print(f"[POOL-SLOT] Slot {i:2d} | PSNR: {psnrs_all[i]:.2f} dB | Drift: {drifts[i]:.6f}")

    # Evaluate top candidates on inpainting benchmark to find the best contractive basin
    max_p = max(psnrs_all)
    valid_candidates = [i for i in range(pool_size) if psnrs_all[i] >= max_p - 0.5]
    top_candidates = sorted(valid_candidates, key=lambda i: drifts[i])[:6]
    candidate_scores = []
    mask_half0 = jnp.ones((1, 1, size, size), dtype=jnp.float32).at[:, :, :, size // 2:].set(0.0)
    tgt_clean_base = clean_np[0] * 255.0 if out_channels == 1 else np.transpose(clean_np, (1, 2, 0)) * 255.0

    for c_idx in top_candidates:
        cand_z = (pool0[c_idx:c_idx + 1], pool1[c_idx:c_idx + 1])
        zc0, zc1 = cand_z[0] * mask_half0, cand_z[1] * downsample_2x(mask_half0)
        k_m1 = downsample_2x(mask_half0)
        for _ in range(8):
            _, d1 = pyramid_delta((zc0, zc1), cond_pyr, params)
            zc1 = zc1 + 0.5 * d1
            zc1 = jnp.where(k_m1 > 0.5, cand_z[1], zc1)
        for _ in range(40):
            d0, d1 = pyramid_delta((zc0, zc1), cond_pyr, params)
            zc0 = zc0 + 0.5 * d0
            zc1 = zc1 + 0.5 * d1
            zc0 = jnp.where(mask_half0 > 0.5, cand_z[0], zc0)
            zc1 = jnp.where(k_m1 > 0.5, cand_z[1], zc1)
        p_eval = readout(zc0, params)[0]
        p_eval_np = np.asarray(p_eval[0] if out_channels == 1 else np.transpose(p_eval, (1, 2, 0)))
        diff = p_eval_np * 255.0 - tgt_clean_base
        h_psnr = float(10.0 * np.log10(255.0 ** 2 / (np.mean(diff ** 2) + 1e-10)))
        candidate_scores.append((h_psnr, c_idx))
        print(f"[CANDIDATE-EVAL] Slot {c_idx:2d} | Clean PSNR: {psnrs_all[c_idx]:.2f} dB | Drift: {drifts[c_idx]:.6f} | Half-Wipe PSNR: {h_psnr:.2f} dB")

    best_psnr, best_idx = max(candidate_scores, key=lambda x: x[0])
    print(f"[POOL-SELECT] Winning Slot: {best_idx} with Half-Wipe PSNR: {best_psnr:.2f} dB!")
    z_eq = (pool0[best_idx:best_idx + 1], pool1[best_idx:best_idx + 1])

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

        # Run Decimation Battery on Pyramid
        run_pyramid_decimation_battery(params, z_eq, cond_pyr, clean_np, out_channels, size, save_dir)

    return history


def run_pyramid_decimation_battery(
    params: dict,
    z_eq: tuple[jax.Array, jax.Array],
    cond_pyr: tuple[jax.Array, jax.Array],
    clean_np: np.ndarray,
    out_channels: int,
    size: int,
    save_dir: Path,
):
    """Executes decimation tests using forward Pyramid NCA relaxation."""
    print("=== Running Pyramid Decimation + Forward NCA Self-Healing Battery ===")

    lam_max = compute_pyramid_spectral_radius(params, z_eq, cond_pyr)
    print(f"[THEORY] Spectral radius of joint pyramid relaxation at z_eq: |lambda_max| = {lam_max:.4f}")

    def forward_pyramid_relax(z_start_pyr, keep_mask0, n_steps=40, step_size=0.5):
        from scipy.ndimage import distance_transform_edt

        zc0, zc1 = z_start_pyr
        keep_mask1 = downsample_2x(keep_mask0)
        dam_mask0 = 1.0 - keep_mask0

        dam_np = np.asarray(dam_mask0[0, 0])
        max_depth = float(distance_transform_edt(dam_np).max()) if float(dam_np.max()) > 0 else 1.0
        # In pyramid, coarse scale doubles propagation speed: min_steps drops by half!
        min_steps = max(5, int(max_depth / 2.0))

        # Boundary seam mask on fine scale with edge padding
        m_padded = jnp.pad(keep_mask0, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
        up = m_padded[:, :, :size, 1:size + 1]
        down = m_padded[:, :, 2:, 1:size + 1]
        left = m_padded[:, :, 1:size + 1, :size]
        right = m_padded[:, :, 1:size + 1, 2:]
        seam_mask = ((up + down + left + right) > 0.5) & (keep_mask0 < 0.5)
        seam_weight = float(jnp.sum(seam_mask)) + 1e-6

        # Phase 1: Hierarchical Coarse Warmup (wake up the coarse compass across the void)
        for _ in range(8):
            _, d1 = pyramid_delta((zc0, zc1), cond_pyr, params)
            zc1 = zc1 + step_size * d1
            zc1 = jnp.where(keep_mask1 > 0.5, z_eq[1], zc1)

        history0 = [zc0]
        seam_errors = []
        residuals = []
        settle_count = 0

        for s in range(1, n_steps + 1):
            d0, d1 = pyramid_delta((zc0, zc1), cond_pyr, params)
            res = float(jnp.mean((dam_mask0 * d0) ** 2))
            residuals.append(res)

            effective_lr = step_size

            zc0 = zc0 + effective_lr * d0
            zc1 = zc1 + effective_lr * d1
            zc0 = jnp.where(keep_mask0 > 0.5, z_eq[0], zc0)
            zc1 = jnp.where(keep_mask1 > 0.5, z_eq[1], zc1)
            history0.append(zc0)

            seam_err = float(jnp.sum(jnp.abs(zc0 - z_eq[0]) * seam_mask) / seam_weight)
            seam_errors.append(seam_err)

            if s >= 36 and res < 1e-7:
                break

        t_final = len(history0) - 1
        indices = np.linspace(0, t_final, 5, dtype=int)
        print(f"[PYRAMID-RELAX] max_depth: {max_depth:.1f}, min_steps: {min_steps} | Converged at step {t_final} (seam_err: {seam_errors[-1]:.5f}, res: {residuals[-1]:.6f}) | linspace frames: {indices.tolist()}")

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

    snaps_half = forward_pyramid_relax((z_half0, z_half1), mask_half0, n_steps=40, step_size=0.5)
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

    snaps_circle = forward_pyramid_relax((z_circle0, z_circle1), jnp.asarray(mask_circle0), n_steps=40, step_size=0.5)
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

    snaps_pepper = forward_pyramid_relax((z_pepper0, z_pepper1), jnp.asarray(mask_pepper0), n_steps=40, step_size=0.5)
    imgs_pepper = [to_img(s) for s in snaps_pepper]
    if out_channels == 1:
        imgs_pepper = [np.stack([im, im, im], axis=-1) for im in imgs_pepper]
    strip_pepper = np.concatenate(imgs_pepper, axis=1)
    Image.fromarray(strip_pepper).save(save_dir / "regen_pepper_strip.png")

    tgt_clean = (clean_np[0] * 255.0) if out_channels == 1 else np.transpose(clean_np, (1, 2, 0)) * 255.0

    def eval_metric(name, strip_path):
        strip_im = Image.open(strip_path).convert("L" if out_channels == 1 else "RGB")
        strip_arr = np.array(strip_im, dtype=float)
        h, w = strip_arr.shape[:2]
        cols = w // h
        pred = strip_arr[:, (cols - 1) * h : cols * h]
        diff = pred - tgt_clean
        mse = float(np.mean(diff ** 2))
        psnr_val = 10.0 * np.log10(255.0 ** 2 / (mse + 1e-10))
        corr_val = float(np.corrcoef(pred.flatten(), tgt_clean.flatten())[0, 1])
        print(f"[REGEN-EVAL] {name:10s} | PSNR: {psnr_val:.2f} dB | Corr: {corr_val:.4f} | MSE: {mse:.2f}")
        return psnr_val

    eval_metric("half_wipe", save_dir / "regen_half_wipe_strip.png")
    eval_metric("crater", save_dir / "regen_crater_strip.png")
    eval_metric("pepper", save_dir / "regen_pepper_strip.png")

    # Save 6x upscaled strips
    for strip_name, strip_arr in [
        ("regen_half_wipe_strip", strip_half),
        ("regen_crater_strip", strip_circle),
        ("regen_pepper_strip", strip_pepper),
    ]:
        im = Image.fromarray(strip_arr)
        im_large = im.resize((im.width * 6, im.height * 6), Image.Resampling.NEAREST)
        im_large.save(save_dir / f"{strip_name}_large.png")

    print(f"Saved Pyramid decimation strips to {save_dir}")
