from __future__ import annotations

import math
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .deq_conditioned import (
    init_conditioned_deq,
    load_target_image,
    nca_cond_delta,
    readout,
)
from .dip_experiment import adam_apply, adam_init, psnr
from .nca_experiment import extract_discrete_segmentation, extract_segmentation_pca


def create_damage_mask(key: jax.Array, h: int, w: int, damage_type: int) -> jax.Array:
    """Creates a binary mask (1 = keep, 0 = destroy) based on damage_type (0=half, 1=circle, 2=box)."""
    k_dir, k_pos = jax.random.split(key)

    # damage_type 0: Half plane wipe (0: top, 1: bottom, 2: left, 3: right)
    half_dir = jax.random.randint(k_dir, (), 0, 4)
    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")

    mask_top = yy >= (h // 2)
    mask_bot = yy < (h // 2)
    mask_left = xx >= (w // 2)
    mask_right = xx < (w // 2)

    half_mask = jnp.where(
        half_dir == 0, mask_top,
        jnp.where(
            half_dir == 1, mask_bot,
            jnp.where(half_dir == 2, mask_left, mask_right)
        )
    )

    # damage_type 1: Circular crater
    cy = jax.random.randint(k_pos, (), h // 4, 3 * h // 4)
    cx = jax.random.randint(k_dir, (), w // 4, 3 * w // 4)
    radius = max(8.0, min(h, w) / 3.0)
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    circle_mask = dist_sq >= (radius ** 2)

    # damage_type 2: Box cutout
    b_size = max(10, min(h, w) // 3)
    top = jax.random.randint(k_pos, (), 0, h - b_size)
    left = jax.random.randint(k_dir, (), 0, w - b_size)
    box_mask = ~((yy >= top) & (yy < top + b_size) & (xx >= left) & (xx < left + b_size))

    mask = jnp.where(damage_type == 0, half_mask, jnp.where(damage_type == 1, circle_mask, box_mask))
    return mask.astype(jnp.float32)


def apply_pool_damage(key: jax.Array, batch_z: jax.Array, num_damaged: int = 2) -> jax.Array:
    """Damages the first num_damaged samples in batch_z with severe decimation."""
    b, c, h, w = batch_z.shape
    keys = jax.random.split(key, b)

    def damage_one(k, z_single, i):
        k_type, k_mask = jax.random.split(k)
        d_type = jax.random.randint(k_type, (), 0, 3)
        mask = create_damage_mask(k_mask, h, w, d_type)  # (H, W)
        mask = mask[None, :, :]  # (1, H, W)
        # Apply damage only if index i < num_damaged
        return jnp.where(i < num_damaged, z_single * mask, z_single)

    indices = jnp.arange(b)
    damaged = jax.vmap(damage_one)(keys, batch_z, indices)
    return damaged


def rollout_blind_nca(
    params: dict,
    batch_z: jax.Array,
    cond: jax.Array,
    key: jax.Array,
    steps: int = 32,
    step_size: float = 0.5,
) -> jax.Array:
    """True Distill-style blind forward rollout with stochastic cell updates."""
    def step_fn(zc, k):
        delta = nca_cond_delta(zc, cond, params)
        mask = jax.random.bernoulli(k, p=0.5, shape=(zc.shape[0], 1, zc.shape[2], zc.shape[3])).astype(jnp.float32)
        z_next = zc + step_size * delta * mask
        return z_next, None

    keys = jax.random.split(key, steps)
    z_final, _ = jax.lax.scan(step_fn, batch_z, keys)
    return z_final


def run_pool_experiment(
    image_name: str = "camera",
    pool_size: int = 32,
    batch_size: int = 8,
    steps: int = 400,
    lr: float = 2e-3,
    channels: int = 16,
    hidden_dim: int = 64,
    size: int = 48,
    deq_steps: int = 32,
    step_size: float = 0.5,
    tv_weight: float = 0.02,
    num_damaged_per_batch: int = 2,
    seed: int = 42,
    save_dir: Path | None = None,
) -> dict:
    key = jax.random.PRNGKey(seed)
    k_net, k_pool, k_loop = jax.random.split(key, 3)

    clean_np, out_channels = load_target_image(image_name, size=size)
    y_target = jnp.asarray(clean_np[None, ...], dtype=jnp.float32)

    # Condition: normalized 2D coordinate grid (x, y)
    yy, xx = np.mgrid[:size, :size].astype(np.float32) / float(size)
    cond_np = np.stack([xx, yy], axis=0)[None, ...]
    cond = jnp.asarray(cond_np)

    params = init_conditioned_deq(k_net, channels=channels, hidden_dim=hidden_dim, in_cond_dim=2, out_channels=out_channels)
    opt_state = adam_init(params)

    # Initialize persistent pool with small noise
    pool = jax.random.normal(k_pool, (pool_size, channels, size, size)) * 0.05

    @jax.jit
    def pool_train_step(p, opt_s, pool_state, k):
        k_idx, k_dam, k_seed, k_roll = jax.random.split(k, 4)
        # Sample random batch
        idx = jax.random.choice(k_idx, pool_size, (batch_size,), replace=False)
        batch_z = pool_state[idx]

        # Reseed worst sample: find sample with highest loss in batch
        preds = readout(batch_z, p)
        sample_losses = jnp.mean((preds - y_target) ** 2, axis=(1, 2, 3))
        worst_in_batch = jnp.argmax(sample_losses)
        fresh_seed = jax.random.normal(k_seed, (channels, size, size)) * 0.05
        batch_z = batch_z.at[worst_in_batch].set(fresh_seed)

        # Apply decimation damage to first num_damaged samples
        batch_z_dam = apply_pool_damage(k_dam, batch_z, num_damaged=num_damaged_per_batch)

        # Blind rollout & gradient computation
        def loss_fn(p_curr):
            z_settled = rollout_blind_nca(
                p_curr, batch_z_dam, cond, k_roll,
                steps=deq_steps, step_size=step_size,
            )
            pred_y = readout(z_settled, p_curr)
            l_sup = 0.5 * jnp.mean((pred_y - y_target) ** 2)

            diff_x = z_settled[:, :, :, 1:] - z_settled[:, :, :, :-1]
            diff_y = z_settled[:, :, 1:, :] - z_settled[:, :, :-1, :]
            l_tv = tv_weight * (jnp.mean(jnp.sqrt(diff_x ** 2 + 1e-6)) + jnp.mean(jnp.sqrt(diff_y ** 2 + 1e-6)))

            return l_sup + l_tv, z_settled

        (total_loss, z_final), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        p, opt_s = adam_apply(p, grads, opt_s, lr=lr)

        # Update persistent pool with relaxed states (stop grad)
        new_pool = pool_state.at[idx].set(jax.lax.stop_gradient(z_final))
        return p, opt_s, new_pool, total_loss

    print(f"=== Starting Regenerative Sample-Pool DEQ on '{image_name}' (pool={pool_size}, batch={batch_size}, steps={steps}, size={size}x{size}) ===")
    history = {"step": [], "loss": [], "psnr": []}
    t0 = time.time()
    k_steps = jax.random.split(k_loop, steps)

    for s in range(1, steps + 1):
        params, opt_state, pool, loss_val = pool_train_step(params, opt_state, pool, k_steps[s - 1])

        if s % 20 == 0 or s == 1 or s == steps:
            # Evaluate best sample in pool
            preds_all = readout(pool, params)
            mses = [float(np.mean((np.asarray(preds_all[i]) - clean_np) ** 2)) for i in range(pool_size)]
            best_idx = int(np.argmin(mses))
            best_p = psnr(np.asarray(preds_all[best_idx]), clean_np)
            history["step"].append(s)
            history["loss"].append(float(loss_val))
            history["psnr"].append(best_p)
            print(f"[POOL-DEQ] Step {s:3d}/{steps} | Loss: {float(loss_val):.5f} | Best Pool PSNR: {best_p:.2f} dB (idx {best_idx})")

    elapsed = time.time() - t0
    print(f"[POOL-DEQ] Training finished in {elapsed:.2f}s.")

    # Find the best equilibrium state in pool
    preds_all = readout(pool, params)
    mses = [float(np.mean((np.asarray(preds_all[i]) - clean_np) ** 2)) for i in range(pool_size)]
    best_idx = int(np.argmin(mses))
    z_eq = pool[best_idx:best_idx + 1]

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save target and clean reconstruction
        pred_clean = readout(z_eq, params)
        pred_np = np.asarray(pred_clean[0])

        if out_channels == 1:
            Image.fromarray((clean_np[0] * 255.0).astype(np.uint8)).save(save_dir / "target.png")
            Image.fromarray(np.clip(pred_np[0] * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_recon.png")
        else:
            Image.fromarray(np.clip(np.transpose(clean_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "target.png")
            Image.fromarray(np.clip(np.transpose(pred_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_recon.png")

        # Segmentation map
        hidden_np = np.asarray(z_eq[0])
        seg_pca = extract_segmentation_pca(hidden_np)
        Image.fromarray(seg_pca).save(save_dir / "deq_segmentation_pca.png")

        labels_2d, discrete_rgb, _ = extract_discrete_segmentation(hidden_np, n_clusters=4, seed=seed)
        Image.fromarray(discrete_rgb).save(save_dir / "deq_discrete_seg.png")

        # === DECIMATION & REGENERATION BATTERY ===
        run_decimation_battery(params, z_eq, cond, clean_np, out_channels, size, save_dir, step_size=step_size)

    return history


def run_decimation_battery(
    params: dict,
    z_eq: jax.Array,
    cond: jax.Array,
    clean_np: np.ndarray,
    out_channels: int,
    size: int,
    save_dir: Path,
    step_size: float = 0.5,
):
    """Executes Distill-style decimation tests: Half-wipe, Crater hole, and Pepper noise."""
    print("=== Running Distill-style Decimation and Regeneration Battery ===")

    def autonomous_relax(z_start, n_steps=60, step_size=0.5):
        snapshots = [z_start]
        checkpoint_steps = [10, 20, 40, n_steps]
        zc = z_start
        for s in range(1, n_steps + 1):
            delta = nca_cond_delta(zc, cond, params)
            zc = zc + step_size * delta
            if s in checkpoint_steps:
                snapshots.append(zc)
        return snapshots

    def to_img(z_state):
        pred = readout(z_state, params)
        p_np = np.asarray(pred[0])
        if out_channels == 1:
            return np.clip(p_np[0] * 255.0, 0, 255).astype(np.uint8)
        else:
            return np.clip(np.transpose(p_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)

    # --- Test 1: Half-Wipe (cut off right half) ---
    mask_half = jnp.ones((1, 1, size, size), dtype=jnp.float32)
    mask_half = mask_half.at[:, :, :, size // 2:].set(0.0)
    z_half = z_eq * mask_half

    snaps_half = autonomous_relax(z_half, n_steps=60)
    imgs_half = [to_img(s) for s in snaps_half]
    if out_channels == 1:
        imgs_half = [np.stack([im, im, im], axis=-1) for im in imgs_half]
    strip_half = np.concatenate(imgs_half, axis=1)
    Image.fromarray(strip_half).save(save_dir / "regen_half_wipe_strip.png")

    # --- Test 2: Center Crater (circular blackout) ---
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = size / 2.0, size / 2.0
    r = size / 3.0
    mask_circle = ((yy - cy) ** 2 + (xx - cx) ** 2 >= r ** 2).astype(np.float32)[None, None, ...]
    z_circle = z_eq * jnp.asarray(mask_circle)

    snaps_circle = autonomous_relax(z_circle, n_steps=60)
    imgs_circle = [to_img(s) for s in snaps_circle]
    if out_channels == 1:
        imgs_circle = [np.stack([im, im, im], axis=-1) for im in imgs_circle]
    strip_circle = np.concatenate(imgs_circle, axis=1)
    Image.fromarray(strip_circle).save(save_dir / "regen_crater_strip.png")

    # --- Test 3: Pepper Noise (50% random pixel wipe) ---
    np.random.seed(42)
    mask_pepper = (np.random.rand(1, 1, size, size) > 0.5).astype(np.float32)
    z_pepper = z_eq * jnp.asarray(mask_pepper)

    snaps_pepper = autonomous_relax(z_pepper, n_steps=60)
    imgs_pepper = [to_img(s) for s in snaps_pepper]
    if out_channels == 1:
        imgs_pepper = [np.stack([im, im, im], axis=-1) for im in imgs_pepper]
    strip_pepper = np.concatenate(imgs_pepper, axis=1)
    Image.fromarray(strip_pepper).save(save_dir / "regen_pepper_strip.png")

    print(f"Saved decimation regeneration strips (half-wipe, crater, pepper) to {save_dir}")
