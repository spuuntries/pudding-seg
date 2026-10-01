from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .dip_experiment import adam_apply, adam_init, create_synthetic_target, psnr
from .nca_pcalm import (
    NCAParams,
    compute_nca_grads,
    compute_trajectory_grads,
    forward_trajectory,
    init_nca_params,
    nca_step,
)


def extract_segmentation_pca(hidden_states: np.ndarray) -> np.ndarray:
    """Projects hidden channels (C-1, H, W) to 3 RGB channels via PCA to visualize neural grouping."""
    c, h, w = hidden_states.shape
    features = hidden_states.reshape(c, -1).T.copy()
    features = features - np.mean(features, axis=0, keepdims=True)

    # SVD for top 3 components
    _, _, vh = np.linalg.svd(features, full_matrices=False)
    proj = features @ vh[:3].T  # (H*W, 3)

    # Normalize to [0, 255]
    p_min = proj.min(axis=0, keepdims=True)
    p_max = proj.max(axis=0, keepdims=True)
    norm = (proj - p_min) / np.maximum(p_max - p_min, 1e-6)
    rgb = (norm.reshape(h, w, 3) * 255.0).astype(np.uint8)
    return rgb


def extract_discrete_segmentation(
    hidden_states: np.ndarray,
    n_clusters: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clusters hidden channels (C, H, W) into n_clusters discrete masks.

    Returns:
        labels_2d: (H, W) integer cluster map in [0, n_clusters-1]
        discrete_rgb: (H, W, 3) uint8 image with distinct colors per cluster
        masks_mosaic: (H, W * n_clusters, 3) uint8 image of binary masks side-by-side
    """
    from scipy.cluster.vq import kmeans2

    c, h, w = hidden_states.shape
    features = hidden_states.reshape(c, -1).T.copy()
    f_mean = np.mean(features, axis=0, keepdims=True)
    f_std = np.std(features, axis=0, keepdims=True) + 1e-6
    feats_norm = (features - f_mean) / f_std

    _, labels = kmeans2(feats_norm, k=n_clusters, minit="points", seed=seed)
    labels_2d = labels.reshape(h, w)

    palette = np.array([
        [230, 25, 75],    # Red
        [60, 180, 75],    # Green
        [255, 225, 25],   # Yellow
        [0, 130, 200],    # Blue
        [245, 130, 48],   # Orange
        [145, 30, 180],   # Purple
        [70, 240, 240],   # Cyan
        [240, 50, 230],   # Magenta
        [210, 245, 60],   # Lime
        [250, 190, 212],  # Pink
    ], dtype=np.uint8)

    discrete_rgb = palette[labels_2d % len(palette)]

    mask_strips = []
    for k in range(n_clusters):
        mask_k = (labels_2d == k).astype(np.uint8) * 255
        mask_rgb = np.stack([mask_k, mask_k, mask_k], axis=-1)
        mask_strips.append(mask_rgb)
    masks_mosaic = np.concatenate(mask_strips, axis=1)

    return labels_2d, discrete_rgb, masks_mosaic



def run_nca_deq(
    steps: int = 150,
    lr: float = 3e-3,
    channels: int = 16,
    hidden_dim: int = 64,
    size: int = 32,
    deq_steps: int = 8,
    inner_steps: int = 2,
    state_lr: float = 0.05,
    alpha: float = 0.1,
    rho: float = 1.0,
    seed: int = 42,
    save_dir: Path | None = None,
) -> dict:
    key = jax.random.PRNGKey(seed)
    k_net, k_init = jax.random.split(key)

    # Target image: (1, 1, H, W)
    clean_np = create_synthetic_target(size)
    y_target = jnp.asarray(clean_np[None, None, ...], dtype=jnp.float32)

    # Initial state grid z_0: (1, C, H, W) with small noise
    z_init = jax.random.normal(k_init, (1, channels, size, size)) * 0.1

    params = init_nca_params(k_net, channels=channels, hidden_dim=hidden_dim)
    opt_state = adam_init(params)

    @jax.jit
    def train_step(p, opt_s, z_curr):
        grads, loss, z_next = compute_nca_grads(
            p,
            z_curr,
            y_target,
            steps=deq_steps,
            inner_steps=inner_steps,
            state_lr=state_lr,
            rho=rho,
            alpha=alpha,
            out_channels=1,
        )
        p, opt_s = adam_apply(p, grads, opt_s, lr=lr)
        return p, opt_s, loss, z_next

    history = {"step": [], "loss": [], "psnr": []}
    print(f"=== Starting NCA DEQ PC-ALM (channels={channels}, steps={steps}, size={size}x{size}) ===")
    t0 = time.time()

    z_curr = z_init
    for s in range(1, steps + 1):
        params, opt_state, loss, z_curr = train_step(params, opt_state, z_curr)

        if s % 10 == 0 or s == 1 or s == steps:
            pred_img = np.asarray(z_curr[0, 0])
            p_val = psnr(pred_img, clean_np)
            history["step"].append(s)
            history["loss"].append(float(loss))
            history["psnr"].append(p_val)
            print(f"[NCA-DEQ] Step {s:3d}/{steps} | Loss: {float(loss):.5f} | PSNR: {p_val:.2f} dB")

    elapsed = time.time() - t0
    print(f"[NCA-DEQ] Done in {elapsed:.2f}s.")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        # 1. Target
        Image.fromarray((clean_np * 255.0).astype(np.uint8)).save(save_dir / "target.png")

        # 2. Reconstructed image
        recon_np = np.clip(np.asarray(z_curr[0, 0]) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(recon_np).save(save_dir / "nca_recon.png")

        # 3. Neural Grouping segmentation from hidden channels
        hidden_np = np.asarray(z_curr[0, 1:])  # (C-1, H, W)
        seg_rgb = extract_segmentation_pca(hidden_np)
        Image.fromarray(seg_rgb).save(save_dir / "nca_segmentation_pca.png")
        print(f"Saved reconstruction and segmentation maps to {save_dir}")

    return history


def run_nca_trajectory(
    steps: int = 150,
    nca_steps: int = 16,
    budget: int = 6,
    inner_steps: int = 2,
    lr: float = 3e-3,
    channels: int = 16,
    hidden_dim: int = 64,
    size: int = 32,
    state_lr: float = 0.05,
    alpha: float = 0.1,
    rho: float = 1.0,
    seed: int = 42,
    save_dir: Path | None = None,
) -> dict:
    key = jax.random.PRNGKey(seed)
    k_net, k_init = jax.random.split(key)

    clean_np = create_synthetic_target(size)
    y_target = jnp.asarray(clean_np[None, None, ...], dtype=jnp.float32)

    # Seed state: fixed grid with center seed activation + gentle spatial coordinates
    # Channels 0..1: normalized coordinate grid (x, y)
    yy, xx = np.mgrid[:size, :size].astype(np.float32) / float(size)
    z_seed_np = np.zeros((1, channels, size, size), dtype=np.float32)
    z_seed_np[0, 0] = xx
    z_seed_np[0, 1] = yy
    z_seed_np[0, 2, size // 2, size // 2] = 1.0  # Center seed
    z_0 = jnp.asarray(z_seed_np)

    params = init_nca_params(k_net, channels=channels, hidden_dim=hidden_dim)
    opt_state = adam_init(params)

    @jax.jit
    def train_step(p, opt_s):
        grads, loss, _ = compute_trajectory_grads(
            p,
            z_0,
            y_target,
            steps=nca_steps,
            budget=budget,
            inner_steps=inner_steps,
            state_lr=state_lr,
            rho=rho,
            alpha=alpha,
            out_channels=1,
        )
        p, opt_s = adam_apply(p, grads, opt_s, lr=lr)
        return p, opt_s, loss

    eval_fn = jax.jit(lambda p: forward_trajectory(z_0, p, nca_steps)[-1])

    history = {"step": [], "loss": [], "psnr": []}
    print(f"=== Starting Trajectory NCA PC-ALM (T={nca_steps}, channels={channels}, steps={steps}, size={size}x{size}) ===")
    t0 = time.time()

    for s in range(1, steps + 1):
        params, opt_state, loss = train_step(params, opt_state)

        if s % 10 == 0 or s == 1 or s == steps:
            final_state = eval_fn(params)
            pred_img = np.asarray(final_state[0, 0])
            p_val = psnr(pred_img, clean_np)
            history["step"].append(s)
            history["loss"].append(float(loss))
            history["psnr"].append(p_val)
            print(f"[NCA-Trajectory] Step {s:3d}/{steps} | Loss: {float(loss):.5f} | PSNR: {p_val:.2f} dB")

    elapsed = time.time() - t0
    print(f"[NCA-Trajectory] Done in {elapsed:.2f}s.")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((clean_np * 255.0).astype(np.uint8)).save(save_dir / "target.png")

        final_state = eval_fn(params)
        recon_np = np.clip(np.asarray(final_state[0, 0]) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(recon_np).save(save_dir / "nca_recon.png")

        hidden_np = np.asarray(final_state[0, 1:])
        seg_rgb = extract_segmentation_pca(hidden_np)
        Image.fromarray(seg_rgb).save(save_dir / "nca_segmentation_pca.png")
        print(f"Saved reconstruction and segmentation maps to {save_dir}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--nca-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--size", type=int, default=32)
    args = parser.parse_args()

    run_nca_trajectory(
        steps=args.steps,
        nca_steps=args.nca_steps,
        lr=args.lr,
        channels=args.channels,
        size=args.size,
        save_dir=Path("results/nca_trajectory_run"),
    )
