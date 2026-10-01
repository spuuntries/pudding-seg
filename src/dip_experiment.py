from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .conv_pcalm import (
    ConvLayer,
    Params,
    compute_grads,
    forward,
    init_conv_params,
    skip_mask,
)


def adam_init(params: Params):
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    t = jnp.asarray(0, dtype=jnp.int32)
    return m, v, t


def adam_apply(
    params: Params,
    grads: Params,
    state,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Params, tuple]:
    m, v, t = state
    t = t + 1
    m = jax.tree_util.tree_map(lambda mi, gi: beta1 * mi + (1.0 - beta1) * gi, m, grads)
    v = jax.tree_util.tree_map(lambda vi, gi: beta2 * vi + (1.0 - beta2) * (gi * gi), v, grads)
    bc1 = 1.0 - beta1**t
    bc2 = 1.0 - beta2**t
    updates = jax.tree_util.tree_map(lambda mi, vi: (mi / bc1) / (jnp.sqrt(vi / bc2) + eps), m, v)
    params = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, updates)
    return params, (m, v, t)


def create_synthetic_target(size: int = 32) -> np.ndarray:
    """Creates a geometric multi-shape target image in [0, 1]."""
    img = np.zeros((size, size), dtype=np.float32)
    # Circle in center
    yy, xx = np.mgrid[:size, :size]
    cy, cx = size // 2, size // 2
    r = size // 4
    mask_circle = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r**2)
    img[mask_circle] = 0.8

    # Square in top left
    s_end = size // 3
    img[2:s_end, 2:s_end] = 0.5

    # Ramp gradient bar at bottom
    img[-size // 5 :, :] = np.linspace(0.1, 0.9, size)[None, :]
    return img


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def run_dip(
    method: str = "pcalm",
    steps: int = 200,
    lr: float = 1e-3,
    depth: int = 6,
    channels: int = 16,
    in_channels: int = 4,
    size: int = 32,
    noise_sigma: float = 0.1,
    budget: int = 4,
    inner_steps: int = 2,
    state_lr: float = 0.05,
    alpha: float = 0.1,
    seed: int = 42,
    save_dir: Path | None = None,
) -> dict:
    key = jax.random.PRNGKey(seed)
    k_net, k_noise, k_input = jax.random.split(key, 3)

    # Clean target image: (1, 1, H, W)
    clean_np = create_synthetic_target(size)
    clean_jax = jnp.asarray(clean_np[None, None, ...], dtype=jnp.float32)

    # Noisy observation: y = clean + noise
    noise_np = np.random.default_rng(seed).normal(0.0, noise_sigma, clean_np.shape).astype(np.float32)
    noisy_np = np.clip(clean_np + noise_np, 0.0, 1.0)
    y_target = jnp.asarray(noisy_np[None, None, ...], dtype=jnp.float32)

    # Fixed DIP input code z ~ N(0, 1): (1, in_channels, H, W)
    z_input = jax.random.normal(k_input, (1, in_channels, size, size))

    # Network params
    params = init_conv_params(
        k_net,
        depth=depth,
        channels=channels,
        in_channels=in_channels,
        out_channels=1,
    )
    skips = skip_mask(depth)
    opt_state = adam_init(params)

    history = {
        "step": [],
        "loss": [],
        "psnr_noisy": [],
        "psnr_clean": [],
    }

    print(f"[{method.upper()}] Starting DIP optimization (depth={depth}, steps={steps}, size={size}x{size})...")
    t0 = time.time()

    @jax.jit
    def train_step(p, opt_s):
        grads, loss = compute_grads(
            method,
            p,
            skips,
            z_input,
            y_target,
            state_lr=state_lr,
            rho=1.0,
            alpha=alpha,
            budget=budget,
            inner_steps=inner_steps,
        )
        p, opt_s = adam_apply(p, grads, opt_s, lr=lr)
        return p, opt_s, loss

    eval_fn = jax.jit(lambda p: forward(p, skips, z_input, jax.nn.relu)[1])

    for s in range(1, steps + 1):
        params, opt_state, loss = train_step(params, opt_state)

        if s % 10 == 0 or s == 1 or s == steps:
            pred = eval_fn(params)
            pred_np = np.asarray(pred[0, 0])
            p_noisy = psnr(pred_np, noisy_np)
            p_clean = psnr(pred_np, clean_np)

            history["step"].append(s)
            history["loss"].append(loss)
            history["psnr_noisy"].append(p_noisy)
            history["psnr_clean"].append(p_clean)

            print(
                f"[{method.upper()}] Step {s:3d}/{steps} | Loss: {loss:.5f} | "
                f"PSNR(noisy): {p_noisy:.2f} dB | PSNR(clean): {p_clean:.2f} dB"
            )

    elapsed = time.time() - t0
    print(f"[{method.upper()}] Done in {elapsed:.2f}s.")

    # Save final reconstruction if save_dir provided
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        _, final_pred = forward(params, skips, z_input, jax.nn.relu)
        final_np = np.clip(np.asarray(final_pred[0, 0]) * 255.0, 0, 255).astype(np.uint8)
        clean_out = (clean_np * 255.0).astype(np.uint8)
        noisy_out = (noisy_np * 255.0).astype(np.uint8)

        Image.fromarray(clean_out).save(save_dir / "target_clean.png")
        Image.fromarray(noisy_out).save(save_dir / "target_noisy.png")
        Image.fromarray(final_np).save(save_dir / f"recon_{method}.png")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["bp", "pc", "pcalm"], default="pcalm")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=2)
    parser.add_argument("--state-lr", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    run_dip(
        method=args.method,
        steps=args.steps,
        lr=args.lr,
        depth=args.depth,
        channels=args.channels,
        size=args.size,
        budget=args.budget,
        inner_steps=args.inner_steps,
        state_lr=args.state_lr,
        alpha=args.alpha,
        save_dir=Path("results/dip_first_run"),
    )
