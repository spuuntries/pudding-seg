from __future__ import annotations

import math
import jax
import jax.numpy as jnp
import numpy as np

from .nca_pcalm import NCAParams, perceive


class DEQModelParams:
    def __init__(self, nca_params: NCAParams, w_out: jax.Array, b_out: jax.Array):
        self.nca = nca_params
        self.w_out = w_out  # (out_c, hidden_c, 1, 1)
        self.b_out = b_out  # (out_c, 1, 1)


def init_conditioned_deq(
    key: jax.Array,
    channels: int = 16,
    hidden_dim: int = 64,
    in_cond_dim: int = 2,  # e.g. (x, y) coordinates
    out_channels: int = 1,
):
    k1, k2, k3 = jax.random.split(key, 3)
    # Perception has 3*channels + in_cond_dim
    total_in = 3 * channels + in_cond_dim
    std1 = 1.0 / math.sqrt(total_in)

    w1 = jax.random.normal(k1, (hidden_dim, total_in, 1, 1)) * std1
    b1 = jnp.zeros((hidden_dim, 1, 1))
    w2 = jnp.zeros((channels, hidden_dim, 1, 1))
    b2 = jnp.zeros((channels, 1, 1))

    # Readout head: channels -> out_channels
    std_out = 1.0 / math.sqrt(channels)
    w_out = jax.random.normal(k2, (out_channels, channels, 1, 1)) * std_out
    b_out = jnp.zeros((out_channels, 1, 1))

    return {
        "w1": w1, "b1": b1,
        "w2": w2, "b2": b2,
        "w_out": w_out, "b_out": b_out,
    }


def nca_cond_delta(z: jax.Array, cond: jax.Array, params: dict) -> jax.Array:
    """NCA step conditioned on input x (e.g. coordinates/noise)."""
    p = perceive(z)  # (B, 3*C, H, W)
    p_full = jnp.concatenate([p, cond], axis=1)  # (B, 3*C + cond_dim, H, W)

    h = jnp.tensordot(p_full, params["w1"][:, :, 0, 0], axes=([1], [1]))
    h = jnp.transpose(h, (0, 3, 1, 2)) + params["b1"]
    h = jax.nn.relu(h)

    delta = jnp.tensordot(h, params["w2"][:, :, 0, 0], axes=([1], [1]))
    delta = jnp.transpose(delta, (0, 3, 1, 2)) + params["b2"]
    return delta


def readout(z: jax.Array, params: dict) -> jax.Array:
    """Projects hidden equilibrium state to image."""
    out = jnp.tensordot(z, params["w_out"][:, :, 0, 0], axes=([1], [1]))
    return jnp.transpose(out, (0, 3, 1, 2)) + params["b_out"]


def deq_energy(
    params: dict,
    z_eq: jax.Array,
    dual_eq: jax.Array,
    cond: jax.Array,
    y: jax.Array,
    rho: float = 1.0,
    tv_weight: float = 0.05,
) -> jax.Array:
    batch_size = z_eq.shape[0]
    pred_y = readout(z_eq, params)
    loss_sup = 0.5 * jnp.sum((pred_y - y) ** 2) / batch_size

    # Equilibrium constraint: Delta_z(z*, x) = 0
    delta = nca_cond_delta(z_eq, cond, params)
    shifted = delta + dual_eq / rho
    loss_eq = 0.5 * rho * jnp.sum(shifted * shifted) / batch_size

    # Edge-preserving spatial Total Variation penalty on hidden states
    diff_x = z_eq[:, :, :, 1:] - z_eq[:, :, :, :-1]
    diff_y = z_eq[:, :, 1:, :] - z_eq[:, :, :-1, :]
    loss_tv = tv_weight * (jnp.sum(jnp.sqrt(diff_x ** 2 + 1e-6)) + jnp.sum(jnp.sqrt(diff_y ** 2 + 1e-6))) / batch_size

    return loss_sup + loss_eq + loss_tv


def settle_deq_pcalm(
    params: dict,
    z_init: jax.Array,
    cond: jax.Array,
    y: jax.Array,
    *,
    steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    tv_weight: float = 0.05,
) -> tuple[jax.Array, jax.Array]:
    z = z_init
    duals = jnp.zeros_like(z)

    def inner_step(z_curr, dual_curr):
        def energy(zc):
            return deq_energy(params, zc, dual_curr, cond, y, rho=rho, tv_weight=tv_weight)

        grad_fn = jax.grad(energy)

        def step(zc, _):
            return zc - state_lr * grad_fn(zc), None

        z_next, _ = jax.lax.scan(step, z_curr, xs=None, length=inner_steps)
        return z_next

    def outer(carry, _):
        zc, dualc = carry
        zc = inner_step(zc, dualc)
        r = nca_cond_delta(zc, cond, params)
        dual_next = dualc + alpha * r
        return (zc, dual_next), None

    if steps > 1:
        (z, duals), _ = jax.lax.scan(outer, (z, duals), xs=None, length=steps - 1)

    z = inner_step(z, duals)
    return z, duals


def compute_deq_grads(
    params: dict,
    z_init: jax.Array,
    cond: jax.Array,
    y: jax.Array,
    *,
    steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    rho: float = 1.0,
    alpha: float = 0.1,
    tv_weight: float = 0.05,
) -> tuple[dict, jax.Array, jax.Array]:
    batch_size = y.shape[0]
    z_eq, dual_eq = settle_deq_pcalm(
        params, z_init, cond, y,
        steps=steps, inner_steps=inner_steps, state_lr=state_lr, rho=rho, alpha=alpha, tv_weight=tv_weight,
    )

    pred_y = readout(z_eq, params)
    loss = 0.5 * jnp.sum((pred_y - y) ** 2) / batch_size

    z_eq_stop = jax.lax.stop_gradient(z_eq)
    dual_eq_stop = jax.lax.stop_gradient(dual_eq)

    def p_loss(p):
        return deq_energy(p, z_eq_stop, dual_eq_stop, cond, y, rho=rho, tv_weight=tv_weight)

    grads = jax.grad(p_loss)(params)
    return grads, loss, z_eq


def load_target_image(image_name: str = "coins", size: int = 48) -> tuple[np.ndarray, int]:
    """Loads a target image (normalized to [0, 1]) and returns (img_np with shape (C, H, W), out_channels)."""
    from PIL import Image

    if image_name == "synthetic":
        from .dip_experiment import create_synthetic_target
        img = create_synthetic_target(size)
        return img[None, ...], 1

    if image_name in {"camera", "coins", "astronaut", "coffee", "chelsea"}:
        import skimage.data as skdata
        raw = getattr(skdata, image_name)()
    else:
        raw = np.array(Image.open(image_name))

    pil_img = Image.fromarray(raw)
    pil_resized = pil_img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(pil_resized, dtype=np.float32) / 255.0

    if arr.ndim == 2:
        return arr[None, ...], 1
    else:
        return np.transpose(arr, (2, 0, 1)), arr.shape[2]


def run_deq_experiment(
    image_name: str = "coins",
    steps: int = 150,
    lr: float = 3e-3,
    channels: int = 16,
    hidden_dim: int = 64,
    size: int = 48,
    deq_steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    alpha: float = 0.1,
    rho: float = 1.0,
    tv_weight: float = 0.05,
    seed: int = 42,
    save_dir=None,
) -> dict:
    import time
    from PIL import Image
    from .dip_experiment import adam_apply, adam_init, psnr
    from .nca_experiment import extract_segmentation_pca

    key = jax.random.PRNGKey(seed)
    k_net, k_init = jax.random.split(key)

    clean_np, out_channels = load_target_image(image_name, size=size)
    y_target = jnp.asarray(clean_np[None, ...], dtype=jnp.float32)

    # Condition: normalized 2D coordinate grid (x, y)
    yy, xx = np.mgrid[:size, :size].astype(np.float32) / float(size)
    cond_np = np.stack([xx, yy], axis=0)[None, ...]  # (1, 2, H, W)
    cond = jnp.asarray(cond_np)

    params = init_conditioned_deq(k_net, channels=channels, hidden_dim=hidden_dim, in_cond_dim=2, out_channels=out_channels)
    opt_state = adam_init(params)
    z_curr = jax.random.normal(k_init, (1, channels, size, size)) * 0.05

    @jax.jit
    def train_step(p, opt_s, zc):
        grads, loss, z_next = compute_deq_grads(
            p, zc, cond, y_target,
            steps=deq_steps, inner_steps=inner_steps, state_lr=state_lr, rho=rho, alpha=alpha, tv_weight=tv_weight,
        )
        p, opt_s = adam_apply(p, grads, opt_s, lr=lr)
        return p, opt_s, loss, z_next

    history = {"step": [], "loss": [], "psnr": []}
    print(f"=== Starting Conditioned DEQ PC-ALM on '{image_name}' (out_c={out_channels}, channels={channels}, steps={steps}, size={size}x{size}) ===")
    t0 = time.time()

    for s in range(1, steps + 1):
        params, opt_state, loss, z_curr = train_step(params, opt_state, z_curr)

        if s % 10 == 0 or s == 1 or s == steps:
            pred = readout(z_curr, params)
            pred_np = np.asarray(pred[0])
            p_val = psnr(pred_np, clean_np)
            history["step"].append(s)
            history["loss"].append(float(loss))
            history["psnr"].append(p_val)
            print(f"[DEQ-PCALM] Step {s:3d}/{steps} | Loss: {float(loss):.5f} | PSNR: {p_val:.2f} dB")

    elapsed = time.time() - t0
    print(f"[DEQ-PCALM] Done in {elapsed:.2f}s.")

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        pred = readout(z_curr, params)
        pred_np = np.asarray(pred[0])

        if out_channels == 1:
            Image.fromarray((clean_np[0] * 255.0).astype(np.uint8)).save(save_dir / "target.png")
            recon_img = np.clip(pred_np[0] * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(recon_img).save(save_dir / "deq_recon.png")
        else:
            target_rgb = np.clip(np.transpose(clean_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
            recon_rgb = np.clip(np.transpose(pred_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(target_rgb).save(save_dir / "target.png")
            Image.fromarray(recon_rgb).save(save_dir / "deq_recon.png")

        hidden_np = np.asarray(z_curr[0])
        seg_rgb = extract_segmentation_pca(hidden_np)
        Image.fromarray(seg_rgb).save(save_dir / "deq_segmentation_pca.png")
        print(f"Saved DEQ reconstruction and segmentation maps to {save_dir}")

    return history
