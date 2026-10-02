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


def perceive_dilated(z: jax.Array, dilations: tuple[int, ...] = (1, 2, 4)) -> jax.Array:
    """Multiscale perception using Sobel filters at multiple dilation rates."""
    feats = [z]
    for d in dilations:
        tl = jnp.roll(jnp.roll(z, d, axis=2), d, axis=3)
        tc = jnp.roll(z, d, axis=2)
        tr = jnp.roll(jnp.roll(z, d, axis=2), -d, axis=3)
        ml = jnp.roll(z, d, axis=3)
        mr = jnp.roll(z, -d, axis=3)
        bl = jnp.roll(jnp.roll(z, -d, axis=2), d, axis=3)
        bc = jnp.roll(z, -d, axis=2)
        br = jnp.roll(jnp.roll(z, -d, axis=2), -d, axis=3)
        dx = (-tl + tr - 2.0 * ml + 2.0 * mr - bl + br) / 8.0
        dy = (-tl - 2.0 * tc - tr + bl + 2.0 * bc + br) / 8.0
        feats.extend([dx, dy])
    return jnp.concatenate(feats, axis=1)


def init_conditioned_deq(
    key: jax.Array,
    channels: int = 16,
    hidden_dim: int = 96,
    in_cond_dim: int = 2,  # e.g. (x, y) coordinates
    out_channels: int = 1,
    dilations: tuple[int, ...] = (1, 2, 4),
):
    k1, k2 = jax.random.split(key)
    # Perception has channels * (1 + 2 * len(dilations)) + in_cond_dim
    num_perceive = channels * (1 + 2 * len(dilations))
    total_in = num_perceive + in_cond_dim
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


def nca_cond_delta(z: jax.Array, cond: jax.Array, params: dict, dilations: tuple[int, ...] = (1, 2, 4)) -> jax.Array:
    """NCA step conditioned on input x with multiscale dilated perception."""
    p = perceive_dilated(z, dilations=dilations)
    if cond.shape[0] != p.shape[0]:
        cond_b = jnp.broadcast_to(cond, (p.shape[0], *cond.shape[1:]))
    else:
        cond_b = cond
    p_full = jnp.concatenate([p, cond_b], axis=1)  # (B, 3*C + cond_dim, H, W)

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
    loss_sup_l2 = 0.5 * jnp.sum((pred_y - y) ** 2) / batch_size
    loss_sup_l1 = 0.5 * jnp.sum(jnp.sqrt((pred_y - y) ** 2 + 1e-6)) / batch_size
    loss_sup = loss_sup_l2 + loss_sup_l1

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
    loss = (0.5 * jnp.sum((pred_y - y) ** 2) + 0.5 * jnp.sum(jnp.sqrt((pred_y - y) ** 2 + 1e-6))) / batch_size

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


def make_fourier_coords(size: int, octaves: int = 4) -> np.ndarray:
    """Generates normalized coordinate grid with gentle multiscale Fourier features."""
    yy, xx = np.mgrid[:size, :size].astype(np.float32) / float(size)
    feats = [xx, yy]
    for k in range(octaves):
        freq = float(2 ** k) * np.pi
        feats.extend([np.sin(freq * xx), np.cos(freq * xx), np.sin(freq * yy), np.cos(freq * yy)])
    return np.stack(feats, axis=0)[None, ...].astype(np.float32)


def run_deq_experiment(
    image_name: str = "coins",
    steps: int = 150,
    lr: float = 3e-3,
    channels: int = 16,
    hidden_dim: int = 96,
    size: int = 48,
    deq_steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    alpha: float = 0.1,
    rho: float = 1.0,
    tv_weight: float = 0.05,
    n_clusters: int = 4,
    damage_prob: float = 0.5,
    damage_box: int = 14,
    damage_warmup: int = 25,
    seed: int = 42,
    save_dir=None,
) -> dict:
    import time
    from PIL import Image
    from .dip_experiment import adam_apply, adam_init, psnr
    from .nca_experiment import extract_discrete_segmentation, extract_segmentation_pca

    key = jax.random.PRNGKey(seed)
    k_net, k_init, k_train = jax.random.split(key, 3)

    clean_np, out_channels = load_target_image(image_name, size=size)
    y_target = jnp.asarray(clean_np[None, ...], dtype=jnp.float32)

    # Condition: normalized 2D coordinate grid + gentle Fourier features (4 octaves)
    cond_np = make_fourier_coords(size, octaves=4)
    cond = jnp.asarray(cond_np)
    in_cond_dim = cond.shape[1]

    params = init_conditioned_deq(k_net, channels=channels, hidden_dim=hidden_dim, in_cond_dim=in_cond_dim, out_channels=out_channels)
    opt_state = adam_init(params)
    z_curr = jax.random.normal(k_init, (1, channels, size, size)) * 0.05

    def apply_damage(k, z):
        k_choice, k_pos = jax.random.split(k)
        should_damage = jax.random.uniform(k_choice) < damage_prob
        top = jax.random.randint(k_pos, (), 0, size - damage_box)
        left = jax.random.randint(k_pos, (), 0, size - damage_box)
        grid_y, grid_x = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
        mask = ~((grid_y >= top) & (grid_y < top + damage_box) & (grid_x >= left) & (grid_x < left + damage_box))
        mask = mask[None, None, :, :].astype(jnp.float32)
        return jnp.where(should_damage, z * mask, z)

    @jax.jit
    def train_step_clean(p, opt_s, zc):
        grads, loss, z_next = compute_deq_grads(
            p, zc, cond, y_target,
            steps=deq_steps, inner_steps=inner_steps, state_lr=state_lr, rho=rho, alpha=alpha, tv_weight=tv_weight,
        )
        p, opt_s = adam_apply(p, grads, opt_s, lr=lr)
        return p, opt_s, loss, z_next

    @jax.jit
    def train_step_damage(p, opt_s, zc, k):
        zc_dam = apply_damage(k, zc)
        grads, loss, z_next = compute_deq_grads(
            p, zc_dam, cond, y_target,
            steps=deq_steps, inner_steps=inner_steps, state_lr=state_lr, rho=rho, alpha=alpha, tv_weight=tv_weight,
        )
        p, opt_s = adam_apply(p, grads, opt_s, lr=lr)
        return p, opt_s, loss, z_next

    history = {"step": [], "loss": [], "psnr": []}
    print(f"=== Starting Conditioned DEQ PC-ALM on '{image_name}' (damage_prob={damage_prob}, out_c={out_channels}, channels={channels}, steps={steps}, size={size}x{size}) ===")
    t0 = time.time()
    k_steps = jax.random.split(k_train, steps)

    for s in range(1, steps + 1):
        if s <= damage_warmup or damage_prob <= 0.0:
            params, opt_state, loss, z_curr = train_step_clean(params, opt_state, z_curr)
        else:
            params, opt_state, loss, z_curr = train_step_damage(params, opt_state, z_curr, k_steps[s - 1])

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

        # Discrete clustering
        labels_2d, discrete_rgb, masks_mosaic = extract_discrete_segmentation(
            hidden_np, n_clusters=n_clusters, seed=seed
        )
        Image.fromarray(discrete_rgb).save(save_dir / "deq_discrete_seg.png")
        Image.fromarray(masks_mosaic).save(save_dir / "deq_cluster_masks.png")
        np.save(save_dir / "z_equilibrium.npy", hidden_np)
        np.save(save_dir / "cluster_labels.npy", labels_2d)

        print(f"Saved DEQ reconstruction, PCA maps, and {n_clusters}-cluster discrete masks to {save_dir}")

        # Test self-healing / regeneration
        box_size = max(8, size // 3)
        r0, r1 = size // 2 - box_size // 2, size // 2 + box_size // 2
        c0, c1 = size // 2 - box_size // 2, size // 2 + box_size // 2
        z_damaged = z_curr.at[:, :, r0:r1, c0:c1].set(0.0)

        # Autonomous equilibrium relaxation on ||Delta z||^2 + TV without y supervision
        def heal_energy(zc):
            delta = nca_cond_delta(zc, cond, params)
            loss_eq = 0.5 * rho * jnp.sum(delta ** 2) / zc.shape[0]
            diff_x = zc[:, :, :, 1:] - zc[:, :, :, :-1]
            diff_y = zc[:, :, 1:, :] - zc[:, :, :-1, :]
            loss_tv = tv_weight * (jnp.sum(jnp.sqrt(diff_x ** 2 + 1e-6)) + jnp.sum(jnp.sqrt(diff_y ** 2 + 1e-6))) / zc.shape[0]
            return loss_eq + loss_tv

        heal_grad = jax.grad(heal_energy)
        z_healed = z_damaged
        for _ in range(60):
            z_healed = z_healed - state_lr * heal_grad(z_healed)

        # Inpainting test: DEQ relaxation with supervision outside the hole
        keep_mask = jnp.ones((1, 1, size, size), dtype=jnp.float32)
        keep_mask = keep_mask.at[:, :, r0:r1, c0:c1].set(0.0)

        def inpaint_energy(zc, dualc):
            pred_y = readout(zc, params)
            loss_sup = 0.5 * jnp.sum((keep_mask * (pred_y - y_target)) ** 2) / zc.shape[0]
            delta = nca_cond_delta(zc, cond, params)
            shifted = delta + dualc / rho
            loss_eq = 0.5 * rho * jnp.sum(shifted * shifted) / zc.shape[0]
            diff_x = zc[:, :, :, 1:] - zc[:, :, :, :-1]
            diff_y = zc[:, :, 1:, :] - zc[:, :, :-1, :]
            loss_tv = tv_weight * (jnp.sum(jnp.sqrt(diff_x ** 2 + 1e-6)) + jnp.sum(jnp.sqrt(diff_y ** 2 + 1e-6))) / zc.shape[0]
            return loss_sup + loss_eq + loss_tv

        inp_grad = jax.grad(inpaint_energy)
        z_inp = z_damaged
        dual_inp = jnp.zeros_like(z_inp)
        for _ in range(deq_steps + 10):
            for _ in range(inner_steps):
                z_inp = z_inp - state_lr * inp_grad(z_inp, dual_inp)
            r = nca_cond_delta(z_inp, cond, params)
            dual_inp = dual_inp + alpha * r

        # Save damaged, healed, and inpainted images
        dam_pred = readout(z_damaged, params)
        heal_pred = readout(z_healed, params)
        inp_pred = readout(z_inp, params)
        dam_np = np.asarray(dam_pred[0])
        heal_np = np.asarray(heal_pred[0])
        inp_np = np.asarray(inp_pred[0])

        if out_channels == 1:
            Image.fromarray(np.clip(dam_np[0] * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_damage_recon.png")
            Image.fromarray(np.clip(heal_np[0] * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_healed_recon.png")
            Image.fromarray(np.clip(inp_np[0] * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_inpaint_recon.png")
        else:
            Image.fromarray(np.clip(np.transpose(dam_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_damage_recon.png")
            Image.fromarray(np.clip(np.transpose(heal_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_healed_recon.png")
            Image.fromarray(np.clip(np.transpose(inp_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)).save(save_dir / "deq_inpaint_recon.png")

        healed_pca = extract_segmentation_pca(np.asarray(z_healed[0]))
        Image.fromarray(healed_pca).save(save_dir / "deq_healed_pca.png")
        print(f"Saved self-healing and inpainting tests to {save_dir}")

    return history
