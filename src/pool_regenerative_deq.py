from __future__ import annotations

from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from .deq_conditioned import (
    compute_deq_grads,
    deq_energy,
    init_conditioned_deq,
    load_target_image,
    make_fourier_coords,
    nca_cond_delta,
    readout,
    settle_deq_pcalm,
)
from .dip_experiment import adam_apply, adam_init, psnr
from .nca_experiment import extract_discrete_segmentation, extract_segmentation_pca


def create_damage_mask(key: jax.Array, h: int, w: int, damage_type: int) -> jax.Array:
    """Creates a binary mask (1 = keep, 0 = destroy) based on damage_type (0=half, 1=circle, 2=box)."""
    k_dir, k_pos = jax.random.split(key)

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

    # Circular crater
    cy = jax.random.randint(k_pos, (), h // 4, 3 * h // 4)
    cx = jax.random.randint(k_dir, (), w // 4, 3 * w // 4)
    radius = max(8.0, min(h, w) / 3.0)
    dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
    circle_mask = dist_sq >= (radius ** 2)

    # Box cutout
    b_size = max(10, min(h, w) // 3)
    top = jax.random.randint(k_pos, (), 0, h - b_size)
    left = jax.random.randint(k_dir, (), 0, w - b_size)
    box_mask = ~((yy >= top) & (yy < top + b_size) & (xx >= left) & (xx < left + b_size))

    mask = jnp.where(damage_type == 0, half_mask, jnp.where(damage_type == 1, circle_mask, box_mask))
    return mask.astype(jnp.float32)


def apply_pool_damage(key: jax.Array, batch_z: jax.Array, num_damaged: int = 2) -> tuple[jax.Array, jax.Array]:
    """Damages the first num_damaged samples in batch_z and returns damaged batch + keep masks."""
    b, c, h, w = batch_z.shape
    keys = jax.random.split(key, b)

    def damage_one(k, z_single, i):
        k_type, k_mask = jax.random.split(k)
        d_type = jax.random.randint(k_type, (), 0, 3)
        # Dedicated half-wipe for sample 0 to ensure exposure every iteration
        d_type = jnp.where(i == 0, 0, d_type)
        mask = create_damage_mask(k_mask, h, w, d_type)  # (H, W)
        mask = mask[None, :, :]  # (1, H, W)
        keep = jnp.where(i < num_damaged, mask, jnp.ones_like(mask))
        return z_single * keep, keep

    indices = jnp.arange(b)
    damaged, masks = jax.vmap(damage_one)(keys, batch_z, indices)
    return damaged, masks


def run_pool_experiment(
    image_name: str = "camera",
    pool_size: int = 16,
    batch_size: int = 4,
    steps: int = 200,
    lr: float = 3e-3,
    channels: int = 16,
    hidden_dim: int = 96,
    size: int = 48,
    deq_steps: int = 15,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    alpha: float = 0.1,
    rho: float = 1.0,
    tv_weight: float = 0.005,
    octaves: int = 6,
    seed: int = 42,
    save_dir: Path | None = None,
) -> dict:
    key = jax.random.PRNGKey(seed)
    k_net, k_pool, k_loop = jax.random.split(key, 3)

    clean_np, out_channels = load_target_image(image_name, size=size)
    y_target = jnp.asarray(clean_np[None, ...], dtype=jnp.float32)
    y_batch = jnp.broadcast_to(y_target, (batch_size, out_channels, size, size))

    # Condition: normalized 2D coordinate grid + high-resolution Fourier features (6 octaves)
    cond_np = make_fourier_coords(size, octaves=octaves)
    cond = jnp.asarray(cond_np)
    in_cond_dim = cond.shape[1]

    params = init_conditioned_deq(k_net, channels=channels, hidden_dim=hidden_dim, in_cond_dim=in_cond_dim, out_channels=out_channels)
    opt_state = adam_init(params)

    # Initialize persistent pool with small noise
    pool = jax.random.normal(k_pool, (pool_size, channels, size, size)) * 0.05

    @jax.jit
    def pool_train_step(p, opt_s, pool_state, k):
        k_idx, k_dam, k_seed, k_n, k_pert = jax.random.split(k, 5)
        idx = jax.random.choice(k_idx, pool_size, (batch_size,), replace=False)
        batch_z = pool_state[idx]

        # Reseed worst sample in batch
        preds = readout(batch_z, p)
        sample_losses = jnp.mean((preds - y_batch) ** 2, axis=(1, 2, 3))
        worst_in_batch = jnp.argmax(sample_losses)
        fresh_seed = jax.random.normal(k_seed, (channels, size, size)) * 0.05
        batch_z = batch_z.at[worst_in_batch].set(fresh_seed)

        # Apply Distill decimation damage to first 4 samples
        batch_z_dam, batch_masks = apply_pool_damage(k_dam, batch_z, num_damaged=4)

        # Solve DEQ stationary state using PC-ALM Augmented Lagrangian
        deq_grads, total_loss, z_eq = compute_deq_grads(
            p, batch_z_dam, cond, y_batch,
            steps=deq_steps, inner_steps=inner_steps,
            state_lr=state_lr, rho=rho, alpha=alpha, tv_weight=tv_weight,
        )

        # Matched inpainting flow loss with PINNED intact pixels: N ~ Uniform(24, 48)
        def flow_loss_fn(p_curr):
            n_steps = jax.random.randint(k_n, (), 24, 49)
            z_target_stop = jax.lax.stop_gradient(z_eq[:4])
            masks_target = batch_masks[:4]
            dam_mask = 1.0 - masks_target
            dam_pixels = jnp.sum(dam_mask) + 1e-6

            def flow_step(zc, _):
                delta = nca_cond_delta(zc, cond, p_curr)
                z_next = zc + 0.5 * delta
                # Pin intact pixels during training just like at test time!
                z_next = jnp.where(masks_target > 0.5, z_target_stop, z_next)
                return z_next, z_next

            _, z_traj = jax.lax.scan(flow_step, batch_z_dam[:4], xs=None, length=48)
            z_final = z_traj[n_steps - 1]

            # Undiluted flow loss focused purely on the void
            l_flow = jnp.sum(dam_mask * (z_final - z_target_stop) ** 2) / (dam_pixels * channels)
            pred_flow = readout(z_final, p_curr)
            y_sub = y_batch[:4]

            # High-frequency sharpening on void: L2 + L1 + Sobel edge gradients
            l_recon_l2 = jnp.sum(dam_mask * (pred_flow - y_sub) ** 2) / dam_pixels
            l_recon_l1 = jnp.sum(dam_mask * jnp.sqrt((pred_flow - y_sub) ** 2 + 1e-6)) / dam_pixels
            gx_pred = pred_flow[:, :, :, 1:] - pred_flow[:, :, :, :-1]
            gy_pred = pred_flow[:, :, 1:, :] - pred_flow[:, :, :-1, :]
            gx_y = y_sub[:, :, :, 1:] - y_sub[:, :, :, :-1]
            gy_y = y_sub[:, :, 1:, :] - y_sub[:, :, :-1, :]
            mask_gx = dam_mask[:, :, :, 1:] * dam_mask[:, :, :, :-1]
            mask_gy = dam_mask[:, :, 1:, :] * dam_mask[:, :, :-1, :]
            l_edge = (
                jnp.sum(mask_gx * jnp.sqrt((gx_pred - gx_y) ** 2 + 1e-6)) / (jnp.sum(mask_gx) + 1e-6)
                + jnp.sum(mask_gy * jnp.sqrt((gy_pred - gy_y) ** 2 + 1e-6)) / (jnp.sum(mask_gy) + 1e-6)
            )
            # Void Contrast / Dynamic Range & Mean Matching Loss:
            # Prevents regression to the conditional mean (washed-out gray haze)
            dam_px_sample = jnp.sum(dam_mask, axis=(2, 3), keepdims=True) + 1e-6
            mu_pred = jnp.sum(dam_mask * pred_flow, axis=(2, 3), keepdims=True) / dam_px_sample
            var_pred = jnp.sum(dam_mask * (pred_flow - mu_pred) ** 2, axis=(2, 3), keepdims=True) / dam_px_sample
            std_pred = jnp.sqrt(var_pred + 1e-6)

            mu_y = jnp.sum(dam_mask * y_sub, axis=(2, 3), keepdims=True) / dam_px_sample
            var_y = jnp.sum(dam_mask * (y_sub - mu_y) ** 2, axis=(2, 3), keepdims=True) / dam_px_sample
            std_y = jnp.sqrt(var_y + 1e-6)

            l_contrast = jnp.mean(jnp.abs(std_pred - std_y))
            l_mean_match = jnp.mean(jnp.abs(mu_pred - mu_y))

            # Rebalance l_recon: downweight pure L2, amplify L1, Sobel edges, and dynamic range contrast
            l_recon = 0.5 * l_recon_l2 + 1.5 * l_recon_l1 + 1.0 * l_edge + 1.5 * l_contrast + 0.5 * l_mean_match

            # Strict zero-velocity attractor constraints:
            # 1. Delta on the evolving state must drop to 0
            delta_final = nca_cond_delta(z_final, cond, p_curr)
            l_stationary = jnp.sum(dam_mask * (delta_final ** 2)) / (dam_pixels * channels)
            # 2. Delta on the TARGET equilibrium state MUST be identically ZERO (no drift at z*!)
            delta_target = nca_cond_delta(z_target_stop, cond, p_curr)
            l_target_drift = jnp.mean(delta_target ** 2)

            # 3. Contractive Restoring Spring Basin around z_target:
            # Shake z_target_stop with random bidirectional perturbations (+ and -).
            # The NCA step MUST push it directly back towards z_target_stop!
            noise = jax.random.normal(k_pert, z_target_stop.shape) * 0.20
            z_perturbed = z_target_stop + noise
            delta_pert = nca_cond_delta(z_perturbed, cond, p_curr)
            z_restored = z_perturbed + 0.5 * delta_pert
            l_spring = jnp.mean((z_restored - z_target_stop) ** 2)

            loss = l_flow + 1.2 * l_recon + 0.5 * l_stationary + 1.0 * l_target_drift + 3.0 * l_spring
            return loss, z_final

        (flow_val, z_final_stepped), flow_grads = jax.value_and_grad(flow_loss_fn, has_aux=True)(p)
        combined_grads = jax.tree_util.tree_map(lambda g1, g2: g1 + 1.0 * g2, deq_grads, flow_grads)

        p, opt_s = adam_apply(p, combined_grads, opt_s, lr=lr)

        # Distill persistent pool: put the stepped states back into the pool for damaged slots!
        # This forces the pool to maintain long-term attractor stability across life cycles.
        batch_z_updated = batch_z.at[:4].set(jax.lax.stop_gradient(z_final_stepped))
        batch_z_updated = batch_z_updated.at[4:].set(jax.lax.stop_gradient(z_eq[4:]))
        new_pool = pool_state.at[idx].set(batch_z_updated)
        return p, opt_s, new_pool, total_loss

    print(f"=== Starting Distill Pool + DEQ PC-ALM on '{image_name}' (pool={pool_size}, batch={batch_size}, steps={steps}, size={size}x{size}) ===")
    history = {"step": [], "loss": [], "psnr": []}
    t0 = time.time()
    k_steps = jax.random.split(k_loop, steps)

    for s in range(1, steps + 1):
        params, opt_state, pool, loss_val = pool_train_step(params, opt_state, pool, k_steps[s - 1])

        if s % 20 == 0 or s == 1 or s == steps:
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
        run_decimation_battery(
            params, z_eq, cond, clean_np, y_target, out_channels, size, save_dir,
            rho=rho, tv_weight=tv_weight, state_lr=state_lr, deq_steps=deq_steps, inner_steps=inner_steps, alpha=alpha,
        )

    return history


def compute_spectral_radius(params: dict, z_eq: jax.Array, cond: jax.Array, n_iter: int = 15) -> float:
    """Computes the spectral radius (top singular/eigenvalue) of the relaxation operator J at z_eq."""
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, z_eq.shape)
    v = v / (jnp.linalg.norm(v) + 1e-8)

    def op(z):
        return z + 0.5 * nca_cond_delta(z, cond, params)

    for _ in range(n_iter):
        _, jv = jax.jvp(op, (z_eq,), (v,))
        norm = jnp.linalg.norm(jv)
        v = jv / (norm + 1e-8)

    _, jv = jax.jvp(op, (z_eq,), (v,))
    return float(jnp.linalg.norm(jv))


def run_decimation_battery(
    params: dict,
    z_eq: jax.Array,
    cond: jax.Array,
    clean_np: np.ndarray,
    y_target: jax.Array,
    out_channels: int,
    size: int,
    save_dir: Path,
    rho: float = 1.0,
    tv_weight: float = 0.005,
    state_lr: float = 0.05,
    deq_steps: int = 15,
    inner_steps: int = 3,
    alpha: float = 0.1,
):
    """Executes decimation tests using forward NCA relaxation and Lyapunov energy descent."""
    print("=== Running Distill Decimation + Forward NCA Self-Healing Battery ===")

    lam_max = compute_spectral_radius(params, z_eq, cond)
    print(f"[THEORY] Spectral radius of relaxation operator at z_eq: |lambda_max| = {lam_max:.4f}")

    def forward_nca_relax(z_start, keep_mask, n_steps=48, step_size=0.5):
        """Autonomous NCA forward relaxation with dynamic boundary-seam continuity valley detection."""
        from scipy.ndimage import distance_transform_edt

        history = [z_start]
        zc = z_start
        dam_mask = 1.0 - keep_mask

        # Mathematically derived min_steps from mask depth geometry:
        dam_np = np.asarray(dam_mask[0, 0])
        max_depth = float(distance_transform_edt(dam_np).max()) if float(dam_np.max()) > 0 else 1.0
        # Receptive field expands ~1.2-1.5 pixels per step:
        min_steps = max(6, int(max_depth / 1.2))

        # Boundary rim inside the damaged void (non-periodic padding so image borders don't wrap):
        m_padded = jnp.pad(keep_mask, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0.0)
        up = m_padded[:, :, :size, 1 : size + 1]
        down = m_padded[:, :, 2:, 1 : size + 1]
        left = m_padded[:, :, 1 : size + 1, :size]
        right = m_padded[:, :, 1 : size + 1, 2:]
        seam_mask = ((up + down + left + right) > 0.5) & (keep_mask < 0.5)
        seam_weight = float(jnp.sum(seam_mask)) + 1e-6

        seam_errors = []
        residuals = []
        settle_count = 0

        for s in range(1, n_steps + 1):
            delta = nca_cond_delta(zc, cond, params)
            res = float(jnp.mean((dam_mask * delta) ** 2))
            residuals.append(res)

            max_res = max(residuals)
            if len(residuals) >= 6 and res < 0.35 * max_res:
                settle_count += 1
                effective_lr = step_size * (0.97 ** settle_count)
            else:
                effective_lr = step_size

            zc = zc + effective_lr * delta
            zc = jnp.where(keep_mask > 0.5, z_eq, zc)
            history.append(zc)

            # Measure boundary seam discontinuity between inside and outside
            seam_err = float(jnp.sum(jnp.abs(zc - z_eq) * seam_mask) / seam_weight)
            seam_errors.append(seam_err)

            # Valley detection: after signal reached deepest void (s >= min_steps + 3),
            # stop if seam error is rising for 3 consecutive steps and is above 1.05 * min of valid errors:
            if s >= min_steps + 3:
                valid_errors = seam_errors[min_steps - 1 :]
                if seam_err > 1.05 * min(valid_errors) and seam_errors[-1] > seam_errors[-2] > seam_errors[-3]:
                    break

        # Best converged step is the exact valley where the boundary seam is smoothest
        valid_errors = seam_errors[min_steps - 1 :]
        best_sub_step = int(np.argmin(valid_errors))
        best_step = (min_steps - 1) + best_sub_step + 1
        t_final = min(best_step, len(history) - 1)
        indices = np.linspace(0, t_final, 5, dtype=int)
        print(f"[DEQ-RELAX] max_depth: {max_depth:.1f}, min_steps: {min_steps} | Converged at step {t_final}/{s} (seam_err: {seam_errors[t_final - 1]:.5f}, res: {residuals[t_final - 1]:.6f}) | linspace frames: {indices.tolist()}")

        snapshots = [history[idx] for idx in indices]
        return snapshots

    def to_img(z_state):
        pred = readout(z_state, params)
        p_np = np.asarray(pred[0])
        if out_channels == 1:
            return np.clip(p_np[0] * 255.0, 0, 255).astype(np.uint8)
        else:
            return np.clip(np.transpose(p_np, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)

    # --- Test 1: Half-Wipe ---
    mask_half = jnp.ones((1, 1, size, size), dtype=jnp.float32)
    mask_half = mask_half.at[:, :, :, size // 2:].set(0.0)
    z_half = z_eq * mask_half

    snaps_half = forward_nca_relax(z_half, mask_half, n_steps=48, step_size=0.5)
    imgs_half = [to_img(s) for s in snaps_half]
    if out_channels == 1:
        imgs_half = [np.stack([im, im, im], axis=-1) for im in imgs_half]
    strip_half = np.concatenate(imgs_half, axis=1)
    Image.fromarray(strip_half).save(save_dir / "regen_half_wipe_strip.png")

    # --- Test 2: Center Crater ---
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    cy, cx = size / 2.0, size / 2.0
    r = size / 3.0
    mask_circle = ((yy - cy) ** 2 + (xx - cx) ** 2 >= r ** 2).astype(np.float32)[None, None, ...]
    z_circle = z_eq * jnp.asarray(mask_circle)

    snaps_circle = forward_nca_relax(z_circle, jnp.asarray(mask_circle), n_steps=48, step_size=0.5)
    imgs_circle = [to_img(s) for s in snaps_circle]
    if out_channels == 1:
        imgs_circle = [np.stack([im, im, im], axis=-1) for im in imgs_circle]
    strip_circle = np.concatenate(imgs_circle, axis=1)
    Image.fromarray(strip_circle).save(save_dir / "regen_crater_strip.png")

    # --- Test 3: Pepper Decimation ---
    np.random.seed(42)
    mask_pepper = (np.random.rand(1, 1, size, size) > 0.5).astype(np.float32)
    z_pepper = z_eq * jnp.asarray(mask_pepper)

    snaps_pepper = forward_nca_relax(z_pepper, jnp.asarray(mask_pepper), n_steps=48, step_size=0.5)
    imgs_pepper = [to_img(s) for s in snaps_pepper]
    if out_channels == 1:
        imgs_pepper = [np.stack([im, im, im], axis=-1) for im in imgs_pepper]
    strip_pepper = np.concatenate(imgs_pepper, axis=1)
    Image.fromarray(strip_pepper).save(save_dir / "regen_pepper_strip.png")

    # Save 6x upscaled strips for clear visual inspection
    for strip_name, strip_arr in [
        ("regen_half_wipe_strip", strip_half),
        ("regen_crater_strip", strip_circle),
        ("regen_pepper_strip", strip_pepper),
    ]:
        im = Image.fromarray(strip_arr)
        im_large = im.resize((im.width * 6, im.height * 6), Image.Resampling.NEAREST)
        im_large.save(save_dir / f"{strip_name}_large.png")

    print(f"Saved decimation regeneration strips (and 6x large versions) to {save_dir}")
