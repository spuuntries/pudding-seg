"""
modal_nca.py — Run NCA DEQ PC-ALM Neural Grouping on Modal GPU.

Usage:
    modal run modal_nca.py
    modal run modal_nca.py --steps 300 --size 64 --channels 24
"""

from pathlib import Path
import modal

app = modal.App("pudding-nca-deq")

LOCAL_DIR = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "jax[cuda12]",
        "numpy",
        "scipy",
        "pillow",
        "matplotlib",
        "scikit-image",
        "tqdm",
    ])
    .add_local_dir(LOCAL_DIR / "src", remote_path="/root/src")
)

volume = modal.Volume.from_name("pudding-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/root/results": volume},
)
def run_modal_nca(
    steps: int = 250,
    lr: float = 3e-3,
    channels: int = 16,
    hidden_dim: int = 64,
    size: int = 32,
    deq_steps: int = 8,
    inner_steps: int = 2,
    state_lr: float = 0.05,
    alpha: float = 0.1,
    rho: float = 1.0,
):
    import sys
    sys.path.insert(0, "/root")
    from pathlib import Path
    from src.nca_experiment import run_nca_trajectory

    out_dir = Path("/root/results") / f"nca_trajectory_{size}x{size}_c{channels}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Modal GPU Trajectory NCA PC-ALM on A10G (size={size}x{size}, channels={channels}) ===")
    hist = run_nca_trajectory(
        steps=steps,
        nca_steps=16,
        budget=6,
        inner_steps=2,
        lr=lr,
        channels=channels,
        hidden_dim=hidden_dim,
        size=size,
        state_lr=state_lr,
        alpha=alpha,
        rho=rho,
        save_dir=out_dir,
    )
    volume.commit()

    images = {}
    for name in ["target.png", "nca_recon.png", "nca_segmentation_pca.png"]:
        file_p = out_dir / name
        if file_p.is_file():
            images[name] = file_p.read_bytes()

    return {"history": hist, "images": images}


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/root/results": volume},
)
def run_modal_deq(
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
    clusters: int = 4,
    damage_prob: float = 0.5,
):
    import sys
    sys.path.insert(0, "/root")
    from pathlib import Path
    from src.deq_conditioned import run_deq_experiment

    out_dir = Path("/root/results") / f"deq_{image_name}_{size}x{size}_c{channels}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Modal GPU Conditioned DEQ PC-ALM on '{image_name}' on A10G (size={size}x{size}, tv_weight={tv_weight}, clusters={clusters}, damage_prob={damage_prob}) ===")
    hist = run_deq_experiment(
        image_name=image_name,
        steps=steps,
        lr=lr,
        channels=channels,
        hidden_dim=hidden_dim,
        size=size,
        deq_steps=deq_steps,
        inner_steps=inner_steps,
        state_lr=state_lr,
        alpha=alpha,
        rho=rho,
        tv_weight=tv_weight,
        n_clusters=clusters,
        damage_prob=damage_prob,
        save_dir=out_dir,
    )
    volume.commit()

    images = {}
    for name in [
        "target.png", "deq_recon.png", "deq_segmentation_pca.png",
        "deq_discrete_seg.png", "deq_cluster_masks.png",
        "deq_damage_recon.png", "deq_healed_recon.png", "deq_inpaint_recon.png", "deq_healed_pca.png"
    ]:
        file_p = out_dir / name
        if file_p.is_file():
            images[name] = file_p.read_bytes()

    return {"history": hist, "images": images}


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    volumes={"/root/results": volume},
)
def run_modal_pool(
    image_name: str = "camera",
    steps: int = 400,
    pool_size: int = 32,
    batch_size: int = 8,
    size: int = 48,
    channels: int = 16,
    lr: float = 2e-3,
    deq_steps: int = 32,
    step_size: float = 0.5,
    tv_weight: float = 0.02,
):
    import sys
    sys.path.insert(0, "/root")
    from pathlib import Path
    from src.pool_regenerative_deq import run_pool_experiment

    out_dir = Path("/root/results") / f"pool_regen_{image_name}_{size}x{size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Regenerative Sample-Pool DEQ on '{image_name}' on A10G (size={size}x{size}, steps={steps}) ===")
    hist = run_pool_experiment(
        image_name=image_name,
        pool_size=pool_size,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        channels=channels,
        size=size,
        deq_steps=deq_steps,
        step_size=step_size,
        tv_weight=tv_weight,
        save_dir=out_dir,
    )
    volume.commit()

    images = {}
    for name in [
        "target.png", "deq_recon.png", "deq_segmentation_pca.png", "deq_discrete_seg.png",
        "regen_half_wipe_strip.png", "regen_crater_strip.png", "regen_pepper_strip.png"
    ]:
        file_p = out_dir / name
        if file_p.is_file():
            images[name] = file_p.read_bytes()

    return {"history": hist, "images": images}


@app.local_entrypoint()
def main(
    image: str = "camera",
    mode: str = "deq",
    steps: int = 180,
    size: int = 48,
    channels: int = 16,
    deq_steps: int = 15,
    tv_weight: float = 0.05,
    clusters: int = 4,
    damage_prob: float = 0.5,
    pool_size: int = 32,
    batch_size: int = 8,
):
    from pathlib import Path
    if mode == "pool":
        res = run_modal_pool.remote(
            image_name=image,
            steps=steps,
            pool_size=pool_size,
            batch_size=batch_size,
            size=size,
            channels=channels,
            deq_steps=deq_steps,
            tv_weight=tv_weight,
        )
        local_out = Path(f"results/pool_{image}")
    elif mode == "deq":
        res = run_modal_deq.remote(
            image_name=image,
            steps=steps,
            size=size,
            channels=channels,
            deq_steps=deq_steps,
            tv_weight=tv_weight,
            clusters=clusters,
            damage_prob=damage_prob,
        )
        local_out = Path(f"results/deq_{image}")
    else:
        res = run_modal_nca.remote(
            steps=steps,
            size=size,
            channels=channels,
            deq_steps=deq_steps,
        )
        local_out = Path("results/nca_deq")

    local_out.mkdir(parents=True, exist_ok=True)
    import time
    for name, b in res["images"].items():
        out_f = (local_out / name).resolve()
        for _ in range(5):
            try:
                out_f.write_bytes(b)
                break
            except OSError:
                time.sleep(0.3)
    print(f"\n[Local] Downloaded all result images to {local_out.resolve()}")
