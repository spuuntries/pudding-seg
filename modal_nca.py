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
    steps: int = 150,
    lr: float = 3e-3,
    channels: int = 16,
    hidden_dim: int = 64,
    size: int = 32,
    deq_steps: int = 10,
    inner_steps: int = 3,
    state_lr: float = 0.05,
    alpha: float = 0.1,
    rho: float = 1.0,
):
    import sys
    sys.path.insert(0, "/root")
    from pathlib import Path
    from src.deq_conditioned import run_deq_experiment

    out_dir = Path("/root/results") / f"deq_conditioned_{size}x{size}_c{channels}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Modal GPU Conditioned DEQ PC-ALM on A10G (size={size}x{size}) ===")
    hist = run_deq_experiment(
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
        save_dir=out_dir,
    )
    volume.commit()

    images = {}
    for name in ["target.png", "deq_recon.png", "deq_segmentation_pca.png"]:
        file_p = out_dir / name
        if file_p.is_file():
            images[name] = file_p.read_bytes()

    return {"history": hist, "images": images}


@app.local_entrypoint()
def main(
    mode: str = "deq",
    steps: int = 150,
    size: int = 32,
    channels: int = 16,
    deq_steps: int = 10,
):
    from pathlib import Path
    if mode == "deq":
        res = run_modal_deq.remote(
            steps=steps,
            size=size,
            channels=channels,
            deq_steps=deq_steps,
        )
        local_out = Path("results/deq_conditioned")
    else:
        res = run_modal_nca.remote(
            steps=steps,
            size=size,
            channels=channels,
            deq_steps=deq_steps,
        )
        local_out = Path("results/nca_deq")

    local_out.mkdir(parents=True, exist_ok=True)
    for name, b in res["images"].items():
        (local_out / name).write_bytes(b)
    print(f"\n[Local] Downloaded all result images to {local_out.resolve()}")
