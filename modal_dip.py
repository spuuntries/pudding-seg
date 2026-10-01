"""
modal_dip.py — Run DIP with PC-ALM on Modal GPU.

Usage:
    modal run modal_dip.py
    modal run modal_dip.py --method pcalm --steps 500 --size 64
"""

from pathlib import Path
import modal

app = modal.App("pudding-dip-pcalm")

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
def run_modal_dip(
    method: str = "pcalm",
    steps: int = 300,
    lr: float = 3e-3,
    depth: int = 8,
    channels: int = 32,
    size: int = 64,
    budget: int = 6,
    inner_steps: int = 2,
    state_lr: float = 0.05,
    alpha: float = 0.1,
):
    import sys
    sys.path.insert(0, "/root")
    from pathlib import Path
    from src.dip_experiment import run_dip

    out_dir = Path("/root/results") / f"dip_{method}_{size}x{size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Running Modal GPU DIP with {method.upper()} on A10G ===")
    hist = run_dip(
        method=method,
        steps=steps,
        lr=lr,
        depth=depth,
        channels=channels,
        size=size,
        budget=budget,
        inner_steps=inner_steps,
        state_lr=state_lr,
        alpha=alpha,
        save_dir=out_dir,
    )
    volume.commit()
    return hist


@app.local_entrypoint()
def main(
    method: str = "pcalm",
    steps: int = 300,
    size: int = 64,
    depth: int = 8,
):
    run_modal_dip.remote(
        method=method,
        steps=steps,
        size=size,
        depth=depth,
    )
