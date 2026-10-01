from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from .dip_experiment import run_dip


def main():
    save_dir = Path("results/comparison_dip")
    save_dir.mkdir(parents=True, exist_ok=True)

    methods = ["bp", "pc", "pcalm"]
    results = {}

    common_kwargs = {
        "steps": 120,
        "lr": 4e-3,
        "depth": 5,
        "channels": 16,
        "size": 32,
        "noise_sigma": 0.12,
        "budget": 4,
        "inner_steps": 2,
        "state_lr": 0.05,
        "alpha": 0.1,
        "seed": 42,
    }

    for m in methods:
        print(f"\n================ Running {m.upper()} ================")
        hist = run_dip(method=m, save_dir=save_dir, **common_kwargs)
        results[m] = hist

    # Plot Comparison Curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    for m in methods:
        axes[0].plot(results[m]["step"], results[m]["loss"], label=m.upper())
    axes[0].set_title("Reconstruction Loss")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PSNR Clean (DIP Denoising Prior metric)
    for m in methods:
        axes[1].plot(results[m]["step"], results[m]["psnr_clean"], label=m.upper())
    axes[1].set_title("PSNR vs Clean Image (dB)")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # PSNR Noisy (Fitting target)
    for m in methods:
        axes[2].plot(results[m]["step"], results[m]["psnr_noisy"], label=m.upper())
    axes[2].set_title("PSNR vs Noisy Target (dB)")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = save_dir / "dip_methods_comparison.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nComparison chart saved to {chart_path}")


if __name__ == "__main__":
    main()
