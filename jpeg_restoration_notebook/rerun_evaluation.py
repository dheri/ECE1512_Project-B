"""Reproduce the restoration results from the bundled checkpoint."""
import argparse
import json
import os
from pathlib import Path
from tempfile import mkdtemp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from evaluate import TEST_QUALITIES, evaluate_all, load_checkpoint
from visualize import show_comparison

PROJECT = Path(__file__).resolve().parents[1]


def plot_curves(results, output):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    reference = results[results.Method == "JPEG"]
    reference[["Q", "BPP", "Images"]].to_csv(output / "quality_bpp.csv", index=False)
    axes[0].plot(reference.Q, reference.BPP, "o-")
    axes[0].set(xlabel="JPEG Q", ylabel="JPEG file bpp")
    for name, group in results.groupby("Method", sort=False):
        axes[1].plot(group.BPP, group.PSNR, "o-", label=name)
        axes[2].plot(group.BPP, group.SSIM, "o-", label=name)
    axes[1].set(xlabel="JPEG input bpp", ylabel="PSNR (dB)")
    axes[2].set(xlabel="JPEG input bpp", ylabel="SSIM")
    for ax in axes:
        ax.grid(alpha=.2)
    for ax in axes[1:]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "bitrate_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT / "evidence/restoration/spatial_only/best.pt")
    parser.add_argument("--data-root", type=Path,
                        default=os.environ.get("PROJECT_B_DATA_ROOT", PROJECT / "data_1um_bicubic"))
    parser.add_argument("--output", type=Path, help="New directory for results")
    parser.add_argument("--qualities", type=int, nargs="+", default=TEST_QUALITIES)
    args = parser.parse_args()
    test_dir = args.data_root / "test"
    if not test_dir.is_dir():
        parser.error(f"Test split not found: {test_dir}. Set --data-root to the prepared dataset.")
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new directory.")

    torch.set_num_threads(2)
    device = torch.device("cpu")
    model, saved = load_checkpoint(args.checkpoint, device)
    models = {"CNN + L1": model}
    cfg = dict(saved, test_dir=str(test_dir), num_workers=0)
    output = args.output
    if output is None:
        runs = Path(__file__).resolve().parent / "runs"
        runs.mkdir(exist_ok=True)
        output = Path(mkdtemp(prefix="evaluation-", dir=runs))
    else:
        output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "test_dir": str(test_dir),
        "qualities": list(args.qualities), "patch_size": cfg["eval_patch_size"],
        "ssim": "11x11 valid Gaussian windows, sigma 1.5",
    }, indent=2) + "\n")
    print("Output:", output, flush=True)
    results = evaluate_all(models, cfg, args.qualities, device=device)
    results.to_csv(output / "metrics_long.csv", index=False)
    results.pivot(index=["Q", "BPP"], columns="Method", values=["PSNR", "SSIM"]).to_csv(
        output / "metrics_table.csv"
    )
    plot_curves(results, output)
    for quality in args.qualities:
        show_comparison(models, cfg, 0, quality, output, device=device)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
