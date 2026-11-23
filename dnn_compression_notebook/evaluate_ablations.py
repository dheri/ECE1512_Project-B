"""Evaluate a saved codec on its frozen crops, with two post-training controls."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from codec import _parts, compress, decompress, payload_size
from comparison import jpeg_roundtrip
from data import ImageDataset
from engine import load_checkpoint
from entropy_context import fit_context_checkpoint
from model import CompressionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_quality import rgb_psnr, rgb_ssim


def inference_only(*args):
    raise RuntimeError("This experiment uses hard quantization only")


@torch.no_grad()
def prune(model, calibration):
    """Keep the 128 most used codewords on training center crops; no fine-tuning."""
    counts = torch.zeros(model.config["codebook_size"], dtype=torch.long)
    for batch in calibration:
        ids = model(batch)["indices"]
        counts += torch.bincount(ids.flatten(), minlength=len(counts))
    k = len(counts) // 2
    # Deterministic ties, retaining original relative order of the selected IDs.
    keep = counts.argsort(descending=True, stable=True)[:k].sort().values
    reduced = CompressionModel(inference_only, **dict(model.config, codebook_size=k))
    state = model.state_dict()
    state["codebook"] = state["codebook"][keep]
    state["prior_logits"] = state["prior_logits"][keep]
    reduced.load_state_dict(state)
    return reduced.eval(), keep.tolist(), float(counts[keep].sum() / counts.sum())


def metrics(decoded, original):
    return float(rgb_psnr(decoded, original)[0]), float(rgb_ssim(decoded, original)[0])


def plot_curves(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.15), layout="constrained")
    labels = dict(jpeg="JPEG (standard tables)", jpeg_opt="JPEG (optimized tables)")
    markers = {"VQ 256 / zlib": "o", "VQ 128 / zlib": "s", "VQ 256 / context": "D"}
    for ax, metric, label in zip(axes, ("psnr", "ssim"), ("Mean PSNR (dB)", "Mean RGB SSIM")):
        for method, color in (("jpeg", "#516177"), ("jpeg_opt", "#99a6b1")):
            rows = sorted((r for r in summary if r["method"] == method), key=lambda r: r["bpp"])
            ax.plot([r["bpp"] for r in rows], [r[metric] for r in rows], ".-",
                    color=color, linewidth=1.1, label=labels[method])
        for method, marker in markers.items():
            row = next(r for r in summary if r["method"] == method)
            ax.scatter(row["bpp"], row[metric], marker=marker, s=36, label=method, zorder=3)
        ax.set(xlabel="Complete file bits per pixel", ylabel=label)
        ax.grid(alpha=.22)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.savefig(output / "rate_distortion.png", dpi=185)
    plt.close(fig)


def plot_examples(examples, output):
    columns = ("Clean", "JPEG Q10", "VQ 256 / zlib", "VQ 128 / zlib")
    fig, axes = plt.subplots(len(examples), len(columns), figsize=(8, 4.15), squeeze=False)
    for i, example in enumerate(examples.values()):
        for ax, name in zip(axes[i], columns):
            array, caption = example[name]
            ax.imshow(array, interpolation="nearest")
            ax.set_title(name + ("\n" + caption if caption else ""), fontsize=8)
            ax.axis("off")
    fig.tight_layout(pad=.65)
    fig.savefig(output / "decoded_examples.png", dpi=185)
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    paths = protocol["used_paths"]
    for split in ("train", "test"):
        if any(p.split("/")[0] != split or ".." in Path(p).parts for p in paths[split]):
            raise ValueError(f"Invalid relative {split} paths")
    crop = protocol["data_config"]["crop_size"]
    device = torch.device("cpu")
    calibration = DataLoader(ImageDataset([args.data_root / p for p in paths["train"]], crop),
                             batch_size=16)
    test = DataLoader(ImageDataset([args.data_root / p for p in paths["test"]], crop))
    base, state = load_checkpoint(args.checkpoint, inference_only)
    reduced, keep, coverage = prune(base, calibration)
    state.update(model_config=reduced.config, state_dict=reduced.state_dict(), retained_ids=keep)
    torch.save(state, args.output / "pruned.pt")
    path = fit_context_checkpoint(args.checkpoint, calibration, inference_only, device,
                                  output=args.output / "context.pt")
    context, _ = load_checkpoint(path, inference_only)
    choices = {"VQ 256 / zlib": base, "VQ 128 / zlib": reduced, "VQ 256 / context": context}
    qualities = (*range(1, 33), 50, 70, 90)
    rows, examples = [], {1: {}, 3: {}}
    for i, x in enumerate(test):
        array = x[0].permute(1, 2, 0).numpy()
        if i in examples:
            examples[i]["Clean"] = (array, "")
        reference = base(x)
        for name, model in choices.items():
            hard = reduced(x) if model is reduced else reference
            stream = compress(model, x)
            with patch.object(model, "decode_indices", wraps=model.decode_indices) as lookup:
                decoded = decompress(model, stream)
                torch.testing.assert_close(lookup.call_args.args[0].long(), hard["indices"], rtol=0, atol=0)
            torch.testing.assert_close(decoded, hard["reconstruction"], rtol=0, atol=0)
            psnr, ssim = metrics(decoded, x)
            mode = _parts(stream)[1].get("c", 0)
            rows.append(dict(image=i, method=name, q="", file_bytes=len(stream), payload_bytes=payload_size(stream),
                             pixels=crop**2, psnr=psnr, ssim=ssim, mode=mode))
            if i in examples:
                examples[i][name] = (decoded[0].permute(1, 2, 0).numpy(),
                                     f"{psnr:.1f} dB, {8*len(stream)/crop**2:.2f} bpp")
        for q in qualities:
            stream, jpeg = jpeg_roundtrip(array, q)
            optimized, opt_pixels = jpeg_roundtrip(array, q, optimize=True)
            np.testing.assert_array_equal(jpeg, opt_pixels)
            psnr, ssim = metrics(torch.from_numpy(jpeg).permute(2, 0, 1)[None], x)
            for method, bitstream in (("jpeg", stream), ("jpeg_opt", optimized)):
                rows.append(dict(image=i, method=method, q=q,
                                 file_bytes=len(bitstream), payload_bytes="", pixels=crop**2,
                                 psnr=psnr, ssim=ssim, mode=""))
            if i in examples and q == 10:
                examples[i]["JPEG Q10"] = (jpeg, f"{psnr:.1f} dB, {8*len(stream)/crop**2:.2f} bpp")
    with (args.output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method, q in dict.fromkeys((r["method"], r["q"]) for r in rows):
        group = [r for r in rows if (r["method"], r["q"]) == (method, q)]
        pixels = sum(r["pixels"] for r in group)
        entry = dict(method=method, q=q, n=len(group), bpp=8*sum(r["file_bytes"] for r in group)/pixels,
                     psnr=float(np.mean([r["psnr"] for r in group])), ssim=float(np.mean([r["ssim"] for r in group])))
        if method in choices:
            entry.update(payload_bpp=8*sum(r["payload_bytes"] for r in group)/pixels,
                         modes={mode: sum(r["mode"] == mode for r in group) for mode in (0, 1, 2)})
        summary.append(entry)
    for field in ("psnr", "ssim"):
        assert math.isclose(summary[0][field], summary[2][field], rel_tol=0, abs_tol=1e-10)
    record = dict(calibration_n=len(calibration.dataset), coverage=coverage, retained_ids=keep,
                  checkpoints={"base": args.checkpoint.stat().st_size, "context": path.stat().st_size,
                               "pruned": (args.output / "pruned.pt").stat().st_size},
                  examples=list(examples), exact_index_and_image_checks=3*len(test.dataset), results=summary)
    (args.output / "summary.json").write_text(json.dumps(record, indent=2))
    plot_curves(summary, args.output)
    plot_examples(examples, args.output)
    print(f"Checked {record['exact_index_and_image_checks']} exact codec round trips "
          f"and identical JPEG pixels across Huffman settings for {len(qualities)} qualities.")
    for row in summary[:3]:
        print(f"{row['method']}: {row['bpp']:.4f} BPP, {row['psnr']:.2f} dB, SSIM {row['ssim']:.4f}")


if __name__ == "__main__":
    main()
