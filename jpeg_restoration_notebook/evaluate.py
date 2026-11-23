import argparse
import torch
from torch.utils.data import DataLoader

from dataset import JPEGRestorationDataset
from model import ResidualRestorationCNN
from metrics import psnr, batch_ssim
import pandas as pd
from tqdm.auto import tqdm

TEST_QUALITIES = (10, 15, 20, 25, 30)


@torch.no_grad()
def evaluate_quality(model, data_dir, quality, patch_size, batch_size, device):
    cfg = dict(test_dir=data_dir, eval_patch_size=patch_size, batch_size=batch_size,
               num_workers=0, seed=0, qualities=())
    rows = evaluate_all({"CNN": model}, cfg, [quality], device=device).set_index("Method")
    return dict(jpeg_psnr=rows.at["JPEG", "PSNR"], model_psnr=rows.at["CNN", "PSNR"],
                jpeg_ssim=rows.at["JPEG", "SSIM"], model_ssim=rows.at["CNN", "SSIM"],
                jpeg_bpp=rows.at["JPEG", "BPP"])


def load_checkpoint(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint["args"]
    model = ResidualRestorationCNN(cfg["channels"], cfg["depth"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg



@torch.no_grad()
def evaluate_all(models, cfg, test_qualities, device=None, train_qualities=None, *, metric_fns=None):
    # Notebook callers can supply their shared metric functions.
    psnr_fn, ssim_fn = (psnr, batch_ssim) if metric_fns is None else metric_fns
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual_train_qualities = set(cfg["qualities"] if train_qualities is None else train_qualities)
    for model in models.values():
        model.eval()
    rows = []
    for quality in tqdm(test_qualities, desc="Test quality factors"):
        ds = JPEGRestorationDataset(cfg["test_dir"], cfg["eval_patch_size"],
                                    [quality], train=False, return_bpp=True,
                                    sample_limit=cfg.get("test_limit"), sample_seed=cfg["seed"] + 2)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])
        totals = {name: {"PSNR": 0.0, "SSIM": 0.0}
                  for name in ["JPEG", *models]}
        count, total_bpp = 0, 0.0
        for degraded, clean, _, bpp in loader:
            degraded, clean = degraded.to(device), clean.to(device)
            predictions = {"JPEG": degraded}
            predictions.update({name: model(degraded) for name, model in models.items()})
            n = clean.shape[0]
            for name, pred in predictions.items():
                totals[name]["PSNR"] += psnr_fn(pred, clean).sum().item()
                totals[name]["SSIM"] += ssim_fn(pred, clean).item() * n
            count += n
            total_bpp += bpp.sum().item()
        if not count:
            raise ValueError("Test loader is empty.")
        for name, values in totals.items():
            rows.append({"Q": quality, "Seen in training": quality in actual_train_qualities,
                         "Method": name, "Images": count, "BPP": total_bpp / count,
                         **{key: value / count for key, value in values.items()}})
    return pd.DataFrame(rows)



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--qualities', type=int, nargs='+', default=TEST_QUALITIES)
    p.add_argument('--patch_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_checkpoint(args.checkpoint, device)

    print('Q | JPEG BPP | JPEG PSNR | Model PSNR | JPEG SSIM | Model SSIM')
    print('-' * 90)
    for q in args.qualities:
        r = evaluate_quality(model, args.data_dir, q, args.patch_size, args.batch_size, device)
        print(
            f'{q:2d} | {r["jpeg_bpp"]:8.4f} | {r["jpeg_psnr"]:9.3f} | '
            f'{r["model_psnr"]:10.3f} | {r["jpeg_ssim"]:.4f} | '
            f'{r["model_ssim"]:.4f}'
        )


if __name__ == '__main__':
    main()
