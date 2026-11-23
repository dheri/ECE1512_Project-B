"""RGB PSNR and SSIM for normalized images."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_quality import rgb_psnr, rgb_ssim

def psnr(pred, target, data_range=1.0):
    return rgb_psnr(pred, target, data_range=data_range)


def batch_ssim(pred, target):
    return rgb_ssim(pred, target).mean()
