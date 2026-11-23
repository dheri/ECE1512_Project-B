"""Image quality for normalized RGB tensors, without metric-library calls."""

import torch
import torch.nn.functional as F


def _rgb_pair(pred, target):
    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 3:
        raise ValueError("Expected matching [B, 3, H, W] RGB tensors")
    if pred.device != target.device:
        raise ValueError("Prediction and target must be on the same device")
    return pred.float(), target.float()


def rgb_psnr(pred, target, *, data_range=1.0, mse_floor=1e-12):
    """Return one PSNR in dB per image, averaging MSE over RGB and pixels."""
    pred, target = _rgb_pair(pred, target)
    mse = (pred - target).square().flatten(1).mean(1)
    return 10.0 * (data_range**2 / mse.clamp_min(mse_floor)).log10()


def rgb_ssim(pred, target):
    """Return per-image SSIM using valid 11x11 Gaussian windows, sigma 1.5."""
    pred, target = _rgb_pair(pred, target)
    if min(pred.shape[-2:]) < 11:
        raise ValueError("SSIM requires images at least 11x11 pixels")
    offsets = torch.arange(-5, 6, device=pred.device, dtype=pred.dtype)
    gaussian = (-0.5 * (offsets / 1.5).square()).softmax(0)
    horizontal = gaussian.view(1, 1, 1, 11).repeat(3, 1, 1, 1)
    vertical = gaussian.view(1, 1, 11, 1).repeat(3, 1, 1, 1)
    # Stack the images and their second moments, preserving each RGB channel.
    moments = torch.cat((pred, target, pred.square(), target.square(), pred * target))
    moments = F.conv2d(moments, horizontal, groups=3)
    mu_x, mu_y, ex2, ey2, exy = F.conv2d(moments, vertical, groups=3).chunk(5)
    var_x = (ex2 - mu_x.square()).clamp_min(0)
    var_y = (ey2 - mu_y.square()).clamp_min(0)
    cov = exy - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    window_ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)
                   / ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)))
    return window_ssim.mean(dim=(1, 2, 3))
