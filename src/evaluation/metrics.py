"""
metrics.py

Image quality evaluation utilities.
"""

import math

import numpy as np
import torch

from skimage.metrics import structural_similarity

import lpips


_lpips_model = lpips.LPIPS(
    net="alex"
)

_lpips_model.eval()


def compute_psnr(
    prediction,
    target,
):

    prediction = prediction.detach().cpu()
    target = target.detach().cpu()

    mse = torch.mean(
        (prediction - target) ** 2
    ).item()

    if mse == 0:
        return 100.0

    return -10.0 * math.log10(mse)


def compute_ssim(
    prediction,
    target,
):

    prediction = (
        prediction.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    target = (
        target.detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    prediction = np.clip(
        prediction,
        0.0,
        1.0,
    )

    target = np.clip(
        target,
        0.0,
        1.0,
    )

    return float(
        structural_similarity(
            prediction,
            target,
            channel_axis=2,
            data_range=1.0,
        )
    )


def compute_lpips(
    prediction,
    target,
):
    """
    Compute LPIPS perceptual distance.

    Lower is better.
    """

    prediction = (
        prediction.detach()
        .cpu()
        * 2.0
        - 1.0
    )

    target = (
        target.detach()
        .cpu()
        * 2.0
        - 1.0
    )

    value = _lpips_model(
        prediction.unsqueeze(0),
        target.unsqueeze(0),
    )

    return float(value.item())


def evaluate_images(
    prediction,
    target,
):

    return {

        "psnr": compute_psnr(
            prediction,
            target,
        ),

        "ssim": compute_ssim(
            prediction,
            target,
        ),

        "lpips": compute_lpips(
            prediction,
            target,
        ),

    }