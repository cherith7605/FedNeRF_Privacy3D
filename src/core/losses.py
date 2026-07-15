"""
losses.py

Loss functions and evaluation metrics for NeRF.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from skimage.metrics import structural_similarity


class NeRFLoss(nn.Module):
    """
    Mean Squared Error loss used for NeRF.
    """

    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ):
        return self.mse(
            prediction,
            target,
        )


def compute_psnr(
    mse: torch.Tensor,
):
    """
    Compute Peak Signal-to-Noise Ratio.
    """

    mse_value = mse.item()

    if mse_value == 0:
        return 100.0

    return -10.0 * math.log10(mse_value)


def compute_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
):
    """
    Compute Structural Similarity Index (SSIM).

    Returns
    -------
    float
    """

    prediction = prediction.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

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

    if prediction.ndim == 2:

        prediction = prediction[:, None, :]

        target = target[:, None, :]

    value = structural_similarity(
        prediction,
        target,
        channel_axis=-1,
        data_range=1.0,
    )

    return float(value)