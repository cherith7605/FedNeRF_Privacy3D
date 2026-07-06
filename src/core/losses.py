"""
losses.py

Loss functions and evaluation metrics for NeRF.
"""

import math
import torch
import torch.nn as nn


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
        return self.mse(prediction, target)


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