"""
volume_rendering.py

Differentiable Volume Rendering used by NeRF.

Author: FedNeRF-Privacy3D
"""

import torch
import torch.nn as nn


class VolumeRenderer(nn.Module):
    """
    Differentiable volume renderer.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        rgb: torch.Tensor,
        density: torch.Tensor,
        depth_values: torch.Tensor,
    ):
        """
        Parameters
        ----------
        rgb
            (..., N, 3)

        density
            (..., N, 1)

        depth_values
            (N,)
        """

        sigma = density.squeeze(-1)

        delta = depth_values[1:] - depth_values[:-1]

        delta = torch.cat(
            [
                delta,
                torch.tensor(
                    [1e10],
                    device=depth_values.device,
                ),
            ]
        )

        alpha = 1.0 - torch.exp(
            -sigma * delta
        )

        transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones_like(alpha[..., :1]),
                    1.0 - alpha + 1e-10,
                ],
                dim=-1,
            ),
            dim=-1,
        )[..., :-1]

        weights = alpha * transmittance

        rgb_map = torch.sum(
            weights[..., None] * rgb,
            dim=-2,
        )

        depth_map = torch.sum(
            weights * depth_values,
            dim=-1,
        )

        acc_map = torch.sum(
            weights,
            dim=-1,
        )

        return rgb_map, depth_map, acc_map, weights