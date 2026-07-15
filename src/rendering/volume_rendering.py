"""
volume_rendering.py

Stable Differentiable Volume Rendering
FedNeRF-Privacy3D
"""

import torch
import torch.nn as nn


class VolumeRenderer(nn.Module):
    """
    Stable NeRF Volume Renderer.

    Supports:
        (..., N_samples, 3)
    """

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

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
            (..., N_samples, 3)

        density
            (..., N_samples, 1)

        depth_values
            (N_samples,)
        """

        sigma = density.squeeze(-1)

        delta = depth_values[1:] - depth_values[:-1]

        delta = torch.cat(
            [
                delta,
                delta[-1:].clone(),
            ],
            dim=0,
        )

        view_shape = [1] * (sigma.dim() - 1) + [delta.shape[0]]

        delta = delta.view(*view_shape)

        sigma_delta = sigma * delta

        sigma_delta = torch.clamp(
            sigma_delta,
            min=0.0,
            max=50.0,
        )

        alpha = 1.0 - torch.exp(-sigma_delta)

        transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones_like(alpha[..., :1]),
                    1.0 - alpha + self.eps,
                ],
                dim=-1,
            ),
            dim=-1,
        )[..., :-1]

        weights = alpha * transmittance

        rgb_map = torch.sum(
            weights.unsqueeze(-1) * rgb,
            dim=-2,
        )

        depth_map = torch.sum(
            weights * depth_values.view(*view_shape),
            dim=-1,
        )

        acc_map = torch.sum(
            weights,
            dim=-1,
        )

        return (
            rgb_map,
            depth_map,
            acc_map,
            weights,
        )