"""
sampler.py

Ray sampling for NeRF.
"""

import torch


class RaySampler:

    def __init__(
        self,
        near=2.0,
        far=6.0,
        num_samples=64,
    ):
        self.near = near
        self.far = far
        self.num_samples = num_samples

    def sample_points(
        self,
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
    ):

        t_values = torch.linspace(
            self.near,
            self.far,
            self.num_samples,
            device=ray_origins.device,
        )

        # Full image: (H, W, 3)
        if ray_origins.dim() == 3:

            sample_points = (
                ray_origins[..., None, :]
                + ray_directions[..., None, :]
                * t_values.view(1, 1, -1, 1)
            )

        # Ray batch: (N, 3)
        elif ray_origins.dim() == 2:

            sample_points = (
                ray_origins[:, None, :]
                + ray_directions[:, None, :]
                * t_values.view(1, -1, 1)
            )

        else:
            raise ValueError(
                f"Unsupported ray shape: {ray_origins.shape}"
            )

        return sample_points, t_values