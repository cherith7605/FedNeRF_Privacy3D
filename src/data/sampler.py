"""
sampler.py

Ray Sampling utilities for NeRF.
"""

import torch


class RaySampler:
    """
    Uniform ray sampler.

    Samples points between the near
    and far planes.
    """

    def __init__(
        self,
        near: float = 2.0,
        far: float = 6.0,
        num_samples: int = 64,
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

        sample_points = (
            ray_origins[..., None, :]
            + ray_directions[..., None, :]
            * t_values[None, None, :, None]
        )

        return sample_points, t_values