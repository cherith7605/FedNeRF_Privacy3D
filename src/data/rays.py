"""
rays.py

Ray generation utilities for NeRF.
"""

import torch


class RayGenerator:
    """
    Generates camera rays from camera intrinsics
    and camera-to-world pose.
    """

    def __init__(self, camera):

        self.camera = camera

    def generate_rays(self, pose):

        H = self.camera.height
        W = self.camera.width
        focal = self.camera.focal

        i, j = torch.meshgrid(
            torch.arange(W, dtype=torch.float32),
            torch.arange(H, dtype=torch.float32),
            indexing="xy"
        )

        directions = torch.stack([
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i)
        ], dim=-1)

        rotation = pose[:3, :3]
        translation = pose[:3, 3]

        rays_d = directions @ rotation.T

        rays_d = rays_d / torch.norm(
            rays_d,
            dim=-1,
            keepdim=True
        )

        rays_o = translation.expand_as(rays_d)

        return rays_o, rays_d