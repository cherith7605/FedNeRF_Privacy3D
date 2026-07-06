"""
ray_batch_sampler.py

Random Ray Batch Sampler

Implements the training strategy used in the
original NeRF paper.

Author: FedNeRF-Privacy3D
"""

import torch

from src.data.camera import Camera
from src.data.rays import RayGenerator


class RayBatchSampler:

    def __init__(self, blender_dataset):

        self.dataset = blender_dataset

        self.camera = Camera(
            blender_dataset.image_width,
            blender_dataset.image_height,
            blender_dataset.camera_angle_x,
        )

        self.ray_generator = RayGenerator(self.camera)

    def sample_batch(
        self,
        image_index: int,
        batch_size: int = 1024,
    ):

        sample = self.dataset[image_index]

        image = sample["image"].permute(1, 2, 0)

        pose = sample["pose"]

        ray_origins, ray_directions = (
            self.ray_generator.generate_rays(pose)
        )

        H, W, _ = image.shape

        ray_origins = ray_origins.reshape(-1, 3)

        ray_directions = ray_directions.reshape(-1, 3)

        rgb = image.reshape(-1, 3)

        total_rays = ray_origins.shape[0]

        indices = torch.randperm(total_rays)[:batch_size]

        return {
            "origin": ray_origins[indices],
            "direction": ray_directions[indices],
            "rgb": rgb[indices],
        }