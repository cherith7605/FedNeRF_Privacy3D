"""
image_evaluator.py

Offline image renderer for FedNeRF-Privacy3D.
"""

from pathlib import Path

import torch
from torchvision.utils import save_image

from src.data.dataset import BlenderDataset
from src.data.camera import Camera
from src.data.rays import RayGenerator
from src.data.sampler import RaySampler

from src.models.tiny_nerf import TinyNeRF
from src.models.positional_encoding import PositionalEncoding

from src.rendering.volume_rendering import VolumeRenderer

from src.core.checkpoint import CheckpointManager

from src.config import DEVICE


class ImageEvaluator:

    def __init__(self):

        self.dataset = BlenderDataset("val")

        self.camera = Camera(
            self.dataset.image_width,
            self.dataset.image_height,
            self.dataset.camera_angle_x,
        )

        self.ray_generator = RayGenerator(self.camera)

        self.model = TinyNeRF().to(DEVICE)

        self.encoder = PositionalEncoding().to(DEVICE)

        self.renderer = VolumeRenderer().to(DEVICE)

        self.ray_sampler = RaySampler()

        self.checkpoint = CheckpointManager()

        Path("outputs/renders").mkdir(
            parents=True,
            exist_ok=True,
        )

    def load_checkpoint(
        self,
        filename="latest.pth",
    ):

        optimizer = torch.optim.Adam(
            self.model.parameters()
        )

        self.checkpoint.load(
            self.model,
            optimizer,
            filename,
        )

        self.model.eval()

    def export_ground_truth(
        self,
        image_index=0,
    ):

        sample = self.dataset[image_index]

        save_image(
            sample["image"],
            f"outputs/renders/target_{image_index:03d}.png",
        )

        return sample

    @torch.no_grad()
    def render_prediction(
        self,
        image_index=0,
        chunk_size=2048,
    ):

        sample = self.dataset[image_index]

        pose = sample["pose"]

        rays_o, rays_d = self.ray_generator.generate_rays(pose)

        H, W, _ = rays_o.shape

        rays_o = rays_o.reshape(-1, 3).to(DEVICE)
        rays_d = rays_d.reshape(-1, 3).to(DEVICE)

        rendered_chunks = []

        self.model.eval()

        for start in range(0, rays_o.shape[0], chunk_size):

            end = min(start + chunk_size, rays_o.shape[0])

            batch_o = rays_o[start:end]
            batch_d = rays_d[start:end]

            points, depth = self.ray_sampler.sample_points(
                batch_o,
                batch_d,
            )

            encoded = self.encoder(
                points.reshape(-1, 3)
            )

            rgb, density = self.model(encoded)

            rgb = rgb.reshape(
                batch_o.shape[0],
                self.ray_sampler.num_samples,
                3,
            )

            density = density.reshape(
                batch_o.shape[0],
                self.ray_sampler.num_samples,
                1,
            )

            prediction, _, _, _ = self.renderer(
                rgb,
                density,
                depth.to(DEVICE),
            )

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0,
            )

            rendered_chunks.append(prediction.cpu())

        prediction = torch.cat(
            rendered_chunks,
            dim=0,
        )

        prediction = prediction.reshape(
            H,
            W,
            3,
        ).permute(
            2,
            0,
            1,
        )

        save_image(
            prediction,
            f"outputs/renders/prediction_{image_index:03d}.png",
        )

        return prediction