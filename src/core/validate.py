"""
validate.py

Validation pipeline for FedNeRF-Privacy3D.
"""

import random
import torch

from src.data.dataset import BlenderDataset
from src.data.ray_batch_sampler import RayBatchSampler

from src.core.losses import (
    NeRFLoss,
    compute_psnr,
)


def validate_model(
    trainer,
    batch_size=512,
):

    dataset = BlenderDataset("val")

    sampler = RayBatchSampler(dataset)

    loss_fn = NeRFLoss()

    trainer.model.eval()

    total_loss = 0.0
    total_psnr = 0.0

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    with torch.no_grad():

        for image_index in indices:

            batch = sampler.sample_batch(
                image_index=image_index,
                batch_size=batch_size,
            )

            ray_origins = batch["origin"].to(trainer.device)
            ray_directions = batch["direction"].to(trainer.device)
            target_rgb = batch["rgb"].to(trainer.device)

            points, depth = trainer.ray_sampler.sample_points(
                ray_origins,
                ray_directions,
            )

            batch_size_actual = points.shape[0]
            num_samples = points.shape[1]

            encoded = trainer.encoder(
                points.reshape(-1, 3)
            )

            rgb, density = trainer.model(encoded)

            rgb = rgb.reshape(
                batch_size_actual,
                num_samples,
                3,
            )

            density = density.reshape(
                batch_size_actual,
                num_samples,
                1,
            )

            prediction, _, _, _ = trainer.renderer(
                rgb,
                density,
                depth,
            )

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0,
            )

            loss = loss_fn(
                prediction,
                target_rgb,
            )

            total_loss += loss.item()
            total_psnr += compute_psnr(loss)

    trainer.model.train()

    total_loss /= len(dataset)
    total_psnr /= len(dataset)

    return {
        "loss": total_loss,
        "psnr": total_psnr,
    }