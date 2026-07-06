"""
train.py

Main training loop for FedNeRF-Privacy3D.
"""

import random

from src.data.dataset import BlenderDataset
from src.data.ray_batch_sampler import RayBatchSampler

from src.models.positional_encoding import PositionalEncoding
from src.models.tiny_nerf import TinyNeRF

from src.rendering.volume_rendering import VolumeRenderer

from src.core.trainer import Trainer
from src.core.checkpoint import CheckpointManager


def train_model(
    epochs=5,
    batch_size=1024,
):

    dataset = BlenderDataset("train")

    sampler = RayBatchSampler(dataset)

    trainer = Trainer(
        model=TinyNeRF(),
        encoder=PositionalEncoding(),
        renderer=VolumeRenderer(),
    )

    checkpoint = CheckpointManager()

    print("=" * 60)
    print("FedNeRF Local Training")
    print("=" * 60)

    for epoch in range(epochs):

        epoch_loss = 0.0
        epoch_psnr = 0.0

        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for image_index in indices:

            batch = sampler.sample_batch(
                image_index=image_index,
                batch_size=batch_size,
            )

            result = trainer.train_batch(
                batch["origin"],
                batch["direction"],
                batch["rgb"],
            )

            epoch_loss += result["loss"]
            epoch_psnr += result["psnr"]

        epoch_loss /= len(dataset)
        epoch_psnr /= len(dataset)

        print()

        print(f"Epoch {epoch+1}/{epochs}")

        print(f"Loss : {epoch_loss:.6f}")

        print(f"PSNR : {epoch_psnr:.2f}")

        checkpoint.save(
            trainer.model,
            trainer.optimizer,
            epoch + 1,
            epoch_loss,
        )

    print()

    print("Training Finished!")

    return trainer