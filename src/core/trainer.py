"""
trainer.py

Stable Generic NeRF Trainer
"""

import torch
import torch.optim as optim

from src.data.sampler import RaySampler
from src.core.losses import (
    NeRFLoss,
    compute_psnr,
)


class Trainer:

    def __init__(
        self,
        model,
        encoder,
        renderer,
        learning_rate=5e-4,
        device="cpu",
    ):

        self.device = device

        self.model = model.to(device)

        self.encoder = encoder.to(device)

        self.renderer = renderer.to(device)

        self.loss_fn = NeRFLoss()

        from src.privacy.dp_optimizer import DPOptimizer

        base_optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.optimizer = DPOptimizer(
            base_optimizer,
            max_grad_norm=1.0,
            noise_multiplier=0.05,
        )

        self.ray_sampler = RaySampler()

    def train_batch(
        self,
        ray_origins,
        ray_directions,
        target_rgb,
    ):

        self.model.train()

        ray_origins = ray_origins.to(self.device)
        ray_directions = ray_directions.to(self.device)
        target_rgb = target_rgb.to(self.device)

        points, depth = self.ray_sampler.sample_points(
            ray_origins,
            ray_directions,
        )

        batch_size = points.shape[0]
        num_samples = points.shape[1]

        encoded = self.encoder(
            points.reshape(-1, 3)
        )

        rgb, density = self.model(encoded)

        rgb = rgb.reshape(
            batch_size,
            num_samples,
            3,
        )

        density = density.reshape(
            batch_size,
            num_samples,
            1,
        )

        prediction, _, _, _ = self.renderer(
            rgb,
            density,
            depth,
        )

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0,
        )

        if torch.isnan(prediction).any():
            raise RuntimeError("NaN detected in renderer output.")

        if torch.isinf(prediction).any():
            raise RuntimeError("Inf detected in renderer output.")

        loss = self.loss_fn(
            prediction,
            target_rgb,
        )

        if torch.isnan(loss):
            raise RuntimeError("Loss became NaN.")

        if torch.isinf(loss):
            raise RuntimeError("Loss became Inf.")

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "psnr": compute_psnr(loss),
        }