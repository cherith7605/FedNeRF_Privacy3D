"""
local_trainer.py

Local training wrapper for Flower clients.

FedNeRF-Privacy3D
"""

import torch

from src.data.dataset import BlenderDataset
from src.data.ray_batch_sampler import RayBatchSampler

from src.models.positional_encoding import PositionalEncoding
from src.models.tiny_nerf import TinyNeRF

from src.rendering.volume_rendering import VolumeRenderer

from src.core.trainer import Trainer
from src.core.validate import validate_model

from src.config import (
    DEVICE,
    LEARNING_RATE,
)


class LocalTrainer:
    """
    Wrapper around the NeRF trainer used by each
    federated learning client.
    """

    def __init__(self):

        self.dataset = BlenderDataset("train")

        self.sampler = RayBatchSampler(self.dataset)

        self.trainer = Trainer(
            model=TinyNeRF(),
            encoder=PositionalEncoding(),
            renderer=VolumeRenderer(),
            learning_rate=LEARNING_RATE,
            device=DEVICE,
        )

    def get_model(self):
        return self.trainer.model

    def get_parameters(self):
        """
        Return model parameters as NumPy arrays.
        """

        return [
            parameter.detach().cpu().numpy()
            for parameter in self.trainer.model.parameters()
        ]

    def set_parameters(self, parameters):
        """
        Load parameters received from the server.
        """

        model = self.trainer.model

        state_dict = model.state_dict()

        keys = list(state_dict.keys())

        for key, value in zip(keys, parameters):

            state_dict[key] = torch.tensor(value)

        model.load_state_dict(state_dict)

    def train(
        self,
        epochs=1,
        batch_size=512,
    ):
        """
        Perform local client training.
        """

        for _ in range(epochs):

            for image_index in range(len(self.dataset)):

                batch = self.sampler.sample_batch(
                    image_index=image_index,
                    batch_size=batch_size,
                )

                self.trainer.train_batch(
                    batch["origin"],
                    batch["direction"],
                    batch["rgb"],
                )

    def evaluate(
        self,
        batch_size=512,
    ):
        """
        Evaluate the local model.
        """

        metrics = validate_model(
            self.trainer,
            batch_size=batch_size,
        )

        return metrics