"""
evaluator.py

Offline evaluation pipeline for FedNeRF-Privacy3D.
"""

from src.models.tiny_nerf import TinyNeRF
from src.models.positional_encoding import PositionalEncoding

from src.rendering.volume_rendering import VolumeRenderer

from src.core.trainer import Trainer
from src.core.checkpoint import CheckpointManager
from src.core.validate import validate_model

from src.config import (
    DEVICE,
    BATCH_SIZE,
)


class Evaluator:
    """
    Offline model evaluator.
    """

    def __init__(self):

        self.trainer = Trainer(
            model=TinyNeRF(),
            encoder=PositionalEncoding(),
            renderer=VolumeRenderer(),
            device=DEVICE,
        )

        self.checkpoint = CheckpointManager()

    def load_checkpoint(
        self,
        filename="latest.pth",
    ):

        checkpoint = self.checkpoint.load(
            self.trainer.model,
            self.trainer.optimizer,
            filename,
        )

        return checkpoint

    def evaluate(
        self,
        batch_size=BATCH_SIZE,
    ):

        metrics = validate_model(
            self.trainer,
            batch_size=batch_size,
        )

        return metrics