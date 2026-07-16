"""
checkpoint.py

Checkpoint utilities for FedNeRF-Privacy3D.
"""

from pathlib import Path
import torch


class CheckpointManager:

    def __init__(self, checkpoint_dir="checkpoints"):

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        epoch,
        loss,
        filename="latest.pth",
    ):

        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        path = self.checkpoint_dir / filename

        torch.save(checkpoint, path)

        return path

    def load(
        self,
        model,
        optimizer,
        filename="latest.pth",
    ):

        path = self.checkpoint_dir / filename

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        return checkpoint