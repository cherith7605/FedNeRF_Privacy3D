"""
image_renderer.py

Validation image renderer for FedNeRF-Privacy3D.
"""

from pathlib import Path

import torch
from torchvision.utils import save_image


class ImageRenderer:
    """
    Utility class for saving rendered validation images.
    """

    def __init__(self, output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_prediction(
        self,
        prediction,
        filename,
    ):
        """
        Save rendered prediction.
        """

        prediction = prediction.detach().cpu()

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0,
        )

        save_image(
            prediction,
            self.output_dir / filename,
        )

    def save_target(
        self,
        target,
        filename,
    ):
        """
        Save ground truth.
        """

        target = target.detach().cpu()

        target = torch.clamp(
            target,
            0.0,
            1.0,
        )

        save_image(
            target,
            self.output_dir / filename,
        )