"""
visualizer.py

Visualization utilities for FedNeRF-Privacy3D.
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self):

        self.output_dir = Path(
            "outputs/figures"
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_comparison(
        self,
        prediction,
        target,
        image_index,
    ):

        prediction = prediction.detach().cpu()

        target = target.detach().cpu()

        prediction = torch.clamp(
            prediction,
            0,
            1,
        )

        target = torch.clamp(
            target,
            0,
            1,
        )

        error = torch.abs(
            prediction - target
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15,5),
        )

        axes[0].imshow(
            target.permute(1,2,0)
        )
        axes[0].set_title(
            "Ground Truth"
        )

        axes[1].imshow(
            prediction.permute(1,2,0)
        )
        axes[1].set_title(
            "Prediction"
        )

        axes[2].imshow(
            error.permute(1,2,0)
        )
        axes[2].set_title(
            "Absolute Error"
        )

        for ax in axes:

            ax.axis("off")

        plt.tight_layout()

        filename = (
            self.output_dir /
            f"comparison_{image_index:03d}.png"
        )

        plt.savefig(
            filename,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        return filename