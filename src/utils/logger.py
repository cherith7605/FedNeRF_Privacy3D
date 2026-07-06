"""
logger.py

TensorBoard Logger for FedNeRF-Privacy3D.
"""

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, log_dir="runs"):

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=str(self.log_dir)
        )

    def log_train(
        self,
        epoch,
        loss,
        psnr,
    ):

        self.writer.add_scalar(
            "Train/Loss",
            loss,
            epoch,
        )

        self.writer.add_scalar(
            "Train/PSNR",
            psnr,
            epoch,
        )

    def log_validation(
        self,
        epoch,
        loss,
        psnr,
    ):

        self.writer.add_scalar(
            "Validation/Loss",
            loss,
            epoch,
        )

        self.writer.add_scalar(
            "Validation/PSNR",
            psnr,
            epoch,
        )

    def close(self):
        self.writer.close()