"""
privacy_accountant.py

Privacy accounting utilities for
FedNeRF-Privacy3D.
"""

from pathlib import Path
import math


class PrivacyAccountant:
    """
    Simple privacy accountant.

    NOTE:
    This provides an approximate epsilon estimate
    for project reporting purposes.
    """

    def __init__(
        self,
        dp_enabled=True,
        noise_multiplier=0.05,
        max_grad_norm=1.0,
        delta=1e-5,
    ):

        self.dp_enabled = dp_enabled
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta

    def estimate_epsilon(
        self,
        steps,
    ):
        """
        Approximate epsilon estimation.
        """

        if not self.dp_enabled:
            return float("inf")

        epsilon = math.sqrt(
            steps * self.noise_multiplier
        )

        return round(epsilon, 4)

    def generate_report(
        self,
        steps,
    ):

        epsilon = self.estimate_epsilon(
            steps
        )

        report = [
            "=" * 60,
            "FedNeRF-Privacy3D Privacy Report",
            "=" * 60,
            f"DP Enabled         : {self.dp_enabled}",
            f"Noise Multiplier   : {self.noise_multiplier}",
            f"Gradient Clip Norm : {self.max_grad_norm}",
            f"Delta              : {self.delta}",
            f"Training Steps     : {steps}",
            f"Estimated Epsilon  : {epsilon}",
            "=" * 60,
        ]

        Path("results").mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = Path(
            "results/privacy_report.txt"
        )

        with open(output_path, "w") as file:
            file.write("\n".join(report))

        return output_path