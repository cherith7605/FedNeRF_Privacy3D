"""
dp_optimizer.py

Differential Privacy optimizer wrapper
for FedNeRF-Privacy3D.
"""

import torch


class DPOptimizer:
    """
    Implements Differential Privacy by

    1. Gradient clipping
    2. Gaussian noise addition
    """

    def __init__(
        self,
        optimizer,
        max_grad_norm=1.0,
        noise_multiplier=0.1,
    ):

        self.optimizer = optimizer

        self.max_grad_norm = max_grad_norm

        self.noise_multiplier = noise_multiplier

    def zero_grad(self):

        self.optimizer.zero_grad()

    def step(self):

        parameters = []

        for group in self.optimizer.param_groups:

            for parameter in group["params"]:

                if parameter.grad is not None:

                    parameters.append(parameter)

        torch.nn.utils.clip_grad_norm_(
            parameters,
            self.max_grad_norm,
        )

        for parameter in parameters:

            noise = torch.normal(
                mean=0.0,
                std=self.noise_multiplier,
                size=parameter.grad.shape,
                device=parameter.grad.device,
            )

            parameter.grad += noise

        self.optimizer.step()