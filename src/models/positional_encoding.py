"""
Positional Encoding
-------------------
Implementation of the positional encoding used in the
original NeRF paper.

Author: FedNeRF-Privacy3D
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Converts a 3D coordinate into a higher-dimensional
    representation using sinusoidal functions.
    """

    def __init__(
        self,
        input_dims: int = 3,
        num_frequencies: int = 10,
        include_input: bool = True,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        self.output_dim = (
            input_dims if include_input else 0
        ) + input_dims * 2 * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        encoding = []

        if self.include_input:
            encoding.append(x)

        for i in range(self.num_frequencies):

            frequency = 2.0 ** i

            encoding.append(
                torch.sin(frequency * math.pi * x)
            )

            encoding.append(
                torch.cos(frequency * math.pi * x)
            )

        return torch.cat(encoding, dim=-1)