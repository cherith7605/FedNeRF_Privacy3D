"""
tiny_nerf.py

Hybrid Tiny-NeRF Network

Predicts:
    RGB (3)
    Density (1)
"""

import torch
import torch.nn as nn


class TinyNeRF(nn.Module):
    """
    Hybrid Tiny-NeRF implementation.
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.density_head = nn.Linear(hidden_dim, 1)

        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, x):

        x1 = self.block1(x)

        x2 = torch.cat([x1, x], dim=-1)

        features = self.block2(x2)

        density = self.density_head(features)

        rgb = self.rgb_head(features)

        return rgb, density