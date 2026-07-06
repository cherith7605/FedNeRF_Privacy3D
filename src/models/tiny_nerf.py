"""
tiny_nerf.py

Tiny-NeRF Network
FedNeRF-Privacy3D

Stable Version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNeRF(nn.Module):
    """
    Tiny-NeRF MLP

    Input:
        Positional Encoding (63)

    Output:
        RGB (3)
        Density (1)
    """

    def __init__(
        self,
        input_dim=63,
        hidden_dim=256,
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
            nn.Sigmoid(),
        )

        self.initialize_weights()

    def initialize_weights(self):
        """
        Xavier initialization improves stability.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)

                nn.init.zeros_(module.bias)

    def forward(self, x):

        x1 = self.block1(x)

        x2 = torch.cat(
            [x1, x],
            dim=-1,
        )

        features = self.block2(x2)

        rgb = self.rgb_head(features)

        density = F.softplus(
            self.density_head(features)
        )

        return rgb, density