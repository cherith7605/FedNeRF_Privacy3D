"""
test_dp_optimizer.py
"""

import torch

from src.models.tiny_nerf import TinyNeRF
from src.privacy.dp_optimizer import DPOptimizer

print("=" * 60)
print("Differential Privacy Optimizer Test")
print("=" * 60)

model = TinyNeRF()

optimizer = DPOptimizer(
    torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
)

loss = 0.0

for parameter in model.parameters():

    loss = loss + parameter.mean()

optimizer.zero_grad()

loss.backward()

optimizer.step()

print()

print("DP Optimizer Executed Successfully")

print()

print("=" * 60)