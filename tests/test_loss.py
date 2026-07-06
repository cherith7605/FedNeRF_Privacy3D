import torch

from src.core.losses import (
    NeRFLoss,
    compute_psnr,
)

criterion = NeRFLoss()

prediction = torch.rand(1024, 3)

target = torch.rand(1024, 3)

loss = criterion(
    prediction,
    target,
)

print("=" * 60)

print("NeRF Loss Test")

print("=" * 60)

print()

print("Loss")

print(loss)

print()

print("PSNR")

print(compute_psnr(loss))

print("=" * 60)