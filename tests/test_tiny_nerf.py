import torch

from src.models.tiny_nerf import TinyNeRF

model = TinyNeRF()

print("=" * 60)
print("FedNeRF Tiny-NeRF Test")
print("=" * 60)

print(model)

dummy = torch.randn(5, 63)

rgb, density = model(dummy)

print()

print("Input Shape")

print(dummy.shape)

print()

print("RGB Shape")

print(rgb.shape)

print()

print("Density Shape")

print(density.shape)

print()

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total Parameters :", total_params)
print("Trainable Parameters :", trainable_params)

print("=" * 60)