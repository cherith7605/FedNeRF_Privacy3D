import torch

from src.rendering.volume_rendering import VolumeRenderer

renderer = VolumeRenderer()

num_samples = 64

rgb = torch.rand(1, num_samples, 3)

density = torch.rand(1, num_samples, 1)

depth = torch.linspace(
    2.0,
    6.0,
    num_samples,
)

rgb_map, depth_map, acc_map, weights = renderer(
    rgb,
    density,
    depth,
)

print("=" * 60)
print("Volume Rendering Test")
print("=" * 60)

print()

print("RGB Map Shape")

print(rgb_map.shape)

print()

print("Depth Map Shape")

print(depth_map.shape)

print()

print("Accumulation Shape")

print(acc_map.shape)

print()

print("Weights Shape")

print(weights.shape)

print()

print("Weight Sum")

print(weights.sum())

print("=" * 60)