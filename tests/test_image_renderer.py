"""
test_image_renderer.py
"""

import torch

from src.rendering.image_renderer import ImageRenderer

print("=" * 60)
print("Validation Image Renderer Test")
print("=" * 60)

renderer = ImageRenderer(
    "outputs/renders",
)

prediction = torch.rand(
    3,
    64,
    64,
)

target = torch.rand(
    3,
    64,
    64,
)

renderer.save_prediction(
    prediction,
    "prediction.png",
)

renderer.save_target(
    target,
    "target.png",
)

print()

print("Images saved successfully.")

print()

print("outputs/renders/prediction.png")

print("outputs/renders/target.png")

print("=" * 60)