import torch.optim as optim

from src.models.tiny_nerf import TinyNeRF
from src.core.checkpoint import CheckpointManager

model = TinyNeRF()

optimizer = optim.Adam(
    model.parameters(),
    lr=5e-4,
)

manager = CheckpointManager()

path = manager.save(
    model=model,
    optimizer=optimizer,
    epoch=5,
    loss=0.123,
)

print("=" * 60)
print("Checkpoint Test")
print("=" * 60)

print()

print("Saved To")

print(path)

checkpoint = manager.load(
    model,
    optimizer,
)

print()

print("Epoch")

print(checkpoint["epoch"])

print()

print("Loss")

print(checkpoint["loss"])

print("=" * 60)