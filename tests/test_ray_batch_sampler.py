from src.data.dataset import BlenderDataset
from src.data.ray_batch_sampler import RayBatchSampler

dataset = BlenderDataset("train")

sampler = RayBatchSampler(dataset)

batch = sampler.sample_batch(
    image_index=0,
    batch_size=1024,
)

print("=" * 60)
print("Ray Batch Sampler Test")
print("=" * 60)

print()

print("Origin Shape")

print(batch["origin"].shape)

print()

print("Direction Shape")

print(batch["direction"].shape)

print()

print("RGB Shape")

print(batch["rgb"].shape)

print()

print("First Origin")

print(batch["origin"][0])

print("=" * 60)