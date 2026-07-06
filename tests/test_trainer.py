from src.data.dataset import BlenderDataset
from src.data.ray_batch_sampler import RayBatchSampler

from src.models.positional_encoding import PositionalEncoding
from src.models.tiny_nerf import TinyNeRF

from src.rendering.volume_rendering import VolumeRenderer

from src.core.trainer import Trainer

dataset = BlenderDataset("train")

batch_sampler = RayBatchSampler(dataset)

batch = batch_sampler.sample_batch(
    image_index=0,
    batch_size=512,
)

trainer = Trainer(
    model=TinyNeRF(),
    encoder=PositionalEncoding(),
    renderer=VolumeRenderer(),
)

print("=" * 60)
print("Trainer Test")
print("=" * 60)

for i in range(3):

    result = trainer.train_batch(
        batch["origin"],
        batch["direction"],
        batch["rgb"],
    )

    print(
        f"Step {i+1}"
    )

    print(
        "Loss :",
        result["loss"]
    )

    print(
        "PSNR :",
        result["psnr"]
    )

    print()