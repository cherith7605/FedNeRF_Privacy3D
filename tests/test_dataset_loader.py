from src.data.dataset import BlenderDataset

dataset = BlenderDataset("train")

print("=" * 60)

print("Dataset Length :", len(dataset))

sample = dataset[0]

print()

print("Image Shape :", sample["image"].shape)

print()

print("Pose Shape :", sample["pose"].shape)

print()

print(sample["pose"])

print("=" * 60)