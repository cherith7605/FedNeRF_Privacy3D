from src.data.dataset import BlenderDataset
from src.data.camera import Camera
from src.data.rays import RayGenerator

dataset = BlenderDataset("train")

camera = Camera(
    dataset.image_width,
    dataset.image_height,
    dataset.camera_angle_x
)

sample = dataset[0]

pose = sample["pose"]

generator = RayGenerator(camera)

origins, directions = generator.generate_rays(pose)

print("=" * 60)

print("Origins Shape :", origins.shape)

print("Directions Shape :", directions.shape)

print()

print("Center Ray Origin")

print(origins[400, 400])

print()

print("Center Ray Direction")

print(directions[400, 400])

print("=" * 60)