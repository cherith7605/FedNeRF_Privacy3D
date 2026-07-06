from src.data.dataset import BlenderDataset
from src.data.camera import Camera
from src.data.rays import RayGenerator
from src.data.sampler import RaySampler

dataset = BlenderDataset("train")

camera = Camera(
    dataset.image_width,
    dataset.image_height,
    dataset.camera_angle_x
)

sample = dataset[0]

generator = RayGenerator(camera)

origins, directions = generator.generate_rays(
    sample["pose"]
)

sampler = RaySampler()

points, t = sampler.sample_points(
    origins,
    directions
)

print("=" * 60)

print("Origins Shape")

print(origins.shape)

print()

print("Directions Shape")

print(directions.shape)

print()

print("Sample Points Shape")

print(points.shape)

print()

print("Depth Samples")

print(t.shape)

print()

print("Center Pixel")

print(points[400, 400])

print("=" * 60)