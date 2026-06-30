from src.data.dataset import BlenderDataset
from src.data.camera import Camera

dataset = BlenderDataset("train")

camera = Camera(
    dataset.image_width,
    dataset.image_height,
    dataset.camera_angle_x
)

camera.summary()

print()

print("Intrinsic Matrix")

print(camera.intrinsic_matrix)