"""
camera.py

Camera utilities for Blender NeRF dataset.
"""

import math
import torch


class Camera:
    """
    Represents the intrinsic parameters of a Blender camera.
    """

    def __init__(self, image_width, image_height, camera_angle_x):

        self.width = image_width
        self.height = image_height
        self.camera_angle_x = camera_angle_x

        self.focal = (
            0.5 * image_width /
            math.tan(0.5 * camera_angle_x)
        )

    @property
    def intrinsic_matrix(self):

        return torch.tensor([
            [self.focal, 0, self.width / 2],
            [0, self.focal, self.height / 2],
            [0, 0, 1]
        ], dtype=torch.float32)

    def summary(self):

        print("=" * 60)
        print("Camera Information")
        print("=" * 60)
        print(f"Image Width  : {self.width}")
        print(f"Image Height : {self.height}")
        print(f"Camera FOV   : {self.camera_angle_x:.6f}")
        print(f"Focal Length : {self.focal:.2f}")
        print("=" * 60)