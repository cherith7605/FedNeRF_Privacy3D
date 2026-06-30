"""
dataset.py

PyTorch Dataset Loader for the Blender NeRF Synthetic Dataset.

Author: FedNeRF-Privacy3D
"""

from pathlib import Path
import json
from typing import Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from src.config import SCENE_PATH


class BlenderDataset(Dataset):
    """
    Blender NeRF Dataset Loader.

    Loads RGB images together with camera poses.
    """

    def __init__(self, split: str = "train"):

        self.split = split

        self.scene_path = Path(SCENE_PATH)

        self.image_folder = self.scene_path / split

        self.transform_file = self.scene_path / f"transforms_{split}.json"

        with open(self.transform_file, "r") as f:
            self.metadata = json.load(f)

        self.frames = self.metadata["frames"]
        self.camera_angle_x = self.metadata["camera_angle_x"]

        first_image = self.scene_path / (
            str(self.frames[0]["file_path"]).replace("./", "") + ".png"
        )

        from PIL import Image
        img = Image.open(first_image)

        self.image_width, self.image_height = img.size
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):

        return len(self.frames)

    def __getitem__(self, index: int) -> Dict:
    
        frame = self.frames[index]

        image_path = Path(frame["file_path"])

        # Remove leading "./" if present
        image_path = Path(str(image_path).replace("./", ""))

        # Make path relative to scene folder
        image_path = self.scene_path / image_path

        # Blender JSON usually omits the extension
        if image_path.suffix == "":
            image_path = image_path.with_suffix(".png")

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        pose = torch.tensor(
            frame["transform_matrix"],
            dtype=torch.float32
        )

        return {
            "image": image,
            "pose": pose,
            "index": index
        }