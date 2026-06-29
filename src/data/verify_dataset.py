from pathlib import Path
import json
from PIL import Image

from src.config import SCENE_PATH


class DatasetVerifier:
    def __init__(self, scene_path):
        self.scene_path = Path(scene_path)

        self.splits = {
            "train": self.scene_path / "train",
            "test": self.scene_path / "test",
            "val": self.scene_path / "val",
        }

        self.json_files = {
            "train": self.scene_path / "transforms_train.json",
            "test": self.scene_path / "transforms_test.json",
            "val": self.scene_path / "transforms_val.json",
        }

    def verify_folders(self):
        print("\nChecking folders...")

        for name, folder in self.splits.items():
            if folder.exists():
                print(f"[OK] {name:<5} folder found")
            else:
                print(f"[ERROR] {name:<5} folder missing")

    def verify_json(self):
        print("\nChecking transform files...")

        for name, file in self.json_files.items():
            if file.exists():
                print(f"[OK] transforms_{name}.json")
            else:
                print(f"[ERROR] transforms_{name}.json missing")

    def verify_images(self):
        print("\nChecking images...")

        total = 0

        for split, folder in self.splits.items():

            images = sorted(
                img for img in folder.glob("*.png")
                if "_depth_" not in img.name
            )

            print(f"{split:<5}: {len(images)} images")

            total += len(images)

            if len(images):

                try:
                    Image.open(images[0]).verify()
                    print(f"      first image OK")

                except Exception:
                    print(f"      image corrupted")

        print("\nTotal Images :", total)

    def verify_json_content(self):
        print("\nChecking camera poses...")

        for split, file in self.json_files.items():

            with open(file, "r") as f:
                data = json.load(f)

            frames = data["frames"]

            print(f"{split:<5}: {len(frames)} camera poses")


if __name__ == "__main__":

    print("=" * 60)
    print("FedNeRF Dataset Verification")
    print("=" * 60)

    verifier = DatasetVerifier(SCENE_PATH)

    verifier.verify_folders()

    verifier.verify_json()

    verifier.verify_images()

    verifier.verify_json_content()

    print("\nDataset verification completed successfully.")