from src.data.verify_dataset import DatasetVerifier
from src.config import SCENE_PATH

verifier = DatasetVerifier(SCENE_PATH)

verifier.verify_folders()

verifier.verify_json()

verifier.verify_images()

verifier.verify_json_content()