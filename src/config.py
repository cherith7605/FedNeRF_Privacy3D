from pathlib import Path

# ==============================
# Project Information
# ==============================

PROJECT_NAME = "FedNeRF-Privacy3D"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================
# Dataset
# ==============================

DATASET_NAME = "blender"

SCENE_NAME = "lego"

DATASET_ROOT = PROJECT_ROOT / "datasets"

SCENE_PATH = DATASET_ROOT / DATASET_NAME / SCENE_NAME

TRAIN_FOLDER = SCENE_PATH / "train"

TEST_FOLDER = SCENE_PATH / "test"

VAL_FOLDER = SCENE_PATH / "val"

TRAIN_JSON = SCENE_PATH / "transforms_train.json"

TEST_JSON = SCENE_PATH / "transforms_test.json"

VAL_JSON = SCENE_PATH / "transforms_val.json"

# ==============================
# Outputs
# ==============================

OUTPUT_PATH = PROJECT_ROOT / "outputs"

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints"

GRAPH_PATH = OUTPUT_PATH / "graphs"

RENDER_PATH = OUTPUT_PATH / "renders"

# ==============================
# Random Seed
# ==============================

RANDOM_SEED = 42