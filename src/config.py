"""
config.py

Central configuration for FedNeRF-Privacy3D.
"""

from pathlib import Path
import torch

# ==========================================================
# Project Information
# ==========================================================

PROJECT_NAME = "FedNeRF-Privacy3D"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Device
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# Dataset
# ==========================================================

DATASET_NAME = "blender"

SCENE_NAME = "lego"

DATASET_ROOT = PROJECT_ROOT / "datasets"

SCENE_PATH = DATASET_ROOT / DATASET_NAME / SCENE_NAME

TRAIN_FOLDER = SCENE_PATH / "train"

VAL_FOLDER = SCENE_PATH / "val"

TEST_FOLDER = SCENE_PATH / "test"

TRAIN_JSON = SCENE_PATH / "transforms_train.json"

VAL_JSON = SCENE_PATH / "transforms_val.json"

TEST_JSON = SCENE_PATH / "transforms_test.json"

# ==========================================================
# Training
# ==========================================================

EPOCHS = 5

BATCH_SIZE = 512

LEARNING_RATE = 5e-4

NUM_WORKERS = 0

# ==========================================================
# NeRF Model
# ==========================================================

POSITIONAL_ENCODING_FREQ = 10

INPUT_DIM = 63

HIDDEN_DIM = 256

# ==========================================================
# Ray Sampling
# ==========================================================

NEAR = 2.0

FAR = 6.0

NUM_SAMPLES = 64

# ==========================================================
# Rendering
# ==========================================================

WHITE_BACKGROUND = False

# ==========================================================
# Federated Learning
# ==========================================================

NUM_CLIENTS = 3

LOCAL_EPOCHS = 1

GLOBAL_ROUNDS = 30

FEDERATED_BATCH_SIZE = 512

SERVER_ADDRESS = "127.0.0.1:8080"

SIMULATION_BACKEND = "ray"

CLIENT_RESOURCES = {
    "num_cpus": 1,
    "num_gpus": 0.0,
}

# ==========================================================
# Privacy
# ==========================================================

# ==========================================================
# Privacy
# ==========================================================

ENABLE_DIFFERENTIAL_PRIVACY = True

ENABLE_SECURE_AGGREGATION = False

MAX_GRAD_NORM = 1.0

NOISE_MULTIPLIER = 0.05

DELTA = 1e-5

# ==========================================================
# Logging
# ==========================================================

RUNS_PATH = PROJECT_ROOT / "runs"

LOG_PATH = PROJECT_ROOT / "logs"

# ==========================================================
# Outputs
# ==========================================================

OUTPUT_PATH = PROJECT_ROOT / "outputs"

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints"

GRAPH_PATH = OUTPUT_PATH / "graphs"

RENDER_PATH = OUTPUT_PATH / "renders"

SAVED_MODEL_PATH = PROJECT_ROOT / "saved_models"

RESULTS_PATH = PROJECT_ROOT / "results"

# ==========================================================
# Checkpoint
# ==========================================================

CHECKPOINT_NAME = "latest.pth"

# ==========================================================
# TensorBoard
# ==========================================================

TENSORBOARD_LOGDIR = RUNS_PATH