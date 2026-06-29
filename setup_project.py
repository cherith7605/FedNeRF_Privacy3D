from pathlib import Path

PROJECT_ROOT = Path.cwd()

folders = [
    "notebooks",
    "src",
    "src/data",
    "src/core",
    "src/federated",
    "src/evaluation",
    "src/utils",
    "datasets",
    "datasets/blender",
    "checkpoints",
    "checkpoints/local",
    "checkpoints/global",
    "saved_models",
    "outputs",
    "outputs/renders",
    "outputs/graphs",
    "outputs/videos",
    "outputs/heatmaps",
    "outputs/reports",
    "experiments",
    "results",
    "logs",
    "docs",
    "docs/architecture",
    "docs/diagrams",
    "docs/notes",
    "report",
    "presentation",
    "tests",
    "assets"
]

files = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
    "run_experiment.py",

    "src/__init__.py",
    "src/config.py",

    "src/data/__init__.py",
    "src/data/dataset.py",
    "src/data/preprocessing.py",
    "src/data/camera.py",
    "src/data/transforms.py",
    "src/data/verify_dataset.py",

    "src/core/__init__.py",
    "src/core/rays.py",
    "src/core/encoding.py",
    "src/core/tinynerf.py",
    "src/core/renderer.py",
    "src/core/losses.py",
    "src/core/trainer.py",

    "src/federated/__init__.py",
    "src/federated/client.py",
    "src/federated/server.py",
    "src/federated/fedavg.py",
    "src/federated/aggregation.py",
    "src/federated/privacy.py",
    "src/federated/communication.py",

    "src/evaluation/__init__.py",
    "src/evaluation/metrics.py",
    "src/evaluation/visualization.py",
    "src/evaluation/plots.py",
    "src/evaluation/reports.py",

    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/helpers.py",
    "src/utils/seed.py",
    "src/utils/io.py"
]

notebooks = [
    "01_Environment_Setup.ipynb",
    "02_Dataset_Management.ipynb",
    "03_Dataset_Loader.ipynb",
    "04_Camera_Mathematics.ipynb",
    "05_Ray_Generation.ipynb",
    "06_Positional_Encoding.ipynb",
    "07_Hybrid_TinyNeRF.ipynb",
    "08_Local_Training.ipynb",
    "09_Federated_Learning.ipynb",
    "10_Privacy_Module.ipynb",
    "11_Global_Reconstruction.ipynb",
    "12_Evaluation.ipynb",
    "13_Experiments.ipynb",
    "14_Demo.ipynb"
]

for folder in folders:
    (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)

for file in files:
    path = PROJECT_ROOT / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

for notebook in notebooks:
    (PROJECT_ROOT / "notebooks" / notebook).touch(exist_ok=True)

print("=" * 70)
print("FedNeRF-Privacy3D Project Initialized Successfully")
print("=" * 70)
print(f"Project Location : {PROJECT_ROOT}")
print(f"Folders Created  : {len(folders)}")
print(f"Files Created    : {len(files)}")
print(f"Notebooks Created: {len(notebooks)}")
print("=" * 70)