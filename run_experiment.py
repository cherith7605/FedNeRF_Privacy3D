"""
run_experiment.py

FedNeRF-Privacy3D

Main Experiment Runner
"""

from src.core.train import train_model


def main():

    print("=" * 60)
    print("FedNeRF-Privacy3D Experiment")
    print("=" * 60)

    train_model(
        epochs=5,
        batch_size=512,
    )

    print()
    print("=" * 60)
    print("Experiment Completed Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()