"""
run_experiment.py

FedNeRF-Privacy3D

Complete experiment runner.
"""

from src.core.train import train_model
from src.core.validate import validate_model
from src.utils.logger import Logger


def main():

    logger = Logger()

    print("=" * 60)
    print("FedNeRF-Privacy3D Experiment")
    print("=" * 60)

    trainer = train_model(
        epochs=5,
        batch_size=512,
    )

    metrics = validate_model(
        trainer,
        batch_size=512,
    )

    logger.log_validation(
        epoch=5,
        loss=metrics["loss"],
        psnr=metrics["psnr"],
    )

    logger.close()

    print()

    print("=" * 60)
    print("Final Validation")
    print("=" * 60)

    print(f"Loss : {metrics['loss']:.6f}")
    print(f"PSNR : {metrics['psnr']:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    main()