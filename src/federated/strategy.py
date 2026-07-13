"""
strategy.py

Federated Learning Strategy
FedNeRF-Privacy3D

Compatible with Flower 1.32.x
"""

from flwr.server.strategy import FedAvg

from src.config import (
    NUM_CLIENTS,
)


def get_strategy():
    """
    Create the default FedAvg strategy.

    This function acts as a factory so that future
    strategies (FedProx, FedAdam, FedYogi, etc.)
    can be added without changing the rest of
    the project.
    """

    strategy = FedAvg(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=NUM_CLIENTS,

        min_evaluate_clients=NUM_CLIENTS,

        min_available_clients=NUM_CLIENTS,

        accept_failures=False,

    )

    return strategy