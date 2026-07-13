"""
strategy.py

Federated learning strategy for FedNeRF-Privacy3D.
"""

from flwr.server.strategy import FedAvg

from src.config import (
    GLOBAL_ROUNDS,
)


def get_strategy():
    """
    Create the federated averaging strategy.
    """

    strategy = FedAvg(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=3,

        min_evaluate_clients=3,

        min_available_clients=3,

        accept_failures=False,

    )

    return strategy