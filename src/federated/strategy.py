"""
strategy.py

Federated Learning Strategy
FedNeRF-Privacy3D

Compatible with Flower 1.32.x
"""

from typing import List, Tuple

from flwr.common import Metrics
from flwr.server.strategy import FedAvg

from src.config import NUM_CLIENTS


def fit_metrics_aggregation_fn(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """
    Aggregate training metrics from all clients.
    """

    total_examples = sum(num_examples for num_examples, _ in metrics)

    avg_loss = (
        sum(
            num_examples * m.get("loss", 0.0)
            for num_examples, m in metrics
        )
        / total_examples
    )

    avg_psnr = (
        sum(
            num_examples * m.get("psnr", 0.0)
            for num_examples, m in metrics
        )
        / total_examples
    )

    return {
        "loss": avg_loss,
        "psnr": avg_psnr,
    }


def evaluate_metrics_aggregation_fn(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """
    Aggregate validation metrics.
    """

    total_examples = sum(num_examples for num_examples, _ in metrics)

    avg_loss = (
        sum(
            num_examples * m.get("loss", 0.0)
            for num_examples, m in metrics
        )
        / total_examples
    )

    avg_psnr = (
        sum(
            num_examples * m.get("psnr", 0.0)
            for num_examples, m in metrics
        )
        / total_examples
    )

    return {
        "loss": avg_loss,
        "psnr": avg_psnr,
    }


def get_strategy():
    """
    Create the default FedAvg strategy.
    """

    strategy = FedAvg(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=NUM_CLIENTS,

        min_evaluate_clients=NUM_CLIENTS,

        min_available_clients=NUM_CLIENTS,

        accept_failures=False,

        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,

        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,

    )

    return strategy