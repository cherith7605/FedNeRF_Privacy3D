"""
server_app.py

Modern Flower ServerApp
FedNeRF-Privacy3D

Compatible with Flower 1.32.x
"""

from flwr.server import ServerApp
from flwr.server import ServerAppComponents
from flwr.common import Context

from src.federated.strategy import get_strategy


def server_fn(context: Context):
    """
    Create the Flower server components.
    """

    strategy = get_strategy()

    return ServerAppComponents(
        strategy=strategy,
    )


app = ServerApp(
    server_fn=server_fn,
)