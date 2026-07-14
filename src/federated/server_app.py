"""
server_app.py

Modern Flower ServerApp
FedNeRF-Privacy3D

Compatible with Flower 1.32.x
"""

from flwr.common import Context
from flwr.compat.server import ServerAppComponents
from flwr.server import ServerApp, ServerConfig

from src.config import GLOBAL_ROUNDS
from src.federated.strategy import get_strategy


def server_fn(context: Context):

    strategy = get_strategy()

    config = ServerConfig(
        num_rounds=GLOBAL_ROUNDS,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


app = ServerApp(
    server_fn=server_fn,
)