"""
client_app.py

Modern Flower ClientApp for FedNeRF-Privacy3D.
Compatible with Flower 1.32.x
"""

from flwr.client import ClientApp
from flwr.common import Context

from src.federated.client import FedNeRFClient


def client_fn(context: Context):
    """
    Create one Flower client.

    A new LocalTrainer is created for every client.
    """

    client = FedNeRFClient()

    return client.to_client()


app = ClientApp(
    client_fn=client_fn,
)