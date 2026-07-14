"""
simulation.py

Flower Simulation Runner
FedNeRF-Privacy3D
"""

from flwr.simulation import run_simulation

from src.config import (
    NUM_CLIENTS,
    CLIENT_RESOURCES,
)

from src.federated.client_app import app as client_app
from src.federated.server_app import app as server_app


def run():

    print("=" * 60)
    print("FedNeRF Flower Simulation")
    print("=" * 60)

    backend_config = {
        "client_resources": CLIENT_RESOURCES,
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_name="ray",
        backend_config=backend_config,
    )


if __name__ == "__main__":
    run()