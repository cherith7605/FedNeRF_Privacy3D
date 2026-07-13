"""
client.py

Flower client for FedNeRF-Privacy3D.
"""

from collections import OrderedDict

import flwr as fl
import torch

from src.config import (
    BATCH_SIZE,
    LOCAL_EPOCHS,
)

from src.federated.local_trainer import LocalTrainer


class FedNeRFClient(fl.client.NumPyClient):
    """
    Flower client wrapping the LocalTrainer.
    """

    def __init__(self):

        self.local_trainer = LocalTrainer()

    def get_parameters(self, config):

        return self.local_trainer.get_parameters()

    def set_parameters(self, parameters):

        model = self.local_trainer.get_model()

        params_dict = zip(
            model.state_dict().keys(),
            parameters,
        )

        state_dict = OrderedDict(
            {
                key: torch.tensor(value)
                for key, value in params_dict
            }
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    def fit(
        self,
        parameters,
        config,
    ):

        self.set_parameters(parameters)

        self.local_trainer.train(
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
        )

        return (
            self.get_parameters(config),
            len(self.local_trainer.dataset),
            {},
        )

    def evaluate(
        self,
        parameters,
        config,
    ):

        self.set_parameters(parameters)

        metrics = self.local_trainer.evaluate(
            batch_size=BATCH_SIZE,
        )

        return (
            metrics["loss"],
            len(self.local_trainer.dataset),
            {
                "psnr": metrics["psnr"],
            },
        )


def client_fn(context):

    return FedNeRFClient().to_client()