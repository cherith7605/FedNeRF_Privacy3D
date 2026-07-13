"""
test_flower_client.py

Tests the Flower client.
"""

from src.federated.client import FedNeRFClient

print("=" * 60)
print("Flower Client Test")
print("=" * 60)

client = FedNeRFClient()

print("\nFlower Client Created Successfully")

parameters = client.get_parameters(config={})

print(f"\nNumber of Parameter Tensors : {len(parameters)}")

print("\nFirst Tensor Shape")

print(parameters[0].shape)

print("=" * 60)