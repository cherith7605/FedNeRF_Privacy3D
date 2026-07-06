import torch

from src.models.positional_encoding import PositionalEncoding

encoder = PositionalEncoding()

print("=" * 60)
print("FedNeRF Positional Encoding Test")
print("=" * 60)

sample = torch.tensor([[1.0, 2.0, 3.0]])

encoded = encoder(sample)

print("Input Shape")

print(sample.shape)

print()

print("Encoded Shape")

print(encoded.shape)

print()

print("Output Dimension")

print(encoder.output_dim)

print()

print("First 15 Values")

print(encoded[0][:15])

print("=" * 60)