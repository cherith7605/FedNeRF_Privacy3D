"""
test_local_trainer.py

Tests the LocalTrainer wrapper.
"""

from src.federated.local_trainer import LocalTrainer


print("=" * 60)
print("Local Trainer Test")
print("=" * 60)

trainer = LocalTrainer()

print("\nModel Loaded Successfully")

parameters = trainer.get_parameters()

print(f"\nTotal Parameters : {len(parameters)}")

trainer.train(
    epochs=1,
    batch_size=512,
)

metrics = trainer.evaluate(
    batch_size=512,
)

print("\nValidation Results")

print(f"Loss : {metrics['loss']:.6f}")

print(f"PSNR : {metrics['psnr']:.2f}")

print("=" * 60)