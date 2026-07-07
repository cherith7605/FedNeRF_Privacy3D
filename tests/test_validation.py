from src.core.train import train_model
from src.core.validate import validate_model

print("=" * 60)
print("Training...")
print("=" * 60)

trainer = train_model(
    epochs=1,
    batch_size=512,
)

print()

print("=" * 60)
print("Running Validation")
print("=" * 60)

metrics = validate_model(
    trainer,
    batch_size=512,
)

print()

print("=" * 60)
print("Validation Results")
print("=" * 60)

print("Validation Loss")

print(metrics["loss"])

print()

print("Validation PSNR")

print(metrics["psnr"])

print("=" * 60)