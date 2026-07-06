from src.core.train import train_model

trainer = train_model(
    epochs=1,
    batch_size=512,
)

print()

print("=" * 60)

print("Training Test Completed")

print("=" * 60)