from src.utils.logger import Logger

logger = Logger()

for epoch in range(5):

    logger.log_train(
        epoch,
        loss=0.20 - epoch * 0.02,
        psnr=8 + epoch,
    )

logger.close()

print("=" * 60)
print("TensorBoard Logger Test")
print("=" * 60)
print()
print("Logs written to:")
print("runs/")
print("=" * 60)