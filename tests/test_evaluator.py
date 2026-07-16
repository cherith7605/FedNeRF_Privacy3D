"""
test_evaluator.py
"""

from src.evaluation.evaluator import Evaluator

print("=" * 60)
print("Offline Evaluator Test")
print("=" * 60)

evaluator = Evaluator()

checkpoint = evaluator.load_checkpoint()

print()

print("Checkpoint Loaded")

print(f"Epoch : {checkpoint['epoch']}")

metrics = evaluator.evaluate()

print()

print("Evaluation Complete")

print(f"Loss : {metrics['loss']:.6f}")

print(f"PSNR : {metrics['psnr']:.2f}")

print("=" * 60)