"""
test_image_evaluator.py
"""

from pathlib import Path

from src.evaluation.image_evaluator import ImageEvaluator

print("=" * 60)
print("Validation Image Renderer Test")
print("=" * 60)

evaluator = ImageEvaluator()

evaluator.load_checkpoint()

evaluator.export_ground_truth(0)

prediction = evaluator.render_prediction(0)

print()

print("Ground Truth Saved")

print(Path("outputs/renders/target_000.png"))

print()

print("Prediction Saved")

print(Path("outputs/renders/prediction_000.png"))

print()

print(prediction.shape)

print("=" * 60)