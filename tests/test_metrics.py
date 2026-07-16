"""
test_metrics.py
"""

from src.evaluation.image_evaluator import ImageEvaluator

from src.evaluation.metrics import (
    evaluate_images,
)

print("=" * 60)
print("Image Quality Metrics Test")
print("=" * 60)

evaluator = ImageEvaluator()

evaluator.load_checkpoint()

sample = evaluator.export_ground_truth(0)

prediction = evaluator.render_prediction(0)

metrics = evaluate_images(
    prediction,
    sample["image"],
)

print()

for key, value in metrics.items():

    if isinstance(value, float):

        print(
            f"{key.upper():<8}: {value:.4f}"
        )

print()

print("=" * 60)