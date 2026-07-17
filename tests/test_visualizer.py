"""
test_visualizer.py
"""

from src.evaluation.image_evaluator import (
    ImageEvaluator,
)

from src.evaluation.visualizer import (
    Visualizer,
)

print("="*60)
print("Visualization Test")
print("="*60)

evaluator = ImageEvaluator()

evaluator.load_checkpoint()

sample = evaluator.export_ground_truth(0)

prediction = evaluator.render_prediction(0)

visualizer = Visualizer()

path = visualizer.save_comparison(
    prediction,
    sample["image"],
    0,
)

print()

print("Figure Generated")

print(path)

print()

print("="*60)