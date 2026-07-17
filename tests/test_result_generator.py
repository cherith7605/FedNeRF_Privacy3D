"""
test_result_generator.py
"""

from src.evaluation.result_generator import ResultGenerator

print("=" * 60)
print("Research Result Generator Test")
print("=" * 60)

generator = ResultGenerator()

csv_path = generator.evaluate_dataset()

print()

print("CSV Generated Successfully")

print(csv_path)

print()

print("=" * 60)