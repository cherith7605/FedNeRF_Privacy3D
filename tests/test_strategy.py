"""
test_strategy.py

Tests the Flower strategy.
"""

from src.federated.strategy import get_strategy

print("=" * 60)
print("Flower Strategy Test")
print("=" * 60)

strategy = get_strategy()

print()

print("Strategy Created Successfully")

print()

print(type(strategy).__name__)

print()

print(strategy)

print("=" * 60)