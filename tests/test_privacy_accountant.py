"""
test_privacy_accountant.py
"""

from src.privacy.privacy_accountant import (
    PrivacyAccountant,
)

print("=" * 60)
print("Privacy Accountant Test")
print("=" * 60)

accountant = PrivacyAccountant()

report = accountant.generate_report(
    steps=1000,
)

print()

print("Privacy Report Generated")

print(report)

print()

print("=" * 60)