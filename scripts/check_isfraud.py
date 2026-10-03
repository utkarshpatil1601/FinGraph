from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "processed" / "paysim_engineered.csv"

value_counts = {}
null_count = 0
total = 0
bad_examples = []

for chunk in pd.read_csv(
    PATH,
    usecols=["isFraud"],
    chunksize=500_000,
):

    total += len(chunk)

    null_count += int(
        chunk["isFraud"].isna().sum()
    )

    counts = chunk["isFraud"].value_counts(
        dropna=False
    )

    for value, count in counts.items():
        value_counts[value] = (
            value_counts.get(value, 0) + int(count)
        )

    invalid = chunk[
        ~chunk["isFraud"].isin([0, 1])
    ]

    if not invalid.empty and len(bad_examples) < 20:
        bad_examples.extend(
            invalid["isFraud"]
            .tolist()
        )

print("=" * 60)
print("isFraud CHECK")
print("=" * 60)

print("Total rows:", total)
print("Null values:", null_count)

print("\nUnique values:")
for value, count in value_counts.items():
    print(repr(value), "->", count)

print("\nInvalid examples:")
print(bad_examples[:20])