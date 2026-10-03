from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def find_input_file(explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    candidates = sorted(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {RAW_DIR}. Put your PaySim CSV there."
        )
    if len(candidates) > 1:
        print("Multiple CSV files found. Using:", candidates[0])
    return candidates[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    args = parser.parse_args()

    input_path = find_input_file(args.input)
    df = pd.read_csv(input_path)

    print("=" * 70)
    print("FinGraph Phase 1 — EDA")
    print("=" * 70)
    print("File:", input_path)
    print("Shape:", df.shape)
    print("\nColumns:")
    for c in df.columns:
        print(" -", c)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))

    print("\nDuplicate rows:", int(df.duplicated().sum()))

    if "isFraud" in df.columns:
        print("\nFraud distribution:")
        print(df["isFraud"].value_counts(dropna=False))
        print("\nFraud percentage:")
        print((df["isFraud"].value_counts(normalize=True) * 100).round(4))

    if "type" in df.columns:
        print("\nTransaction types:")
        print(df["type"].value_counts())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print("\nNumeric summary:")
        print(df[numeric_cols].describe().T)

    if "isFraud" in df.columns and "type" in df.columns:
        ctab = pd.crosstab(df["type"], df["isFraud"], normalize="index") * 100
        ctab.to_csv(REPORT_DIR / "fraud_rate_by_transaction_type.csv")

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="type",
            y="isFraud",
            estimator="mean",
            errorbar=None,
        )
        plt.ylabel("Fraud rate")
        plt.xlabel("Transaction type")
        plt.title("Fraud Rate by Transaction Type")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "fraud_rate_by_type.png", dpi=160)
        plt.close()

    if "isFraud" in df.columns:
        plt.figure(figsize=(6, 5))
        sns.countplot(data=df, x="isFraud")
        plt.xlabel("isFraud (0 = Genuine, 1 = Fraud)")
        plt.ylabel("Number of transactions")
        plt.title("Fraud Class Distribution")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "fraud_class_distribution.png", dpi=160)
        plt.close()

    print("\nEDA completed.")
    print("Reports saved to:", REPORT_DIR)

if __name__ == "__main__":
    main()
