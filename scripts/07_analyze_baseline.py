from pathlib import Path
import json
import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "baseline_random_forest_phase2.joblib"
PRED_PATH = ROOT / "reports" / "baseline_phase2_test_predictions.csv"

FEATURE_NAMES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "is_merchant",
    "orig_balance_change",
    "dest_balance_change",
    "amount_to_orig_balance",
    "orig_transaction_count",
    "dest_transaction_count",
    "hour",
    "day",
    "is_night",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
]

print("=" * 70)
print("FinGraph - RANDOM FOREST BASELINE DIAGNOSTIC")
print("=" * 70)

# ---------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------

print("\nLoading model...")
model = joblib.load(MODEL_PATH)

print(f"Model loaded: {MODEL_PATH}")

# ---------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importances = model.feature_importances_

if len(importances) != len(FEATURE_NAMES):
    print(
        f"WARNING: model has {len(importances)} features, "
        f"but expected {len(FEATURE_NAMES)}"
    )
else:
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances
    }).sort_values(
        "importance",
        ascending=False
    )

    print(
        importance_df.to_string(
            index=False,
            formatters={
                "importance": "{:.6f}".format
            }
        )
    )

    output_path = ROOT / "results" / "baseline_feature_importance.json"

    output_path.write_text(
        json.dumps(
            importance_df.to_dict(orient="records"),
            indent=2
        ),
        encoding="utf-8"
    )

    print(f"\nSaved: {output_path}")

# ---------------------------------------------------------------
# LOAD PREDICTIONS
# ---------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST PREDICTION FILE")
print("=" * 70)

pred_df = pd.read_csv(PRED_PATH)

print(f"Rows: {len(pred_df)}")
print("\nColumns:")
print(list(pred_df.columns))

print("\nFirst 5 rows:")
print(pred_df.head().to_string(index=False))

# ---------------------------------------------------------------
# FIND PROBABILITY / LABEL COLUMNS
# ---------------------------------------------------------------

prob_candidates = [
    "fraud_probability",
    "probability",
    "y_prob",
    "prediction_probability",
    "predicted_probability"
]

label_candidates = [
    "isFraud",
    "label",
    "y_true",
    "target"
]

prob_col = next(
    (c for c in prob_candidates if c in pred_df.columns),
    None
)

label_col = next(
    (c for c in label_candidates if c in pred_df.columns),
    None
)

print("\nDetected probability column:", prob_col)
print("Detected label column:", label_col)

# ---------------------------------------------------------------
# PROBABILITY DISTRIBUTION
# ---------------------------------------------------------------

if prob_col is not None:

    print("\n" + "=" * 70)
    print("FRAUD PROBABILITY DISTRIBUTION")
    print("=" * 70)

    print(pred_df[prob_col].describe())

    if label_col is not None:

        print("\n" + "=" * 70)
        print("PROBABILITY BY ACTUAL CLASS")
        print("=" * 70)

        grouped = pred_df.groupby(label_col)[prob_col].agg([
            "count",
            "min",
            "mean",
            "median",
            "max"
        ])

        print(grouped)

# ---------------------------------------------------------------
# TOP FRAUD PREDICTIONS
# ---------------------------------------------------------------

if prob_col is not None:

    print("\n" + "=" * 70)
    print("TOP 20 HIGHEST FRAUD PROBABILITIES")
    print("=" * 70)

    cols = [
        c for c in [
            "transaction_id",
            "step",
            "isFraud",
            "label",
            "y_true",
            prob_col
        ]
        if c in pred_df.columns
    ]

    print(
        pred_df
        .sort_values(prob_col, ascending=False)
        [cols]
        .head(20)
        .to_string(index=False)
    )

# ---------------------------------------------------------------
# ACTUAL FRAUD ROWS
# ---------------------------------------------------------------

if prob_col is not None and label_col is not None:

    fraud_df = pred_df[pred_df[label_col] == 1]

    print("\n" + "=" * 70)
    print("ACTUAL FRAUD TRANSACTIONS")
    print("=" * 70)

    print(f"Fraud transactions: {len(fraud_df)}")

    print(
        fraud_df
        .sort_values(prob_col, ascending=False)
        [cols]
        .head(20)
        .to_string(index=False)
    )

# ---------------------------------------------------------------
# FINAL
# ---------------------------------------------------------------

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)