from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "paysim_engineered.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "baseline_random_forest_phase2.joblib"
)

TRAIN_RESULT_PATH = (
    ROOT
    / "results"
    / "baseline_phase2_validation.json"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "baseline_phase2_test.json"
)

REPORT_DIR = (
    ROOT
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SAMPLE_SIZE = 100_000
SEED = 42
CHUNK_SIZE = 500_000


NUMERIC_FEATURES = [
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
]

TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


# ============================================================
# COUNT ROWS
# ============================================================

def count_rows():

    count = 0

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.reader(f)

        next(reader)

        for _ in reader:
            count += 1

    return count


# ============================================================
# REPRODUCE SAME SAMPLE
# ============================================================

def sample_full_dataset():

    print()
    print("=" * 70)
    print("REPRODUCING SAME 100K SAMPLE")
    print("=" * 70)

    total_rows = count_rows()

    rng = np.random.default_rng(
        SEED
    )

    current_sample = None

    processed = 0

    for chunk in pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        chunk = chunk.copy()

        chunk["_random_key"] = (
            rng.random(
                len(chunk)
            )
        )

        if current_sample is None:

            current_sample = chunk

        else:

            current_sample = pd.concat(
                [
                    current_sample,
                    chunk,
                ],
                ignore_index=True,
            )

        if len(current_sample) > SAMPLE_SIZE:

            current_sample = (
                current_sample
                .nsmallest(
                    SAMPLE_SIZE,
                    "_random_key",
                )
                .reset_index(drop=True)
            )

        processed += len(chunk)

        print(
            f"\rScanned: "
            f"{processed:,}/{total_rows:,}",
            end="",
            flush=True,
        )

    print()

    df = (
        current_sample
        .drop(
            columns=["_random_key"]
        )
        .reset_index(drop=True)
    )

    df = (
        df
        .sort_values(
            [
                "step",
                "transaction_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    print(
        f"Selected: {len(df):,}"
    )

    return df


# ============================================================
# TEMPORAL CUTOFFS
# ============================================================

def determine_cutoffs(df):

    step_counts = (
        df.groupby("step")
        .size()
        .sort_index()
    )

    steps = (
        step_counts.index
        .to_numpy(
            dtype=np.int32
        )
    )

    counts = (
        step_counts.values
        .astype(np.int64)
    )

    cumulative = np.cumsum(
        counts
    )

    total = len(df)

    train_target = (
        0.70 * total
    )

    val_target = (
        0.85 * total
    )

    train_index = int(
        np.searchsorted(
            cumulative,
            train_target,
            side="left",
        )
    )

    val_index = int(
        np.searchsorted(
            cumulative,
            val_target,
            side="left",
        )
    )

    train_index = min(
        train_index,
        len(steps) - 2,
    )

    val_index = max(
        val_index,
        train_index + 1,
    )

    val_index = min(
        val_index,
        len(steps) - 1,
    )

    train_end = int(
        steps[train_index]
    )

    val_end = int(
        steps[val_index]
    )

    test_end = int(
        steps[-1]
    )

    return (
        train_end,
        val_end,
        test_end,
    )


# ============================================================
# FEATURES
# ============================================================

def prepare_features(df):

    X_numeric = (
        df[
            NUMERIC_FEATURES
        ]
        .copy()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    type_matrix = np.zeros(
        (
            len(df),
            len(TRANSACTION_TYPES),
        ),
        dtype=np.float32,
    )

    type_to_index = {
        value: index
        for index, value
        in enumerate(
            TRANSACTION_TYPES
        )
    }

    for i, value in enumerate(
        df["type"]
        .astype(str)
        .to_numpy()
    ):

        if value in type_to_index:

            type_matrix[
                i,
                type_to_index[value],
            ] = 1.0

    type_df = pd.DataFrame(
        type_matrix,
        columns=[
            f"type_{value}"
            for value in TRANSACTION_TYPES
        ],
        index=df.index,
    )

    X = pd.concat(
        [
            X_numeric,
            type_df,
        ],
        axis=1,
    )

    return X


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": None,
        "pr_auc": None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    if len(
        np.unique(y_true)
    ) == 2:

        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        result["pr_auc"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FinGraph - PHASE 2 RANDOM FOREST TEST EVALUATION")
    print("=" * 70)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n"
            "Run scripts/05_train_baseline.py first."
        )

    if not TRAIN_RESULT_PATH.exists():

        raise FileNotFoundError(
            f"Training metadata not found:\n"
            f"{TRAIN_RESULT_PATH}"
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print(
        "Loading model:"
    )

    print(
        MODEL_PATH
    )

    model = joblib.load(
        MODEL_PATH
    )

    # ========================================================
    # REPRODUCE SAMPLE
    # ========================================================

    df = sample_full_dataset()

    # ========================================================
    # CUTOFFS
    # ========================================================

    (
        train_end,
        val_end,
        test_end,
    ) = determine_cutoffs(
        df
    )

    print()
    print(
        "Temporal split:"
    )

    print(
        f"  Train: 1 -> {train_end}"
    )

    print(
        f"  Val:   {train_end + 1} -> {val_end}"
    )

    print(
        f"  Test:  {val_end + 1} -> {test_end}"
    )

    # ========================================================
    # TEST MASK
    # ========================================================

    test_mask = (
        df["step"]
        > val_end
    )

    test_df = df[
        test_mask
    ].copy()

    print()
    print(
        "Test transactions:",
        len(test_df)
    )

    print(
        "Test fraud:",
        int(
            test_df["isFraud"].sum()
        )
    )

    print(
        "Test genuine:",
        int(
            (
                test_df["isFraud"] == 0
            ).sum()
        )
    )

    # ========================================================
    # FEATURES
    # ========================================================

    X = prepare_features(
        df
    )

    X_test = X.loc[
        test_mask
    ]

    y_test = (
        test_df["isFraud"]
        .astype(int)
        .to_numpy()
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print()
    print(
        "Generating frozen test predictions..."
    )

    probabilities = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        y_test,
        probabilities,
        threshold=0.5,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("BASELINE TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy:       {metrics['accuracy']:.6f}"
    )

    print(
        f"Precision:      {metrics['precision']:.6f}"
    )

    print(
        f"Recall:         {metrics['recall']:.6f}"
    )

    print(
        f"F1:             {metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC:        {metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC:         {metrics['pr_auc']:.6f}"
    )

    print()
    print(
        "Confusion matrix @ threshold 0.5:"
    )

    print(
        f"  TN: {metrics['true_negative']:,}"
    )

    print(
        f"  FP: {metrics['false_positive']:,}"
    )

    print(
        f"  FN: {metrics['false_negative']:,}"
    )

    print(
        f"  TP: {metrics['true_positive']:,}"
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions = (
        test_df[
            [
                "transaction_id",
                "step",
                "nameOrig",
                "nameDest",
                "amount",
                "isFraud",
            ]
        ]
        .copy()
    )

    predictions[
        "fraud_probability"
    ] = probabilities

    predictions[
        "prediction"
    ] = (
        probabilities >= 0.5
    ).astype(int)

    predictions.to_csv(
        REPORT_DIR
        / "baseline_phase2_test_predictions.csv",
        index=False,
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "model":
            "FinGraph Random Forest",
        "phase":
            "Phase 2 fair baseline",
        "sample_size":
            SAMPLE_SIZE,
        "seed":
            SEED,
        "train_end_step":
            train_end,
        "validation_end_step":
            val_end,
        "test_end_step":
            test_end,
        "test_transactions":
            int(len(test_df)),
        "test_fraud":
            int(y_test.sum()),
        "test_genuine":
            int((y_test == 0).sum()),
        "feature_count":
            int(X.shape[1]),
        "metrics":
            metrics,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print(
        "Predictions saved:"
    )

    print(
        REPORT_DIR
        / "baseline_phase2_test_predictions.csv"
    )

    print()
    print(
        "Metrics saved:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("BASELINE TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()