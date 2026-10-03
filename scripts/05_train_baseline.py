from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

MODEL_DIR = (
    ROOT
    / "models"
)

RESULT_DIR = (
    ROOT
    / "results"
)

REPORT_DIR = (
    ROOT
    / "reports"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_PATH = (
    MODEL_DIR
    / "baseline_random_forest_phase2.joblib"
)

RESULT_PATH = (
    RESULT_DIR
    / "baseline_phase2_validation.json"
)

FEATURE_COLUMNS_PATH = (
    RESULT_DIR
    / "baseline_phase2_feature_columns.json"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 100_000
SEED = 42
CHUNK_SIZE = 500_000

N_ESTIMATORS = 250
MAX_DEPTH = 18
MIN_SAMPLES_LEAF = 2


# ============================================================
# FEATURES
# ============================================================
#
# IMPORTANT:
# These are deliberately aligned with the transaction inputs
# used by the GraphSAGE model.
#
# GraphSAGE:
# 16 numeric + 5 transaction-type one-hot = 21 features
#
# Random Forest:
# exactly the same 21 features
# ============================================================

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
# REPRODUCE EXACT GRAPH SAMPLE
# ============================================================

def sample_full_dataset():

    print()
    print("=" * 70)
    print("REPRODUCING EXACT GRAPH SAMPLE")
    print("=" * 70)

    total_rows = count_rows()

    print(
        f"Full dataset rows: {total_rows:,}"
    )

    print(
        f"Target sample: {SAMPLE_SIZE:,}"
    )

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

    # --------------------------------------------------------
    # EXACT ordering used by temporal graph builder
    # --------------------------------------------------------

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
        f"Selected sample: "
        f"{len(df):,}"
    )

    print(
        f"Fraud: "
        f"{int((df['isFraud'] == 1).sum()):,}"
    )

    print(
        f"Genuine: "
        f"{int((df['isFraud'] == 0).sum()):,}"
    )

    print(
        f"Step range: "
        f"{df['step'].min()} -> "
        f"{df['step'].max()}"
    )

    return df


# ============================================================
# TEMPORAL CUTOFFS
# ============================================================

def determine_cutoffs(df):

    print()
    print("=" * 70)
    print("DETERMINING SAME TEMPORAL SPLIT AS GRAPHSAGE")
    print("=" * 70)

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

    print(
        f"Train steps: "
        f"1 -> {train_end}"
    )

    print(
        f"Validation steps: "
        f"{train_end + 1} -> {val_end}"
    )

    print(
        f"Test steps: "
        f"{val_end + 1} -> {test_end}"
    )

    return (
        train_end,
        val_end,
        test_end,
    )


# ============================================================
# PREPARE SHARED TRANSACTION FEATURES
# ============================================================

def prepare_features(df):

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    X_numeric = (
        df[
            NUMERIC_FEATURES
        ]
        .copy()
    )

    X_numeric = (
        X_numeric
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # Fixed transaction-type one-hot features
    # --------------------------------------------------------

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

    values = (
        df["type"]
        .astype(str)
        .to_numpy()
    )

    for i, value in enumerate(values):

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

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    X = pd.concat(
        [
            X_numeric,
            type_df,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Ensure finite
    # --------------------------------------------------------

    matrix = X.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(
        matrix
    ).all():

        raise ValueError(
            "NaN/Inf detected in baseline features."
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

    y_true = np.asarray(
        y_true
    ).astype(int)

    probabilities = np.asarray(
        probabilities
    ).astype(float)

    predictions = (
        probabilities >= threshold
    ).astype(int)

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
    print("FinGraph - PHASE 2 FAIR RANDOM FOREST BASELINE")
    print("=" * 70)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    # ========================================================
    # SAMPLE
    # ========================================================

    df = sample_full_dataset()

    # ========================================================
    # CUT OFFS
    # ========================================================

    (
        train_end,
        val_end,
        test_end,
    ) = determine_cutoffs(
        df
    )

    # ========================================================
    # TEMPORAL SPLIT
    # ========================================================

    train_mask = (
        df["step"]
        <= train_end
    )

    val_mask = (
        (df["step"] > train_end)
        & (df["step"] <= val_end)
    )

    test_mask = (
        df["step"]
        > val_end
    )

    train_df = df[
        train_mask
    ].copy()

    val_df = df[
        val_mask
    ].copy()

    test_df = df[
        test_mask
    ].copy()

    print()
    print(
        "Split sizes:"
    )

    print(
        f"  Train:      {len(train_df):,}"
    )

    print(
        f"  Validation: {len(val_df):,}"
    )

    print(
        f"  Test:       {len(test_df):,}"
    )

    print()
    print(
        "Fraud counts:"
    )

    print(
        f"  Train:      "
        f"{int(train_df['isFraud'].sum()):,}"
    )

    print(
        f"  Validation: "
        f"{int(val_df['isFraud'].sum()):,}"
    )

    print(
        f"  Test:       "
        f"{int(test_df['isFraud'].sum()):,}"
    )

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    print()
    print(
        "Preparing shared 21-feature transaction matrix..."
    )

    X = prepare_features(
        df
    )

    y = (
        df["isFraud"]
        .astype(int)
    )

    X_train = X.loc[
        train_mask
    ]

    X_val = X.loc[
        val_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    y_val = y.loc[
        val_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    print(
        "Feature matrix shape:",
        X.shape
    )

    # ========================================================
    # TRAIN RANDOM FOREST
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    print(
        f"Estimators: {N_ESTIMATORS}"
    )

    print(
        f"Max depth: {MAX_DEPTH}"
    )

    print(
        f"Min samples leaf: {MIN_SAMPLES_LEAF}"
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    val_prob = (
        model
        .predict_proba(
            X_val
        )[:, 1]
    )

    val_metrics = calculate_metrics(
        y_val,
        val_prob,
        threshold=0.5,
    )

    for key, value in (
        val_metrics.items()
    ):

        print(
            f"{key:12s}: {value}"
        )

    # ========================================================
    # TEST IS NOT TOUCHED FOR MODEL SELECTION
    # ========================================================
    #
    # We deliberately do not calculate test metrics here.
    # scripts/06_evaluate_baseline.py will perform the final
    # test evaluation after the model is frozen.
    # ========================================================

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save feature names
    # --------------------------------------------------------

    with open(
        FEATURE_COLUMNS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            list(X.columns),
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    validation_predictions = (
        val_df[
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

    validation_predictions[
        "fraud_probability"
    ] = val_prob

    validation_predictions[
        "prediction"
    ] = (
        val_prob >= 0.5
    ).astype(int)

    validation_predictions.to_csv(
        REPORT_DIR
        / "baseline_phase2_validation_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance":
                model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        REPORT_DIR
        / "baseline_phase2_feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "model":
            "Random Forest",
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
        "train_rows":
            int(len(train_df)),
        "validation_rows":
            int(len(val_df)),
        "test_rows":
            int(len(test_df)),
        "train_fraud":
            int(y_train.sum()),
        "validation_fraud":
            int(y_val.sum()),
        "test_fraud":
            int(y_test.sum()),
        "feature_count":
            int(X.shape[1]),
        "feature_columns":
            list(X.columns),
        "parameters": {
            "n_estimators":
                N_ESTIMATORS,
            "max_depth":
                MAX_DEPTH,
            "min_samples_leaf":
                MIN_SAMPLES_LEAF,
            "class_weight":
                "balanced_subsample",
        },
        "validation_metrics":
            val_metrics,
    }

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print("PHASE 2 BASELINE TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Validation results:",
        RESULT_PATH
    )

    print(
        "Feature count:",
        X.shape[1]
    )

    print()
    print(
        "Test data has NOT been evaluated."
    )

    print(
        "Run 06_evaluate_baseline.py for final test evaluation."
    )


if __name__ == "__main__":
    main()