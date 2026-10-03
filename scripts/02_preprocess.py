from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPECTED NUMERIC COLUMNS
# ============================================================

EXPECTED_NUMERIC = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


# ============================================================
# FIND INPUT FILE
# ============================================================

def find_input_file(explicit=None):
    """
    Find the PaySim CSV.
    """

    if explicit:

        path = Path(explicit)

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists():
            raise FileNotFoundError(path)

        return path

    candidates = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {RAW_DIR}. "
            "Put your PaySim CSV there."
        )

    return candidates[0]


# ============================================================
# HISTORICAL TRANSACTION COUNTS
# ============================================================

def add_temporal_transaction_counts(df):
    """
    Calculate transaction counts using ONLY information
    available BEFORE the current transaction's time step.

    Example:

        Account A:
        step 10 -> 3 transactions
        step 11 -> 2 transactions
        step 12 -> 4 transactions

    Every transaction at step 12 receives:

        historical_count = 5

    The transactions occurring at step 12 are NOT counted.

    This avoids temporal leakage from:
        - future steps
        - other transactions in the same step
    """

    print()
    print("=" * 60)
    print("CALCULATING TEMPORAL TRANSACTION COUNTS")
    print("=" * 60)

    # --------------------------------------------------------
    # IMPORTANT:
    # PaySim step represents time.
    #
    # Sort chronologically using a stable sort so that the
    # original order is preserved inside the same step.
    # --------------------------------------------------------

    df = df.sort_values(
        ["step", "_original_row"],
        kind="mergesort"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Sender historical count
    # --------------------------------------------------------

    # Number of previous transactions by sender
    cumulative_sender_count = (
        df.groupby(
            "nameOrig",
            sort=False
        ).cumcount()
    )

    # Number of previous transactions from the SAME
    # sender within the CURRENT step.
    #
    # Subtracting this leaves only transactions from
    # earlier time steps.
    same_step_sender_rank = (
        df.groupby(
            ["nameOrig", "step"],
            sort=False
        ).cumcount()
    )

    df["orig_transaction_count"] = (
        cumulative_sender_count
        - same_step_sender_rank
    ).astype(np.int32)

    # --------------------------------------------------------
    # Receiver historical count
    # --------------------------------------------------------

    cumulative_receiver_count = (
        df.groupby(
            "nameDest",
            sort=False
        ).cumcount()
    )

    same_step_receiver_rank = (
        df.groupby(
            ["nameDest", "step"],
            sort=False
        ).cumcount()
    )

    df["dest_transaction_count"] = (
        cumulative_receiver_count
        - same_step_receiver_rank
    ).astype(np.int32)

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    if (
        df["orig_transaction_count"] < 0
    ).any():

        raise ValueError(
            "Negative orig_transaction_count detected."
        )

    if (
        df["dest_transaction_count"] < 0
    ).any():

        raise ValueError(
            "Negative dest_transaction_count detected."
        )

    print(
        "Temporal sender counts calculated."
    )

    print(
        "Temporal receiver counts calculated."
    )

    print(
        "Maximum historical sender count:",
        int(
            df["orig_transaction_count"].max()
        )
    )

    print(
        "Maximum historical receiver count:",
        int(
            df["dest_transaction_count"].max()
        )
    )

    return df


# ============================================================
# CLEAN PAYSim
# ============================================================

def clean_paysim(df):
    """
    Clean PaySim and create leakage-safe engineered features.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = [
        c.strip()
        for c in df.columns
    ]

    # --------------------------------------------------------
    # Preserve original row order
    # --------------------------------------------------------

    df["_original_row"] = np.arange(
        len(df),
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Drop exact duplicates
    # --------------------------------------------------------

    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Removed duplicates: "
        f"{before - len(df)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
    ]

    missing_required = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing_required:

        raise ValueError(
            "Missing required PaySim columns: "
            + ", ".join(missing_required)
        )

    # ========================================================
    # NUMERIC CONVERSION
    # ========================================================

    print()
    print("=" * 60)
    print("VALIDATING NUMERIC COLUMNS")
    print("=" * 60)

    for col in EXPECTED_NUMERIC:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="raise"
            )

    # --------------------------------------------------------
    # Explicitly validate required numeric fields
    # --------------------------------------------------------

    if df[required].isna().any().any():

        missing_counts = (
            df[required]
            .isna()
            .sum()
        )

        print(
            "\nMissing values:"
        )

        print(
            missing_counts[
                missing_counts > 0
            ]
        )

        raise ValueError(
            "Required columns contain missing values."
        )

    print(
        "Numeric validation successful."
    )

    # ========================================================
    # BASIC VALIDITY CHECKS
    # ========================================================

    print()
    print("=" * 60)
    print("BASIC VALIDITY CHECKS")
    print("=" * 60)

    # Negative amounts
    if (
        df["amount"] < 0
    ).any():

        raise ValueError(
            "Negative transaction amounts found."
        )

    # --------------------------------------------------------
    # Strict fraud-label validation
    # --------------------------------------------------------

    fraud_values = (
        pd.to_numeric(
            df["isFraud"],
            errors="coerce"
        )
    )

    if fraud_values.isna().any():

        raise ValueError(
            "isFraud contains null/non-numeric values."
        )

    if not fraud_values.isin([0, 1]).all():

        invalid_values = (
            fraud_values[
                ~fraud_values.isin([0, 1])
            ]
            .unique()
        )

        raise ValueError(
            "isFraud contains values other than 0/1: "
            f"{invalid_values}"
        )

    df["isFraud"] = (
        fraud_values
        .astype(np.int8)
    )

    print(
        "isFraud validation successful."
    )

    # ========================================================
    # TYPE / ACCOUNT IDS
    # ========================================================

    df["type"] = (
        df["type"]
        .astype(str)
        .str.strip()
    )

    df["nameOrig"] = (
        df["nameOrig"]
        .astype(str)
        .str.strip()
    )

    df["nameDest"] = (
        df["nameDest"]
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # MERCHANT INDICATOR
    # ========================================================

    df["is_merchant"] = (
        df["nameDest"]
        .astype(str)
        .str.startswith("M")
        .astype(np.int8)
    )

    # ========================================================
    # ROW-LOCAL ENGINEERED FEATURES
    # ========================================================

    print()
    print(
        "Creating row-local engineered features..."
    )

    # --------------------------------------------------------
    # Origin balance change
    # --------------------------------------------------------

    df["orig_balance_change"] = (
        df["oldbalanceOrg"]
        - df["newbalanceOrig"]
    )

    # --------------------------------------------------------
    # Destination balance change
    # --------------------------------------------------------

    df["dest_balance_change"] = (
        df["newbalanceDest"]
        - df["oldbalanceDest"]
    )

    # --------------------------------------------------------
    # Amount relative to sender's old balance
    # --------------------------------------------------------

    df["amount_to_orig_balance"] = (
        df["amount"]
        / (
            df["oldbalanceOrg"]
            + 1.0
        )
    )

    # Prevent infinite values
    df["amount_to_orig_balance"] = (
        df["amount_to_orig_balance"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .astype(np.float32)
    )

    # ========================================================
    # TEMPORAL TRANSACTION COUNTS
    # ========================================================

    df = add_temporal_transaction_counts(
        df
    )

    # ========================================================
    # TIME FEATURES
    # ========================================================

    print()
    print(
        "Creating temporal features..."
    )

    # PaySim step is hourly
    df["step"] = (
        df["step"]
        .astype(np.int32)
    )

    df["hour"] = (
        df["step"] % 24
    ).astype(np.int16)

    df["day"] = (
        df["step"] // 24
    ).astype(np.int32)

    df["is_night"] = (
        df["hour"]
        .between(0, 5)
        .astype(np.int8)
    )

    # ========================================================
    # SORT FINAL DATA CHRONOLOGICALLY
    # ========================================================

    df = df.sort_values(
        ["step", "_original_row"],
        kind="mergesort"
    ).reset_index(drop=True)

    # ========================================================
    # CREATE OUR OWN TRANSACTION ID
    # ========================================================

    print()
    print(
        "Creating FinGraph transaction IDs..."
    )

    df.insert(
        0,
        "transaction_id",
        [
            f"TXN{i:09d}"
            for i in range(
                1,
                len(df) + 1
            )
        ]
    )

    # ========================================================
    # REMOVE INTERNAL HELPER COLUMN
    # ========================================================

    df = df.drop(
        columns=["_original_row"]
    )

    # ========================================================
    # FINAL NUMERIC CLEANUP
    # ========================================================

    integer_columns = [
        "step",
        "isFraud",
        "isFlaggedFraud",
        "is_merchant",
        "orig_transaction_count",
        "dest_transaction_count",
        "hour",
        "day",
        "is_night",
    ]

    for col in integer_columns:

        if col in df.columns:

            if col in [
                "isFraud",
                "isFlaggedFraud",
                "is_merchant",
                "is_night",
            ]:

                df[col] = (
                    df[col]
                    .astype(np.int8)
                )

            elif col in [
                "orig_transaction_count",
                "dest_transaction_count",
                "hour",
            ]:

                df[col] = (
                    df[col]
                    .astype(np.int32)
                )

            else:

                df[col] = (
                    df[col]
                    .astype(np.int32)
                )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL DATASET VALIDATION")
    print("=" * 60)

    if df["transaction_id"].duplicated().any():

        raise ValueError(
            "Duplicate transaction_id values detected."
        )

    if not df["isFraud"].isin([0, 1]).all():

        raise ValueError(
            "Final isFraud validation failed."
        )

    if (
        df["orig_transaction_count"] < 0
    ).any():

        raise ValueError(
            "Negative historical sender counts."
        )

    if (
        df["dest_transaction_count"] < 0
    ).any():

        raise ValueError(
            "Negative historical receiver counts."
        )

    # Check numerical features
    numeric_check_columns = [
        col
        for col in [
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "orig_balance_change",
            "dest_balance_change",
            "amount_to_orig_balance",
            "orig_transaction_count",
            "dest_transaction_count",
            "hour",
            "day",
            "is_night",
        ]
        if col in df.columns
    ]

    numeric_values = (
        df[numeric_check_columns]
        .to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        numeric_values
    ).all():

        raise ValueError(
            "NaN or infinite values detected "
            "in engineered numeric features."
        )

    print(
        "Final validation successful."
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess PaySim for FinGraph "
            "with temporal-safe features."
        )
    )

    parser.add_argument(
        "--input",
        default=None
    )

    parser.add_argument(
        "--output",
        default=(
            "data/processed/"
            "paysim_engineered.csv"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Find input
    # --------------------------------------------------------

    input_path = find_input_file(
        args.input
    )

    output_path = (
        ROOT / args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("FinGraph PaySim Preprocessing")
    print("=" * 60)

    print()
    print(
        "Reading:",
        input_path
    )

    # --------------------------------------------------------
    # Read raw PaySim
    # --------------------------------------------------------

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    print(
        "Raw shape:",
        df.shape
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    clean = clean_paysim(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    clean.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print()
    print(
        "Processed shape:",
        clean.shape
    )

    print(
        "Fraud count:",
        int(
            clean["isFraud"].sum()
        )
    )

    print(
        "Genuine count:",
        int(
            (
                clean["isFraud"] == 0
            ).sum()
        )
    )

    print(
        "Unique accounts:",
        len(
            set(
                clean["nameOrig"]
            )
            |
            set(
                clean["nameDest"]
            )
        )
    )

    print(
        "Step range:",
        int(clean["step"].min()),
        "->",
        int(clean["step"].max())
    )

    print(
        "Unique steps:",
        clean["step"].nunique()
    )

    print()
    print(
        "Saved:",
        output_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()