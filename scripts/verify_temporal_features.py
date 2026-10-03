from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "paysim_engineered.csv"
)


# ============================================================
# FUNCTION:
# CALCULATE HISTORICAL COUNTS INDEPENDENTLY
# ============================================================

def calculate_historical_counts(
    df,
    account_column,
    output_column_name,
):
    """
    Independently calculate the number of transactions
    belonging to each account from STRICTLY EARLIER
    time steps.

    Transactions occurring in the current step are NOT
    included.
    """

    print(
        f"Calculating historical counts for "
        f"{account_column}..."
    )

    # --------------------------------------------------------
    # Count transactions for every (account, step)
    # --------------------------------------------------------

    step_counts = (
        df.groupby(
            [account_column, "step"],
            sort=True,
            observed=True,
        )
        .size()
        .reset_index(name="_transactions_in_step")
    )

    # --------------------------------------------------------
    # For each account, cumulative count through each step
    # --------------------------------------------------------

    step_counts["_cumulative_count"] = (
        step_counts
        .groupby(
            account_column,
            sort=False,
            observed=True,
        )["_transactions_in_step"]
        .cumsum()
    )

    # --------------------------------------------------------
    # Remove current step transactions
    #
    # cumulative through current step
    # MINUS
    # transactions in current step
    #
    # = transactions from earlier steps
    # --------------------------------------------------------

    step_counts[output_column_name] = (
        step_counts["_cumulative_count"]
        - step_counts["_transactions_in_step"]
    ).astype(np.int64)

    # --------------------------------------------------------
    # Keep only mapping required for merge
    # --------------------------------------------------------

    result = step_counts[
        [
            account_column,
            "step",
            output_column_name,
        ]
    ]

    # --------------------------------------------------------
    # Merge reference count back to every transaction
    # --------------------------------------------------------

    df = df.merge(
        result,
        on=[
            account_column,
            "step",
        ],
        how="left",
        sort=False,
        validate="many_to_one",
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FinGraph TEMPORAL FEATURE VERIFICATION")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_PATH}"
        )

    print("\nLoading:")
    print(DATA_PATH)

    columns = [
        "transaction_id",
        "step",
        "nameOrig",
        "nameDest",
        "orig_transaction_count",
        "dest_transaction_count",
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
        low_memory=False,
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    print()
    print("=" * 60)
    print("BASIC DATA CHECK")
    print("=" * 60)

    required_columns = [
        "transaction_id",
        "step",
        "nameOrig",
        "nameDest",
        "orig_transaction_count",
        "dest_transaction_count",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df["transaction_id"].isna().any():
        raise ValueError(
            "transaction_id contains null values."
        )

    if df["step"].isna().any():
        raise ValueError(
            "step contains null values."
        )

    if df["nameOrig"].isna().any():
        raise ValueError(
            "nameOrig contains null values."
        )

    if df["nameDest"].isna().any():
        raise ValueError(
            "nameDest contains null values."
        )

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    print()
    print(
        "Sorting transactions chronologically..."
    )

    df = df.sort_values(
        [
            "step",
            "transaction_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ========================================================
    # INDEPENDENT SENDER CALCULATION
    # ========================================================

    print()
    print("=" * 60)
    print("VERIFYING SENDER COUNTS")
    print("=" * 60)

    df = calculate_historical_counts(
        df=df,
        account_column="nameOrig",
        output_column_name="_expected_orig_count",
    )

    # ========================================================
    # INDEPENDENT RECEIVER CALCULATION
    # ========================================================

    print()
    print("=" * 60)
    print("VERIFYING RECEIVER COUNTS")
    print("=" * 60)

    df = calculate_historical_counts(
        df=df,
        account_column="nameDest",
        output_column_name="_expected_dest_count",
    )

    # ========================================================
    # COMPARE SENDER COUNTS
    # ========================================================

    print()
    print("=" * 60)
    print("COMPARING SENDER COUNTS")
    print("=" * 60)

    actual_sender = (
        df["orig_transaction_count"]
        .astype(np.int64)
    )

    expected_sender = (
        df["_expected_orig_count"]
        .astype(np.int64)
    )

    sender_match = (
        actual_sender
        == expected_sender
    )

    sender_mismatches = int(
        (~sender_match).sum()
    )

    print(
        "Sender temporal-count mismatches:",
        f"{sender_mismatches:,}"
    )

    if sender_mismatches > 0:

        print(
            "\nSender mismatch examples:"
        )

        mismatch_rows = df.loc[
            ~sender_match,
            [
                "transaction_id",
                "step",
                "nameOrig",
                "orig_transaction_count",
                "_expected_orig_count",
            ],
        ]

        print(
            mismatch_rows.head(10).to_string(
                index=False
            )
        )

    # ========================================================
    # COMPARE RECEIVER COUNTS
    # ========================================================

    print()
    print("=" * 60)
    print("COMPARING RECEIVER COUNTS")
    print("=" * 60)

    actual_receiver = (
        df["dest_transaction_count"]
        .astype(np.int64)
    )

    expected_receiver = (
        df["_expected_dest_count"]
        .astype(np.int64)
    )

    receiver_match = (
        actual_receiver
        == expected_receiver
    )

    receiver_mismatches = int(
        (~receiver_match).sum()
    )

    print(
        "Receiver temporal-count mismatches:",
        f"{receiver_mismatches:,}"
    )

    if receiver_mismatches > 0:

        print(
            "\nReceiver mismatch examples:"
        )

        mismatch_rows = df.loc[
            ~receiver_match,
            [
                "transaction_id",
                "step",
                "nameDest",
                "dest_transaction_count",
                "_expected_dest_count",
            ],
        ]

        print(
            mismatch_rows.head(10).to_string(
                index=False
            )
        )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    print()
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    actual_orig = (
        df["orig_transaction_count"]
        .astype(np.int64)
    )

    actual_dest = (
        df["dest_transaction_count"]
        .astype(np.int64)
    )

    print(
        "Minimum sender historical count:",
        int(actual_orig.min())
    )

    print(
        "Maximum sender historical count:",
        int(actual_orig.max())
    )

    print(
        "Minimum receiver historical count:",
        int(actual_dest.min())
    )

    print(
        "Maximum receiver historical count:",
        int(actual_dest.max())
    )

    if (actual_orig < 0).any():
        raise ValueError(
            "Negative sender historical counts detected."
        )

    if (actual_dest < 0).any():
        raise ValueError(
            "Negative receiver historical counts detected."
        )

    # ========================================================
    # MANUAL TEMPORAL EXAMPLE
    # ========================================================

    print()
    print("=" * 60)
    print("TEMPORAL LOGIC EXAMPLE")
    print("=" * 60)

    # Find an account appearing at multiple steps
    sender_counts_per_account = (
        df.groupby("nameOrig")["step"]
        .nunique()
    )

    repeated_accounts = (
        sender_counts_per_account[
            sender_counts_per_account > 1
        ]
        .sort_values(ascending=False)
    )

    if len(repeated_accounts) > 0:

        example_account = (
            repeated_accounts.index[0]
        )

        example = df[
            df["nameOrig"]
            == example_account
        ][
            [
                "transaction_id",
                "step",
                "nameOrig",
                "orig_transaction_count",
            ]
        ].head(10)

        print(
            f"\nExample sender account: "
            f"{example_account}"
        )

        print(
            example.to_string(
                index=False
            )
        )

        print(
            "\nThe count should increase only when "
            "an earlier time step has occurred."
        )

    else:

        print(
            "No repeated sender account found "
            "for manual example."
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 60)

    if (
        sender_mismatches == 0
        and receiver_mismatches == 0
    ):

        print(
            "TEMPORAL FEATURE VERIFICATION PASSED"
        )

        print()
        print(
            "Both transaction-count features use "
            "only strictly earlier time steps."
        )

    else:

        print(
            "TEMPORAL FEATURE VERIFICATION FAILED"
        )

        raise ValueError(
            "Temporal feature mismatches detected."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()