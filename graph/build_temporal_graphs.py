"""
FinGraph - Strict Temporal Heterogeneous Graph Builder

Creates three cumulative temporal graph snapshots:

TRAIN:
    Transactions through the training cutoff.

VALIDATION:
    Transactions through the validation cutoff.
    Aggregate/network node features use HISTORY ONLY
    through the training cutoff.

TEST:
    Transactions through the test cutoff.
    Aggregate/network node features use HISTORY ONLY
    through the validation cutoff.

Important:
- Sampling is uniform across the FULL PaySim dataset.
- Same sample is reproducible with seed=42.
- Transaction-count features are already temporal-safe.
- Account aggregate features are history-only.
- Device/IP/Merchant/Session degree features are history-only.
- CONNECTED_TO is built from historical transactions only.
- Synthetic relationships are retained.
- Transaction-driven edges carry edge_time.
- Account fraud labels are NOT stored in the graph.
- Original fingraph_graph.pt is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PAYSIM_PATH = (
    ROOT
    / "data"
    / "processed"
    / "paysim_engineered.csv"
)

RELATIONSHIP_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "account_relationships.csv"
)

GRAPH_DIR = (
    ROOT
    / "data"
    / "graphs"
)

TRAIN_GRAPH_PATH = (
    GRAPH_DIR
    / "fingraph_train.pt"
)

VAL_GRAPH_PATH = (
    GRAPH_DIR
    / "fingraph_val.pt"
)

TEST_GRAPH_PATH = (
    GRAPH_DIR
    / "fingraph_test.pt"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build temporal FinGraph HeteroData snapshots."
        )
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help=(
            "Number of transactions sampled from the "
            "full PaySim dataset."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="CSV chunk size.",
    )

    return parser.parse_args()


# ============================================================
# COUNT FULL DATASET
# ============================================================

def count_rows():
    print()
    print(
        "Counting rows in full PaySim dataset..."
    )

    count = 0

    with open(
        PAYSIM_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.reader(f)

        next(reader)

        for _ in reader:
            count += 1

    print(
        f"Total rows: {count:,}"
    )

    return count


# ============================================================
# UNIFORM SAMPLE
# ============================================================

def uniform_sample(
    sample_size: int,
    seed: int,
    chunk_size: int,
):
    print()
    print("=" * 60)
    print("SAMPLING FULL DATASET")
    print("=" * 60)

    total_rows = count_rows()

    if sample_size >= total_rows:

        print(
            "Requested sample size is larger than or "
            "equal to full dataset."
        )

        print(
            "Loading complete dataset."
        )

        return pd.read_csv(
            PAYSIM_PATH,
            low_memory=False,
        )

    rng = np.random.default_rng(seed)

    current_sample = None

    processed = 0

    for chunk in pd.read_csv(
        PAYSIM_PATH,
        chunksize=chunk_size,
        low_memory=False,
    ):

        chunk = chunk.copy()

        chunk["_random_key"] = (
            rng.random(len(chunk))
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

        if len(current_sample) > sample_size:

            current_sample = (
                current_sample
                .nsmallest(
                    sample_size,
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

    current_sample = (
        current_sample
        .drop(
            columns=["_random_key"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Selected sample: "
        f"{len(current_sample):,}"
    )

    return current_sample


# ============================================================
# VALIDATE SAMPLE
# ============================================================

def validate_sample(df: pd.DataFrame):

    print()
    print("=" * 60)
    print("VALIDATING SAMPLE")
    print("=" * 60)

    required_columns = [
        "transaction_id",
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "isFraud",
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
            "Missing columns:\n"
            f"{missing}"
        )

    # --------------------------------------------------------
    # FRAUD LABEL
    # --------------------------------------------------------

    fraud = pd.to_numeric(
        df["isFraud"],
        errors="coerce",
    )

    if fraud.isna().any():

        raise ValueError(
            "Sample contains null/non-numeric "
            "isFraud values."
        )

    if not fraud.isin([0, 1]).all():

        invalid = (
            fraud[
                ~fraud.isin([0, 1])
            ]
            .unique()
        )

        raise ValueError(
            "Invalid isFraud values:\n"
            f"{invalid}"
        )

    df["isFraud"] = (
        fraud.astype(np.int8)
    )

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    df["step"] = pd.to_numeric(
        df["step"],
        errors="raise",
    ).astype(np.int32)

    # --------------------------------------------------------
    # IDS
    # --------------------------------------------------------

    for col in [
        "transaction_id",
        "type",
        "nameOrig",
        "nameDest",
    ]:

        df[col] = (
            df[col]
            .astype(str)
        )

    # --------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------

    numeric_columns = [
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

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="raise",
            )

    # --------------------------------------------------------
    # NUMERICAL SANITY
    # --------------------------------------------------------

    for col in numeric_columns:

        if col not in df.columns:
            continue

        values = df[col].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():

            raise ValueError(
                f"NaN/Inf detected in {col}."
            )

    print(
        f"Sample rows: {len(df):,}"
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

    print(
        f"Unique steps: "
        f"{df['step'].nunique()}"
    )

    return df


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def determine_cutoffs(df: pd.DataFrame):

    print()
    print("=" * 60)
    print("DETERMINING TEMPORAL CUTOFFS")
    print("=" * 60)

    step_counts = (
        df.groupby("step")
        .size()
        .sort_index()
    )

    steps = (
        step_counts.index
        .to_numpy(dtype=np.int32)
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

    train_df = df[
        df["step"] <= train_end
    ].copy()

    val_period_df = df[
        (df["step"] > train_end)
        & (df["step"] <= val_end)
    ].copy()

    test_period_df = df[
        df["step"] > val_end
    ].copy()

    print(
        f"Train steps: "
        f"{steps[0]} -> {train_end}"
    )

    print(
        f"Validation steps: "
        f"{train_end + 1} -> {val_end}"
    )

    print(
        f"Test steps: "
        f"{val_end + 1} -> {test_end}"
    )

    print()

    print(
        f"Train transactions: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation transactions: "
        f"{len(val_period_df):,}"
    )

    print(
        f"Test transactions: "
        f"{len(test_period_df):,}"
    )

    print()

    print(
        "Validation period fraud:",
        int(
            (
                val_period_df["isFraud"] == 1
            ).sum()
        )
    )

    print(
        "Test period fraud:",
        int(
            (
                test_period_df["isFraud"] == 1
            ).sum()
        )
    )

    return (
        train_end,
        val_end,
        test_end,
    )


# ============================================================
# FIND RELATIONSHIP COLUMN
# ============================================================

def find_column(
    columns,
    candidates,
):
    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


# ============================================================
# LOAD SYNTHETIC RELATIONSHIPS
# ============================================================

def load_relationships(
    account_ids,
    chunk_size,
):

    print()
    print("=" * 60)
    print("LOADING SYNTHETIC RELATIONSHIPS")
    print("=" * 60)

    if not RELATIONSHIP_PATH.exists():

        raise FileNotFoundError(
            RELATIONSHIP_PATH
        )

    account_ids = set(
        str(x)
        for x in account_ids
    )

    header = pd.read_csv(
        RELATIONSHIP_PATH,
        nrows=0,
    )

    columns = list(
        header.columns
    )

    account_col = find_column(
        columns,
        [
            "account_id",
            "account",
            "name",
        ],
    )

    device_col = find_column(
        columns,
        [
            "device_id",
            "device",
        ],
    )

    ip_col = find_column(
        columns,
        [
            "ip_address",
            "ip_id",
            "ip",
        ],
    )

    merchant_col = find_column(
        columns,
        [
            "merchant_id",
            "merchant",
        ],
    )

    session_col = find_column(
        columns,
        [
            "session_id",
            "session",
        ],
    )

    required = {
        "account_id": account_col,
        "device_id": device_col,
        "ip_address": ip_col,
        "merchant_id": merchant_col,
        "session_id": session_col,
    }

    missing = [
        name
        for name, column in required.items()
        if column is None
    ]

    if missing:

        raise ValueError(
            "Missing relationship columns: "
            f"{missing}\n"
            f"Available: {columns}"
        )

    usecols = [
        account_col,
        device_col,
        ip_col,
        merchant_col,
        session_col,
    ]

    pieces = []

    processed = 0

    for chunk in pd.read_csv(
        RELATIONSHIP_PATH,
        usecols=usecols,
        chunksize=chunk_size,
        dtype=str,
    ):

        mask = (
            chunk[account_col]
            .isin(account_ids)
        )

        selected = chunk.loc[
            mask
        ]

        if not selected.empty:
            pieces.append(
                selected
            )

        processed += len(chunk)

        print(
            f"\rRelationship rows scanned: "
            f"{processed:,}",
            end="",
            flush=True,
        )

    print()

    if not pieces:

        raise ValueError(
            "No required accounts were found "
            "in relationship file."
        )

    relations = pd.concat(
        pieces,
        ignore_index=True,
    )

    relations = relations.rename(
        columns={
            account_col: "account_id",
            device_col: "device_id",
            ip_col: "ip_address",
            merchant_col: "merchant_id",
            session_col: "session_id",
        }
    )

    relations = (
        relations
        .drop_duplicates(
            subset=["account_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"Loaded relationships: "
        f"{len(relations):,}"
    )

    return relations


# ============================================================
# CREATE DETERMINISTIC MAPPING
# ============================================================

def create_mapping(values):

    values = (
        pd.Series(values)
        .astype(str)
        .unique()
    )

    return {
        value: i
        for i, value in enumerate(values)
    }


# ============================================================
# BUILD TRANSACTION FEATURES
# ============================================================

def build_transaction_features(
    df: pd.DataFrame
):

    numeric_features = [
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

    available = [
        col
        for col in numeric_features
        if col in df.columns
    ]

    numeric_df = (
        df[available]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    numeric_matrix = (
        numeric_df
        .to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # TRANSACTION TYPE ONE-HOT
    # --------------------------------------------------------

    transaction_types = [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]

    type_to_index = {
        value: i
        for i, value
        in enumerate(transaction_types)
    }

    type_matrix = np.zeros(
        (
            len(df),
            len(transaction_types),
        ),
        dtype=np.float32,
    )

    for row_index, value in enumerate(
        df["type"].astype(str)
    ):

        if value in type_to_index:

            type_matrix[
                row_index,
                type_to_index[value],
            ] = 1.0

    matrix = np.concatenate(
        [
            numeric_matrix,
            type_matrix,
        ],
        axis=1,
    )

    return torch.tensor(
        matrix,
        dtype=torch.float32,
    )


# ============================================================
# ENTITY DEGREE FEATURE
# ============================================================

def degree_feature(
    number_nodes: int,
    destination_index: np.ndarray,
):

    if len(destination_index) == 0:

        degrees = np.zeros(
            number_nodes,
            dtype=np.float32,
        )

    else:

        degrees = np.bincount(
            destination_index,
            minlength=number_nodes,
        ).astype(
            np.float32
        )

    degrees = np.log1p(
        degrees
    )

    return torch.tensor(
        degrees.reshape(-1, 1),
        dtype=torch.float32,
    )


# ============================================================
# BUILD ONE SNAPSHOT
# ============================================================

def build_snapshot(
    snapshot_df: pd.DataFrame,
    history_end_step: int,
    target_start_step: int,
    target_end_step: int,
    snapshot_name: str,
    relationship_df: pd.DataFrame,
    seed: int,
):

    print()
    print("=" * 60)
    print(
        f"BUILDING {snapshot_name.upper()} SNAPSHOT"
    )
    print("=" * 60)

    # ========================================================
    # SORT SNAPSHOT
    # ========================================================

    df = (
        snapshot_df
        .sort_values(
            [
                "step",
                "transaction_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    # ========================================================
    # HISTORY MASK
    #
    # This is TRANSACTION-level.
    # ========================================================

    history_mask = (
        df["step"] <= history_end_step
    )

    history_transaction_mask = (
        history_mask.to_numpy(
            dtype=bool
        )
    )

    history_df = df.loc[
        history_mask
    ].copy()

    print()
    print(
        f"History cutoff: "
        f"step <= {history_end_step}"
    )

    print(
        f"History transactions: "
        f"{len(history_df):,}"
    )

    print(
        f"Snapshot transactions: "
        f"{len(df):,}"
    )

    # ========================================================
    # BASIC VALUES
    # ========================================================

    n_transactions = len(df)

    transaction_index = np.arange(
        n_transactions,
        dtype=np.int64,
    )

    sender_ids = (
        df["nameOrig"]
        .astype(str)
        .to_numpy()
    )

    receiver_ids = (
        df["nameDest"]
        .astype(str)
        .to_numpy()
    )

    # ========================================================
    # ACCOUNT NODES
    # ========================================================

    all_accounts = np.concatenate(
        [
            sender_ids,
            receiver_ids,
        ]
    )

    account_to_idx = create_mapping(
        all_accounts
    )

    n_accounts = len(
        account_to_idx
    )

    sender_idx = np.array(
        [
            account_to_idx[x]
            for x in sender_ids
        ],
        dtype=np.int64,
    )

    receiver_idx = np.array(
        [
            account_to_idx[x]
            for x in receiver_ids
        ],
        dtype=np.int64,
    )

    account_ids = list(
        account_to_idx.keys()
    )

    # ========================================================
    # SYNTHETIC RELATIONSHIP LOOKUP
    # ========================================================

    relation_lookup = (
        relationship_df
        .set_index("account_id")
    )

    device_ids = []
    ip_ids = []
    merchant_ids = []
    session_ids = []

    for account in account_ids:

        if account in relation_lookup.index:

            row = relation_lookup.loc[
                account
            ]

            device_ids.append(
                str(row["device_id"])
            )

            ip_ids.append(
                str(row["ip_address"])
            )

            merchant_ids.append(
                str(row["merchant_id"])
            )

            session_ids.append(
                str(row["session_id"])
            )

        else:

            device_ids.append(
                f"DEVICE_UNKNOWN_{account}"
            )

            ip_ids.append(
                f"IP_UNKNOWN_{account}"
            )

            merchant_ids.append(
                f"MERCHANT_UNKNOWN_{account}"
            )

            session_ids.append(
                f"SESSION_UNKNOWN_{account}"
            )

    device_to_idx = create_mapping(
        device_ids
    )

    ip_to_idx = create_mapping(
        ip_ids
    )

    merchant_to_idx = create_mapping(
        merchant_ids
    )

    session_to_idx = create_mapping(
        session_ids
    )

    account_number = np.arange(
        n_accounts,
        dtype=np.int64,
    )

    account_device_idx = np.array(
        [
            device_to_idx[x]
            for x in device_ids
        ],
        dtype=np.int64,
    )

    account_ip_idx = np.array(
        [
            ip_to_idx[x]
            for x in ip_ids
        ],
        dtype=np.int64,
    )

    account_merchant_idx = np.array(
        [
            merchant_to_idx[x]
            for x in merchant_ids
        ],
        dtype=np.int64,
    )

    account_session_idx = np.array(
        [
            session_to_idx[x]
            for x in session_ids
        ],
        dtype=np.int64,
    )

    # ========================================================
    # HETERODATA
    # ========================================================

    data = HeteroData()

    # ========================================================
    # TRANSACTION NODE
    # ========================================================

    data["transaction"].x = (
        build_transaction_features(df)
    )

    data["transaction"].y = torch.tensor(
        df["isFraud"].to_numpy(
            dtype=np.int64
        ),
        dtype=torch.long,
    )

    data["transaction"].step = torch.tensor(
        df["step"].to_numpy(
            dtype=np.int64
        ),
        dtype=torch.long,
    )

    data["transaction"].transaction_id = (
        df["transaction_id"]
        .astype(str)
        .tolist()
    )

    # ========================================================
    # ACCOUNT FEATURES
    #
    # IMPORTANT:
    # Count only historical transactions.
    # ========================================================

    historical_sender_accounts = (
        sender_idx[
            history_transaction_mask
        ]
    )

    historical_receiver_accounts = (
        receiver_idx[
            history_transaction_mask
        ]
    )

    sent_count = np.bincount(
        historical_sender_accounts,
        minlength=n_accounts,
    ).astype(
        np.float32
    )

    received_count = np.bincount(
        historical_receiver_accounts,
        minlength=n_accounts,
    ).astype(
        np.float32
    )

    account_features = np.column_stack(
        [
            np.log1p(sent_count),
            np.log1p(received_count),
        ]
    )

    data["account"].x = torch.tensor(
        account_features,
        dtype=torch.float32,
    )

    # IMPORTANT:
    # No account.y is created.
    #
    # The previous account.y was derived from fraudulent
    # sender labels and is not needed for the current
    # transaction-level GraphSAGE model.

    # ========================================================
    # HISTORICAL ACCOUNT INDICES
    #
    # IMPORTANT:
    # This is ACCOUNT-level, unlike history_mask.
    # ========================================================

    historical_account_indices = np.unique(
        np.concatenate(
            [
                historical_sender_accounts,
                historical_receiver_accounts,
            ]
        )
    )

    print(
        "Historical accounts:",
        f"{len(historical_account_indices):,}"
    )

    # ========================================================
    # DEVICE
    #
    # Degree based on HISTORICAL accounts only.
    # ========================================================

    n_devices = len(
        device_to_idx
    )

    data["device"].num_nodes = (
        n_devices
    )

    historical_device_idx = (
        account_device_idx[
            historical_account_indices
        ]
    )

    data["device"].x = (
        degree_feature(
            n_devices,
            historical_device_idx,
        )
    )

    # ========================================================
    # IP
    #
    # Degree based on HISTORICAL accounts only.
    # ========================================================

    n_ips = len(
        ip_to_idx
    )

    data["ip"].num_nodes = (
        n_ips
    )

    historical_ip_idx = (
        account_ip_idx[
            historical_account_indices
        ]
    )

    data["ip"].x = (
        degree_feature(
            n_ips,
            historical_ip_idx,
        )
    )

    # ========================================================
    # MERCHANT
    #
    # Degree based on HISTORICAL accounts only.
    # ========================================================

    n_merchants = len(
        merchant_to_idx
    )

    data["merchant"].num_nodes = (
        n_merchants
    )

    historical_merchant_idx = (
        account_merchant_idx[
            historical_account_indices
        ]
    )

    data["merchant"].x = (
        degree_feature(
            n_merchants,
            historical_merchant_idx,
        )
    )

    # ========================================================
    # SESSION
    #
    # Degree based on HISTORICAL accounts only.
    # ========================================================

    n_sessions = len(
        session_to_idx
    )

    data["session"].num_nodes = (
        n_sessions
    )

    historical_session_idx = (
        account_session_idx[
            historical_account_indices
        ]
    )

    data["session"].x = (
        degree_feature(
            n_sessions,
            historical_session_idx,
        )
    )

    # ========================================================
    # TIME
    # ========================================================

    step_tensor = torch.tensor(
        df["step"].to_numpy(
            dtype=np.int64
        ),
        dtype=torch.long,
    )

    # ========================================================
    # ACCOUNT -> TRANSACTION
    # SENT
    # ========================================================

    edge_index = torch.tensor(
        np.vstack(
            [
                sender_idx,
                transaction_index,
            ]
        ),
        dtype=torch.long,
    )

    data[
        "account",
        "SENT",
        "transaction",
    ].edge_index = (
        edge_index
    )

    data[
        "account",
        "SENT",
        "transaction",
    ].edge_time = (
        step_tensor.clone()
    )

    # ========================================================
    # TRANSACTION -> ACCOUNT
    # SENT_BY
    # ========================================================

    edge_index = torch.tensor(
        np.vstack(
            [
                transaction_index,
                sender_idx,
            ]
        ),
        dtype=torch.long,
    )

    data[
        "transaction",
        "SENT_BY",
        "account",
    ].edge_index = (
        edge_index
    )

    data[
        "transaction",
        "SENT_BY",
        "account",
    ].edge_time = (
        step_tensor.clone()
    )

    # ========================================================
    # ACCOUNT -> TRANSACTION
    # RECEIVED
    # ========================================================

    edge_index = torch.tensor(
        np.vstack(
            [
                receiver_idx,
                transaction_index,
            ]
        ),
        dtype=torch.long,
    )

    data[
        "account",
        "RECEIVED",
        "transaction",
    ].edge_index = (
        edge_index
    )

    data[
        "account",
        "RECEIVED",
        "transaction",
    ].edge_time = (
        step_tensor.clone()
    )

    # ========================================================
    # TRANSACTION -> ACCOUNT
    # RECEIVED_BY
    # ========================================================

    edge_index = torch.tensor(
        np.vstack(
            [
                transaction_index,
                receiver_idx,
            ]
        ),
        dtype=torch.long,
    )

    data[
        "transaction",
        "RECEIVED_BY",
        "account",
    ].edge_index = (
        edge_index
    )

    data[
        "transaction",
        "RECEIVED_BY",
        "account",
    ].edge_time = (
        step_tensor.clone()
    )

    # ========================================================
    # CONNECTED_TO
    #
    # HISTORY ONLY
    # ========================================================

    history_sender = (
        sender_idx[
            history_transaction_mask
        ]
    )

    history_receiver = (
        receiver_idx[
            history_transaction_mask
        ]
    )

    history_steps = step_tensor[
        torch.from_numpy(
            history_transaction_mask
        )
    ]

    connected_mask = (
        history_sender
        != history_receiver
    )

    connected_src = (
        history_sender[
            connected_mask
        ]
    )

    connected_dst = (
        history_receiver[
            connected_mask
        ]
    )

    connected_time = (
        history_steps[
            torch.from_numpy(
                connected_mask
            )
        ]
    )

    if len(connected_src) > 0:

        connected_edges = torch.tensor(
            np.vstack(
                [
                    connected_src,
                    connected_dst,
                ]
            ),
            dtype=torch.long,
        )

        data[
            "account",
            "CONNECTED_TO",
            "account",
        ].edge_index = (
            connected_edges
        )

        data[
            "account",
            "CONNECTED_TO",
            "account",
        ].edge_time = (
            connected_time
        )

    else:

        data[
            "account",
            "CONNECTED_TO",
            "account",
        ].edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        data[
            "account",
            "CONNECTED_TO",
            "account",
        ].edge_time = torch.empty(
            (0,),
            dtype=torch.long,
        )

    # ========================================================
    # ACCOUNT -> DEVICE
    # ========================================================

    data[
        "account",
        "USES_DEVICE",
        "device",
    ].edge_index = torch.tensor(
        np.vstack(
            [
                account_number,
                account_device_idx,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # ACCOUNT -> IP
    # ========================================================

    data[
        "account",
        "USES_IP",
        "ip",
    ].edge_index = torch.tensor(
        np.vstack(
            [
                account_number,
                account_ip_idx,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # ACCOUNT -> MERCHANT
    # ========================================================

    data[
        "account",
        "USES_MERCHANT",
        "merchant",
    ].edge_index = torch.tensor(
        np.vstack(
            [
                account_number,
                account_merchant_idx,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # ACCOUNT -> SESSION
    # ========================================================

    data[
        "account",
        "LOGIN_FROM",
        "session",
    ].edge_index = torch.tensor(
        np.vstack(
            [
                account_number,
                account_session_idx,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # TARGET MASKS
    # ========================================================

    tx_steps = (
        df["step"]
        .to_numpy(
            dtype=np.int32
        )
    )

    train_mask = (
        tx_steps <= target_start_step
    )

    val_mask = (
        (tx_steps > target_start_step)
        & (tx_steps <= target_end_step)
    )

    test_mask = (
        tx_steps > target_end_step
    )

    data[
        "transaction"
    ].train_mask = torch.tensor(
        train_mask,
        dtype=torch.bool,
    )

    data[
        "transaction"
    ].val_mask = torch.tensor(
        val_mask,
        dtype=torch.bool,
    )

    data[
        "transaction"
    ].test_mask = torch.tensor(
        test_mask,
        dtype=torch.bool,
    )

    # ========================================================
    # METADATA
    # ========================================================

    data.graph_name = "FinGraph"

    data.snapshot_name = (
        snapshot_name
    )

    data.snapshot_start_step = int(
        tx_steps.min()
    )

    data.snapshot_end_step = int(
        tx_steps.max()
    )

    data.history_end_step = int(
        history_end_step
    )

    data.target_start_step = int(
        target_start_step
    )

    data.target_end_step = int(
        target_end_step
    )

    data.random_seed = int(
        seed
    )

    # ========================================================
    # SNAPSHOT SUMMARY
    # ========================================================

    print()
    print(
        "Nodes:"
    )

    for node_type in data.node_types:

        print(
            f"  {node_type:15s}: "
            f"{data[node_type].num_nodes:,}"
        )

    print()

    print(
        "Edges:"
    )

    for edge_type in data.edge_types:

        count = data[
            edge_type
        ].edge_index.shape[1]

        print(
            f"  {str(edge_type):55s}: "
            f"{count:,}"
        )

    print()

    print(
        "Transactions in snapshot:",
        f"{len(df):,}"
    )

    print(
        "Fraud transactions:",
        int(
            (
                df["isFraud"] == 1
            ).sum()
        )
    )

    print(
        "History transactions used "
        "for aggregate/network features:",
        f"{len(history_df):,}"
    )

    return data


# ============================================================
# VALIDATE SNAPSHOT
# ============================================================

def validate_snapshot(
    data,
    snapshot_name,
):

    print()
    print(
        f"Validating {snapshot_name} snapshot..."
    )

    # --------------------------------------------------------
    # REQUIRED NODE TYPES
    # --------------------------------------------------------

    required_nodes = {
        "transaction",
        "account",
        "device",
        "ip",
        "merchant",
        "session",
    }

    if not required_nodes.issubset(
        set(data.node_types)
    ):

        raise ValueError(
            "Missing required node types."
        )

    # --------------------------------------------------------
    # REQUIRED EDGE TYPES
    # --------------------------------------------------------

    required_edges = [
        (
            "account",
            "SENT",
            "transaction",
        ),
        (
            "transaction",
            "SENT_BY",
            "account",
        ),
        (
            "account",
            "RECEIVED",
            "transaction",
        ),
        (
            "transaction",
            "RECEIVED_BY",
            "account",
        ),
        (
            "account",
            "CONNECTED_TO",
            "account",
        ),
        (
            "account",
            "USES_DEVICE",
            "device",
        ),
        (
            "account",
            "USES_IP",
            "ip",
        ),
        (
            "account",
            "USES_MERCHANT",
            "merchant",
        ),
        (
            "account",
            "LOGIN_FROM",
            "session",
        ),
    ]

    for edge in required_edges:

        if edge not in data.edge_types:

            raise ValueError(
                f"Missing edge: {edge}"
            )

    # --------------------------------------------------------
    # TRANSACTION LABELS
    # --------------------------------------------------------

    labels = data[
        "transaction"
    ].y

    unique_labels = set(
        torch.unique(
            labels
        ).tolist()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):

        raise ValueError(
            "Transaction labels are invalid."
        )

    # --------------------------------------------------------
    # ACCOUNT LABEL CHECK
    #
    # There should be NO account.y.
    # --------------------------------------------------------

    if hasattr(
        data["account"],
        "y",
    ):

        raise ValueError(
            "Unexpected account.y found. "
            "Account labels must not be stored."
        )

    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

    for node_type in data.node_types:

        if not hasattr(
            data[node_type],
            "x",
        ):
            continue

        x = data[node_type].x

        if x.is_floating_point():

            if not bool(
                torch.isfinite(x).all()
            ):

                raise ValueError(
                    f"NaN/Inf detected in "
                    f"{node_type}.x"
                )

    # --------------------------------------------------------
    # EDGE TIME
    # --------------------------------------------------------

    temporal_edges = [
        (
            "account",
            "SENT",
            "transaction",
        ),
        (
            "transaction",
            "SENT_BY",
            "account",
        ),
        (
            "account",
            "RECEIVED",
            "transaction",
        ),
        (
            "transaction",
            "RECEIVED_BY",
            "account",
        ),
        (
            "account",
            "CONNECTED_TO",
            "account",
        ),
    ]

    for edge_type in temporal_edges:

        store = data[
            edge_type
        ]

        if not hasattr(
            store,
            "edge_time",
        ):

            raise ValueError(
                f"Missing edge_time for "
                f"{edge_type}"
            )

        if len(store.edge_time) != (
            store.edge_index.shape[1]
        ):

            raise ValueError(
                f"edge_time length mismatch "
                f"for {edge_type}"
            )

    # --------------------------------------------------------
    # MASK CHECK
    # --------------------------------------------------------

    transaction_store = (
        data["transaction"]
    )

    train_mask = (
        transaction_store.train_mask
    )

    val_mask = (
        transaction_store.val_mask
    )

    test_mask = (
        transaction_store.test_mask
    )

    overlap_train_val = int(
        (
            train_mask
            & val_mask
        ).sum()
    )

    overlap_train_test = int(
        (
            train_mask
            & test_mask
        ).sum()
    )

    overlap_val_test = int(
        (
            val_mask
            & test_mask
        ).sum()
    )

    if (
        overlap_train_val > 0
        or overlap_train_test > 0
        or overlap_val_test > 0
    ):

        raise ValueError(
            "Mask overlap detected."
        )

    print(
        f"{snapshot_name} validation PASSED."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print("=" * 60)
    print(
        "FinGraph STRICT TEMPORAL GRAPH BUILDER"
    )
    print("=" * 60)

    print()
    print("PaySim:")
    print(PAYSIM_PATH)

    print()
    print("Relationships:")
    print(RELATIONSHIP_PATH)

    # ========================================================
    # FILE CHECKS
    # ========================================================

    if not PAYSIM_PATH.exists():

        raise FileNotFoundError(
            PAYSIM_PATH
        )

    if not RELATIONSHIP_PATH.exists():

        raise FileNotFoundError(
            RELATIONSHIP_PATH
        )

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SAMPLE
    # ========================================================

    df = uniform_sample(
        sample_size=args.sample_size,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    df = validate_sample(
        df
    )

    # ========================================================
    # TEMPORAL CUTOFFS
    # ========================================================

    (
        train_end,
        val_end,
        test_end,
    ) = determine_cutoffs(
        df
    )

    # ========================================================
    # UNIQUE ACCOUNTS
    # ========================================================

    required_accounts = set(
        df["nameOrig"]
        .astype(str)
    )

    required_accounts.update(
        df["nameDest"]
        .astype(str)
    )

    print()
    print(
        f"Unique sampled accounts: "
        f"{len(required_accounts):,}"
    )

    # ========================================================
    # LOAD RELATIONSHIPS
    # ========================================================

    relationship_df = load_relationships(
        account_ids=required_accounts,
        chunk_size=args.chunk_size,
    )

    # ========================================================
    # CUMULATIVE DATAFRAMES
    # ========================================================

    train_df = df[
        df["step"] <= train_end
    ].copy()

    val_df = df[
        df["step"] <= val_end
    ].copy()

    test_df = df[
        df["step"] <= test_end
    ].copy()

    # ========================================================
    # TRAIN GRAPH
    #
    # Snapshot:
    #     steps <= 323
    #
    # History:
    #     steps <= 323
    # ========================================================

    train_graph = build_snapshot(
        snapshot_df=train_df,
        history_end_step=train_end,
        target_start_step=train_end,
        target_end_step=train_end,
        snapshot_name="train",
        relationship_df=relationship_df,
        seed=args.seed,
    )

    # Only train targets exist here.
    train_graph[
        "transaction"
    ].val_mask = torch.zeros(
        len(train_df),
        dtype=torch.bool,
    )

    train_graph[
        "transaction"
    ].test_mask = torch.zeros(
        len(train_df),
        dtype=torch.bool,
    )

    validate_snapshot(
        train_graph,
        "TRAIN",
    )

    torch.save(
        train_graph,
        TRAIN_GRAPH_PATH,
    )

    print()
    print(
        "Saved:",
        TRAIN_GRAPH_PATH,
    )

    # ========================================================
    # VALIDATION GRAPH
    #
    # Snapshot:
    #     steps <= 378
    #
    # History:
    #     steps <= 323
    #
    # Targets:
    #     steps 324 -> 378
    # ========================================================

    val_graph = build_snapshot(
        snapshot_df=val_df,
        history_end_step=train_end,
        target_start_step=train_end,
        target_end_step=val_end,
        snapshot_name="validation",
        relationship_df=relationship_df,
        seed=args.seed,
    )

    validate_snapshot(
        val_graph,
        "VALIDATION",
    )

    torch.save(
        val_graph,
        VAL_GRAPH_PATH,
    )

    print()
    print(
        "Saved:",
        VAL_GRAPH_PATH,
    )

    # ========================================================
    # TEST GRAPH
    #
    # Snapshot:
    #     steps <= 741
    #
    # History:
    #     steps <= 378
    #
    # Targets:
    #     steps 379 -> 741
    # ========================================================

    test_graph = build_snapshot(
        snapshot_df=test_df,
        history_end_step=val_end,
        target_start_step=val_end,
        target_end_step=test_end,
        snapshot_name="test",
        relationship_df=relationship_df,
        seed=args.seed,
    )

    test_steps_tensor = (
        test_graph[
            "transaction"
        ].step
    )

    # --------------------------------------------------------
    # Test graph masks
    # --------------------------------------------------------

    test_graph[
        "transaction"
    ].train_mask = (
        test_steps_tensor <= val_end
    )

    test_graph[
        "transaction"
    ].val_mask = torch.zeros(
        len(test_steps_tensor),
        dtype=torch.bool,
    )

    test_graph[
        "transaction"
    ].test_mask = (
        test_steps_tensor > val_end
    )

    validate_snapshot(
        test_graph,
        "TEST",
    )

    torch.save(
        test_graph,
        TEST_GRAPH_PATH,
    )

    print()
    print(
        "Saved:",
        TEST_GRAPH_PATH,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print(
        "STRICT TEMPORAL GRAPH BUILD COMPLETE"
    )
    print("=" * 60)

    print()
    print(
        "Train graph:",
        TRAIN_GRAPH_PATH,
    )

    print(
        "Validation graph:",
        VAL_GRAPH_PATH,
    )

    print(
        "Test graph:",
        TEST_GRAPH_PATH,
    )

    print()
    print(
        "Temporal cutoffs:"
    )

    print(
        f"  Train ends at step: "
        f"{train_end}"
    )

    print(
        f"  Validation ends at step: "
        f"{val_end}"
    )

    print(
        f"  Test ends at step: "
        f"{test_end}"
    )

    print()
    print(
        "Aggregate/network history:"
    )

    print(
        f"  Train: "
        f"<= {train_end}"
    )

    print(
        f"  Validation: "
        f"<= {train_end}"
    )

    print(
        f"  Test: "
        f"<= {val_end}"
    )

    print()
    print(
        "Account fraud labels:"
    )

    print(
        "  NOT stored in graph."
    )

    print()
    print(
        "Original fingraph_graph.pt:"
    )

    print(
        "  NOT overwritten."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()