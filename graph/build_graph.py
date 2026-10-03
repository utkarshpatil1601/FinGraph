"""
FinGraph - Heterogeneous Graph Builder

Builds a PyTorch Geometric HeteroData graph from:
    data/processed/paysim_engineered.csv
    data/synthetic/account_relationships.csv

Features:
- Uniform random sampling across the FULL PaySim dataset
- Deterministic sampling with seed
- Strict isFraud validation
- Numeric type conversion
- Temporal train/validation/test masks
- Account, transaction, device, IP, merchant, session nodes
- SENT / SENT_BY
- RECEIVED / RECEIVED_BY
- USES_DEVICE
- USES_IP
- USES_MERCHANT
- LOGIN_FROM
- CONNECTED_TO
- Transaction and account labels
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAYSIM_PATH = PROJECT_ROOT / "data" / "processed" / "paysim_engineered.csv"
RELATIONSHIP_PATH = (
    PROJECT_ROOT / "data" / "synthetic" / "account_relationships.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "graphs"
OUTPUT_PATH = OUTPUT_DIR / "fingraph_graph.pt"


# ============================================================
# EXPECTED COLUMNS
# ============================================================

REQUIRED_PAYSim_COLUMNS = [
    "transaction_id",
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

OPTIONAL_NUMERIC_COLUMNS = [
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


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the FinGraph heterogeneous graph"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of transactions to sample from the full dataset",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="CSV chunk size",
    )

    return parser.parse_args()


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_paySim_columns():
    if not PAYSIM_PATH.exists():
        raise FileNotFoundError(
            f"PaySim file not found:\n{PAYSIM_PATH}"
        )

    header = pd.read_csv(PAYSIM_PATH, nrows=0)

    missing = [
        col
        for col in REQUIRED_PAYSim_COLUMNS
        if col not in header.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required PaySim columns:\n{missing}"
        )

    print("PaySim columns validated.")


# ============================================================
# FULL DATASET ROW COUNT
# ============================================================

def count_rows():
    """
    Count rows in the full CSV without loading the full dataframe.
    """

    print()
    print("Counting rows in full dataset...")

    count = 0

    with open(
        PAYSIM_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        reader = csv.reader(f)

        # skip header
        next(reader)

        for _ in reader:
            count += 1

    print(f"Total transaction rows: {count:,}")

    return count


# ============================================================
# UNIFORM SAMPLE ACROSS FULL DATASET
# ============================================================

def uniform_sample_full_dataset(
    sample_size: int,
    seed: int,
    chunk_size: int,
):
    """
    Select a uniform random sample from the ENTIRE CSV.

    This avoids the earlier problem where:
        df.head(100000)

    selected only early PaySim time steps.

    We assign every row a deterministic random key and
    retain the smallest `sample_size` keys.
    """

    rng = np.random.default_rng(seed)

    print()
    print("=" * 60)
    print("UNIFORM SAMPLING ACROSS FULL DATASET")
    print("=" * 60)

    total_rows = count_rows()

    if sample_size >= total_rows:
        print(
            f"Requested sample size ({sample_size:,}) >= "
            f"dataset size ({total_rows:,})."
        )
        print("Using the complete dataset.")

        df = pd.read_csv(PAYSIM_PATH)

        return df

    print(f"Target sample size: {sample_size:,}")
    print(f"Random seed: {seed}")
    print()

    current_sample = None
    processed = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            PAYSIM_PATH,
            chunksize=chunk_size,
            low_memory=False,
        ),
        start=1,
    ):

        chunk = chunk.copy()

        # Random key for each row
        chunk["_random_key"] = rng.random(len(chunk))

        if current_sample is None:
            current_sample = chunk

        else:
            current_sample = pd.concat(
                [current_sample, chunk],
                ignore_index=True,
            )

        # Keep only the best sample_size random keys
        if len(current_sample) > sample_size:
            current_sample = (
                current_sample
                .nsmallest(sample_size, "_random_key")
                .reset_index(drop=True)
            )

        processed += len(chunk)

        print(
            f"\rScanned: {processed:,}/{total_rows:,} rows",
            end="",
            flush=True,
        )

    print()
    print(f"Selected: {len(current_sample):,} rows")

    current_sample = current_sample.drop(
        columns=["_random_key"]
    )

    return current_sample.reset_index(drop=True)


# ============================================================
# STRICT TYPE CONVERSION
# ============================================================

def clean_and_validate_sample(df: pd.DataFrame):
    """
    Convert sampled fields to correct numeric types.

    IMPORTANT:
    Do NOT silently replace invalid isFraud values with 0.
    """

    print()
    print("=" * 60)
    print("VALIDATING SAMPLED DATA")
    print("=" * 60)

    # --------------------------------------------------------
    # isFraud
    # --------------------------------------------------------

    raw_fraud = df["isFraud"]

    fraud_numeric = pd.to_numeric(
        raw_fraud,
        errors="coerce",
    )

    # Null / non-numeric values
    if fraud_numeric.isna().any():

        bad_rows = df.loc[
            fraud_numeric.isna(),
            ["transaction_id", "isFraud"]
        ].head(10)

        raise ValueError(
            "isFraud contains null or non-numeric values "
            "after sampling.\n\n"
            f"{bad_rows}"
        )

    # Only 0 or 1 allowed
    invalid_mask = ~fraud_numeric.isin([0, 1])

    if invalid_mask.any():

        bad_rows = df.loc[
            invalid_mask,
            ["transaction_id", "isFraud"]
        ].head(10)

        raise ValueError(
            "isFraud must contain only 0 or 1 "
            "after conversion.\n\n"
            f"Invalid examples:\n{bad_rows}"
        )

    # Safe conversion
    df["isFraud"] = fraud_numeric.astype("int8")

    # --------------------------------------------------------
    # step
    # --------------------------------------------------------

    df["step"] = pd.to_numeric(
        df["step"],
        errors="raise",
    ).astype("int32")

    # --------------------------------------------------------
    # Numeric feature columns
    # --------------------------------------------------------

    for col in OPTIONAL_NUMERIC_COLUMNS:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="raise",
            )

    # --------------------------------------------------------
    # String columns
    # --------------------------------------------------------

    for col in [
        "transaction_id",
        "type",
        "nameOrig",
        "nameDest",
    ]:

        if col in df.columns:
            df[col] = df[col].astype(str)

    # --------------------------------------------------------
    # Print validation
    # --------------------------------------------------------

    fraud_counts = (
        df["isFraud"]
        .value_counts()
        .sort_index()
    )

    print("Sample validation successful.")
    print()
    print("Sample rows:", f"{len(df):,}")
    print("Fraud distribution:")

    for label, count in fraud_counts.items():
        print(f"  {label}: {count:,}")

    print()
    print(
        "Unique steps:",
        df["step"].nunique()
    )

    print(
        "Step range:",
        int(df["step"].min()),
        "->",
        int(df["step"].max()),
    )

    return df


# ============================================================
# RELATIONSHIP COLUMN DETECTION
# ============================================================

def find_relationship_column(columns, possible_names):
    """
    Find a relationship column using several accepted names.
    """

    for name in possible_names:
        if name in columns:
            return name

    return None


# ============================================================
# LOAD ONLY REQUIRED ACCOUNT RELATIONSHIPS
# ============================================================

def load_relationships_for_accounts(
    account_ids,
    chunk_size: int,
):
    """
    Load synthetic relationship data only for accounts
    appearing in the sampled transaction graph.

    This prevents loading all ~9 million accounts into memory.
    """

    print()
    print("=" * 60)
    print("LOADING SYNTHETIC RELATIONSHIPS")
    print("=" * 60)

    if not RELATIONSHIP_PATH.exists():
        raise FileNotFoundError(
            f"Relationship file not found:\n{RELATIONSHIP_PATH}"
        )

    target_accounts = set(
        str(x)
        for x in account_ids
    )

    print(
        f"Accounts needed for sampled graph: "
        f"{len(target_accounts):,}"
    )

    header = pd.read_csv(
        RELATIONSHIP_PATH,
        nrows=0,
    )

    columns = list(header.columns)

    account_col = find_relationship_column(
        columns,
        [
            "account_id",
            "account",
            "name",
        ],
    )

    device_col = find_relationship_column(
        columns,
        [
            "device_id",
            "device",
        ],
    )

    ip_col = find_relationship_column(
        columns,
        [
            "ip_address",
            "ip_id",
            "ip",
        ],
    )

    merchant_col = find_relationship_column(
        columns,
        [
            "merchant_id",
            "merchant",
        ],
    )

    session_col = find_relationship_column(
        columns,
        [
            "session_id",
            "session",
        ],
    )

    if account_col is None:
        raise ValueError(
            "Could not find account column in relationship file.\n"
            f"Available columns: {columns}"
        )

    required_relation_columns = [
        ("device", device_col),
        ("ip", ip_col),
        ("merchant", merchant_col),
        ("session", session_col),
    ]

    missing = [
        label
        for label, col in required_relation_columns
        if col is None
    ]

    if missing:
        raise ValueError(
            "Missing relationship columns: "
            f"{missing}\n"
            f"Available columns: {columns}"
        )

    selected_columns = [
        account_col,
        device_col,
        ip_col,
        merchant_col,
        session_col,
    ]

    selected_parts = []

    processed = 0

    for chunk in pd.read_csv(
        RELATIONSHIP_PATH,
        usecols=selected_columns,
        chunksize=chunk_size,
        dtype=str,
    ):

        processed += len(chunk)

        mask = chunk[account_col].isin(
            target_accounts
        )

        selected = chunk.loc[
            mask,
            selected_columns,
        ]

        if not selected.empty:
            selected_parts.append(selected)

        print(
            f"\rRelationship rows scanned: "
            f"{processed:,}",
            end="",
            flush=True,
        )

    print()

    if not selected_parts:
        raise ValueError(
            "No sampled accounts were found in "
            "account_relationships.csv."
        )

    relation_df = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    relation_df = relation_df.rename(
        columns={
            account_col: "account_id",
            device_col: "device_id",
            ip_col: "ip_address",
            merchant_col: "merchant_id",
            session_col: "session_id",
        }
    )

    # Remove duplicate accounts while preserving first record
    duplicated_accounts = (
        relation_df["account_id"]
        .duplicated()
    )

    if duplicated_accounts.any():

        duplicate_count = int(
            duplicated_accounts.sum()
        )

        print(
            f"WARNING: {duplicate_count:,} duplicate "
            "relationship rows found."
        )

        relation_df = (
            relation_df
            .drop_duplicates(
                subset=["account_id"],
                keep="first",
            )
        )

    print()
    print(
        f"Relationships loaded: "
        f"{len(relation_df):,}"
    )

    print(
        f"Sampled accounts with relationships: "
        f"{relation_df['account_id'].nunique():,}"
    )

    return relation_df


# ============================================================
# INDEX MAPPING
# ============================================================

def make_mapping(values):
    """
    Create deterministic string -> integer node mapping.
    """

    unique_values = pd.unique(
        pd.Series(values).astype(str)
    )

    return {
        value: index
        for index, value in enumerate(unique_values)
    }


# ============================================================
# TRANSACTION FEATURES
# ============================================================

def build_transaction_features(df):
    """
    Build transaction feature matrix.

    Includes numeric PaySim / engineered features plus
    transaction type one-hot encoding.
    """

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

    available_numeric = [
        col
        for col in numeric_features
        if col in df.columns
    ]

    feature_parts = []

    if available_numeric:

        numeric_df = df[
            available_numeric
        ].copy()

        numeric_df = numeric_df.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        numeric_df = numeric_df.fillna(0)

        feature_parts.append(
            numeric_df.to_numpy(
                dtype=np.float32
            )
        )

    # --------------------------------------------------------
    # Transaction type one-hot
    # --------------------------------------------------------

    transaction_types = [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    ]

    type_matrix = np.zeros(
        (len(df), len(transaction_types)),
        dtype=np.float32,
    )

    type_to_index = {
        t: i
        for i, t in enumerate(transaction_types)
    }

    for i, transaction_type in enumerate(
        df["type"].astype(str)
    ):

        if transaction_type in type_to_index:
            type_matrix[
                i,
                type_to_index[transaction_type]
            ] = 1.0

    feature_parts.append(type_matrix)

    features = np.concatenate(
        feature_parts,
        axis=1,
    )

    return torch.tensor(
        features,
        dtype=torch.float32,
    )


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def build_temporal_masks(steps):
    """
    Create approximately:
        70% train
        15% validation
        15% test

    based on complete PaySim time steps.

    No random mixing between time periods.
    """

    steps = np.asarray(
        steps,
        dtype=np.int32,
    )

    unique_steps, counts = np.unique(
        steps,
        return_counts=True,
    )

    if len(unique_steps) < 3:
        raise ValueError(
            "Not enough unique time steps for "
            "train/validation/test split."
        )

    cumulative_counts = np.cumsum(counts)

    n = len(steps)

    train_target = 0.70 * n
    validation_target = 0.85 * n

    train_end_idx = int(
        np.searchsorted(
            cumulative_counts,
            train_target,
            side="left",
        )
    )

    validation_end_idx = int(
        np.searchsorted(
            cumulative_counts,
            validation_target,
            side="left",
        )
    )

    # Clamp indices
    train_end_idx = min(
        train_end_idx,
        len(unique_steps) - 2,
    )

    validation_end_idx = max(
        validation_end_idx,
        train_end_idx + 1,
    )

    validation_end_idx = min(
        validation_end_idx,
        len(unique_steps) - 1,
    )

    train_steps = unique_steps[
        :train_end_idx + 1
    ]

    validation_steps = unique_steps[
        train_end_idx + 1:
        validation_end_idx + 1
    ]

    test_steps = unique_steps[
        validation_end_idx + 1:
    ]

    # Safety adjustments
    if len(validation_steps) == 0:

        validation_steps = np.array(
            [unique_steps[train_end_idx + 1]]
        )

    if len(test_steps) == 0:

        test_steps = np.array(
            [unique_steps[-1]]
        )

    train_mask = np.isin(
        steps,
        train_steps,
    )

    validation_mask = np.isin(
        steps,
        validation_steps,
    )

    test_mask = np.isin(
        steps,
        test_steps,
    )

    return (
        train_mask,
        validation_mask,
        test_mask,
        train_steps,
        validation_steps,
        test_steps,
    )


# ============================================================
# NODE FEATURES FOR RELATIONSHIP NODES
# ============================================================

def build_degree_features(
    number_of_nodes,
    edge_destination_indices,
):
    """
    Simple structural feature:
        log1p(number of connections)
    """

    degree = np.bincount(
        edge_destination_indices,
        minlength=number_of_nodes,
    ).astype(np.float32)

    degree = np.log1p(degree)

    return torch.tensor(
        degree.reshape(-1, 1),
        dtype=torch.float32,
    )


# ============================================================
# MAIN GRAPH CONSTRUCTION
# ============================================================

def build_graph(
    df: pd.DataFrame,
    relationship_df: pd.DataFrame,
):
    print()
    print("=" * 60)
    print("BUILDING FINGRAPH")
    print("=" * 60)

    # --------------------------------------------------------
    # Convert IDs to strings
    # --------------------------------------------------------

    df["nameOrig"] = df["nameOrig"].astype(str)
    df["nameDest"] = df["nameDest"].astype(str)

    relationship_df["account_id"] = (
        relationship_df["account_id"]
        .astype(str)
    )

    for col in [
        "device_id",
        "ip_address",
        "merchant_id",
        "session_id",
    ]:
        relationship_df[col] = (
            relationship_df[col]
            .astype(str)
        )

    # --------------------------------------------------------
    # Transaction mapping
    # --------------------------------------------------------

    transaction_ids = (
        df["transaction_id"]
        .astype(str)
        .tolist()
    )

    transaction_to_idx = make_mapping(
        transaction_ids
    )

    transaction_index = np.arange(
        len(df),
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Account mapping
    # --------------------------------------------------------

    sender_accounts = (
        df["nameOrig"]
        .astype(str)
        .tolist()
    )

    receiver_accounts = (
        df["nameDest"]
        .astype(str)
        .tolist()
    )

    all_accounts = (
        sender_accounts +
        receiver_accounts
    )

    account_to_idx = make_mapping(
        all_accounts
    )

    sender_idx = np.array(
        [
            account_to_idx[x]
            for x in sender_accounts
        ],
        dtype=np.int64,
    )

    receiver_idx = np.array(
        [
            account_to_idx[x]
            for x in receiver_accounts
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Relationship mappings
    # --------------------------------------------------------

    account_relation = (
        relationship_df
        .set_index("account_id")
    )

    number_accounts = len(
        account_to_idx
    )

    account_device_ids = []
    account_ip_ids = []
    account_merchant_ids = []
    account_session_ids = []

    for account in account_to_idx.keys():

        if account in account_relation.index:

            row = account_relation.loc[account]

            account_device_ids.append(
                str(row["device_id"])
            )

            account_ip_ids.append(
                str(row["ip_address"])
            )

            account_merchant_ids.append(
                str(row["merchant_id"])
            )

            account_session_ids.append(
                str(row["session_id"])
            )

        else:
            # This should be rare, but keeps the graph valid.
            account_device_ids.append(
                f"DEVICE_UNKNOWN_{account}"
            )

            account_ip_ids.append(
                f"IP_UNKNOWN_{account}"
            )

            account_merchant_ids.append(
                f"MERCHANT_UNKNOWN_{account}"
            )

            account_session_ids.append(
                f"SESSION_UNKNOWN_{account}"
            )

    device_to_idx = make_mapping(
        account_device_ids
    )

    ip_to_idx = make_mapping(
        account_ip_ids
    )

    merchant_to_idx = make_mapping(
        account_merchant_ids
    )

    session_to_idx = make_mapping(
        account_session_ids
    )

    # --------------------------------------------------------
    # Relationship edge indices
    # --------------------------------------------------------

    account_order = list(
        account_to_idx.keys()
    )

    account_device_idx = np.array(
        [
            device_to_idx[
                account_device_ids[i]
            ]
            for i in range(number_accounts)
        ],
        dtype=np.int64,
    )

    account_ip_idx = np.array(
        [
            ip_to_idx[
                account_ip_ids[i]
            ]
            for i in range(number_accounts)
        ],
        dtype=np.int64,
    )

    account_merchant_idx = np.array(
        [
            merchant_to_idx[
                account_merchant_ids[i]
            ]
            for i in range(number_accounts)
        ],
        dtype=np.int64,
    )

    account_session_idx = np.array(
        [
            session_to_idx[
                account_session_ids[i]
            ]
            for i in range(number_accounts)
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Create HeteroData
    # --------------------------------------------------------

    data = HeteroData()

    # ========================================================
    # TRANSACTION NODE
    # ========================================================

    transaction_x = build_transaction_features(
        df
    )

    data["transaction"].x = transaction_x

    data["transaction"].y = torch.tensor(
        df["isFraud"].to_numpy(),
        dtype=torch.long,
    )

    data["transaction"].step = torch.tensor(
        df["step"].to_numpy(),
        dtype=torch.long,
    )

    # ========================================================
    # ACCOUNT NODE
    # ========================================================

    sent_counts = np.bincount(
        sender_idx,
        minlength=number_accounts,
    ).astype(np.float32)

    received_counts = np.bincount(
        receiver_idx,
        minlength=number_accounts,
    ).astype(np.float32)

    account_x = np.column_stack(
        [
            np.log1p(sent_counts),
            np.log1p(received_counts),
        ]
    )

    data["account"].x = torch.tensor(
        account_x,
        dtype=torch.float32,
    )

    # Account fraud label:
    # account considered fraudulent if it was the
    # sender of at least one fraudulent transaction.
    account_labels = np.zeros(
        number_accounts,
        dtype=np.int64,
    )

    fraud_sender_mask = (
        df["isFraud"].to_numpy() == 1
    )

    fraudulent_sender_accounts = np.unique(
        sender_idx[fraud_sender_mask]
    )

    account_labels[
        fraudulent_sender_accounts
    ] = 1

    data["account"].y = torch.tensor(
        account_labels,
        dtype=torch.long,
    )

    # ========================================================
    # DEVICE NODE
    # ========================================================

    number_devices = len(device_to_idx)

    data["device"].num_nodes = number_devices

    data["device"].x = build_degree_features(
        number_devices,
        account_device_idx,
    )

    # ========================================================
    # IP NODE
    # ========================================================

    number_ips = len(ip_to_idx)

    data["ip"].num_nodes = number_ips

    data["ip"].x = build_degree_features(
        number_ips,
        account_ip_idx,
    )

    # ========================================================
    # MERCHANT NODE
    # ========================================================

    number_merchants = len(merchant_to_idx)

    data["merchant"].num_nodes = number_merchants

    data["merchant"].x = build_degree_features(
        number_merchants,
        account_merchant_idx,
    )

    # ========================================================
    # SESSION NODE
    # ========================================================

    number_sessions = len(session_to_idx)

    data["session"].num_nodes = number_sessions

    data["session"].x = build_degree_features(
        number_sessions,
        account_session_idx,
    )

    # ========================================================
    # ACCOUNT -> TRANSACTION
    # SENT
    # ========================================================

    data[
        "account",
        "SENT",
        "transaction",
    ].edge_index = torch.tensor(
        np.vstack(
            [
                sender_idx,
                transaction_index,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # TRANSACTION -> ACCOUNT
    # SENT_BY
    # ========================================================

    data[
        "transaction",
        "SENT_BY",
        "account",
    ].edge_index = torch.tensor(
        np.vstack(
            [
                transaction_index,
                sender_idx,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # ACCOUNT -> TRANSACTION
    # RECEIVED
    # ========================================================

    data[
        "account",
        "RECEIVED",
        "transaction",
    ].edge_index = torch.tensor(
        np.vstack(
            [
                receiver_idx,
                transaction_index,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # TRANSACTION -> ACCOUNT
    # RECEIVED_BY
    # ========================================================

    data[
        "transaction",
        "RECEIVED_BY",
        "account",
    ].edge_index = torch.tensor(
        np.vstack(
            [
                transaction_index,
                receiver_idx,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # ACCOUNT -> ACCOUNT
    # CONNECTED_TO
    # ========================================================

    connected_mask = (
        sender_idx != receiver_idx
    )

    connected_src = sender_idx[
        connected_mask
    ]

    connected_dst = receiver_idx[
        connected_mask
    ]

    if len(connected_src) > 0:

        connected_edges = np.vstack(
            [
                connected_src,
                connected_dst,
            ]
        )

        data[
            "account",
            "CONNECTED_TO",
            "account",
        ].edge_index = torch.tensor(
            connected_edges,
            dtype=torch.long,
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
                np.arange(
                    number_accounts,
                    dtype=np.int64,
                ),
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
                np.arange(
                    number_accounts,
                    dtype=np.int64,
                ),
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
                np.arange(
                    number_accounts,
                    dtype=np.int64,
                ),
                account_merchant_idx,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # ACCOUNT -> SESSION
    # LOGIN_FROM
    # ========================================================

    data[
        "account",
        "LOGIN_FROM",
        "session",
    ].edge_index = torch.tensor(
        np.vstack(
            [
                np.arange(
                    number_accounts,
                    dtype=np.int64,
                ),
                account_session_idx,
            ]
        ),
        dtype=torch.long,
    )

    # ========================================================
    # TEMPORAL MASKS
    # ========================================================

    (
        train_mask_np,
        validation_mask_np,
        test_mask_np,
        train_steps,
        validation_steps,
        test_steps,
    ) = build_temporal_masks(
        df["step"].to_numpy()
    )

    data["transaction"].train_mask = torch.tensor(
        train_mask_np,
        dtype=torch.bool,
    )

    data["transaction"].val_mask = torch.tensor(
        validation_mask_np,
        dtype=torch.bool,
    )

    data["transaction"].test_mask = torch.tensor(
        test_mask_np,
        dtype=torch.bool,
    )

    # --------------------------------------------------------
    # Account masks
    #
    # Account masks are derived from the earliest/latest
    # transaction step in which each account appears.
    # --------------------------------------------------------

    account_first_step = np.full(
        number_accounts,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )

    account_last_step = np.full(
        number_accounts,
        -1,
        dtype=np.int32,
    )

    steps_array = df["step"].to_numpy(
        dtype=np.int32
    )

    for i in range(len(df)):

        s = int(steps_array[i])

        sender = int(sender_idx[i])
        receiver = int(receiver_idx[i])

        if s < account_first_step[sender]:
            account_first_step[sender] = s

        if s < account_first_step[receiver]:
            account_first_step[receiver] = s

        if s > account_last_step[sender]:
            account_last_step[sender] = s

        if s > account_last_step[receiver]:
            account_last_step[receiver] = s

    train_min = int(train_steps.min())
    train_max = int(train_steps.max())

    val_min = int(validation_steps.min())
    val_max = int(validation_steps.max())

    test_min = int(test_steps.min())
    test_max = int(test_steps.max())

    account_train_mask = (
        (account_first_step >= train_min)
        & (account_first_step <= train_max)
    )

    account_val_mask = (
        (account_first_step >= val_min)
        & (account_first_step <= val_max)
    )

    account_test_mask = (
        (account_first_step >= test_min)
        & (account_first_step <= test_max)
    )

    data["account"].train_mask = torch.tensor(
        account_train_mask,
        dtype=torch.bool,
    )

    data["account"].val_mask = torch.tensor(
        account_val_mask,
        dtype=torch.bool,
    )

    data["account"].test_mask = torch.tensor(
        account_test_mask,
        dtype=torch.bool,
    )

    # ========================================================
    # METADATA
    # ========================================================

    data.graph_name = "FinGraph"
    data.sample_size = len(df)
    data.random_seed = int(
        args.seed if "args" in globals() else 42
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)

    print()
    print("NODE COUNTS")

    for node_type in data.node_types:

        print(
            f"  {str(node_type):15s}: "
            f"{data[node_type].num_nodes:,}"
        )

    print()
    print("EDGE COUNTS")

    for edge_type in data.edge_types:

        count = data[
            edge_type
        ].edge_index.shape[1]

        print(
            f"  {str(edge_type)}: "
            f"{count:,}"
        )

    # --------------------------------------------------------
    # Transaction split
    # --------------------------------------------------------

    print()
    print("TRANSACTION TEMPORAL SPLIT")

    train_count = int(
        data["transaction"].train_mask.sum()
    )

    val_count = int(
        data["transaction"].val_mask.sum()
    )

    test_count = int(
        data["transaction"].test_mask.sum()
    )

    total_count = (
        train_count +
        val_count +
        test_count
    )

    print(
        f"  Train:      {train_count:,} "
        f"({train_count / total_count * 100:.2f}%)"
    )

    print(
        f"  Validation: {val_count:,} "
        f"({val_count / total_count * 100:.2f}%)"
    )

    print(
        f"  Test:       {test_count:,} "
        f"({test_count / total_count * 100:.2f}%)"
    )

    print()
    print(
        f"  Train steps: "
        f"{train_steps.min()} -> {train_steps.max()}"
    )

    print(
        f"  Val steps:   "
        f"{validation_steps.min()} -> "
        f"{validation_steps.max()}"
    )

    print(
        f"  Test steps:  "
        f"{test_steps.min()} -> {test_steps.max()}"
    )

    # --------------------------------------------------------
    # Fraud distribution
    # --------------------------------------------------------

    y = data["transaction"].y.numpy()

    fraud_total = int(
        np.sum(y == 1)
    )

    genuine_total = int(
        np.sum(y == 0)
    )

    print()
    print("TRANSACTION LABELS")

    print(
        f"  Genuine: {genuine_total:,}"
    )

    print(
        f"  Fraud:   {fraud_total:,}"
    )

    print(
        f"  Fraud %: "
        f"{fraud_total / len(y) * 100:.4f}%"
    )

    return data


# ============================================================
# GRAPH VALIDATION
# ============================================================

def validate_graph(data):
    print()
    print("=" * 60)
    print("VALIDATING GRAPH")
    print("=" * 60)

    # --------------------------------------------------------
    # Node validation
    # --------------------------------------------------------

    expected_nodes = {
        "transaction",
        "account",
        "device",
        "ip",
        "merchant",
        "session",
    }

    actual_nodes = set(
        data.node_types
    )

    missing_nodes = (
        expected_nodes - actual_nodes
    )

    if missing_nodes:
        raise ValueError(
            f"Missing node types: {missing_nodes}"
        )

    # --------------------------------------------------------
    # Required edges
    # --------------------------------------------------------

    required_edges = [
        ("account", "SENT", "transaction"),
        ("transaction", "SENT_BY", "account"),
        ("account", "RECEIVED", "transaction"),
        ("transaction", "RECEIVED_BY", "account"),
        ("account", "CONNECTED_TO", "account"),
        ("account", "USES_DEVICE", "device"),
        ("account", "USES_IP", "ip"),
        ("account", "USES_MERCHANT", "merchant"),
        ("account", "LOGIN_FROM", "session"),
    ]

    for edge in required_edges:

        if edge not in data.edge_types:
            raise ValueError(
                f"Missing edge type: {edge}"
            )

    # --------------------------------------------------------
    # Transaction labels
    # --------------------------------------------------------

    labels = data["transaction"].y

    unique_labels = torch.unique(
        labels
    ).tolist()

    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            "Transaction labels contain values "
            f"other than 0/1: {unique_labels}"
        )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    train_mask = data[
        "transaction"
    ].train_mask

    val_mask = data[
        "transaction"
    ].val_mask

    test_mask = data[
        "transaction"
    ].test_mask

    if not (
        train_mask.dtype == torch.bool
        and val_mask.dtype == torch.bool
        and test_mask.dtype == torch.bool
    ):
        raise ValueError(
            "Transaction masks must be boolean."
        )

    overlap = (
        train_mask & val_mask
        | train_mask & test_mask
        | val_mask & test_mask
    )

    if overlap.any():
        raise ValueError(
            "Train/validation/test masks overlap."
        )

    combined = (
        train_mask
        | val_mask
        | test_mask
    )

    if not bool(torch.all(combined)):
        raise ValueError(
            "Some transaction nodes are not assigned "
            "to train/validation/test."
        )

    # --------------------------------------------------------
    # Edge index validation
    # --------------------------------------------------------

    for edge_type in data.edge_types:

        edge_index = data[
            edge_type
        ].edge_index

        if edge_index.ndim != 2:
            raise ValueError(
                f"Bad edge index shape for {edge_type}: "
                f"{edge_index.shape}"
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                f"Edge index must have shape [2, E] "
                f"for {edge_type}."
            )

    print("Graph validation PASSED.")

    print()
    print("Node types:")
    print(data.node_types)

    print()
    print("Edge types:")
    print(data.edge_types)

    print()
    print(
        "Transaction feature shape:",
        tuple(
            data["transaction"].x.shape
        ),
    )

    print(
        "Account feature shape:",
        tuple(
            data["account"].x.shape
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global args

    args = parse_args()

    print("=" * 60)
    print("FinGraph Graph Builder")
    print("=" * 60)

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("PaySim:")
    print(PAYSIM_PATH)

    print()
    print("Relationships:")
    print(RELATIONSHIP_PATH)

    print()
    print("Output:")
    print(OUTPUT_PATH)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not PAYSIM_PATH.exists():
        raise FileNotFoundError(
            f"Missing PaySim file:\n{PAYSIM_PATH}"
        )

    if not RELATIONSHIP_PATH.exists():
        raise FileNotFoundError(
            f"Missing relationship file:\n"
            f"{RELATIONSHIP_PATH}"
        )

    # --------------------------------------------------------
    # Validate PaySim header
    # --------------------------------------------------------

    validate_paySim_columns()

    # --------------------------------------------------------
    # Sample full dataset
    # --------------------------------------------------------

    df = uniform_sample_full_dataset(
        sample_size=args.sample_size,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    # --------------------------------------------------------
    # Clean + validate
    # --------------------------------------------------------

    df = clean_and_validate_sample(
        df
    )

    # --------------------------------------------------------
    # Required account list
    # --------------------------------------------------------

    required_accounts = set(
        df["nameOrig"].astype(str)
    )

    required_accounts.update(
        df["nameDest"].astype(str)
    )

    print()
    print(
        f"Unique accounts in sample: "
        f"{len(required_accounts):,}"
    )

    # --------------------------------------------------------
    # Load relevant relationships
    # --------------------------------------------------------

    relationship_df = (
        load_relationships_for_accounts(
            account_ids=required_accounts,
            chunk_size=args.chunk_size,
        )
    )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    data = build_graph(
        df,
        relationship_df,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_graph(
        data
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        data,
        OUTPUT_PATH,
    )

    print()
    print("=" * 60)
    print("GRAPH BUILD SUCCESSFUL")
    print("=" * 60)

    print()
    print(
        f"Saved graph to:\n{OUTPUT_PATH}"
    )

    print()
    print(
        f"File size: "
        f"{OUTPUT_PATH.stat().st_size / (1024 ** 2):.2f} MB"
    )

    print()
    print("DONE.")


if __name__ == "__main__":
    main()