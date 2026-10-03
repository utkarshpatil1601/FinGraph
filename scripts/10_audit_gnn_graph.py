"""
FinGraph - GraphSAGE Graph Audit

Purpose:
    Audit the temporal PyG graphs before modifying the GNN.

This script does NOT train or modify any model.
It only examines graph structure and transaction connectivity.
"""

from __future__ import annotations

from pathlib import Path

import torch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATHS = {
    "TRAIN": ROOT / "data" / "graphs" / "fingraph_train.pt",
    "VALIDATION": ROOT / "data" / "graphs" / "fingraph_val.pt",
    "TEST": ROOT / "data" / "graphs" / "fingraph_test.pt",
}


# ============================================================
# HELPERS
# ============================================================

def print_node_info(data, name):
    print("\n" + "-" * 70)
    print(f"{name} NODE INFORMATION")
    print("-" * 70)

    for node_type in data.node_types:

        store = data[node_type]

        num_nodes = store.num_nodes

        print(f"\n{node_type}:")
        print(f"  Nodes: {num_nodes}")

        if hasattr(store, "x"):
            print(
                f"  Features: {tuple(store.x.shape)}"
            )

        if hasattr(store, "y"):
            y = store.y

            fraud = int((y == 1).sum())
            genuine = int((y == 0).sum())

            print(f"  Labels: {len(y)}")
            print(f"  Fraud: {fraud}")
            print(f"  Genuine: {genuine}")

        for mask_name in [
            "train_mask",
            "val_mask",
            "test_mask",
        ]:
            if hasattr(store, mask_name):

                mask = getattr(
                    store,
                    mask_name,
                )

                print(
                    f"  {mask_name}: "
                    f"{int(mask.sum())}"
                )


def print_edge_info(data, name):
    print("\n" + "-" * 70)
    print(f"{name} EDGE INFORMATION")
    print("-" * 70)

    for edge_type in data.edge_types:

        store = data[edge_type]

        edge_index = store.edge_index

        print(
            f"\n{edge_type}:"
        )

        print(
            f"  Edges: {edge_index.shape[1]:,}"
        )

        if hasattr(store, "edge_time"):

            edge_time = store.edge_time

            print(
                f"  edge_time min: "
                f"{int(edge_time.min())}"
            )

            print(
                f"  edge_time max: "
                f"{int(edge_time.max())}"
            )


def transaction_degree_report(data):
    print("\n" + "-" * 70)
    print("TRANSACTION CONNECTIVITY")
    print("-" * 70)

    tx_count = data["transaction"].num_nodes

    total_degree = torch.zeros(
        tx_count,
        dtype=torch.long,
    )

    incoming_degree = torch.zeros(
        tx_count,
        dtype=torch.long,
    )

    outgoing_degree = torch.zeros(
        tx_count,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Count every edge touching transaction nodes
    # --------------------------------------------------------

    for edge_type in data.edge_types:

        src_type, relation, dst_type = edge_type

        edge_index = data[edge_type].edge_index

        if src_type == "transaction":

            src = edge_index[0]

            counts = torch.bincount(
                src,
                minlength=tx_count,
            )

            total_degree += counts

            if relation in [
                "SENT",
                "RECEIVED",
            ]:
                outgoing_degree += counts

        if dst_type == "transaction":

            dst = edge_index[1]

            counts = torch.bincount(
                dst,
                minlength=tx_count,
            )

            total_degree += counts

            if relation in [
                "SENT",
                "RECEIVED",
            ]:
                incoming_degree += counts

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"\nTransaction nodes: {tx_count:,}"
    )

    print(
        f"Minimum total degree: "
        f"{int(total_degree.min())}"
    )

    print(
        f"Maximum total degree: "
        f"{int(total_degree.max())}"
    )

    print(
        f"Mean total degree: "
        f"{float(total_degree.float().mean()):.4f}"
    )

    isolated = int(
        (total_degree == 0).sum()
    )

    print(
        f"Isolated transactions: "
        f"{isolated:,}"
    )

    print(
        f"Transactions with degree > 0: "
        f"{int((total_degree > 0).sum()):,}"
    )

    # --------------------------------------------------------
    # Individual relation connectivity
    # --------------------------------------------------------

    relations_to_check = [
        ("SENT", "transaction", "account"),
        ("RECEIVED", "account", "transaction"),
        ("SENT_BY", "account", "transaction"),
        ("RECEIVED_BY", "transaction", "account"),
        ("USES_DEVICE", "transaction", "device"),
        ("USES_IP", "transaction", "ip"),
        ("USES_MERCHANT", "transaction", "merchant"),
        ("LOGIN_FROM", "transaction", "session"),
    ]

    print("\nTransaction relation coverage:")

    for relation in relations_to_check:

        if relation not in data.edge_types:
            print(
                f"  {relation}: NOT PRESENT"
            )
            continue

        src_type, _, dst_type = relation

        edge_index = data[relation].edge_index

        if src_type == "transaction":

            unique_tx = torch.unique(
                edge_index[0]
            ).numel()

        elif dst_type == "transaction":

            unique_tx = torch.unique(
                edge_index[1]
            ).numel()

        else:
            unique_tx = 0

        coverage = (
            unique_tx / tx_count
        ) * 100

        print(
            f"  {relation}: "
            f"{unique_tx:,}/{tx_count:,} "
            f"({coverage:.2f}%)"
        )

    # --------------------------------------------------------
    # Fraud connectivity
    # --------------------------------------------------------

    if hasattr(
        data["transaction"],
        "y",
    ):

        y = data["transaction"].y

        fraud_mask = y == 1
        genuine_mask = y == 0

        print("\nFraud transaction connectivity:")

        fraud_degree = (
            total_degree[fraud_mask]
        )

        genuine_degree = (
            total_degree[genuine_mask]
        )

        if fraud_degree.numel() > 0:

            print(
                f"  Fraud count: "
                f"{fraud_degree.numel():,}"
            )

            print(
                f"  Fraud min degree: "
                f"{int(fraud_degree.min())}"
            )

            print(
                f"  Fraud max degree: "
                f"{int(fraud_degree.max())}"
            )

            print(
                f"  Fraud mean degree: "
                f"{float(fraud_degree.float().mean()):.4f}"
            )

        if genuine_degree.numel() > 0:

            print(
                f"  Genuine mean degree: "
                f"{float(genuine_degree.float().mean()):.4f}"
            )


def check_masks(data, name):
    print("\n" + "-" * 70)
    print(f"{name} MASK CHECK")
    print("-" * 70)

    if not hasattr(
        data["transaction"],
        "y",
    ):
        print("No transaction labels found.")
        return

    y = data["transaction"].y

    train_mask = getattr(
        data["transaction"],
        "train_mask",
        None,
    )

    val_mask = getattr(
        data["transaction"],
        "val_mask",
        None,
    )

    test_mask = getattr(
        data["transaction"],
        "test_mask",
        None,
    )

    if train_mask is not None:
        print(
            f"Train mask: "
            f"{int(train_mask.sum()):,}"
        )

    if val_mask is not None:
        print(
            f"Validation mask: "
            f"{int(val_mask.sum()):,}"
        )

    if test_mask is not None:
        print(
            f"Test mask: "
            f"{int(test_mask.sum()):,}"
        )

    # --------------------------------------------------------
    # Overlap checks
    # --------------------------------------------------------

    masks = {
        "train": train_mask,
        "validation": val_mask,
        "test": test_mask,
    }

    names = list(masks.keys())

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            a_name = names[i]
            b_name = names[j]

            a = masks[a_name]
            b = masks[b_name]

            if a is None or b is None:
                continue

            overlap = int(
                (a & b).sum()
            )

            print(
                f"{a_name} ∩ {b_name}: "
                f"{overlap}"
            )

    # --------------------------------------------------------
    # Label counts per mask
    # --------------------------------------------------------

    for mask_name, mask in masks.items():

        if mask is None:
            continue

        labels = y[mask]

        print(
            f"\n{mask_name} labels:"
        )

        print(
            f"  Transactions: "
            f"{len(labels):,}"
        )

        print(
            f"  Fraud: "
            f"{int((labels == 1).sum()):,}"
        )

        print(
            f"  Genuine: "
            f"{int((labels == 0).sum()):,}"
        )


def graph_summary(data, name):

    print("\n" + "=" * 70)
    print(f"{name} GRAPH SUMMARY")
    print("=" * 70)

    print(
        "\nNode types:"
    )

    for node_type in data.node_types:
        print(
            f"  {node_type}"
        )

    print(
        "\nEdge types:"
    )

    for edge_type in data.edge_types:
        print(
            f"  {edge_type}"
        )

    print(
        "\nTotal nodes:"
    )

    total_nodes = 0

    for node_type in data.node_types:

        count = data[node_type].num_nodes
        total_nodes += count

        print(
            f"  {node_type}: {count:,}"
        )

    print(
        f"  TOTAL: {total_nodes:,}"
    )

    print(
        "\nTotal edges:"
    )

    total_edges = 0

    for edge_type in data.edge_types:

        count = data[edge_type].edge_index.shape[1]

        total_edges += count

    print(
        f"  TOTAL: {total_edges:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FinGraph - GRAPHSAGE GRAPH AUDIT")
    print("=" * 70)

    for name, path in GRAPH_PATHS.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Graph not found:\n{path}"
            )

        print(
            f"\nLoading {name}:"
        )

        data = torch.load(
            path,
            weights_only=False,
        )

        graph_summary(
            data,
            name,
        )

        print_node_info(
            data,
            name,
        )

        print_edge_info(
            data,
            name,
        )

        check_masks(
            data,
            name,
        )

        transaction_degree_report(
            data,
        )

    print("\n" + "=" * 70)
    print("GRAPH AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()