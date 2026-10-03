from pathlib import Path

import numpy as np
import torch


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "graphs"
    / "fingraph_graph.pt"
)


# ============================================================
# HELPERS
# ============================================================

def check_tensor(name, tensor):
    print(f"\n{name}")
    print("-" * 60)

    print("Shape:", tuple(tensor.shape))
    print("Dtype:", tensor.dtype)

    if tensor.numel() == 0:
        print("Empty tensor")
        return

    if tensor.is_floating_point():
        print(
            "Finite:",
            bool(torch.isfinite(tensor).all())
        )

        print(
            "Min:",
            float(tensor.min())
        )

        print(
            "Max:",
            float(tensor.max())
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FinGraph GRAPH VERIFICATION")
    print("=" * 60)

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Graph file not found:\n{GRAPH_PATH}"
        )

    print("\nLoading:")
    print(GRAPH_PATH)

    # Explicitly use weights_only=False because HeteroData
    # is a PyG object rather than a plain state dictionary.
    data = torch.load(
        GRAPH_PATH,
        weights_only=False
    )

    # ========================================================
    # NODE TYPES
    # ========================================================

    expected_nodes = {
        "transaction",
        "account",
        "device",
        "ip",
        "merchant",
        "session",
    }

    print("\n" + "=" * 60)
    print("NODE TYPES")
    print("=" * 60)

    print(data.node_types)

    actual_nodes = set(data.node_types)

    missing_nodes = expected_nodes - actual_nodes

    if missing_nodes:
        raise ValueError(
            f"Missing node types: {missing_nodes}"
        )

    for node_type in data.node_types:

        store = data[node_type]

        print(
            f"{node_type:15s} -> "
            f"{store.num_nodes:,} nodes"
        )

        if hasattr(store, "x"):
            print(
                f"{'':15s}    Features: "
                f"{tuple(store.x.shape)}"
            )

        if hasattr(store, "y"):
            print(
                f"{'':15s}    Labels: "
                f"{tuple(store.y.shape)}"
            )

    # ========================================================
    # EDGE TYPES
    # ========================================================

    print("\n" + "=" * 60)
    print("EDGE TYPES")
    print("=" * 60)

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

    for edge_type in data.edge_types:

        edge_index = data[
            edge_type
        ].edge_index

        print(
            f"{str(edge_type):65s} -> "
            f"{edge_index.shape[1]:,} edges"
        )

        if edge_index.shape[0] != 2:
            raise ValueError(
                f"Invalid edge shape for {edge_type}: "
                f"{edge_index.shape}"
            )

    missing_edges = [
        edge
        for edge in required_edges
        if edge not in data.edge_types
    ]

    if missing_edges:
        raise ValueError(
            f"Missing required edges: {missing_edges}"
        )

    # ========================================================
    # TRANSACTION FEATURES
    # ========================================================

    print("\n" + "=" * 60)
    print("TRANSACTION DATA")
    print("=" * 60)

    tx_x = data["transaction"].x
    tx_y = data["transaction"].y
    tx_step = data["transaction"].step

    check_tensor(
        "Transaction features",
        tx_x
    )

    print(
        "\nTransaction labels:",
        torch.unique(tx_y).tolist()
    )

    print(
        "Fraud:",
        int((tx_y == 1).sum())
    )

    print(
        "Genuine:",
        int((tx_y == 0).sum())
    )

    print(
        "\nStep minimum:",
        int(tx_step.min())
    )

    print(
        "Step maximum:",
        int(tx_step.max())
    )

    print(
        "Unique steps:",
        tx_step.unique().numel()
    )

    # ========================================================
    # TEMPORAL MASKS
    # ========================================================

    print("\n" + "=" * 60)
    print("TEMPORAL MASKS")
    print("=" * 60)

    train_mask = data[
        "transaction"
    ].train_mask

    val_mask = data[
        "transaction"
    ].val_mask

    test_mask = data[
        "transaction"
    ].test_mask

    print(
        "Train:",
        int(train_mask.sum()),
        f"({train_mask.float().mean() * 100:.2f}%)"
    )

    print(
        "Validation:",
        int(val_mask.sum()),
        f"({val_mask.float().mean() * 100:.2f}%)"
    )

    print(
        "Test:",
        int(test_mask.sum()),
        f"({test_mask.float().mean() * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------

    overlap_train_val = (
        train_mask & val_mask
    ).sum()

    overlap_train_test = (
        train_mask & test_mask
    ).sum()

    overlap_val_test = (
        val_mask & test_mask
    ).sum()

    print(
        "\nTrain/Val overlap:",
        int(overlap_train_val)
    )

    print(
        "Train/Test overlap:",
        int(overlap_train_test)
    )

    print(
        "Val/Test overlap:",
        int(overlap_val_test)
    )

    if (
        overlap_train_val > 0
        or overlap_train_test > 0
        or overlap_val_test > 0
    ):
        raise ValueError(
            "Temporal masks overlap."
        )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    coverage = (
        train_mask
        | val_mask
        | test_mask
    )

    uncovered = (~coverage).sum()

    print(
        "\nUnassigned transactions:",
        int(uncovered)
    )

    if uncovered > 0:
        raise ValueError(
            "Some transactions are not assigned "
            "to train/validation/test."
        )

    # ========================================================
    # STEP RANGES
    # ========================================================

    print("\n" + "=" * 60)
    print("TEMPORAL STEP RANGES")
    print("=" * 60)

    for name, mask in [
        ("Train", train_mask),
        ("Validation", val_mask),
        ("Test", test_mask),
    ]:

        selected_steps = tx_step[mask]

        print(
            f"{name:12s}: "
            f"{int(selected_steps.min())} -> "
            f"{int(selected_steps.max())}"
        )

    # ========================================================
    # FRAUD BY SPLIT
    # ========================================================

    print("\n" + "=" * 60)
    print("FRAUD DISTRIBUTION BY SPLIT")
    print("=" * 60)

    for name, mask in [
        ("Train", train_mask),
        ("Validation", val_mask),
        ("Test", test_mask),
    ]:

        labels = tx_y[mask]

        total = len(labels)
        fraud = int((labels == 1).sum())
        genuine = int((labels == 0).sum())

        fraud_rate = (
            fraud / total * 100
            if total > 0
            else 0
        )

        print(
            f"{name:12s}: "
            f"Total={total:,}, "
            f"Fraud={fraud:,}, "
            f"Genuine={genuine:,}, "
            f"Fraud%={fraud_rate:.4f}%"
        )

    # ========================================================
    # ACCOUNT DATA
    # ========================================================

    print("\n" + "=" * 60)
    print("ACCOUNT DATA")
    print("=" * 60)

    account_x = data["account"].x
    account_y = data["account"].y

    print(
        "Account features:",
        tuple(account_x.shape)
    )

    print(
        "Account labels:",
        torch.unique(account_y).tolist()
    )

    print(
        "Fraud-labelled accounts:",
        int((account_y == 1).sum())
    )

    print(
        "Normal accounts:",
        int((account_y == 0).sum())
    )

    # ========================================================
    # NAN / INF CHECK
    # ========================================================

    print("\n" + "=" * 60)
    print("NUMERICAL VALIDATION")
    print("=" * 60)

    for node_type in data.node_types:

        if hasattr(
            data[node_type],
            "x"
        ):

            x = data[node_type].x

            if x.is_floating_point():

                finite = bool(
                    torch.isfinite(x).all()
                )

                print(
                    f"{node_type:15s}: "
                    f"finite={finite}"
                )

                if not finite:
                    raise ValueError(
                        f"NaN/Inf detected in "
                        f"{node_type} features."
                    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("GRAPH VERIFICATION PASSED")
    print("=" * 60)

    print("\nThe saved HeteroData graph is structurally valid.")


if __name__ == "__main__":
    main()