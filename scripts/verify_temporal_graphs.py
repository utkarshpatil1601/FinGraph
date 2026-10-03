from pathlib import Path
import torch


ROOT = Path(__file__).resolve().parents[1]

GRAPH_FILES = {
    "TRAIN": ROOT / "data" / "graphs" / "fingraph_train.pt",
    "VALIDATION": ROOT / "data" / "graphs" / "fingraph_val.pt",
    "TEST": ROOT / "data" / "graphs" / "fingraph_test.pt",
}


def verify_graph(name, path):
    print()
    print("=" * 60)
    print(f"{name} GRAPH")
    print("=" * 60)

    if not path.exists():
        raise FileNotFoundError(path)

    data = torch.load(
        path,
        weights_only=False
    )

    mask_train = data["transaction"].train_mask
    mask_val = data["transaction"].val_mask
    mask_test = data["transaction"].test_mask

    steps = data["transaction"].step

    print("File:", path)
    print(
        "Transactions:",
        len(steps)
    )

    print()
    print("Mask counts:")
    print(
        "  Train:",
        int(mask_train.sum())
    )
    print(
        "  Validation:",
        int(mask_val.sum())
    )
    print(
        "  Test:",
        int(mask_test.sum())
    )

    print()
    print("Step ranges represented in graph:")
    print(
        "  Min:",
        int(steps.min())
    )
    print(
        "  Max:",
        int(steps.max())
    )

    # --------------------------------------------------------
    # No overlap
    # --------------------------------------------------------

    train_val = int(
        (mask_train & mask_val).sum()
    )

    train_test = int(
        (mask_train & mask_test).sum()
    )

    val_test = int(
        (mask_val & mask_test).sum()
    )

    print()
    print("Mask overlap:")
    print("  Train/Val :", train_val)
    print("  Train/Test:", train_test)
    print("  Val/Test  :", val_test)

    if (
        train_val != 0
        or train_test != 0
        or val_test != 0
    ):
        raise ValueError(
            f"{name}: mask overlap detected."
        )

    # --------------------------------------------------------
    # Expected composition
    # --------------------------------------------------------

    if name == "TRAIN":
        if (
            mask_train.sum() != len(steps)
            or mask_val.sum() != 0
            or mask_test.sum() != 0
        ):
            raise ValueError(
                "TRAIN graph masks are incorrect."
            )

    elif name == "VALIDATION":
        if (
            mask_train.sum() == 0
            or mask_val.sum() == 0
            or mask_test.sum() != 0
        ):
            raise ValueError(
                "VALIDATION graph masks are incorrect."
            )

    elif name == "TEST":
        if (
            mask_train.sum() == 0
            or mask_val.sum() != 0
            or mask_test.sum() == 0
        ):
            raise ValueError(
                "TEST graph masks are incorrect."
            )

    # --------------------------------------------------------
    # Edge-time validation
    # --------------------------------------------------------

    temporal_edges = [
        ("account", "SENT", "transaction"),
        ("transaction", "SENT_BY", "account"),
        ("account", "RECEIVED", "transaction"),
        ("transaction", "RECEIVED_BY", "account"),
        ("account", "CONNECTED_TO", "account"),
    ]

    for edge_type in temporal_edges:

        store = data[edge_type]

        if not hasattr(store, "edge_time"):
            raise ValueError(
                f"{name}: missing edge_time for {edge_type}"
            )

        if len(store.edge_time) != store.edge_index.shape[1]:
            raise ValueError(
                f"{name}: edge_time mismatch for {edge_type}"
            )

    print()
    print(f"{name} VERIFICATION PASSED")


def main():
    print("=" * 60)
    print("FinGraph TEMPORAL GRAPH VERIFICATION")
    print("=" * 60)

    for name, path in GRAPH_FILES.items():
        verify_graph(name, path)

    print()
    print("=" * 60)
    print("ALL TEMPORAL GRAPHS VERIFIED")
    print("=" * 60)


if __name__ == "__main__":
    main()