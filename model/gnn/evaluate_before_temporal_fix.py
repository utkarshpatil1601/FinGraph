"""
FinGraph - GraphSAGE Test Evaluation

Loads the best GraphSAGE checkpoint selected using the
validation PR-AUC and evaluates it ONCE on the test period.

The test graph contains:
    historical context: steps 1 -> 378
    test targets:       steps 379 -> 741

The test labels are never used for training or checkpoint
selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from gnn_model import (
    add_reverse_relationships,
    create_model,
    validate_model_relations,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_GRAPH_PATH = (
    ROOT
    / "data"
    / "graphs"
    / "fingraph_train.pt"
)

TEST_GRAPH_PATH = (
    ROOT
    / "data"
    / "graphs"
    / "fingraph_test.pt"
)

MODEL_PATH = (
    ROOT
    / "saved_models"
    / "gnn_fraud.pt"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "graphsage_test_results.json"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

HIDDEN_CHANNELS = 32
DROPOUT = 0.30


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Graph not found:\n{path}"
        )

    print(
        f"\nLoading:\n{path}"
    )

    data = torch.load(
        path,
        weights_only=False,
    )

    data = add_reverse_relationships(
        data
    )

    validate_model_relations(
        data
    )

    return data


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_with_training_stats(
    data,
    checkpoint,
):

    stats = (
        checkpoint[
            "normalization_stats"
        ]
    )

    for node_type in data.node_types:

        if node_type not in stats:
            continue

        mean = (
            stats[node_type]["mean"]
            .float()
        )

        std = (
            stats[node_type]["std"]
            .float()
        )

        data[node_type].x = (
            data[node_type].x.float()
            - mean
        ) / std

    return data


# ============================================================
# MOVE GRAPH
# ============================================================

def move_graph_to_device(
    data,
    device,
):

    for node_type in data.node_types:

        store = data[node_type]

        if hasattr(store, "x"):
            store.x = store.x.to(device)

        if hasattr(store, "y"):
            store.y = store.y.to(device)

        if hasattr(store, "test_mask"):
            store.test_mask = (
                store.test_mask.to(device)
            )

    for edge_type in data.edge_types:

        store = data[edge_type]

        if hasattr(store, "edge_index"):

            store.edge_index = (
                store.edge_index.to(device)
            )

        if hasattr(store, "edge_time"):

            store.edge_time = (
                store.edge_time.to(device)
            )

    return data


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

    metrics = {}

    metrics["accuracy"] = float(
        accuracy_score(
            y_true,
            predictions,
        )
    )

    metrics["precision"] = float(
        precision_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    metrics["recall"] = float(
        recall_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    metrics["f1"] = float(
        f1_score(
            y_true,
            predictions,
            zero_division=0,
        )
    )

    if len(
        np.unique(y_true)
    ) == 2:

        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        metrics["pr_auc"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    else:

        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_positive"] = int(tp)

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FinGraph - GRAPHSAGE TEST EVALUATION")
    print("=" * 70)

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(
        "\nDevice:",
        device
    )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        weights_only=False,
    )

    print(
        "\nBest validation epoch:",
        checkpoint["best_epoch"]
    )

    print(
        "Best validation PR-AUC:",
        checkpoint["best_val_pr_auc"]
    )

    # ========================================================
    # LOAD TEST GRAPH
    # ========================================================

    test_data = load_graph(
        TEST_GRAPH_PATH
    )

    # ========================================================
    # TEST TARGETS
    # ========================================================

    test_mask = (
        test_data[
            "transaction"
        ].test_mask
    )

    y = (
        test_data[
            "transaction"
        ].y[
            test_mask
        ]
        .cpu()
        .numpy()
        .astype(int)
    )

    print()
    print(
        "Test transactions:",
        len(y)
    )

    print(
        "Test fraud:",
        int((y == 1).sum())
    )

    print(
        "Test genuine:",
        int((y == 0).sum())
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print(
        "\nApplying TRAIN normalization statistics..."
    )

    test_data = (
        normalize_with_training_stats(
            test_data,
            checkpoint,
        )
    )

    # ========================================================
    # MOVE TO DEVICE
    # ========================================================

    test_data = move_graph_to_device(
        test_data,
        device,
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = create_model(
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    # Initialize LazyLinear layers
    with torch.no_grad():

        _ = model(
            test_data.x_dict,
            test_data.edge_index_dict,
        )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ========================================================
    # PREDICT
    # ========================================================

    print(
        "\nGenerating test predictions..."
    )

    with torch.no_grad():

        logits = model(
            test_data.x_dict,
            test_data.edge_index_dict,
        )

        probabilities = (
            torch.sigmoid(
                logits[test_mask]
            )
            .cpu()
            .numpy()
        )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        y,
        probabilities,
        threshold=0.5,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("TEST RESULTS")
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
    # TOP PREDICTIONS
    # ========================================================

    test_indices = (
        torch.where(
            test_mask
        )[0]
        .cpu()
        .numpy()
    )

    top_order = np.argsort(
        probabilities
    )[::-1]

    top_n = min(
        20,
        len(top_order),
    )

    print()
    print(
        "Top test fraud probabilities:"
    )

    top_predictions = []

    transaction_ids = (
        test_data[
            "transaction"
        ].transaction_id
    )

    steps = (
        test_data[
            "transaction"
        ].step
        .cpu()
        .numpy()
    )

    for position in top_order[:top_n]:

        original_index = int(
            test_indices[position]
        )

        probability = float(
            probabilities[position]
        )

        label = int(
            test_data[
                "transaction"
            ].y[
                original_index
            ]
            .cpu()
            .item()
        )

        transaction_id = (
            transaction_ids[
                original_index
            ]
        )

        step = int(
            steps[
                original_index
            ]
        )

        print(
            f"  {transaction_id} | "
            f"step={step} | "
            f"probability={probability:.6f} | "
            f"label={label}"
        )

        top_predictions.append(
            {
                "transaction_id":
                    transaction_id,
                "step":
                    step,
                "probability":
                    probability,
                "actual_label":
                    label,
            }
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "model":
            "FinGraph GraphSAGE",
        "best_epoch":
            int(
                checkpoint["best_epoch"]
            ),
        "best_validation_pr_auc":
            float(
                checkpoint[
                    "best_val_pr_auc"
                ]
            ),
        "test_transactions":
            int(len(y)),
        "test_fraud":
            int((y == 1).sum()),
        "test_genuine":
            int((y == 0).sum()),
        "metrics":
            metrics,
        "top_predictions":
            top_predictions,
    }

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

    print()
    print(
        "Saved:",
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()