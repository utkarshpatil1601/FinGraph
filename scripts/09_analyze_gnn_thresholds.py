"""
FinGraph - GraphSAGE Validation Threshold Analysis

Purpose:
    Analyze GraphSAGE classification thresholds using ONLY
    the validation period.

Important:
    - Test data is NOT loaded.
    - Test labels are NOT used.
    - The existing best GraphSAGE checkpoint is used.
    - Training normalization statistics are reused.
    - Threshold selection is based on validation F1.

Output:
    results/graphsage_validation_thresholds.csv
    results/graphsage_selected_threshold.json
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import csv

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

GNN_DIR = ROOT / "model" / "gnn"

# Make model/gnn importable from scripts/
if str(GNN_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_DIR))


# ============================================================
# GRAPH / MODEL IMPORTS
# ============================================================

from gnn_model import (
    add_reverse_relationships,
    create_model,
    validate_model_relations,
)


# ============================================================
# PATHS
# ============================================================

TRAIN_GRAPH_PATH = (
    ROOT
    / "data"
    / "graphs"
    / "fingraph_train.pt"
)

VAL_GRAPH_PATH = (
    ROOT
    / "data"
    / "graphs"
    / "fingraph_val.pt"
)

MODEL_PATH = (
    ROOT
    / "saved_models"
    / "gnn_fraud.pt"
)

OUTPUT_CSV = (
    ROOT
    / "results"
    / "graphsage_validation_thresholds.csv"
)

OUTPUT_JSON = (
    ROOT
    / "results"
    / "graphsage_selected_threshold.json"
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
    """
    Load a PyG graph, add reverse relationships,
    and validate all model relations.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Graph not found:\n{path}"
        )

    print(f"\nLoading graph:")
    print(path)

    data = torch.load(
        path,
        weights_only=False,
    )

    # Required because GraphSAGE expects reverse relations
    data = add_reverse_relationships(data)

    validate_model_relations(data)

    return data


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_with_training_stats(
    data,
    checkpoint,
):
    """
    Normalize every node type using statistics saved
    during GraphSAGE training.

    No validation/test statistics are calculated here.
    """

    if "normalization_stats" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'normalization_stats'."
        )

    stats = checkpoint["normalization_stats"]

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

        # Protection against zero standard deviation
        std = torch.where(
            std == 0,
            torch.ones_like(std),
            std,
        )

        if not hasattr(
            data[node_type],
            "x",
        ):
            continue

        data[node_type].x = (
            data[node_type].x.float()
            - mean
        ) / std

    return data


# ============================================================
# MOVE GRAPH TO DEVICE
# ============================================================

def move_graph_to_device(
    data,
    device,
):
    """
    Move node features, labels, masks and edges to
    CPU/GPU device.
    """

    # --------------------------------------------------------
    # NODE STORES
    # --------------------------------------------------------

    for node_type in data.node_types:

        store = data[node_type]

        if hasattr(store, "x"):
            store.x = store.x.to(device)

        if hasattr(store, "y"):
            store.y = store.y.to(device)

        if hasattr(store, "val_mask"):
            store.val_mask = (
                store.val_mask.to(device)
            )

        if hasattr(store, "train_mask"):
            store.train_mask = (
                store.train_mask.to(device)
            )

        if hasattr(store, "test_mask"):
            store.test_mask = (
                store.test_mask.to(device)
            )

    # --------------------------------------------------------
    # EDGE STORES
    # --------------------------------------------------------

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
# THRESHOLD METRICS
# ============================================================

def calculate_threshold_metrics(
    y_true,
    probabilities,
    threshold,
):
    """
    Calculate classification metrics at a given threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),

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

        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FinGraph - GRAPHSAGE VALIDATION THRESHOLD ANALYSIS")
    print("=" * 70)

    # ========================================================
    # DEVICE
    # ========================================================

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print("\nDevice:", device)

    # ========================================================
    # CHECK REQUIRED FILES
    # ========================================================

    required_files = [
        TRAIN_GRAPH_PATH,
        VAL_GRAPH_PATH,
        MODEL_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING BEST GRAPHSAGE CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_PATH,
        weights_only=False,
    )

    print(
        "Checkpoint:",
        MODEL_PATH,
    )

    print(
        "Best epoch:",
        checkpoint["best_epoch"],
    )

    print(
        "Best validation PR-AUC:",
        checkpoint["best_val_pr_auc"],
    )

    # ========================================================
    # LOAD VALIDATION GRAPH
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING VALIDATION GRAPH")
    print("=" * 70)

    val_data = load_graph(
        VAL_GRAPH_PATH
    )

    # ========================================================
    # VALIDATION MASK
    # ========================================================

    if not hasattr(
        val_data["transaction"],
        "val_mask",
    ):
        raise AttributeError(
            "Validation graph does not contain transaction.val_mask"
        )

    val_mask = (
        val_data[
            "transaction"
        ].val_mask
    )

    # ========================================================
    # VALIDATION LABELS
    # ========================================================

    y = (
        val_data[
            "transaction"
        ].y[
            val_mask
        ]
        .cpu()
        .numpy()
        .astype(int)
    )

    print(
        "\nValidation transactions:",
        len(y),
    )

    print(
        "Validation fraud:",
        int((y == 1).sum()),
    )

    print(
        "Validation genuine:",
        int((y == 0).sum()),
    )

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Validation set contains only one class. "
            "ROC-AUC and PR-AUC cannot be calculated."
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print("\n" + "=" * 70)
    print("APPLYING TRAINING NORMALIZATION STATISTICS")
    print("=" * 70)

    val_data = normalize_with_training_stats(
        val_data,
        checkpoint,
    )

    print("Normalization complete.")

    # ========================================================
    # MOVE TO DEVICE
    # ========================================================

    val_data = move_graph_to_device(
        val_data,
        device,
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING GRAPHSAGE MODEL")
    print("=" * 70)

    model = create_model(
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    # ========================================================
    # INITIALIZE LAZY LAYERS
    # ========================================================

    print(
        "\nInitializing model layers..."
    )

    with torch.no_grad():

        _ = model(
            val_data.x_dict,
            val_data.edge_index_dict,
        )

    # ========================================================
    # LOAD TRAINED PARAMETERS
    # ========================================================

    print(
        "Loading trained parameters..."
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print("Model ready.")

    # ========================================================
    # GENERATE VALIDATION PREDICTIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATING VALIDATION PREDICTIONS")
    print("=" * 70)

    with torch.no_grad():

        logits = model(
            val_data.x_dict,
            val_data.edge_index_dict,
        )

        probabilities = (
            torch.sigmoid(
                logits[val_mask]
            )
            .cpu()
            .numpy()
        )

    print(
        "Prediction generation complete."
    )

    # ========================================================
    # THRESHOLD-INDEPENDENT METRICS
    # ========================================================

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    print("\n" + "=" * 70)
    print("THRESHOLD-INDEPENDENT VALIDATION METRICS")
    print("=" * 70)

    print(
        f"ROC-AUC: {roc_auc:.6f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.6f}"
    )

    # ========================================================
    # THRESHOLD SWEEP
    # ========================================================

    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    thresholds = np.round(
        np.arange(
            0.05,
            0.951,
            0.01,
        ),
        2,
    )

    results = []

    for threshold in thresholds:

        metrics = calculate_threshold_metrics(
            y_true=y,
            probabilities=probabilities,
            threshold=float(threshold),
        )

        results.append(metrics)

    # ========================================================
    # BEST THRESHOLD BY F1
    # ========================================================

    best = max(
        results,
        key=lambda row: (
            row["f1"],
            row["precision"],
            -row["false_positive"],
        ),
    )

    print("\n" + "=" * 70)
    print("BEST VALIDATION THRESHOLD")
    print("=" * 70)

    print(
        f"Selection criterion: Maximum validation F1"
    )

    print(
        f"Threshold:  {best['threshold']:.2f}"
    )

    print(
        f"Accuracy:   {best['accuracy']:.6f}"
    )

    print(
        f"Precision:  {best['precision']:.6f}"
    )

    print(
        f"Recall:     {best['recall']:.6f}"
    )

    print(
        f"F1:         {best['f1']:.6f}"
    )

    print(
        f"TN:         {best['true_negative']}"
    )

    print(
        f"FP:         {best['false_positive']}"
    )

    print(
        f"FN:         {best['false_negative']}"
    )

    print(
        f"TP:         {best['true_positive']}"
    )

    # ========================================================
    # SELECTED THRESHOLDS TABLE
    # ========================================================

    interesting_thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    print("\n" + "=" * 70)
    print("SELECTED THRESHOLD RESULTS")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FP':<10}"
        f"{'FN':<10}"
        f"{'TP':<10}"
    )

    print("-" * 78)

    for target in interesting_thresholds:

        closest = min(
            results,
            key=lambda row: abs(
                row["threshold"] - target
            ),
        )

        print(
            f"{closest['threshold']:<12.2f}"
            f"{closest['precision']:<12.4f}"
            f"{closest['recall']:<12.4f}"
            f"{closest['f1']:<12.4f}"
            f"{closest['false_positive']:<10}"
            f"{closest['false_negative']:<10}"
            f"{closest['true_positive']:<10}"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nThreshold results saved:"
    )

    print(
        OUTPUT_CSV
    )

    # ========================================================
    # SAVE SELECTED THRESHOLD JSON
    # ========================================================

    selected_result = {
        "model": "FinGraph GraphSAGE",

        "selection_method":
            "Maximum validation F1",

        "threshold":
            float(best["threshold"]),

        "validation_transactions":
            int(len(y)),

        "validation_fraud":
            int((y == 1).sum()),

        "validation_genuine":
            int((y == 0).sum()),

        "validation_roc_auc":
            float(roc_auc),

        "validation_pr_auc":
            float(pr_auc),

        "metrics_at_selected_threshold":
            best,

        "test_set_used":
            False,
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            selected_result,
            f,
            indent=2,
        )

    print(
        "Selected threshold saved:"
    )

    print(
        OUTPUT_JSON
    )

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The test graph was NOT loaded."
    )

    print(
        "The test labels were NOT used."
    )

    print(
        "Threshold selection used validation data only."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()