"""
FinGraph - First-pass GraphSAGE Training

Training setup:
    Train graph:
        data/graphs/fingraph_train.pt

    Validation graph:
        data/graphs/fingraph_val.pt

    Final test graph:
        data/graphs/fingraph_test.pt

The test graph is NOT used during training.

Metrics:
    Accuracy
    Precision
    Recall
    F1
    ROC-AUC
    PR-AUC

Because fraud is highly imbalanced, BCEWithLogitsLoss uses
a positive-class weight derived only from the training graph.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from gnn_model import (
    MODEL_RELATIONS,
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

VAL_GRAPH_PATH = (
    ROOT
    / "data"
    / "graphs"
    / "fingraph_val.pt"
)

MODEL_DIR = (
    ROOT
    / "saved_models"
)

RESULT_DIR = (
    ROOT
    / "results"
)

MODEL_PATH = (
    MODEL_DIR
    / "gnn_fraud.pt"
)

RESULT_PATH = (
    RESULT_DIR
    / "graphsage_phase2_results.json"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

HIDDEN_CHANNELS = 32

DROPOUT = 0.30

LEARNING_RATE = 0.003

WEIGHT_DECAY = 1e-4

EPOCHS = 20


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():

        print(
            "CUDA available: using GPU"
        )

        return torch.device("cuda")

    print(
        "CUDA unavailable: using CPU"
    )

    return torch.device("cpu")


# ============================================================
# LOAD GRAPH
# ============================================================

def load_graph(path: Path):

    if not path.exists():

        raise FileNotFoundError(
            f"Graph not found:\n{path}"
        )

    print(
        f"\nLoading graph:\n{path}"
    )

    data = torch.load(
        path,
        weights_only=False,
    )

    # --------------------------------------------------------
    # First create reverse relations from the forward
    # relations stored in the graph.
    # --------------------------------------------------------

    data = add_reverse_relationships(
        data
    )

    # --------------------------------------------------------
    # Now validate both the original and generated
    # relations.
    # --------------------------------------------------------

    validate_model_relations(
        data
    )

    return data


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def calculate_normalization_stats(
    train_data
):
    """
    Calculate feature normalization only from the training graph.
    """

    stats = {}

    for node_type in train_data.node_types:

        x = train_data[node_type].x.float()

        mean = x.mean(
            dim=0
        )

        std = x.std(
            dim=0,
            unbiased=False,
        )

        std = torch.where(
            std < 1e-8,
            torch.ones_like(std),
            std,
        )

        stats[node_type] = {
            "mean": mean,
            "std": std,
        }

    return stats


def normalize_data(
    data,
    stats,
):
    """
    Apply training-derived normalization to a graph.
    """

    for node_type in data.node_types:

        if node_type not in stats:
            continue

        mean = stats[node_type]["mean"]

        std = stats[node_type]["std"]

        data[node_type].x = (
            data[node_type].x.float()
            - mean
        ) / std

    return data


# ============================================================
# GRAPH TO DEVICE
# ============================================================

def move_graph_to_device(
    data,
    device,
):

    for node_type in data.node_types:

        store = data[node_type]

        if hasattr(
            store,
            "x",
        ):

            store.x = store.x.to(
                device
            )

        if hasattr(
            store,
            "y",
        ):

            store.y = store.y.to(
                device
            )

        if hasattr(
            store,
            "train_mask",
        ):

            store.train_mask = (
                store.train_mask.to(
                    device
                )
            )

        if hasattr(
            store,
            "val_mask",
        ):

            store.val_mask = (
                store.val_mask.to(
                    device
                )
            )

        if hasattr(
            store,
            "test_mask",
        ):

            store.test_mask = (
                store.test_mask.to(
                    device
                )
            )

    for edge_type in data.edge_types:

        store = data[edge_type]

        if hasattr(
            store,
            "edge_index",
        ):

            store.edge_index = (
                store.edge_index.to(
                    device
                )
            )

        if hasattr(
            store,
            "edge_time",
        ):

            store.edge_time = (
                store.edge_time.to(
                    device
                )
            )

    return data


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):
    """
    Calculate binary fraud metrics.
    """

    y_true = np.asarray(
        y_true
    ).astype(int)

    probabilities = np.asarray(
        probabilities
    ).astype(float)

    predictions = (
        probabilities >= 0.5
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

    # ROC-AUC needs both classes
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

    return metrics


# ============================================================
# EVALUATE GRAPH
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    data,
    mask_name,
):
    """
    Evaluate transaction predictions using the requested mask.
    """

    model.eval()

    logits = model(
        data.x_dict,
        data.edge_index_dict,
    )

    mask = getattr(
        data["transaction"],
        mask_name,
    )

    y = (
        data["transaction"]
        .y[mask]
        .detach()
        .cpu()
        .numpy()
    )

    probabilities = (
        torch.sigmoid(
            logits[mask]
        )
        .detach()
        .cpu()
        .numpy()
    )

    metrics = calculate_metrics(
        y,
        probabilities,
    )

    return (
        metrics,
        y,
        probabilities,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FinGraph - FIRST-PASS GRAPHSAGE")
    print("=" * 70)

    set_seed(
        SEED
    )

    device = get_device()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD TRAIN GRAPH
    # ========================================================

    train_data = load_graph(
        TRAIN_GRAPH_PATH
    )

    print(
        "\nTrain transaction nodes:",
        train_data["transaction"].num_nodes,
    )

    print(
        "Train fraud:",
        int(
            (
                train_data[
                    "transaction"
                ].y
                == 1
            ).sum()
        ),
    )

    print(
        "Train genuine:",
        int(
            (
                train_data[
                    "transaction"
                ].y
                == 0
            ).sum()
        ),
    )

    # ========================================================
    # LOAD VALIDATION GRAPH
    # ========================================================

    val_data = load_graph(
        VAL_GRAPH_PATH
    )

    print(
        "\nValidation target transactions:",
        int(
            val_data[
                "transaction"
            ].val_mask.sum()
        ),
    )

    print(
        "Validation fraud:",
        int(
            (
                val_data[
                    "transaction"
                ].y[
                    val_data[
                        "transaction"
                    ].val_mask
                ]
                == 1
            ).sum()
        ),
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print(
        "\nCalculating normalization statistics "
        "from TRAIN graph only..."
    )

    normalization_stats = (
        calculate_normalization_stats(
            train_data
        )
    )

    train_data = normalize_data(
        train_data,
        normalization_stats,
    )

    val_data = normalize_data(
        val_data,
        normalization_stats,
    )

    # ========================================================
    # MOVE TO CPU / GPU
    # ========================================================

    train_data = move_graph_to_device(
        train_data,
        device,
    )

    val_data = move_graph_to_device(
        val_data,
        device,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model(
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    # ========================================================
    # INITIALIZE LAZY LAYERS
    # ========================================================

    print(
        "\nInitializing model..."
    )

    with torch.no_grad():

        _ = model(
            train_data.x_dict,
            train_data.edge_index_dict,
        )

    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    train_y = (
        train_data[
            "transaction"
        ].y[
            train_data[
                "transaction"
            ].train_mask
        ]
    )

    positive_count = int(
        (train_y == 1).sum()
    )

    negative_count = int(
        (train_y == 0).sum()
    )

    if positive_count == 0:

        raise ValueError(
            "Training graph contains no fraud "
            "transactions."
        )

    pos_weight = (
        negative_count
        / positive_count
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CLASS IMBALANCE"
    )

    print(
        "Positive fraud:",
        positive_count
    )

    print(
        "Negative genuine:",
        negative_count
    )

    print(
        "Positive-class weight:",
        f"{pos_weight:.2f}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    best_val_pr_auc = -1.0
    best_epoch = -1
    history = []

    print()
    print(
        "=" * 70
    )
    print(
        "TRAINING"
    )
    print(
        "=" * 70
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad()

        logits = model(
            train_data.x_dict,
            train_data.edge_index_dict,
        )

        mask = (
            train_data[
                "transaction"
            ].train_mask
        )

        y = (
            train_data[
                "transaction"
            ].y[mask]
            .float()
        )

        train_logits = (
            logits[mask]
        )

        loss = criterion(
            train_logits,
            y,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=2.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Training metrics
        # ----------------------------------------------------

        with torch.no_grad():

            train_probabilities = (
                torch.sigmoid(
                    train_logits
                )
                .cpu()
                .numpy()
            )

            train_y_np = (
                y
                .cpu()
                .numpy()
            )

            train_metrics = (
                calculate_metrics(
                    train_y_np,
                    train_probabilities,
                )
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics, _, _ = evaluate(
            model,
            val_data,
            "val_mask",
        )

        row = {
            "epoch": epoch,
            "train_loss": float(
                loss.item()
            ),
            "train_pr_auc":
                train_metrics["pr_auc"],
            "train_roc_auc":
                train_metrics["roc_auc"],
            "val_pr_auc":
                val_metrics["pr_auc"],
            "val_roc_auc":
                val_metrics["roc_auc"],
            "val_precision":
                val_metrics["precision"],
            "val_recall":
                val_metrics["recall"],
            "val_f1":
                val_metrics["f1"],
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {loss.item():.5f} | "
            f"Train PR-AUC "
            f"{train_metrics['pr_auc']:.4f} | "
            f"Val PR-AUC "
            f"{val_metrics['pr_auc']:.4f} | "
            f"Val Recall "
            f"{val_metrics['recall']:.4f} | "
            f"Val F1 "
            f"{val_metrics['f1']:.4f}"
        )

        # ----------------------------------------------------
        # Save best model by validation PR-AUC
        # ----------------------------------------------------

        current_val_pr_auc = (
            val_metrics["pr_auc"]
        )

        if (
            current_val_pr_auc is not None
            and current_val_pr_auc
            > best_val_pr_auc
        ):

            best_val_pr_auc = (
                current_val_pr_auc
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "hidden_channels":
                        HIDDEN_CHANNELS,
                    "dropout":
                        DROPOUT,
                    "seed":
                        SEED,
                    "best_epoch":
                        best_epoch,
                    "best_val_pr_auc":
                        best_val_pr_auc,
                    "normalization_stats": {
                        node_type: {
                            "mean":
                                values["mean"]
                                .cpu(),
                            "std":
                                values["std"]
                                .cpu(),
                        }
                        for node_type, values
                        in normalization_stats.items()
                    },
                },
                MODEL_PATH,
            )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "FINAL VALIDATION"
    )
    print(
        "=" * 70
    )

    checkpoint = torch.load(
        MODEL_PATH,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    val_metrics, _, _ = evaluate(
        model,
        val_data,
        "val_mask",
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        f"Validation PR-AUC: "
        f"{val_metrics['pr_auc']}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{val_metrics['roc_auc']}"
    )

    print(
        f"Validation Precision: "
        f"{val_metrics['precision']}"
    )

    print(
        f"Validation Recall: "
        f"{val_metrics['recall']}"
    )

    print(
        f"Validation F1: "
        f"{val_metrics['f1']}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "model": "FinGraph GraphSAGE",
        "phase": "Phase 2",
        "seed": SEED,
        "hidden_channels":
            HIDDEN_CHANNELS,
        "dropout":
            DROPOUT,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "epochs":
            EPOCHS,
        "train_fraud":
            positive_count,
        "train_genuine":
            negative_count,
        "positive_class_weight":
            pos_weight,
        "best_epoch":
            best_epoch,
        "best_validation_pr_auc":
            best_val_pr_auc,
        "final_validation_metrics":
            val_metrics,
        "history":
            history,
        "model_path":
            str(MODEL_PATH),
    }

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "GRAPHSAGE TRAINING COMPLETE"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "Saved model:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "Saved results:"
    )

    print(
        RESULT_PATH
    )


if __name__ == "__main__":
    main()