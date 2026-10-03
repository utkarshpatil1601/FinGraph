from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = ROOT / "results" / "baseline_phase2_test.json"
GNN_PATH = ROOT / "results" / "graphsage_test_results.json"

OUTPUT_JSON = ROOT / "results" / "phase2_model_comparison.json"
OUTPUT_CSV = ROOT / "results" / "phase2_model_comparison.csv"


print("=" * 70)
print("FinGraph - PHASE 2 MODEL COMPARISON")
print("=" * 70)


# -------------------------------------------------------------------
# LOAD RESULTS
# -------------------------------------------------------------------

print("\nLoading baseline results...")
with open(BASELINE_PATH, "r", encoding="utf-8") as f:
    baseline = json.load(f)

print("Loading GraphSAGE results...")
with open(GNN_PATH, "r", encoding="utf-8") as f:
    gnn = json.load(f)


# -------------------------------------------------------------------
# DISPLAY RAW STRUCTURE
# -------------------------------------------------------------------

print("\nBaseline result keys:")
print(list(baseline.keys()))

print("\nGraphSAGE result keys:")
print(list(gnn.keys()))


# -------------------------------------------------------------------
# HELPER
# -------------------------------------------------------------------

def get_metric(data, name):
    """
    Handles slightly different JSON structures safely.
    """
    if name in data:
        return data[name]

    for parent in ["metrics", "test_metrics", "results"]:
        if isinstance(data.get(parent), dict):
            if name in data[parent]:
                return data[parent][name]

    return None


metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]


# -------------------------------------------------------------------
# BUILD COMPARISON
# -------------------------------------------------------------------

rows = []

for metric in metrics:
    rows.append({
        "metric": metric.upper(),
        "Random Forest": get_metric(baseline, metric),
        "GraphSAGE": get_metric(gnn, metric),
    })

comparison_df = pd.DataFrame(rows)


# -------------------------------------------------------------------
# PRINT
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST SET COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# -------------------------------------------------------------------
# CONFUSION MATRICES
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)

def get_confusion_matrix(data):
    if "confusion_matrix" in data:
        return data["confusion_matrix"]

    if isinstance(data.get("metrics"), dict):
        return data["metrics"].get("confusion_matrix")

    return None


baseline_cm = baseline["metrics"].get("confusion_matrix")

if baseline_cm is None:
    baseline_cm = {
        "TN": baseline["metrics"].get("true_negative"),
        "FP": baseline["metrics"].get("false_positive"),
        "FN": baseline["metrics"].get("false_negative"),
        "TP": baseline["metrics"].get("true_positive"),
    }

gnn_cm = gnn["metrics"].get("confusion_matrix")

if gnn_cm is None:
    gnn_cm = {
        "TN": gnn["metrics"].get("true_negative"),
        "FP": gnn["metrics"].get("false_positive"),
        "FN": gnn["metrics"].get("false_negative"),
        "TP": gnn["metrics"].get("true_positive"),
    }

print("\nRandom Forest:")
print(baseline_cm)

print("\nGraphSAGE:")
print(gnn_cm)


# -------------------------------------------------------------------
# SAMPLE INFORMATION
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST SET INFORMATION")
print("=" * 70)

for name, data in [
    ("Random Forest", baseline),
    ("GraphSAGE", gnn)
]:
    print(f"\n{name}:")

    for key in [
        "test_transactions",
        "test_fraud",
        "test_genuine"
    ]:
        value = data.get(key)

        if value is not None:
            print(f"  {key}: {value}")


# -------------------------------------------------------------------
# SAVE CSV
# -------------------------------------------------------------------

comparison_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# -------------------------------------------------------------------
# SAVE JSON
# -------------------------------------------------------------------

comparison_json = {
    "evaluation": "Phase 2 fair temporal test comparison",
    "models": {
        "random_forest": {
            metric: get_metric(baseline, metric)
            for metric in metrics
        },
        "graphsage": {
            metric: get_metric(gnn, metric)
            for metric in metrics
        }
    },
    "confusion_matrices": {
        "random_forest": baseline_cm,
        "graphsage": gnn_cm
    }
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        comparison_json,
        f,
        indent=2
    )


# -------------------------------------------------------------------
# COMPLETE
# -------------------------------------------------------------------

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)

print(f"\nCSV saved:")
print(OUTPUT_CSV)

print(f"\nJSON saved:")
print(OUTPUT_JSON)