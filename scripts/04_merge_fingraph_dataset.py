from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transactions",
        default="data/processed/paysim_engineered.csv"
    )
    parser.add_argument(
        "--relationships",
        default="data/synthetic/account_relationships.csv"
    )
    parser.add_argument(
        "--output",
        default="data/processed/fingraph_phase1_dataset.csv"
    )
    args = parser.parse_args()

    tx_path = ROOT / args.transactions
    rel_path = ROOT / args.relationships
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tx_path)
    rel = pd.read_csv(rel_path)

    orig_rel = rel.add_prefix("orig_")
    dest_rel = rel.add_prefix("dest_")

    # Rename joining keys.
    orig_rel = orig_rel.rename(columns={"orig_account_id": "nameOrig"})
    dest_rel = dest_rel.rename(columns={"dest_account_id": "nameDest"})

    # Keep relationship metadata with explicit prefixes.
    merged = df.merge(
        orig_rel[
            [
                "nameOrig",
                "orig_device_id",
                "orig_ip_address",
                "orig_merchant_id",
                "orig_session_id"
            ]
        ],
        on="nameOrig",
        how="left",
        validate="many_to_one"
    )

    merged = merged.merge(
        dest_rel[
            [
                "nameDest",
                "dest_device_id",
                "dest_ip_address",
                "dest_merchant_id",
                "dest_session_id"
            ]
        ],
        on="nameDest",
        how="left",
        validate="many_to_one"
    )

    # Convenience flags for network-like investigation.
    merged["same_device_sender_receiver"] = (
        merged["orig_device_id"] == merged["dest_device_id"]
    ).astype(int)

    merged["same_ip_sender_receiver"] = (
        merged["orig_ip_address"] == merged["dest_ip_address"]
    ).astype(int)

    merged.to_csv(out_path, index=False)

    print("Final Phase 1 dataset shape:", merged.shape)
    print("Saved:", out_path)
    print("\nColumns:")
    print("\n".join(f" - {c}" for c in merged.columns))

if __name__ == "__main__":
    main()
