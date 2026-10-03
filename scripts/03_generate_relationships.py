
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ACCOUNTS = (
    ROOT / "data" / "synthetic" / "account_relationships_old.csv"
)

DEFAULT_OUTPUT = (
    ROOT / "data" / "synthetic" / "account_relationships.csv"
)

SEED = 42


def stable_hash(value: str) -> int:
    """Stable hash, reproducible across Python runs."""
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


def is_selected(
    account_id: str,
    share: float,
    entity_type: str,
) -> bool:
    """Select accounts independently for each entity type."""
    if share <= 0:
        return False

    if share >= 1:
        return True

    value = stable_hash(
        f"{SEED}:SELECT:{entity_type}:{account_id}"
    )

    return (value % 1_000_000) < int(
        share * 1_000_000
    )


def create_relationship_row(
    account_id: str,
    shared_entity_share: float,
    cluster_count: int,
) -> tuple[str, str, str, str, str, str]:

    def entity_id(entity_type: str, prefix: str) -> str:
        """
        Assign an entity deterministically.

        Most accounts get their own entity.
        A selected subset shares small clusters.
        Selection does not use transaction labels.
        """
        selected = is_selected(
            account_id,
            shared_entity_share,
            entity_type,
        )

        if selected:
            cluster_hash = stable_hash(
                f"{SEED}:CLUSTER:{entity_type}:{account_id}"
            )

            cluster_id = cluster_hash % cluster_count

            return f"{prefix}_SHARED_{cluster_id:06d}"

        unique_hash = stable_hash(
            f"{SEED}:UNIQUE:{entity_type}:{account_id}"
        )

        return f"{prefix}_{unique_hash:016X}"

    # Separate namespaces ensure that sharing a device
    # does not automatically imply sharing an IP/session.
    device_id = entity_id("DEVICE", "D")
    ip_address = entity_id("IP", "IP")
    session_id = entity_id("SESSION", "S")

    # Merchant reuse is generated independently.
    merchant_hash = stable_hash(
        f"{SEED}:MERCHANT:{account_id}"
    )

    merchant_id = f"M_{merchant_hash % 100_000:06d}"

    return (
        account_id,
        device_id,
        ip_address,
        merchant_id,
        session_id,
        "synthetic",
    )


def generate_relationships(
    accounts_path: Path,
    output_path: Path,
    shared_entity_share: float = 0.025,
    cluster_size: int = 10,
):
    if not 0 <= shared_entity_share <= 1:
        raise ValueError(
            "shared_entity_share must be between 0 and 1."
        )

    if cluster_size < 2:
        raise ValueError("cluster_size must be at least 2.")

    print("Counting accounts...")

    total_accounts = 0

    for chunk in pd.read_csv(
        accounts_path,
        usecols=["account_id"],
        chunksize=250_000,
    ):
        total_accounts += chunk["account_id"].nunique()

    if total_accounts == 0:
        raise ValueError("No accounts found in input file.")

    cluster_count = max(
        1,
        int(total_accounts * shared_entity_share)
        // cluster_size,
    )

    print(f"Total accounts: {total_accounts:,}")
    print(f"Seed: {SEED}")
    print(f"Shared-entity selection rate: "
          f"{shared_entity_share:.2%}")
    print(f"Clusters per entity type: {cluster_count:,}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_write = True
    processed = 0

    # Prevent accidental duplication if the source contains
    # repeated account IDs across different chunks.
    # The source should ideally already contain unique IDs.
    seen_accounts: set[str] = set()

    for chunk in pd.read_csv(
        accounts_path,
        usecols=["account_id"],
        chunksize=250_000,
    ):
        rows = []

        for raw_account in chunk["account_id"].dropna():
            account_id = str(raw_account)

            if account_id in seen_accounts:
                continue

            seen_accounts.add(account_id)

            rows.append(
                create_relationship_row(
                    account_id=account_id,
                    shared_entity_share=shared_entity_share,
                    cluster_count=cluster_count,
                )
            )

        out_chunk = pd.DataFrame(
            rows,
            columns=[
                "account_id",
                "device_id",
                "ip_address",
                "merchant_id",
                "session_id",
                "relationship_source",
            ],
        )

        out_chunk["generator_seed"] = SEED
        out_chunk["mode"] = "research_safe"

        out_chunk.to_csv(
            output_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )

        first_write = False
        processed += len(out_chunk)

        print(f"Processed {processed:,} accounts")

    print("\nGeneration completed.")
    print(f"Accounts written: {processed:,}")
    print(f"Output: {output_path}")
    print("Fraud labels were not loaded or used.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate label-independent synthetic "
            "FinGraph relationships."
        )
    )

    parser.add_argument(
        "--accounts",
        type=Path,
        default=DEFAULT_ACCOUNTS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--share",
        type=float,
        default=0.025,
        help=(
            "Fraction selected for shared clusters "
            "per entity type. Default: 0.025"
        ),
    )

    parser.add_argument(
        "--cluster-size",
        type=int,
        default=10,
        help="Approximate target accounts per cluster.",
    )

    args = parser.parse_args()

    generate_relationships(
        accounts_path=args.accounts,
        output_path=args.output,
        shared_entity_share=args.share,
        cluster_size=args.cluster_size,
    )


if __name__ == "__main__":
    main()