"""
FinGraph -> Neo4j Loader

Loads the same deterministic 100,000-transaction sample used by
the FinGraph temporal graph builder into Neo4j.

Neo4j node labels:
    Account
    Transaction
    Device
    IP
    Merchant
    Session

Neo4j relationships:
    SENT
    SENT_BY
    RECEIVED
    RECEIVED_BY
    CONNECTED_TO
    USES_DEVICE
    USES_IP
    USES_MERCHANT
    LOGIN_FROM

The loader is designed to be safely re-runnable by using MERGE.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from neo4j import GraphDatabase


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PAYSIM_PATH = (
    ROOT
    / "data"
    / "processed"
    / "paysim_engineered.csv"
)

RELATIONSHIP_PATH = (
    ROOT
    / "data"
    / "synthetic"
    / "account_relationships.csv"
)


# ============================================================
# NEO4J CONFIG
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "fingraphNeo4j2026"


# ============================================================
# FINGRAPH SAMPLE CONFIG
# ============================================================

SAMPLE_SIZE = 100_000
SEED = 42

CHUNK_SIZE = 500_000

NODE_BATCH_SIZE = 2_000
REL_BATCH_SIZE = 5_000


# ============================================================
# COUNT DATASET ROWS
# ============================================================

def count_rows():

    count = 0

    with open(
        PAYSIM_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.reader(f)

        next(reader)

        for _ in reader:
            count += 1

    return count


# ============================================================
# REPRODUCE FINgraph SAMPLE
# ============================================================

def sample_dataset():

    print()
    print("=" * 60)
    print("REPRODUCING FIN GRAPH SAMPLE")
    print("=" * 60)

    total_rows = count_rows()

    print(
        f"Full dataset rows: {total_rows:,}"
    )

    print(
        f"Target sample: {SAMPLE_SIZE:,}"
    )

    rng = np.random.default_rng(
        SEED
    )

    current_sample = None

    processed = 0

    for chunk in pd.read_csv(
        PAYSIM_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        chunk = chunk.copy()

        chunk["_random_key"] = (
            rng.random(len(chunk))
        )

        if current_sample is None:

            current_sample = chunk

        else:

            current_sample = pd.concat(
                [
                    current_sample,
                    chunk,
                ],
                ignore_index=True,
            )

        if len(current_sample) > SAMPLE_SIZE:

            current_sample = (
                current_sample
                .nsmallest(
                    SAMPLE_SIZE,
                    "_random_key",
                )
                .reset_index(drop=True)
            )

        processed += len(chunk)

        print(
            f"\rScanned: "
            f"{processed:,}/{total_rows:,}",
            end="",
            flush=True,
        )

    print()

    df = (
        current_sample
        .drop(
            columns=["_random_key"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Ensure exactly the same ordering used for the
    # temporal graph.
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "step",
                "transaction_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    print(
        f"Selected sample: {len(df):,}"
    )

    print(
        f"Step range: "
        f"{df['step'].min()} -> "
        f"{df['step'].max()}"
    )

    print(
        f"Fraud transactions: "
        f"{int(df['isFraud'].sum()):,}"
    )

    return df


# ============================================================
# LOAD RELEVANT SYNTHETIC RELATIONSHIPS
# ============================================================

def find_column(
    columns,
    candidates,
):

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


def load_relationships(
    account_ids,
):

    print()
    print("=" * 60)
    print("LOADING SYNTHETIC ACCOUNT RELATIONSHIPS")
    print("=" * 60)

    account_ids = set(
        str(x)
        for x in account_ids
    )

    header = pd.read_csv(
        RELATIONSHIP_PATH,
        nrows=0,
    )

    columns = list(
        header.columns
    )

    account_col = find_column(
        columns,
        [
            "account_id",
            "account",
            "name",
        ],
    )

    device_col = find_column(
        columns,
        [
            "device_id",
            "device",
        ],
    )

    ip_col = find_column(
        columns,
        [
            "ip_address",
            "ip_id",
            "ip",
        ],
    )

    merchant_col = find_column(
        columns,
        [
            "merchant_id",
            "merchant",
        ],
    )

    session_col = find_column(
        columns,
        [
            "session_id",
            "session",
        ],
    )

    required = [
        account_col,
        device_col,
        ip_col,
        merchant_col,
        session_col,
    ]

    if any(
        column is None
        for column in required
    ):

        raise ValueError(
            "Could not find all required relationship "
            f"columns.\nAvailable columns: {columns}"
        )

    usecols = [
        account_col,
        device_col,
        ip_col,
        merchant_col,
        session_col,
    ]

    pieces = []

    processed = 0

    for chunk in pd.read_csv(
        RELATIONSHIP_PATH,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        dtype=str,
    ):

        selected = chunk[
            chunk[account_col].isin(
                account_ids
            )
        ]

        if not selected.empty:
            pieces.append(
                selected
            )

        processed += len(chunk)

        print(
            f"\rRelationship rows scanned: "
            f"{processed:,}",
            end="",
            flush=True,
        )

    print()

    if not pieces:

        raise ValueError(
            "No synthetic relationships found "
            "for sampled accounts."
        )

    relations = pd.concat(
        pieces,
        ignore_index=True,
    )

    relations = relations.rename(
        columns={
            account_col: "account_id",
            device_col: "device_id",
            ip_col: "ip_address",
            merchant_col: "merchant_id",
            session_col: "session_id",
        }
    )

    relations = (
        relations
        .drop_duplicates(
            subset=["account_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"Relationships loaded: "
        f"{len(relations):,}"
    )

    return relations


# ============================================================
# CREATE CONSTRAINTS
# ============================================================

def create_constraints(driver):

    print()
    print("=" * 60)
    print("CREATING NEO4J CONSTRAINTS")
    print("=" * 60)

    queries = [

        """
        CREATE CONSTRAINT account_id_unique IF NOT EXISTS
        FOR (n:Account)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS
        FOR (n:Transaction)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT device_id_unique IF NOT EXISTS
        FOR (n:Device)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT ip_id_unique IF NOT EXISTS
        FOR (n:IP)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT merchant_id_unique IF NOT EXISTS
        FOR (n:Merchant)
        REQUIRE n.id IS UNIQUE
        """,

        """
        CREATE CONSTRAINT session_id_unique IF NOT EXISTS
        FOR (n:Session)
        REQUIRE n.id IS UNIQUE
        """,
    ]

    with driver.session() as session:

        for query in queries:

            session.run(query).consume()

    print(
        "Constraints created/verified."
    )


# ============================================================
# LOAD TRANSACTION NODES
# ============================================================

def load_transaction_nodes(
    driver,
    df,
):

    print()
    print("=" * 60)
    print("LOADING TRANSACTION NODES")
    print("=" * 60)

    rows = []

    with driver.session() as session:

        for index, row in df.iterrows():

            rows.append(
                {
                    "id": str(
                        row["transaction_id"]
                    ),
                    "step": int(
                        row["step"]
                    ),
                    "type": str(
                        row["type"]
                    ),
                    "amount": float(
                        row["amount"]
                    ),
                    "origin": str(
                        row["nameOrig"]
                    ),
                    "destination": str(
                        row["nameDest"]
                    ),
                    "isFraud": int(
                        row["isFraud"]
                    ),
                    "isFlaggedFraud": int(
                        row.get(
                            "isFlaggedFraud",
                            0
                        )
                    ),
                    "hour": int(
                        row.get(
                            "hour",
                            0
                        )
                    ),
                    "day": int(
                        row.get(
                            "day",
                            0
                        )
                    ),
                }
            )

            if len(rows) >= NODE_BATCH_SIZE:

                session.run(
                    """
                    UNWIND $rows AS row

                    MERGE (t:Transaction {
                        id: row.id
                    })

                    SET
                        t.step = row.step,
                        t.type = row.type,
                        t.amount = row.amount,
                        t.origin = row.origin,
                        t.destination = row.destination,
                        t.isFraud = row.isFraud,
                        t.isFlaggedFraud = row.isFlaggedFraud,
                        t.hour = row.hour,
                        t.day = row.day
                    """,
                    rows=rows,
                ).consume()

                rows.clear()

                print(
                    f"\rTransactions loaded: "
                    f"{index + 1:,}/{len(df):,}",
                    end="",
                    flush=True,
                )

        if rows:

            session.run(
                """
                UNWIND $rows AS row

                MERGE (t:Transaction {
                    id: row.id
                })

                SET
                    t.step = row.step,
                    t.type = row.type,
                    t.amount = row.amount,
                    t.origin = row.origin,
                    t.destination = row.destination,
                    t.isFraud = row.isFraud,
                    t.isFlaggedFraud = row.isFlaggedFraud,
                    t.hour = row.hour,
                    t.day = row.day
                """,
                rows=rows,
            ).consume()

    print()

    print(
        f"Transaction nodes loaded: "
        f"{len(df):,}"
    )


# ============================================================
# LOAD ACCOUNT NODES
# ============================================================

def load_account_nodes(
    driver,
    df,
):

    print()
    print("=" * 60)
    print("LOADING ACCOUNT NODES")
    print("=" * 60)

    sent_counts = (
        df["nameOrig"]
        .astype(str)
        .value_counts()
    )

    received_counts = (
        df["nameDest"]
        .astype(str)
        .value_counts()
    )

    fraud_sender_accounts = set(
        df.loc[
            df["isFraud"] == 1,
            "nameOrig",
        ]
        .astype(str)
        .tolist()
    )

    account_ids = sorted(
        set(
            df["nameOrig"]
            .astype(str)
        )
        |
        set(
            df["nameDest"]
            .astype(str)
        )
    )

    rows = []

    with driver.session() as session:

        for index, account_id in enumerate(
            account_ids,
            start=1,
        ):

            rows.append(
                {
                    "id": account_id,
                    "sent_count": int(
                        sent_counts.get(
                            account_id,
                            0,
                        )
                    ),
                    "received_count": int(
                        received_counts.get(
                            account_id,
                            0,
                        )
                    ),
                    "isFraud": int(
                        account_id
                        in fraud_sender_accounts
                    ),
                }
            )

            if len(rows) >= NODE_BATCH_SIZE:

                session.run(
                    """
                    UNWIND $rows AS row

                    MERGE (a:Account {
                        id: row.id
                    })

                    SET
                        a.sent_count = row.sent_count,
                        a.received_count =
                            row.received_count,
                        a.isFraud = row.isFraud
                    """,
                    rows=rows,
                ).consume()

                rows.clear()

                print(
                    f"\rAccounts loaded: "
                    f"{index:,}/{len(account_ids):,}",
                    end="",
                    flush=True,
                )

        if rows:

            session.run(
                """
                UNWIND $rows AS row

                MERGE (a:Account {
                    id: row.id
                })

                SET
                    a.sent_count = row.sent_count,
                    a.received_count =
                        row.received_count,
                    a.isFraud = row.isFraud
                """,
                rows=rows,
            ).consume()

    print()

    print(
        f"Account nodes loaded: "
        f"{len(account_ids):,}"
    )


# ============================================================
# LOAD RELATIONSHIP NODES
# ============================================================

def load_static_relationship_nodes(
    driver,
    relations,
):

    print()
    print("=" * 60)
    print("LOADING DEVICE/IP/MERCHANT/SESSION NODES")
    print("=" * 60)

    devices = sorted(
        relations["device_id"]
        .astype(str)
        .unique()
    )

    ips = sorted(
        relations["ip_address"]
        .astype(str)
        .unique()
    )

    merchants = sorted(
        relations["merchant_id"]
        .astype(str)
        .unique()
    )

    sessions = sorted(
        relations["session_id"]
        .astype(str)
        .unique()
    )

    with driver.session() as session:

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        session.run(
            """
            UNWIND $rows AS row

            MERGE (d:Device {
                id: row
            })
            """,
            rows=devices,
        ).consume()

        print(
            f"Device nodes: {len(devices):,}"
        )

        # ----------------------------------------------------
        # IP
        # ----------------------------------------------------

        session.run(
            """
            UNWIND $rows AS row

            MERGE (i:IP {
                id: row
            })
            """,
            rows=ips,
        ).consume()

        print(
            f"IP nodes: {len(ips):,}"
        )

        # ----------------------------------------------------
        # Merchant
        # ----------------------------------------------------

        session.run(
            """
            UNWIND $rows AS row

            MERGE (m:Merchant {
                id: row
            })
            """,
            rows=merchants,
        ).consume()

        print(
            f"Merchant nodes: {len(merchants):,}"
        )

        # ----------------------------------------------------
        # Session
        # ----------------------------------------------------

        session.run(
            """
            UNWIND $rows AS row

            MERGE (s:Session {
                id: row
            })
            """,
            rows=sessions,
        ).consume()

        print(
            f"Session nodes: {len(sessions):,}"
        )


# ============================================================
# LOAD TRANSACTION RELATIONSHIPS
# ============================================================

def load_transaction_relationships(
    driver,
    df,
):

    print()
    print("=" * 60)
    print("LOADING TRANSACTION RELATIONSHIPS")
    print("=" * 60)

    rows = []

    with driver.session() as session:

        for index, row in df.iterrows():

            transaction_id = str(
                row["transaction_id"]
            )

            origin = str(
                row["nameOrig"]
            )

            destination = str(
                row["nameDest"]
            )

            step = int(
                row["step"]
            )

            amount = float(
                row["amount"]
            )

            rows.append(
                {
                    "transaction_id":
                        transaction_id,
                    "origin":
                        origin,
                    "destination":
                        destination,
                    "step":
                        step,
                    "amount":
                        amount,
                }
            )

            if len(rows) >= REL_BATCH_SIZE:

                session.run(
                    """
                    UNWIND $rows AS row

                    MERGE (a:Account {
                        id: row.origin
                    })

                    MERGE (b:Account {
                        id: row.destination
                    })

                    MERGE (t:Transaction {
                        id: row.transaction_id
                    })

                    MERGE (a)-[s:SENT {
                        transaction_id:
                            row.transaction_id
                    }]->(t)

                    SET
                        s.step = row.step,
                        s.amount = row.amount

                    MERGE (t)-[sb:SENT_BY {
                        transaction_id:
                            row.transaction_id
                    }]->(a)

                    SET
                        sb.step = row.step,
                        sb.amount = row.amount

                    MERGE (b)-[r:RECEIVED {
                        transaction_id:
                            row.transaction_id
                    }]->(t)

                    SET
                        r.step = row.step,
                        r.amount = row.amount

                    MERGE (t)-[rb:RECEIVED_BY {
                        transaction_id:
                            row.transaction_id
                    }]->(b)

                    SET
                        rb.step = row.step,
                        rb.amount = row.amount

                    FOREACH (
                        ignored IN CASE
                            WHEN row.origin <>
                                 row.destination
                            THEN [1]
                            ELSE []
                        END |

                        MERGE (a)-[c:CONNECTED_TO {
                            transaction_id:
                                row.transaction_id
                        }]->(b)

                        SET
                            c.step = row.step,
                            c.amount = row.amount
                    )
                    """,
                    rows=rows,
                ).consume()

                rows.clear()

                print(
                    f"\rTransactions processed: "
                    f"{index + 1:,}/{len(df):,}",
                    end="",
                    flush=True,
                )

        if rows:

            session.run(
                """
                UNWIND $rows AS row

                MERGE (a:Account {
                    id: row.origin
                })

                MERGE (b:Account {
                    id: row.destination
                })

                MERGE (t:Transaction {
                    id: row.transaction_id
                })

                MERGE (a)-[s:SENT {
                    transaction_id:
                        row.transaction_id
                }]->(t)

                SET
                    s.step = row.step,
                    s.amount = row.amount

                MERGE (t)-[sb:SENT_BY {
                    transaction_id:
                        row.transaction_id
                }]->(a)

                SET
                    sb.step = row.step,
                    sb.amount = row.amount

                MERGE (b)-[r:RECEIVED {
                    transaction_id:
                        row.transaction_id
                }]->(t)

                SET
                    r.step = row.step,
                    r.amount = row.amount

                MERGE (t)-[rb:RECEIVED_BY {
                    transaction_id:
                        row.transaction_id
                }]->(b)

                SET
                    rb.step = row.step,
                    rb.amount = row.amount

                FOREACH (
                    ignored IN CASE
                        WHEN row.origin <>
                             row.destination
                        THEN [1]
                        ELSE []
                    END |

                    MERGE (a)-[c:CONNECTED_TO {
                        transaction_id:
                            row.transaction_id
                    }]->(b)

                    SET
                        c.step = row.step,
                        c.amount = row.amount
                )
                """,
                rows=rows,
            ).consume()

    print()

    print(
        "Transaction relationships loaded."
    )


# ============================================================
# LOAD STATIC ACCOUNT RELATIONSHIPS
# ============================================================

def load_static_relationships(
    driver,
    relations,
):

    print()
    print("=" * 60)
    print("LOADING ACCOUNT RELATIONSHIPS")
    print("=" * 60)

    rows = []

    with driver.session() as session:

        for index, row in relations.iterrows():

            rows.append(
                {
                    "account_id": str(
                        row["account_id"]
                    ),
                    "device_id": str(
                        row["device_id"]
                    ),
                    "ip_address": str(
                        row["ip_address"]
                    ),
                    "merchant_id": str(
                        row["merchant_id"]
                    ),
                    "session_id": str(
                        row["session_id"]
                    ),
                }
            )

            if len(rows) >= REL_BATCH_SIZE:

                session.run(
                    """
                    UNWIND $rows AS row

                    MERGE (a:Account {
                        id: row.account_id
                    })

                    MERGE (d:Device {
                        id: row.device_id
                    })

                    MERGE (i:IP {
                        id: row.ip_address
                    })

                    MERGE (m:Merchant {
                        id: row.merchant_id
                    })

                    MERGE (s:Session {
                        id: row.session_id
                    })

                    MERGE (a)-[:USES_DEVICE]->(d)

                    MERGE (a)-[:USES_IP]->(i)

                    MERGE (a)-[:USES_MERCHANT]->(m)

                    MERGE (a)-[:LOGIN_FROM]->(s)
                    """,
                    rows=rows,
                ).consume()

                rows.clear()

                print(
                    f"\rStatic relationships processed: "
                    f"{index + 1:,}/{len(relations):,}",
                    end="",
                    flush=True,
                )

        if rows:

            session.run(
                """
                UNWIND $rows AS row

                MERGE (a:Account {
                    id: row.account_id
                })

                MERGE (d:Device {
                    id: row.device_id
                })

                MERGE (i:IP {
                    id: row.ip_address
                })

                MERGE (m:Merchant {
                    id: row.merchant_id
                })

                MERGE (s:Session {
                    id: row.session_id
                })

                MERGE (a)-[:USES_DEVICE]->(d)

                MERGE (a)-[:USES_IP]->(i)

                MERGE (a)-[:USES_MERCHANT]->(m)

                MERGE (a)-[:LOGIN_FROM]->(s)
                """,
                rows=rows,
            ).consume()

    print()

    print(
        "Static account relationships loaded."
    )


# ============================================================
# DATABASE SUMMARY
# ============================================================

def print_database_summary(
    driver,
):

    print()
    print("=" * 60)
    print("NEO4J DATABASE SUMMARY")
    print("=" * 60)

    with driver.session() as session:

        # ----------------------------------------------------
        # Node counts
        # ----------------------------------------------------

        node_query = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(*) AS count
        ORDER BY label
        """

        result = session.run(
            node_query
        )

        print()
        print("Nodes:")

        for record in result:

            print(
                f"  {str(record['label']):15s} "
                f"{record['count']:,}"
            )

        # ----------------------------------------------------
        # Relationship counts
        # ----------------------------------------------------

        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS relationship, count(*) AS count
        ORDER BY relationship
        """

        result = session.run(
            rel_query
        )

        print()
        print("Relationships:")

        for record in result:

            print(
                f"  {str(record['relationship']):20s} "
                f"{record['count']:,}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FinGraph -> Neo4j LOADER")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not PAYSIM_PATH.exists():

        raise FileNotFoundError(
            PAYSIM_PATH
        )

    if not RELATIONSHIP_PATH.exists():

        raise FileNotFoundError(
            RELATIONSHIP_PATH
        )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    df = sample_dataset()

    # --------------------------------------------------------
    # Accounts
    # --------------------------------------------------------

    account_ids = set(
        df["nameOrig"]
        .astype(str)
    )

    account_ids.update(
        df["nameDest"]
        .astype(str)
    )

    print()
    print(
        f"Unique sampled accounts: "
        f"{len(account_ids):,}"
    )

    # --------------------------------------------------------
    # Synthetic relationships
    # --------------------------------------------------------

    relations = load_relationships(
        account_ids
    )

    # --------------------------------------------------------
    # Neo4j connection
    # --------------------------------------------------------

    print()
    print(
        "Connecting to:",
        NEO4J_URI
    )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USERNAME,
            NEO4J_PASSWORD,
        ),
    )

    try:

        driver.verify_connectivity()

        print(
            "Neo4j connectivity: PASSED"
        )

        # ----------------------------------------------------
        # Constraints
        # ----------------------------------------------------

        create_constraints(
            driver
        )

        # ----------------------------------------------------
        # Nodes
        # ----------------------------------------------------

        load_transaction_nodes(
            driver,
            df,
        )

        load_account_nodes(
            driver,
            df,
        )

        load_static_relationship_nodes(
            driver,
            relations,
        )

        # ----------------------------------------------------
        # Transaction relationships
        # ----------------------------------------------------

        load_transaction_relationships(
            driver,
            df,
        )

        # ----------------------------------------------------
        # Static relationships
        # ----------------------------------------------------

        load_static_relationships(
            driver,
            relations,
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print_database_summary(
            driver
        )

        print()
        print("=" * 60)
        print("NEO4J LOAD COMPLETE")
        print("=" * 60)

    finally:

        driver.close()


if __name__ == "__main__":
    main()