from neo4j import GraphDatabase


URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "fingraphNeo4j2026"


def run_query(session, title, query, **params):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    result = session.run(query, **params)

    records = list(result)

    for record in records:
        print(dict(record))

    if not records:
        print("No records returned.")

    return records


def main():

    print("=" * 70)
    print("FinGraph Neo4j VERIFICATION")
    print("=" * 70)

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD),
    )

    try:

        driver.verify_connectivity()

        with driver.session() as session:

            # ========================================================
            # 1. NODE COUNTS
            # ========================================================

            run_query(
                session,
                "1. NODE COUNTS",
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN
                    label,
                    count(*) AS count
                ORDER BY label
                """
            )

            # ========================================================
            # 2. RELATIONSHIP COUNTS
            # ========================================================

            run_query(
                session,
                "2. RELATIONSHIP COUNTS",
                """
                MATCH ()-[r]->()
                RETURN
                    type(r) AS relationship,
                    count(*) AS count
                ORDER BY relationship
                """
            )

            # ========================================================
            # 3. FRAUD TRANSACTION COUNT
            # ========================================================

            run_query(
                session,
                "3. FRAUD TRANSACTIONS",
                """
                MATCH (t:Transaction)
                RETURN
                    count(*) AS total_transactions,
                    sum(
                        CASE
                            WHEN t.isFraud = 1
                            THEN 1
                            ELSE 0
                        END
                    ) AS fraud_transactions
                """
            )

            # ========================================================
            # 4. TRANSACTION -> ACCOUNT CONNECTIVITY
            # ========================================================

            run_query(
                session,
                "4. TRANSACTION CONNECTIVITY",
                """
                MATCH (t:Transaction)
                OPTIONAL MATCH (a1:Account)-[:SENT]->(t)
                OPTIONAL MATCH (a2:Account)-[:RECEIVED]->(t)

                WITH
                    t,
                    count(DISTINCT a1) AS senders,
                    count(DISTINCT a2) AS receivers

                RETURN
                    count(t) AS transactions,
                    sum(
                        CASE WHEN senders = 1
                        THEN 1 ELSE 0 END
                    ) AS transactions_with_sender,
                    sum(
                        CASE WHEN receivers = 1
                        THEN 1 ELSE 0 END
                    ) AS transactions_with_receiver
                """
            )

            # ========================================================
            # 5. SHARED DEVICES
            # ========================================================

            run_query(
                session,
                "5. TOP SHARED DEVICES",
                """
                MATCH (d:Device)<-[:USES_DEVICE]-(a:Account)

                WITH
                    d,
                    count(DISTINCT a) AS account_count

                WHERE account_count > 1

                RETURN
                    d.id AS device_id,
                    account_count

                ORDER BY account_count DESC

                LIMIT 10
                """
            )

            # ========================================================
            # 6. SHARED IPs
            # ========================================================

            run_query(
                session,
                "6. TOP SHARED IPs",
                """
                MATCH (i:IP)<-[:USES_IP]-(a:Account)

                WITH
                    i,
                    count(DISTINCT a) AS account_count

                WHERE account_count > 1

                RETURN
                    i.id AS ip_id,
                    account_count

                ORDER BY account_count DESC

                LIMIT 10
                """
            )

            # ========================================================
            # 7. ACCOUNT NETWORK CONNECTIONS
            # ========================================================

            run_query(
                session,
                "7. TOP CONNECTED ACCOUNTS",
                """
                MATCH (a:Account)-[r:CONNECTED_TO]->(b:Account)

                RETURN
                    a.id AS source_account,
                    b.id AS destination_account,
                    r.amount AS amount,
                    r.step AS step

                ORDER BY r.amount DESC

                LIMIT 10
                """
            )

            # ========================================================
            # 8. FRAUD TRANSACTION EXAMPLE
            # ========================================================

            fraud_records = run_query(
                session,
                "8. SAMPLE FRAUD TRANSACTION",
                """
                MATCH (t:Transaction {isFraud: 1})
                RETURN
                    t.id AS transaction_id,
                    t.amount AS amount,
                    t.type AS type,
                    t.step AS step,
                    t.origin AS origin,
                    t.destination AS destination
                ORDER BY t.step
                LIMIT 5
                """
            )

            # ========================================================
            # 9. NETWORK AROUND FRAUD TRANSACTION
            # ========================================================

            if fraud_records:

                transaction_id = fraud_records[0][
                    "transaction_id"
                ]

                run_query(
                    session,
                    "9. NETWORK AROUND SAMPLE FRAUD TRANSACTION",
                    """
                    MATCH (t:Transaction {
                        id: $transaction_id
                    })

                    OPTIONAL MATCH
                        (sender:Account)-[:SENT]->(t)

                    OPTIONAL MATCH
                        (receiver:Account)-[:RECEIVED]->(t)

                    OPTIONAL MATCH
                        (sender)-[:USES_DEVICE]->(d:Device)

                    OPTIONAL MATCH
                        (sender)-[:USES_IP]->(i:IP)

                    RETURN
                        t.id AS transaction_id,
                        sender.id AS sender,
                        receiver.id AS receiver,
                        collect(DISTINCT d.id)[..5]
                            AS sender_devices,
                        collect(DISTINCT i.id)[..5]
                            AS sender_ips
                    """,
                    transaction_id=transaction_id,
                )

            # ========================================================
            # 10. MONEY FLOW SAMPLE
            # ========================================================

            run_query(
                session,
                "10. SAMPLE MONEY FLOW PATHS",
                """
                MATCH
                    (a:Account)
                    -[:CONNECTED_TO*2..3]->
                    (b:Account)

                WHERE a <> b

                RETURN
                    a.id AS start_account,
                    b.id AS end_account

                LIMIT 10
                """
            )

            # ========================================================
            # 11. CONSTRAINTS
            # ========================================================

            run_query(
                session,
                "11. NEO4J CONSTRAINTS",
                """
                SHOW CONSTRAINTS
                YIELD name, type, entityType, labelsOrTypes, properties
                RETURN
                    name,
                    type,
                    entityType,
                    labelsOrTypes,
                    properties
                ORDER BY name
                """
            )

        print()
        print("=" * 70)
        print("NEO4J VERIFICATION COMPLETE")
        print("=" * 70)

    finally:

        driver.close()


if __name__ == "__main__":
    main()