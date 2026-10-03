from neo4j import GraphDatabase


URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "fingraphNeo4j2026"


def main():
    print("=" * 60)
    print("FinGraph Neo4j CONNECTION TEST")
    print("=" * 60)

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD),
    )

    try:
        # ----------------------------------------------------
        # Connectivity
        # ----------------------------------------------------

        print("\nChecking connectivity...")

        driver.verify_connectivity()

        print("Connectivity: PASSED")

        # ----------------------------------------------------
        # Authentication
        # ----------------------------------------------------

        print("\nChecking authentication...")

        auth_ok = driver.verify_authentication()

        print(
            "Authentication:",
            "PASSED" if auth_ok else "FAILED"
        )

        if not auth_ok:
            raise RuntimeError(
                "Neo4j authentication failed."
            )

        # ----------------------------------------------------
        # Basic query
        # ----------------------------------------------------

        with driver.session() as session:

            result = session.run(
                """
                RETURN
                    1 AS value,
                    datetime() AS server_time
                """
            )

            record = result.single()

            print(
                "\nQuery result:",
                record["value"]
            )

            print(
                "Server time:",
                record["server_time"]
            )

        # ----------------------------------------------------
        # DBMS information
        # ----------------------------------------------------

        with driver.session() as session:

            result = session.run(
                """
                CALL dbms.components()
                YIELD name, versions, edition
                RETURN name, versions, edition
                ORDER BY name
                """
            )

            print("\nNeo4j DBMS components:")

            records = list(result)

            for record in records:

                print(
                    f"  Name:     {record['name']}"
                )

                print(
                    f"  Version:  {record['versions']}"
                )

                print(
                    f"  Edition:  {record['edition']}"
                )

                print()

        # ----------------------------------------------------
        # Database existence / current database
        # ----------------------------------------------------

        with driver.session() as session:

            result = session.run(
                """
                SHOW DATABASES
                YIELD name, type, currentStatus
                RETURN name, type, currentStatus
                ORDER BY name
                """
            )

            print("Available databases:")

            for record in result:

                print(
                    f"  {record['name']:15s} "
                    f"type={record['type']:10s} "
                    f"status={record['currentStatus']}"
                )

        # ----------------------------------------------------
        # Final
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("NEO4J CONNECTION SUCCESSFUL")
        print("=" * 60)

    except Exception as exc:

        print()
        print("=" * 60)
        print("NEO4J CONNECTION FAILED")
        print("=" * 60)

        print(
            "\nError type:",
            type(exc).__name__
        )

        print(
            "Error:",
            str(exc)
        )

        raise

    finally:

        driver.close()


if __name__ == "__main__":
    main()