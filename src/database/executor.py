from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.database.duckdb_engine import DuckDBEngine
from src.validation.sql_guard import SQLGuard, SQLGuardError


class QueryExecutionError(Exception):
    """Raised when a query cannot be executed safely."""


class QueryExecutor:
    """
    Executes validated SQL against the configured DuckDB dataset.

    Execution flow:

        SQL
        ↓
        SQL Guard
        ↓
        DuckDB Engine
        ↓
        DataFrame
    """

    def __init__(
        self,
        dataset_path: str | Path,
        table_name: str = "ecommerce_orders",
    ):
        self.dataset_path = Path(dataset_path)
        self.table_name = table_name

        self.guard = SQLGuard(
            allowed_table=table_name
        )

        self.engine = DuckDBEngine(
            dataset_path=self.dataset_path,
            table_name=self.table_name,
        )

    def execute(self, sql: str) -> pd.DataFrame:
        """
        Validate and execute SQL.

        SQL is NEVER sent to DuckDB before passing the SQL Guard.
        """

        if not isinstance(sql, str):
            raise QueryExecutionError(
                "SQL query must be a string."
            )

        if not sql.strip():
            raise QueryExecutionError(
                "SQL query cannot be empty."
            )

        # --------------------------------------------------------------
        # Step 1: Safety validation
        # --------------------------------------------------------------

        try:
            self.guard.validate(sql)

        except SQLGuardError as exc:
            raise QueryExecutionError(
                f"SQL query rejected by safety guard: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Step 2: Execute validated SQL
        # --------------------------------------------------------------

        try:
            result = self.engine.execute(sql)

        except Exception as exc:
            raise QueryExecutionError(
                f"DuckDB execution failed: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Step 3: Validate returned result
        # --------------------------------------------------------------

        if not isinstance(result, pd.DataFrame):
            raise QueryExecutionError(
                "DuckDB execution did not return a pandas DataFrame."
            )

        return result

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        self.engine.close()


# ======================================================================
# TESTS
# ======================================================================

if __name__ == "__main__":

    print("\n=== QUERY EXECUTOR TEST ===\n")

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "data"
        / "raw"
        / "ecommerce_orders.csv"
    )

    executor = QueryExecutor(
        dataset_path=dataset_path,
        table_name="ecommerce_orders",
    )

    # ------------------------------------------------------------------
    # SAFE QUERY
    # ------------------------------------------------------------------

    safe_sql = """
        SELECT
            region,
            SUM(sales) AS total_sales
        FROM ecommerce_orders
        GROUP BY region
        ORDER BY total_sales DESC;
    """

    print("SAFE QUERY")
    print("=" * 80)
    print(safe_sql.strip())

    try:
        result = executor.execute(safe_sql)

        print("\nExecution: PASS")
        print("\nResult:")
        print(result.to_string(index=False))

    except QueryExecutionError as exc:
        print(f"\nExecution: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # BLOCKED QUERY
    # ------------------------------------------------------------------

    unsafe_sql = """
        DROP TABLE ecommerce_orders;
    """

    print("\n\nUNSAFE QUERY")
    print("=" * 80)
    print(unsafe_sql.strip())

    try:
        executor.execute(unsafe_sql)

        print("\nExecution: FAIL")
        print("Unsafe query was executed.")

    except QueryExecutionError as exc:
        print("\nExecution: BLOCKED")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # EXTRACT / DATE QUERY
    # ------------------------------------------------------------------

    date_sql = """
        SELECT
            SUM(sales) AS total_sales
        FROM ecommerce_orders
        WHERE EXTRACT(YEAR FROM order_date) = 2024;
    """

    print("\n\nDATE QUERY")
    print("=" * 80)
    print(date_sql.strip())

    try:
        result = executor.execute(date_sql)

        print("\nExecution: PASS")
        print("\nResult:")
        print(result.to_string(index=False))

    except QueryExecutionError as exc:
        print("\nExecution: FAIL")
        print(f"Reason: {exc}")

    executor.close()

    print("\n\nQuery executor test completed.")