from pathlib import Path

import duckdb
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".parquet"}


class DuckDBEngine:
    """
    Lightweight DuckDB execution engine for CSV and Parquet datasets.

    The engine is responsible for:
    - registering a dataset as a queryable table
    - executing SQL
    - returning results as pandas DataFrames
    - exposing basic database metadata
    """

    def __init__(
        self,
        dataset_path: str | Path,
        table_name: str = "dataset",
    ):
        self.dataset_path = Path(dataset_path)
        self.table_name = table_name

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        if self.dataset_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported dataset format: "
                f"{self.dataset_path.suffix}"
            )

        self.connection = duckdb.connect(database=":memory:")

        self._register_dataset()

    def _register_dataset(self) -> None:
        """
        Register the source dataset as a DuckDB view.
        """

        file_path = str(self.dataset_path.resolve()).replace(
            "\\",
            "/",
        )

        extension = self.dataset_path.suffix.lower()

        if extension == ".csv":
            query = f"""
                CREATE OR REPLACE VIEW {self.table_name} AS
                SELECT *
                FROM read_csv_auto('{file_path}')
            """

        elif extension == ".parquet":
            query = f"""
                CREATE OR REPLACE VIEW {self.table_name} AS
                SELECT *
                FROM read_parquet('{file_path}')
            """

        else:
            raise ValueError(
                f"Unsupported dataset format: {extension}"
            )

        self.connection.execute(query)

    def execute(
        self,
        sql: str,
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a DataFrame.
        """

        if not sql or not sql.strip():
            raise ValueError("SQL query cannot be empty.")

        return self.connection.execute(sql).df()

    def get_schema(self) -> pd.DataFrame:
        """
        Return the schema of the registered dataset.
        """

        return self.connection.execute(
            f"DESCRIBE {self.table_name}"
        ).df()

    def count_rows(self) -> int:
        """
        Return the number of rows in the dataset.
        """

        result = self.connection.execute(
            f"SELECT COUNT(*) AS row_count "
            f"FROM {self.table_name}"
        ).fetchone()

        return int(result[0])

    def preview(
        self,
        limit: int = 5,
    ) -> pd.DataFrame:
        """
        Return a small preview of the dataset.
        """

        if limit <= 0:
            raise ValueError("Preview limit must be greater than zero.")

        return self.connection.execute(
            f"""
            SELECT *
            FROM {self.table_name}
            LIMIT {int(limit)}
            """
        ).df()

    def close(self) -> None:
        """
        Close the DuckDB connection.
        """

        self.connection.close()


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_dir
        / "data"
        / "raw"
        / "ecommerce_orders.csv"
    )

    engine = DuckDBEngine(
        dataset_path=dataset_path,
        table_name="ecommerce_orders",
    )

    print("\n=== DUCKDB ENGINE TEST ===\n")

    print("Dataset:")
    print(dataset_path)

    print("\nRow count:")
    print(engine.count_rows())

    print("\nSchema:")
    print(engine.get_schema().to_string(index=False))

    print("\nPreview:")
    print(engine.preview(5).to_string(index=False))

    print("\nTest query:")

    result = engine.execute(
        """
        SELECT
            region,
            ROUND(SUM(sales), 2) AS total_sales
        FROM ecommerce_orders
        GROUP BY region
        ORDER BY total_sales DESC
        """
    )

    print(result.to_string(index=False))

    engine.close()

    print("\nDuckDB engine test completed successfully.")