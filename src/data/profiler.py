from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".parquet"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV or Parquet dataset into a pandas DataFrame.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file type: {path.suffix}. "
        f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Classify columns into useful analytical categories.
    """

    numeric_columns = []
    categorical_columns = []
    date_columns = []
    boolean_columns = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_bool_dtype(series):
            boolean_columns.append(column)

        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)

        elif pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(column)

        else:
            # Try detecting date-like string columns.
            parsed = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed",
            )

            valid_ratio = parsed.notna().mean()

            if valid_ratio >= 0.90:
                date_columns.append(column)
            else:
                categorical_columns.append(column)

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "date": date_columns,
        "boolean": boolean_columns,
    }


def profile_numeric_columns(df: pd.DataFrame) -> dict:
    """
    Generate descriptive statistics for numeric columns.
    """

    numeric_df = df.select_dtypes(include="number")

    profile = {}

    for column in numeric_df.columns:
        series = numeric_df[column]

        profile[column] = {
            "min": float(series.min()) if not series.dropna().empty else None,
            "max": float(series.max()) if not series.dropna().empty else None,
            "mean": float(series.mean()) if not series.dropna().empty else None,
            "median": float(series.median()) if not series.dropna().empty else None,
            "std": float(series.std()) if not series.dropna().empty else None,
        }

    return profile


def profile_categorical_columns(
    df: pd.DataFrame,
    max_unique_values: int = 20,
) -> dict:
    """
    Generate useful information about categorical columns.

    For columns with a manageable number of unique values,
    include their most common values.
    """

    profile = {}

    for column in df.columns:
        series = df[column]

        dtype = series.dtype
        
        if not (
            pd.api.types.is_object_dtype(dtype)
              or isinstance(dtype, pd.CategoricalDtype)):
            
            continue

        unique_count = int(series.nunique(dropna=True))

        column_profile = {
            "unique_count": unique_count,
        }

        if unique_count <= max_unique_values:
            value_counts = series.value_counts(dropna=False)

            column_profile["values"] = [
                {
                    "value": None if pd.isna(value) else str(value),
                    "count": int(count),
                }
                for value, count in value_counts.items()
            ]

        else:
            column_profile["top_values"] = [
                {
                    "value": None if pd.isna(value) else str(value),
                    "count": int(count),
                }
                for value, count in series.value_counts(
                    dropna=False
                ).head(10).items()
            ]

        profile[column] = column_profile

    return profile


def profile_date_columns(
    df: pd.DataFrame,
    date_columns: list[str],
) -> dict:
    """
    Generate date ranges and basic temporal information.
    """

    profile = {}

    for column in date_columns:
        series = pd.to_datetime(
            df[column],
            errors="coerce",
            format="mixed",
        ).dropna()

        if series.empty:
            profile[column] = {
                "min": None,
                "max": None,
            }
            continue

        profile[column] = {
            "min": series.min().strftime("%Y-%m-%d"),
            "max": series.max().strftime("%Y-%m-%d"),
        }

    return profile


def detect_missing_values(df: pd.DataFrame) -> dict:
    """
    Profile missing values for every column.
    """

    profile = {}

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percentage = (
            missing_count / len(df) * 100
            if len(df) > 0
            else 0.0
        )

        profile[column] = {
            "missing_count": missing_count,
            "missing_percentage": round(
                missing_percentage,
                4,
            ),
        }

    return profile


def detect_candidate_metrics(
    df: pd.DataFrame,
) -> list[str]:
    """
    Identify numeric columns that could plausibly be used
    as analytical measures.
    """

    metrics = []

    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()

        if series.empty:
            continue

        # Exclude columns that look like identifiers.
        unique_ratio = series.nunique() / len(series)

        if unique_ratio >= 0.98:
            continue

        metrics.append(column)

    return metrics


def detect_candidate_dimensions(
    df: pd.DataFrame,
) -> list[str]:
    """
    Identify columns that could plausibly be used
    as analytical dimensions.
    """

    dimensions = []

    for column in df.columns:
        dtype = df[column].dtype

        if (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
        ):
            dimensions.append(column)

    return dimensions

def profile_dataset(path: str | Path) -> dict:
    """
    Build a complete deterministic profile of a dataset.

    Supports:
        - CSV
        - Parquet
    """

    path = Path(path)

    df = load_dataset(path)

    column_types = detect_column_types(df)

    profile = {
        "dataset": {
            "name": path.name,
            "path": str(path),
            "format": path.suffix.lower().replace(".", ""),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
        },
        "columns": list(df.columns),
        "column_types": column_types,
        "missing_values": detect_missing_values(df),
        "numeric_profile": profile_numeric_columns(df),
        "categorical_profile": profile_categorical_columns(df),
        "date_profile": profile_date_columns(
            df,
            column_types["date"],
        ),
        "candidate_metrics": detect_candidate_metrics(df),
        "candidate_dimensions": column_types["categorical"],
    }

    return profile


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_dir
        / "data"
        / "raw"
        / "ecommerce_orders.csv"
    )

    profile = profile_dataset(dataset_path)

    print("\n=== DATASET PROFILE ===\n")

    print("Dataset:")
    print(profile["dataset"])

    print("\nColumn types:")
    for key, columns in profile["column_types"].items():
        print(f"  {key}: {columns}")

    print("\nCandidate metrics:")
    print(profile["candidate_metrics"])

    print("\nCandidate dimensions:")
    print(profile["candidate_dimensions"])

    print("\nDate profile:")
    print(profile["date_profile"])

    print("\nMissing values:")
    for column, values in profile["missing_values"].items():
        if values["missing_count"] > 0:
            print(f"  {column}: {values}")

    print("\nNumeric profile:")
    for column, values in profile["numeric_profile"].items():
        print(f"  {column}: {values}")