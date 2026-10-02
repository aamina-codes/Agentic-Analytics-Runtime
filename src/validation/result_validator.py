from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class ResultValidationError(Exception):
    """Raised when a query result fails validation."""


@dataclass
class ResultValidationReport:
    """Summary of result validation."""

    valid: bool
    row_count: int
    column_count: int
    columns: list[str]
    warnings: list[str]


class ResultValidator:
    """
    Validates DataFrame results produced by the analytics runtime.

    The validator checks structural integrity and basic numeric sanity.
    It does not attempt to determine whether a result is semantically
    correct for the user's question.
    """

    def __init__(
        self,
        allow_empty: bool = False,
    ):
        self.allow_empty = allow_empty

    # ------------------------------------------------------------------
    # Basic type validation
    # ------------------------------------------------------------------

    def _validate_type(self, result: object) -> pd.DataFrame:
        """Ensure the result is a pandas DataFrame."""

        if result is None:
            raise ResultValidationError(
                "Query returned no result."
            )

        if not isinstance(result, pd.DataFrame):
            raise ResultValidationError(
                "Query result must be a pandas DataFrame."
            )

        return result

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    def _validate_shape(self, result: pd.DataFrame) -> None:
        """Validate result dimensions."""

        if result.shape[1] == 0:
            raise ResultValidationError(
                "Query result contains zero columns."
            )

        if result.empty and not self.allow_empty:
            raise ResultValidationError(
                "Query returned zero rows."
            )

    # ------------------------------------------------------------------
    # Column validation
    # ------------------------------------------------------------------

    def _validate_columns(self, result: pd.DataFrame) -> None:
        """Validate result column names."""

        columns = list(result.columns)

        if any(
            column is None or str(column).strip() == ""
            for column in columns
        ):
            raise ResultValidationError(
                "Query result contains an empty column name."
            )

        normalized = [str(column).lower() for column in columns]

        if len(normalized) != len(set(normalized)):
            raise ResultValidationError(
                "Query result contains duplicate column names."
            )

    # ------------------------------------------------------------------
    # Numeric validation
    # ------------------------------------------------------------------

    def _validate_numeric_values(
        self,
        result: pd.DataFrame,
    ) -> list[str]:
        """
        Check numeric columns for non-finite values.

        NaN and infinite values can make downstream metrics and charts
        unreliable.
        """

        warnings: list[str] = []

        numeric_columns = result.select_dtypes(
            include=[np.number]
        ).columns

        for column in numeric_columns:
            values = result[column].to_numpy()

            if np.isinf(values).any():
                raise ResultValidationError(
                    f"Numeric column '{column}' contains infinite values."
                )

            if np.isnan(values).any():
                warnings.append(
                    f"Numeric column '{column}' contains null values."
                )

        return warnings

    # ------------------------------------------------------------------
    # Public validation
    # ------------------------------------------------------------------

    def validate(
        self,
        result: object,
    ) -> ResultValidationReport:
        """
        Validate a query result.

        Returns a validation report when successful.

        Raises:
            ResultValidationError: if the result fails validation.
        """

        dataframe = self._validate_type(result)

        self._validate_shape(dataframe)
        self._validate_columns(dataframe)

        warnings = self._validate_numeric_values(dataframe)

        return ResultValidationReport(
            valid=True,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            columns=[str(column) for column in dataframe.columns],
            warnings=warnings,
        )


# ======================================================================
# TESTS
# ======================================================================

if __name__ == "__main__":

    print("\n=== RESULT VALIDATOR TEST ===\n")

    validator = ResultValidator()

    # ------------------------------------------------------------------
    # VALID RESULT
    # ------------------------------------------------------------------

    valid_result = pd.DataFrame(
        {
            "region": [
                "North America",
                "Europe",
                "Asia Pacific",
            ],
            "total_sales": [
                13669971.27,
                10700974.13,
                8265143.74,
            ],
        }
    )

    print("VALID RESULT")
    print("=" * 80)

    try:
        report = validator.validate(valid_result)

        print("Validation: PASS")
        print(f"Rows: {report.row_count}")
        print(f"Columns: {report.column_count}")
        print(f"Column names: {report.columns}")
        print(f"Warnings: {report.warnings}")

    except ResultValidationError as exc:
        print("Validation: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # EMPTY RESULT
    # ------------------------------------------------------------------

    empty_result = pd.DataFrame(
        columns=["region", "total_sales"]
    )

    print("\n\nEMPTY RESULT")
    print("=" * 80)

    try:
        validator.validate(empty_result)

        print("Validation: FAIL")
        print("Empty result was incorrectly accepted.")

    except ResultValidationError as exc:
        print("Validation: BLOCKED")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # DUPLICATE COLUMNS
    # ------------------------------------------------------------------

    duplicate_columns = pd.DataFrame(
        [
            ["North America", 1000],
        ],
        columns=["region", "region"],
    )

    print("\n\nDUPLICATE COLUMNS")
    print("=" * 80)

    try:
        validator.validate(duplicate_columns)

        print("Validation: FAIL")
        print("Duplicate columns were incorrectly accepted.")

    except ResultValidationError as exc:
        print("Validation: BLOCKED")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # INFINITE VALUE
    # ------------------------------------------------------------------

    infinite_result = pd.DataFrame(
        {
            "metric": [100.0, np.inf],
        }
    )

    print("\n\nINFINITE VALUE")
    print("=" * 80)

    try:
        validator.validate(infinite_result)

        print("Validation: FAIL")
        print("Infinite value was incorrectly accepted.")

    except ResultValidationError as exc:
        print("Validation: BLOCKED")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # NULL NUMERIC VALUE
    # ------------------------------------------------------------------

    null_result = pd.DataFrame(
        {
            "metric": [100.0, np.nan, 200.0],
        }
    )

    print("\n\nNULL NUMERIC VALUE")
    print("=" * 80)

    try:
        report = validator.validate(null_result)

        print("Validation: PASS WITH WARNING")
        print(f"Warnings: {report.warnings}")

    except ResultValidationError as exc:
        print("Validation: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # ALLOW EMPTY
    # ------------------------------------------------------------------

    empty_allowed_validator = ResultValidator(
        allow_empty=True
    )

    print("\n\nEMPTY RESULT — ALLOWED MODE")
    print("=" * 80)

    try:
        report = empty_allowed_validator.validate(
            empty_result
        )

        print("Validation: PASS")
        print(f"Rows: {report.row_count}")
        print(f"Columns: {report.column_count}")

    except ResultValidationError as exc:
        print("Validation: FAIL")
        print(f"Reason: {exc}")

    print("\n\nResult validator test completed.")