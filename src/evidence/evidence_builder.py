from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class EvidenceBuildError(Exception):
    """Raised when evidence cannot be constructed."""


@dataclass
class Evidence:
    """
    Structured evidence supporting an analytics answer.
    """

    question: str
    sql: str
    result: pd.DataFrame
    summary: str
    key_values: dict[str, object] = field(default_factory=dict)


class EvidenceBuilder:
    """
    Builds a structured evidence object from a question,
    executed SQL, and validated query result.

    This class does not generate SQL and does not execute queries.
    It only packages the evidence produced by earlier stages.
    """

    def __init__(self, max_rows: int = 10):
        if max_rows <= 0:
            raise ValueError("max_rows must be greater than zero.")

        self.max_rows = max_rows

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        question: str,
        sql: str,
        result: pd.DataFrame,
    ) -> None:
        """Validate evidence inputs."""

        if not isinstance(question, str) or not question.strip():
            raise EvidenceBuildError(
                "Question must be a non-empty string."
            )

        if not isinstance(sql, str) or not sql.strip():
            raise EvidenceBuildError(
                "SQL must be a non-empty string."
            )

        if not isinstance(result, pd.DataFrame):
            raise EvidenceBuildError(
                "Result must be a pandas DataFrame."
            )

        if result.empty:
            raise EvidenceBuildError(
                "Cannot build evidence from an empty result."
            )

    # ------------------------------------------------------------------
    # Key-value extraction
    # ------------------------------------------------------------------

    def _extract_key_values(
        self,
        result: pd.DataFrame,
    ) -> dict[str, object]:
        """
        Extract the first result row as a compact evidence dictionary.

        This is intentionally deterministic.
        """

        first_row = result.iloc[0]

        key_values: dict[str, object] = {}

        for column in result.columns:
            value = first_row[column]

            if pd.isna(value):
                key_values[column] = None
            else:
                # Convert numpy scalar values to normal Python values.
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except (ValueError, TypeError):
                        pass

                key_values[column] = value

        return key_values

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        result: pd.DataFrame,
    ) -> str:
        """
        Build a compact deterministic description of the result.

        The final natural-language answer will be generated later.
        This summary describes the evidence itself.
        """

        row_count = len(result)
        column_count = len(result.columns)

        return (
            f"Query returned {row_count} row"
            f"{'s' if row_count != 1 else ''} "
            f"across {column_count} column"
            f"{'s' if column_count != 1 else ''}."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        question: str,
        sql: str,
        result: pd.DataFrame,
    ) -> Evidence:
        """
        Build structured evidence from a validated query result.
        """

        self._validate_inputs(
            question,
            sql,
            result,
        )

        evidence_result = result.head(
            self.max_rows
        ).copy()

        key_values = self._extract_key_values(
            evidence_result
        )

        summary = self._build_summary(
            evidence_result
        )

        return Evidence(
            question=question,
            sql=sql,
            result=evidence_result,
            summary=summary,
            key_values=key_values,
        )


# ======================================================================
# TESTS
# ======================================================================

if __name__ == "__main__":

    print("\n=== EVIDENCE BUILDER TEST ===\n")

    builder = EvidenceBuilder(
        max_rows=10
    )

    # ------------------------------------------------------------------
    # TEST 1 — NORMAL RESULT
    # ------------------------------------------------------------------

    question = (
        "Which region generated the highest total sales?"
    )

    sql = """
    SELECT
        region,
        SUM(sales) AS sum_sales
    FROM ecommerce_orders
    GROUP BY region
    ORDER BY sum_sales DESC
    LIMIT 1;
    """

    result = pd.DataFrame(
        {
            "region": [
                "North America"
            ],
            "sum_sales": [
                13669971.27
            ],
        }
    )

    print("NORMAL RESULT")
    print("=" * 80)

    try:
        evidence = builder.build(
            question,
            sql,
            result,
        )

        print("Evidence build: PASS")
        print(f"Question: {evidence.question}")
        print(f"Summary: {evidence.summary}")
        print(f"Key values: {evidence.key_values}")

        print("\nSQL:")
        print(evidence.sql.strip())

        print("\nEvidence rows:")
        print(evidence.result.to_string(index=False))

    except EvidenceBuildError as exc:
        print("Evidence build: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 2 — MULTI-ROW RESULT
    # ------------------------------------------------------------------

    multi_result = pd.DataFrame(
        {
            "region": [
                "North America",
                "Europe",
                "Asia Pacific",
                "Latin America",
            ],
            "sum_sales": [
                13669971.27,
                10700974.13,
                8265143.74,
                3485465.26,
            ],
        }
    )

    print("\n\nMULTI-ROW RESULT")
    print("=" * 80)

    try:
        evidence = builder.build(
            "Show total sales by region.",
            sql,
            multi_result,
        )

        print("Evidence build: PASS")
        print(f"Summary: {evidence.summary}")
        print(f"Rows retained: {len(evidence.result)}")
        print(f"First row: {evidence.key_values}")

    except EvidenceBuildError as exc:
        print("Evidence build: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 3 — ROW LIMIT
    # ------------------------------------------------------------------

    limited_builder = EvidenceBuilder(
        max_rows=2
    )

    print("\n\nROW LIMIT")
    print("=" * 80)

    try:
        evidence = limited_builder.build(
            "Show total sales by region.",
            sql,
            multi_result,
        )

        print("Evidence build: PASS")
        print(f"Configured limit: 2")
        print(f"Rows retained: {len(evidence.result)}")

    except EvidenceBuildError as exc:
        print("Evidence build: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 4 — EMPTY RESULT
    # ------------------------------------------------------------------

    empty_result = pd.DataFrame(
        columns=[
            "region",
            "sum_sales",
        ]
    )

    print("\n\nEMPTY RESULT")
    print("=" * 80)

    try:
        builder.build(
            question,
            sql,
            empty_result,
        )

        print("Validation: FAIL")
        print("Empty result was incorrectly accepted.")

    except EvidenceBuildError as exc:
        print("Validation: BLOCKED")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 5 — INVALID QUESTION
    # ------------------------------------------------------------------

    print("\n\nINVALID QUESTION")
    print("=" * 80)

    try:
        builder.build(
            "",
            sql,
            result,
        )

        print("Validation: FAIL")
        print("Empty question was incorrectly accepted.")

    except EvidenceBuildError as exc:
        print("Validation: BLOCKED")
        print(f"Reason: {exc}")

    print("\n\nEvidence builder test completed.")