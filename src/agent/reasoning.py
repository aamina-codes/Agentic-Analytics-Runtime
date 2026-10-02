from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.evidence.evidence_builder import Evidence


class ReasoningError(Exception):
    """Raised when an answer cannot be generated from the evidence."""


@dataclass
class Answer:
    """
    Human-readable answer generated from deterministic evidence.
    """

    question: str
    answer: str
    evidence_summary: str
    sql: str


class ReasoningEngine:
    """
    Converts structured evidence into a concise human-readable answer.

    Important:
    - Does not execute SQL.
    - Does not generate SQL.
    - Does not invent metrics.
    - Uses only values present in the supplied evidence.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_value(self, value: object) -> str:
        """Format a value for human-readable output."""

        if value is None:
            return "N/A"

        if isinstance(value, bool):
            return str(value)

        if isinstance(value, int):
            return f"{value:,}"

        if isinstance(value, float):
            return f"{value:,.2f}"

        return str(value)

    def _format_metric(
        self,
        column_name: str,
        value: object,
    ) -> str:
        """
        Format a metric according to common analytics naming conventions.
        """

        formatted = self._format_value(value)

        lowered = column_name.lower()

        if "percentage" in lowered or "percent" in lowered:
            return f"{formatted}%"

        if any(
            keyword in lowered
            for keyword in [
                "sales",
                "revenue",
                "profit",
                "cost",
            ]
        ):
            return f"${formatted}"

        return formatted

    # ------------------------------------------------------------------
    # Question interpretation
    # ------------------------------------------------------------------

    def _detect_question_type(
        self,
        question: str,
    ) -> str:
        """
        Detect the broad answer style.

        Comparison questions are checked before ranking questions because
        a comparison may contain words such as "highest" while actually
        asking for a difference between groups.
        """

        lowered = question.lower()

        # --------------------------------------------------------------
        # Comparison questions
        # --------------------------------------------------------------

        if (
            (
                "difference" in lowered
                or "differ" in lowered
                or "differed" in lowered
            )
            and (
                "between" in lowered
                or "among" in lowered
            )
        ):
            return "comparison"

        if (
            "how did" in lowered
            and (
                "differ" in lowered
                or "difference" in lowered
            )
        ):
            return "comparison"

        # --------------------------------------------------------------
        # Ranking questions
        # --------------------------------------------------------------

        if any(
            phrase in lowered
            for phrase in [
                "highest",
                "top",
                "most",
                "largest",
                "best",
            ]
        ):
            return "ranking"

        if any(
            phrase in lowered
            for phrase in [
                "lowest",
                "least",
                "smallest",
                "worst",
            ]
        ):
            return "ranking_low"

        # --------------------------------------------------------------
        # Percentage questions
        # --------------------------------------------------------------

        if any(
            phrase in lowered
            for phrase in [
                "percentage",
                "percent",
                "proportion",
                "share",
            ]
        ):
            return "percentage"

        # --------------------------------------------------------------
        # Count questions
        # --------------------------------------------------------------

        if any(
            phrase in lowered
            for phrase in [
                "how many",
                "count",
                "number of",
            ]
        ):
            return "count"

        # --------------------------------------------------------------
        # Aggregation questions
        # --------------------------------------------------------------

        if any(
            phrase in lowered
            for phrase in [
                "total",
                "sum",
                "revenue",
            ]
        ):
            return "aggregation"

        # --------------------------------------------------------------
        # Average questions
        # --------------------------------------------------------------

        if any(
            phrase in lowered
            for phrase in [
                "average",
                "mean",
                "avg",
            ]
        ):
            return "average"

        return "general"

    # ------------------------------------------------------------------
    # Answer builders
    # ------------------------------------------------------------------

    def _build_ranking_answer(
        self,
        evidence: Evidence,
        descending: bool = True,
    ) -> str:
        """
        Build an answer for ranking questions.

        Expected result shape:
            dimension | metric
        """

        result = evidence.result

        if result.empty:
            raise ReasoningError(
                "Cannot answer a ranking question from empty evidence."
            )

        if len(result.columns) < 2:
            raise ReasoningError(
                "Ranking evidence requires at least two columns."
            )

        dimension_column = result.columns[0]
        metric_column = result.columns[1]

        row = result.iloc[0]

        dimension_value = row[dimension_column]
        metric_value = row[metric_column]

        metric_text = self._format_metric(
            metric_column,
            metric_value,
        )

        direction = "highest" if descending else "lowest"

        return (
            f"{dimension_value} generated the {direction} "
            f"{self._humanize_metric(metric_column)}, "
            f"with {metric_text}."
        )

    def _build_percentage_answer(
        self,
        evidence: Evidence,
    ) -> str:
        """Build an answer for percentage questions."""

        result = evidence.result

        if result.empty:
            raise ReasoningError(
                "Cannot answer a percentage question from empty evidence."
            )

        column = result.columns[0]
        value = result.iloc[0][column]

        formatted = self._format_value(value)

        return f"The result is {formatted}%."

    def _build_count_answer(
        self,
        evidence: Evidence,
    ) -> str:
        """Build an answer for count questions."""

        result = evidence.result

        if result.empty:
            raise ReasoningError(
                "Cannot answer a count question from empty evidence."
            )

        if len(result.columns) == 1:
            column = result.columns[0]
            value = result.iloc[0][column]

            return (
                f"The count is "
                f"{self._format_value(value)}."
            )

        dimension_column = result.columns[0]
        metric_column = result.columns[1]

        dimension = result.iloc[0][dimension_column]
        count = result.iloc[0][metric_column]

        return (
            f"{dimension} has the highest count, "
            f"with {self._format_value(count)}."
        )

    def _build_single_value_answer(
        self,
        evidence: Evidence,
        question_type: str,
    ) -> str:
        """Build an answer for a single-value result."""

        result = evidence.result

        if result.empty:
            raise ReasoningError(
                "Cannot answer from empty evidence."
            )

        column = result.columns[0]
        value = result.iloc[0][column]

        metric_text = self._format_metric(
            column,
            value,
        )

        if question_type == "average":
            return (
                f"The average "
                f"{self._humanize_metric(column)} "
                f"is {metric_text}."
            )

        metric_name = self._humanize_metric(column)

        if metric_name.startswith("total "):
            return (
                f"The {metric_name} "
                f"is {metric_text}."
            )

        return (
            f"The total {metric_name} "
            f"is {metric_text}."
        )

    def _build_comparison_answer(
        self,
        evidence: Evidence,
    ) -> str:
        """
        Build an answer for comparison questions.

        q17 is intentionally a scalar result:

            difference
            ----------
            9208933.63

        The SQL layer has already performed the comparison, so the
        reasoning layer only needs to explain the resulting value.
        """

        result = evidence.result

        if result.empty:
            raise ReasoningError(
                "Cannot answer a comparison question from empty evidence."
            )

        if len(result.columns) != 1:
            raise ReasoningError(
                "Comparison evidence must contain exactly one result column."
            )

        column = result.columns[0]
        value = result.iloc[0][column]

        formatted = self._format_metric(
            column,
            value,
        )

        lowered = column.lower()

        if (
            "difference" in lowered
            or "change" in lowered
            or "delta" in lowered
        ):
            return (
                f"The difference is {formatted}."
            )

        return (
            f"The comparison result is {formatted}."
        )

    # ------------------------------------------------------------------
    # Generic answer
    # ------------------------------------------------------------------

    def _build_general_answer(
        self,
        evidence: Evidence,
    ) -> str:
        """Build a generic answer from the first evidence row."""

        result = evidence.result

        if result.empty:
            raise ReasoningError(
                "Cannot answer from empty evidence."
            )

        row = result.iloc[0]

        parts: list[str] = []

        for column in result.columns:
            value = row[column]

            parts.append(
                f"{column} = {self._format_value(value)}"
            )

        return (
            "The query returned: "
            + ", ".join(parts)
            + "."
        )

    # ------------------------------------------------------------------
    # Human-readable column names
    # ------------------------------------------------------------------

    def _humanize_metric(
        self,
        column_name: str,
    ) -> str:
        """Convert SQL-style metric names into readable text."""

        name = column_name.lower()

        replacements = {
            "sum_": "total ",
            "avg_": "average ",
            "count_": "count ",
            "min_": "minimum ",
            "max_": "maximum ",
        }

        for prefix, replacement in replacements.items():
            if name.startswith(prefix):
                name = replacement + name[len(prefix):]
                break

        return name.replace("_", " ")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        evidence: Evidence,
    ) -> Answer:
        """
        Generate a human-readable answer from evidence.
        """

        if not isinstance(evidence, Evidence):
            raise ReasoningError(
                "ReasoningEngine requires an Evidence object."
            )

        if evidence.result.empty:
            raise ReasoningError(
                "Cannot generate an answer from empty evidence."
            )

        question_type = self._detect_question_type(
            evidence.question
        )

        if question_type == "comparison":
            answer_text = self._build_comparison_answer(
                evidence
            )

        elif question_type == "ranking":
            answer_text = self._build_ranking_answer(
                evidence,
                descending=True,
            )

        elif question_type == "ranking_low":
            answer_text = self._build_ranking_answer(
                evidence,
                descending=False,
            )

        elif question_type == "percentage":
            answer_text = self._build_percentage_answer(
                evidence
            )

        elif question_type == "count":
            answer_text = self._build_count_answer(
                evidence
            )

        elif question_type in {
            "aggregation",
            "average",
        }:
            answer_text = self._build_single_value_answer(
                evidence,
                question_type,
            )

        else:
            answer_text = self._build_general_answer(
                evidence
            )

        return Answer(
            question=evidence.question,
            answer=answer_text,
            evidence_summary=evidence.summary,
            sql=evidence.sql,
        )


# ======================================================================
# TESTS
# ======================================================================

if __name__ == "__main__":

    print("\n=== REASONING ENGINE TEST ===\n")

    from src.evidence.evidence_builder import EvidenceBuilder

    evidence_builder = EvidenceBuilder()

    reasoning_engine = ReasoningEngine()

    # ------------------------------------------------------------------
    # TEST 1 — RANKING
    # ------------------------------------------------------------------

    ranking_result = pd.DataFrame(
        {
            "region": [
                "North America"
            ],
            "sum_sales": [
                13669971.27
            ],
        }
    )

    ranking_question = (
        "Which region generated the highest total sales?"
    )

    ranking_sql = """
    SELECT
        region,
        SUM(sales) AS sum_sales
    FROM ecommerce_orders
    GROUP BY region
    ORDER BY sum_sales DESC
    LIMIT 1;
    """

    ranking_evidence = evidence_builder.build(
        ranking_question,
        ranking_sql,
        ranking_result,
    )

    print("RANKING QUESTION")
    print("=" * 80)

    try:
        answer = reasoning_engine.answer(
            ranking_evidence
        )

        print("Reasoning: PASS")
        print(f"Answer: {answer.answer}")

    except ReasoningError as exc:
        print("Reasoning: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 2 — PERCENTAGE
    # ------------------------------------------------------------------

    percentage_result = pd.DataFrame(
        {
            "percentage": [
                4.69
            ]
        }
    )

    percentage_question = (
        "What percentage of orders were cancelled?"
    )

    percentage_sql = """
    SELECT
        ROUND(
            100.0 * COUNT(*) FILTER (
                WHERE order_status = 'Cancelled'
            ) / COUNT(*),
            2
        ) AS percentage
    FROM ecommerce_orders;
    """

    percentage_evidence = evidence_builder.build(
        percentage_question,
        percentage_sql,
        percentage_result,
    )

    print("\n\nPERCENTAGE QUESTION")
    print("=" * 80)

    try:
        answer = reasoning_engine.answer(
            percentage_evidence
        )

        print("Reasoning: PASS")
        print(f"Answer: {answer.answer}")

    except ReasoningError as exc:
        print("Reasoning: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 3 — AGGREGATION
    # ------------------------------------------------------------------

    aggregation_result = pd.DataFrame(
        {
            "sum_sales": [
                18173443.65
            ]
        }
    )

    aggregation_question = (
        "What were the total sales in 2024?"
    )

    aggregation_sql = """
    SELECT
        SUM(sales) AS sum_sales
    FROM ecommerce_orders
    WHERE EXTRACT(YEAR FROM order_date) = 2024;
    """

    aggregation_evidence = evidence_builder.build(
        aggregation_question,
        aggregation_sql,
        aggregation_result,
    )

    print("\n\nAGGREGATION QUESTION")
    print("=" * 80)

    try:
        answer = reasoning_engine.answer(
            aggregation_evidence
        )

        print("Reasoning: PASS")
        print(f"Answer: {answer.answer}")

    except ReasoningError as exc:
        print("Reasoning: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 4 — COMPARISON
    # ------------------------------------------------------------------

    comparison_result = pd.DataFrame(
        {
            "difference": [
                9208933.63
            ]
        }
    )

    comparison_question = (
        "How did total sales differ between the two "
        "customer segments with the highest average "
        "order value?"
    )

    comparison_sql = """
    WITH top_groups AS (
        SELECT
            customer_segment
        FROM ecommerce_orders
        GROUP BY customer_segment
        ORDER BY AVG(sales) DESC
        LIMIT 2
    ),
    group_totals AS (
        SELECT
            customer_segment,
            SUM(sales) AS total_sales
        FROM ecommerce_orders
        WHERE customer_segment IN (
            SELECT customer_segment
            FROM top_groups
        )
        GROUP BY customer_segment
    )
    SELECT
        MAX(total_sales) - MIN(total_sales) AS difference
    FROM group_totals;
    """

    comparison_evidence = evidence_builder.build(
        comparison_question,
        comparison_sql,
        comparison_result,
    )

    print("\n\nCOMPARISON QUESTION")
    print("=" * 80)

    try:
        answer = reasoning_engine.answer(
            comparison_evidence
        )

        print("Reasoning: PASS")
        print(f"Answer: {answer.answer}")

    except ReasoningError as exc:
        print("Reasoning: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 5 — INVALID EVIDENCE
    # ------------------------------------------------------------------

    print("\n\nINVALID EVIDENCE")
    print("=" * 80)

    try:
        reasoning_engine.answer(
            "not evidence"
        )

        print("Validation: FAIL")
        print("Invalid evidence was incorrectly accepted.")

    except ReasoningError as exc:
        print("Validation: BLOCKED")
        print(f"Reason: {exc}")

    print("\n\nReasoning engine test completed.")