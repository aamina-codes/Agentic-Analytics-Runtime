from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.models.schemas import QueryPlan


class ChartSelectionError(Exception):
    """Raised when a chart cannot be selected from the available result."""


@dataclass
class ChartSpec:
    """
    Deterministic visualization specification.

    This describes WHAT should be visualized.
    Actual rendering happens in chart_builder.py.
    """

    chart_type: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    title: Optional[str] = None
    reason: Optional[str] = None


class ChartSelector:
    """
    Deterministically selects an appropriate chart from a QueryPlan
    and validated query result.

    The selector does not generate charts.
    It only decides what visualization should be used.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _numeric_columns(
        self,
        result: pd.DataFrame,
    ) -> list[str]:
        """Return numeric columns from the result."""

        return [
            str(column)
            for column in result.select_dtypes(
                include="number"
            ).columns
        ]

    def _categorical_columns(
        self,
        result: pd.DataFrame,
    ) -> list[str]:
        """Return non-numeric columns from the result."""

        return [
            str(column)
            for column in result.columns
            if not pd.api.types.is_numeric_dtype(result[column])
        ]

    # ------------------------------------------------------------------
    # Metric column detection
    # ------------------------------------------------------------------

    def _select_metric_column(
        self,
        result: pd.DataFrame,
        plan: QueryPlan,
        excluded_columns: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Identify the numeric column representing the analytical metric.

        Example:

            year | sum_sales

        The year column is numeric, but it is the time dimension.
        Therefore sum_sales should be selected as the metric.
        """

        excluded = set(excluded_columns or [])

        numeric_columns = [
            column
            for column in self._numeric_columns(result)
            if column not in excluded
        ]

        if not numeric_columns:
            return None

        # Prefer a column whose name contains the planned metric.
        if plan.metric:
            metric_lower = plan.metric.lower()

            for column in numeric_columns:
                if metric_lower in column.lower():
                    return column

        # Prefer common aggregation prefixes.
        aggregation_prefixes = (
            "sum_",
            "avg_",
            "count_",
            "min_",
            "max_",
        )

        for column in numeric_columns:
            if column.lower().startswith(aggregation_prefixes):
                return column

        # Fall back to the first remaining numeric column.
        return numeric_columns[0]

    # ------------------------------------------------------------------
    # KPI detection
    # ------------------------------------------------------------------

    def _select_kpi(
        self,
        result: pd.DataFrame,
        plan: QueryPlan,
    ) -> Optional[ChartSpec]:
        """
        Select KPI when the result contains one row and one numeric value.
        """

        if len(result) != 1:
            return None

        numeric_columns = self._numeric_columns(result)

        if len(numeric_columns) != 1:
            return None

        y_column = numeric_columns[0]

        return ChartSpec(
            chart_type="kpi",
            y_column=y_column,
            title=plan.metric or y_column,
            reason=(
                "Single numeric result detected; "
                "KPI presentation is appropriate."
            ),
        )

    # ------------------------------------------------------------------
    # Bar chart detection
    # ------------------------------------------------------------------

    def _select_bar(
        self,
        result: pd.DataFrame,
        plan: QueryPlan,
    ) -> Optional[ChartSpec]:
        """
        Select bar chart for categorical comparisons.
        """

        categorical_columns = self._categorical_columns(result)

        if not categorical_columns:
            return None

        x_column = None

        # Prefer the planned dimension when available.
        if plan.dimension in result.columns:
            x_column = plan.dimension

        if x_column is None:
            x_column = categorical_columns[0]

        y_column = self._select_metric_column(
            result=result,
            plan=plan,
            excluded_columns=[x_column],
        )

        if y_column is None:
            return None

        return ChartSpec(
            chart_type="bar",
            x_column=x_column,
            y_column=y_column,
            title=f"{plan.metric or y_column} by {x_column}",
            reason=(
                "Categorical dimension with a numeric measure "
                "detected; bar chart selected."
            ),
        )

    # ------------------------------------------------------------------
    # Line chart detection
    # ------------------------------------------------------------------

    def _select_line(
        self,
        result: pd.DataFrame,
        plan: QueryPlan,
    ) -> Optional[ChartSpec]:
        """
        Select line chart for time-oriented results.
        """

        if plan.time_granularity is None:
            return None

        # Identify time/date dimension first.
        time_columns = [
            str(column)
            for column in result.columns
            if (
                pd.api.types.is_datetime64_any_dtype(result[column])
                or "date" in str(column).lower()
                or "year" in str(column).lower()
                or "month" in str(column).lower()
            )
        ]

        if not time_columns:
            return None

        x_column = time_columns[0]

        # IMPORTANT:
        # Exclude the time dimension from metric selection.
        y_column = self._select_metric_column(
            result=result,
            plan=plan,
            excluded_columns=[x_column],
        )

        if y_column is None:
            return None

        return ChartSpec(
            chart_type="line",
            x_column=x_column,
            y_column=y_column,
            title=f"{plan.metric or y_column} over time",
            reason=(
                "Time-oriented query detected; "
                "line chart selected."
            ),
        )

    # ------------------------------------------------------------------
    # Table fallback
    # ------------------------------------------------------------------

    def _select_table(
        self,
        result: pd.DataFrame,
        plan: QueryPlan,
    ) -> ChartSpec:
        """Use a table when no specialized visualization applies."""

        return ChartSpec(
            chart_type="table",
            title="Query Result",
            reason=(
                "No specialized visualization matched the result; "
                "table selected as a safe fallback."
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        result: pd.DataFrame,
        plan: QueryPlan,
    ) -> ChartSpec:
        """
        Select the most appropriate visualization.

        Selection order:

        1. KPI
        2. Line chart for time-based results
        3. Bar chart for categorical comparisons
        4. Table fallback
        """

        if not isinstance(result, pd.DataFrame):
            raise ChartSelectionError(
                "Chart selection requires a pandas DataFrame."
            )

        if result.empty:
            raise ChartSelectionError(
                "Cannot select a chart for an empty result."
            )

        # 1. Single-value KPI.
        kpi = self._select_kpi(result, plan)

        if kpi is not None:
            return kpi

        # 2. Time-series line chart.
        line = self._select_line(result, plan)

        if line is not None:
            return line

        # 3. Categorical comparison.
        bar = self._select_bar(result, plan)

        if bar is not None:
            return bar

        # 4. Safe fallback.
        return self._select_table(result, plan)


# ======================================================================
# TESTS
# ======================================================================

if __name__ == "__main__":

    print("\n=== CHART SELECTOR TEST ===\n")

    selector = ChartSelector()

    # ------------------------------------------------------------------
    # TEST 1 — BAR CHART
    # ------------------------------------------------------------------

    bar_result = pd.DataFrame(
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

    bar_plan = QueryPlan(
        question="Which region generated the highest total sales?",
        intent="ranking",
        metric="sales",
        aggregation="sum",
        dimension="region",
    )

    print("BAR CHART")
    print("=" * 80)

    try:
        spec = selector.select(
            result=bar_result,
            plan=bar_plan,
        )

        print("Selection: PASS")
        print(spec)

    except ChartSelectionError as exc:
        print("Selection: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 2 — KPI
    # ------------------------------------------------------------------

    kpi_result = pd.DataFrame(
        {
            "sum_sales": [
                18173443.65,
            ]
        }
    )

    kpi_plan = QueryPlan(
        question="What were total sales in 2024?",
        intent="aggregation",
        metric="sales",
        aggregation="sum",
    )

    print("\n\nKPI")
    print("=" * 80)

    try:
        spec = selector.select(
            result=kpi_result,
            plan=kpi_plan,
        )

        print("Selection: PASS")
        print(spec)

    except ChartSelectionError as exc:
        print("Selection: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 3 — LINE CHART
    # ------------------------------------------------------------------

    line_result = pd.DataFrame(
        {
            "year": [
                2024,
                2025,
            ],
            "sum_sales": [
                18173443.65,
                17948110.75,
            ],
        }
    )

    line_plan = QueryPlan(
        question="How did total sales change by year?",
        intent="comparison",
        metric="sales",
        aggregation="sum",
        time_granularity="year",
    )

    print("\n\nLINE CHART")
    print("=" * 80)

    try:
        spec = selector.select(
            result=line_result,
            plan=line_plan,
        )

        print("Selection: PASS")
        print(spec)

        # Explicit correctness check.
        if (
            spec.x_column == "year"
            and spec.y_column == "sum_sales"
        ):
            print("Axis mapping: PASS")
        else:
            print("Axis mapping: FAIL")
            print(
                f"Expected x='year', y='sum_sales'; "
                f"got x='{spec.x_column}', y='{spec.y_column}'"
            )

    except ChartSelectionError as exc:
        print("Selection: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 4 — CATEGORICAL FALLBACK
    # ------------------------------------------------------------------

    table_result = pd.DataFrame(
        {
            "customer_segment": [
                "Consumer",
                "Corporate",
                "Home Office",
            ],
            "avg_sales": [
                462.27,
                900.36,
                602.47,
            ],
        }
    )

    table_plan = QueryPlan(
        question="Show me the customer segment data.",
        intent="aggregation",
        metric=None,
        aggregation=None,
    )

    print("\n\nCATEGORICAL RESULT")
    print("=" * 80)

    try:
        spec = selector.select(
            result=table_result,
            plan=table_plan,
        )

        print("Selection: PASS")
        print(spec)

    except ChartSelectionError as exc:
        print("Selection: FAIL")
        print(f"Reason: {exc}")

    print("\n\nChart selector test completed.")