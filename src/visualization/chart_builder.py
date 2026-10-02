from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from src.visualization.chart_selector import ChartSpec


class ChartBuildError(Exception):
    """Raised when a chart cannot be built from the supplied result."""


class ChartBuilder:
    """
    Builds Plotly figures from deterministic ChartSpec objects.

    The builder does not decide which chart should be used.
    That decision belongs to ChartSelector.
    """

    def __init__(self):
        self.default_height = 450

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_columns(
        self,
        result: pd.DataFrame,
        spec: ChartSpec,
    ) -> None:
        """Ensure requested chart columns exist."""

        for column in [spec.x_column, spec.y_column]:
            if column is None:
                continue

            if column not in result.columns:
                raise ChartBuildError(
                    f"Column '{column}' required by chart was not found."
                )

    def _base_layout(
        self,
        figure: go.Figure,
        title: Optional[str],
    ) -> go.Figure:
        """Apply consistent presentation settings."""

        figure.update_layout(
            title=title,
            height=self.default_height,
            template="plotly_white",
            margin=dict(
                l=60,
                r=30,
                t=70,
                b=60,
            ),
            hovermode="x unified",
        )

        return figure

    # ------------------------------------------------------------------
    # Bar chart
    # ------------------------------------------------------------------

    def _build_bar(
        self,
        result: pd.DataFrame,
        spec: ChartSpec,
    ) -> go.Figure:
        """Build a categorical bar chart."""

        if spec.x_column is None or spec.y_column is None:
            raise ChartBuildError(
                "Bar chart requires both x_column and y_column."
            )

        figure = go.Figure()

        figure.add_trace(
            go.Bar(
                x=result[spec.x_column],
                y=result[spec.y_column],
                text=result[spec.y_column],
                texttemplate="%{text:.2f}",
                textposition="outside",
                hovertemplate=(
                    f"{spec.x_column}: %{{x}}"
                    "<br>"
                    f"{spec.y_column}: %{{y:,.2f}}"
                    "<extra></extra>"
                ),
            )
        )

        figure.update_xaxes(
            title_text=spec.x_column
        )

        figure.update_yaxes(
            title_text=spec.y_column
        )

        return self._base_layout(
            figure,
            spec.title,
        )

    # ------------------------------------------------------------------
    # Line chart
    # ------------------------------------------------------------------

    def _build_line(
        self,
        result: pd.DataFrame,
        spec: ChartSpec,
    ) -> go.Figure:
        """Build a time-series line chart."""

        if spec.x_column is None or spec.y_column is None:
            raise ChartBuildError(
                "Line chart requires both x_column and y_column."
            )

        figure = go.Figure()

        figure.add_trace(
            go.Scatter(
                x=result[spec.x_column],
                y=result[spec.y_column],
                mode="lines+markers",
                hovertemplate=(
                    f"{spec.x_column}: %{{x}}"
                    "<br>"
                    f"{spec.y_column}: %{{y:,.2f}}"
                    "<extra></extra>"
                ),
            )
        )

        figure.update_xaxes(
            title_text=spec.x_column
        )

        figure.update_yaxes(
            title_text=spec.y_column
        )

        return self._base_layout(
            figure,
            spec.title,
        )

    # ------------------------------------------------------------------
    # KPI
    # ------------------------------------------------------------------

    def _build_kpi(
        self,
        result: pd.DataFrame,
        spec: ChartSpec,
    ) -> go.Figure:
        """Build a single-value KPI visualization."""

        if spec.y_column is None:
            raise ChartBuildError(
                "KPI requires y_column."
            )

        if len(result) != 1:
            raise ChartBuildError(
                "KPI requires exactly one result row."
            )

        value = result.iloc[0][spec.y_column]

        if pd.isna(value):
            raise ChartBuildError(
                "KPI value cannot be null."
            )

        figure = go.Figure(
            go.Indicator(
                mode="number",
                value=float(value),
                title={
                    "text": spec.title or spec.y_column
                },
                number={
                    "valueformat": ",.2f"
                },
            )
        )

        return self._base_layout(
            figure,
            None,
        )

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _build_table(
        self,
        result: pd.DataFrame,
        spec: ChartSpec,
    ) -> go.Figure:
        """
        Build a Plotly table.

        This is a fallback visualization and is intentionally simple.
        """

        figure = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(result.columns),
                        align="left",
                    ),
                    cells=dict(
                        values=[
                            result[column].tolist()
                            for column in result.columns
                        ],
                        align="left",
                    ),
                )
            ]
        )

        return self._base_layout(
            figure,
            spec.title or "Query Result",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        result: pd.DataFrame,
        spec: ChartSpec,
    ) -> go.Figure:
        """
        Build a Plotly figure from a ChartSpec.
        """

        if not isinstance(result, pd.DataFrame):
            raise ChartBuildError(
                "Chart builder requires a pandas DataFrame."
            )

        if result.empty:
            raise ChartBuildError(
                "Cannot build a chart from an empty result."
            )

        self._validate_columns(
            result,
            spec,
        )

        chart_type = spec.chart_type.lower()

        if chart_type == "bar":
            return self._build_bar(
                result,
                spec,
            )

        if chart_type == "line":
            return self._build_line(
                result,
                spec,
            )

        if chart_type == "kpi":
            return self._build_kpi(
                result,
                spec,
            )

        if chart_type == "table":
            return self._build_table(
                result,
                spec,
            )

        raise ChartBuildError(
            f"Unsupported chart type: {spec.chart_type}"
        )


# ======================================================================
# TESTS
# ======================================================================

if __name__ == "__main__":

    print("\n=== CHART BUILDER TEST ===\n")

    builder = ChartBuilder()

    # ------------------------------------------------------------------
    # TEST 1 — BAR
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

    bar_spec = ChartSpec(
        chart_type="bar",
        x_column="region",
        y_column="sum_sales",
        title="Total Sales by Region",
    )

    print("BAR CHART")
    print("=" * 80)

    try:
        figure = builder.build(
            bar_result,
            bar_spec,
        )

        print("Build: PASS")
        print(f"Traces: {len(figure.data)}")
        print(f"Chart type: {figure.data[0].type}")

    except ChartBuildError as exc:
        print("Build: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 2 — LINE
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

    line_spec = ChartSpec(
        chart_type="line",
        x_column="year",
        y_column="sum_sales",
        title="Sales Over Time",
    )

    print("\n\nLINE CHART")
    print("=" * 80)

    try:
        figure = builder.build(
            line_result,
            line_spec,
        )

        print("Build: PASS")
        print(f"Traces: {len(figure.data)}")
        print(f"Chart type: {figure.data[0].type}")

    except ChartBuildError as exc:
        print("Build: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 3 — KPI
    # ------------------------------------------------------------------

    kpi_result = pd.DataFrame(
        {
            "sum_sales": [
                18173443.65,
            ]
        }
    )

    kpi_spec = ChartSpec(
        chart_type="kpi",
        y_column="sum_sales",
        title="Total Sales 2024",
    )

    print("\n\nKPI")
    print("=" * 80)

    try:
        figure = builder.build(
            kpi_result,
            kpi_spec,
        )

        print("Build: PASS")
        print(f"Traces: {len(figure.data)}")
        print(f"Chart type: {figure.data[0].type}")

    except ChartBuildError as exc:
        print("Build: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 4 — TABLE
    # ------------------------------------------------------------------

    table_spec = ChartSpec(
        chart_type="table",
        title="Sales Results",
    )

    print("\n\nTABLE")
    print("=" * 80)

    try:
        figure = builder.build(
            bar_result,
            table_spec,
        )

        print("Build: PASS")
        print(f"Traces: {len(figure.data)}")
        print(f"Chart type: {figure.data[0].type}")

    except ChartBuildError as exc:
        print("Build: FAIL")
        print(f"Reason: {exc}")

    # ------------------------------------------------------------------
    # TEST 5 — INVALID COLUMN
    # ------------------------------------------------------------------

    invalid_spec = ChartSpec(
        chart_type="bar",
        x_column="region",
        y_column="does_not_exist",
        title="Invalid Chart",
    )

    print("\n\nINVALID COLUMN")
    print("=" * 80)

    try:
        builder.build(
            bar_result,
            invalid_spec,
        )

        print("Validation: FAIL")
        print("Invalid column was incorrectly accepted.")

    except ChartBuildError as exc:
        print("Validation: BLOCKED")
        print(f"Reason: {exc}")

    print("\n\nChart builder test completed.")