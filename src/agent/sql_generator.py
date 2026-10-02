from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.schemas import QueryPlan


class SQLGenerationError(Exception):
    """Raised when a QueryPlan cannot be converted into valid SQL."""


class SQLGenerator:
    """
    Deterministic SQL generator.

    Converts a validated QueryPlan into DuckDB-compatible SQL.

    Important design principle:

        QueryPlan → SQL

    No LLM is involved here.

    The planner decides what the question means.
    The SQL generator decides how that structured plan
    should be expressed as SQL.
    """

    ALLOWED_AGGREGATIONS = {
        "sum": "SUM",
        "avg": "AVG",
        "count": "COUNT",
        "min": "MIN",
        "max": "MAX",
    }

    ALLOWED_OPERATORS = {
    "equals": "=",
    "not_equals": "<>",
    "greater_than": ">",
    "greater_than_or_equal": ">=",
    "less_than": "<",
    "less_than_or_equal": "<=",

    # Also accept SQL-style operators emitted by the planner.
    "=": "=",
    "<>": "<>",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
}

    ALLOWED_DATE_OPERATORS = {
        "year_equals",
    }

    def __init__(
        self,
        table_name: str = "ecommerce_orders",
    ):
        self.table_name = table_name

    # ------------------------------------------------------------------
    # IDENTIFIER VALIDATION
    # ------------------------------------------------------------------

    def _validate_identifier(
        self,
        identifier: str,
    ) -> str:
        """
        Validate a SQL identifier.

        Only simple alphanumeric identifiers and underscores
        are allowed.
        """

        if not identifier:
            raise SQLGenerationError(
                "SQL identifier cannot be empty."
            )

        if not all(
            character.isalnum()
            or character == "_"
            for character in identifier
        ):
            raise SQLGenerationError(
                f"Unsafe SQL identifier: {identifier}"
            )

        return identifier

    # ------------------------------------------------------------------
    # VALUE ESCAPING
    # ------------------------------------------------------------------

    def _escape_sql_value(
        self,
        value: Any,
    ) -> str:
        """
        Convert a Python value into a safe SQL literal.
        """

        if value is None:
            return "NULL"

        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"

        if isinstance(value, (int, float)):
            return str(value)

        value_string = str(value)

        value_string = value_string.replace(
            "'",
            "''",
        )

        return f"'{value_string}'"

    # ------------------------------------------------------------------
    # AGGREGATION
    # ------------------------------------------------------------------

    def _build_aggregation(
        self,
        aggregation: str,
        metric: str | None,
    ) -> str:
        """
        Build an aggregate SQL expression.
        """

        aggregation = aggregation.lower()

        if aggregation not in self.ALLOWED_AGGREGATIONS:
            raise SQLGenerationError(
                f"Unsupported aggregation: {aggregation}"
            )

        sql_function = self.ALLOWED_AGGREGATIONS[
            aggregation
        ]

        if aggregation == "count":
            if metric:
                metric = self._validate_identifier(
                    metric
                )
                return f"{sql_function}({metric})"

            return f"{sql_function}(*)"

        if not metric:
            raise SQLGenerationError(
                f"Metric required for {aggregation}."
            )

        metric = self._validate_identifier(
            metric
        )

        return f"{sql_function}({metric})"

    # ------------------------------------------------------------------
    # STANDARD FILTER
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        filter_condition: dict[str, Any],
    ) -> str:
        """
        Convert a standard planner filter into SQL.
        """

        column = filter_condition.get(
            "column"
        )

        operator = filter_condition.get(
            "operator"
        )

        value = filter_condition.get(
            "value"
        )

        if not column:
            raise SQLGenerationError(
                "Filter column is missing."
            )

        column = self._validate_identifier(
            column
        )

        if operator in self.ALLOWED_OPERATORS:
            sql_operator = self.ALLOWED_OPERATORS[
                operator
            ]

            return (
                f"{column} "
                f"{sql_operator} "
                f"{self._escape_sql_value(value)}"
            )

        if operator == "year_equals":
            if not isinstance(
                value,
                int,
            ):
                raise SQLGenerationError(
                    "year_equals requires an integer year."
                )

            return (
                f"EXTRACT(YEAR FROM {column}) "
                f"= {value}"
            )

        raise SQLGenerationError(
            f"Unsupported filter operator: {operator}"
        )

    # ------------------------------------------------------------------
    # WHERE CLAUSE
    # ------------------------------------------------------------------

    def _build_where_clause(
        self,
        filters: list[dict[str, Any]],
    ) -> str:
        """
        Build a WHERE clause from planner filters.
        """

        if not filters:
            return ""

        conditions = [
            self._build_filter(
                filter_condition
            )
            for filter_condition in filters
        ]

        return (
            "WHERE "
            + " AND ".join(conditions)
        )

    # ------------------------------------------------------------------
    # SORT DIRECTION
    # ------------------------------------------------------------------

    def _validate_sort_direction(
        self,
        direction: str | None,
    ) -> str:
        if not direction:
            raise SQLGenerationError(
                "Sort direction is required."
            )

        direction = direction.upper()

        if direction not in {
            "ASC",
            "DESC",
        }:
            raise SQLGenerationError(
                f"Unsupported sort direction: {direction}"
            )

        return direction

    # ------------------------------------------------------------------
    # LIMIT
    # ------------------------------------------------------------------

    def _build_limit(
        self,
        limit: int | None,
    ) -> str:
        if limit is None:
            return ""

        if limit <= 0:
            raise SQLGenerationError(
                "LIMIT must be greater than zero."
            )

        return f"LIMIT {int(limit)}"

    # ------------------------------------------------------------------
    # STANDARD QUERY
    # ------------------------------------------------------------------

    def _generate_standard_query(
        self,
        plan: QueryPlan,
    ) -> str:
        """
        Generate SQL for standard analytical queries.

        Supports:

        - aggregations
        - grouped aggregations
        - rankings
        - filters
        - limits
        """

        if not plan.aggregation:
            raise SQLGenerationError(
                "Query plan is missing aggregation."
            )

        aggregate_expression = (
            self._build_aggregation(
                plan.aggregation,
                plan.metric,
            )
        )

        table = self._validate_identifier(
            self.table_name
        )

        where_clause = (
            self._build_where_clause(
                plan.filters
            )
        )

        if plan.dimension:
            dimension = (
                self._validate_identifier(
                    plan.dimension
                )
            )

            alias = (
                f"{plan.aggregation}_"
                f"{plan.metric or 'rows'}"
            )

            select_clause = (
                f"{dimension}, "
                f"{aggregate_expression} "
                f"AS {alias}"
            )

            group_clause = (
                f"GROUP BY {dimension}"
            )

            sort_column = alias

        else:
            alias = (
                f"{plan.aggregation}_"
                f"{plan.metric or 'rows'}"
            )

            select_clause = (
                f"{aggregate_expression} "
                f"AS {alias}"
            )

            group_clause = ""

            sort_column = (
                alias
                if plan.sort_column
                else None
            )

        sql_parts = [
            "SELECT",
            f"    {select_clause}",
            f"FROM {table}",
        ]

        if where_clause:
            sql_parts.append(
                where_clause
            )

        if group_clause:
            sql_parts.append(
                group_clause
            )

        if sort_column and plan.sort_direction:
            direction = (
                self._validate_sort_direction(
                    plan.sort_direction
                )
            )

            sql_parts.append(
                f"ORDER BY {sort_column} "
                f"{direction}"
            )

        limit_clause = self._build_limit(
            plan.limit
        )

        if limit_clause:
            sql_parts.append(
                limit_clause
            )

        return "\n".join(
            sql_parts
        ) + ";"

    # ------------------------------------------------------------------
    # COUNT RANKING
    # ------------------------------------------------------------------

    def _generate_count_ranking_query(
        self,
        plan: QueryPlan,
    ) -> str:
        """
        Generate frequency-based ranking SQL.

        Example:

            Which payment method was used most often?

        becomes:

            SELECT
                payment_method,
                COUNT(*) AS count
            FROM ecommerce_orders
            GROUP BY payment_method
            ORDER BY count DESC
            LIMIT 1;
        """

        if not plan.dimension:
            raise SQLGenerationError(
                "Count ranking requires a dimension."
            )

        dimension = self._validate_identifier(
            plan.dimension
        )

        table = self._validate_identifier(
            self.table_name
        )

        where_clause = (
            self._build_where_clause(
                plan.filters
            )
        )

        sql_parts = [
            "SELECT",
            f"    {dimension},",
            "    COUNT(*) AS count",
            f"FROM {table}",
        ]

        if where_clause:
            sql_parts.append(
                where_clause
            )

        sql_parts.extend(
            [
                f"GROUP BY {dimension}",
                "ORDER BY count DESC",
            ]
        )

        limit_clause = self._build_limit(
            plan.limit
        )

        if limit_clause:
            sql_parts.append(
                limit_clause
            )

        return "\n".join(
            sql_parts
        ) + ";"

    # ------------------------------------------------------------------
    # MULTI-METRIC QUERY
    # ------------------------------------------------------------------

    def _generate_multi_metric_query(
        self,
        plan: QueryPlan,
    ) -> str:
        """
        Generate grouped SQL containing multiple metrics.

        Example:

            What were total sales and total profit by region?

        becomes:

            SELECT
                region,
                SUM(sales) AS total_sales,
                SUM(profit) AS total_profit
            FROM ecommerce_orders
            GROUP BY region;
        """

        if not plan.dimension:
            raise SQLGenerationError(
                "Multi-metric aggregation requires a dimension."
            )

        if not plan.metrics:
            raise SQLGenerationError(
                "Multi-metric aggregation requires metrics."
            )

        dimension = self._validate_identifier(
            plan.dimension
        )

        table = self._validate_identifier(
            self.table_name
        )

        expressions = []

        for metric_spec in plan.metrics:
            if metric_spec.derived:
                raise SQLGenerationError(
                    "Derived metrics require the derived_ratio strategy."
                )

            aggregation = metric_spec.aggregation
            metric = metric_spec.metric

            if not aggregation:
                raise SQLGenerationError(
                    "Metric specification is missing aggregation."
                )

            expression = self._build_aggregation(
                aggregation,
                metric,
            )

            alias = (
                metric_spec.alias
                or f"{aggregation}_{metric}"
            )

            alias = self._validate_identifier(
                alias
            )

            expressions.append(
                f"{expression} AS {alias}"
            )

        select_lines = [
            f"    {dimension},"
        ]

        for index, expression in enumerate(
            expressions
        ):
            comma = (
                ","
                if index < len(expressions) - 1
                else ""
            )

            select_lines.append(
                f"    {expression}{comma}"
            )

        where_clause = (
            self._build_where_clause(
                plan.filters
            )
        )

        sql_parts = [
            "SELECT",
            *select_lines,
            f"FROM {table}",
        ]

        if where_clause:
            sql_parts.append(
                where_clause
            )

        sql_parts.append(
            f"GROUP BY {dimension}"
        )

                # Ranking questions should sort by the planner's
        # primary metric rather than whichever metric happens
        # to appear first in the metrics list.
        if (
            plan.sort_column
            and plan.sort_direction
        ):
            primary_alias = None

            for metric_spec in plan.metrics:
                if (
                    metric_spec.metric
                    == plan.metric
                ):
                    primary_alias = (
                        metric_spec.alias
                        or f"{metric_spec.aggregation}_{metric_spec.metric}"
                    )
                    break

            if primary_alias:
                primary_alias = (
                    self._validate_identifier(
                        primary_alias
                    )
                )

                direction = (
                    self._validate_sort_direction(
                        plan.sort_direction
                    )
                )

                sql_parts.append(
                    f"ORDER BY {primary_alias} "
                    f"{direction}"
                )

        limit_clause = self._build_limit(
            plan.limit
        )

        if limit_clause:
            sql_parts.append(
                limit_clause
            )

        return "\n".join(
            sql_parts
        ) + ";"

    # ------------------------------------------------------------------
    # DERIVED RATIO
    # ------------------------------------------------------------------

    def _generate_derived_ratio_query(
        self,
        plan: QueryPlan,
    ) -> str:
        """
        Generate a ratio such as profit margin.

        Example:

            SUM(profit) / SUM(sales) * 100
        """

        if not plan.dimension:
            raise SQLGenerationError(
                "Derived ratio requires a dimension."
            )

        if not plan.metrics:
            raise SQLGenerationError(
                "Derived ratio requires a metric specification."
            )

        metric_spec = plan.metrics[0]

        if not metric_spec.derived:
            raise SQLGenerationError(
                "Derived ratio specification is missing."
            )

        derived = metric_spec.derived

        if derived.get("type") != "ratio":
            raise SQLGenerationError(
                "Unsupported derived metric type."
            )

        numerator = derived.get(
            "numerator"
        )

        denominator = derived.get(
            "denominator"
        )

        multiplier = derived.get(
            "multiplier",
            1.0,
        )

        if not numerator or not denominator:
            raise SQLGenerationError(
                "Ratio requires numerator and denominator."
            )

        numerator_expression = (
            self._build_aggregation(
                numerator["aggregation"],
                numerator["metric"],
            )
        )

        denominator_expression = (
            self._build_aggregation(
                denominator["aggregation"],
                denominator["metric"],
            )
        )

        dimension = self._validate_identifier(
            plan.dimension
        )

        alias = self._validate_identifier(
            metric_spec.alias
            or "ratio"
        )

        table = self._validate_identifier(
            self.table_name
        )

        where_clause = (
            self._build_where_clause(
                plan.filters
            )
        )

        ratio_expression = (
            f"100.0 * "
            f"{numerator_expression} / "
            f"NULLIF({denominator_expression}, 0)"
        )

        if multiplier != 100.0:
            ratio_expression = (
                f"{multiplier} * "
                f"{numerator_expression} / "
                f"NULLIF({denominator_expression}, 0)"
            )

        sql_parts = [
            "SELECT",
            f"    {dimension},",
            f"    {ratio_expression} AS {alias}",
            f"FROM {table}",
        ]

        if where_clause:
            sql_parts.append(
                where_clause
            )

        sql_parts.extend(
            [
                f"GROUP BY {dimension}",
                f"ORDER BY {alias} DESC",
            ]
        )

        limit_clause = self._build_limit(
            plan.limit
        )

        if limit_clause:
            sql_parts.append(
                limit_clause
            )

        return "\n".join(
            sql_parts
        ) + ";"

    # ------------------------------------------------------------------
    # YEAR-OVER-YEAR COMPARISON
    # ------------------------------------------------------------------

    def _generate_year_over_year_query(
        self,
        plan: QueryPlan,
    ) -> str:
        """
        Generate SQL for a two-year metric comparison.

        Example:

            What was the change in total sales from 2024 to 2025?

        Returns one row containing both yearly values and their difference.
        """

        if not plan.comparison:
            raise SQLGenerationError(
                "Year-over-year comparison requires comparison metadata."
            )

        comparison = plan.comparison

        if not hasattr(
            comparison,
            "periods",
        ):
            raise SQLGenerationError(
                "Invalid comparison specification."
            )

        periods = comparison.periods

        if len(periods) != 2:
            raise SQLGenerationError(
                "Year-over-year comparison requires exactly two years."
            )

        metric = comparison.metric
        aggregation = comparison.aggregation

        if not metric or not aggregation:
            raise SQLGenerationError(
                "Year-over-year comparison requires metric and aggregation."
            )

        aggregate_expression = (
            self._build_aggregation(
                aggregation,
                metric,
            )
        )

        table = self._validate_identifier(
            self.table_name
        )

        first_year = int(periods[0])
        second_year = int(periods[1])

        alias_base = (
            f"{aggregation}_{metric}"
        )

        first_alias = (
            f"{alias_base}_{first_year}"
        )

        second_alias = (
            f"{alias_base}_{second_year}"
        )

        sql = f"""
WITH yearly_values AS (
    SELECT
        EXTRACT(YEAR FROM order_date) AS year,
        {aggregate_expression} AS value
    FROM {table}
    WHERE EXTRACT(YEAR FROM order_date)
        IN ({first_year}, {second_year})
    GROUP BY year
)
SELECT
    MAX(
        CASE
            WHEN year = {first_year}
            THEN value
        END
    ) AS {first_alias},
    MAX(
        CASE
            WHEN year = {second_year}
            THEN value
        END
    ) AS {second_alias},
    MAX(
        CASE
            WHEN year = {second_year}
            THEN value
        END
    )
    -
    MAX(
        CASE
            WHEN year = {first_year}
            THEN value
        END
    ) AS change
FROM yearly_values;
""".strip()

        return sql

    # ------------------------------------------------------------------
    # TOP GROUPS THEN COMPARE
    # ------------------------------------------------------------------

    def _generate_top_groups_then_compare_query(
        self,
        plan: QueryPlan,
    ) -> str:
        """
        Generate a two-stage comparison.

        Example:

            What is the difference in total sales between the two
            customer segments with the highest average order value?

        Stage 1:
            Find top two segments by AVG(sales).

        Stage 2:
            Calculate SUM(sales) for those two segments and subtract.
        """

        if not plan.dimension:
            raise SQLGenerationError(
                "Top-group comparison requires a dimension."
            )

        if plan.limit != 2:
            raise SQLGenerationError(
                "Top-group comparison currently requires LIMIT 2."
            )

        dimension = self._validate_identifier(
            plan.dimension
        )

        table = self._validate_identifier(
            self.table_name
        )

        filters = self._build_where_clause(
            plan.filters
        )

        sql = f"""
WITH top_groups AS (
    SELECT
        {dimension}
    FROM {table}
    {filters}
    GROUP BY {dimension}
    ORDER BY AVG(sales) DESC
    LIMIT 2
),
group_totals AS (
    SELECT
        {dimension},
        SUM(sales) AS total_sales
    FROM {table}
    WHERE {dimension} IN (
        SELECT {dimension}
        FROM top_groups
    )
    {"AND " + filters[6:] if filters.startswith("WHERE ") else ""}
    GROUP BY {dimension}
)
SELECT
    MAX(total_sales) -
    MIN(total_sales) AS difference
FROM group_totals;
""".strip()

        return sql

    # ------------------------------------------------------------------
    # RANK WITHIN TIME PERIOD
    # ------------------------------------------------------------------

    def _generate_rank_within_time_period_query(
        self,
        plan: QueryPlan,
    ) -> str:
        """
        Rank a dimension independently within each year.

        Example:

            Which region generated the most sales for each year?

        Returns the highest-sales region for each year.
        """

        if not plan.dimension:
            raise SQLGenerationError(
                "Time-period ranking requires a dimension."
            )

        if plan.time_granularity != "year":
            raise SQLGenerationError(
                "Time-period ranking currently supports year granularity only."
            )

        dimension = self._validate_identifier(
            plan.dimension
        )

        table = self._validate_identifier(
            self.table_name
        )

        where_clause = (
            self._build_where_clause(
                plan.filters
            )
        )

        sql = f"""
WITH yearly_region_sales AS (
    SELECT
        EXTRACT(YEAR FROM order_date) AS year,
        {dimension},
        SUM(sales) AS total_sales
    FROM {table}
    {where_clause}
    GROUP BY
        year,
        {dimension}
),
ranked AS (
    SELECT
        year,
        {dimension},
        total_sales,
        ROW_NUMBER() OVER (
            PARTITION BY year
            ORDER BY total_sales DESC
        ) AS rank
    FROM yearly_region_sales
)
SELECT
    year,
    {dimension},
    total_sales
FROM ranked
WHERE rank = 1
ORDER BY year;
""".strip()

        return sql

    # ------------------------------------------------------------------
    # PERCENTAGE QUERY
    # ------------------------------------------------------------------

    def _generate_percentage_query(
        self,
        plan: QueryPlan,
    ) -> str:
        """
        Generate SQL for percentage/share questions.
        """

        if plan.percentage is None:
            raise SQLGenerationError(
                "Percentage plan is missing."
            )

        percentage = plan.percentage

        numerator_conditions = []

        for filter_condition in (
            percentage.numerator_filters
        ):
            numerator_conditions.append(
                self._build_filter_condition_object(
                    filter_condition
                )
            )

        denominator_conditions = []

        for filter_condition in (
            percentage.denominator_filters
        ):
            denominator_conditions.append(
                self._build_filter_condition_object(
                    filter_condition
                )
            )

        numerator_where = ""

        if numerator_conditions:
            numerator_where = (
                "WHERE "
                + " AND ".join(
                    numerator_conditions
                )
            )

        denominator_where = ""

        if denominator_conditions:
            denominator_where = (
                "WHERE "
                + " AND ".join(
                    denominator_conditions
                )
            )

        table = self._validate_identifier(
            self.table_name
        )

        sql = f"""
SELECT
    ROUND(
        100.0 *
        (
            SELECT COUNT(*)
            FROM {table}
            {numerator_where}
        )
        /
        NULLIF(
            (
                SELECT COUNT(*)
                FROM {table}
                {denominator_where}
            ),
            0
        ),
        2
    ) AS percentage;
""".strip()

        return sql

    # ------------------------------------------------------------------
    # FILTER CONDITION OBJECT
    # ------------------------------------------------------------------

    def _build_filter_condition_object(
        self,
        filter_condition: Any,
    ) -> str:
        """
        Convert a FilterCondition dataclass
        into a normal filter dictionary.
        """

        return self._build_filter(
            {
                "column": filter_condition.column,
                "operator": filter_condition.operator,
                "value": filter_condition.value,
            }
        )

    # ------------------------------------------------------------------
    # PUBLIC GENERATION METHOD
    # ------------------------------------------------------------------

    def generate(
        self,
        plan: QueryPlan,
    ) -> str:
        """
        Generate SQL from a QueryPlan.
        """

        if plan.intent == "percentage":
            return self._generate_percentage_query(
                plan
            )

        if plan.strategy == "count_ranking":
            return self._generate_count_ranking_query(
                plan
            )

        if plan.strategy == "multi_metric_aggregation":
            return self._generate_multi_metric_query(
                plan
            )

        if plan.strategy == "derived_ratio":
            return self._generate_derived_ratio_query(
                plan
            )

        if plan.strategy == "year_over_year_comparison":
            return self._generate_year_over_year_query(
                plan
            )

        if plan.strategy == "top_groups_then_compare":
            return self._generate_top_groups_then_compare_query(
                plan
            )

        if plan.strategy == "rank_within_time_period":
            return self._generate_rank_within_time_period_query(
                plan
            )

        return self._generate_standard_query(
            plan
        )


# ----------------------------------------------------------------------
# LOCAL TEST
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from src.agent.planner import QueryPlanner
    from src.data.profiler import profile_dataset

    project_dir = (
        Path(__file__).resolve().parents[2]
    )

    dataset_path = (
        project_dir
        / "data"
        / "raw"
        / "ecommerce_orders.csv"
    )

    dataset_profile = profile_dataset(
        dataset_path
    )

    planner = QueryPlanner(
        dataset_profile
    )

    generator = SQLGenerator(
        table_name="ecommerce_orders"
    )

    test_questions = [
        "Which region generated the highest total sales?",
        "Which product category generated the highest total profit?",
        "What was the total sales revenue in 2024?",
        "Which customer segment has the highest average sales per order?",
        "What percentage of all orders were cancelled?",
        "What was the change in total sales from 2024 to 2025?",
        "What was the total profit for completed orders?",
        "Which payment method was used most often?",
        "What were total sales and total profit by region?",
        "Which region had the highest profit margin?",
        "What is the difference in total sales between the two customer segments with the highest average order value?",
        "Which category had the highest average discount and what was its average profit?",
        "Which region generated the most sales for each year?",
        "Which category generated the highest profit among orders with a discount of at least 20%?",
    ]

    print("\n=== SQL GENERATOR TEST ===\n")

    for question in test_questions:
        print(f"Question: {question}")

        try:
            plan = planner.create_plan(
                question
            )

            sql = generator.generate(
                plan
            )

            print("\nGenerated SQL:")
            print(sql)

        except Exception as exc:
            print(
                f"\nERROR: {exc}"
            )

        print("-" * 80)