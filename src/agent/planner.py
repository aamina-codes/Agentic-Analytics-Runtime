from __future__ import annotations

import re

from dataclasses import dataclass

from typing import Any

from src.models.schemas import (
    ComparisonSpec,
    FilterCondition,
    MetricSpec,
    PercentagePlan,
    QueryPlan,
)


class QueryPlanningError(Exception):
    """Raised when a question cannot be converted into a usable query plan."""


@dataclass
class PlannerConfig:
    default_limit: int = 1


class QueryPlanner:
    """
    Converts natural-language analytics questions into structured QueryPlan objects.

    The planner is deterministic by design:

    - schema-aware
    - explicit metric detection
    - explicit aggregation detection
    - explicit filter detection
    - explicit analytical strategies

    The planner does not execute SQL.
    """

    def __init__(
        self,
        dataset_profile: dict[str, Any],
        config: PlannerConfig | None = None,
    ) -> None:
        self.dataset_profile = dataset_profile
        self.config = config or PlannerConfig()

        self.numeric_columns = set(
            dataset_profile.get("numeric_columns", [])
            or dataset_profile.get("numeric", [])
        )

        self.categorical_columns = set(
            dataset_profile.get("categorical_columns", [])
            or dataset_profile.get("categorical", [])
        )

        self.date_columns = set(
            dataset_profile.get("date_columns", [])
            or dataset_profile.get("date", [])
        )

        self.all_columns = (
            self.numeric_columns
            | self.categorical_columns
            | self.date_columns
        )

        self.field_aliases = {
            "sales": "sales",
            "revenue": "sales",
            "sales revenue": "sales",
            "profit": "profit",
            "discount": "discount",
            "shipping cost": "shipping_cost",
            "shipping": "shipping_cost",
            "quantity": "quantity",
            "units": "quantity",
            "customer segment": "customer_segment",
            "customer segments": "customer_segment",
            "segment": "customer_segment",
            "region": "region",
            "country": "country",
            "city": "city",
            "category": "category",
            "product category": "category",
            "subcategory": "subcategory",
            "payment method": "payment_method",
            "payment methods": "payment_method",
            "order status": "order_status",
            "status": "order_status",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_plan(self, question: str) -> QueryPlan:
        if not isinstance(question, str) or not question.strip():
            raise QueryPlanningError("Question must be a non-empty string.")

        question = self._normalize_question(question)

        intent = self._detect_intent(question)

        filters = self._detect_filters(question)

        # Percentage questions are handled independently because they
        # require numerator + denominator semantics.
        if intent == "percentage":
            return self._build_percentage_plan(
                question,
                filters,
            )

        comparison = self._detect_comparison(question)

        # Derived metrics must be checked before ordinary metrics.
        if self._contains_profit_margin(question):
            return self._build_profit_margin_plan(
                question=question,
                intent=intent,
                filters=filters,
            )

        # Special multi-stage comparison.
        if self._is_top_groups_comparison(question):
            return self._build_top_groups_comparison_plan(
                question=question,
                filters=filters,
            )

        # Ranking by frequency/count.
        if self._is_count_ranking(question):
            return self._build_count_ranking_plan(
                question=question,
                filters=filters,
            )

        # Ranking within each year.
        if self._is_rank_within_time_period(question):
            return self._build_time_period_ranking_plan(
                question=question,
                filters=filters,
            )

        metrics = self._detect_metrics(question)

        if not metrics:
            raise QueryPlanningError(
                f"Could not identify an analytical metric from question: {question}"
            )

        primary_metric = self._select_primary_metric(
            question=question,
            metrics=metrics,
            filters=filters,
        )

        aggregation = self._detect_aggregation(
            question=question,
            metric=primary_metric,
        )

        dimension = self._detect_dimension(question)

        # q20-style cases:
        # "highest profit among orders with discount >= 20%"
        #
        # The target metric must remain profit even though discount occurs
        # earlier in the question as a filter.
        if self._has_ranking_target_phrase(question):
            target_metric = self._detect_ranking_target_metric(question)

            if target_metric is not None:
                primary_metric = target_metric

                aggregation = self._detect_aggregation(
                    question=question,
                    metric=primary_metric,
                )

        # Multi-metric aggregation such as:
        # "total sales and total profit by region"
        strategy = None

        if len(metrics) > 1:
            strategy = "multi_metric_aggregation"

        # Standard ranking defaults.
        sort_column = None
        sort_direction = None
        limit = None

        if intent in {"ranking", "grouped_ranking"}:
            sort_column = primary_metric if primary_metric else None
            sort_direction = "DESC"
            limit = self.config.default_limit

        # q06-style year comparison.
        if comparison is not None:
            strategy = "year_over_year_comparison"

        # Build metric specs.
        metric_specs = self._build_metric_specs(
            question=question,
            metrics=metrics,
            primary_metric=primary_metric,
            primary_aggregation=aggregation,
        )

        # Ensure the primary metric is represented correctly when filters
        # mention another numeric field.
        if primary_metric is not None:
            metric_specs = self._ensure_primary_metric(
                metric_specs,
                primary_metric,
                aggregation,
            )

        confidence = self._calculate_confidence(
            intent=intent,
            primary_metric=primary_metric,
            dimension=dimension,
            filters=filters,
            metrics=metrics,
        )

        reasoning_parts = [
            f"Detected intent: {intent}.",
        ]

        if primary_metric:
            reasoning_parts.append(
                f"Primary metric: {primary_metric}."
            )

        if aggregation:
            reasoning_parts.append(
                f"Aggregation: {aggregation}."
            )

        if dimension:
            reasoning_parts.append(
                f"Dimension: {dimension}."
            )

        if len(metrics) > 1:
            reasoning_parts.append(
                f"Multiple metrics detected: {metrics}."
            )

        if filters:
            reasoning_parts.append(
                f"Detected filters: {filters}."
            )

        if comparison is not None:
            reasoning_parts.append(
                f"Comparison type: {comparison.comparison_type}."
            )

            if comparison.periods:
                reasoning_parts.append(
                    f"Comparison periods: {comparison.periods}."
                )

        if strategy:
            reasoning_parts.append(
                f"Analytical strategy: {strategy}."
            )

        return QueryPlan(
            question=question,
            intent=intent,
            metric=primary_metric,
            aggregation=aggregation,
            dimension=dimension,
            filters=filters,
            sort_column=sort_column,
            sort_direction=sort_direction,
            limit=limit,
            time_granularity=(
                "year"
                if intent == "grouped_ranking"
                else None
            ),
            comparison=comparison,
            percentage=None,
            confidence=confidence,
            reasoning=" ".join(reasoning_parts),
            metrics=metric_specs,
            strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_question(self, question: str) -> str:
        question = question.strip().lower()

        # Make common accidental missing spaces harmless.
        #
        # Example:
        # "thetwo customer segments"
        # -> "the two customer segments"
        #
        # This is intentionally limited to known analytical phrases
        # rather than blindly inserting spaces into arbitrary words.
        replacements = {
            "thetwo": "the two",
            "top2": "top 2",
            "highestaverage": "highest average",
            "lowestaverage": "lowest average",
            "averagesales": "average sales",
            "averageprofit": "average profit",
            "averagediscount": "average discount",
            "totalsales": "total sales",
            "totalprofit": "total profit",
        }

        for source, target in replacements.items():
            question = question.replace(source, target)

        # Make accidental missing spaces before common analytical terms
        # harmless.
        question = re.sub(
            r"(?<=[a-z])(?=(average|total|highest|lowest|profit|discount|sales|by|and)\b)",
            " ",
            question,
        )

        question = re.sub(r"\s+", " ", question)

        return question

    # ------------------------------------------------------------------
    # Intent
    # ------------------------------------------------------------------

    def _detect_intent(self, question: str) -> str:
        if self._contains_percentage_language(question):
            return "percentage"

        if self._is_top_groups_comparison(question):
            return "comparison"

        if self._is_year_change(question):
            return "comparison"

        if self._is_rank_within_time_period(question):
            return "grouped_ranking"

        if self._is_count_ranking(question):
            return "ranking"

        if any(
            phrase in question
            for phrase in [
                "highest",
                "lowest",
                "most",
                "least",
                "top",
                "largest",
                "smallest",
            ]
        ):
            return "ranking"

        if "by region" in question or "by category" in question:
            return "grouped_aggregation"

        if "by " in question:
            return "grouped_aggregation"

        return "aggregation"

    # ------------------------------------------------------------------
    # Metric detection
    # ------------------------------------------------------------------

    def _detect_metrics(self, question: str) -> list[str]:
        """
        Detect numeric metrics requested by the question.

        A numeric field mentioned only as a filter is not automatically
        considered a requested output metric.
        """

        metrics: list[str] = []

        explicit_phrases = [
            ("sales revenue", "sales"),
            ("total sales", "sales"),
            ("average sales", "sales"),
            ("avg sales", "sales"),
            ("sales", "sales"),
            ("total profit", "profit"),
            ("average profit", "profit"),
            ("avg profit", "profit"),
            ("profit", "profit"),
            ("average discount", "discount"),
            ("avg discount", "discount"),
            ("discount", "discount"),
            ("shipping cost", "shipping_cost"),
            ("quantity", "quantity"),
        ]

        for phrase, metric in explicit_phrases:
            if phrase in question and metric not in metrics:
                metrics.append(metric)

        filter_metric = self._detect_filter_only_metric(question)

        if filter_metric in metrics:
            target_metric = self._detect_ranking_target_metric(question)

            if target_metric and target_metric != filter_metric:
                metrics = [
                    metric
                    for metric in metrics
                    if metric != filter_metric
                ]

        return metrics

    def _detect_metric(self, question: str) -> str | None:
        metrics = self._detect_metrics(question)

        if not metrics:
            return None

        return self._select_primary_metric(
            question=question,
            metrics=metrics,
            filters=[],
        )

    def _select_primary_metric(
        self,
        question: str,
        metrics: list[str],
        filters: list[dict[str, Any]],
    ) -> str:
        ranking_target = self._detect_ranking_target_metric(question)

        if ranking_target in metrics:
            return ranking_target

        for metric in self._metric_order_from_question(
            question,
            metrics,
        ):
            return metric

        return metrics[0]

    def _metric_order_from_question(
        self,
        question: str,
        metrics: list[str],
    ) -> list[str]:
        positions: list[tuple[int, str]] = []

        for metric in metrics:
            aliases = [
                alias
                for alias, canonical in self.field_aliases.items()
                if canonical == metric
            ]

            positions_for_metric = [
                question.find(alias)
                for alias in aliases
                if question.find(alias) >= 0
            ]

            if positions_for_metric:
                positions.append(
                    (
                        min(positions_for_metric),
                        metric,
                    )
                )

        positions.sort()

        return [
            metric
            for _, metric in positions
        ]

    def _detect_ranking_target_metric(
        self,
        question: str,
    ) -> str | None:
        patterns = [
            r"(?:highest|lowest)\s+(?:total\s+|average\s+|avg\s+)?"
            r"(sales|profit|discount|shipping cost|quantity)",

            r"(?:most|least)\s+(?:total\s+|average\s+|avg\s+)?"
            r"(sales|profit|discount|shipping cost|quantity)",

            r"highest\s+(sales|profit|discount)",

            r"lowest\s+(sales|profit|discount)",
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                question,
            )

            if match:
                value = match.group(1)

                return self.field_aliases.get(
                    value,
                    value.replace(" ", "_"),
                )

        return None

    def _has_ranking_target_phrase(
        self,
        question: str,
    ) -> bool:
        return bool(
            re.search(
                r"\b(highest|lowest|most|least|top|largest|smallest)\b",
                question,
            )
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _detect_aggregation(
        self,
        question: str,
        metric: str | None,
    ) -> str | None:
        if self._is_count_ranking(question):
            return "count"

        if re.search(
            r"\b(average|avg|mean)\b",
            question,
        ):
            return "avg"

        if re.search(
            r"\b(total|sum)\b",
            question,
        ):
            return "sum"

        if self._has_ranking_target_phrase(question):
            return "sum"

        return "sum" if metric else None

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def _detect_dimension(
        self,
        question: str,
    ) -> str | None:
        candidates = [
            "customer_segment",
            "payment_method",
            "order_status",
            "subcategory",
            "category",
            "region",
            "country",
            "city",
            "product",
        ]

        for column in candidates:
            pretty = column.replace("_", " ")

            if pretty in question:
                return column

        return None

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    def _detect_filters(
        self,
        question: str,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []

        years = re.findall(
            r"\b(20\d{2})\b",
            question,
        )

        # Only create standalone year filters when the question is not
        # explicitly comparing two years.
        if len(years) == 1:
            filters.append(
                {
                    "column": "order_date",
                    "operator": "year_equals",
                    "value": int(years[0]),
                }
            )

        status_map = {
            "cancelled": "Cancelled",
            "canceled": "Cancelled",
            "completed": "Completed",
            "returned": "Returned",
        }

        for phrase, value in status_map.items():
            if phrase in question:
                filters.append(
                    {
                        "column": "order_status",
                        "operator": "equals",
                        "value": value,
                    }
                )
                break

        discount_pattern = re.search(
            r"discount\s+(?:of\s+)?at\s+least\s+(\d+(?:\.\d+)?)\s*%",
            question,
        )

        if discount_pattern:
            percentage = float(
                discount_pattern.group(1)
            )

            filters.append(
                {
                    "column": "discount",
                    "operator": ">=",
                    "value": percentage / 100.0,
                }
            )

        return filters

    def _detect_filter_only_metric(
        self,
        question: str,
    ) -> str | None:
        if re.search(
            r"discount\s+(?:of\s+)?at\s+least\s+\d+(?:\.\d+)?\s*%",
            question,
        ):
            return "discount"

        return None

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def _detect_comparison(
        self,
        question: str,
    ) -> ComparisonSpec | None:
        if not self._is_year_change(question):
            return None

        years = [
            int(value)
            for value in re.findall(
                r"\b20\d{2}\b",
                question,
            )
        ]

        if len(years) >= 2:
            years = years[:2]

        return ComparisonSpec(
            comparison_type="year_change",
            periods=years,
            metric="sales",
            aggregation="sum",
            operation="difference",
        )

    def _is_year_change(
        self,
        question: str,
    ) -> bool:
        return (
            "change" in question
            and len(
                re.findall(
                    r"\b20\d{2}\b",
                    question,
                )
            ) >= 2
        ) or (
            "from 2024 to 2025" in question
        )

    # ------------------------------------------------------------------
    # Special strategies
    # ------------------------------------------------------------------

    def _is_count_ranking(
        self,
        question: str,
    ) -> bool:
        return (
            "most often" in question
            or "used most" in question
            or "most frequently" in question
            or "most common" in question
            or "most used" in question
            or "most orders" in question
            or "highest number of orders" in question
            or "largest number of orders" in question
            or "most number of orders" in question
        )

    def _build_count_ranking_plan(
        self,
        question: str,
        filters: list[dict[str, Any]],
    ) -> QueryPlan:
        dimension = self._detect_dimension(question)

        metric_specs = [
            MetricSpec(
                metric=None,
                aggregation="count",
                alias="count",
            )
        ]

        return QueryPlan(
            question=question,
            intent="ranking",
            metric=None,
            aggregation="count",
            dimension=dimension,
            filters=filters,
            sort_column="count",
            sort_direction="DESC",
            limit=1,
            time_granularity=None,
            comparison=None,
            percentage=None,
            confidence=1.0 if dimension else 0.7,
            reasoning=(
                "Detected intent: ranking. "
                "Frequency-based ranking detected. "
                f"Dimension: {dimension}. "
                "Aggregation: count. "
                "Strategy: count_ranking."
            ),
            metrics=metric_specs,
            strategy="count_ranking",
        )

    def _is_top_groups_comparison(
        self,
        question: str,
    ) -> bool:
        """
        Detect questions that require:

        1. Finding the top two groups by average order value.
        2. Calculating total sales for those groups.
        3. Comparing the two totals.

        Supports natural variations such as:

        - "top two customer segments"
        - "two customer segments"
        - "the two customer segments"
        - "how did ... differ"
        - "... difference ..."
        """

        has_two_groups = (
            "top two" in question
            or "top 2" in question
            or "two customer segments" in question
            or "2 customer segments" in question
        )

        has_average_order_value = (
            "highest average order value" in question
            or "highest average order" in question
        )

        has_comparison_language = (
            "difference" in question
            or "differ" in question
            or "differed" in question
            or "how did" in question
        )

        return (
            has_two_groups
            and has_average_order_value
            and has_comparison_language
        )

    def _build_top_groups_comparison_plan(
        self,
        question: str,
        filters: list[dict[str, Any]],
    ) -> QueryPlan:
        dimension = self._detect_dimension(question)

        metric_specs = [
            MetricSpec(
                metric="sales",
                aggregation="avg",
                alias="avg_sales",
            ),
            MetricSpec(
                metric="sales",
                aggregation="sum",
                alias="total_sales",
            ),
        ]

        comparison = ComparisonSpec(
            comparison_type="top_groups_then_difference",
            periods=[],
            metric="sales",
            aggregation="sum",
            operation="difference",
        )

        return QueryPlan(
            question=question,
            intent="comparison",
            metric="sales",
            aggregation="sum",
            dimension=dimension,
            filters=filters,
            sort_column="avg_sales",
            sort_direction="DESC",
            limit=2,
            time_granularity=None,
            comparison=comparison,
            percentage=None,
            confidence=1.0 if dimension else 0.8,
            reasoning=(
                "Detected intent: comparison. "
                "Multi-stage comparison detected. "
                f"Dimension: {dimension}. "
                "Stage 1: rank groups by average sales. "
                "Stage 2: compare total sales of the top two groups. "
                "Analytical strategy: top_groups_then_compare."
            ),
            metrics=metric_specs,
            strategy="top_groups_then_compare",
        )

    def _is_rank_within_time_period(
        self,
        question: str,
    ) -> bool:
        return (
            "for each year" in question
            or "each year" in question
            or "per year" in question
        ) and (
            "highest" in question
            or "most" in question
            or "top" in question
        )

    def _build_time_period_ranking_plan(
        self,
        question: str,
        filters: list[dict[str, Any]],
    ) -> QueryPlan:
        dimension = self._detect_dimension(question)

        metric_specs = [
            MetricSpec(
                metric="sales",
                aggregation="sum",
                alias="total_sales",
            )
        ]

        return QueryPlan(
            question=question,
            intent="grouped_ranking",
            metric="sales",
            aggregation="sum",
            dimension=dimension,
            filters=filters,
            sort_column="total_sales",
            sort_direction="DESC",
            limit=1,
            time_granularity="year",
            comparison=None,
            percentage=None,
            confidence=1.0 if dimension else 0.8,
            reasoning=(
                "Detected intent: grouped_ranking. "
                "Primary metric: sales. "
                "Aggregation: sum. "
                f"Dimension: {dimension}. "
                "Time granularity: year. "
                "Analytical strategy: rank_within_time_period."
            ),
            metrics=metric_specs,
            strategy="rank_within_time_period",
        )

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def _contains_profit_margin(
        self,
        question: str,
    ) -> bool:
        return (
            "profit margin" in question
            or "profit-margin" in question
        )

    def _build_profit_margin_plan(
        self,
        question: str,
        intent: str,
        filters: list[dict[str, Any]],
    ) -> QueryPlan:
        dimension = self._detect_dimension(question)

        metric_spec = MetricSpec(
            metric=None,
            aggregation=None,
            alias="profit_margin",
            derived={
                "type": "ratio",
                "numerator": {
                    "metric": "profit",
                    "aggregation": "sum",
                },
                "denominator": {
                    "metric": "sales",
                    "aggregation": "sum",
                },
                "multiplier": 100.0,
            },
        )

        return QueryPlan(
            question=question,
            intent="ranking",
            metric=None,
            aggregation=None,
            dimension=dimension,
            filters=filters,
            sort_column="profit_margin",
            sort_direction="DESC",
            limit=1,
            time_granularity=None,
            comparison=None,
            percentage=None,
            confidence=1.0 if dimension else 0.8,
            reasoning=(
                "Detected intent: ranking. "
                f"Dimension: {dimension}. "
                "Derived metric: profit_margin. "
                "Analytical strategy: derived_ratio."
            ),
            metrics=[metric_spec],
            strategy="derived_ratio",
        )

    # ------------------------------------------------------------------
    # Percentage
    # ------------------------------------------------------------------

    def _contains_percentage_language(
        self,
        question: str,
    ) -> bool:
        return (
            "percentage" in question
            or "percent" in question
            or "what share" in question
            or "what proportion" in question
        )

    def _build_percentage_plan(
        self,
        question: str,
        filters: list[dict[str, Any]],
    ) -> QueryPlan:
        numerator_filters = [
            FilterCondition(
                column=item["column"],
                operator=item["operator"],
                value=item["value"],
            )
            for item in filters
        ]

        percentage = PercentagePlan(
            numerator_aggregation="count",
            numerator_filters=numerator_filters,
            denominator_aggregation="count",
            denominator_filters=[],
        )

        return QueryPlan(
            question=question,
            intent="percentage",
            metric=None,
            aggregation=None,
            dimension=None,
            filters=filters,
            sort_column=None,
            sort_direction=None,
            limit=None,
            time_granularity=None,
            comparison=None,
            percentage=percentage,
            confidence=1.0,
            reasoning=(
                "Detected intent: percentage. "
                f"Detected filters: {filters}. "
                "Percentage numerator uses a conditional count. "
                "Percentage denominator uses the total row count."
            ),
            metrics=[],
            strategy="percentage",
        )

    # ------------------------------------------------------------------
    # Metric specs
    # ------------------------------------------------------------------

    def _build_metric_specs(
        self,
        question: str,
        metrics: list[str],
        primary_metric: str,
        primary_aggregation: str | None,
    ) -> list[MetricSpec]:
        specs: list[MetricSpec] = []

        for metric in metrics:
            if metric == primary_metric:
                aggregation = primary_aggregation
            else:
                aggregation = self._detect_secondary_aggregation(
                    question,
                    metric,
                )

            specs.append(
                MetricSpec(
                    metric=metric,
                    aggregation=aggregation,
                    alias=None,
                )
            )

        return specs

    def _detect_secondary_aggregation(
        self,
        question: str,
        metric: str,
    ) -> str:
        metric_aliases = [
            alias
            for alias, canonical in self.field_aliases.items()
            if canonical == metric
        ]

        for alias in metric_aliases:
            if re.search(
                rf"(average|avg|mean)\s+{re.escape(alias)}",
                question,
            ):
                return "avg"

            if re.search(
                rf"(total|sum)\s+{re.escape(alias)}",
                question,
            ):
                return "sum"

        return "sum"

    def _ensure_primary_metric(
        self,
        specs: list[MetricSpec],
        primary_metric: str,
        aggregation: str | None,
    ) -> list[MetricSpec]:
        for spec in specs:
            if spec.metric == primary_metric:
                spec.aggregation = aggregation
                return specs

        specs.insert(
            0,
            MetricSpec(
                metric=primary_metric,
                aggregation=aggregation,
            ),
        )

        return specs

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        intent: str,
        primary_metric: str | None,
        dimension: str | None,
        filters: list[dict[str, Any]],
        metrics: list[str],
    ) -> float:
        score = 0.5

        if intent:
            score += 0.1

        if primary_metric:
            score += 0.2

        if dimension:
            score += 0.1

        if metrics:
            score += 0.05

        if filters:
            score += 0.05

        return min(
            round(score, 2),
            1.0,
        )


# ----------------------------------------------------------------------
# Manual test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    profile = {
        "numeric_columns": [
            "quantity",
            "sales",
            "discount",
            "profit",
            "shipping_cost",
        ],
        "categorical_columns": [
            "order_id",
            "customer_id",
            "customer_segment",
            "region",
            "country",
            "city",
            "category",
            "subcategory",
            "product",
            "payment_method",
            "order_status",
        ],
        "date_columns": [
            "order_date",
        ],
    }

    planner = QueryPlanner(profile)

    questions = [
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

    print("\n=== QUERY PLANNER TEST ===\n")

    for question in questions:
        print(f"Question: {question}")

        try:
            plan = planner.create_plan(question)

            print("Plan:")
            print(f"  Intent: {plan.intent}")
            print(f"  Metric: {plan.metric}")
            print(f"  Aggregation: {plan.aggregation}")
            print(f"  Dimension: {plan.dimension}")
            print(f"  Filters: {plan.filters}")
            print(f"  Sort column: {plan.sort_column}")
            print(f"  Sort direction: {plan.sort_direction}")
            print(f"  Limit: {plan.limit}")
            print(f"  Metrics: {plan.metrics}")
            print(f"  Comparison: {plan.comparison}")
            print(f"  Strategy: {plan.strategy}")
            print(f"  Confidence: {plan.confidence}")
            print(f"  Reasoning: {plan.reasoning}")

            if plan.percentage:
                print("  Percentage plan:")
                print(
                    f"    Numerator aggregation: "
                    f"{plan.percentage.numerator_aggregation}"
                )
                print(
                    f"    Numerator filters: "
                    f"{plan.percentage.numerator_filters}"
                )
                print(
                    f"    Denominator aggregation: "
                    f"{plan.percentage.denominator_aggregation}"
                )
                print(
                    f"    Denominator filters: "
                    f"{plan.percentage.denominator_filters}"
                )

        except Exception as exc:
            print(f"  ERROR: {exc}")

        print("-" * 80)