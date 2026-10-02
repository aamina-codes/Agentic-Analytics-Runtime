from dataclasses import dataclass, field
from typing import Any


@dataclass
class FilterCondition:
    """
    Represents a single filter condition.

    Example:
        order_status = 'Cancelled'
        discount >= 0.20
        year(order_date) = 2024
    """

    column: str
    operator: str
    value: Any


@dataclass
class MetricSpec:
    """
    Represents one analytical metric.

    Examples:

        SUM(sales)

        AVG(profit)

        COUNT(*)

        SUM(profit) / SUM(sales)
    """

    metric: str | None = None

    aggregation: str | None = None

    alias: str | None = None

    # Optional derived-metric definition.
    #
    # Example:
    #
    # {
    #     "numerator": {
    #         "metric": "profit",
    #         "aggregation": "sum"
    #     },
    #     "denominator": {
    #         "metric": "sales",
    #         "aggregation": "sum"
    #     }
    # }
    derived: dict[str, Any] | None = None


@dataclass
class PercentagePlan:
    """
    Represents a percentage calculation.

    Example:

        What percentage of orders were cancelled?

        numerator:
            COUNT(*) WHERE order_status = 'Cancelled'

        denominator:
            COUNT(*)
    """

    numerator_aggregation: str = "count"

    numerator_filters: list[FilterCondition] = field(
        default_factory=list
    )

    denominator_aggregation: str = "count"

    denominator_filters: list[FilterCondition] = field(
        default_factory=list
    )


@dataclass
class ComparisonSpec:
    """
    Represents a comparison between analytical values.

    Example:

        How much did sales change between 2024 and 2025?

    periods:
        2024
        2025

    operation:
        difference
    """

    comparison_type: str = "difference"

    periods: list[int] = field(
        default_factory=list
    )

    metric: str | None = None

    aggregation: str | None = None

    # Optional comparison operations.
    #
    # Supported concepts can include:
    #
    # difference
    # absolute_difference
    # percentage_change
    # growth
    operation: str | None = None


@dataclass
class QueryPlan:
    """
    Structured representation of an analytical question.

    The query planner converts natural-language questions
    into an intermediate representation consumed by the
    deterministic SQL generator.

    Existing single-metric fields are retained for
    backward compatibility.

    Newer analytical capabilities can use:

        metrics
        derived metrics
        comparison
        time_granularity
    """

    question: str

    # ------------------------------------------------------------------
    # HIGH-LEVEL INTENT
    # ------------------------------------------------------------------

    intent: str

    # ------------------------------------------------------------------
    # BACKWARD-COMPATIBLE SINGLE METRIC
    # ------------------------------------------------------------------

    metric: str | None = None

    aggregation: str | None = None

    # ------------------------------------------------------------------
    # GROUPING / DIMENSION
    # ------------------------------------------------------------------

    dimension: str | None = None

    # ------------------------------------------------------------------
    # FILTERS
    # ------------------------------------------------------------------

    filters: list[dict[str, Any]] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # SORTING
    # ------------------------------------------------------------------

    sort_column: str | None = None

    sort_direction: str | None = None

    # ------------------------------------------------------------------
    # LIMIT
    # ------------------------------------------------------------------

    limit: int | None = None

    # ------------------------------------------------------------------
    # TIME GROUPING
    # ------------------------------------------------------------------

    time_granularity: str | None = None

    # ------------------------------------------------------------------
    # MULTIPLE METRICS
    # ------------------------------------------------------------------

    metrics: list[MetricSpec] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------------

    comparison: ComparisonSpec | dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # PERCENTAGE
    # ------------------------------------------------------------------

    percentage: PercentagePlan | None = None

    # ------------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------------

    confidence: float = 1.0

    # ------------------------------------------------------------------
    # HUMAN-READABLE REASONING
    # ------------------------------------------------------------------

    reasoning: str | None = None

    # ------------------------------------------------------------------
    # ANALYTICAL STRATEGY
    # ------------------------------------------------------------------

    strategy: str | None = None