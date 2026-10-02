from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ----------------------------------------------------------------------
# PROJECT PATH
# ----------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ----------------------------------------------------------------------
# PROJECT IMPORTS
# ----------------------------------------------------------------------

from src.agent.orchestrator import (
    AnalyticsAgent,
    AgentOrchestrationError,
)


# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------

QUESTIONS_PATH = (
    PROJECT_DIR
    / "evaluation"
    / "questions.json"
)

EXPECTED_RESULTS_PATH = (
    PROJECT_DIR
    / "evaluation"
    / "expected_results.json"
)

RESULTS_PATH = (
    PROJECT_DIR
    / "evaluation"
    / "results.json"
)

DATASET_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "ecommerce_orders.csv"
)


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

NUMERIC_TOLERANCE = 0.01


# ----------------------------------------------------------------------
# VALUE HELPERS
# ----------------------------------------------------------------------

def is_numeric_value(
    value: Any,
) -> bool:
    """
    Return True for Python, NumPy, and pandas numeric scalars,
    while excluding booleans.
    """

    if isinstance(value, bool):
        return False

    try:
        return bool(
            pd.api.types.is_number(value)
        )
    except Exception:
        return False


def numbers_match(
    actual: Any,
    expected: Any,
    tolerance: float = NUMERIC_TOLERANCE,
) -> bool:
    """
    Compare two numeric values using absolute tolerance.
    """

    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (
        TypeError,
        ValueError,
    ):
        return False

    if not (
        math.isfinite(actual_value)
        and math.isfinite(expected_value)
    ):
        return False

    return abs(
        actual_value - expected_value
    ) <= tolerance


def normalize_scalar(
    value: Any,
) -> Any:
    """
    Convert pandas / NumPy scalar values into
    JSON-friendly Python values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


# ----------------------------------------------------------------------
# DATAFRAME HELPERS
# ----------------------------------------------------------------------

def dataframe_to_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert a DataFrame into JSON-safe records.
    """

    if dataframe.empty:
        return []

    records = dataframe.to_dict(
        orient="records"
    )

    normalized_records = []

    for record in records:
        normalized_record = {
            str(key): normalize_scalar(value)
            for key, value in record.items()
        }

        normalized_records.append(
            normalized_record
        )

    return normalized_records


# ----------------------------------------------------------------------
# JSON HELPERS
# ----------------------------------------------------------------------

def load_json(
    path: Path,
) -> Any:
    """
    Load JSON from disk.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def extract_questions(
    payload: Any,
) -> dict[str, str]:
    """
    Extract question IDs and question text.
    """

    if isinstance(payload, dict):

        if "questions" in payload:

            questions = payload["questions"]

            if isinstance(
                questions,
                list,
            ):
                extracted = {}

                for item in questions:

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    question_id = (
                        item.get("id")
                        or item.get("question_id")
                    )

                    question_text = (
                        item.get("question")
                        or item.get("text")
                    )

                    if (
                        question_id
                        and question_text
                    ):
                        extracted[
                            str(question_id)
                        ] = str(question_text)

                return extracted

        extracted = {}

        for key, value in payload.items():

            if isinstance(
                value,
                str,
            ):
                extracted[str(key)] = value

            elif isinstance(
                value,
                dict,
            ):
                question_text = (
                    value.get("question")
                    or value.get("text")
                )

                if question_text:
                    extracted[
                        str(key)
                    ] = str(question_text)

        return extracted

    if isinstance(
        payload,
        list,
    ):
        extracted = {}

        for index, item in enumerate(
            payload,
            start=1,
        ):

            if isinstance(
                item,
                dict,
            ):
                question_id = (
                    item.get("id")
                    or item.get("question_id")
                    or f"q{index:02d}"
                )

                question_text = (
                    item.get("question")
                    or item.get("text")
                )

                if question_text:
                    extracted[
                        str(question_id)
                    ] = str(question_text)

        return extracted

    raise ValueError(
        "Unsupported questions.json format."
    )


def extract_expected_results(
    payload: Any,
) -> dict[str, dict[str, Any]]:
    """
    Extract expected results.
    """

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "expected_results.json must contain an object."
        )

    if "results" in payload:

        results = payload["results"]

        if isinstance(
            results,
            dict,
        ):
            return results

    return payload


# ----------------------------------------------------------------------
# EXPECTED-RESULT TYPE DETECTION
# ----------------------------------------------------------------------

def is_ranking_expected(
    expected: dict[str, Any],
) -> bool:

    return (
        "answer" in expected
        and "value" in expected
    )


def is_single_numeric_expected(
    expected: dict[str, Any],
) -> bool:

    answer = expected.get(
        "answer"
    )

    return (
        is_numeric_value(answer)
    )


def is_percentage_expected(
    expected: dict[str, Any],
) -> bool:

    return (
        "percentage" in expected
    )


def is_multi_row_expected(
    expected: dict[str, Any],
) -> bool:

    return isinstance(
        expected.get("rows"),
        list,
    )


# ----------------------------------------------------------------------
# BASIC VALUE COMPARISON
# ----------------------------------------------------------------------

def values_equal(
    actual: Any,
    expected: Any,
) -> bool:

    if is_numeric_value(
        expected
    ):
        return numbers_match(
            actual,
            expected,
        )

    if is_numeric_value(
        actual
    ):
        return numbers_match(
            actual,
            expected,
        )

    return (
        str(actual).strip().lower()
        == str(expected).strip().lower()
    )


# ----------------------------------------------------------------------
# RANKING COMPARISON
# ----------------------------------------------------------------------

def compare_ranking_result(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:

    if actual.empty:
        return (
            False,
            "Agent returned an empty result.",
        )

    expected_answer = expected.get(
        "answer"
    )

    expected_value = expected.get(
        "value"
    )

    first_row = actual.iloc[0]

    entity_value = None
    numeric_value = None

    for column in actual.columns:

        value = first_row[column]

        if (
            entity_value is None
            and isinstance(
                value,
                str,
            )
        ):
            entity_value = value

        if (
            numeric_value is None
            and is_numeric_value(value)
        ):
            numeric_value = value

    if entity_value is None:
        return (
            False,
            "Could not identify the ranked entity.",
        )

    if numeric_value is None:
        return (
            False,
            "Could not identify the ranked numeric value.",
        )

    entity_match = (
        str(entity_value).strip().lower()
        == str(expected_answer).strip().lower()
    )

    value_match = numbers_match(
        numeric_value,
        expected_value,
    )

    if entity_match and value_match:
        return (
            True,
            "Ranking result matches.",
        )

    reasons = []

    if not entity_match:
        reasons.append(
            f"expected entity "
            f"'{expected_answer}', "
            f"got '{entity_value}'"
        )

    if not value_match:
        reasons.append(
            f"expected value "
            f"{expected_value}, "
            f"got {numeric_value}"
        )

    return (
        False,
        "; ".join(reasons),
    )


# ----------------------------------------------------------------------
# SINGLE NUMERIC COMPARISON
# ----------------------------------------------------------------------

def compare_single_numeric_result(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:

    if actual.empty:
        return (
            False,
            "Agent returned an empty result.",
        )

    expected_value = expected.get(
        "answer"
    )

    first_row = actual.iloc[0]

    actual_value = None

    for column in actual.columns:

        value = first_row[column]

        if is_numeric_value(value):
            actual_value = value
            break

    if actual_value is None:
        return (
            False,
            "Could not identify numeric result.",
        )

    if numbers_match(
        actual_value,
        expected_value,
    ):
        return (
            True,
            "Numeric result matches.",
        )

    return (
        False,
        f"Expected {expected_value}, "
        f"got {actual_value}.",
    )


# ----------------------------------------------------------------------
# PERCENTAGE COMPARISON
# ----------------------------------------------------------------------

def compare_percentage_result(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:

    if actual.empty:
        return (
            False,
            "Agent returned an empty result.",
        )

    expected_percentage = expected.get(
        "percentage"
    )

    first_row = actual.iloc[0]

    actual_percentage = None

    for column in actual.columns:

        value = first_row[column]

        if is_numeric_value(value):
            actual_percentage = value
            break

    if actual_percentage is None:
        return (
            False,
            "Could not identify percentage result.",
        )

    if numbers_match(
        actual_percentage,
        expected_percentage,
    ):
        return (
            True,
            "Percentage result matches.",
        )

    return (
        False,
        f"Expected percentage "
        f"{expected_percentage}, "
        f"got {actual_percentage}.",
    )


# ----------------------------------------------------------------------
# YEAR-OVER-YEAR CHANGE
# ----------------------------------------------------------------------

def compare_change_result(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:

    if actual.empty:
        return (
            False,
            "Agent returned an empty result.",
        )

    expected_change = expected.get(
        "absolute_change"
    )

    first_row = actual.iloc[0]

    actual_change = None

    for column in actual.columns:

        if (
            str(column).lower()
            == "change"
        ):
            actual_change = first_row[column]
            break

    if actual_change is None:

        for column in actual.columns:

            value = first_row[column]

            if is_numeric_value(value):
                actual_change = value
                break

    if actual_change is None:
        return (
            False,
            "Could not find the absolute change "
            "in agent result.",
        )

    if numbers_match(
        actual_change,
        expected_change,
    ):
        return (
            True,
            "Absolute year-over-year change matches.",
        )

    return (
        False,
        f"Expected absolute change "
        f"{expected_change}, "
        f"got {actual_change}.",
    )


# ----------------------------------------------------------------------
# COUNT RANKING
# ----------------------------------------------------------------------

def compare_count_ranking_result(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:

    if actual.empty:
        return (
            False,
            "Agent returned an empty result.",
        )

    expected_answer = expected.get(
        "answer"
    )

    expected_count = expected.get(
        "order_count"
    )

    first_row = actual.iloc[0]

    entity_value = None
    numeric_value = None

    for column in actual.columns:

        value = first_row[column]

        if (
            entity_value is None
            and isinstance(
                value,
                str,
            )
        ):
            entity_value = value

        if (
            numeric_value is None
            and is_numeric_value(value)
        ):
            numeric_value = value

    if entity_value is None:
        return (
            False,
            "Could not find the ranked payment method.",
        )

    if numeric_value is None:
        return (
            False,
            "Could not find the order count.",
        )

    entity_match = (
        str(entity_value).strip().lower()
        == str(expected_answer).strip().lower()
    )

    count_match = numbers_match(
        numeric_value,
        expected_count,
    )

    if entity_match and count_match:
        return (
            True,
            "Ranked payment method and order count match.",
        )

    reasons = []

    if not entity_match:
        reasons.append(
            f"expected payment method "
            f"'{expected_answer}', "
            f"got '{entity_value}'"
        )

    if not count_match:
        reasons.append(
            f"expected order count "
            f"{expected_count}, "
            f"got {numeric_value}"
        )

    return (
        False,
        "; ".join(reasons),
    )


# ----------------------------------------------------------------------
# PROFIT MARGIN
# ----------------------------------------------------------------------

def compare_profit_margin_result(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:

    if actual.empty:
        return (
            False,
            "Agent returned an empty result.",
        )

    expected_answer = expected.get(
        "answer"
    )

    expected_margin = expected.get(
        "profit_margin"
    )

    first_row = actual.iloc[0]

    entity_value = None
    margin_value = None

    for column in actual.columns:

        value = first_row[column]

        if (
            entity_value is None
            and isinstance(
                value,
                str,
            )
        ):
            entity_value = value

        if (
            margin_value is None
            and is_numeric_value(value)
        ):
            margin_value = value

    if entity_value is None:
        return (
            False,
            "Could not find the ranked region.",
        )

    if margin_value is None:
        return (
            False,
            "Could not find the profit margin.",
        )

    entity_match = (
        str(entity_value).strip().lower()
        == str(expected_answer).strip().lower()
    )

    margin_match = numbers_match(
        margin_value,
        expected_margin,
    )

    if entity_match and margin_match:
        return (
            True,
            "Ranked region and profit margin match.",
        )

    reasons = []

    if not entity_match:
        reasons.append(
            f"expected region "
            f"'{expected_answer}', "
            f"got '{entity_value}'"
        )

    if not margin_match:
        reasons.append(
            f"expected profit margin "
            f"{expected_margin}, "
            f"got {margin_value}"
        )

    return (
        False,
        "; ".join(reasons),
    )


# ----------------------------------------------------------------------
# TOP-GROUP COMPARISON
# ----------------------------------------------------------------------

def compare_top_groups_result(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:

    if actual.empty:
        return (
            False,
            "Agent returned an empty result.",
        )

    expected_difference = expected.get(
        "difference"
    )

    first_row = actual.iloc[0]

    actual_difference = None

    for column in actual.columns:

        if (
            str(column).lower()
            == "difference"
        ):
            actual_difference = first_row[column]
            break

    if actual_difference is None:

        for column in actual.columns:

            value = first_row[column]

            if is_numeric_value(value):
                actual_difference = value
                break

    if actual_difference is None:
        return (
            False,
            "Could not find the sales difference.",
        )

    if numbers_match(
        actual_difference,
        expected_difference,
    ):
        return (
            True,
            "Top-group sales difference matches.",
        )

    return (
        False,
        f"Expected sales difference "
        f"{expected_difference}, "
        f"got {actual_difference}.",
    )


# ----------------------------------------------------------------------
# MULTI-METRIC RANKING
# ----------------------------------------------------------------------

def compare_multi_metric_ranking_result(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:

    if actual.empty:
        return (
            False,
            "Agent returned an empty result.",
        )

    expected_category = expected.get(
        "category"
    )

    expected_discount = expected.get(
        "average_discount"
    )

    expected_profit = expected.get(
        "average_profit"
    )

    first_row = actual.iloc[0]

    category_value = None

    for column in actual.columns:

        value = first_row[column]

        if isinstance(
            value,
            str,
        ):
            category_value = value
            break

    if category_value is None:
        return (
            False,
            "Could not find the ranked category.",
        )

    actual_discount = None
    actual_profit = None

    for column in actual.columns:

        normalized = str(
            column
        ).lower()

        value = first_row[column]

        if (
            "discount" in normalized
            and is_numeric_value(value)
        ):
            actual_discount = value

        elif (
            "profit" in normalized
            and is_numeric_value(value)
        ):
            actual_profit = value

    numeric_values = []

    for column in actual.columns:

        value = first_row[column]

        if is_numeric_value(value):
            numeric_values.append(value)

    if actual_discount is None:

        if numeric_values:
            actual_discount = numeric_values[0]

    if actual_profit is None:

        if len(numeric_values) >= 2:
            actual_profit = numeric_values[1]

    if actual_discount is None:
        return (
            False,
            "Could not find average discount.",
        )

    if actual_profit is None:
        return (
            False,
            "Could not find average profit.",
        )

    category_match = (
        str(category_value).strip().lower()
        == str(expected_category).strip().lower()
    )

    discount_match = numbers_match(
        actual_discount,
        expected_discount,
    )

    profit_match = numbers_match(
        actual_profit,
        expected_profit,
    )

    if (
        category_match
        and discount_match
        and profit_match
    ):
        return (
            True,
            "Category and both average metrics match.",
        )

    reasons = []

    if not category_match:
        reasons.append(
            f"expected category "
            f"'{expected_category}', "
            f"got '{category_value}'"
        )

    if not discount_match:
        reasons.append(
            f"expected average discount "
            f"{expected_discount}, "
            f"got {actual_discount}"
        )

    if not profit_match:
        reasons.append(
            f"expected average profit "
            f"{expected_profit}, "
            f"got {actual_profit}"
        )

    return (
        False,
        "; ".join(reasons),
    )


# ----------------------------------------------------------------------
# MULTI-ROW COMPARISON
# ----------------------------------------------------------------------

def resolve_actual_column(
    actual_row: dict[str, Any],
    expected_column: str,
) -> str | None:
    """
    Resolve semantic aliases between expected and actual
    result column names.

    Examples:

        total_sales -> sum_sales
        total_profit -> sum_profit
        sales -> total_sales
    """

    if expected_column in actual_row:
        return expected_column

    aliases = {
        "total_sales": [
            "total_sales",
            "sum_sales",
            "sales",
        ],
        "total_profit": [
            "total_profit",
            "sum_profit",
            "profit",
        ],
        "sales": [
            "sales",
            "total_sales",
            "sum_sales",
        ],
        "profit": [
            "profit",
            "total_profit",
            "sum_profit",
        ],
    }

    candidates = aliases.get(
        expected_column,
        [expected_column],
    )

    for candidate in candidates:

        if candidate in actual_row:
            return candidate

    return None


def row_matches(
    actual_row: dict[str, Any],
    expected_row: dict[str, Any],
) -> bool:
    """
    Determine whether one actual row semantically matches
    one expected row.
    """

    for key, expected_value in expected_row.items():

        actual_key = resolve_actual_column(
            actual_row,
            key,
        )

        if actual_key is None:
            return False

        actual_value = actual_row[
            actual_key
        ]

        if not values_equal(
            actual_value,
            expected_value,
        ):
            return False

    return True


def compare_multi_row_result(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:
    """
    Compare multi-row results without assuming row ordering.

    SQL GROUP BY queries do not guarantee row order unless an
    ORDER BY clause is explicitly present. Therefore the evaluator
    should compare rows by semantic content rather than position.

    Matching is one-to-one:
    each expected row must match exactly one actual row.
    """

    expected_rows = expected.get(
        "rows"
    )

    if not isinstance(
        expected_rows,
        list,
    ):
        return (
            False,
            "Expected rows must be a list.",
        )

    actual_records = dataframe_to_records(
        actual
    )

    if len(actual_records) != len(
        expected_rows
    ):
        return (
            False,
            f"Expected {len(expected_rows)} "
            f"rows, got {len(actual_records)}.",
        )

    # Track actual rows already matched so that one actual row
    # cannot satisfy multiple expected rows.
    unmatched_actual_indices = set(
        range(len(actual_records))
    )

    for expected_index, expected_row in enumerate(
        expected_rows,
        start=1,
    ):

        matching_index = None

        for actual_index in unmatched_actual_indices:

            actual_row = actual_records[
                actual_index
            ]

            if row_matches(
                actual_row,
                expected_row,
            ):
                matching_index = actual_index
                break

        if matching_index is None:
            return (
                False,
                f"Could not find a matching actual row "
                f"for expected row {expected_index}: "
                f"{expected_row}. "
                f"Actual rows: {actual_records}",
            )

        unmatched_actual_indices.remove(
            matching_index
        )

    return (
        True,
        "All expected rows match, independent of row order.",
    )


# ----------------------------------------------------------------------
# SEMANTIC COMPARISON DISPATCHER
# ----------------------------------------------------------------------

def compare_semantically(
    actual: pd.DataFrame,
    expected: dict[str, Any],
) -> tuple[bool, str]:

    if not isinstance(
        expected,
        dict,
    ):
        return (
            False,
            "Unsupported expected-result format.",
        )

    # Multi-row
    if is_multi_row_expected(
        expected
    ):
        return compare_multi_row_result(
            actual,
            expected,
        )

    # Percentage
    if is_percentage_expected(
        expected
    ):
        return compare_percentage_result(
            actual,
            expected,
        )

    # Year-over-year change
    if (
        "absolute_change" in expected
        and "percentage_change" in expected
    ):
        return compare_change_result(
            actual,
            expected,
        )

    # Count ranking
    if "order_count" in expected:
        return compare_count_ranking_result(
            actual,
            expected,
        )

    # Profit margin
    if "profit_margin" in expected:
        return compare_profit_margin_result(
            actual,
            expected,
        )

    # Top groups
    if (
        "segment_1" in expected
        and "segment_2" in expected
        and "difference" in expected
    ):
        return compare_top_groups_result(
            actual,
            expected,
        )

    # Multi-metric ranking
    if (
        "category" in expected
        and "average_discount" in expected
        and "average_profit" in expected
    ):
        return compare_multi_metric_ranking_result(
            actual,
            expected,
        )

    # Standard ranking
    if is_ranking_expected(
        expected
    ):
        return compare_ranking_result(
            actual,
            expected,
        )

    # Single numeric
    if is_single_numeric_expected(
        expected
    ):
        return compare_single_numeric_result(
            actual,
            expected,
        )

    return (
        False,
        "Could not determine semantic comparison strategy.",
    )


# ----------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------

def evaluate() -> dict[str, Any]:

    questions_payload = load_json(
        QUESTIONS_PATH
    )

    expected_payload = load_json(
        EXPECTED_RESULTS_PATH
    )

    questions = extract_questions(
        questions_payload
    )

    expected_results = extract_expected_results(
        expected_payload
    )

    if not questions:
        raise ValueError(
            "No evaluation questions found."
        )

    if not expected_results:
        raise ValueError(
            "No expected results found."
        )

    print()
    print("=" * 80)
    print(
        "AGENTIC ANALYTICS RUNTIME — EVALUATION"
    )
    print("=" * 80)
    print()

    print(
        f"Dataset: {DATASET_PATH}"
    )

    print(
        f"Questions: {len(questions)}"
    )

    print()

    evaluation_results = []

    passed = 0

    agent = None

    try:

        agent = AnalyticsAgent(
            dataset_path=DATASET_PATH
        )

        for index, (
            question_id,
            question,
        ) in enumerate(
            questions.items(),
            start=1,
        ):

            print(
                f"[{index}/{len(questions)}] "
                f"{question_id}",
                end=" ",
            )

            expected = expected_results.get(
                question_id
            )

            if expected is None:

                print(
                    "FAIL: Missing expected result."
                )

                evaluation_results.append(
                    {
                        "question_id": question_id,
                        "question": question,
                        "status": "FAIL",
                        "error": (
                            "Missing expected result."
                        ),
                    }
                )

                continue

            record = {
                "question_id": question_id,
                "question": question,
                "expected_result": expected,
            }

            try:

                response = agent.run(
                    question
                )

                actual_result = response.result

                matches, reason = (
                    compare_semantically(
                        actual_result,
                        expected,
                    )
                )

                record.update(
                    {
                        "status": (
                            "PASS"
                            if matches
                            else "FAIL"
                        ),
                        "answer": response.answer,
                        "sql": response.sql,
                        "actual_result": (
                            dataframe_to_records(
                                actual_result
                            )
                        ),
                        "comparison": reason,
                    }
                )

                if matches:

                    passed += 1

                    print(
                        "PASS"
                    )

                else:

                    print(
                        f"FAIL: {reason}"
                    )

            except AgentOrchestrationError as exc:

                record.update(
                    {
                        "status": "FAIL",
                        "error": str(exc),
                    }
                )

                print(
                    f"FAIL: {exc}"
                )

            except Exception as exc:

                record.update(
                    {
                        "status": "FAIL",
                        "error": (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    }
                )

                print(
                    f"FAIL: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            evaluation_results.append(
                record
            )

    finally:

        if agent is not None:

            try:
                agent.close()
            except Exception:
                pass

    total = len(
        questions
    )

    accuracy = (
        passed / total * 100
        if total
        else 0.0
    )

    output = {
        "summary": {
            "total_questions": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy": accuracy,
        },
        "results": evaluation_results,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print(
        "EVALUATION SUMMARY"
    )
    print("=" * 80)
    print()

    print(
        f"Passed:   {passed}/{total}"
    )

    print(
        f"Failed:   {total - passed}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        RESULTS_PATH
    )

    print()

    return output


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()