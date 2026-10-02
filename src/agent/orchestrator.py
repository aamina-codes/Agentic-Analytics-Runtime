from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.agent.planner import QueryPlanner
from src.agent.reasoning import ReasoningEngine
from src.agent.sql_generator import SQLGenerator
from src.data.profiler import profile_dataset
from src.database.executor import QueryExecutor
from src.evidence.evidence_builder import Evidence, EvidenceBuilder
from src.validation.result_validator import (
    ResultValidationError,
    ResultValidationReport,
    ResultValidator,
)
from src.visualization.chart_builder import ChartBuilder
from src.visualization.chart_selector import ChartSelector, ChartSpec


class AgentOrchestrationError(Exception):
    """Raised when the analytics agent pipeline cannot complete."""


@dataclass
class AgentResponse:
    """
    Complete response produced by the analytics agent.

    Contains both the human-readable answer and the
    underlying evidence needed to inspect how it was produced.
    """

    question: str
    answer: str
    sql: str
    result: pd.DataFrame
    evidence: Evidence
    chart: object | None
    chart_spec: ChartSpec | None
    validation: ResultValidationReport


class AnalyticsAgent:
    """
    End-to-end deterministic analytics agent.

    Pipeline:

        Question
            ↓
        Planner
            ↓
        SQL Generator
            ↓
        SQL Guard + Executor
            ↓
        Result Validator
            ↓
        Chart Selector
            ↓
        Chart Builder
            ↓
        Evidence Builder
            ↓
        Reasoning Engine
            ↓
        AgentResponse

    Design principle:

        LLM proposes; deterministic code decides.

    This current implementation is fully deterministic.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        table_name: str = "ecommerce_orders",
        evidence_max_rows: int = 10,
    ):
        self.dataset_path = Path(dataset_path)
        self.table_name = table_name

        if not self.dataset_path.exists():
            raise AgentOrchestrationError(
                f"Dataset not found: {self.dataset_path}"
            )

        # --------------------------------------------------------------
        # Dataset understanding
        # --------------------------------------------------------------

        try:
            self.dataset_profile = profile_dataset(
                self.dataset_path
            )
        except Exception as exc:
            raise AgentOrchestrationError(
                f"Dataset profiling failed: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Pipeline components
        # --------------------------------------------------------------

        self.planner = QueryPlanner(
            self.dataset_profile
        )

        self.sql_generator = SQLGenerator(
            table_name=self.table_name
        )

        self.executor = QueryExecutor(
            dataset_path=self.dataset_path,
            table_name=self.table_name,
        )

        self.result_validator = ResultValidator()

        self.chart_selector = ChartSelector()
        self.chart_builder = ChartBuilder()

        self.evidence_builder = EvidenceBuilder(
            max_rows=evidence_max_rows
        )

        self.reasoning_engine = ReasoningEngine()

    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
    ) -> AgentResponse:
        """
        Run a natural-language question through the full
        analytics pipeline.
        """

        if not isinstance(question, str):
            raise AgentOrchestrationError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:
            raise AgentOrchestrationError(
                "Question cannot be empty."
            )

        # --------------------------------------------------------------
        # Step 1: Planning
        # --------------------------------------------------------------

        try:
            plan = self.planner.create_plan(
                question
            )
        except Exception as exc:
            raise AgentOrchestrationError(
                f"Query planning failed: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Step 2: SQL generation
        # --------------------------------------------------------------

        try:
            sql = self.sql_generator.generate(
                plan
            )
        except Exception as exc:
            raise AgentOrchestrationError(
                f"SQL generation failed: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Step 3: SQL validation + execution
        # --------------------------------------------------------------

        try:
            result = self.executor.execute(
                sql
            )
        except Exception as exc:
            raise AgentOrchestrationError(
                f"Query execution failed: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Step 4: Result validation
        # --------------------------------------------------------------

        try:
            validation = self.result_validator.validate(
                result
            )
        except ResultValidationError as exc:
            raise AgentOrchestrationError(
                f"Result validation failed: {exc}"
            ) from exc
        except Exception as exc:
            raise AgentOrchestrationError(
                f"Unexpected result validation error: {exc}"
            ) from exc

        if not validation.valid:
            raise AgentOrchestrationError(
                "Query result failed validation."
            )

        # --------------------------------------------------------------
        # Step 5: Chart selection
        # --------------------------------------------------------------

        chart_spec: ChartSpec | None = None
        chart = None

        try:
            chart_spec = self.chart_selector.select(
                result,
                plan,
            )

            chart = self.chart_builder.build(
                result,
                chart_spec,
            )

        except Exception as exc:
            # Visualization is useful but should not destroy
            # an otherwise valid analytical answer.
            chart_spec = None
            chart = None

            print(
                f"Visualization warning: {exc}"
            )

        # --------------------------------------------------------------
        # Step 6: Evidence construction
        # --------------------------------------------------------------

        try:
            evidence = self.evidence_builder.build(
                question,
                sql,
                result,
            )
        except Exception as exc:
            raise AgentOrchestrationError(
                f"Evidence construction failed: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Step 7: Human-readable reasoning
        # --------------------------------------------------------------

        try:
            answer = self.reasoning_engine.answer(
                evidence
            )
        except Exception as exc:
            raise AgentOrchestrationError(
                f"Answer generation failed: {exc}"
            ) from exc

        # --------------------------------------------------------------
        # Step 8: Final response
        # --------------------------------------------------------------

        return AgentResponse(
            question=question,
            answer=answer.answer,
            sql=sql,
            result=result,
            evidence=evidence,
            chart=chart,
            chart_spec=chart_spec,
            validation=validation,
        )

    # ------------------------------------------------------------------
    # RESOURCE MANAGEMENT
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""

        self.executor.close()

    def __enter__(self) -> "AnalyticsAgent":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()


# ======================================================================
# LOCAL END-TO-END TEST
# ======================================================================

if __name__ == "__main__":

    print("\n=== AGENTIC ANALYTICS RUNTIME TEST ===\n")

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "data"
        / "raw"
        / "ecommerce_orders.csv"
    )

    agent = AnalyticsAgent(
        dataset_path=dataset_path,
        table_name="ecommerce_orders",
    )

    test_questions = [
        "Which region generated the highest total sales?",
        "Which product category generated the highest total profit?",
        "What was the total sales revenue in 2024?",
        "What percentage of all orders were cancelled?",
    ]

    for question in test_questions:

        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")
        print("=" * 80)

        try:

            response = agent.run(
                question
            )

            print("\nANSWER")
            print("-" * 80)
            print(response.answer)

            print("\nSQL")
            print("-" * 80)
            print(response.sql)

            print("\nRESULT")
            print("-" * 80)
            print(
                response.result.to_string(
                    index=False
                )
            )

            print("\nEVIDENCE")
            print("-" * 80)
            print(
                response.evidence.summary
            )

            print(
                f"Rows retained: "
                f"{len(response.evidence.result)}"
            )

            print("\nVALIDATION")
            print("-" * 80)
            print(
                f"Valid: {response.validation.valid}"
            )
            print(
                f"Rows: {response.validation.row_count}"
            )
            print(
                f"Columns: "
                f"{response.validation.column_count}"
            )

            print("\nVISUALIZATION")
            print("-" * 80)

            if response.chart_spec is not None:
                print(
                    f"Chart type: "
                    f"{response.chart_spec.chart_type}"
                )
                print(
                    f"X column: "
                    f"{response.chart_spec.x_column}"
                )
                print(
                    f"Y column: "
                    f"{response.chart_spec.y_column}"
                )
            else:
                print(
                    "No visualization generated."
                )

        except AgentOrchestrationError as exc:

            print("\nPIPELINE: FAIL")
            print(f"Reason: {exc}")

    agent.close()

    print(
        "\n\nAgent orchestration test completed."
    )