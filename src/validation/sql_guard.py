from __future__ import annotations

import re


class SQLGuardError(Exception):
    """Raised when a SQL query violates a safety rule."""


class SQLGuard:
    """
    Deterministic SQL safety guard for Agentic Analytics Runtime.

    Allows read-only SELECT/WITH queries against the configured dataset
    table and internally declared CTEs.

    Blocks:
    - destructive SQL
    - multiple statements
    - unauthorized tables
    - dangerous DuckDB constructs
    """

    FORBIDDEN_KEYWORDS = {
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "REPLACE",
        "ATTACH",
        "DETACH",
        "COPY",
        "INSTALL",
        "LOAD",
        "CALL",
        "EXPORT",
        "IMPORT",
    }

    ALLOWED_STATEMENT_TYPES = {"SELECT", "WITH"}

    def __init__(self, allowed_table: str = "ecommerce_orders"):
        self.allowed_table = allowed_table.lower()

    # ------------------------------------------------------------------
    # Basic normalization
    # ------------------------------------------------------------------

    def _normalize_sql(self, sql: str) -> str:
        """Normalize whitespace and remove SQL comments."""

        sql = sql.strip()

        # Remove single-line comments.
        sql = re.sub(
            r"--.*?$",
            "",
            sql,
            flags=re.MULTILINE,
        )

        # Remove block comments.
        sql = re.sub(
            r"/\*.*?\*/",
            "",
            sql,
            flags=re.DOTALL,
        )

        # Normalize whitespace.
        sql = re.sub(
            r"\s+",
            " ",
            sql,
        )

        return sql.strip()

    # ------------------------------------------------------------------
    # Statement validation
    # ------------------------------------------------------------------

    def _validate_statement_type(self, sql: str) -> None:
        """Allow only SELECT or WITH statements."""

        match = re.match(
            r"^\s*([A-Za-z]+)",
            sql,
        )

        if not match:
            raise SQLGuardError(
                "Unable to determine SQL statement type."
            )

        statement_type = match.group(1).upper()

        if statement_type not in self.ALLOWED_STATEMENT_TYPES:
            raise SQLGuardError(
                f"Statement type '{statement_type}' is not allowed."
            )

    # ------------------------------------------------------------------
    # Forbidden keywords
    # ------------------------------------------------------------------

    def _validate_forbidden_keywords(self, sql: str) -> None:
        """
        Block dangerous SQL keywords anywhere in the query.
        """

        for keyword in self.FORBIDDEN_KEYWORDS:

            pattern = rf"\b{re.escape(keyword)}\b"

            if re.search(
                pattern,
                sql,
                flags=re.IGNORECASE,
            ):
                raise SQLGuardError(
                    f"Forbidden SQL keyword detected: {keyword}."
                )

    # ------------------------------------------------------------------
    # Multiple statement detection
    # ------------------------------------------------------------------

    def _validate_single_statement(self, sql: str) -> None:
        """
        Allow at most one SQL statement.

        A trailing semicolon is allowed.
        """

        stripped = sql.strip()

        if ";" not in stripped:
            return

        without_trailing = (
            stripped.rstrip(";").strip()
        )

        if ";" in without_trailing:
            raise SQLGuardError(
                "Multiple SQL statements are not allowed."
            )

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, sql: str) -> list[str]:
        """
        Tokenize identifiers and parentheses.

        This is intentionally a small deterministic tokenizer rather
        than a full SQL parser.
        """

        pattern = (
            r"[A-Za-z_][A-Za-z0-9_]*"
            r"|[(),]"
        )

        return [
            match.group(0)
            for match in re.finditer(
                pattern,
                sql,
                flags=re.IGNORECASE,
            )
        ]

    # ------------------------------------------------------------------
    # CTE extraction
    # ------------------------------------------------------------------

    def _find_cte_names(self, sql: str) -> set[str]:
        """
        Find CTE names declared by a WITH clause.

        Example:

            WITH yearly_values AS (...),
                 ranked AS (...)
            SELECT ...
            FROM ranked;

        Returns:

            {"yearly_values", "ranked"}
        """

        tokens = self._tokenize(sql)

        if not tokens:
            return set()

        if tokens[0].upper() != "WITH":
            return set()

        cte_names: set[str] = set()

        index = 1

        # Support WITH RECURSIVE.
        if (
            index < len(tokens)
            and tokens[index].upper() == "RECURSIVE"
        ):
            index += 1

        while index < len(tokens):

            token = tokens[index]

            # A CTE declaration looks like:
            #
            # name AS (
            #
            if (
                re.match(
                    r"^[A-Za-z_][A-Za-z0-9_]*$",
                    token,
                )
                and index + 2 < len(tokens)
                and tokens[index + 1].upper() == "AS"
                and tokens[index + 2] == "("
            ):

                cte_names.add(
                    token.lower()
                )

                # Move into the CTE body.
                depth = 1
                index += 3

                while (
                    index < len(tokens)
                    and depth > 0
                ):

                    current = tokens[index]

                    if current == "(":
                        depth += 1

                    elif current == ")":
                        depth -= 1

                    index += 1

                # After closing the CTE body, the next token
                # may be a comma followed by another CTE.
                if (
                    index < len(tokens)
                    and tokens[index] == ","
                ):
                    index += 1

                continue

            break

        return cte_names

    # ------------------------------------------------------------------
    # FROM / JOIN extraction
    # ------------------------------------------------------------------

    def _find_table_references(
        self,
        sql: str,
    ) -> list[str]:
        """
        Find identifiers appearing after FROM or JOIN at any nesting
        level.

        This intentionally examines nested CTEs and subqueries.

        Example:

            WITH safe_data AS (
                SELECT *
                FROM another_table
            )
            SELECT *
            FROM safe_data;

        Returns:

            ["another_table", "safe_data"]

        EXTRACT(YEAR FROM order_date) is ignored because the FROM is
        inside an EXTRACT expression rather than being a table clause.
        """

        tokens = self._tokenize(sql)

        tables: list[str] = []

        # Track parentheses so we can identify function/expression
        # contexts and distinguish them from table references.
        depth = 0

        for index, token in enumerate(tokens):

            upper_token = token.upper()

            if token == "(":
                depth += 1
                continue

            if token == ")":
                depth = max(
                    0,
                    depth - 1,
                )
                continue

            if upper_token not in {
                "FROM",
                "JOIN",
            }:
                continue

            # ----------------------------------------------------------
            # Ignore FROM inside EXTRACT(...)
            #
            # EXTRACT(YEAR FROM order_date)
            #
            # `order_date` is a column, not a table.
            # ----------------------------------------------------------

            expression_context = False

            lookback_start = max(
                0,
                index - 3,
            )

            for previous_index in range(
                lookback_start,
                index,
            ):
                if (
                    tokens[previous_index].upper()
                    == "EXTRACT"
                ):
                    expression_context = True
                    break

            if expression_context:
                continue

            # ----------------------------------------------------------
            # Find the identifier immediately following FROM/JOIN.
            # ----------------------------------------------------------

            next_index = index + 1

            while next_index < len(tokens):

                next_token = tokens[next_index]

                if next_token in {
                    ",",
                    "(",
                    ")",
                }:
                    break

                if re.match(
                    r"^[A-Za-z_][A-Za-z0-9_]*$",
                    next_token,
                ):
                    tables.append(
                        next_token
                    )
                    break

                next_index += 1

        return tables

    # ------------------------------------------------------------------
    # Table authorization
    # ------------------------------------------------------------------

    def _validate_table_usage(
        self,
        sql: str,
    ) -> None:
        """
        Allow the configured dataset table and internally declared CTEs.

        Every FROM/JOIN reference is still inspected, including nested
        CTEs and subqueries.
        """

        tables = self._find_table_references(
            sql
        )

        cte_names = self._find_cte_names(
            sql
        )

        for table in tables:

            table_lower = table.lower()

            if table_lower == self.allowed_table:
                continue

            if table_lower in cte_names:
                continue

            raise SQLGuardError(
                f"Unauthorized table referenced: {table}"
            )

    # ------------------------------------------------------------------
    # Dangerous constructs
    # ------------------------------------------------------------------

    def _validate_dangerous_constructs(
        self,
        sql: str,
    ) -> None:
        """Block additional DuckDB features that should not be exposed."""

        dangerous_patterns = {
            r"\bPRAGMA\b":
                "PRAGMA statements are not allowed.",

            r"\bSEQUENCE\b":
                "SEQUENCE operations are not allowed.",

            r"\bFUNCTION\b":
                "Function definitions are not allowed.",

            r"\bMACRO\b":
                "Macro definitions are not allowed.",
        }

        for pattern, message in (
            dangerous_patterns.items()
        ):

            if re.search(
                pattern,
                sql,
                flags=re.IGNORECASE,
            ):
                raise SQLGuardError(
                    message
                )

    # ------------------------------------------------------------------
    # Public validation method
    # ------------------------------------------------------------------

    def validate(
        self,
        sql: str,
    ) -> bool:
        """
        Validate SQL.

        Returns:
            True if the query passes all safety checks.

        Raises:
            SQLGuardError if any safety rule is violated.
        """

        if not isinstance(sql, str):
            raise SQLGuardError(
                "SQL query must be a string."
            )

        if not sql.strip():
            raise SQLGuardError(
                "SQL query cannot be empty."
            )

        normalized_sql = self._normalize_sql(
            sql
        )

        self._validate_single_statement(
            normalized_sql
        )

        self._validate_statement_type(
            normalized_sql
        )

        self._validate_forbidden_keywords(
            normalized_sql
        )

        self._validate_table_usage(
            normalized_sql
        )

        self._validate_dangerous_constructs(
            normalized_sql
        )

        return True


# ======================================================================
# TESTS
# ======================================================================

if __name__ == "__main__":

    print("\n=== SQL GUARD TEST ===\n")

    guard = SQLGuard(
        allowed_table="ecommerce_orders"
    )

    safe_queries = [

        """
        SELECT
            region,
            SUM(sales) AS total_sales
        FROM ecommerce_orders
        GROUP BY region
        ORDER BY total_sales DESC
        LIMIT 1;
        """,

        """
        SELECT
            SUM(sales) AS total_sales
        FROM ecommerce_orders
        WHERE EXTRACT(YEAR FROM order_date) = 2024;
        """,

        """
        SELECT
            category,
            AVG(profit) AS avg_profit
        FROM ecommerce_orders
        GROUP BY category;
        """,

        """
        WITH yearly_values AS (
            SELECT
                EXTRACT(YEAR FROM order_date) AS year,
                SUM(sales) AS value
            FROM ecommerce_orders
            GROUP BY year
        )
        SELECT
            MAX(value) AS max_value
        FROM yearly_values;
        """,

        """
        WITH yearly_values AS (
            SELECT
                EXTRACT(YEAR FROM order_date) AS year,
                SUM(sales) AS value
            FROM ecommerce_orders
            GROUP BY year
        ),
        ranked AS (
            SELECT
                year,
                value,
                ROW_NUMBER() OVER (
                    ORDER BY value DESC
                ) AS rank
            FROM yearly_values
        )
        SELECT
            year,
            value
        FROM ranked
        WHERE rank = 1;
        """,
    ]

    unsafe_queries = [

        """
        DROP TABLE ecommerce_orders;
        """,

        """
        DELETE FROM ecommerce_orders
        WHERE region = 'Europe';
        """,

        """
        UPDATE ecommerce_orders
        SET sales = 0;
        """,

        """
        SELECT *
        FROM another_table;
        """,

        """
        SELECT *
        FROM ecommerce_orders;

        DROP TABLE ecommerce_orders;
        """,

        """
        WITH safe_data AS (
            SELECT *
            FROM another_table
        )
        SELECT *
        FROM safe_data;
        """,
    ]

    print("SAFE QUERIES")
    print("=" * 80)

    for index, query in enumerate(
        safe_queries,
        start=1,
    ):

        try:
            guard.validate(query)

            print(
                f"Safe query {index}: PASS"
            )

        except SQLGuardError as exc:

            print(
                f"Safe query {index}: FAIL"
            )

            print(
                f"  Reason: {exc}"
            )

    print("\nUNSAFE QUERIES")
    print("=" * 80)

    for index, query in enumerate(
        unsafe_queries,
        start=1,
    ):

        try:
            guard.validate(query)

            print(
                f"Unsafe query {index}: FAIL"
            )

        except SQLGuardError as exc:

            print(
                f"Unsafe query {index}: BLOCKED"
            )

            print(
                f"  Reason: {exc}"
            )

    print("\nSQL guard test completed.")