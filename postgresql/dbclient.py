"""
DatabaseClient (psycopg v3) with:
- Connection pooling
- Safe SELECT-only guard
- Query timeout (server-side statement_timeout)
- LIMIT injection (wraps query as subquery unless it already has a LIMIT)
- Query logging (duration + rowcount + truncated SQL)

Requires:
  pip install "psycopg[binary,pool]"
"""

from __future__ import annotations

import base64
import datetime as dt
import logging
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool

_ = load_dotenv("env.example")


def _to_json_safe_scalar(x: Any) -> Any:
    # Missing
    if x is None:
        return None
    # pandas missing values (pd.NA, NaT) & numpy nan
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # Decimal (Postgres numeric)
    if isinstance(x, Decimal):
        return float(x)

    # UUID
    if isinstance(x, UUID):
        return str(x)

    # datetime/date/time
    if isinstance(x, (dt.datetime, pd.Timestamp)):
        ts = pd.Timestamp(x)
        # make timezone-safe: convert to UTC and drop tz (Altair often prefers naive)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime()

    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return x.isoformat()

    if isinstance(x, dt.time):
        return x.isoformat()

    # timedelta
    if isinstance(x, (dt.timedelta, pd.Timedelta, np.timedelta64)):
        return pd.Timedelta(x).total_seconds()

    # bytes-like
    if isinstance(x, (bytes, bytearray, memoryview)):
        b = bytes(x)
        # keep it compact and JSON-safe
        return base64.b64encode(b).decode("ascii")

    # numpy scalar types
    if isinstance(x, np.generic):
        return x.item()

    return x


def make_df_json_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with problematic types converted to JSON/Altair-friendly types.
    """
    df2 = df.copy()

    # Handle tz-aware datetime64 columns efficiently
    for c in df2.columns:
        col = df2[c]
        if pd.api.types.is_datetime64tz_dtype(col.dtype):
            # convert to UTC naive
            df2[c] = pd.to_datetime(col, utc=True).dt.tz_localize(None)

    # Apply scalar conversion column-wise (robust)
    for c in df2.columns:
        df2[c] = df2[c].map(_to_json_safe_scalar)

    return df2


@dataclass(frozen=True)
class QueryResult:
    columns: List[str]
    rows: List[Tuple[Any, ...]]
    duration_ms: float
    limited: bool


class DatabaseClient:
    """
    Safe-ish query runner for analytics/LLM use.

    Notes:
    - This intentionally only allows SELECT queries (no CTE DML, no multi-statements).
    - LIMIT injection is done by wrapping into: SELECT * FROM (<sql>) AS subquery LIMIT <n>
      unless a top-level LIMIT is detected.
    - Timeout is enforced by setting `SET LOCAL statement_timeout` per query.
    """

    _FORBIDDEN_KEYWORDS = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
        "vacuum",
        "analyze",
        "reindex",
        "copy",
        "call",
        "do",
        "execute",
        "prepare",
        "deallocate",
    )

    def __init__(
        self,
        *,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
        pool_timeout_s: int = 10,
        statement_timeout_ms: int = 5000,
        default_limit: int = 1000,
        log_sql_max_chars: int = 500,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.conninfo = self._build_conninfo()

        self.pool = ConnectionPool(
            conninfo=self.conninfo,
            min_size=min_pool_size,
            max_size=max_pool_size,
            timeout=pool_timeout_s,
        )

        self.statement_timeout_ms = int(statement_timeout_ms)
        self.default_limit = int(default_limit)
        self.log_sql_max_chars = int(log_sql_max_chars)

        self.logger = logger or logging.getLogger("db")
        if not self.logger.handlers:
            # sensible default logging if user hasn't configured logging yet
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.CRITICAL)

    # -------------------------
    # Connection info
    # -------------------------
    def _build_conninfo(self) -> str:
        host = os.environ.get("DB_HOST")
        port = os.environ.get("DB_PORT")
        user = os.environ.get("DB_USER")
        password = os.environ.get("DB_PASSWORD")
        db = os.environ.get("DB_DATABASE")

        missing = [
            k
            for k, v in {
                "DB_HOST": host,
                "DB_PORT": port,
                "DB_USER": user,
                "DB_PASSWORD": password,
                "DB_DATABASE": db,
            }.items()
            if not v
        ]
        if missing:
            raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

        return f"user={user} password={password} host={host} port={port} dbname={db}"

    # -------------------------
    # Public API
    # -------------------------
    def get_schema_summary(self) -> str:
        sql = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
        """
        res = self.query(sql, limit=None)
        schema: Dict[str, List[str]] = {}
        for table, column, dtype in res.rows:
            schema.setdefault(table, []).append(f"{column} ({dtype})")

        lines = ["Database structure:"]
        for table, cols in schema.items():
            lines.append(f"- {table}: {', '.join(cols)}")
        return "\n".join(lines)

    def query(
        self,
        sql: str,
        *,
        limit: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        log: bool = True,
    ) -> QueryResult:
        """
        Execute a SELECT query safely with optional LIMIT injection and timeout.

        Args:
            sql: SQL text (must be SELECT; single statement).
            limit: If provided, injects LIMIT unless query already has a top-level LIMIT.
                   If None: uses default_limit.
                   If 0: no LIMIT injection.
            timeout_ms: If provided, overrides default statement_timeout_ms.
            log: If True, logs query duration and row count.

        Returns:
            QueryResult(columns, rows, duration_ms, limited)
        """
        cleaned = self._normalize_sql(sql)

        if not self.is_safe_sql(cleaned):
            raise ValueError("Unsafe SQL query detected (only SELECT allowed).")

        effective_limit: Optional[int]
        if limit == 0:
            effective_limit = None
        elif limit is None:
            effective_limit = self.default_limit
        else:
            effective_limit = int(limit)

        final_sql, limited = self._inject_limit(cleaned, effective_limit)

        effective_timeout_ms = (
            int(timeout_ms) if timeout_ms is not None else self.statement_timeout_ms
        )

        t0 = time.perf_counter()
        rows: List[Tuple[Any, ...]]
        cols: List[str]

        with self.pool.connection() as conn:
            conn.row_factory = tuple_row
            with conn.cursor() as cur:
                # server-side timeout for this transaction only
                cur.execute(f"SET LOCAL statement_timeout = {int(effective_timeout_ms)};")
                cur.execute(final_sql)
                rows = cur.fetchall()
                cols = [d.name for d in cur.description] if cur.description else []

        duration_ms = (time.perf_counter() - t0) * 1000.0

        if log:
            self._log_query(
                final_sql, duration_ms=duration_ms, rowcount=len(rows), limited=limited
            )

        return QueryResult(columns=cols, rows=rows, duration_ms=duration_ms, limited=limited)

    def close(self) -> None:
        """Close pool connections."""
        self.pool.close()

    # -------------------------
    # Safety
    # -------------------------
    def is_safe_sql(self, sql: str) -> bool:
        """
        Conservative safety check:
        - single statement only (no semicolons inside)
        - must start with SELECT (allows WITH ... SELECT)
        - forbid common DDL/DML keywords anywhere
        """
        s = sql.strip().lower()

        # disallow multi statements
        if ";" in s.rstrip(";"):
            return False

        # allow WITH ... SELECT ... (CTE) and plain SELECT
        if not (s.startswith("select") or s.startswith("with")):
            return False

        # If starts with WITH, ensure it eventually contains SELECT
        if s.startswith("with") and "select" not in s:
            return False

        # forbid dangerous keywords
        return not any(k in s for k in self._FORBIDDEN_KEYWORDS)

    # -------------------------
    # LIMIT injection
    # -------------------------
    def _inject_limit(self, sql: str, limit: Optional[int]) -> Tuple[str, bool]:
        """
        If limit is None -> no injection.
        If SQL already has a top-level LIMIT -> no injection.
        Else wrap as subquery and apply LIMIT.
        """
        if limit is None:
            return sql, False

        if self._has_top_level_limit(sql):
            return sql, False

        wrapped = f"SELECT * FROM (\n{sql}\n) AS subquery\nLIMIT {int(limit)}"
        return wrapped, True

    def _has_top_level_limit(self, sql: str) -> bool:
        """
        Best-effort detection of a top-level LIMIT.
        This is not a full SQL parser; it tries to ignore 'limit' inside parentheses/strings.
        """
        s = sql.strip()

        depth = 0
        in_single = False
        in_double = False
        in_dollar: Optional[str] = None  # dollar-quoted tag like $$ or $tag$
        i = 0
        while i < len(s):
            ch = s[i]

            # dollar-quoted strings (rare for SELECT, but handle)
            if not in_single and not in_double:
                if in_dollar:
                    if s.startswith(in_dollar, i):
                        in_dollar = None
                        i += len(in_dollar or "")
                        continue
                else:
                    m = re.match(r"\$[A-Za-z_0-9]*\$", s[i:])
                    if m:
                        in_dollar = m.group(0)
                        i += len(in_dollar)
                        continue

            if in_dollar:
                i += 1
                continue

            # quoted strings/idents
            if ch == "'" and not in_double:
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue

            if in_single or in_double:
                i += 1
                continue

            if ch == "(":
                depth += 1
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                i += 1
                continue

            # only look for LIMIT at top-level
            if depth == 0:
                # match whole word 'limit'
                if re.match(r"(?i)\blimit\b", s[i:]):
                    return True

            i += 1

        return False

    # -------------------------
    # Logging / utils
    # -------------------------
    def _log_query(self, sql: str, *, duration_ms: float, rowcount: int, limited: bool) -> None:
        compact = re.sub(r"\s+", " ", sql).strip()
        if len(compact) > self.log_sql_max_chars:
            compact = compact[: self.log_sql_max_chars] + "…"
        self.logger.info(
            "SQL (%s) %d rows in %.1f ms | %s",
            "LIMITED" if limited else "NO_LIMIT",
            rowcount,
            duration_ms,
            compact,
        )

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        s = sql.strip()
        # remove trailing semicolons (common in user-entered SQL)
        s = re.sub(r";+\s*$", "", s)
        return s

    # -------------------------
    # Presentation helper
    # -------------------------
    @staticmethod
    def to_markdown_table(result: QueryResult, max_rows: int = 10) -> str:
        if not result.rows:
            return "No results found."

        cols = result.columns
        header = "| " + " | ".join(cols) + " |"
        separator = "| " + " | ".join(["---"] * len(cols)) + " |"

        def fmt(v: Any) -> str:
            if v is None:
                return "NULL"
            return str(v)

        body = "\n".join(
            "| " + " | ".join(fmt(v) for v in row) + " |"
            for row in result.rows[:max_rows]
        )
        if len(result.rows) > max_rows:
            body += f"\n\n… showing first {max_rows} rows (query returned {len(result.rows)})."

        return "\n".join([header, separator, body])

    # -------------------------
    # LLM response helper
    # -------------------------
    @staticmethod
    def extract_sql(text: str) -> Optional[str]:
        """
        Extract a SQL SELECT query from a text message.
        Supports code blocks (```sql ...```), inline backticks, or plain SELECT statements.
        Returns None if no SELECT is found.
        """
        pattern = r"""
            ```(?:sql)?\s*(SELECT[\s\S]+?)\s*```   # code block ```sql ... ```
            |`\s*(SELECT[\s\S]+?)\s*`              # inline `SELECT ...`
            |(SELECT[\s\S]+?;)                     # SELECT ending with semicolon
            |(SELECT[\s\S]+)$                      # SELECT till end of text
        """
        m = re.search(pattern, text, flags=re.IGNORECASE | re.VERBOSE)
        if not m:
            return None

        # return the first non-None match group
        for i in range(1, len(m.groups()) + 1):
            if m.group(i):
                return m.group(i).strip()
        return None

    # -------------------------
    # Pandas helper
    # -------------------------
    @staticmethod
    def to_dataframe(result: QueryResult):
        """
        Convert a QueryResult into a pandas DataFrame.
        Requires pandas to be installed.
        """
        try:
            import pandas as pd
        except Exception as e:
            raise ImportError("pandas is required to convert results to a DataFrame") from e

        df = pd.DataFrame(result.rows, columns=result.columns)
        df = make_df_json_safe(df)
        return df

    @staticmethod
    def describe_dataframe_for_llm(df) -> str:
        """
        Describe a pandas DataFrame for an LLM: row count and columns with dtypes.
        """
        try:
            import pandas as pd  # noqa: F401
        except Exception as e:
            raise ImportError("pandas is required to describe a DataFrame") from e

        row_count = len(df)
        col_parts = [f"{name} ({dtype})" for name, dtype in df.dtypes.items()]
        cols = ", ".join(col_parts) if col_parts else "none"
        if row_count == 0:
            return f"No data available."

        preview = df.head(5).to_dict(orient="records")
        return (
            "Result available in DataFrame df. It has "
            f"{row_count} rows and {df.shape[1]} cols. "
            f"Columns with types: {cols}. First 5 rows: {preview}."
        )
