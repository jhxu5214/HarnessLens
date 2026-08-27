from __future__ import annotations

import sqlite3
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_RESULT_ROWS = 100_000


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool
    elapsed_s: float


def execute_readonly_sql(
    database: str | Path,
    sql: str,
    *,
    timeout_s: float,
    max_rows: int = MAX_RESULT_ROWS,
    deadline: float | None = None,
) -> QueryResult:
    statement = _validate_statement(sql)
    path = Path(database).resolve()
    if not path.is_file():
        raise ValueError(f"BIRD database is unavailable: {path}")
    started = time.monotonic()
    effective_deadline = deadline or started + float(timeout_s)
    if effective_deadline <= started:
        raise TimeoutError("SQL execution timed out")
    connection = sqlite3.connect(
        f"file:{path}?mode=ro", uri=True, timeout=min(5.0, float(timeout_s))
    )
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_readonly_authorizer)

        def interrupt_expired_query() -> int:
            return int(time.monotonic() >= effective_deadline)

        connection.set_progress_handler(interrupt_expired_query, 10_000)
        try:
            cursor = connection.execute(statement)
            columns = tuple(item[0] for item in (cursor.description or ()))
            rows = cursor.fetchmany(int(max_rows) + 1)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower() and time.monotonic() >= effective_deadline:
                raise TimeoutError("SQL execution timed out") from exc
            raise
        truncated = len(rows) > int(max_rows)
        return QueryResult(
            columns=columns,
            rows=tuple(tuple(item) for item in rows[: int(max_rows)]),
            truncated=truncated,
            elapsed_s=time.monotonic() - started,
        )
    finally:
        connection.close()


def grade_execution_accuracy(
    database: str | Path,
    predicted_sql: str,
    gold_sql: str,
    *,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + float(timeout_s)
    try:
        predicted = execute_readonly_sql(
            database,
            predicted_sql,
            timeout_s=timeout_s,
            deadline=deadline,
        )
        gold = execute_readonly_sql(
            database,
            gold_sql,
            timeout_s=timeout_s,
            deadline=deadline,
        )
        if predicted.truncated or gold.truncated:
            raise RuntimeError(
                f"query returned more than the {MAX_RESULT_ROWS}-row safety limit"
            )
        passed = set(predicted.rows) == set(gold.rows)
        diagnostic = _sanitized_execution_diagnostic(predicted, gold, passed=passed)
        return {
            "passed": bool(passed),
            "error": "",
            "predicted_row_count": len(predicted.rows),
            "gold_row_count": len(gold.rows),
            "diagnostic": diagnostic,
            "elapsed_s": round(time.monotonic() - started, 4),
        }
    except Exception as exc:  # benchmark SQL failures are charged failures
        return {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}"[:2000],
            "predicted_row_count": None,
            "gold_row_count": None,
            "diagnostic": {
                "mismatch_type": "query_execution_error",
                "predicted_row_count": None,
                "reference_row_count": None,
                "predicted_column_count": None,
                "reference_column_count": None,
                "predicted_unique_row_count": None,
                "reference_unique_row_count": None,
                "duplicate_profile_mismatch": False,
            },
            "elapsed_s": round(time.monotonic() - started, 4),
        }


def _sanitized_execution_diagnostic(
    predicted: QueryResult,
    reference: QueryResult,
    *,
    passed: bool,
) -> dict[str, Any]:
    predicted_unique = len(set(predicted.rows))
    reference_unique = len(set(reference.rows))
    predicted_columns = len(predicted.columns)
    reference_columns = len(reference.columns)
    if passed:
        mismatch_type = "match"
    elif predicted_columns != reference_columns:
        mismatch_type = "column_count_mismatch"
    elif len(predicted.rows) != len(reference.rows):
        mismatch_type = "row_count_mismatch"
    elif predicted_unique != reference_unique:
        mismatch_type = "duplicate_profile_mismatch"
    else:
        mismatch_type = "value_or_type_mismatch"
    return {
        "mismatch_type": mismatch_type,
        "predicted_row_count": len(predicted.rows),
        "reference_row_count": len(reference.rows),
        "predicted_column_count": predicted_columns,
        "reference_column_count": reference_columns,
        "predicted_unique_row_count": predicted_unique,
        "reference_unique_row_count": reference_unique,
        "duplicate_profile_mismatch": (
            len(predicted.rows) - predicted_unique
            != len(reference.rows) - reference_unique
        ),
    }


def _validate_statement(sql: str) -> str:
    statement = str(sql or "").strip()
    if not statement:
        raise ValueError("SQL is empty")
    inspected = re.sub(
        r"\A\s*(?:(?:--[^\n]*(?:\n|\Z))|(?:/\*.*?\*/\s*))*",
        "",
        statement,
        flags=re.DOTALL,
    )
    first = inspected.lstrip(" \t\r\n(").split(None, 1)[0].lower()
    if first not in {"select", "with"}:
        raise ValueError("only SELECT or WITH queries are allowed")
    return statement


_DENIED_ACTIONS = {
    value
    for name in (
        "SQLITE_ALTER_TABLE",
        "SQLITE_ATTACH",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_DELETE",
        "SQLITE_DETACH",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_INSERT",
        "SQLITE_PRAGMA",
        "SQLITE_REINDEX",
        "SQLITE_TRANSACTION",
        "SQLITE_UPDATE",
    )
    if (value := getattr(sqlite3, name, None)) is not None
}


def _readonly_authorizer(
    action: int,
    _argument1: str | None,
    _argument2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    return sqlite3.SQLITE_DENY if action in _DENIED_ACTIONS else sqlite3.SQLITE_OK
