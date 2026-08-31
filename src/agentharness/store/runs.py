"""SQLite persistence for runs and their steps. stdlib sqlite3, no ORM.

Pure storage. This module knows about columns and rows; it knows nothing about
outcomes, policies or the loop. Everything it stores is already primitive by the
time it arrives, which is what lets a trace be read back and rendered without
importing the harness that produced it.
"""

import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

STEP_COLUMNS = (
    "run_id",
    "sequence",
    "iteration",
    "kind",
    "duration_ms",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "attempts",
    "tool_call_id",
    "tool_name",
    "arguments_json",
    "outcome_class",
    "is_write",
    "error_detail",
)


class RunStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        # check_same_thread=False because the API's test client and any ASGI
        # server hand requests to worker threads. Runs are short and writes are
        # serialized by SQLite's own locking.
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    def close(self) -> None:
        self._connection.close()

    def start_run(self, run_id: str, goal: str, model: str, started_at: float) -> None:
        self._connection.execute(
            "INSERT INTO runs (run_id, goal, model, started_at) VALUES (?, ?, ?, ?)",
            (run_id, goal, model, started_at),
        )
        self._connection.commit()

    def finish_run(self, run_id: str, **fields: Any) -> None:
        assignments = ", ".join(f"{column} = ?" for column in fields)
        self._connection.execute(
            f"UPDATE runs SET {assignments} WHERE run_id = ?",
            (*fields.values(), run_id),
        )
        self._connection.commit()

    def add_step(self, **fields: Any) -> None:
        columns = [column for column in STEP_COLUMNS if column in fields]
        placeholders = ", ".join("?" for _ in columns)
        self._connection.execute(
            f"INSERT INTO steps ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(fields[column] for column in columns),
        )
        self._connection.commit()

    def next_sequence(self, run_id: str) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next FROM steps WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return int(row["next"])

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def get_steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM steps WHERE run_id = ? ORDER BY sequence", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def column_names(self, table: str) -> list[str]:
        """Used by the schema test that asserts nothing here can hold a transcript."""
        rows = self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        return [row["name"] for row in rows]
