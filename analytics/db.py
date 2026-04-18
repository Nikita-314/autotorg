"""
Лёгкая обёртка над SQLite для analytics.db (сигналы, решения, adaptive_actions).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple


_ADAPTIVE_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS adaptive_actions (
  action_id TEXT PRIMARY KEY,
  action_ts TEXT NOT NULL,
  mode TEXT NOT NULL,
  scope TEXT NOT NULL,
  bucket_key TEXT,
  parameter_name TEXT NOT NULL,
  old_value REAL,
  new_value REAL,
  confidence_score REAL,
  sample_size INTEGER,
  action_status TEXT NOT NULL,
  reason_text TEXT,
  metrics_json TEXT,
  window_summary_json TEXT,
  created_at TEXT NOT NULL,
  applied INTEGER DEFAULT 1,
  reverted INTEGER DEFAULT 0,
  evaluation_status TEXT DEFAULT 'n/a'
);
"""


class AnalyticsDB:
    def __init__(self, path: Path):
        self.path = Path(path)

    def ensure_adaptive_actions_table(self) -> None:
        self.execute(_ADAPTIVE_CREATE_SQL)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def cursor(self) -> Iterable[sqlite3.Cursor]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        finally:
            conn.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self.cursor() as cur:
            cur.execute(sql, params)

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None

    def migrate_adaptive_actions_columns(self) -> None:
        """Добавляет колонки журнала, если их ещё нет (идемпотентно)."""
        self.ensure_adaptive_actions_table()
        with self.cursor() as cur:
            cur.execute("PRAGMA table_info(adaptive_actions)")
            existing = {row[1] for row in cur.fetchall()}
        alters = []
        if "applied" not in existing:
            alters.append("ALTER TABLE adaptive_actions ADD COLUMN applied INTEGER DEFAULT 1")
        if "reverted" not in existing:
            alters.append("ALTER TABLE adaptive_actions ADD COLUMN reverted INTEGER DEFAULT 0")
        if "evaluation_status" not in existing:
            alters.append(
                "ALTER TABLE adaptive_actions ADD COLUMN evaluation_status TEXT DEFAULT 'n/a'"
            )
        for stmt in alters:
            self.execute(stmt)
