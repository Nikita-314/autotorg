"""
Лёгкая обёртка над SQLite для analytics.db (сигналы, решения, adaptive_actions).
"""
from __future__ import annotations

import json
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

# Схема должна совпадать с INSERT в signal_logger.py / adaptive engine.
_SCHEMA_STATEMENTS: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS signal_runs (
      run_id TEXT PRIMARY KEY,
      strategy_code TEXT NOT NULL,
      strategy_version TEXT NOT NULL,
      run_started_at TEXT NOT NULL,
      run_finished_at TEXT,
      universe_size INTEGER,
      tradable_universe_size INTEGER,
      instruments_checked INTEGER,
      signals_found INTEGER,
      run_status TEXT,
      comment TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signals (
      signal_id TEXT PRIMARY KEY,
      run_id TEXT,
      ticker TEXT NOT NULL,
      signal_ts TEXT NOT NULL,
      strategy_code TEXT,
      strategy_version TEXT,
      side TEXT,
      entry_price REAL,
      stop_price REAL,
      target_price REAL,
      confidence_score REAL,
      reason_code TEXT,
      reason_text TEXT,
      feature_snapshot_json TEXT,
      model_snapshot_json TEXT,
      market_regime TEXT,
      execution_mode TEXT,
      status TEXT,
      created_at TEXT,
      updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decision_logs (
      decision_id TEXT PRIMARY KEY,
      run_id TEXT,
      signal_id TEXT,
      ticker TEXT NOT NULL,
      decision_ts TEXT NOT NULL,
      decision_type TEXT NOT NULL,
      decision_label TEXT NOT NULL,
      reason_code TEXT,
      reason_text TEXT,
      details_json TEXT,
      created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_inference_logs (
      inference_id TEXT PRIMARY KEY,
      signal_id TEXT,
      run_id TEXT,
      ticker TEXT,
      model_type TEXT NOT NULL,
      model_version TEXT,
      inference_ts TEXT,
      input_features_json TEXT,
      raw_output_json TEXT,
      decision_label TEXT,
      confidence_score REAL,
      action_recommendation TEXT,
      model_used_in_final_decision INTEGER,
      created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS signal_outcomes (
      outcome_id TEXT PRIMARY KEY,
      signal_id TEXT NOT NULL UNIQUE,
      price_after_5m REAL,
      price_after_15m REAL,
      price_after_60m REAL,
      price_eod REAL,
      mfe REAL,
      mae REAL,
      outcome_5m_pct REAL,
      outcome_15m_pct REAL,
      outcome_60m_pct REAL,
      outcome_eod_pct REAL,
      evaluated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS paper_trade_links (
      link_id TEXT PRIMARY KEY,
      signal_id TEXT,
      local_trade_id TEXT,
      ticker TEXT NOT NULL,
      side TEXT,
      event_ts TEXT,
      qty REAL,
      price REAL,
      paper_position_id TEXT,
      comment TEXT
    )
    """,
    _ADAPTIVE_CREATE_SQL,
)


class AnalyticsDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.ensure_schema()

    @staticmethod
    def dumps_json(obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False)

    def ensure_schema(self) -> None:
        """Создаёт все таблицы, если их ещё нет (идемпотентно)."""
        for stmt in _SCHEMA_STATEMENTS:
            self.execute(stmt.strip())
        self.migrate_adaptive_actions_columns()

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
