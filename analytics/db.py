import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


def safe_to_json(value: Any) -> Any:
    """Convert numpy/pandas/datetime objects to JSON-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [safe_to_json(v) for v in value.tolist()]

    if pd is not None:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, pd.Series):
            return {str(k): safe_to_json(v) for k, v in value.to_dict().items()}
        if isinstance(value, pd.DataFrame):
            return [safe_to_json(v) for v in value.to_dict(orient="records")]

    if isinstance(value, dict):
        return {str(k): safe_to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_to_json(v) for v in value]

    return str(value)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_runs (
  run_id TEXT PRIMARY KEY,
  strategy_code TEXT NOT NULL,
  strategy_version TEXT,
  run_started_at TEXT NOT NULL,
  run_finished_at TEXT,
  universe_size INTEGER,
  tradable_universe_size INTEGER,
  instruments_checked INTEGER,
  signals_found INTEGER,
  run_status TEXT,
  comment TEXT
);

CREATE TABLE IF NOT EXISTS signals (
  signal_id TEXT PRIMARY KEY,
  run_id TEXT,
  ticker TEXT NOT NULL,
  signal_ts TEXT NOT NULL,
  strategy_code TEXT NOT NULL,
  strategy_version TEXT,
  side TEXT NOT NULL, -- BUY/SELL/HOLD/BLOCK (HOLD/BLOCK are diagnostic decisions, not only executed trades)
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
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES signal_runs(run_id)
);

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
  evaluated_at TEXT NOT NULL,
  FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
);

CREATE TABLE IF NOT EXISTS signal_reviews (
  review_id TEXT PRIMARY KEY,
  signal_id TEXT NOT NULL,
  review_ts TEXT NOT NULL,
  is_good_signal INTEGER,
  is_execution_problem INTEGER,
  is_market_regime_problem INTEGER,
  is_false_breakout INTEGER,
  is_low_liquidity_problem INTEGER,
  is_volume_anomaly INTEGER,
  is_time_of_day_problem INTEGER,
  is_model_conflict INTEGER,
  root_cause TEXT,
  review_comment TEXT,
  next_action TEXT,
  reviewed_by TEXT,
  FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
);

CREATE TABLE IF NOT EXISTS model_inference_logs (
  inference_id TEXT PRIMARY KEY,
  signal_id TEXT,
  run_id TEXT,
  ticker TEXT,
  model_type TEXT NOT NULL,
  model_version TEXT,
  inference_ts TEXT NOT NULL,
  input_features_json TEXT,
  raw_output_json TEXT,
  decision_label TEXT,
  confidence_score REAL,
  action_recommendation TEXT,
  model_used_in_final_decision INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY (signal_id) REFERENCES signals(signal_id),
  FOREIGN KEY (run_id) REFERENCES signal_runs(run_id)
);

CREATE TABLE IF NOT EXISTS decision_logs (
  decision_id TEXT PRIMARY KEY,
  run_id TEXT,
  signal_id TEXT,
  ticker TEXT NOT NULL,
  decision_ts TEXT NOT NULL,
  decision_type TEXT NOT NULL, -- STRATEGY_DECISION / TRADE_OPEN / TRADE_CLOSE
  decision_label TEXT,          -- BUY/SELL/HOLD/BLOCK
  reason_code TEXT,
  reason_text TEXT,
  details_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES signal_runs(run_id),
  FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
);

CREATE TABLE IF NOT EXISTS paper_trade_links (
  link_id TEXT PRIMARY KEY,
  signal_id TEXT,
  local_trade_id TEXT,
  ticker TEXT NOT NULL,
  side TEXT NOT NULL,
  event_ts TEXT NOT NULL,
  qty REAL,
  price REAL,
  paper_position_id TEXT,
  comment TEXT,
  FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
);

CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(signal_ts);
CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_signals_run ON signals(run_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_signal ON signal_outcomes(signal_id);
CREATE INDEX IF NOT EXISTS idx_decisions_ticker_ts ON decision_logs(ticker, decision_ts);
CREATE INDEX IF NOT EXISTS idx_inference_run ON model_inference_logs(run_id);
"""


class AnalyticsDB:
    """SQLite storage adapter for analytics layer (easy to replace later)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        self._run_migrations()
        self.conn.commit()

    def _has_column(self, table: str, column: str) -> bool:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(r[1] == column for r in rows)

    def _run_migrations(self) -> None:
        # decision_logs.created_at for DB write timestamp
        if not self._has_column("decision_logs", "created_at"):
            self.conn.execute("ALTER TABLE decision_logs ADD COLUMN created_at TEXT")
            self.conn.execute("UPDATE decision_logs SET created_at = decision_ts WHERE created_at IS NULL")

        # paper_trade_links.local_trade_id for robust OPEN/CLOSE linking
        if not self._has_column("paper_trade_links", "local_trade_id"):
            self.conn.execute("ALTER TABLE paper_trade_links ADD COLUMN local_trade_id TEXT")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_trade_local_id ON paper_trade_links(local_trade_id)")

        # signal_outcomes must be 1 row per signal_id
        self.conn.execute(
            """
            DELETE FROM signal_outcomes
            WHERE rowid NOT IN (
                SELECT MAX(rowid)
                FROM signal_outcomes
                GROUP BY signal_id
            )
            """
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_signal_outcomes_signal_id ON signal_outcomes(signal_id)"
        )

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> None:
        self.conn.execute(sql, params)
        self.conn.commit()

    def executemany(self, sql: str, params: Iterable[Tuple[Any, ...]]) -> None:
        self.conn.executemany(sql, params)
        self.conn.commit()

    def fetchall(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def fetchone(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    @staticmethod
    def dumps_json(payload: Any) -> str:
        return json.dumps(safe_to_json(payload), ensure_ascii=False, separators=(",", ":"))

