import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd

from .db import AnalyticsDB
from .signal_logger import SignalLogger

LOGGER = logging.getLogger(__name__)

# Outcome columns that must all be NULL to retry evaluation (matches signal_outcomes schema).
_OUTCOME_VALUE_COLS = (
    "price_after_5m",
    "price_after_15m",
    "price_after_60m",
    "price_eod",
    "mfe",
    "mae",
    "outcome_5m_pct",
    "outcome_15m_pct",
    "outcome_60m_pct",
    "outcome_eod_pct",
)


class OutcomeEvaluator:
    """
    Evaluates signal outcomes on available bars.
    Uses existing project data source via provided candle loader callback.
    """

    def __init__(self, db: AnalyticsDB, logger: SignalLogger):
        self.db = db
        self.logger = logger

    @staticmethod
    def _parse_ts(ts: str) -> Optional[datetime]:
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None

    @staticmethod
    def _normalize_df_index(df: Any) -> Any:
        """Align candle index with naive UTC signal time: compare naive-to-naive only."""
        if df is None or df.empty:
            return df
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return df
        if idx.tz is None:
            return df
        out = df.copy()
        try:
            out.index = idx.tz_convert("UTC").tz_localize(None)
        except Exception:
            out.index = pd.DatetimeIndex(
                [
                    pd.Timestamp(t).tz_convert("UTC").tz_localize(None)
                    for t in idx
                ]
            )
        return out

    @staticmethod
    def _first_price_after(df, ts: datetime, minutes: int) -> Optional[float]:
        if df is None or df.empty:
            return None
        target = ts + timedelta(minutes=minutes)
        sub = df[df.index >= target]
        if sub.empty:
            return None
        try:
            return float(sub["close"].iloc[0])
        except Exception:
            return None

    @staticmethod
    def _eod_price(df, ts: datetime) -> Optional[float]:
        if df is None or df.empty:
            return None
        same_day = df[(df.index.date == ts.date()) & (df.index >= ts)]
        if same_day.empty:
            return None
        try:
            return float(same_day["close"].iloc[-1])
        except Exception:
            return None

    @staticmethod
    def _mfe_mae(df, ts: datetime, entry: float) -> tuple:
        if df is None or df.empty or entry <= 0:
            return None, None
        same_day = df[(df.index.date == ts.date()) & (df.index >= ts)]
        if same_day.empty:
            return None, None
        try:
            max_p = float(same_day["high"].max()) if "high" in same_day.columns else float(same_day["close"].max())
            min_p = float(same_day["low"].min()) if "low" in same_day.columns else float(same_day["close"].min())
            mfe = (max_p / entry - 1.0) * 100.0
            mae = (min_p / entry - 1.0) * 100.0
            return mfe, mae
        except Exception:
            return None, None

    @staticmethod
    def _pct(base: Optional[float], val: Optional[float]) -> Optional[float]:
        if base is None or val is None or base == 0:
            return None
        return (val / base - 1.0) * 100.0

    def evaluate_signal(self, signal: Dict[str, Any], load_candles_fn) -> Optional[Dict[str, Any]]:
        signal_ts = self._parse_ts(str(signal.get("signal_ts", "")))
        if signal_ts is None:
            return None
        ticker = str(signal.get("ticker", "")).upper()
        entry_price = signal.get("entry_price")
        try:
            entry_price = float(entry_price) if entry_price is not None else None
        except Exception:
            entry_price = None
        if not ticker or entry_price is None:
            return None

        df = load_candles_fn(ticker)
        if df is None or df.empty:
            return None
        df = self._normalize_df_index(df)

        p5 = self._first_price_after(df, signal_ts, 5)
        p15 = self._first_price_after(df, signal_ts, 15)
        p60 = self._first_price_after(df, signal_ts, 60)
        peod = self._eod_price(df, signal_ts)
        mfe, mae = self._mfe_mae(df, signal_ts, entry_price)

        return {
            "price_after_5m": p5,
            "price_after_15m": p15,
            "price_after_60m": p60,
            "price_eod": peod,
            "mfe": mfe,
            "mae": mae,
            "outcome_5m_pct": self._pct(entry_price, p5),
            "outcome_15m_pct": self._pct(entry_price, p15),
            "outcome_60m_pct": self._pct(entry_price, p60),
            "outcome_eod_pct": self._pct(entry_price, peod),
        }

    def evaluate_pending(
        self,
        load_candles_fn,
        limit: int = 300,
        reeval_null_older_than_minutes: int = 30,
    ) -> int:
        """
        Pending = no outcome row yet, OR row exists but all outcome metrics are still NULL
        (e.g. first run had no bars after signal_ts; after MOEX pagination data may exist).

        Re-evaluating rows that stay NULL is throttled by evaluated_at so we do not
        hammer the DB/API every bot tick; use reeval_null_older_than_minutes (default 30).
        """
        null_checks = " AND ".join(f"o.{c} IS NULL" for c in _OUTCOME_VALUE_COLS)
        throttle_mod = f"-{max(1, int(reeval_null_older_than_minutes))} minutes"
        rows = self.db.fetchall(
            f"""
            SELECT s.signal_id, s.ticker, s.signal_ts, s.entry_price
            FROM signals s
            LEFT JOIN signal_outcomes o ON o.signal_id = s.signal_id
            WHERE s.entry_price IS NOT NULL
              AND (
                s.side IN ('BUY','SELL')
                OR (s.side IN ('HOLD','BLOCK') AND s.reason_code IN ('BLOCK_ML','BLOCK_TREND'))
              )
              AND (
                o.signal_id IS NULL
                OR (
                  {null_checks}
                  AND (
                    o.evaluated_at IS NULL
                    OR datetime(replace(substr(o.evaluated_at, 1, 19), 'T', ' '))
                       < datetime('now', ?)
                  )
                )
              )
            ORDER BY s.signal_ts DESC
            LIMIT ?
            """,
            (throttle_mod, int(limit)),
        )
        updated = 0
        for row in rows:
            try:
                outcome = self.evaluate_signal(row, load_candles_fn)
                if outcome is None:
                    continue
                self.logger.upsert_signal_outcome(str(row["signal_id"]), outcome)
                updated += 1
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "Outcome evaluate failed signal_id=%s ticker=%s err=%s",
                    row.get("signal_id"),
                    row.get("ticker"),
                    exc,
                )
        return updated

