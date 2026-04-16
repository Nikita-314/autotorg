import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .db import AnalyticsDB


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SignalLogger:
    def __init__(self, db: AnalyticsDB):
        self.db = db

    def start_run(
        self,
        strategy_code: str,
        strategy_version: str,
        universe_size: int,
        tradable_universe_size: int,
        comment: str = "",
    ) -> str:
        run_id = str(uuid.uuid4())
        now = _now_iso()
        self.db.execute(
            """
            INSERT INTO signal_runs (
              run_id, strategy_code, strategy_version, run_started_at,
              universe_size, tradable_universe_size, instruments_checked, signals_found, run_status, comment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                strategy_code,
                strategy_version,
                now,
                int(universe_size),
                int(tradable_universe_size),
                0,
                0,
                "RUNNING",
                comment,
            ),
        )
        return run_id

    def finish_run(
        self,
        run_id: str,
        instruments_checked: int,
        signals_found: int,
        run_status: str = "OK",
        comment: str = "",
    ) -> None:
        self.db.execute(
            """
            UPDATE signal_runs
            SET run_finished_at=?, instruments_checked=?, signals_found=?, run_status=?, comment=?
            WHERE run_id=?
            """,
            (_now_iso(), int(instruments_checked), int(signals_found), run_status, comment, run_id),
        )

    def log_signal(
        self,
        run_id: Optional[str],
        ticker: str,
        strategy_code: str,
        strategy_version: str,
        side: str,
        entry_price: Optional[float],
        stop_price: Optional[float],
        target_price: Optional[float],
        confidence_score: Optional[float],
        reason_code: Optional[str],
        reason_text: Optional[str],
        feature_snapshot: Dict[str, Any],
        model_snapshot: Dict[str, Any],
        market_regime: Optional[str],
        execution_mode: str,
        status: str,
        signal_ts: Optional[str] = None,
    ) -> str:
        signal_id = str(uuid.uuid4())
        now = _now_iso()
        signal_ts = signal_ts or now
        self.db.execute(
            """
            INSERT INTO signals (
              signal_id, run_id, ticker, signal_ts, strategy_code, strategy_version, side,
              entry_price, stop_price, target_price, confidence_score,
              reason_code, reason_text, feature_snapshot_json, model_snapshot_json, market_regime,
              execution_mode, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal_id,
                run_id,
                ticker,
                signal_ts,
                strategy_code,
                strategy_version,
                side,
                entry_price,
                stop_price,
                target_price,
                confidence_score,
                reason_code,
                reason_text,
                self.db.dumps_json(feature_snapshot),
                self.db.dumps_json(model_snapshot),
                market_regime,
                execution_mode,
                status,
                now,
                now,
            ),
        )
        return signal_id

    def update_signal_status(self, signal_id: Optional[str], status: str) -> None:
        if not signal_id:
            return
        self.db.execute(
            "UPDATE signals SET status=?, updated_at=? WHERE signal_id=?",
            (status, _now_iso(), signal_id),
        )

    def log_decision(
        self,
        run_id: Optional[str],
        signal_id: Optional[str],
        ticker: str,
        decision_type: str,
        decision_label: str,
        reason_code: Optional[str],
        reason_text: Optional[str],
        details: Dict[str, Any],
        decision_ts: Optional[str] = None,
    ) -> str:
        decision_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO decision_logs (
              decision_id, run_id, signal_id, ticker, decision_ts, decision_type,
              decision_label, reason_code, reason_text, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                run_id,
                signal_id,
                ticker,
                decision_ts or _now_iso(),
                decision_type,
                decision_label,
                reason_code,
                reason_text,
                self.db.dumps_json(details),
                _now_iso(),
            ),
        )
        return decision_id

    def log_model_inference(
        self,
        run_id: Optional[str],
        signal_id: Optional[str],
        ticker: Optional[str],
        model_type: str,
        model_version: Optional[str],
        input_features: Dict[str, Any],
        raw_output: Dict[str, Any],
        decision_label: Optional[str],
        confidence_score: Optional[float],
        action_recommendation: Optional[str],
        model_used_in_final_decision: bool,
        inference_ts: Optional[str] = None,
    ) -> str:
        inference_id = str(uuid.uuid4())
        now = _now_iso()
        self.db.execute(
            """
            INSERT INTO model_inference_logs (
              inference_id, signal_id, run_id, ticker, model_type, model_version, inference_ts,
              input_features_json, raw_output_json, decision_label, confidence_score,
              action_recommendation, model_used_in_final_decision, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inference_id,
                signal_id,
                run_id,
                ticker,
                model_type,
                model_version,
                inference_ts or now,
                self.db.dumps_json(input_features),
                self.db.dumps_json(raw_output),
                decision_label,
                confidence_score,
                action_recommendation,
                int(bool(model_used_in_final_decision)),
                now,
            ),
        )
        return inference_id

    def upsert_signal_outcome(self, signal_id: str, outcome: Dict[str, Any]) -> str:
        existing = self.db.fetchone("SELECT outcome_id FROM signal_outcomes WHERE signal_id=?", (signal_id,))
        outcome_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO signal_outcomes (
              outcome_id, signal_id, price_after_5m, price_after_15m, price_after_60m, price_eod,
              mfe, mae, outcome_5m_pct, outcome_15m_pct, outcome_60m_pct, outcome_eod_pct, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(signal_id) DO UPDATE SET
              price_after_5m=excluded.price_after_5m,
              price_after_15m=excluded.price_after_15m,
              price_after_60m=excluded.price_after_60m,
              price_eod=excluded.price_eod,
              mfe=excluded.mfe,
              mae=excluded.mae,
              outcome_5m_pct=excluded.outcome_5m_pct,
              outcome_15m_pct=excluded.outcome_15m_pct,
              outcome_60m_pct=excluded.outcome_60m_pct,
              outcome_eod_pct=excluded.outcome_eod_pct,
              evaluated_at=excluded.evaluated_at
            """,
            (
                outcome_id,
                signal_id,
                outcome.get("price_after_5m"),
                outcome.get("price_after_15m"),
                outcome.get("price_after_60m"),
                outcome.get("price_eod"),
                outcome.get("mfe"),
                outcome.get("mae"),
                outcome.get("outcome_5m_pct"),
                outcome.get("outcome_15m_pct"),
                outcome.get("outcome_60m_pct"),
                outcome.get("outcome_eod_pct"),
                _now_iso(),
            ),
        )
        if existing:
            return str(existing["outcome_id"])
        return outcome_id

    def link_paper_trade(
        self,
        signal_id: Optional[str],
        local_trade_id: Optional[str],
        ticker: str,
        side: str,
        qty: Optional[float],
        price: Optional[float],
        paper_position_id: Optional[str] = None,
        comment: str = "",
        event_ts: Optional[str] = None,
    ) -> str:
        link_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO paper_trade_links (
              link_id, signal_id, local_trade_id, ticker, side, event_ts, qty, price, paper_position_id, comment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                link_id,
                signal_id,
                local_trade_id,
                ticker,
                side,
                event_ts or _now_iso(),
                qty,
                price,
                paper_position_id,
                comment,
            ),
        )
        return link_id

