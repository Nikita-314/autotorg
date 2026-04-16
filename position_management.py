"""
Многоуровневое сопровождение позиции (long/short симметрично по ценовой логике).

Режимы:
- off: не используется (в bot остаётся классический SL/TP).
- shadow: расчёт + логи POSITION_MGMT в аналитику; исполнение — старый SL/TP.
- paper: выходы управляет этот модуль (частичные и полные) в PaperBroker.

Константы — стартовые эвристики (ATR/профиль), подбираются по логам и бэктесту.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class VolatilityBucket(str, Enum):
    CALM = "calm"
    MEDIUM = "medium"
    WILD = "wild"


# --- Коэффициенты (привязка к ATR и корзине волатильности) ---
K_ATR_INITIAL_STOP_CALM = 1.6
K_ATR_INITIAL_STOP_MEDIUM = 2.0
K_ATR_INITIAL_STOP_WILD = 2.6

SWING_BUFFER_ATR_FRAC = 0.35
SOFT_ZONE_FRACTION_OF_RISK = 0.42

PARTIAL_EXIT_FRAC_CALM = 0.28
PARTIAL_EXIT_FRAC_MEDIUM = 0.38
PARTIAL_EXIT_FRAC_WILD = 0.48

TP_ACTIVATION_R_MULT_CALM = 0.85
TP_ACTIVATION_R_MULT_MEDIUM = 1.0
TP_ACTIVATION_R_MULT_WILD = 1.15

PROFIT_ACTIVATION_PARTIAL_FRAC = 0.28

TRAILING_GIVEBACK_FRAC_OF_PEAK_MOVE_CALM = 0.32
TRAILING_GIVEBACK_FRAC_OF_PEAK_MOVE_MEDIUM = 0.38
TRAILING_GIVEBACK_FRAC_OF_PEAK_MOVE_WILD = 0.48

TRAILING_MIN_ATR_FRAC = 0.55


@dataclass
class InstrumentProfile:
    atr: float
    mean_range_pct: float
    mean_true_range_pct: float
    std_returns: float
    typical_pullback_pct: float
    false_break_proxy: float
    mean_volume: float
    volume_spike_ratio: float
    pullback_after_impulse_pct: float
    bucket: VolatilityBucket
    raw: Dict[str, float] = field(default_factory=dict)


@dataclass
class ManagedPositionState:
    entry_price: float
    qty: float
    remaining_qty: float
    side: PositionSide
    opened_at: str
    entry_bar_key: str
    initial_stop: float
    hard_stop: float
    soft_stop: float
    tp_activation_level: float
    trailing_active: bool
    trailing_peak_price: float
    trailing_trough_price: float
    partial_exit_stage: int
    realized_partial_pnl_rub: float
    regime_snapshot: Dict[str, Any]
    ticker_volatility_bucket: str
    ticker_profile: str
    risk_per_unit: float
    bars_in_danger_zone: int
    last_bar_key: str
    peak_unrealized_move_pct: float
    last_shadow_hash: str = ""


@dataclass
class ManagementEvent:
    code: str
    message: str
    details: Dict[str, Any]
    close_fraction: float = 0.0
    full_exit: bool = False


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 2:
        return 0.0
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _swing_low(df: pd.DataFrame, lookback: int = 5) -> float:
    if df is None or len(df) < lookback + 1:
        return float(df["low"].iloc[-1]) if df is not None and len(df) else 0.0
    return float(df["low"].iloc[-(lookback + 1) :].min())


def _swing_high(df: pd.DataFrame, lookback: int = 5) -> float:
    if df is None or len(df) < lookback + 1:
        return float(df["high"].iloc[-1]) if df is not None and len(df) else 0.0
    return float(df["high"].iloc[-(lookback + 1) :].max())


def _profile_from_metrics(
    atr: float,
    atr_pct: float,
    mean_range_pct: float,
    mean_tr_pct: float,
    std_returns: float,
    typical_pullback_pct: float,
    false_break_proxy: float,
    mean_volume: float,
    volume_spike_ratio: float,
    pullback_after_impulse_pct: float,
    last_close: float,
) -> InstrumentProfile:
    vol_score = std_returns * 0.5 + mean_tr_pct * 0.5 + mean_range_pct * 0.3
    if vol_score < 0.012:
        bucket = VolatilityBucket.CALM
    elif vol_score > 0.022:
        bucket = VolatilityBucket.WILD
    else:
        bucket = VolatilityBucket.MEDIUM

    raw = {
        "vol_score": vol_score,
        "atr_pct": atr_pct,
        "mean_range_pct": mean_range_pct,
        "mean_tr_pct": mean_tr_pct,
        "std_returns": std_returns,
    }
    return InstrumentProfile(
        atr=atr,
        mean_range_pct=mean_range_pct,
        mean_true_range_pct=mean_tr_pct,
        std_returns=std_returns,
        typical_pullback_pct=typical_pullback_pct,
        false_break_proxy=false_break_proxy,
        mean_volume=mean_volume,
        volume_spike_ratio=volume_spike_ratio,
        pullback_after_impulse_pct=pullback_after_impulse_pct,
        bucket=bucket,
        raw=raw,
    )


def compute_instrument_profile(df: Optional[pd.DataFrame], n: int = 60) -> InstrumentProfile:
    """Профиль по последним N барам."""
    if df is None or len(df) < 20:
        return InstrumentProfile(
            atr=0.0,
            mean_range_pct=0.02,
            mean_true_range_pct=0.02,
            std_returns=0.015,
            typical_pullback_pct=0.01,
            false_break_proxy=0.3,
            mean_volume=0.0,
            volume_spike_ratio=1.0,
            pullback_after_impulse_pct=0.012,
            bucket=VolatilityBucket.MEDIUM,
            raw={},
        )

    d = df.tail(min(n, len(df))).copy()
    if "open" not in d.columns:
        d["open"] = d["close"]
    c = d["close"].replace(0, math.nan)
    last_close = float(c.iloc[-1])
    rng = (d["high"] - d["low"]) / c
    mean_range_pct = float(rng.mean()) if len(rng) else 0.02
    ret = c.pct_change().dropna()
    std_returns = float(ret.std()) if len(ret) > 1 else 0.015
    atr = _atr(d, 14)
    atr_pct = float(atr / last_close) if last_close else mean_range_pct

    h, l, cl = d["high"], d["low"], d["close"]
    prev = cl.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    mean_tr_pct = float((tr / cl).mean()) if len(tr) else mean_range_pct

    imp = ret.abs() > 0.01
    pullback_after_impulse_pct = float(ret.shift(-1).abs().where(imp).dropna().mean() or 0.012)

    rng_bar = (h - l).replace(0, math.nan)
    upper_wick = (h - pd.concat([d["open"], cl], axis=1).max(axis=1)).clip(lower=0) / rng_bar
    lower_wick = (pd.concat([d["open"], cl], axis=1).min(axis=1) - l).clip(lower=0) / rng_bar
    wick = (upper_wick + lower_wick) / 2.0
    false_break_proxy = float((wick > 0.45).mean()) if len(wick) else 0.3

    vol = d["volume"] if "volume" in d.columns else pd.Series([0.0] * len(d))
    mean_volume = float(vol.mean()) if vol.mean() == vol.mean() else 0.0
    last_vol = float(vol.iloc[-1])
    volume_spike_ratio = float(last_vol / mean_volume) if mean_volume > 1e-9 else 1.0

    typical_pullback_pct = float(ret.abs().rolling(5).mean().iloc[-1])

    return _profile_from_metrics(
        atr,
        atr_pct,
        mean_range_pct,
        mean_tr_pct,
        std_returns,
        typical_pullback_pct,
        false_break_proxy,
        mean_volume,
        volume_spike_ratio,
        pullback_after_impulse_pct,
        last_close,
    )


def _bucket_k_initial(bucket: VolatilityBucket) -> float:
    if bucket == VolatilityBucket.CALM:
        return K_ATR_INITIAL_STOP_CALM
    if bucket == VolatilityBucket.WILD:
        return K_ATR_INITIAL_STOP_WILD
    return K_ATR_INITIAL_STOP_MEDIUM


def _bucket_partial_frac(bucket: VolatilityBucket) -> float:
    if bucket == VolatilityBucket.CALM:
        return PARTIAL_EXIT_FRAC_CALM
    if bucket == VolatilityBucket.WILD:
        return PARTIAL_EXIT_FRAC_WILD
    return PARTIAL_EXIT_FRAC_MEDIUM


def _bucket_tp_r_mult(bucket: VolatilityBucket) -> float:
    if bucket == VolatilityBucket.CALM:
        return TP_ACTIVATION_R_MULT_CALM
    if bucket == VolatilityBucket.WILD:
        return TP_ACTIVATION_R_MULT_WILD
    return TP_ACTIVATION_R_MULT_MEDIUM


def _bucket_trailing_giveback(bucket: VolatilityBucket) -> float:
    if bucket == VolatilityBucket.CALM:
        return TRAILING_GIVEBACK_FRAC_OF_PEAK_MOVE_CALM
    if bucket == VolatilityBucket.WILD:
        return TRAILING_GIVEBACK_FRAC_OF_PEAK_MOVE_WILD
    return TRAILING_GIVEBACK_FRAC_OF_PEAK_MOVE_MEDIUM


def build_initial_state(
    entry_price: float,
    qty: float,
    side: PositionSide,
    opened_at: str,
    entry_bar_key: str,
    df: Optional[pd.DataFrame],
    strategy_sl_floor_pct: float,
    strategy_tp_floor_pct: float,
) -> ManagedPositionState:
    """Уровни: hard — аварийный; soft — между entry и hard (partial); TP activation — зона прибыли."""
    _ = strategy_tp_floor_pct  # зарезервировано для будущих целей
    prof = compute_instrument_profile(df)
    atr = max(prof.atr, entry_price * 1e-6)
    k = _bucket_k_initial(prof.bucket)
    swing_lo = _swing_low(df) if df is not None and len(df) else entry_price
    swing_hi = _swing_high(df) if df is not None and len(df) else entry_price
    buf = SWING_BUFFER_ATR_FRAC * atr

    if side == PositionSide.LONG:
        candidate_hard = min(
            swing_lo - buf,
            entry_price - k * atr,
            entry_price * (1.0 - max(strategy_sl_floor_pct, prof.mean_true_range_pct * 2.5)),
        )
        hard_stop = float(candidate_hard)
        initial_stop = entry_price - k * atr
        soft_stop = entry_price - SOFT_ZONE_FRACTION_OF_RISK * max(entry_price - hard_stop, 1e-9 * entry_price)
        risk_per_unit = max(entry_price - hard_stop, 1e-9 * entry_price)
        tp_activation = entry_price + _bucket_tp_r_mult(prof.bucket) * risk_per_unit
        peak = entry_price
        trough = entry_price
    else:
        candidate_hard = max(
            swing_hi + buf,
            entry_price + k * atr,
            entry_price * (1.0 + max(strategy_sl_floor_pct, prof.mean_true_range_pct * 2.5)),
        )
        hard_stop = float(candidate_hard)
        initial_stop = entry_price + k * atr
        soft_stop = entry_price + SOFT_ZONE_FRACTION_OF_RISK * max(hard_stop - entry_price, 1e-9 * entry_price)
        risk_per_unit = max(hard_stop - entry_price, 1e-9 * entry_price)
        tp_activation = entry_price - _bucket_tp_r_mult(prof.bucket) * risk_per_unit
        peak = entry_price
        trough = entry_price

    return ManagedPositionState(
        entry_price=entry_price,
        qty=qty,
        remaining_qty=qty,
        side=side,
        opened_at=opened_at,
        entry_bar_key=entry_bar_key,
        initial_stop=float(initial_stop),
        hard_stop=float(hard_stop),
        soft_stop=float(soft_stop),
        tp_activation_level=float(tp_activation),
        trailing_active=False,
        trailing_peak_price=float(peak),
        trailing_trough_price=float(trough),
        partial_exit_stage=0,
        realized_partial_pnl_rub=0.0,
        regime_snapshot={"profile_raw": prof.raw, "bucket": prof.bucket.value},
        ticker_volatility_bucket=prof.bucket.value,
        ticker_profile=str(prof.bucket.value),
        risk_per_unit=float(risk_per_unit),
        bars_in_danger_zone=0,
        last_bar_key=entry_bar_key,
        peak_unrealized_move_pct=0.0,
    )


def _bucket_from_str(s: str) -> VolatilityBucket:
    try:
        return VolatilityBucket(s)
    except ValueError:
        return VolatilityBucket.MEDIUM


def evaluate_tick(
    state: ManagedPositionState,
    price: float,
    bar_key: str,
    profile: InstrumentProfile,
) -> Tuple[ManagedPositionState, List[ManagementEvent]]:
    events: List[ManagementEvent] = []
    bucket = _bucket_from_str(state.ticker_volatility_bucket)
    atr = max(profile.atr, state.entry_price * 1e-6)
    giveback = _bucket_trailing_giveback(bucket)
    partial_frac = _bucket_partial_frac(bucket)

    if bar_key != state.last_bar_key:
        state.last_bar_key = bar_key

    rem = state.remaining_qty
    side = state.side

    if side == PositionSide.LONG:
        # 1) Аварийный выход (ниже всего)
        if price <= state.hard_stop:
            events.append(
                ManagementEvent(
                    "HARD_STOP_HIT",
                    "Hard stop (long)",
                    {"price": price, "hard_stop": state.hard_stop},
                    full_exit=True,
                )
            )
            return state, events

        # 2) Мягкая зона — частичный выход (выше hard, ниже/на soft)
        if price <= state.soft_stop and state.partial_exit_stage < 1 and rem > 0:
            state.partial_exit_stage = 1
            events.append(
                ManagementEvent(
                    "SOFT_STOP_HIT",
                    "Soft zone — partial exit (long)",
                    {**_snap(state, price), "partial_frac": partial_frac},
                    close_fraction=partial_frac,
                )
            )

        # 3) Зона прибыли + trailing
        if not state.trailing_active and price >= state.tp_activation_level:
            state.trailing_active = True
            state.trailing_peak_price = price
            events.append(
                ManagementEvent(
                    "TP_ZONE_ENTERED",
                    "Profit zone entered (long)",
                    _snap(state, price),
                    close_fraction=0.0,
                )
            )
            events.append(
                ManagementEvent(
                    "TRAILING_ACTIVATED",
                    "Trailing mode on (long)",
                    {"tp_activation": state.tp_activation_level, "price": price},
                    close_fraction=PROFIT_ACTIVATION_PARTIAL_FRAC,
                )
            )
            state.partial_exit_stage = max(state.partial_exit_stage, 2)

        if state.trailing_active and rem > 0:
            if price > state.trailing_peak_price:
                state.trailing_peak_price = price
                state.peak_unrealized_move_pct = max(
                    state.peak_unrealized_move_pct,
                    (state.trailing_peak_price - state.entry_price) / state.entry_price * 100.0,
                )
                events.append(
                    ManagementEvent(
                        "TRAILING_UPDATED",
                        "New peak (long)",
                        {"peak": state.trailing_peak_price},
                    )
                )
            peak_move = state.trailing_peak_price - state.entry_price
            giveback_abs = max(giveback * peak_move, TRAILING_MIN_ATR_FRAC * atr)
            if price <= state.trailing_peak_price - giveback_abs and peak_move > 0:
                events.append(
                    ManagementEvent(
                        "TRAILING_EXIT",
                        "Trailing giveback exit (long)",
                        {
                            "price": price,
                            "peak": state.trailing_peak_price,
                            "giveback_abs": giveback_abs,
                        },
                        full_exit=True,
                    )
                )
                return state, events

    else:
        # SHORT
        if price >= state.hard_stop:
            events.append(
                ManagementEvent(
                    "HARD_STOP_HIT",
                    "Hard stop (short)",
                    {"price": price, "hard_stop": state.hard_stop},
                    full_exit=True,
                )
            )
            return state, events

        if price >= state.soft_stop and state.partial_exit_stage < 1 and rem > 0:
            state.partial_exit_stage = 1
            events.append(
                ManagementEvent(
                    "SOFT_STOP_HIT",
                    "Soft zone — partial exit (short)",
                    {**_snap(state, price), "partial_frac": partial_frac},
                    close_fraction=partial_frac,
                )
            )

        if not state.trailing_active and price <= state.tp_activation_level:
            state.trailing_active = True
            state.trailing_trough_price = price
            events.append(
                ManagementEvent(
                    "TP_ZONE_ENTERED",
                    "Profit zone entered (short)",
                    _snap(state, price),
                )
            )
            events.append(
                ManagementEvent(
                    "TRAILING_ACTIVATED",
                    "Trailing mode on (short)",
                    {"tp_activation": state.tp_activation_level, "price": price},
                    close_fraction=PROFIT_ACTIVATION_PARTIAL_FRAC,
                )
            )
            state.partial_exit_stage = max(state.partial_exit_stage, 2)

        if state.trailing_active and rem > 0:
            if price < state.trailing_trough_price:
                state.trailing_trough_price = price
                state.peak_unrealized_move_pct = max(
                    state.peak_unrealized_move_pct,
                    (state.entry_price - state.trailing_trough_price) / state.entry_price * 100.0,
                )
                events.append(
                    ManagementEvent(
                        "TRAILING_UPDATED",
                        "New trough (short)",
                        {"trough": state.trailing_trough_price},
                    )
                )
            peak_move = state.entry_price - state.trailing_trough_price
            giveback_abs = max(giveback * peak_move, TRAILING_MIN_ATR_FRAC * atr)
            if price >= state.trailing_trough_price + giveback_abs and peak_move > 0:
                events.append(
                    ManagementEvent(
                        "TRAILING_EXIT",
                        "Trailing rebound exit (short)",
                        {
                            "price": price,
                            "trough": state.trailing_trough_price,
                            "giveback_abs": giveback_abs,
                        },
                        full_exit=True,
                    )
                )
                return state, events

    return state, events


def _snap(state: ManagedPositionState, price: float) -> Dict[str, Any]:
    return {
        "entry": state.entry_price,
        "remaining_qty": state.remaining_qty,
        "price": price,
        "soft_stop": state.soft_stop,
        "hard_stop": state.hard_stop,
        "tp_activation": state.tp_activation_level,
        "trailing_active": state.trailing_active,
        "side": state.side.value,
    }


def clone_state(s: ManagedPositionState) -> ManagedPositionState:
    return ManagedPositionState(
        entry_price=s.entry_price,
        qty=s.qty,
        remaining_qty=s.remaining_qty,
        side=s.side,
        opened_at=s.opened_at,
        entry_bar_key=s.entry_bar_key,
        initial_stop=s.initial_stop,
        hard_stop=s.hard_stop,
        soft_stop=s.soft_stop,
        tp_activation_level=s.tp_activation_level,
        trailing_active=s.trailing_active,
        trailing_peak_price=s.trailing_peak_price,
        trailing_trough_price=s.trailing_trough_price,
        partial_exit_stage=s.partial_exit_stage,
        realized_partial_pnl_rub=s.realized_partial_pnl_rub,
        regime_snapshot=dict(s.regime_snapshot),
        ticker_volatility_bucket=s.ticker_volatility_bucket,
        ticker_profile=s.ticker_profile,
        risk_per_unit=s.risk_per_unit,
        bars_in_danger_zone=s.bars_in_danger_zone,
        last_bar_key=s.last_bar_key,
        peak_unrealized_move_pct=s.peak_unrealized_move_pct,
        last_shadow_hash=s.last_shadow_hash,
    )


def shadow_preview(
    state: ManagedPositionState,
    price: float,
    bar_key: str,
    profile: InstrumentProfile,
) -> List[ManagementEvent]:
    """Оценка без изменения переданного state (копия)."""
    st, ev = evaluate_tick(clone_state(state), price, bar_key, profile)
    return ev
