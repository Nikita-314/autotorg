import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests
import yfinance as yf
from aiogram import Bot, Dispatcher, F, Router

from data_loader import load_candles
from features import build_features
from position_management import (
    InstrumentProfile,
    ManagedPositionState,
    PositionSide,
    build_initial_state,
    compute_instrument_profile,
    evaluate_tick,
)
from analytics import AnalyticsDB, OutcomeEvaluator, PaperTradeMapper, SignalLogger, safe_to_json
from analytics.trade_signal_id import resolve_buy_signal_id_at_close
from aiogram.filters import Command, CommandStart
from aiogram.types import KeyboardButton, Message, ReplyKeyboardMarkup
from dotenv import load_dotenv
from openai import OpenAI
from tradingview_ta import Interval, TA_Handler
from adaptive_analysis import AdaptiveAnalysisEngine, AdaptiveDecision


def _setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, stream=sys.stdout)
    log_file = os.getenv("BOT_LOG_FILE", "").strip()
    if log_file:
        root = logging.getLogger()
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)


LOGGER = logging.getLogger("tv-bcs-bot")


INTERVAL_MAP = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
    "1w": Interval.INTERVAL_1_WEEK,
}


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min(value, max_value), min_value)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def format_rub(value: Any) -> str:
    amount = safe_float(value)
    return f"{amount:,.2f} RUB".replace(",", " ")


def format_close_pnl_line(pnl: float, pnl_pct: float) -> str:
    """Строка PnL для закрытия сделки: при малых |pnl| не маскировать под 0.00 RUB."""
    if abs(pnl) < 10:
        pnl_part = f"{pnl:.4f} RUB"
    else:
        pnl_part = format_rub(pnl)
    return f"{pnl_part} ({pnl_pct:.4f}%)"


def format_signed_rub(value: float) -> str:
    amount = safe_float(value)
    if abs(amount) < 10:
        return f"{amount:+.4f} RUB"
    return f"{amount:+,.2f} RUB".replace(",", " ")


def _bar_key_at(dt: datetime, interval: str) -> str:
    """UTC-метка текущего бара (округление под TV_INTERVAL), без смены стратегии/ML."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    iv = (interval or "1h").strip().lower()
    if iv in ("1h", "60m"):
        floored = dt.replace(minute=0, second=0, microsecond=0)
    elif iv == "15m":
        floored = dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)
    elif iv == "5m":
        floored = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
    elif iv in ("1d", "d", "1D"):
        floored = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        floored = dt.replace(minute=0, second=0, microsecond=0)
    return floored.isoformat(timespec="seconds")


def _compute_supertrend(
    highs: List[float], lows: List[float], closes: List[float],
    period: int = 10, multiplier: float = 3.0,
) -> Tuple[float, float]:
    """SuperTrend (ATR-based). Returns (supertrend_value, direction). direction: 1=бычий, -1=медвежий."""
    if len(closes) < period + 1:
        return 0.0, 0.0
    n = len(closes)
    tr_list: List[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)
    atr_list: List[float] = []
    for i in range(len(tr_list)):
        if i < period - 1:
            atr_list.append(sum(tr_list[: i + 1]) / (i + 1))
        else:
            atr_list.append(sum(tr_list[i - period + 1 : i + 1]) / period)
    hl2 = [(highs[i + 1] + lows[i + 1]) / 2 for i in range(len(atr_list))]
    upper = [hl2[i] + multiplier * atr_list[i] for i in range(len(atr_list))]
    lower = [hl2[i] - multiplier * atr_list[i] for i in range(len(atr_list))]
    supertrend: List[float] = []
    direction: List[float] = []
    for i in range(len(atr_list)):
        c = closes[i + 1]
        if i == 0:
            sup = lower[i]
            dir_val = 1.0
        else:
            prev_sup = supertrend[-1]
            prev_dir = direction[-1]
            if prev_dir == 1.0:
                if lower[i] > prev_sup:
                    sup = lower[i]
                    dir_val = 1.0
                elif c < prev_sup:
                    sup = upper[i]
                    dir_val = -1.0
                else:
                    sup = prev_sup
                    dir_val = 1.0
            else:
                if upper[i] < prev_sup:
                    sup = upper[i]
                    dir_val = -1.0
                elif c > prev_sup:
                    sup = lower[i]
                    dir_val = 1.0
                else:
                    sup = prev_sup
                    dir_val = -1.0
        supertrend.append(sup)
        direction.append(dir_val)
    return supertrend[-1], direction[-1]


def _fetch_ohlc_moex(symbol: str, interval: str) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """Fetch OHLC from MOEX ISS. Returns (highs, lows, closes) or None."""
    from datetime import datetime, timedelta

    interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 60, "1d": 24, "1w": 24}
    moex_interval = interval_map.get(interval, 60)
    till = datetime.utcnow()
    if interval in ("1d", "1w"):
        from_dt = till - timedelta(days=90)
    else:
        from_dt = till - timedelta(days=30)
    from_str = from_dt.strftime("%Y-%m-%d")
    till_str = till.strftime("%Y-%m-%d")
    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{symbol}/candles.json"
        f"?from={from_str}&till={till_str}&interval={moex_interval}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        candles = data.get("candles", {})
        rows = candles.get("data", [])
        cols = candles.get("columns", [])
        if not rows or not cols:
            return None
        idx = {c: i for i, c in enumerate(cols)}
        highs = [float(r[idx["high"]]) for r in rows if "high" in idx]
        lows = [float(r[idx["low"]]) for r in rows if "low" in idx]
        closes = [float(r[idx["close"]]) for r in rows if "close" in idx]
        if len(closes) < 15:
            return None
        return highs, lows, closes
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.debug("MOEX OHLC fetch failed for %s: %s", symbol, exc)
        return None


def _fetch_supertrend(symbol: str, exchange: str, interval: str) -> Tuple[float, float]:
    """Fetch OHLC, compute SuperTrend. MOEX: ISS API. US: yfinance. Returns (supertrend, direction)."""
    if exchange.upper() == "MOEX":
        return _fetch_supertrend_moex(symbol, interval)
    return _fetch_supertrend_yf(symbol, interval)


def _fetch_supertrend_moex(symbol: str, interval: str) -> Tuple[float, float]:
    """Fetch OHLC from MOEX ISS, compute SuperTrend."""
    ohlc = _fetch_ohlc_moex(symbol, interval)
    if ohlc is None:
        return 0.0, 0.0
    highs, lows, closes = ohlc
    return _compute_supertrend(highs, lows, closes, period=10, multiplier=3.0)


def _compute_ema(closes: List[float], period: int) -> float:
    """Exponential Moving Average. Returns last value."""
    if len(closes) < period:
        return closes[-1] if closes else 0.0
    alpha = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for c in closes[period:]:
        ema = alpha * c + (1 - alpha) * ema
    return ema


def _compute_rsi(closes: List[float], period: int = 14) -> float:
    """RSI (Relative Strength Index). Returns 0-100."""
    if len(closes) < period + 1:
        return 50.0
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(0, diff))
        losses.append(max(0, -diff))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss <= 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd(
    closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[float, float]:
    """MACD line and signal. Returns (macd, macd_signal)."""
    if len(closes) < slow + signal:
        return 0.0, 0.0
    ema_fast = _compute_ema(closes[:slow], fast)
    ema_slow = sum(closes[:slow]) / slow
    alpha_f = 2.0 / (fast + 1)
    alpha_s = 2.0 / (slow + 1)
    macd_line: List[float] = []
    for i in range(slow, len(closes)):
        ema_fast = alpha_f * closes[i] + (1 - alpha_f) * ema_fast
        ema_slow = alpha_s * closes[i] + (1 - alpha_s) * ema_slow
        macd_line.append(ema_fast - ema_slow)
    if len(macd_line) < signal:
        return macd_line[-1] if macd_line else 0.0, 0.0
    macd_sig = sum(macd_line[:signal]) / signal
    alpha_sig = 2.0 / (signal + 1)
    for v in macd_line[signal:]:
        macd_sig = alpha_sig * v + (1 - alpha_sig) * macd_sig
    return macd_line[-1], macd_sig


def _compute_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    """ADX (Average Directional Index). Returns 0-100."""
    if len(closes) < period + 2:
        return 25.0
    tr_list: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_list.append(tr)
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    atr = sum(tr_list[:period]) / period
    plus_di = 100 * sum(plus_dm[:period]) / period / atr if atr > 0 else 0
    minus_di = 100 * sum(minus_dm[:period]) / period / atr if atr > 0 else 0
    dx_list: List[float] = []
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_di = 100 * (plus_dm[i] if plus_dm[i] > minus_dm[i] else 0) / atr if atr > 0 else 0
        minus_di = 100 * (minus_dm[i] if minus_dm[i] > plus_dm[i] else 0) / atr if atr > 0 else 0
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
        dx_list.append(dx)
    if len(dx_list) < period:
        return 25.0
    adx = sum(dx_list[:period]) / period
    alpha = 2.0 / (period + 1)
    for dx in dx_list[period:]:
        adx = alpha * dx + (1 - alpha) * adx
    return adx


def _fetch_supertrend_yf(symbol: str, interval: str) -> Tuple[float, float]:
    """Fetch OHLC via yfinance, compute SuperTrend."""
    period_map = {"1m": "5d", "5m": "5d", "15m": "5d", "1h": "1mo", "4h": "2mo", "1d": "3mo", "1w": "1y"}
    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d", "1w": "1wk"}
    period = period_map.get(interval, "1mo")
    yf_interval = interval_map.get(interval, "1h")
    try:
        data = yf.download(symbol, period=period, interval=yf_interval, progress=False, auto_adjust=True)
        if data is None or data.empty or len(data) < 15:
            return 0.0, 0.0
        highs = data["High"].tolist() if "High" in data.columns else data["Close"].tolist()
        lows = data["Low"].tolist() if "Low" in data.columns else data["Close"].tolist()
        closes = data["Close"].tolist()
        if len(closes) < 15:
            return 0.0, 0.0
        return _compute_supertrend(highs, lows, closes, period=10, multiplier=3.0)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.debug("SuperTrend yfinance fetch failed for %s: %s", symbol, exc)
        return 0.0, 0.0


@dataclass
class StockInstrument:
    symbol: str
    name: str = ""
    exchange: str = ""
    tradable: bool = True
    status: str = ""
    source_meta: Dict[str, Any] = None


@dataclass
class IndicatorSnapshot:
    symbol: str
    price: float
    ema20: float
    ema50: float
    rsi: float
    adx: float
    macd: float
    macd_signal: float
    buy_count: float
    sell_count: float
    neutral_count: float
    supertrend: float  # 0 = нет данных
    supertrend_direction: float  # 1 = бычий, -1 = медвежий, 0 = нет данных


def _get_snapshot_from_moex(symbol: str, exchange: str, interval: str) -> Optional[IndicatorSnapshot]:
    """Compute IndicatorSnapshot from MOEX OHLC (or yfinance for non-MOEX). Returns None if fetch fails."""
    if exchange.upper() != "MOEX":
        ohlc = None
        try:
            data = yf.download(
                symbol,
                period="3mo",
                interval="1h",
                progress=False,
                auto_adjust=True,
                timeout=15,
            )
            if data is not None and not data.empty and len(data) >= 50:
                highs = data["High"].tolist() if "High" in data.columns else data["Close"].tolist()
                lows = data["Low"].tolist() if "Low" in data.columns else data["Close"].tolist()
                closes = data["Close"].tolist()
                ohlc = (highs, lows, closes)
        except Exception:  # pylint: disable=broad-except
            pass
    else:
        ohlc = _fetch_ohlc_moex(symbol, interval)
    if ohlc is None:
        return None
    highs, lows, closes = ohlc
    price = closes[-1] if closes else 0.0
    if price <= 0:
        return None
    ema20 = _compute_ema(closes, 20)
    ema50 = _compute_ema(closes, 50)
    rsi = _compute_rsi(closes, 14)
    adx = _compute_adx(highs, lows, closes, 14)
    macd, macd_signal = _compute_macd(closes, 12, 26, 9)
    supertrend_val, supertrend_dir = _compute_supertrend(highs, lows, closes, period=10, multiplier=3.0)
    return IndicatorSnapshot(
        symbol=symbol,
        price=price,
        ema20=ema20,
        ema50=ema50,
        rsi=rsi,
        adx=adx,
        macd=macd,
        macd_signal=macd_signal,
        buy_count=0.0,
        sell_count=0.0,
        neutral_count=0.0,
        supertrend=supertrend_val,
        supertrend_direction=supertrend_dir,
    )


@dataclass
class StrategyConfig:
    buy_threshold: float
    sell_threshold: float
    ema_weight: float
    macd_weight: float
    rsi_weight: float
    adx_weight: float
    supertrend_weight: float
    rsi_overbought: float
    rsi_oversold: float
    stop_loss_pct: float
    take_profit_pct: float

    @classmethod
    def defaults(cls, settings: "Settings") -> "StrategyConfig":
        return cls(
            buy_threshold=settings.buy_threshold,
            sell_threshold=settings.sell_threshold,
            ema_weight=0.30,
            macd_weight=0.25,
            rsi_weight=0.15,
            adx_weight=0.20,
            supertrend_weight=0.15,
            rsi_overbought=70.0,
            rsi_oversold=30.0,
            stop_loss_pct=settings.stop_loss_pct,
            take_profit_pct=settings.take_profit_pct,
        )


@dataclass
class Settings:
    seed_symbols: List[str]
    screener: str
    exchange: str
    interval: str
    max_symbols: int
    buy_threshold: float
    sell_threshold: float
    position_size_rub: float
    stop_loss_pct: float
    take_profit_pct: float
    min_price: float
    max_price: float
    trading_mode: str
    paper_initial_balance_rub: float
    paper_state_file: str
    paper_balance_state_file: str
    paper_commission_rate: float
    paper_commission_min_rub: float
    paper_include_commission: bool
    bcs_stocks_api_url: str
    bcs_stocks_cache_file: str
    universe_refresh_cycles: int
    bcs_api_base_url: str
    bcs_api_token: str
    bcs_account_id: str
    bcs_dry_run: bool
    openai_api_key: str
    openai_model: str
    openai_rebalance_cycles: int
    enable_ai_risk_filter: bool
    strategy_state_file: str
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_notify_hold: bool
    poll_seconds: int
    tv_request_delay_sec: float
    strategy_mode: str  # "indicator" | "supertrend_ml"
    ml_enabled: bool
    ml_prob_threshold: float
    analytics_enabled: bool
    analytics_db_path: str
    analytics_strategy_version: str
    analytics_log_holds: bool
    analytics_outcome_eval_enabled: bool
    # position_mgmt_mode: off | shadow | paper — многоуровневое сопровождение (только paper execution)
    position_mgmt_mode: str
    adaptive_mode: str
    adaptive_state_file: str
    adaptive_lookback_days: int
    adaptive_refresh_cycles: int
    adaptive_min_observations: int
    adaptive_negative_outcome_pct: float
    adaptive_positive_outcome_pct: float
    adaptive_max_threshold_delta_frac: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        seed_symbols = [s.strip().upper() for s in os.getenv("TV_SYMBOLS", "AAPL,MSFT").split(",") if s.strip()]
        return cls(
            seed_symbols=seed_symbols,
            screener=os.getenv("TV_SCREENER", "america"),
            exchange=os.getenv("TV_EXCHANGE", "NASDAQ"),
            interval=os.getenv("TV_INTERVAL", "1h"),
            max_symbols=int(os.getenv("MAX_SYMBOLS", "30")),
            buy_threshold=float(os.getenv("BUY_THRESHOLD", "0.55")),
            sell_threshold=float(os.getenv("SELL_THRESHOLD", "-0.55")),
            position_size_rub=float(os.getenv("POSITION_SIZE_RUB", "10000")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.03")),
            take_profit_pct=float(os.getenv("TAKE_PROFIT_PCT", "0.06")),
            min_price=float(os.getenv("MIN_PRICE", "10")),
            max_price=float(os.getenv("MAX_PRICE", "1000000")),
            trading_mode=os.getenv("TRADING_MODE", "paper").strip().lower(),
            paper_initial_balance_rub=float(os.getenv("PAPER_INITIAL_BALANCE_RUB", "50000")),
            paper_state_file=os.getenv("PAPER_STATE_FILE", "paper_state.json"),
            paper_balance_state_file=os.getenv("PAPER_BALANCE_STATE_FILE", "paper_balance_state.json"),
            paper_commission_rate=float(os.getenv("PAPER_COMMISSION_RATE", "0.0005")),
            paper_commission_min_rub=float(os.getenv("PAPER_COMMISSION_MIN_RUB", "0")),
            paper_include_commission=os.getenv("PAPER_INCLUDE_COMMISSION", "true").lower() == "true",
            bcs_stocks_api_url=os.getenv("BCS_STOCKS_API_URL", "").strip(),
            bcs_stocks_cache_file=os.getenv("BCS_STOCKS_CACHE_FILE", "bcs_stocks_cache.json"),
            universe_refresh_cycles=int(os.getenv("UNIVERSE_REFRESH_CYCLES", "30")),
            bcs_api_base_url=os.getenv("BCS_API_BASE_URL", "").strip(),
            bcs_api_token=os.getenv("BCS_API_TOKEN", "").strip(),
            bcs_account_id=os.getenv("BCS_ACCOUNT_ID", "").strip(),
            bcs_dry_run=os.getenv("BCS_DRY_RUN", "true").lower() == "true",
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            openai_rebalance_cycles=int(os.getenv("OPENAI_REBALANCE_CYCLES", "20")),
            enable_ai_risk_filter=os.getenv("ENABLE_AI_RISK_FILTER", "true").lower() == "true",
            strategy_state_file=os.getenv("STRATEGY_STATE_FILE", "strategy_state.json"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            telegram_notify_hold=os.getenv("TELEGRAM_NOTIFY_HOLD", "false").lower() == "true",
            poll_seconds=int(os.getenv("POLL_SECONDS", "60")),
            tv_request_delay_sec=float(os.getenv("TV_REQUEST_DELAY_SEC", "2.5")),
            strategy_mode=os.getenv("STRATEGY_MODE", "supertrend_ml").strip().lower(),
            ml_enabled=os.getenv("ML_ENABLED", "true").lower() == "true",
            ml_prob_threshold=float(os.getenv("ML_PROB_THRESHOLD", "0.10")),
            analytics_enabled=os.getenv("ANALYTICS_ENABLED", "false").lower() == "true",
            analytics_db_path=os.getenv("ANALYTICS_DB_PATH", "analytics.db").strip(),
            analytics_strategy_version=os.getenv("ANALYTICS_STRATEGY_VERSION", "stage-a").strip(),
            analytics_log_holds=os.getenv("ANALYTICS_LOG_HOLDS", "true").lower() == "true",
            analytics_outcome_eval_enabled=os.getenv("ANALYTICS_OUTCOME_EVAL_ENABLED", "false").lower() == "true",
            position_mgmt_mode=os.getenv("POSITION_MGMT_MODE", "off").strip().lower(),
            adaptive_mode=os.getenv("ADAPTIVE_MODE", "off").strip().lower(),
            adaptive_state_file=os.getenv("ADAPTIVE_STATE_FILE", "adaptive_state.json").strip(),
            adaptive_lookback_days=int(os.getenv("ADAPTIVE_LOOKBACK_DAYS", "14")),
            adaptive_refresh_cycles=int(os.getenv("ADAPTIVE_REFRESH_CYCLES", "5")),
            adaptive_min_observations=int(os.getenv("ADAPTIVE_MIN_OBSERVATIONS", "5")),
            adaptive_negative_outcome_pct=float(os.getenv("ADAPTIVE_NEGATIVE_OUTCOME_PCT", "-0.25")),
            adaptive_positive_outcome_pct=float(os.getenv("ADAPTIVE_POSITIVE_OUTCOME_PCT", "0.05")),
            adaptive_max_threshold_delta_frac=float(os.getenv("ADAPTIVE_MAX_THRESHOLD_DELTA_FRAC", "0.15")),
        )


class TradingViewClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def get_snapshot(self, symbol: str) -> IndicatorSnapshot:
        try:
            interval = INTERVAL_MAP.get(self.settings.interval, Interval.INTERVAL_1_HOUR)
            handler = TA_Handler(
                symbol=symbol,
                screener=self.settings.screener,
                exchange=self.settings.exchange,
                interval=interval,
            )
            analysis = handler.get_analysis()
            indicators = analysis.indicators
            summary = analysis.summary

            supertrend_val, supertrend_dir = _fetch_supertrend(
                symbol, self.settings.exchange, self.settings.interval
            )
            snapshot = IndicatorSnapshot(
                symbol=symbol,
                price=safe_float(indicators.get("close")),
                ema20=safe_float(indicators.get("EMA20")),
                ema50=safe_float(indicators.get("EMA50")),
                rsi=safe_float(indicators.get("RSI")),
                adx=safe_float(indicators.get("ADX")),
                macd=safe_float(indicators.get("MACD.macd")),
                macd_signal=safe_float(indicators.get("MACD.signal")),
                buy_count=safe_float(summary.get("BUY")),
                sell_count=safe_float(summary.get("SELL")),
                neutral_count=safe_float(summary.get("NEUTRAL")),
                supertrend=supertrend_val,
                supertrend_direction=supertrend_dir,
            )
            if snapshot.price <= 0:
                raise RuntimeError(f"Cannot get market price for {symbol}.")
            return snapshot
        except Exception as exc:  # pylint: disable=broad-except
            fallback = _get_snapshot_from_moex(
                symbol, self.settings.exchange, self.settings.interval
            )
            if fallback is not None:
                LOGGER.info("TradingView failed for %s (%s), using MOEX/yf fallback", symbol, exc)
                return fallback
            raise


class BCSUniverseProvider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_path = Path(settings.bcs_stocks_cache_file)
        self.moex_tqbr_url = (
            "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json?iss.meta=off"
        )

    def _cache(self, instruments: List[StockInstrument]) -> None:
        payload = [asdict(item) for item in instruments]
        with self.cache_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)

    def _read_cache(self) -> List[StockInstrument]:
        if not self.cache_path.exists():
            return []
        with self.cache_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        result: List[StockInstrument] = []
        for row in raw:
            result.append(
                StockInstrument(
                    symbol=str(row.get("symbol", "")).upper(),
                    name=str(row.get("name", "")),
                    exchange=str(row.get("exchange", "")),
                    tradable=bool(row.get("tradable", True)),
                    status=str(row.get("status", "")),
                    source_meta=row.get("source_meta", {}),
                )
            )
        return [x for x in result if x.symbol]

    def _parse_api_payload(self, payload: Any) -> List[StockInstrument]:
        candidates = None
        if isinstance(payload, list):
            candidates = payload
        elif isinstance(payload, dict):
            for key in ("instruments", "data", "items", "stocks", "result"):
                value = payload.get(key)
                if isinstance(value, list):
                    candidates = value
                    break

        if not candidates:
            raise RuntimeError("Unsupported BCS stocks payload format.")

        instruments: List[StockInstrument] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            symbol = (
                str(item.get("symbol") or item.get("ticker") or item.get("code") or item.get("securityCode") or "")
                .strip()
                .upper()
            )
            if not symbol:
                continue
            tradable = bool(item.get("tradable", item.get("canTrade", item.get("isTradingAllowed", True))))
            instruments.append(
                StockInstrument(
                    symbol=symbol,
                    name=str(item.get("name") or item.get("shortName") or ""),
                    exchange=str(item.get("exchange") or item.get("market") or ""),
                    tradable=tradable,
                    status=str(item.get("status") or item.get("tradingStatus") or ""),
                    source_meta=item,
                )
            )
        return instruments

    def _load_moex_universe(self) -> List[StockInstrument]:
        response = requests.get(self.moex_tqbr_url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        securities = payload.get("securities", {})
        columns = securities.get("columns", [])
        data = securities.get("data", [])
        if not columns or not data:
            raise RuntimeError("MOEX payload does not contain securities list.")

        col_idx = {name: idx for idx, name in enumerate(columns)}
        secid_idx = col_idx.get("SECID")
        shortname_idx = col_idx.get("SHORTNAME")
        status_idx = col_idx.get("STATUS")

        if secid_idx is None:
            raise RuntimeError("MOEX payload missing SECID column.")

        instruments: List[StockInstrument] = []
        for row in data:
            if secid_idx >= len(row):
                continue
            symbol = str(row[secid_idx] or "").strip().upper()
            if not symbol:
                continue
            name = ""
            status = ""
            if shortname_idx is not None and shortname_idx < len(row):
                name = str(row[shortname_idx] or "")
            if status_idx is not None and status_idx < len(row):
                status = str(row[status_idx] or "")

            instruments.append(
                StockInstrument(
                    symbol=symbol,
                    name=name,
                    exchange="MOEX",
                    tradable=True,
                    status=status,
                    source_meta={"source": "moex_tqbr"},
                )
            )
        return instruments

    def fetch(self) -> List[StockInstrument]:
        if self.settings.bcs_stocks_api_url:
            try:
                response = requests.get(self.settings.bcs_stocks_api_url, timeout=30)
                response.raise_for_status()
                payload = response.json()
                instruments = self._parse_api_payload(payload)
                if instruments:
                    self._cache(instruments)
                    return instruments
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to load BCS universe from API: %s", exc)

        try:
            instruments = self._load_moex_universe()
            if instruments:
                self._cache(instruments)
                LOGGER.info("Loaded MOEX TQBR universe: %d symbols", len(instruments))
                return instruments
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Failed to load MOEX universe: %s", exc)

        cached = self._read_cache()
        if cached:
            LOGGER.info("Using cached BCS stocks universe: %d symbols", len(cached))
            return cached

        LOGGER.warning("Using fallback seed symbols (set BCS_STOCKS_API_URL or use MOEX network access).")
        return [StockInstrument(symbol=symbol, exchange=self.settings.exchange) for symbol in self.settings.seed_symbols]


class StockRiskFilter:
    BAD_STATUS_WORDS = ("bankrupt", "liquidat", "delist", "suspend", "halt", "default")

    def __init__(self, settings: Settings, openai_client: Optional[OpenAI]):
        self.settings = settings
        self.openai = openai_client
        self._last_ai_blocked: List[Dict[str, str]] = []

    def deterministic_filter(self, instruments: List[StockInstrument]) -> List[StockInstrument]:
        filtered: List[StockInstrument] = []
        for item in instruments:
            status = item.status.lower()
            if not item.tradable:
                continue
            if any(bad in status for bad in self.BAD_STATUS_WORDS):
                continue
            filtered.append(item)
        return filtered

    def ai_filter(self, instruments: List[StockInstrument]) -> List[StockInstrument]:
        if not self.settings.enable_ai_risk_filter or not self.openai or not instruments:
            self._last_ai_blocked = []
            return instruments

        short_list = [
            {
                "symbol": x.symbol,
                "name": x.name,
                "exchange": x.exchange,
                "status": x.status,
            }
            for x in instruments[:200]
        ]
        prompt = (
            "You are a risk filter for stock trading. "
            "Given instruments, remove symbols with high bankruptcy/delisting/default risk or obvious trading bans. "
            "Return strict JSON with keys allow (array of symbols) and block (array of objects with symbol/reason). "
            f"Instruments: {json.dumps(short_list, ensure_ascii=True)}"
        )
        try:
            result = self.openai.chat.completions.create(
                model=self.settings.openai_model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = result.choices[0].message.content or "{}"
            parsed = extract_json_object(raw)
            allowed = {str(x).upper() for x in parsed.get("allow", [])}
            blocked = parsed.get("block", [])
            self._last_ai_blocked = []
            if blocked:
                for row in blocked[:10]:
                    symbol = str(row.get("symbol", "")).upper()
                    reason = str(row.get("reason", "no reason"))
                    LOGGER.info("AI blocked %s: %s", symbol, reason)
                    if symbol:
                        self._last_ai_blocked.append({"symbol": symbol, "reason": reason})
            if not allowed:
                return instruments
            return [x for x in instruments if x.symbol in allowed]
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("AI risk filter failed, using deterministic universe: %s", exc)
            self._last_ai_blocked = []
            return instruments

    def pop_last_ai_blocked(self) -> List[Dict[str, str]]:
        payload = self._last_ai_blocked
        self._last_ai_blocked = []
        return payload


class IndicatorScorer:
    def score(self, snapshot: IndicatorSnapshot, strategy: StrategyConfig) -> float:
        ema_signal = 1.0 if snapshot.ema20 >= snapshot.ema50 else -1.0
        macd_signal = 1.0 if snapshot.macd >= snapshot.macd_signal else -1.0
        rsi_signal = 0.0
        if snapshot.rsi > 0:
            if snapshot.rsi <= strategy.rsi_oversold:
                rsi_signal = 1.0
            elif snapshot.rsi >= strategy.rsi_overbought:
                rsi_signal = -1.0

        supertrend_signal = snapshot.supertrend_direction if snapshot.supertrend_direction != 0 else 0.0

        total_weight = (
            strategy.ema_weight
            + strategy.macd_weight
            + strategy.rsi_weight
            + (strategy.supertrend_weight if supertrend_signal != 0 else 0.0)
        )
        if total_weight <= 0:
            return 0.0

        raw = (
            ema_signal * strategy.ema_weight
            + macd_signal * strategy.macd_weight
            + rsi_signal * strategy.rsi_weight
            + supertrend_signal * strategy.supertrend_weight
        ) / total_weight

        if snapshot.adx > 20:
            adx_boost = clamp((snapshot.adx - 20) / 30, 0.0, 1.0)
            raw *= 1 + strategy.adx_weight * adx_boost

        return clamp(raw, -1.0, 1.0)


class OpenAIStrategyAdvisor:
    def __init__(self, settings: Settings, client: Optional[OpenAI]):
        self.settings = settings
        self.client = client
        self.state_path = Path(settings.strategy_state_file)
        self.last_rebalance_trade_count = 0

    def load(self) -> Optional[StrategyConfig]:
        if not self.state_path.exists():
            return None
        try:
            with self.state_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.last_rebalance_trade_count = int(payload.get("last_rebalance_trade_count", 0))
            return StrategyConfig(
                buy_threshold=float(payload["buy_threshold"]),
                sell_threshold=float(payload["sell_threshold"]),
                ema_weight=float(payload["ema_weight"]),
                macd_weight=float(payload["macd_weight"]),
                rsi_weight=float(payload["rsi_weight"]),
                adx_weight=float(payload["adx_weight"]),
                supertrend_weight=float(payload.get("supertrend_weight", 0.15)),
                rsi_overbought=float(payload["rsi_overbought"]),
                rsi_oversold=float(payload["rsi_oversold"]),
                stop_loss_pct=float(payload.get("stop_loss_pct", 0.03)),
                take_profit_pct=float(payload.get("take_profit_pct", 0.06)),
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Cannot load strategy state: %s", exc)
            return None

    def save(self, strategy: StrategyConfig, *, trade_count: Optional[int] = None) -> None:
        if trade_count is not None:
            self.last_rebalance_trade_count = max(0, int(trade_count))
        payload = asdict(strategy)
        payload["last_rebalance_trade_count"] = int(self.last_rebalance_trade_count)
        with self.state_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)

    def maybe_update(
        self,
        cycle: int,
        strategy: StrategyConfig,
        performance: Dict[str, Any],
        market_sample: List[Dict[str, Any]],
    ) -> Tuple[StrategyConfig, Optional[str]]:
        if not self.client or self.settings.openai_rebalance_cycles <= 0:
            return strategy, None
        if cycle % self.settings.openai_rebalance_cycles != 0:
            return strategy, None
        trade_count = int(safe_float(performance.get("trade_count"), 0.0))
        if trade_count <= 0:
            return strategy, None
        if trade_count <= self.last_rebalance_trade_count:
            return strategy, None

        prompt = (
            "You are a quant assistant for short-term trend strategy tuning. "
            "Given current strategy and recent performance, return strict JSON with keys: "
            "buy_threshold, sell_threshold, ema_weight, macd_weight, rsi_weight, adx_weight, "
            "supertrend_weight, rsi_overbought, rsi_oversold, stop_loss_pct, take_profit_pct, reason. "
            "Keep risk moderate.\n"
            f"current_strategy={json.dumps(asdict(strategy), ensure_ascii=True)}\n"
            f"performance={json.dumps(performance, ensure_ascii=True)}\n"
            f"market_sample={json.dumps(market_sample[:25], ensure_ascii=True)}"
        )

        try:
            result = self.client.chat.completions.create(
                model=self.settings.openai_model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = result.choices[0].message.content or "{}"
            parsed = extract_json_object(raw)
            updated = StrategyConfig(
                buy_threshold=clamp(float(parsed.get("buy_threshold", strategy.buy_threshold)), 0.3, 0.9),
                sell_threshold=clamp(float(parsed.get("sell_threshold", strategy.sell_threshold)), -0.9, -0.3),
                ema_weight=clamp(float(parsed.get("ema_weight", strategy.ema_weight)), 0.0, 2.0),
                macd_weight=clamp(float(parsed.get("macd_weight", strategy.macd_weight)), 0.0, 2.0),
                rsi_weight=clamp(float(parsed.get("rsi_weight", strategy.rsi_weight)), 0.0, 2.0),
                adx_weight=clamp(float(parsed.get("adx_weight", strategy.adx_weight)), 0.0, 2.0),
                supertrend_weight=clamp(float(parsed.get("supertrend_weight", strategy.supertrend_weight)), 0.0, 2.0),
                rsi_overbought=clamp(float(parsed.get("rsi_overbought", strategy.rsi_overbought)), 55, 90),
                rsi_oversold=clamp(float(parsed.get("rsi_oversold", strategy.rsi_oversold)), 10, 45),
                stop_loss_pct=clamp(float(parsed.get("stop_loss_pct", strategy.stop_loss_pct)), 0.005, 0.20),
                take_profit_pct=clamp(float(parsed.get("take_profit_pct", strategy.take_profit_pct)), 0.01, 0.50),
            )
            reason = str(parsed.get("reason", "")).strip()

            changes: List[str] = []
            for field_name, title in (
                ("buy_threshold", "BUY порог"),
                ("sell_threshold", "SELL порог"),
                ("ema_weight", "вес EMA"),
                ("macd_weight", "вес MACD"),
                ("rsi_weight", "вес RSI"),
                ("adx_weight", "усиление ADX"),
                ("supertrend_weight", "вес SuperTrend"),
                ("rsi_overbought", "RSI перекупленность"),
                ("rsi_oversold", "RSI перепроданность"),
                ("stop_loss_pct", "Stop-loss %"),
                ("take_profit_pct", "Take-profit %"),
            ):
                old_value = float(getattr(strategy, field_name))
                new_value = float(getattr(updated, field_name))
                if abs(old_value - new_value) > 1e-9:
                    if field_name.endswith("_pct"):
                        changes.append(f"- {title}: {old_value * 100:.2f}% -> {new_value * 100:.2f}%")
                    else:
                        changes.append(f"- {title}: {old_value:.4f} -> {new_value:.4f}")

            if not changes:
                self.last_rebalance_trade_count = trade_count
                return strategy, None

            self.save(updated, trade_count=trade_count)
            LOGGER.info("Strategy updated by OpenAI: %s", updated)
            summary = "🤖 ИИ изменил параметры стратегии:\n" + "\n".join(changes)
            if reason:
                summary += f"\n📝 Причина от ИИ: {reason}"
            return updated, summary
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("OpenAI strategy update failed: %s", exc)
            return strategy, None


class BCSClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        if settings.bcs_api_token:
            self.session.headers.update({"Authorization": f"Bearer {settings.bcs_api_token}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def place_order(self, symbol: str, side: str, amount_rub: float, market_price: float) -> None:
        payload = {
            "accountId": self.settings.bcs_account_id,
            "symbol": symbol,
            "side": side,
            "amountRub": str(amount_rub),
            "orderType": "MARKET",
        }
        if self.settings.bcs_dry_run:
            LOGGER.info("[DRY-RUN] %s %s for %.2f RUB @ %.4f", side, symbol, amount_rub, market_price)
            return
        endpoint = f"{self.settings.bcs_api_base_url.rstrip('/')}/v1/orders"
        response = self.session.post(endpoint, json=payload, timeout=30)
        response.raise_for_status()
        LOGGER.info("Order sent: %s %s for %.2f RUB", side, symbol, amount_rub)

    def get_position(self, symbol: str) -> Tuple[float, float]:
        """BCS: no local position tracking, assume no position."""
        return 0.0, 0.0

    def can_afford_buy(self, amount_rub: float) -> bool:
        """BCS: assume we can always buy."""
        return True


class PaperBroker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.state_path = Path(settings.paper_state_file)
        self.balance_state_path = Path(settings.paper_balance_state_file)
        self.initial_balance_rub = settings.paper_initial_balance_rub
        self.last_reset_at = datetime.now(timezone.utc).isoformat()
        self.cash_rub = settings.paper_initial_balance_rub
        self.positions: Dict[str, float] = {}
        self.avg_price: Dict[str, float] = {}
        self.open_entry_commission_rub: Dict[str, float] = {}
        self.realized_gross_pnl_rub = 0.0
        self.realized_commission_rub = 0.0
        self.trade_count = 0
        self._load()

    def _load(self) -> None:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.cash_rub = safe_float(payload.get("cash_rub"), self.settings.paper_initial_balance_rub)
            self.positions = {k: safe_float(v) for k, v in payload.get("positions", {}).items()}
            self.avg_price = {k: safe_float(v) for k, v in payload.get("avg_price", {}).items()}
            self.open_entry_commission_rub = {
                k: safe_float(v) for k, v in payload.get("open_entry_commission_rub", {}).items()
            }
            self.realized_gross_pnl_rub = safe_float(
                payload.get("realized_gross_pnl_rub"), safe_float(payload.get("realized_pnl_rub"))
            )
            self.realized_commission_rub = safe_float(payload.get("realized_commission_rub"))
            self.trade_count = int(payload.get("trade_count", 0))
        if self.balance_state_path.exists():
            with self.balance_state_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            self.initial_balance_rub = safe_float(
                payload.get("initial_balance_rub"), self.settings.paper_initial_balance_rub
            )
            self.last_reset_at = str(payload.get("last_reset_at") or self.last_reset_at)
        else:
            self.sync_balance_snapshot({})
        if not self.state_path.exists():
            self._save_trading_state()

    def _save_trading_state(self) -> None:
        payload = {
            "cash_rub": round(self.cash_rub, 2),
            "positions": self.positions,
            "avg_price": self.avg_price,
            "open_entry_commission_rub": self.open_entry_commission_rub,
            "realized_gross_pnl_rub": round(self.realized_gross_pnl_rub, 2),
            "realized_commission_rub": round(self.realized_commission_rub, 2),
            "realized_net_pnl_rub": round(self.realized_net_pnl_rub, 2),
            "realized_pnl_rub": round(self.realized_net_pnl_rub, 2),
            "trade_count": self.trade_count,
        }
        with self.state_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)

    @property
    def realized_net_pnl_rub(self) -> float:
        return self.realized_gross_pnl_rub - self.realized_commission_rub

    def _commission_for_notional(self, notional_rub: float) -> float:
        if not self.settings.paper_include_commission:
            return 0.0
        notional = abs(safe_float(notional_rub))
        if notional <= 0:
            return 0.0
        return max(
            notional * max(0.0, self.settings.paper_commission_rate),
            max(0.0, self.settings.paper_commission_min_rub),
        )

    def _max_affordable_buy_notional(self, cash_rub: float) -> float:
        cash = max(0.0, safe_float(cash_rub))
        if not self.settings.paper_include_commission:
            return cash
        rate = max(0.0, self.settings.paper_commission_rate)
        min_commission = max(0.0, self.settings.paper_commission_min_rub)
        candidates: List[float] = []
        notional_case_b = cash - min_commission
        if notional_case_b > 0:
            if rate <= 0 or (notional_case_b * rate) <= min_commission + 1e-12:
                candidates.append(notional_case_b)
        if rate >= 0:
            notional_case_a = cash / (1.0 + rate) if (1.0 + rate) > 0 else 0.0
            if rate <= 0 or (notional_case_a * rate) + 1e-12 >= min_commission:
                candidates.append(notional_case_a)
        candidates.append(0.0)
        return max(candidates)

    def _fallback_prices(self) -> Dict[str, float]:
        prices = {symbol: self.avg_price.get(symbol, 0.0) for symbol in self.positions}
        return {symbol: price for symbol, price in prices.items() if price > 0}

    def get_balance_snapshot(self, latest_prices: Dict[str, float]) -> Dict[str, Any]:
        merged_prices = dict(self._fallback_prices())
        merged_prices.update({k: safe_float(v) for k, v in latest_prices.items()})
        market_value = sum(qty * merged_prices.get(symbol, 0.0) for symbol, qty in self.positions.items())
        equity = self.cash_rub + market_value
        total_pnl = equity - self.initial_balance_rub
        unrealized_pnl = total_pnl - self.realized_net_pnl_rub
        return {
            "initial_balance_rub": round(self.initial_balance_rub, 2),
            "current_balance_rub": round(equity, 2),
            "cash_rub": round(self.cash_rub, 2),
            "market_value_rub": round(market_value, 2),
            "equity_rub": round(equity, 2),
            "realized_gross_pnl_rub": round(self.realized_gross_pnl_rub, 2),
            "realized_commission_rub": round(self.realized_commission_rub, 2),
            "realized_net_pnl_rub": round(self.realized_net_pnl_rub, 2),
            "realized_pnl_total_rub": round(self.realized_net_pnl_rub, 2),
            "realized_pnl_rub": round(self.realized_net_pnl_rub, 2),
            "unrealized_pnl_rub": round(unrealized_pnl, 2),
            "total_pnl_rub": round(total_pnl, 2),
            "trade_count": self.trade_count,
            "positions_count": len(self.positions),
            "last_reset_at": self.last_reset_at,
        }

    def sync_balance_snapshot(self, latest_prices: Dict[str, float]) -> Dict[str, Any]:
        snapshot = self.get_balance_snapshot(latest_prices)
        with self.balance_state_path.open("w", encoding="utf-8") as file:
            json.dump(snapshot, file, ensure_ascii=True, indent=2)
        return snapshot

    def _save(self, latest_prices: Optional[Dict[str, float]] = None) -> None:
        self._save_trading_state()
        self.sync_balance_snapshot(latest_prices or {})

    def place_order(self, symbol: str, side: str, amount_rub: float, market_price: float) -> Optional[Dict[str, Any]]:
        if market_price <= 0 or amount_rub <= 0:
            return None
        if side == "BUY":
            spend = min(amount_rub, self._max_affordable_buy_notional(self.cash_rub))
            if spend <= 0:
                return None
            commission_rub = self._commission_for_notional(spend)
            total_cash_out = spend + commission_rub
            qty = spend / market_price
            old_qty = self.positions.get(symbol, 0.0)
            old_avg = self.avg_price.get(symbol, 0.0)
            total_qty = old_qty + qty
            self.avg_price[symbol] = ((old_qty * old_avg) + (qty * market_price)) / total_qty
            self.positions[symbol] = total_qty
            self.open_entry_commission_rub[symbol] = self.open_entry_commission_rub.get(symbol, 0.0) + commission_rub
            self.cash_rub -= total_cash_out
            self.realized_commission_rub += commission_rub
            self.trade_count += 1
            LOGGER.info(
                "[PAPER] BUY %s qty=%.6f price=%.4f commission=%.4f",
                symbol,
                qty,
                market_price,
                commission_rub,
            )
            execution = {
                "side": side,
                "qty": qty,
                "filled_notional_rub": spend,
                "notional_rub": spend,
                "commission_rub": commission_rub,
                "commission_rate": self.settings.paper_commission_rate if self.settings.paper_include_commission else 0.0,
                "gross_pnl": 0.0,
                "gross_pnl_pct": 0.0,
                "net_pnl": -commission_rub,
                "net_pnl_pct": ((-commission_rub / spend) * 100.0) if spend > 0 else 0.0,
                "entry_commission_allocated_rub": commission_rub,
                "exit_commission_rub": 0.0,
                "cash_after_rub": self.cash_rub,
            }
        elif side == "SELL":
            held = self.positions.get(symbol, 0.0)
            if held <= 0:
                return None
            qty = min(held, amount_rub / market_price)
            if qty <= 0:
                return None
            notional_rub = qty * market_price
            proceeds = notional_rub
            commission_rub = self._commission_for_notional(notional_rub)
            entry = self.avg_price.get(symbol, market_price)
            gross_pnl = (market_price - entry) * qty
            gross_pnl_pct = ((market_price / entry) - 1.0) * 100 if entry > 0 else 0.0
            entry_commission_pool = self.open_entry_commission_rub.get(symbol, 0.0)
            entry_commission_allocated_rub = entry_commission_pool * (qty / held) if held > 0 else 0.0
            total_trade_commission_rub = entry_commission_allocated_rub + commission_rub
            net_pnl = gross_pnl - total_trade_commission_rub
            notional_entry_rub = entry * qty
            net_pnl_pct = (net_pnl / notional_entry_rub) * 100.0 if notional_entry_rub > 0 else 0.0
            self.realized_gross_pnl_rub += gross_pnl
            self.realized_commission_rub += commission_rub
            left = held - qty
            if left <= 1e-12:
                self.positions.pop(symbol, None)
                self.avg_price.pop(symbol, None)
                self.open_entry_commission_rub.pop(symbol, None)
            else:
                self.positions[symbol] = left
                self.open_entry_commission_rub[symbol] = max(0.0, entry_commission_pool - entry_commission_allocated_rub)
            self.cash_rub += proceeds - commission_rub
            self.trade_count += 1
            LOGGER.info(
                "[PAPER] SELL %s qty=%.6f price=%.4f commission=%.4f net=%.4f",
                symbol,
                qty,
                market_price,
                commission_rub,
                net_pnl,
            )
            execution = {
                "side": side,
                "qty": qty,
                "filled_notional_rub": notional_rub,
                "notional_rub": notional_rub,
                "commission_rub": total_trade_commission_rub,
                "commission_rate": self.settings.paper_commission_rate if self.settings.paper_include_commission else 0.0,
                "gross_pnl": gross_pnl,
                "gross_pnl_pct": gross_pnl_pct,
                "net_pnl": net_pnl,
                "net_pnl_pct": net_pnl_pct,
                "entry_commission_allocated_rub": entry_commission_allocated_rub,
                "exit_commission_rub": commission_rub,
                "cash_after_rub": self.cash_rub,
            }
        else:
            return None
        self._save({symbol: market_price})
        return execution

    def performance(self, latest_prices: Dict[str, float]) -> Dict[str, Any]:
        return self.sync_balance_snapshot(latest_prices)

    def get_open_positions(self, latest_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for symbol in sorted(self.positions.keys()):
            qty = self.positions.get(symbol, 0.0)
            avg = self.avg_price.get(symbol, 0.0)
            last = latest_prices.get(symbol, avg)
            if qty <= 0:
                continue
            market_value = qty * last
            unrealized = (last - avg) * qty
            rows.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "avg_price": avg,
                    "last_price": last,
                    "market_value_rub": market_value,
                    "unrealized_pnl_rub": unrealized,
                }
            )
        return rows

    def get_position(self, symbol: str) -> Tuple[float, float]:
        qty = self.positions.get(symbol, 0.0)
        avg = self.avg_price.get(symbol, 0.0)
        return qty, avg

    def can_afford_buy(self, amount_rub: float) -> bool:
        return self.cash_rub >= amount_rub

    def reset_balance(self, initial_balance_rub: Optional[float] = None) -> Dict[str, Any]:
        amount = safe_float(initial_balance_rub, self.settings.paper_initial_balance_rub)
        if amount <= 0:
            amount = self.settings.paper_initial_balance_rub
        self.initial_balance_rub = amount
        self.last_reset_at = datetime.now(timezone.utc).isoformat()
        self.cash_rub = amount
        self.positions = {}
        self.avg_price = {}
        self.open_entry_commission_rub = {}
        self.realized_gross_pnl_rub = 0.0
        self.realized_commission_rub = 0.0
        self.trade_count = 0
        self._save({})
        return self.get_balance_snapshot({})

    def set_balance(self, initial_balance_rub: float) -> Dict[str, Any]:
        return self.reset_balance(initial_balance_rub)


class TelegramNotifier:
    def __init__(self, settings: Settings):
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.notify_hold = settings.telegram_notify_hold

    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    async def _send(self, message: str) -> Optional[int]:
        async with Bot(token=self.token) as bot:
            sent = await bot.send_message(chat_id=self.chat_id, text=message)
            return sent.message_id

    async def _edit(self, message_id: int, message: str) -> bool:
        async with Bot(token=self.token) as bot:
            try:
                await bot.edit_message_text(chat_id=self.chat_id, message_id=message_id, text=message)
                return True
            except Exception:
                return False

    def send(self, message: str) -> Optional[int]:
        if not self.enabled():
            return None
        try:
            return asyncio.run(self._send(message))
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Telegram send failed: %s", exc)
            return None

    def edit(self, message_id: Optional[int], message: str) -> bool:
        if not self.enabled() or not message_id:
            return False
        try:
            return asyncio.run(self._edit(message_id, message))
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Telegram edit failed: %s", exc)
            return False


class TelegramCommandServer:
    BTN_STATUS = "📊 Статус"
    BTN_STRATEGY = "🧠 Стратегия"
    BTN_PNL = "📈 P&L"
    BTN_BALANCE = "💰 Баланс"
    BTN_POSITIONS = "🧾 Позиции"
    BTN_CLOSE_ALL = "🛑 Закрыть все позиции"
    BTN_CONFIRM_CLOSE_ALL = "✅ Подтвердить закрытие всех позиций"
    BTN_CANCEL_CLOSE_ALL = "❌ Отмена"
    BTN_HELP = "❓ Помощь"

    def __init__(
        self,
        settings: Settings,
        status_provider: Callable[[], str],
        strategy_provider: Callable[[], str],
        pnl_provider: Callable[[], str],
        balance_provider: Callable[[], str],
        positions_provider: Callable[[], str],
        balance_reset_handler: Callable[[Optional[float]], str],
        balance_set_handler: Callable[[float], str],
        close_all_check_handler: Callable[[], Tuple[bool, str]],
        close_all_execute_handler: Callable[[], str],
        set_close_all_entry_pause: Optional[Callable[[bool], None]] = None,
    ):
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.status_provider = status_provider
        self.strategy_provider = strategy_provider
        self.pnl_provider = pnl_provider
        self.balance_provider = balance_provider
        self.positions_provider = positions_provider
        self.balance_reset_handler = balance_reset_handler
        self.balance_set_handler = balance_set_handler
        self.close_all_check_handler = close_all_check_handler
        self.close_all_execute_handler = close_all_execute_handler
        self.set_close_all_entry_pause = set_close_all_entry_pause
        self.default_balance_amount = settings.paper_initial_balance_rub
        self._thread: Optional[Thread] = None
        self._pending_close_all_chats: Set[int] = set()

    def enabled(self) -> bool:
        return bool(self.token)

    async def _run_polling(self) -> None:
        bot = Bot(token=self.token)
        dispatcher = Dispatcher()
        router = Router()
        try:
            me = await bot.get_me()
            LOGGER.info("Telegram: бот @%s, polling запускается", me.username or me.id)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Telegram: токен недействителен или сеть недоступна (%s). Проверьте TELEGRAM_BOT_TOKEN.", exc)
            await bot.session.close()
            return
        try:
            await bot.delete_webhook(drop_pending_updates=False)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Telegram: delete_webhook пропущен (%s)", exc)
        LOGGER.info("Telegram long polling started.")
        actions_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text=self.BTN_STATUS), KeyboardButton(text=self.BTN_STRATEGY)],
                [KeyboardButton(text=self.BTN_PNL), KeyboardButton(text=self.BTN_BALANCE)],
                [KeyboardButton(text=self.BTN_POSITIONS), KeyboardButton(text=self.BTN_CLOSE_ALL)],
                [KeyboardButton(text=self.BTN_HELP)],
            ],
            resize_keyboard=True,
            is_persistent=True,
            input_field_placeholder="Выберите действие",
        )
        confirm_close_keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text=self.BTN_CONFIRM_CLOSE_ALL)],
                [KeyboardButton(text=self.BTN_CANCEL_CLOSE_ALL)],
            ],
            resize_keyboard=True,
            is_persistent=False,
            input_field_placeholder="Подтвердите массовое закрытие",
        )

        async def send_keyboard(chat_id: str, text: str) -> None:
            await bot.send_message(chat_id=chat_id, text=text, reply_markup=actions_keyboard)

        async def ask_close_all_confirmation(message: Message) -> None:
            allowed, text = self.close_all_check_handler()
            if not allowed:
                self._pending_close_all_chats.discard(message.chat.id)
                if self.set_close_all_entry_pause:
                    self.set_close_all_entry_pause(False)
                await message.answer(text, reply_markup=actions_keyboard)
                return
            self._pending_close_all_chats.add(message.chat.id)
            if self.set_close_all_entry_pause:
                self.set_close_all_entry_pause(True)
            await message.answer(
                text
                + "\n\n⏸ Новые входы временно приостановлены до подтверждения или отмены."
                + "\n\nПодтвердите массовое закрытие всех paper-позиций.",
                reply_markup=confirm_close_keyboard,
            )

        @router.message(CommandStart())
        async def command_start(message: Message) -> None:
            await message.answer(
                "👋 Бот запущен.\nВыберите действие на клавиатуре ниже или используйте команды.",
                reply_markup=actions_keyboard,
            )

        @router.message(Command("status"))
        async def command_status(message: Message) -> None:
            await message.answer(self.status_provider())

        @router.message(Command("strategy"))
        async def command_strategy(message: Message) -> None:
            await message.answer(self.strategy_provider())

        @router.message(Command("pnl"))
        async def command_pnl(message: Message) -> None:
            await message.answer(self.pnl_provider())

        @router.message(Command("balance"))
        async def command_balance(message: Message) -> None:
            await message.answer(self.balance_provider())

        @router.message(Command("balance_reset"))
        async def command_balance_reset(message: Message) -> None:
            parts = (message.text or "").split()
            amount = safe_float(parts[1], self._default_balance_amount()) if len(parts) > 1 else None
            await message.answer(self.balance_reset_handler(amount))

        @router.message(Command("balance_set"))
        async def command_balance_set(message: Message) -> None:
            parts = (message.text or "").split()
            if len(parts) < 2:
                await message.answer("💰 Укажите сумму: /balance_set 65000")
                return
            await message.answer(self.balance_set_handler(safe_float(parts[1], self._default_balance_amount())))

        @router.message(Command("positions"))
        async def command_positions(message: Message) -> None:
            await message.answer(self.positions_provider())

        @router.message(Command("close_all_positions"))
        async def command_close_all_positions(message: Message) -> None:
            await ask_close_all_confirmation(message)

        @router.message(Command("help"))
        async def command_help(message: Message) -> None:
            await message.answer(
                "❓ Подсказка:\n"
                "/status, /strategy, /pnl, /balance, /positions, /close_all_positions\n"
                "/balance_reset [amount], /balance_set <amount>\n"
                "Используйте кнопки на клавиатуре ниже.",
                reply_markup=actions_keyboard,
            )

        @router.message(F.text == self.BTN_STATUS)
        async def button_status(message: Message) -> None:
            await message.answer(self.status_provider())

        @router.message(F.text == self.BTN_STRATEGY)
        async def button_strategy(message: Message) -> None:
            await message.answer(self.strategy_provider())

        @router.message(F.text == self.BTN_PNL)
        async def button_pnl(message: Message) -> None:
            await message.answer(self.pnl_provider())

        @router.message(F.text == self.BTN_BALANCE)
        async def button_balance(message: Message) -> None:
            await message.answer(self.balance_provider())

        @router.message(F.text == self.BTN_POSITIONS)
        async def button_positions(message: Message) -> None:
            await message.answer(self.positions_provider())

        @router.message(F.text == self.BTN_CLOSE_ALL)
        async def button_close_all(message: Message) -> None:
            await ask_close_all_confirmation(message)

        @router.message(F.text == self.BTN_CONFIRM_CLOSE_ALL)
        async def button_confirm_close_all(message: Message) -> None:
            if message.chat.id not in self._pending_close_all_chats:
                await message.answer(
                    "ℹ️ Закрытие уже выполнено или подтверждение больше не активно.",
                    reply_markup=actions_keyboard,
                )
                return
            self._pending_close_all_chats.discard(message.chat.id)
            await message.answer(self.close_all_execute_handler(), reply_markup=actions_keyboard)

        @router.message(F.text == self.BTN_CANCEL_CLOSE_ALL)
        async def button_cancel_close_all(message: Message) -> None:
            self._pending_close_all_chats.discard(message.chat.id)
            if self.set_close_all_entry_pause:
                self.set_close_all_entry_pause(False)
            await message.answer("Операция массового закрытия отменена.", reply_markup=actions_keyboard)

        @router.message(F.text == self.BTN_HELP)
        async def button_help(message: Message) -> None:
            await message.answer(
                "❓ Подсказка:\n"
                "/status, /strategy, /pnl, /balance, /positions, /close_all_positions\n"
                "/balance_reset [amount], /balance_set <amount>\n"
                "Используйте кнопки на клавиатуре ниже.",
                reply_markup=actions_keyboard,
            )

        @router.message()
        async def fallback_keyboard(message: Message) -> None:
            await message.answer(
                "⌨️ Клавиатура доступна ниже. Нажмите нужную кнопку.",
                reply_markup=actions_keyboard,
            )

        dispatcher.include_router(router)
        try:
            if self.chat_id:
                try:
                    await send_keyboard(self.chat_id, "✅ Клавиатура подключена. Можно работать через кнопки ниже.")
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning("Cannot pre-send keyboard to configured chat: %s", exc)
            await dispatcher.start_polling(bot, handle_signals=False)
        finally:
            await bot.session.close()
            LOGGER.info("Telegram polling stopped.")

    def _run(self) -> None:
        try:
            asyncio.run(self._run_polling())
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Telegram polling остановлен с ошибкой: %s", exc)

    def start(self) -> None:
        if not self.enabled():
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run, name="telegram-polling", daemon=True)
        self._thread.start()
        LOGGER.info("Telegram command server thread started.")

    def _default_balance_amount(self) -> float:
        return safe_float(self.default_balance_amount, 50000.0)


class SignalEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.project_root = Path(__file__).resolve().parent
        self.openai_client = (
            OpenAI(api_key=settings.openai_api_key, timeout=20.0) if settings.openai_api_key else None
        )
        self.notifier = TelegramNotifier(settings)
        self.tv = TradingViewClient(settings)
        self.universe_provider = BCSUniverseProvider(settings)
        self.risk_filter = StockRiskFilter(settings, self.openai_client)
        self.scorer = IndicatorScorer()
        self.advisor = OpenAIStrategyAdvisor(settings, self.openai_client)
        self.strategy = self.advisor.load() or StrategyConfig.defaults(settings)
        self.ml_model = None
        self.ml_model_version = "n/a"
        if settings.ml_enabled:
            try:
                from models.ml_model import MLSignalModel
                self.ml_model = MLSignalModel(prob_threshold=settings.ml_prob_threshold)
                self.ml_model.load()
                model_path = getattr(self.ml_model, "model_path", None)
                if model_path and Path(model_path).exists():
                    model_mtime = datetime.fromtimestamp(Path(model_path).stat().st_mtime, tz=timezone.utc).isoformat()
                    self.ml_model_version = f"{Path(model_path).name}:{model_mtime}"
                else:
                    self.ml_model_version = "loaded"
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("ML model not loaded: %s", exc)
        self.broker = PaperBroker(settings) if settings.trading_mode == "paper" else BCSClient(settings)
        self.symbols: List[str] = []
        self.cycle = 0
        self.last_prices: Dict[str, float] = {}
        # До первого завершённого run_once get_status/get_pnl брали last_performance без equity/cash → нули.
        self.last_performance: Dict[str, Any] = {"status": "starting"}
        if isinstance(self.broker, PaperBroker):
            self.last_performance = {
                **self.broker.get_balance_snapshot({}),
                "status": "starting",
            }
        self.last_ai_blocked_signature: Tuple[str, ...] = tuple()
        self.trade_meta: Dict[str, Dict[str, Any]] = {}
        self.current_run_id: Optional[str] = None
        self.current_signals_found = 0
        self.current_instruments_checked = 0
        self._operator_lock = Lock()
        self._last_decision_meta: Dict[str, Dict[str, Any]] = {}
        # Защита от повторного BUY / re-entry на том же баре (UTC bucket по TV_INTERVAL)
        self._last_buy_bar_key: Dict[str, str] = {}
        self._last_close_bar_key: Dict[str, str] = {}
        # Пауза новых BUY, пока в Telegram ждём подтверждение «Закрыть все позиции»
        self._manual_close_all_pending = Event()
        # Многоуровневое сопровождение (position_management.py)
        self._pm_states: Dict[str, ManagedPositionState] = {}
        self._pm_profiles: Dict[str, InstrumentProfile] = {}
        # (symbol, signal_id) -> (last_logged reason_code, fingerprint for TRAILING_EXIT only)
        self._pm_last_analytics_event: Dict[Tuple[str, str], Tuple[str, Optional[Tuple[Any, ...]]]] = {}
        self.analytics_db: Optional[AnalyticsDB] = None
        self.analytics_logger: Optional[SignalLogger] = None
        self.outcome_evaluator: Optional[OutcomeEvaluator] = None
        self.paper_mapper: Optional[PaperTradeMapper] = None
        self.adaptive_engine: Optional[AdaptiveAnalysisEngine] = None
        if self.settings.analytics_enabled:
            try:
                analytics_db_path = Path(self.settings.analytics_db_path)
                if not analytics_db_path.is_absolute():
                    analytics_db_path = self.project_root / analytics_db_path
                self.analytics_db = AnalyticsDB(analytics_db_path)
                self.analytics_logger = SignalLogger(self.analytics_db)
                self.outcome_evaluator = OutcomeEvaluator(self.analytics_db, self.analytics_logger)
                self.paper_mapper = PaperTradeMapper(self.analytics_logger)
                LOGGER.info("Analytics layer enabled: %s", analytics_db_path)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to initialize analytics layer: %s", exc)
                self.analytics_db = None
                self.analytics_logger = None
                self.outcome_evaluator = None
                self.paper_mapper = None
                self.adaptive_engine = None
        if self.analytics_db and self.analytics_logger:
            try:
                adaptive_state_path = Path(self.settings.adaptive_state_file)
                if not adaptive_state_path.is_absolute():
                    adaptive_state_path = self.project_root / adaptive_state_path
                self.adaptive_engine = AdaptiveAnalysisEngine(
                    self.analytics_db,
                    self.analytics_logger,
                    state_path=adaptive_state_path,
                    mode=self.settings.adaptive_mode,
                    base_ml_threshold=self.settings.ml_prob_threshold,
                    lookback_days=self.settings.adaptive_lookback_days,
                    refresh_cycles=self.settings.adaptive_refresh_cycles,
                    min_observations=self.settings.adaptive_min_observations,
                    negative_outcome_pct=self.settings.adaptive_negative_outcome_pct,
                    positive_outcome_pct=self.settings.adaptive_positive_outcome_pct,
                    max_threshold_delta_frac=self.settings.adaptive_max_threshold_delta_frac,
                )
                LOGGER.info("Adaptive layer enabled: %s", adaptive_state_path)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Failed to initialize adaptive layer: %s", exc)
                self.adaptive_engine = None
        self.refresh_universe()

    def set_manual_close_all_pending(self, active: bool) -> None:
        if active:
            self._manual_close_all_pending.set()
        else:
            self._manual_close_all_pending.clear()

    def refresh_universe(self) -> None:
        instruments = self.universe_provider.fetch()
        instruments = self.risk_filter.deterministic_filter(instruments)
        ai_pool_size = max(self.settings.max_symbols * 3, self.settings.max_symbols)
        instruments = self.risk_filter.ai_filter(instruments[:ai_pool_size])
        ai_blocked = self.risk_filter.pop_last_ai_blocked()
        blocked_signature = tuple(sorted([row["symbol"] for row in ai_blocked]))
        if ai_blocked and blocked_signature != self.last_ai_blocked_signature and self.notifier.enabled():
            preview = "\n".join([f"- {row['symbol']}: {row['reason']}" for row in ai_blocked[:8]])
            if len(ai_blocked) > 8:
                preview += f"\n... и еще {len(ai_blocked) - 8}"
            self.notifier.send(
                "🛡️ ИИ обновил риск-фильтр бумаг.\n"
                f"Исключено: {len(ai_blocked)}\n{preview}"
            )
        self.last_ai_blocked_signature = blocked_signature
        symbols = []
        for item in instruments:
            if item.symbol not in symbols:
                symbols.append(item.symbol)
            if len(symbols) >= self.settings.max_symbols:
                break
        if not symbols:
            symbols = self.settings.seed_symbols[: self.settings.max_symbols]
        self.symbols = symbols
        LOGGER.info("Active universe size: %d symbols", len(self.symbols))

    def decide(self, score: float) -> str:
        if score >= self.strategy.buy_threshold:
            return "BUY"
        if score <= self.strategy.sell_threshold:
            return "SELL"
        return "HOLD"

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_float_or_none(value: Any) -> Optional[float]:
        try:
            val = float(value)
            if val != val:  # NaN check
                return None
            return val
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _market_session_phase(index_value: Any) -> Optional[str]:
        if not hasattr(index_value, "hour"):
            return None
        hour = int(getattr(index_value, "hour"))
        if hour < 6:
            return "night"
        if hour < 12:
            return "morning"
        if hour < 18:
            return "day"
        return "evening"

    @staticmethod
    def _bars_since_flip(features_df: Any) -> Optional[int]:
        try:
            st_dir = features_df["st_dir"].dropna()
            if len(st_dir) == 0:
                return None
            flip = st_dir.diff().fillna(0).abs()
            group = (flip != 0).cumsum()
            value = int(flip.groupby(group).cumcount().iloc[-1])
            return value
        except Exception:
            return None

    def _build_feature_snapshot(self, df: Any, features_df: Any) -> Dict[str, Any]:
        feature_snapshot: Dict[str, Any] = {
            "close": None,
            "supertrend_value": None,
            "supertrend_direction": None,
            "ema200": None,
            "price_vs_ema200_pct": None,
            "atr": None,
            "volume_1m": None,
            "volume_avg_20": None,
            "volume_ratio": None,
            "return_5m": None,
            "return_15m": None,
            "breakout_flag": None,
            "volatility_20": None,
            "market_session_phase": None,
            "bars_since_flip": None,
        }
        if features_df is None or len(features_df) == 0:
            return feature_snapshot

        try:
            last = features_df.iloc[-1]
            close = self._safe_float_or_none(last.get("close"))
            ema200 = self._safe_float_or_none(last.get("ema200"))
            feature_snapshot["close"] = close
            feature_snapshot["supertrend_value"] = self._safe_float_or_none(last.get("st_line"))
            feature_snapshot["supertrend_direction"] = self._safe_float_or_none(last.get("st_dir"))
            feature_snapshot["ema200"] = ema200
            feature_snapshot["atr"] = self._safe_float_or_none(last.get("atr"))
            feature_snapshot["volume_ratio"] = self._safe_float_or_none(last.get("vol_ratio"))
            if close is not None and ema200 not in (None, 0.0):
                feature_snapshot["price_vs_ema200_pct"] = ((close / ema200) - 1.0) * 100.0
            if hasattr(features_df, "index") and len(features_df.index) > 0:
                feature_snapshot["market_session_phase"] = self._market_session_phase(features_df.index[-1])
        except Exception:
            pass

        try:
            if df is not None and len(df) > 0 and "close" in df.columns:
                close_series = df["close"]
                feature_snapshot["return_5m"] = self._safe_float_or_none(close_series.pct_change(5).iloc[-1])
                feature_snapshot["return_15m"] = self._safe_float_or_none(close_series.pct_change(15).iloc[-1])
                feature_snapshot["volatility_20"] = self._safe_float_or_none(close_series.pct_change().rolling(20).std().iloc[-1])
                prev_high_20 = close_series.shift(1).rolling(20).max().iloc[-1]
                close_val = self._safe_float_or_none(close_series.iloc[-1])
                if close_val is not None and prev_high_20 is not None:
                    feature_snapshot["breakout_flag"] = bool(close_val > float(prev_high_20))
            if df is not None and len(df) > 0 and "volume" in df.columns:
                vol_series = df["volume"]
                feature_snapshot["volume_1m"] = self._safe_float_or_none(vol_series.iloc[-1])
                feature_snapshot["volume_avg_20"] = self._safe_float_or_none(vol_series.rolling(20).mean().iloc[-1])
        except Exception:
            pass

        feature_snapshot["bars_since_flip"] = self._bars_since_flip(features_df)
        return safe_to_json(feature_snapshot)

    def _log_model_inference(
        self,
        symbol: str,
        input_features: Dict[str, Any],
        raw_output: Dict[str, Any],
        decision_label: str,
        confidence_score: Optional[float],
        action_recommendation: str,
        model_used_in_final_decision: bool,
    ) -> None:
        if not self.analytics_logger or not self.current_run_id:
            return
        try:
            self.analytics_logger.log_model_inference(
                run_id=self.current_run_id,
                signal_id=None,
                ticker=symbol,
                model_type="classifier",
                model_version=self.ml_model_version,
                input_features=input_features,
                raw_output=raw_output,
                decision_label=decision_label,
                confidence_score=confidence_score,
                action_recommendation=action_recommendation,
                model_used_in_final_decision=model_used_in_final_decision,
                inference_ts=self._iso_now(),
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Analytics inference log failed for %s: %s", symbol, exc)

    def _decide_supertrend_ml(
        self, symbol: str
    ) -> Optional[Tuple[str, float, IndicatorSnapshot]]:
        """
        Supertrend + ML: вход long только если st_dir=1, close>ema200, P(up)>threshold.
        Возвращает (action, score, snapshot) или None при ошибке.
        """
        self._last_decision_meta[symbol] = {}
        df = load_candles(symbol, self.settings.exchange, self.settings.interval)
        if df is None or len(df) < 100:
            return None
        features = build_features(df)
        features = features.dropna()
        if len(features) < 50:
            return None
        last = features.iloc[-1]
        st_dir = last.get("st_dir", 0)
        close = last.get("close", 0)
        ema200 = last.get("ema200", 0)
        ema50 = last.get("ema50", close)
        rsi = last.get("rsi", 50)
        macd = last.get("macd", 0)
        macd_signal = last.get("macd_signal", 0)
        atr = last.get("atr", 0)
        st_line = last.get("st_line", close)
        feature_snapshot = self._build_feature_snapshot(df, features)

        ml_proba = None
        if self.ml_model:
            ml_proba = self.ml_model.predict_proba_up(features)

        qty, _ = self.broker.get_position(symbol)
        has_position = qty > 0
        market_regime = "trend_up" if st_dir >= 1 else "trend_down"

        if has_position:
            if st_dir <= -1:
                self._last_decision_meta[symbol] = {
                    "reason_code": "EXIT_ON_SUPERTREND_FLIP",
                    "reason_text": "Open position and SuperTrend flipped down.",
                    "feature_snapshot": feature_snapshot,
                    "model_snapshot": {
                        "ml_prob_up": ml_proba,
                        "ml_threshold": self.settings.ml_prob_threshold,
                    },
                    "market_regime": "trend_down",
                    "adaptive_decision": None,
                }
                return ("SELL", -0.8, self._snapshot_from_features(close, ema50, rsi, macd, macd_signal, symbol))
            self._last_decision_meta[symbol] = {
                "reason_code": "HOLD_OPEN_POSITION",
                "reason_text": "Position already open; no new buy.",
                "feature_snapshot": feature_snapshot,
                "model_snapshot": {
                    "ml_prob_up": ml_proba,
                    "ml_threshold": self.settings.ml_prob_threshold,
                },
                "market_regime": market_regime,
                "adaptive_decision": None,
            }
            return ("HOLD", 0.0, self._snapshot_from_features(close, ema50, rsi, macd, macd_signal, symbol))

        trend_ok = st_dir >= 1 and close > ema200
        base_ml_ok = ml_proba is None or ml_proba >= self.settings.ml_prob_threshold
        adaptive_decision: Optional[AdaptiveDecision] = None
        effective_ml_threshold = self.settings.ml_prob_threshold
        if trend_ok and self.adaptive_engine and ml_proba is not None:
            adaptive_decision = self.adaptive_engine.evaluate_entry(
                symbol=symbol,
                feature_snapshot=feature_snapshot,
                market_regime=market_regime,
                ml_prob_up=ml_proba,
                trading_mode=self.settings.trading_mode,
            )
            if adaptive_decision is not None and adaptive_decision.should_apply:
                effective_ml_threshold = adaptive_decision.effective_threshold
        # DB-first volume_ratio adaptive (пакет adaptive/): только paper, флаг ADAPTIVE_USE_DB_VOLUME_LOOP
        volume_ratio_adaptive_details: Optional[Dict[str, Any]] = None
        db_volume_reason_code: Optional[str] = None
        if (
            os.getenv("ADAPTIVE_USE_DB_VOLUME_LOOP", "").strip().lower() in ("1", "true", "yes", "on")
            and self.settings.trading_mode == "paper"
        ):
            from adaptive.bot_hook import prepare_ml_buy_decision

            d = prepare_ml_buy_decision(
                ml_prob=ml_proba,
                feature_snapshot=feature_snapshot,
                trading_mode=self.settings.trading_mode,
                adaptive_mode=self.settings.adaptive_mode,
                base_ml_threshold=float(effective_ml_threshold),
            )
            volume_ratio_adaptive_details = d.get("details_adaptive")
            if not d["allow_buy"]:
                db_volume_reason_code = d.get("reason_code")
            elif d.get("effective_ml_threshold") is not None:
                effective_ml_threshold = float(d["effective_ml_threshold"])
        ml_ok = ml_proba is None or ml_proba >= effective_ml_threshold
        if db_volume_reason_code:
            ml_ok = False
        decision_label = "ALLOW_BUY" if trend_ok and ml_ok else "BLOCK_BUY"
        reason_code = "ENTRY_OK" if trend_ok and ml_ok else "BLOCK_ML" if trend_ok and not ml_ok else "BLOCK_TREND"
        reason_text = (
            "Trend and ML conditions passed."
            if trend_ok and ml_ok
            else "Trend passed, but ML probability below threshold."
            if trend_ok and not ml_ok
            else "Trend filter rejected entry."
        )
        if db_volume_reason_code and trend_ok and not ml_ok:
            reason_code = db_volume_reason_code
            ar = (volume_ratio_adaptive_details or {}).get("adaptive_reason") or ""
            reason_text = f"DB volume-ratio adaptive: {ar}".strip()
            decision_label = "BLOCK_BUY"
        if adaptive_decision is not None and adaptive_decision.changed_decision and adaptive_decision.should_apply:
            if trend_ok and base_ml_ok and not ml_ok and not db_volume_reason_code:
                reason_code = "BLOCK_ADAPTIVE_REGIME"
                reason_text = adaptive_decision.reason_text
                decision_label = "BLOCK_BUY"
            elif trend_ok and not base_ml_ok and ml_ok:
                reason_code = "ENTRY_OK"
                reason_text = "Trend and ML conditions passed. Adaptive regime filter applied."
                decision_label = "ALLOW_BUY"
        trend_bypass = False
        if reason_code == "BLOCK_TREND" and symbol.upper() in ():
            if ml_ok:
                LOGGER.info("BLOCK_TREND override applied for %s", symbol)
                reason_code = "ENTRY_OK"
                trend_bypass = True
                reason_text = "Trend filter bypassed for allowlisted ticker."
                decision_label = "ALLOW_BUY"
            else:
                reason_code = "BLOCK_TREND_BYPASS_ML_FAIL"
                reason_text = (
                    "Allowlisted ticker: trend bypass would apply, but ML probability below threshold."
                )
        model_snapshot = {
            "model_type": "classifier",
            "model_version": self.ml_model_version,
            "ml_prob_up": ml_proba,
            "ml_threshold": self.settings.ml_prob_threshold,
            "ml_threshold_effective": effective_ml_threshold,
            "trend_ok": bool(trend_ok),
            "ml_ok": bool(ml_ok),
            "adaptive_mode": self.settings.adaptive_mode,
            "adaptive_decision": asdict(adaptive_decision) if adaptive_decision is not None else None,
            "volume_ratio_adaptive": volume_ratio_adaptive_details,
        }
        self._last_decision_meta[symbol] = {
            "reason_code": reason_code,
            "reason_text": reason_text,
            "feature_snapshot": feature_snapshot,
            "model_snapshot": model_snapshot,
            "market_regime": market_regime,
            "adaptive_decision": asdict(adaptive_decision) if adaptive_decision is not None else None,
        }
        if self.ml_model is not None:
            allow_buy_ml = (trend_ok and ml_ok) or (trend_bypass and ml_ok)
            self._log_model_inference(
                symbol=symbol,
                input_features=feature_snapshot,
                raw_output={"p_up": ml_proba},
                decision_label=decision_label,
                confidence_score=self._safe_float_or_none(ml_proba),
                action_recommendation="BUY" if allow_buy_ml else "HOLD",
                model_used_in_final_decision=True,
            )

        if (trend_ok and ml_ok) or (trend_bypass and ml_ok):
            score = ml_proba if ml_proba is not None else 0.7
            return ("BUY", score, self._snapshot_from_features(close, ema50, rsi, macd, macd_signal, symbol))

        # Диагностика: почему не вошли. Помогает ловить "глухие" режимы без сигналов.
        LOGGER.info(
            "ML gate block %s: st_dir=%.2f close=%.4f ema200=%.4f trend_ok=%s p_up=%s threshold=%.3f ml_ok=%s",
            symbol,
            float(st_dir),
            float(close),
            float(ema200),
            trend_ok,
            "n/a" if ml_proba is None else f"{ml_proba:.4f}",
            effective_ml_threshold,
            ml_ok,
        )
        return ("HOLD", 0.0, self._snapshot_from_features(close, ema50, rsi, macd, macd_signal, symbol))

    def _snapshot_from_features(
        self, price: float, ema50: float, rsi: float, macd: float, macd_signal: float, symbol: str
    ) -> IndicatorSnapshot:
        return IndicatorSnapshot(
            symbol=symbol,
            price=price,
            ema20=ema50,
            ema50=ema50,
            rsi=rsi,
            adx=25.0,
            macd=macd,
            macd_signal=macd_signal,
            buy_count=0.0,
            sell_count=0.0,
            neutral_count=0.0,
            supertrend=0.0,
            supertrend_direction=1.0,
        )

    def _format_message(self, symbol: str, action: str, score: float, snapshot: IndicatorSnapshot) -> str:
        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
        action_ru = "ПОКУПКА" if action == "BUY" else "ПРОДАЖА" if action == "SELL" else "ОЖИДАНИЕ"
        return (
            f"{emoji} {symbol} | {action_ru}\n"
            f"💵 Цена: {snapshot.price:.4f}\n"
            f"📊 Скор: {score:.3f}\n"
            f"📈 EMA20/EMA50: {snapshot.ema20:.4f}/{snapshot.ema50:.4f}\n"
            f"📉 MACD: {snapshot.macd:.4f}/{snapshot.macd_signal:.4f}\n"
            f"🧭 RSI/ADX: {snapshot.rsi:.2f}/{snapshot.adx:.2f}\n"
            f"⚙️ Режим: {self.settings.trading_mode}"
        )

    def _format_entry_message(
        self,
        symbol: str,
        score: float,
        snapshot: IndicatorSnapshot,
        sl_price: float,
        tp_price: float,
        adaptive_note: Optional[str] = None,
    ) -> str:
        text = (
            f"🟢 {symbol} | ВХОД LONG\n"
            f"💵 Цена входа: {snapshot.price:.4f}\n"
            f"📊 Скор: {score:.3f}\n"
            f"🛑 Stop-loss: {sl_price:.4f}\n"
            f"🎯 Take-profit: {tp_price:.4f}\n"
            f"📈 EMA20/EMA50: {snapshot.ema20:.4f}/{snapshot.ema50:.4f}\n"
            f"📉 MACD: {snapshot.macd:.4f}/{snapshot.macd_signal:.4f}\n"
            f"🧭 RSI/ADX: {snapshot.rsi:.2f}/{snapshot.adx:.2f}\n"
            f"⚙️ Режим: {self.settings.trading_mode}"
        )
        if adaptive_note:
            text += f"\n🧩 {adaptive_note}"
        return text

    def _adaptive_note_from_meta(self, adaptive_meta: Optional[Dict[str, Any]]) -> Optional[str]:
        if not adaptive_meta:
            return None
        if adaptive_meta.get("mode") == "paper" and adaptive_meta.get("changed_decision"):
            return "Adaptive regime filter applied"
        if adaptive_meta.get("mode") == "shadow" and adaptive_meta.get("changed_decision"):
            return "Adaptive shadow recommendation logged"
        return None

    def _balance_lines_for_message(self, latest_prices: Dict[str, float]) -> str:
        if not isinstance(self.broker, PaperBroker):
            return ""
        snapshot = self.broker.get_balance_snapshot(latest_prices)
        total = safe_float(snapshot.get("total_pnl_rub", 0.0))
        return (
            f"\n💰 Баланс: {format_rub(snapshot.get('current_balance_rub', 0))}"
            f"\n📈 С последнего reset: {format_rub(total)}"
        )

    def _trade_result_lines(
        self,
        gross_pnl: float,
        gross_pnl_pct: float,
        commission_rub: float,
        net_pnl: float,
        net_pnl_pct: float,
    ) -> str:
        return (
            f"\n📊 До комиссии: {format_close_pnl_line(gross_pnl, gross_pnl_pct)}"
            f"\n💸 Комиссия: {format_signed_rub(-abs(commission_rub))}"
            f"\n📈 Итог net: {format_close_pnl_line(net_pnl, net_pnl_pct)}"
        )

    def _close_trade(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
        qty: float,
        avg_price: float,
        execution: Optional[Dict[str, Any]] = None,
        close_reason_code: str = "CLOSE_TRADE",
        extra_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._pm_states.pop(symbol, None)
        gross_pnl = safe_float(execution.get("gross_pnl")) if execution else (exit_price - avg_price) * qty
        gross_pnl_pct = (
            safe_float(execution.get("gross_pnl_pct"))
            if execution and execution.get("gross_pnl_pct") is not None
            else (((exit_price / avg_price) - 1.0) * 100 if avg_price > 0 else 0.0)
        )
        commission_rub = safe_float(execution.get("commission_rub")) if execution else 0.0
        net_pnl = safe_float(execution.get("net_pnl")) if execution else gross_pnl - commission_rub
        net_pnl_pct = (
            safe_float(execution.get("net_pnl_pct"))
            if execution and execution.get("net_pnl_pct") is not None
            else gross_pnl_pct
        )
        pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
        meta_before = self.trade_meta.get(symbol)
        signal_id, signal_id_missing_reason = resolve_buy_signal_id_at_close(
            self.analytics_db, symbol, meta_before
        )
        meta = self.trade_meta.pop(symbol, None)
        local_trade_id = meta.get("local_trade_id") if meta else None
        adaptive_note = meta.get("adaptive_note") if meta else None
        balance_lines = self._balance_lines_for_message({**self.last_prices, symbol: exit_price})
        balance_after_rub = None
        if isinstance(self.broker, PaperBroker):
            balance_after_rub = safe_float(
                self.broker.get_balance_snapshot({**self.last_prices, symbol: exit_price}).get("current_balance_rub", 0.0)
            )
        close_text = (
            f"🔔 {symbol} | СДЕЛКА ЗАКРЫТА\n"
            f"Причина: {reason}\n"
            f"💵 Вход: {avg_price:.4f}\n"
            f"💵 Выход: {exit_price:.4f}\n"
            f"📦 Объем: {qty:.6f}\n"
            f"{pnl_emoji} Результат сделки:"
        )
        close_text += self._trade_result_lines(gross_pnl, gross_pnl_pct, commission_rub, net_pnl, net_pnl_pct)
        if adaptive_note:
            close_text += f"\n🧩 {adaptive_note}"
        close_text += balance_lines
        self.notifier.send(close_text)
        if self.analytics_logger:
            try:
                close_details: Dict[str, Any] = {
                    "qty": qty,
                    "avg_price": avg_price,
                    "exit_price": exit_price,
                    "pnl": net_pnl,
                    "pnl_pct": net_pnl_pct,
                    "gross_pnl": gross_pnl,
                    "gross_pnl_pct": gross_pnl_pct,
                    "commission_rub": commission_rub,
                    "net_pnl": net_pnl,
                    "net_pnl_pct": net_pnl_pct,
                    "commission_rate": self.settings.paper_commission_rate if self.settings.paper_include_commission else 0.0,
                    "notional_rub": safe_float(execution.get("notional_rub")) if execution else qty * exit_price,
                    "entry_commission_allocated_rub": safe_float(execution.get("entry_commission_allocated_rub")) if execution else 0.0,
                    "exit_commission_rub": safe_float(execution.get("exit_commission_rub")) if execution else commission_rub,
                    "balance_after_rub": balance_after_rub,
                    **(extra_details or {}),
                }
                if signal_id is None and signal_id_missing_reason:
                    close_details["signal_id_missing_reason"] = signal_id_missing_reason
                self.analytics_logger.update_signal_status(signal_id, "CLOSED")
                self.analytics_logger.log_decision(
                    run_id=self.current_run_id,
                    signal_id=signal_id,
                    ticker=symbol,
                    decision_type="TRADE_CLOSE",
                    decision_label="SELL",
                    reason_code=close_reason_code,
                    reason_text=reason,
                    details=close_details,
                    decision_ts=self._iso_now(),
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Analytics close log failed for %s: %s", symbol, exc)
        if self.paper_mapper:
            try:
                self.paper_mapper.map_close(
                    signal_id=signal_id,
                    local_trade_id=local_trade_id,
                    ticker=symbol,
                    qty=qty,
                    price=exit_price,
                    comment=f"reason={reason}",
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Paper mapping close failed for %s: %s", symbol, exc)
        if meta and meta.get("entry_message_id"):
            closed_entry_text = (
                f"✅ {symbol} | ВХОД LONG (ЗАКРЫТА)\n"
                f"💵 Цена входа: {meta.get('entry_price', avg_price):.4f}\n"
                f"🛑 Stop-loss: {meta.get('sl_price', 0.0):.4f}\n"
                f"🎯 Take-profit: {meta.get('tp_price', 0.0):.4f}\n"
                f"💵 Цена выхода: {exit_price:.4f}\n"
                f"📦 Объем: {qty:.6f}\n"
                f"📌 Причина: {reason}\n"
                f"{pnl_emoji} Результат сделки:"
            )
            closed_entry_text += self._trade_result_lines(
                gross_pnl, gross_pnl_pct, commission_rub, net_pnl, net_pnl_pct
            )
            if adaptive_note:
                closed_entry_text += f"\n🧩 {adaptive_note}"
            closed_entry_text += balance_lines
            self.notifier.edit(meta.get("entry_message_id"), closed_entry_text)

    @staticmethod
    def _pm_trailing_exit_analytics_fingerprint(details: Dict[str, Any]) -> Tuple[Any, ...]:
        """Округлённые поля для подавления повторных одинаковых строк TRAILING_EXIT в analytics."""
        def rf(x: Any) -> Any:
            if x is None:
                return None
            try:
                return round(float(x), 6)
            except (TypeError, ValueError):
                return None

        return (
            bool(details.get("full_exit")),
            rf(details.get("close_fraction")),
            rf(details.get("price")),
            rf(details.get("peak")),
            rf(details.get("trough")),
            rf(details.get("giveback_abs")),
        )

    def _log_pm_event(
        self,
        symbol: str,
        reason_code: str,
        reason_text: str,
        details: Dict[str, Any],
        signal_id: Optional[str] = None,
    ) -> None:
        if not self.analytics_logger:
            return
        key = (symbol, signal_id or "")
        trailing_fp: Optional[Tuple[Any, ...]] = None
        if reason_code == "TRAILING_EXIT":
            trailing_fp = self._pm_trailing_exit_analytics_fingerprint(details)
            prev = self._pm_last_analytics_event.get(key)
            if prev and prev[0] == "TRAILING_EXIT" and prev[1] == trailing_fp:
                return
        try:
            self.analytics_logger.log_decision(
                run_id=self.current_run_id,
                signal_id=signal_id,
                ticker=symbol,
                decision_type="POSITION_MGMT",
                decision_label=reason_code,
                reason_code=reason_code,
                reason_text=reason_text,
                details=details,
                decision_ts=self._iso_now(),
            )
            if reason_code == "TRAILING_EXIT":
                self._pm_last_analytics_event[key] = (reason_code, trailing_fp)
            else:
                self._pm_last_analytics_event[key] = (reason_code, None)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("PM analytics log failed for %s: %s", symbol, exc)

    def _partial_close_trade(
        self,
        symbol: str,
        exit_price: float,
        qty_exit: float,
        reason_code: str,
        details: Dict[str, Any],
        execution: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Частичное закрытие long: trade_meta не удаляем."""
        qty, avg_price = self.broker.get_position(symbol)
        if qty_exit <= 0 or qty <= 0:
            return
        gross_pnl = safe_float(execution.get("gross_pnl")) if execution else (exit_price - avg_price) * qty_exit
        gross_pnl_pct = (
            safe_float(execution.get("gross_pnl_pct"))
            if execution and execution.get("gross_pnl_pct") is not None
            else (((exit_price / avg_price) - 1.0) * 100 if avg_price > 0 else 0.0)
        )
        commission_rub = safe_float(execution.get("commission_rub")) if execution else 0.0
        net_pnl = safe_float(execution.get("net_pnl")) if execution else gross_pnl - commission_rub
        net_pnl_pct = (
            safe_float(execution.get("net_pnl_pct"))
            if execution and execution.get("net_pnl_pct") is not None
            else gross_pnl_pct
        )
        pnl_emoji = "🟢" if net_pnl >= 0 else "🔴"
        balance_lines = self._balance_lines_for_message({**self.last_prices, symbol: exit_price})
        msg = (
            f"📊 {symbol} | ЧАСТИЧНЫЙ ВЫХОД ({reason_code})\n"
            f"💵 Средняя: {avg_price:.4f}  Выход: {exit_price:.4f}\n"
            f"📦 Объём: {qty_exit:.6f}\n"
            f"{pnl_emoji} Результат:"
        )
        msg += self._trade_result_lines(gross_pnl, gross_pnl_pct, commission_rub, net_pnl, net_pnl_pct)
        msg += balance_lines
        self.notifier.send(msg)
        m = self.trade_meta.get(symbol)
        signal_id, signal_id_missing_reason = resolve_buy_signal_id_at_close(
            self.analytics_db, symbol, m
        )
        local_trade_id = m.get("local_trade_id") if m else None
        if self.analytics_logger:
            try:
                part_details: Dict[str, Any] = {
                    **details,
                    "qty_exit": qty_exit,
                    "exit_price": exit_price,
                    "pnl": net_pnl,
                    "pnl_pct": net_pnl_pct,
                    "gross_pnl": gross_pnl,
                    "gross_pnl_pct": gross_pnl_pct,
                    "commission_rub": commission_rub,
                    "net_pnl": net_pnl,
                    "net_pnl_pct": net_pnl_pct,
                    "commission_rate": self.settings.paper_commission_rate if self.settings.paper_include_commission else 0.0,
                    "notional_rub": safe_float(execution.get("notional_rub")) if execution else qty_exit * exit_price,
                    "entry_commission_allocated_rub": safe_float(execution.get("entry_commission_allocated_rub")) if execution else 0.0,
                    "exit_commission_rub": safe_float(execution.get("exit_commission_rub")) if execution else commission_rub,
                }
                if signal_id is None and signal_id_missing_reason:
                    part_details["signal_id_missing_reason"] = signal_id_missing_reason
                self.analytics_logger.log_decision(
                    run_id=self.current_run_id,
                    signal_id=signal_id,
                    ticker=symbol,
                    decision_type="POSITION_MGMT",
                    decision_label="PARTIAL_EXIT",
                    reason_code=reason_code,
                    reason_text="Partial exit (position management)",
                    details=part_details,
                    decision_ts=self._iso_now(),
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("PM partial analytics failed for %s: %s", symbol, exc)
        if self.paper_mapper:
            try:
                self.paper_mapper.map_close(
                    signal_id=signal_id,
                    local_trade_id=local_trade_id,
                    ticker=symbol,
                    qty=qty_exit,
                    price=exit_price,
                    comment=f"pm_partial:{reason_code}",
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("PM partial paper map failed for %s: %s", symbol, exc)

    def _maybe_init_position_management(
        self,
        symbol: str,
        entry_price: float,
        qty: float,
        entry_bar_key: Optional[str],
    ) -> None:
        mode = self.settings.position_mgmt_mode
        if mode not in ("shadow", "paper") or self.settings.trading_mode != "paper":
            return
        df = load_candles(symbol, self.settings.exchange, self.settings.interval)
        st = build_initial_state(
            entry_price,
            qty,
            PositionSide.LONG,
            self._iso_now(),
            entry_bar_key or "",
            df,
            self.strategy.stop_loss_pct,
            self.strategy.take_profit_pct,
        )
        self._pm_states[symbol] = st
        prof = compute_instrument_profile(df)
        self._pm_profiles[symbol] = prof
        sig = self.trade_meta.get(symbol, {}).get("signal_id")
        self._log_pm_event(
            symbol,
            "POSITION_STATE_INIT",
            "Position management state initialized (long)",
            {
                "mode": mode,
                "hard_stop": st.hard_stop,
                "soft_stop": st.soft_stop,
                "tp_activation": st.tp_activation_level,
                "initial_stop": st.initial_stop,
                "bucket": st.ticker_volatility_bucket,
                "profile": st.regime_snapshot,
            },
            sig,
        )

    def _position_management_tick(self, symbol: str, price: float, bar_key: str) -> bool:
        """
        Возвращает True, если позиция полностью закрыта менеджером (paper).
        shadow: обновляет состояние и логирует, без ордеров.
        """
        st = self._pm_states.get(symbol)
        if st is None:
            return False
        df = load_candles(symbol, self.settings.exchange, self.settings.interval)
        profile = compute_instrument_profile(df)
        self._pm_profiles[symbol] = profile

        state, events = evaluate_tick(st, price, bar_key, profile)
        mode = self.settings.position_mgmt_mode
        signal_id = self.trade_meta.get(symbol, {}).get("signal_id")

        for ev in events:
            self._log_pm_event(
                symbol,
                ev.code,
                ev.message,
                {
                    **ev.details,
                    "mode": mode,
                    "price": price,
                    "full_exit": ev.full_exit,
                    "close_fraction": ev.close_fraction,
                },
                signal_id,
            )

        if mode == "shadow":
            self._pm_states[symbol] = state
            return False

        if mode != "paper":
            return False

        for ev in events:
            if ev.full_exit:
                qty, avg_price = self.broker.get_position(symbol)
                if qty <= 0:
                    self._pm_states.pop(symbol, None)
                    return True
                execution = self.broker.place_order(symbol, "SELL", qty * price, price)
                self._close_trade(symbol, price, f"PM:{ev.code}", qty, avg_price, execution=execution)
                self._last_close_bar_key[symbol] = bar_key
                return True
            if ev.close_fraction > 0:
                qty, avg_price = self.broker.get_position(symbol)
                if qty <= 0:
                    continue
                qty_close = qty * ev.close_fraction
                amount_rub = qty_close * price
                execution = self.broker.place_order(symbol, "SELL", amount_rub, price)
                self._partial_close_trade(symbol, price, qty_close, ev.code, ev.details, execution=execution)
                q_new, _ = self.broker.get_position(symbol)
                state.remaining_qty = q_new

        self._pm_states[symbol] = state
        q_left, _ = self.broker.get_position(symbol)
        if q_left <= 0:
            self._pm_states.pop(symbol, None)
            return True
        return False

    def _legacy_risk_exit(self, symbol: str, price: float) -> bool:
        """Классический фиксированный SL/TP от стратегии."""
        if not isinstance(self.broker, PaperBroker):
            return False
        qty, avg_price = self.broker.get_position(symbol)
        if qty <= 0 or avg_price <= 0:
            return False

        sl_price = avg_price * (1.0 - self.strategy.stop_loss_pct)
        tp_price = avg_price * (1.0 + self.strategy.take_profit_pct)

        if price <= sl_price:
            execution = self.broker.place_order(symbol, "SELL", qty * price, price)
            self._close_trade(symbol, price, "Stop-loss", qty, avg_price, execution=execution)
            self._last_close_bar_key[symbol] = _bar_key_at(datetime.now(timezone.utc), self.settings.interval)
            return True
        if price >= tp_price:
            execution = self.broker.place_order(symbol, "SELL", qty * price, price)
            self._close_trade(symbol, price, "Take-profit", qty, avg_price, execution=execution)
            self._last_close_bar_key[symbol] = _bar_key_at(datetime.now(timezone.utc), self.settings.interval)
            return True
        return False

    def get_status_text(self) -> str:
        performance = self.last_performance
        return (
            "🤖 Состояние бота\n"
            f"⚙️ Режим: {self.settings.trading_mode}\n"
            f"🔄 Цикл: {self.cycle}\n"
            f"🧺 Вселенная: {len(self.symbols)} тикеров\n"
            f"🎯 Пороги: BUY>={self.strategy.buy_threshold:.2f} / SELL<={self.strategy.sell_threshold:.2f}\n"
            f"💼 Капитал: {format_rub(performance.get('equity_rub', 0))}\n"
            f"💵 Кэш: {format_rub(performance.get('cash_rub', 0))}\n"
            f"🧾 Сделок: {performance.get('trade_count', 'n/a')}\n"
            f"📌 Открытых позиций: {performance.get('positions_count', 'n/a')}"
        )

    def get_strategy_text(self) -> str:
        base = (
            "🧠 Текущая стратегия\n"
            f"📌 Режим: {self.settings.strategy_mode}\n"
            f"🕒 Таймфрейм: {self.settings.interval}\n"
            f"🌍 Рынок: screener={self.settings.screener}, exchange={self.settings.exchange}\n"
        )
        adaptive_line = (
            f"🧩 Adaptive: {self.adaptive_engine.describe_policy()}\n"
            if self.adaptive_engine is not None
            else "🧩 Adaptive: off\n"
        )
        if self.settings.strategy_mode == "supertrend_ml":
            ml_status = "✅ загружена" if self.ml_model and getattr(self.ml_model, "_is_fitted", False) else "❌ не обучена"
            return (
                base
                + f"📐 Supertrend + ML: ST=направление, close>EMA200, P(up)>{self.settings.ml_prob_threshold}\n"
                + f"🤖 ML-модель: {ml_status}\n"
                + f"🛑 Stop-loss: {self.strategy.stop_loss_pct * 100:.2f}%\n"
                + f"🎯 Take-profit: {self.strategy.take_profit_pct * 100:.2f}%\n"
                + adaptive_line
            )
        return (
            base
            + "📊 Индикаторы: EMA20, EMA50, MACD, RSI, ADX, SuperTrend\n"
            + f"⚖️ Веса: EMA={self.strategy.ema_weight:.2f}, MACD={self.strategy.macd_weight:.2f}, "
            f"RSI={self.strategy.rsi_weight:.2f}, ADX={self.strategy.adx_weight:.2f}, "
            f"SuperTrend={self.strategy.supertrend_weight:.2f}\n"
            + f"🎯 Пороги: BUY>={self.strategy.buy_threshold:.2f}, SELL<={self.strategy.sell_threshold:.2f}\n"
            + f"📏 RSI-границы: перепроданность<={self.strategy.rsi_oversold:.1f}, "
            f"перекупленность>={self.strategy.rsi_overbought:.1f}\n"
            + f"🛑 Stop-loss: {self.strategy.stop_loss_pct * 100:.2f}%\n"
            + f"🎯 Take-profit: {self.strategy.take_profit_pct * 100:.2f}%\n"
            + f"🤖 AI-перенастройка: каждые {self.settings.openai_rebalance_cycles} циклов\n"
            + f"🛡️ AI-фильтр риска: {self.settings.enable_ai_risk_filter}\n"
            + adaptive_line.rstrip()
        )

    def get_pnl_text(self) -> str:
        if not isinstance(self.broker, PaperBroker):
            return "📈 Детальный P&L доступен только в paper-режиме."
        p = self.last_performance
        total = safe_float(p.get("total_pnl_rub", 0.0))
        realized = safe_float(p.get("realized_net_pnl_rub", p.get("realized_pnl_rub", 0.0)))
        commissions = safe_float(p.get("realized_commission_rub", 0.0))
        unrealized = safe_float(p.get("unrealized_pnl_rub", 0.0))
        total_emoji = "🟢" if total >= 0 else "🔴"
        return (
            "📈 Текущий P&L\n"
            f"{total_emoji} Общий P&L: {format_rub(total)}\n"
            f"✅ Реализованный net P&L: {format_rub(realized)}\n"
            f"💸 Комиссии: {format_rub(commissions)}\n"
            f"📌 Нереализованный P&L: {format_rub(unrealized)}\n"
            f"💼 Капитал: {format_rub(p.get('equity_rub', 0))}\n"
            f"💵 Кэш: {format_rub(p.get('cash_rub', 0))}"
        )

    def get_balance_text(self) -> str:
        if not isinstance(self.broker, PaperBroker):
            return "💰 Баланс доступен только в paper-режиме."
        snapshot = self.broker.get_balance_snapshot(self.last_prices)
        total = safe_float(snapshot.get("total_pnl_rub", 0.0))
        total_emoji = "🟢" if total >= 0 else "🔴"
        return (
            "💰 Виртуальный баланс\n"
            f"🏁 Старт: {format_rub(snapshot.get('initial_balance_rub', 0))}\n"
            f"💼 Баланс: {format_rub(snapshot.get('current_balance_rub', 0))}\n"
            f"📈 Realized gross: {format_rub(snapshot.get('realized_gross_pnl_rub', 0))}\n"
            f"💸 Комиссии: {format_rub(snapshot.get('realized_commission_rub', 0))}\n"
            f"✅ Realized net: {format_rub(snapshot.get('realized_net_pnl_rub', 0))}\n"
            f"📌 Нереализовано: {format_rub(snapshot.get('unrealized_pnl_rub', 0))}\n"
            f"🏦 Equity: {format_rub(snapshot.get('equity_rub', 0))}\n"
            f"{total_emoji} С последнего reset: {format_rub(total)}\n"
            f"🕒 Reset: {snapshot.get('last_reset_at', 'n/a')}"
        )

    def reset_balance(self, initial_balance_rub: Optional[float] = None) -> str:
        with self._operator_lock:
            if not isinstance(self.broker, PaperBroker):
                return "💰 Сброс баланса доступен только в paper-режиме."
            snapshot = self.broker.reset_balance(initial_balance_rub)
            self.trade_meta = {}
            self._pm_states = {}
            self._pm_profiles = {}
            self._last_buy_bar_key = {}
            self._last_close_bar_key = {}
            self.last_performance = snapshot
            return (
                "💰 Баланс сброшен.\n"
                f"🏁 Новый старт: {format_rub(snapshot.get('initial_balance_rub', 0))}\n"
                f"🕒 Reset: {snapshot.get('last_reset_at', 'n/a')}"
            )

    def set_balance(self, initial_balance_rub: float) -> str:
        with self._operator_lock:
            if not isinstance(self.broker, PaperBroker):
                return "💰 Установка баланса доступна только в paper-режиме."
            snapshot = self.broker.set_balance(initial_balance_rub)
            self.trade_meta = {}
            self._pm_states = {}
            self._pm_profiles = {}
            self._last_buy_bar_key = {}
            self._last_close_bar_key = {}
            self.last_performance = snapshot
            return (
                "💰 Стартовый баланс обновлён.\n"
                f"🏁 Новый старт: {format_rub(snapshot.get('initial_balance_rub', 0))}\n"
                f"🕒 Reset: {snapshot.get('last_reset_at', 'n/a')}"
            )

    def get_open_positions_text(self) -> str:
        if not isinstance(self.broker, PaperBroker):
            return "🧾 Открытые позиции доступны только в paper-режиме."
        positions = self.broker.get_open_positions(self.last_prices)
        if not positions:
            return "🧾 Открытые позиции\nСейчас активных позиций нет."
        header = f"🧾 Открытые позиции ({len(positions)})"
        lines = [header]
        for row in positions[:20]:
            pnl = safe_float(row.get("unrealized_pnl_rub", 0.0))
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(
                f"\n• {row['symbol']} | кол-во={row['qty']:.4f}\n"
                f"  средняя={row['avg_price']:.4f} текущая={row['last_price']:.4f}\n"
                f"  {pnl_emoji} нереализовано={format_rub(pnl)} | стоимость={format_rub(row['market_value_rub'])}"
            )
        if len(positions) > 20:
            lines.append(f"\n… и еще {len(positions) - 20} позиций")
        return "\n".join(lines)

    def can_close_all_positions(self) -> Tuple[bool, str]:
        if not isinstance(self.broker, PaperBroker) or self.settings.trading_mode != "paper":
            return False, "🛑 Массовое закрытие доступно только в paper-режиме."
        open_symbols = [symbol for symbol, qty in self.broker.positions.items() if safe_float(qty) > 0]
        if not open_symbols:
            return False, "🛑 Сейчас нет открытых paper-позиций. Закрывать нечего."
        return True, f"🛑 Будут закрыты текущие paper-позиции: {len(open_symbols)}"

    def close_all_positions(self) -> str:
        with self._operator_lock:
            try:
                return self._close_all_positions_inner()
            finally:
                self._manual_close_all_pending.clear()

    def _close_all_positions_inner(self) -> str:
        allowed, text = self.can_close_all_positions()
        if not allowed:
            return text

        symbols = [symbol for symbol, qty in self.broker.positions.items() if safe_float(qty) > 0]
        closed = 0
        errors: List[str] = []
        gross_total = 0.0
        commission_total = 0.0
        net_total = 0.0

        for symbol in list(symbols):
            qty_before, avg_before = self.broker.get_position(symbol)
            if qty_before <= 0:
                continue
            try:
                snapshot = self.tv.get_snapshot(symbol)
                price = safe_float(snapshot.price)
                if price <= 0:
                    raise RuntimeError("invalid market price")
                self.last_prices[symbol] = price
                execution = self.broker.place_order(
                    symbol=symbol,
                    side="SELL",
                    amount_rub=qty_before * price,
                    market_price=price,
                )
                if not execution:
                    raise RuntimeError("paper execution returned empty result")
                meta_open = self.trade_meta.get(symbol)
                open_signal_id, _ = resolve_buy_signal_id_at_close(self.analytics_db, symbol, meta_open)
                if self.analytics_logger:
                    self.analytics_logger.update_signal_status(open_signal_id, "CLOSING")
                self._close_trade(
                    symbol,
                    price,
                    "Manual close all",
                    qty_before,
                    avg_before,
                    execution=execution,
                    close_reason_code="MANUAL_CLOSE_ALL",
                    extra_details={
                        "manual_action": True,
                        "operator_action": "close_all_positions",
                    },
                )
                self._last_close_bar_key[symbol] = _bar_key_at(datetime.now(timezone.utc), self.settings.interval)
                closed += 1
                gross_total += safe_float(execution.get("gross_pnl"))
                commission_total += safe_float(execution.get("commission_rub"))
                net_total += safe_float(execution.get("net_pnl"))
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning("Manual close-all failed for %s: %s", symbol, exc)
                errors.append(f"{symbol}: {exc}")

        balance_snapshot = (
            self.broker.get_balance_snapshot(self.last_prices)
            if isinstance(self.broker, PaperBroker)
            else {"current_balance_rub": 0.0}
        )
        balance_before_reset = safe_float(balance_snapshot.get("current_balance_rub", 0.0))
        remaining_open = [sym for sym, qty in self.broker.positions.items() if safe_float(qty) > 0]
        all_closed_ok = len(errors) == 0 and len(remaining_open) == 0
        default_initial_rub = float(self.settings.paper_initial_balance_rub)
        reset_applied = False
        new_balance_after_reset: Optional[float] = None
        if all_closed_ok and isinstance(self.broker, PaperBroker):
            reset_snap = self.broker.reset_balance(None)
            self.trade_meta = {}
            self._pm_states = {}
            self._pm_profiles = {}
            self._last_buy_bar_key = {}
            self._last_close_bar_key = {}
            self.last_performance = reset_snap
            reset_applied = True
            new_balance_after_reset = safe_float(reset_snap.get("current_balance_rub", default_initial_rub))

        lines: List[str] = [
            "🛑 Закрытие всех позиций завершено",
            f"• Закрыто позиций: {closed} (было к закрытию: {len(symbols)})",
            f"• Суммарный gross: {format_signed_rub(gross_total)}",
            f"• Комиссия (всего): {format_rub(commission_total)}",
            f"• Net (сумма): {format_signed_rub(net_total)}",
        ]
        if reset_applied and new_balance_after_reset is not None:
            lines.extend(
                [
                    f"• Баланс до сброса: {format_rub(balance_before_reset)}",
                    "✅ Paper-баланс сброшен на значение по умолчанию "
                    f"(как /balance_reset без аргумента: {format_rub(default_initial_rub)}).",
                    f"• Новый баланс после сброса: {format_rub(new_balance_after_reset)}",
                ]
            )
        else:
            lines.append(
                f"• Текущий баланс (equity): {format_rub(balance_snapshot.get('current_balance_rub', 0))}"
            )
            if not all_closed_ok:
                lines.append(
                    "⚠️ Сброс баланса не выполнён: остались незакрытые позиции или были ошибки закрытия."
                )
        lines.append(f"• Ошибок закрытия: {len(errors)}")
        summary = "\n".join(lines)
        if errors:
            summary += "\n" + "\n".join([f"• {item}" for item in errors[:8]])
            if len(errors) > 8:
                summary += f"\n… и еще {len(errors) - 8}"
        if remaining_open and not reset_applied:
            summary += "\n• Не закрыто (позиции): " + ", ".join(remaining_open[:20])
            if len(remaining_open) > 20:
                summary += f" … +{len(remaining_open) - 20}"
        return summary

    def _check_risk_exit(self, symbol: str, price: float, bar_key: str) -> bool:
        if not isinstance(self.broker, PaperBroker):
            return False
        qty, avg_price = self.broker.get_position(symbol)
        if qty <= 0 or avg_price <= 0:
            return False

        pm_mode = self.settings.position_mgmt_mode
        if pm_mode in ("shadow", "paper") and symbol in self._pm_states:
            closed = self._position_management_tick(symbol, price, bar_key)
            if closed:
                return True
            if pm_mode == "paper":
                return False
            return self._legacy_risk_exit(symbol, price)

        return self._legacy_risk_exit(symbol, price)

    def run_once(self) -> None:
        with self._operator_lock:
            self.cycle += 1
            self.current_signals_found = 0
            self.current_instruments_checked = 0
            self.current_run_id = None
            run_status = "OK"
            run_comment = ""
            if self.analytics_logger:
                try:
                    self.current_run_id = self.analytics_logger.start_run(
                        strategy_code=self.settings.strategy_mode,
                        strategy_version=self.settings.analytics_strategy_version,
                        universe_size=len(self.symbols),
                        tradable_universe_size=len(self.symbols),
                        comment=f"cycle={self.cycle}",
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning("Failed to start analytics run: %s", exc)

            if self.cycle == 1 or self.cycle % self.settings.universe_refresh_cycles == 0:
                self.refresh_universe()
            if self.adaptive_engine is not None:
                try:
                    self.adaptive_engine.refresh(self.cycle)
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning("Adaptive refresh failed: %s", exc)

            latest_prices: Dict[str, float] = {}
            market_sample: List[Dict[str, Any]] = []

            for symbol in self.symbols:
                try:
                    if self.settings.strategy_mode == "supertrend_ml":
                        result = self._decide_supertrend_ml(symbol)
                        if result is None:
                            continue
                        action, score, snapshot = result
                    else:
                        snapshot = self.tv.get_snapshot(symbol)
                        score = self.scorer.score(snapshot, self.strategy)
                        action = self.decide(score)

                    self.current_instruments_checked += 1
                    if snapshot.price < self.settings.min_price or snapshot.price > self.settings.max_price:
                        if self.analytics_logger and self.settings.analytics_log_holds:
                            self.analytics_logger.log_decision(
                                run_id=self.current_run_id,
                                signal_id=None,
                                ticker=symbol,
                                decision_type="STRATEGY_DECISION",
                                decision_label="BLOCK",
                                reason_code="OUT_OF_PRICE_RANGE",
                                reason_text="Price outside configured min/max range.",
                                details={"price": snapshot.price},
                                decision_ts=self._iso_now(),
                            )
                        continue

                    bar_key_risk = _bar_key_at(datetime.now(timezone.utc), self.settings.interval)
                    risk_exit_triggered = self._check_risk_exit(symbol, snapshot.price, bar_key_risk)
                    if risk_exit_triggered:
                        LOGGER.info("Risk exit for %s", symbol)
                        self.last_prices[symbol] = snapshot.price
                        continue

                    latest_prices[symbol] = snapshot.price
                    self.last_prices[symbol] = snapshot.price
                    market_sample.append(
                        {
                            "symbol": symbol,
                            "price": snapshot.price,
                            "score": round(score, 4),
                            "rsi": round(snapshot.rsi, 2),
                            "adx": round(snapshot.adx, 2),
                        }
                    )

                    qty, _ = self.broker.get_position(symbol)
                    has_position = qty > 0
                    decision_meta = self._last_decision_meta.get(symbol, {})
                    feature_snapshot = decision_meta.get("feature_snapshot", {})
                    model_snapshot = decision_meta.get("model_snapshot", {})
                    reason_code = decision_meta.get("reason_code")
                    reason_text = decision_meta.get("reason_text")
                    market_regime = decision_meta.get("market_regime")

                    if has_position and action in ("BUY", "HOLD"):
                        if self.analytics_logger and self.settings.analytics_log_holds:
                            hold_signal_id = self.analytics_logger.log_signal(
                                run_id=self.current_run_id,
                                ticker=symbol,
                                strategy_code=self.settings.strategy_mode,
                                strategy_version=self.settings.analytics_strategy_version,
                                side="HOLD",
                                entry_price=snapshot.price,
                                stop_price=None,
                                target_price=None,
                                confidence_score=self._safe_float_or_none(score),
                                reason_code="HAS_OPEN_POSITION",
                                reason_text="Ticker already has open position.",
                                feature_snapshot=feature_snapshot,
                                model_snapshot=model_snapshot,
                                market_regime=market_regime,
                                execution_mode=self.settings.trading_mode,
                                status="HOLD",
                                signal_ts=self._iso_now(),
                            )
                            self.analytics_logger.log_decision(
                                run_id=self.current_run_id,
                                signal_id=hold_signal_id,
                                ticker=symbol,
                                decision_type="STRATEGY_DECISION",
                                decision_label="HOLD",
                                reason_code="HAS_OPEN_POSITION",
                                reason_text="Ticker already has open position.",
                                details={"action": action},
                                decision_ts=self._iso_now(),
                            )
                        continue

                    if not has_position and action == "SELL":
                        if self.analytics_logger and self.settings.analytics_log_holds:
                            block_signal_id = self.analytics_logger.log_signal(
                                run_id=self.current_run_id,
                                ticker=symbol,
                                strategy_code=self.settings.strategy_mode,
                                strategy_version=self.settings.analytics_strategy_version,
                                side="BLOCK",
                                entry_price=snapshot.price,
                                stop_price=None,
                                target_price=None,
                                confidence_score=self._safe_float_or_none(score),
                                reason_code="NO_OPEN_POSITION",
                                reason_text="Sell signal ignored because there is no open position.",
                                feature_snapshot=feature_snapshot,
                                model_snapshot=model_snapshot,
                                market_regime=market_regime,
                                execution_mode=self.settings.trading_mode,
                                status="BLOCK",
                                signal_ts=self._iso_now(),
                            )
                            self.analytics_logger.log_decision(
                                run_id=self.current_run_id,
                                signal_id=block_signal_id,
                                ticker=symbol,
                                decision_type="STRATEGY_DECISION",
                                decision_label="BLOCK",
                                reason_code="NO_OPEN_POSITION",
                                reason_text="Sell signal ignored because there is no open position.",
                                details={"action": action},
                                decision_ts=self._iso_now(),
                            )
                        continue

                    if action == "BUY" and not self.broker.can_afford_buy(self.settings.position_size_rub):
                        if self.analytics_logger:
                            block_signal_id = self.analytics_logger.log_signal(
                                run_id=self.current_run_id,
                                ticker=symbol,
                                strategy_code=self.settings.strategy_mode,
                                strategy_version=self.settings.analytics_strategy_version,
                                side="BLOCK",
                                entry_price=snapshot.price,
                                stop_price=None,
                                target_price=None,
                                confidence_score=self._safe_float_or_none(score),
                                reason_code="INSUFFICIENT_CASH",
                                reason_text="Cannot afford new buy by position sizing rule.",
                                feature_snapshot=feature_snapshot,
                                model_snapshot=model_snapshot,
                                market_regime=market_regime,
                                execution_mode=self.settings.trading_mode,
                                status="BLOCK",
                                signal_ts=self._iso_now(),
                            )
                            self.analytics_logger.log_decision(
                                run_id=self.current_run_id,
                                signal_id=block_signal_id,
                                ticker=symbol,
                                decision_type="STRATEGY_DECISION",
                                decision_label="BLOCK",
                                reason_code="INSUFFICIENT_CASH",
                                reason_text="Cannot afford new buy by position sizing rule.",
                                details={"required_rub": self.settings.position_size_rub},
                                decision_ts=self._iso_now(),
                            )
                        continue

                    if (
                        action == "BUY"
                        and self.settings.trading_mode == "paper"
                        and self._manual_close_all_pending.is_set()
                    ):
                        LOGGER.info("BUY skipped %s: awaiting Telegram close-all confirmation", symbol)
                        continue

                    bar_key_buy: Optional[str] = None
                    if action == "BUY":
                        bar_key_buy = _bar_key_at(datetime.now(timezone.utc), self.settings.interval)
                        if (
                            self._last_buy_bar_key.get(symbol) == bar_key_buy
                            or self._last_close_bar_key.get(symbol) == bar_key_buy
                        ):
                            LOGGER.info("ENTRY SKIPPED: already traded on this bar [%s]", symbol)
                            if self.analytics_logger:
                                self.analytics_logger.log_decision(
                                    run_id=self.current_run_id,
                                    signal_id=None,
                                    ticker=symbol,
                                    decision_type="STRATEGY_DECISION",
                                    decision_label="SKIP",
                                    reason_code="SAME_BAR_REENTRY",
                                    reason_text="ENTRY SKIPPED: already traded on this bar",
                                    details={
                                        "bar_key": bar_key_buy,
                                        "interval": self.settings.interval,
                                    },
                                    decision_ts=self._iso_now(),
                                )
                            continue

                    signal_id: Optional[str] = None
                    adaptive_meta = decision_meta.get("adaptive_decision") if decision_meta else None
                    if self.analytics_logger and (action in ("BUY", "SELL") or self.settings.analytics_log_holds):
                        stop_price = None
                        target_price = None
                        status = "OPEN" if action == "BUY" else "CLOSE_SIGNAL" if action == "SELL" else "HOLD"
                        if action == "BUY":
                            stop_price = snapshot.price * (1.0 - self.strategy.stop_loss_pct)
                            target_price = snapshot.price * (1.0 + self.strategy.take_profit_pct)
                        signal_id = self.analytics_logger.log_signal(
                            run_id=self.current_run_id,
                            ticker=symbol,
                            strategy_code=self.settings.strategy_mode,
                            strategy_version=self.settings.analytics_strategy_version,
                            side=action,
                            entry_price=snapshot.price,
                            stop_price=stop_price,
                            target_price=target_price,
                            confidence_score=self._safe_float_or_none(score),
                            reason_code=reason_code,
                            reason_text=reason_text,
                            feature_snapshot=feature_snapshot,
                            model_snapshot=model_snapshot,
                            market_regime=market_regime,
                            execution_mode=self.settings.trading_mode,
                            status=status,
                            signal_ts=self._iso_now(),
                        )
                        self.analytics_logger.log_decision(
                            run_id=self.current_run_id,
                            signal_id=signal_id,
                            ticker=symbol,
                            decision_type="STRATEGY_DECISION",
                            decision_label=action,
                            reason_code=reason_code,
                            reason_text=reason_text,
                            details={
                                "score": score,
                                "price": snapshot.price,
                                "strategy_mode": self.settings.strategy_mode,
                                "adaptive_mode": self.settings.adaptive_mode,
                                "adaptive_decision": adaptive_meta,
                            },
                            decision_ts=self._iso_now(),
                        )
                        if self.adaptive_engine is not None and adaptive_meta is not None:
                            try:
                                self.adaptive_engine.log_signal_adaptation(
                                    signal_id=signal_id,
                                    run_id=self.current_run_id,
                                    ticker=symbol,
                                    decision=AdaptiveDecision(**adaptive_meta),
                                )
                            except Exception as exc:  # pylint: disable=broad-except
                                LOGGER.warning("Adaptive signal log failed for %s: %s", symbol, exc)

                    LOGGER.info("Symbol=%s price=%.4f score=%.3f action=%s", symbol, snapshot.price, score, action)

                    if action == "BUY":
                        self.current_signals_found += 1
                        before_qty, _ = self.broker.get_position(symbol)
                        local_trade_id = str(uuid.uuid4())
                        execution = self.broker.place_order(
                            symbol=symbol,
                            side="BUY",
                            amount_rub=self.settings.position_size_rub,
                            market_price=snapshot.price,
                        )
                        after_qty, after_avg = self.broker.get_position(symbol)
                        if after_qty > before_qty:
                            if bar_key_buy is not None:
                                self._last_buy_bar_key[symbol] = bar_key_buy
                            sl_price = after_avg * (1.0 - self.strategy.stop_loss_pct)
                            tp_price = after_avg * (1.0 + self.strategy.take_profit_pct)
                            adaptive_note = self._adaptive_note_from_meta(adaptive_meta)
                            entry_message = self._format_entry_message(
                                symbol, score, snapshot, sl_price, tp_price, adaptive_note=adaptive_note
                            )
                            entry_message_id = self.notifier.send(entry_message)
                            self.trade_meta[symbol] = {
                                "entry_price": after_avg,
                                "entry_message_id": entry_message_id,
                                "sl_price": sl_price,
                                "tp_price": tp_price,
                                "signal_id": signal_id,
                                "local_trade_id": local_trade_id,
                                "entry_bar_key": bar_key_buy,
                                "adaptive_note": adaptive_note,
                            }
                            if self.analytics_logger:
                                self.analytics_logger.log_decision(
                                    run_id=self.current_run_id,
                                    signal_id=signal_id,
                                    ticker=symbol,
                                    decision_type="TRADE_OPEN",
                                    decision_label="BUY",
                                    reason_code="ORDER_FILLED",
                                    reason_text="Paper/live order opened position.",
                                    details={
                                        "qty_after": after_qty,
                                        "avg_price": after_avg,
                                        "gross_pnl": 0.0,
                                        "gross_pnl_pct": 0.0,
                                        "commission_rub": safe_float(execution.get("commission_rub")) if execution else 0.0,
                                        "net_pnl": safe_float(execution.get("net_pnl")) if execution else 0.0,
                                        "net_pnl_pct": safe_float(execution.get("net_pnl_pct")) if execution else 0.0,
                                        "commission_rate": self.settings.paper_commission_rate if self.settings.paper_include_commission else 0.0,
                                        "notional_rub": safe_float(execution.get("notional_rub")) if execution else snapshot.price * (after_qty - before_qty),
                                    },
                                    decision_ts=self._iso_now(),
                                )
                            if self.paper_mapper:
                                self.paper_mapper.map_open(
                                    signal_id=signal_id,
                                    local_trade_id=local_trade_id,
                                    ticker=symbol,
                                    qty=after_qty - before_qty,
                                    price=snapshot.price,
                                    comment="strategy_buy",
                                )
                            self._maybe_init_position_management(symbol, after_avg, after_qty, bar_key_buy)
                    elif action == "SELL":
                        qty_before, avg_before = self.broker.get_position(symbol)
                        if qty_before > 0:
                            bar_key_exit = _bar_key_at(datetime.now(timezone.utc), self.settings.interval)
                            entry_bar_key = self.trade_meta.get(symbol, {}).get("entry_bar_key")
                            if entry_bar_key is not None and entry_bar_key == bar_key_exit:
                                LOGGER.info("EXIT SKIPPED: same bar as entry [%s]", symbol)
                                continue
                        self.current_signals_found += 1
                        if qty_before > 0:
                            meta_open = self.trade_meta.get(symbol)
                            open_signal_id, _ = resolve_buy_signal_id_at_close(
                                self.analytics_db, symbol, meta_open
                            )
                            execution = self.broker.place_order(
                                symbol=symbol,
                                side="SELL",
                                amount_rub=qty_before * snapshot.price,
                                market_price=snapshot.price,
                            )
                            if self.analytics_logger:
                                self.analytics_logger.update_signal_status(open_signal_id, "CLOSING")
                            self._close_trade(
                                symbol, snapshot.price, "Сигнал стратегии", qty_before, avg_before, execution=execution
                            )
                            self._last_close_bar_key[symbol] = _bar_key_at(
                                datetime.now(timezone.utc), self.settings.interval
                            )
                    elif self.notifier.notify_hold:
                        self.notifier.send(self._format_message(symbol, action, score, snapshot))
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning("Failed for %s: %s", symbol, exc)
                    run_status = "ERROR"
                    run_comment = f"{symbol}:{exc}"
                time.sleep(self.settings.tv_request_delay_sec)

            valuation_prices = dict(self.last_prices)
            valuation_prices.update(latest_prices)
            performance = (
                self.broker.performance(valuation_prices)
                if isinstance(self.broker, PaperBroker)
                else {"mode": self.settings.trading_mode}
            )
            if isinstance(self.broker, PaperBroker):
                LOGGER.info("[PAPER] %s", performance)
            self.last_performance = performance

            new_strategy, ai_change_message = self.advisor.maybe_update(
                cycle=self.cycle,
                strategy=self.strategy,
                performance=performance,
                market_sample=market_sample,
            )
            self.strategy = new_strategy
            if ai_change_message:
                self.notifier.send(ai_change_message)
            if self.outcome_evaluator and self.settings.analytics_outcome_eval_enabled:
                try:
                    updated = self.outcome_evaluator.evaluate_pending(
                        load_candles_fn=lambda ticker: load_candles(
                            ticker, self.settings.exchange, self.settings.interval
                        ),
                        limit=200,
                    )
                    if updated:
                        LOGGER.info("Analytics outcomes evaluated: %d", updated)
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning("Outcome evaluator failed: %s", exc)
            if self.analytics_logger and self.current_run_id:
                try:
                    self.analytics_logger.finish_run(
                        run_id=self.current_run_id,
                        instruments_checked=self.current_instruments_checked,
                        signals_found=self.current_signals_found,
                        run_status=run_status,
                        comment=run_comment,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning("Failed to finish analytics run: %s", exc)


def main() -> None:
    load_dotenv()
    _setup_logging()
    settings = Settings.from_env()
    engine_ref: Dict[str, Optional[SignalEngine]] = {"engine": None}

    def status_provider() -> str:
        current = engine_ref["engine"]
        if current is None:
            return "⏳ Бот инициализируется, подождите..."
        return current.get_status_text()

    def strategy_provider() -> str:
        current = engine_ref["engine"]
        if current is None:
            return "⏳ Бот инициализируется, подождите..."
        return current.get_strategy_text()

    def pnl_provider() -> str:
        current = engine_ref["engine"]
        if current is None:
            return "⏳ Бот инициализируется, подождите..."
        return current.get_pnl_text()

    def positions_provider() -> str:
        current = engine_ref["engine"]
        if current is None:
            return "⏳ Бот инициализируется, подождите..."
        return current.get_open_positions_text()

    def balance_provider() -> str:
        current = engine_ref["engine"]
        if current is None:
            return "⏳ Бот инициализируется, подождите..."
        return current.get_balance_text()

    def balance_reset_handler(amount: Optional[float]) -> str:
        current = engine_ref["engine"]
        if current is None:
            return "⏳ Бот инициализируется, подождите..."
        return current.reset_balance(amount)

    def balance_set_handler(amount: float) -> str:
        current = engine_ref["engine"]
        if current is None:
            return "⏳ Бот инициализируется, подождите..."
        return current.set_balance(amount)

    def close_all_check_handler() -> Tuple[bool, str]:
        current = engine_ref["engine"]
        if current is None:
            return False, "⏳ Бот инициализируется, подождите..."
        return current.can_close_all_positions()

    def close_all_execute_handler() -> str:
        current = engine_ref["engine"]
        if current is None:
            return "⏳ Бот инициализируется, подождите..."
        return current.close_all_positions()

    def set_close_all_entry_pause(active: bool) -> None:
        current = engine_ref["engine"]
        if current is not None:
            current.set_manual_close_all_pending(active)

    command_server = TelegramCommandServer(
        settings,
        status_provider,
        strategy_provider,
        pnl_provider,
        balance_provider,
        positions_provider,
        balance_reset_handler,
        balance_set_handler,
        close_all_check_handler,
        close_all_execute_handler,
        set_close_all_entry_pause,
    )
    if command_server.enabled():
        command_server.start()
        LOGGER.info(
            "Telegram commands enabled (/start, /status, /strategy, /pnl, /balance, "
            "/balance_reset, /balance_set, /positions, /close_all_positions, /help)."
        )
    else:
        LOGGER.warning(
            "Telegram command polling disabled: TELEGRAM_BOT_TOKEN is empty. "
            "Проверьте .env в каталоге запуска (screen должен делать cd в корень проекта) и перезапустите бота."
        )

    engine = SignalEngine(settings)
    engine_ref["engine"] = engine
    LOGGER.info(
        "Bot started. mode=%s interval=%s symbols=%d",
        settings.trading_mode,
        settings.interval,
        len(engine.symbols),
    )
    if engine.notifier.enabled():
        engine.notifier.send(
            "✅ Бот запущен.\n"
            f"⚙️ Режим: {settings.trading_mode}\n"
            f"🕒 Таймфрейм: {settings.interval}\n"
            f"🧺 Вселенная: {len(engine.symbols)} тикеров\n"
            "⌨️ Для управления используйте кнопки клавиатуры в чате."
        )
    elif settings.telegram_bot_token.strip():
        LOGGER.info(
            "Стартовое сообщение в чат не отправлено: пустой TELEGRAM_CHAT_ID "
            "(уведомления отключены; команды боту в Telegram при этом работают, если polling запущен)."
        )

    while True:
        engine.run_once()
        time.sleep(settings.poll_seconds)


if __name__ == "__main__":
    main()
