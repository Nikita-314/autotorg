"""
Загрузка OHLCV-данных. MOEX ISS API и yfinance.
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import requests
import yfinance as yf


def load_moex_candles(
    symbol: str,
    interval: str = "1h",
    days_back: int = 90,
) -> Optional[pd.DataFrame]:
    """Загрузка свечей с MOEX ISS. Возвращает DataFrame с open, high, low, close, volume."""
    interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 60, "1d": 24, "1w": 24}
    moex_interval = interval_map.get(interval, 60)
    till = datetime.utcnow()
    from_dt = till - timedelta(days=days_back)
    from_str = from_dt.strftime("%Y-%m-%d")
    till_str = till.strftime("%Y-%m-%d")
    base_url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{symbol}/candles.json"
        f"?from={from_str}&till={till_str}&interval={moex_interval}"
    )
    try:
        # MOEX ISS returns at most 500 rows per request; without pagination only the oldest
        # chunk of [from..till] is loaded, so max(df.index) lags far behind "now".
        rows: list = []
        cols = None
        start = 0
        page_size = 500
        while True:
            resp = requests.get(f"{base_url}&start={start}", timeout=15)
            resp.raise_for_status()
            data = resp.json()
            candles = data.get("candles", {})
            chunk = candles.get("data", [])
            if cols is None:
                cols = candles.get("columns", [])
            if not chunk:
                break
            rows.extend(chunk)
            if len(chunk) < page_size:
                break
            start += page_size
        if not rows or not cols:
            return None
        idx = {c: i for i, c in enumerate(cols)}
        vol_col = "volume" if "volume" in idx else "value" if "value" in idx else None
        df = pd.DataFrame(
            {
                "open": [float(r[idx["open"]]) for r in rows],
                "high": [float(r[idx["high"]]) for r in rows],
                "low": [float(r[idx["low"]]) for r in rows],
                "close": [float(r[idx["close"]]) for r in rows],
                "volume": [float(r[idx[vol_col]]) for r in rows] if vol_col else [0.0] * len(rows),
            }
        )
        if "begin" in idx:
            df["datetime"] = pd.to_datetime([r[idx["begin"]] for r in rows])
            df = df.set_index("datetime")
        df = df.sort_index()
        return df
    except Exception:  # pylint: disable=broad-except
        return None


def load_yf_candles(
    symbol: str,
    interval: str = "1h",
    period: str = "3mo",
) -> Optional[pd.DataFrame]:
    """Загрузка свечей через yfinance."""
    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "1h", "1d": "1d", "1w": "1wk"}
    yf_interval = interval_map.get(interval, "1h")
    try:
        data = yf.download(symbol, period=period, interval=yf_interval, progress=False, auto_adjust=True)
        if data is None or data.empty or len(data) < 50:
            return None
        df = data[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["open", "high", "low", "close"]
        if "Volume" in data.columns:
            df["volume"] = data["Volume"].values
        else:
            df["volume"] = 0.0
        return df
    except Exception:  # pylint: disable=broad-except
        return None


def load_candles(
    symbol: str,
    exchange: str,
    interval: str = "1h",
) -> Optional[pd.DataFrame]:
    """Универсальная загрузка: MOEX или yfinance."""
    if exchange.upper() == "MOEX":
        return load_moex_candles(symbol, interval, days_back=90)
    return load_yf_candles(symbol, interval, period="3mo")
