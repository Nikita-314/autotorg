"""
Признаки для ML: ATR, Supertrend, EMA, RSI, MACD, returns, volatility.
Supertrend как фильтр направления + признаки для модели.
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR (Average True Range)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """SuperTrend line и direction. direction: 1=бычий, -1=медвежий."""
    atr = compute_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)
    sup_prev = np.nan
    dir_prev = 1.0
    for i in range(period, len(close)):
        c = close.iloc[i]
        u, l = upper.iloc[i], lower.iloc[i]
        if np.isnan(sup_prev):
            sup = l
            dir_val = 1.0
        else:
            if dir_prev == 1.0:
                if l > sup_prev:
                    sup, dir_val = l, 1.0
                elif c < sup_prev:
                    sup, dir_val = u, -1.0
                else:
                    sup, dir_val = sup_prev, 1.0
            else:
                if u < sup_prev:
                    sup, dir_val = u, -1.0
                elif c > sup_prev:
                    sup, dir_val = l, 1.0
                else:
                    sup, dir_val = sup_prev, -1.0
        supertrend.iloc[i] = sup
        direction.iloc[i] = dir_val
        sup_prev, dir_prev = sup, dir_val
    return supertrend, direction


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI 0-100."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal, histogram."""
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - macd_signal
    return macd_line, macd_signal, histogram


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит полный набор признаков для ML.
    Без утечки будущего: все индикаторы считаются только по прошлым данным.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)

    # ATR
    atr = compute_atr(high, low, close, 14)

    # SuperTrend
    st_line, st_dir = compute_supertrend(high, low, close, 10, 3.0)
    dist_to_st = ((close - st_line) / atr.replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)
    close_above_st = (close > st_line).astype(float)

    # EMA
    ema50 = compute_ema(close, 50)
    ema200 = compute_ema(close, 200)
    ema_dist_50 = ((close - ema50) / close.replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)
    ema_dist_200 = ((close - ema200) / close.replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)

    # RSI
    rsi = compute_rsi(close, 14)

    # MACD
    macd_line, macd_signal, macd_hist = compute_macd(close, 12, 26, 9)

    # Returns
    ret_1 = close.pct_change(1)
    ret_3 = close.pct_change(3)
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)

    # Volatility (rolling std of returns)
    vol = ret_1.rolling(20).std()

    # Volume
    vol_ma = volume.rolling(20).mean().replace(0, np.nan)
    vol_ratio = (volume / vol_ma).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    # Bars since SuperTrend flip (упрощённо)
    st_flip = st_dir.diff().abs()
    bars_since_flip = st_flip.groupby((st_flip != 0).cumsum()).cumcount()

    # Time features
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        dow = df.index.dayofweek
    else:
        hour = pd.Series(12, index=df.index)
        dow = pd.Series(2, index=df.index)

    atr_pct = (atr / close.replace(0, np.nan)).fillna(0).replace([np.inf, -np.inf], 0)
    features = pd.DataFrame({
        "atr": atr,
        "atr_pct": atr_pct,
        "st_dir": st_dir,
        "st_line": st_line,
        "dist_to_st": dist_to_st,
        "close_above_st": close_above_st,
        "ema50": ema50,
        "ema200": ema200,
        "ema_dist_50": ema_dist_50,
        "ema_dist_200": ema_dist_200,
        "rsi": rsi,
        "macd": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "volatility": vol.fillna(0).replace([np.inf, -np.inf], 0),
        "vol_ratio": vol_ratio,
        "close": close,
        "hour": hour,
        "dow": dow,
    }, index=df.index)
    return features


def make_target(
    close: pd.Series,
    forward_bars: int = 10,
    threshold_pct: float = 0.008,
) -> pd.Series:
    """
    Target для классификации: 1 если доходность через N свечей > threshold, иначе 0.
    Без утечки: target[i] основан на close[i+forward_bars].
    """
    future_return = close.shift(-forward_bars) / close - 1.0
    target = (future_return > threshold_pct).astype(int)
    return target


def get_latest_signal(
    features: pd.DataFrame,
    target_col: str = "target",
) -> Optional[dict]:
    """
    Извлекает последнюю строку признаков для live-прогноза.
    Убирает target (его нет в live).
    """
    if features.empty or len(features) < 50:
        return None
    last = features.iloc[-1]
    exclude = {"target", "close"}
    return {k: float(v) for k, v in last.items() if k not in exclude and pd.notna(v) and np.isfinite(v)}
