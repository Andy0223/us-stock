from __future__ import annotations

import math
from typing import Any

import pandas as pd


def compute_price_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty or "Close" not in frame:
        return empty_metrics()

    prices = frame.copy().sort_index()
    close = pd.to_numeric(prices["Close"], errors="coerce").dropna()
    if close.empty:
        return empty_metrics()

    volume = pd.to_numeric(prices.get("Volume", pd.Series(index=prices.index, dtype=float)), errors="coerce").fillna(0.0)
    latest_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2]) if len(close) >= 2 else math.nan
    latest_volume = float(volume.iloc[-1]) if len(volume) else 0.0
    avg_volume_20 = float(volume.tail(20).mean()) if len(volume) else 0.0
    ma20 = rolling_last(close, 20)
    ma50 = rolling_last(close, 50)
    ma200 = rolling_last(close, 200)
    high = pd.to_numeric(prices.get("High", close), errors="coerce").fillna(close)
    low = pd.to_numeric(prices.get("Low", close), errors="coerce").fillna(close)
    high_20 = float(high.tail(20).max()) if len(high) else math.nan
    low_20 = float(low.tail(20).min()) if len(low) else math.nan
    atr14 = average_true_range(prices, 14)
    rsi14 = relative_strength_index(close, 14)

    return {
        "latest_date": close.index[-1].date().isoformat(),
        "close": latest_close,
        "previous_close": previous_close,
        "return_1d": pct_change(close, 1),
        "return_5d": pct_change(close, 5),
        "return_20d": pct_change(close, 20),
        "return_60d": pct_change(close, 60),
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "pct_from_ma20": pct_from(latest_close, ma20),
        "pct_from_ma50": pct_from(latest_close, ma50),
        "pct_from_ma200": pct_from(latest_close, ma200),
        "rsi14": rsi14,
        "atr14": atr14,
        "atr_pct": safe_div(atr14, latest_close),
        "high_20d": high_20,
        "low_20d": low_20,
        "drawdown_from_20d_high": safe_div(latest_close, high_20) - 1 if is_finite(high_20) else math.nan,
        "pct_above_20d_low": safe_div(latest_close, low_20) - 1 if is_finite(low_20) else math.nan,
        "latest_volume": latest_volume,
        "avg_volume_20d": avg_volume_20,
        "volume_ratio_20d": safe_div(latest_volume, avg_volume_20),
        "avg_dollar_volume_20d": avg_volume_20 * latest_close,
    }


def empty_metrics() -> dict[str, Any]:
    keys = [
        "latest_date",
        "close",
        "previous_close",
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
        "ma20",
        "ma50",
        "ma200",
        "pct_from_ma20",
        "pct_from_ma50",
        "pct_from_ma200",
        "rsi14",
        "atr14",
        "atr_pct",
        "high_20d",
        "low_20d",
        "drawdown_from_20d_high",
        "pct_above_20d_low",
        "latest_volume",
        "avg_volume_20d",
        "volume_ratio_20d",
        "avg_dollar_volume_20d",
    ]
    return {key: math.nan for key in keys}


def pct_change(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return math.nan
    base = float(close.iloc[-periods - 1])
    latest = float(close.iloc[-1])
    return safe_div(latest, base) - 1


def pct_from(value: float, reference: float) -> float:
    return safe_div(value, reference) - 1 if is_finite(value) and is_finite(reference) else math.nan


def rolling_last(series: pd.Series, window: int) -> float:
    if len(series) < window:
        return math.nan
    return float(series.rolling(window).mean().iloc[-1])


def average_true_range(frame: pd.DataFrame, window: int = 14) -> float:
    if frame.empty or len(frame) < window + 1:
        return math.nan
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(true_range.rolling(window).mean().iloc[-1])


def relative_strength_index(close: pd.Series, window: int = 14) -> float:
    if len(close) < window + 1:
        return math.nan
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[-1]
    return float(value) if pd.notna(value) else math.nan


def safe_div(numerator: float, denominator: float) -> float:
    if not is_finite(numerator) or not is_finite(denominator) or denominator == 0:
        return math.nan
    return float(numerator) / float(denominator)


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
