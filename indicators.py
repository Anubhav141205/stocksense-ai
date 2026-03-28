"""
Technical indicators: moving averages, RSI, and volume average.

Expects a DataFrame produced by :func:`data.fetch_stock_data` (lowercase OHLCV).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Columns required before computing derived series
_OHLCV_COLS = frozenset({"open", "high", "low", "close", "volume"})


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index using Wilder's smoothing.

    Formula: RSI = 100 - (100 / (1 + RS)), where RS is the ratio of average
    gain to average loss over ``period`` bars.

    Parameters
    ----------
    close
        Closing prices (same index as the source frame).
    period
        Lookback length; 14 is the common default.

    Returns
    -------
    pandas.Series
        RSI values aligned with ``close``; NaN until ``period`` bars are available.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _require_ohlcv(df: pd.DataFrame) -> None:
    """
    Ensure the frame has the minimum columns needed for indicators.

    Parameters
    ----------
    df
        Candidate price DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    miss = sorted(_OHLCV_COLS - set(df.columns))
    if miss:
        raise ValueError(f"DataFrame is missing required columns: {miss}")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add MA20, MA50, RSI(14), and 20-day volume average.

    Appends columns: ``ma20``, ``ma50``, ``rsi14``, ``volume_ma20``.

    Parameters
    ----------
    df
        OHLCV history; must be non-empty and include open/high/low/close/volume.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with indicator columns added.

    Raises
    ------
    ValueError
        If ``df`` is None, empty, or missing OHLCV columns.
    """
    if df is None:
        raise ValueError("DataFrame is None.")
    if df.empty:
        raise ValueError("Cannot compute indicators on an empty DataFrame.")
    _require_ohlcv(df)

    out = df.copy()
    close = out["close"]
    vol = out["volume"]

    out["ma20"] = close.rolling(window=20, min_periods=20).mean()
    out["ma50"] = close.rolling(window=50, min_periods=50).mean()
    out["rsi14"] = _rsi_wilder(close, period=14)
    out["volume_ma20"] = vol.rolling(window=20, min_periods=20).mean()

    return out


def latest_values(df: pd.DataFrame) -> dict[str, Any]:
    """
    Extract the last row's price and indicator values for UI metrics.

    Parameters
    ----------
    df
        Frame that may include columns from :func:`add_indicators`. May be empty.

    Returns
    -------
    dict
        Keys: ``close``, ``ma20``, ``ma50``, ``rsi14``, ``volume``, ``volume_ma20``.
        Values are ``float`` or ``None`` if the column is missing or the last value is NaN.
    """
    keys = ("close", "ma20", "ma50", "rsi14", "volume", "volume_ma20")
    if df is None or df.empty:
        return {k: None for k in keys}

    last = df.iloc[-1]
    out: dict[str, Any] = {}
    for k in keys:
        if k not in df.columns:
            out[k] = None
            continue
        val = last.get(k)
        out[k] = float(val) if pd.notna(val) else None
    return out
