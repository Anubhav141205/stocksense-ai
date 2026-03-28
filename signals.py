"""
Signal detection: multiple independent conditions on the latest bar.

Each condition may fire together with others. Point weights are defined in
``POINTS_BY_KIND``; use :func:`total_signal_score` for the combined score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import pandas as pd

# Thresholds
RSI_OVERSOLD_MAX = 30.0  # RSI < this → oversold (+2)
RSI_MOMENTUM_LOW = 30.0  # inclusive lower bound for early momentum band
RSI_MOMENTUM_HIGH = 50.0  # inclusive upper bound
VOLUME_VS_AVG_MULTIPLIER = 1.5

# Required columns on the indicator frame
_SIGNAL_COLUMNS = (
    "close",
    "ma20",
    "ma50",
    "rsi14",
    "volume",
    "volume_ma20",
)

# Points per ``Signal.kind`` (scanner / UI use the same weights)
POINTS_BY_KIND: dict[str, int] = {
    "trend_bullish": 2,  # MA20 > MA50
    "momentum_early": 1,  # 30 <= RSI <= 50
    "oversold_rsi": 2,  # RSI < 30
    "volume_elevated": 1,  # volume > 1.5x 20d avg
    "price_above_ma20": 1,  # close > MA20
}


@dataclass
class Signal:
    """
    One active rule on the latest bar.

    Attributes
    ----------
    kind
        Key into ``POINTS_BY_KIND`` (e.g. ``trend_bullish``).
    title
        Short label for comma-separated UI lists.
    detail
        One-line description with numbers where useful.
    metadata
        Optional structured values.
    """

    kind: str
    title: str
    detail: str
    metadata: dict = field(default_factory=dict)


def _signal_columns_ready(df: pd.DataFrame) -> bool:
    """Return True if all indicator columns exist on ``df``."""
    return all(c in df.columns for c in _SIGNAL_COLUMNS)


def _latest_complete_row(df: pd.DataFrame) -> pd.Series | None:
    """
    Return the last row where all signal columns are non-NaN, or ``None``.
    """
    if df is None or df.empty:
        return None
    if not _signal_columns_ready(df):
        return None
    clean = df.dropna(subset=list(_SIGNAL_COLUMNS), how="any")
    if clean.empty:
        return None
    return clean.iloc[-1]


def score_for_signal(signal: Signal) -> int:
    """
    Return the score contribution for a single signal.

    Parameters
    ----------
    signal
        Instance produced by :func:`detect_signals`.

    Returns
    -------
    int
        Points for ``signal.kind``, or 0 if unknown.
    """
    return POINTS_BY_KIND.get(signal.kind, 0)


def total_signal_score(signals: Sequence[Signal]) -> int:
    """
    Sum score contributions for all signals on the latest bar.

    Parameters
    ----------
    signals
        Output of :func:`detect_signals`.

    Returns
    -------
    int
        Total points; empty sequence → 0.
    """
    return sum(score_for_signal(s) for s in signals)


def detect_signals(df: pd.DataFrame) -> List[Signal]:
    """
    Evaluate all rules on the most recent row with complete indicator data.

    Multiple conditions can apply at once. RSI: ``< 30`` is oversold (+2);
    ``30 <= RSI <= 50`` is early momentum (+1) — mutually exclusive bands.

    Parameters
    ----------
    df
        OHLCV + columns from :func:`indicators.add_indicators`.

    Returns
    -------
    list of Signal
        All conditions that are true on that bar; may be empty.
    """
    signals: List[Signal] = []
    cur = _latest_complete_row(df)
    if cur is None:
        return signals

    close = float(cur["close"])
    ma20 = float(cur["ma20"])
    ma50 = float(cur["ma50"])
    rsi = float(cur["rsi14"])
    vol = float(cur["volume"])
    vol_ma = float(cur["volume_ma20"])

    if ma20 > ma50:
        signals.append(
            Signal(
                kind="trend_bullish",
                title="Trend bullish (MA20 > MA50)",
                detail=f"MA20 ({ma20:.2f}) above MA50 ({ma50:.2f}).",
                metadata={"ma20": ma20, "ma50": ma50},
            )
        )

    if rsi < RSI_OVERSOLD_MAX:
        signals.append(
            Signal(
                kind="oversold_rsi",
                title="Oversold (RSI < 30)",
                detail=f"RSI {rsi:.1f} is below {RSI_OVERSOLD_MAX:.0f}.",
                metadata={"rsi14": rsi},
            )
        )
    elif RSI_MOMENTUM_LOW <= rsi <= RSI_MOMENTUM_HIGH:
        signals.append(
            Signal(
                kind="momentum_early",
                title="Early momentum (RSI 30–50)",
                detail=f"RSI {rsi:.1f} is between {RSI_MOMENTUM_LOW:.0f} and {RSI_MOMENTUM_HIGH:.0f}.",
                metadata={"rsi14": rsi},
            )
        )

    if vol_ma > 0 and vol > VOLUME_VS_AVG_MULTIPLIER * vol_ma:
        ratio = vol / vol_ma
        signals.append(
            Signal(
                kind="volume_elevated",
                title="Volume > 1.5× average",
                detail=f"Volume {ratio:.2f}× the 20-day average.",
                metadata={"volume": vol, "volume_ma20": vol_ma, "ratio": ratio},
            )
        )

    if close > ma20:
        signals.append(
            Signal(
                kind="price_above_ma20",
                title="Price above MA20",
                detail=f"Close {close:.2f} above MA20 {ma20:.2f}.",
                metadata={"close": close, "ma20": ma20},
            )
        )

    return signals


def analyze_latest_bar(df: pd.DataFrame) -> tuple[int, List[Signal]]:
    """
    Convenience: detect signals and return ``(total_score, signals)``.

    Parameters
    ----------
    df
        Same as :func:`detect_signals`.

    Returns
    -------
    tuple[int, list of Signal]
        Total score and the list of active signals.
    """
    sigs = detect_signals(df)
    return total_signal_score(sigs), sigs
