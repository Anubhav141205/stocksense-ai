"""
Multi-symbol scan: fetch OHLCV, compute indicators, detect signals, and score rows.

Scoring uses the same weights as ``signals.POINTS_BY_KIND`` via :func:`get_score`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Sequence

import pandas as pd

from ai import generate_stock_insight
from data import StockDataError, fetch_stock_data
from indicators import add_indicators, latest_values
from signals import Signal, detect_signals, score_for_signal, total_signal_score

logger = logging.getLogger(__name__)


def recommendation_for_score(score: int) -> str:
    """
    Map total signal score to a coarse action label (not financial advice).

    Parameters
    ----------
    score
        Sum of rule weights from :func:`signals.total_signal_score`.

    Returns
    -------
    str
        ``BUY`` if score >= 4, ``WATCH`` if 2–3, ``AVOID`` if 0–1.
    """
    if score >= 4:
        return "BUY"
    if score >= 2:
        return "WATCH"
    return "AVOID"


@dataclass
class ScanRow:
    """
    One symbol's scan outcome for tabular display.

    Attributes
    ----------
    symbol
        Yahoo ticker (e.g. ``TCS.NS``).
    score
        Sum of :func:`get_score` over all detected signals; 0 if none or on failure.
    signal_kinds
        Stable ``Signal.kind`` values that fired.
    signal_summary
        Comma-separated signal titles for the UI.
    close
        Last close price when the pipeline succeeded; ``None`` on failure.
    error
        Error message when fetch or indicators failed; ``None`` on success.
    """

    symbol: str
    score: int
    signal_kinds: List[str] = field(default_factory=list)
    signal_summary: str = ""
    close: float | None = None
    error: str | None = None


def get_score(signal: Signal) -> int:
    """
    Return the opportunity score for one detected signal.

    Parameters
    ----------
    signal
        Instance from :func:`signals.detect_signals`.

    Returns
    -------
    int
        Matches ``signals.POINTS_BY_KIND`` (e.g. trend bullish 2, oversold 2).
    """
    return score_for_signal(signal)


def score_signals(signals: Sequence[Signal]) -> int:
    """
    Total score for a sequence of signals (same as :func:`signals.total_signal_score`).

    Parameters
    ----------
    signals
        Output of :func:`signals.detect_signals`.

    Returns
    -------
    int
        Sum of per-signal points.
    """
    return total_signal_score(signals)


def _signal_labels(signals: Sequence[Signal]) -> str:
    """Comma-separated titles for table cells."""
    if not signals:
        return "No signal"
    return ", ".join(s.title for s in signals)


def scan_symbol(symbol: str, period: str = "6mo") -> ScanRow:
    """
    Run data → indicators → signals → total score for one ticker.

    Parameters
    ----------
    symbol
        Yahoo symbol (e.g. ``TCS.NS``).
    period
        yfinance history window.

    Returns
    -------
    ScanRow
        ``score`` is the sum of all active signal weights; ``signal_summary`` is comma-separated.
    """
    sym = symbol.strip().upper()
    try:
        raw = fetch_stock_data(sym, period=period)
        df = add_indicators(raw)
        sigs = detect_signals(df)
        sc = total_signal_score(sigs)
        lv = latest_values(df)
        return ScanRow(
            symbol=sym,
            score=sc,
            signal_kinds=[s.kind for s in sigs],
            signal_summary=_signal_labels(sigs),
            close=lv.get("close"),
            error=None,
        )
    except StockDataError as exc:
        logger.warning("Scan failed for %s: %s", sym, exc)
        return ScanRow(
            symbol=sym,
            score=0,
            signal_summary="No signal",
            error=str(exc),
        )
    except ValueError as exc:
        logger.warning("Scan validation failed for %s: %s", sym, exc)
        return ScanRow(
            symbol=sym,
            score=0,
            signal_summary="No signal",
            error=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected scan error for %s", sym)
        return ScanRow(
            symbol=sym,
            score=0,
            signal_summary="No signal",
            error=f"Unexpected: {exc}",
        )


def scan_universe(symbols: Sequence[str], period: str = "6mo") -> list[ScanRow]:
    """
    Scan each symbol; sort by total score descending, then symbol A–Z.

    Parameters
    ----------
    symbols
        Yahoo tickers.
    period
        yfinance period for each :func:`scan_symbol`.

    Returns
    -------
    list of ScanRow
        Sorted by ``score`` (desc).
    """
    rows: list[ScanRow] = [scan_symbol(sym, period=period) for sym in symbols]
    rows.sort(key=lambda r: (-r.score, r.symbol))
    return rows


def opportunities_dataframe(rows: Sequence[ScanRow]) -> pd.DataFrame:
    """
    Build ``Stock``, ``Signal``, ``Score``, ``Action``, ``AI Insight``.

    Sorted by score descending. Insights use OpenAI when configured, else a plain fallback.

    Parameters
    ----------
    rows
        From :func:`scan_universe` (any order; re-sorted here).

    Returns
    -------
    pandas.DataFrame
    """
    records: list[dict[str, Any]] = []
    for r in rows:
        if r.error:
            signal_cell = f"Unavailable — {r.error}"
        else:
            signal_cell = r.signal_summary
        sc = int(r.score)
        act = recommendation_for_score(sc)
        insight = generate_stock_insight(
            symbol=r.symbol,
            signal_summary=r.signal_summary if not r.error else "",
            score=sc,
            action=act,
            load_error=bool(r.error),
        )
        records.append(
            {
                "Stock": r.symbol,
                "Signal": signal_cell,
                "Score": sc,
                "Action": act,
                "AI Insight": insight,
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


def top_opportunities(rows: Sequence[ScanRow], n: int = 5) -> list[ScanRow]:
    """First ``n`` rows after universe sort (highest scores first)."""
    if n <= 0:
        return []
    return list(rows[:n])
