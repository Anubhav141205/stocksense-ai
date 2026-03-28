"""
Stock data fetching via yfinance.

Normalizes OHLCV columns and maps provider failures to :class:`StockDataError`
so callers (including the UI) can show clear messages for invalid symbols or
empty history.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Yahoo tickers: alphanumerics, dots, hyphens; ^ for indices, = for some instruments
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.\-^=]{1,24}$", re.IGNORECASE)


class StockDataError(Exception):
    """
    Raised when price history cannot be loaded.

    Examples include unknown tickers, delisted symbols, network failures, or
    responses that do not contain the expected OHLCV columns.
    """


def normalize_symbol(symbol: str) -> str:
    """
    Trim whitespace and convert a ticker to uppercase for Yahoo Finance.

    Parameters
    ----------
    symbol
        Raw user or config input (e.g. ``"  aapl  "``).

    Returns
    -------
    str
        Normalized symbol (e.g. ``"AAPL"``).

    Raises
    ------
    ValueError
        If ``symbol`` is empty after stripping, or fails basic format checks.
    """
    if symbol is None:
        raise ValueError("Symbol cannot be empty.")
    cleaned = str(symbol).strip().upper()
    if not cleaned:
        raise ValueError("Symbol cannot be empty.")
    if not _SYMBOL_PATTERN.fullmatch(cleaned):
        raise ValueError(
            f"Invalid ticker format: '{cleaned}'. Use letters, numbers, and common symbols "
            "like . - ^ = (e.g. AAPL, BRK-B, ^GSPC)."
        )
    return cleaned


def fetch_stock_data(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download daily (or other) OHLCV history for a ticker.

    Parameters
    ----------
    symbol
        Stock ticker; normalized via :func:`normalize_symbol`.
    period
        yfinance lookback: ``1d``, ``5d``, ``1mo``, ``3mo``, ``6mo``, ``1y``, etc.
    interval
        Bar size; default ``1d`` for daily bars.

    Returns
    -------
    pandas.DataFrame
        Index is datetime; columns include ``open``, ``high``, ``low``,
        ``close``, ``volume`` (lowercase).

    Raises
    ------
    ValueError
        If the symbol is empty or fails :func:`normalize_symbol`.
    StockDataError
        If Yahoo returns no rows, missing columns, or the request fails.
    """
    ticker = normalize_symbol(symbol)

    try:
        stock = yf.Ticker(ticker)
        # Empty DataFrame for unknown tickers, delisted symbols, or bad period/interval
        df = stock.history(period=period, interval=interval, auto_adjust=True)
    except Exception as exc:
        logger.exception("yfinance request failed for %s", ticker)
        raise StockDataError(
            f"Could not reach data service for '{ticker}'. Check your connection and try again."
        ) from exc

    if df is None or df.empty:
        raise StockDataError(
            f"No price data for '{ticker}'. The symbol may be invalid, delisted, "
            "or have no history for this period — try another ticker or a longer window."
        )

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise StockDataError(
            f"Unexpected data shape for '{ticker}': missing columns {sorted(missing)}."
        )

    return df


def validate_ticker_quick(symbol: str) -> Optional[str]:
    """
    Validate user input before calling :func:`fetch_stock_data`.

    Performs only cheap checks (no network). Full validation still happens
    when history is downloaded.

    Parameters
    ----------
    symbol
        Raw ticker string from a form or CLI.

    Returns
    -------
    str or None
        An error message suitable for display, or ``None`` if input looks OK.
    """
    if symbol is None or not str(symbol).strip():
        return "Please enter a ticker symbol."
    try:
        normalize_symbol(symbol)
    except ValueError as exc:
        return str(exc)
    return None
