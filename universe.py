"""
Predefined equity universes for batch scans (Yahoo Finance tickers).

NSE symbols use the ``.NS`` suffix on Yahoo Finance (National Stock Exchange of India).
"""

from __future__ import annotations

# Large-cap NSE names for the Top Opportunities scanner
NSE_SCANNER_SYMBOLS: tuple[str, ...] = (
    "TCS.NS",
    "INFY.NS",
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "LT.NS",
    "SBIN.NS",
)
