"""
Plain-language explanations for trading signals and AI-style scan insights.

Per-stock insights use OpenAI when ``OPENAI_API_KEY`` is set; otherwise a simple
fallback keeps the app usable offline.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

from signals import Signal

logger = logging.getLogger(__name__)

_DISCLAIMER = "This is informational only, not investment advice."


def explain_signal(signal: Signal) -> str:
    """
    Build a short educational paragraph for one signal.

    Parameters
    ----------
    signal
        Detected signal from :func:`signals.detect_signals`.

    Returns
    -------
    str
        Neutral context text suitable for display under the signal detail.
    """
    if signal.kind == "trend_bullish":
        return (
            "When the 20-day average sits above the 50-day, the short-term trend is "
            "stronger than the medium-term filter. Some read that as bullish structure, "
            f"but trends can persist or reverse — use other risk tools. {_DISCLAIMER}"
        )

    if signal.kind == "momentum_early":
        return (
            "RSI between about 30 and 50 often means the stock is off extreme oversold "
            "levels but not yet overbought — sometimes described as early recovery momentum. "
            f"It is not a timing guarantee. {_DISCLAIMER}"
        )

    if signal.kind == "oversold_rsi":
        return (
            "RSI below 30 suggests heavy recent selling in the lookback window. "
            "Some traders look for bounces; others wait for confirmation because "
            f"oversold can persist in downtrends. {_DISCLAIMER}"
        )

    if signal.kind == "volume_elevated":
        return (
            "Volume well above its recent average means more participation than usual. "
            "It can accompany news or shifts in sentiment; combine with price direction "
            f"for context. {_DISCLAIMER}"
        )

    if signal.kind == "price_above_ma20":
        return (
            "Price above the 20-day average suggests short-term strength relative to "
            "that smoothing line. It is a simple trend filter, not a standalone buy signal. "
            f"{_DISCLAIMER}"
        )

    return f"{signal.detail} {_DISCLAIMER}"


def explain_signals(signals: List[Signal]) -> List[tuple[Signal, str]]:
    """
    Pair each signal with its explanation for iteration in the UI.

    Parameters
    ----------
    signals
        List from :func:`signals.detect_signals` (may be empty).

    Returns
    -------
    list of tuple
        ``(signal, explanation)`` for each input signal, in order.
    """
    return [(s, explain_signal(s)) for s in signals]


def _clamp_two_lines(text: str, max_chars: int = 320) -> str:
    """Keep output short (roughly two lines in a table cell)."""
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text


def _build_insight_prompt(symbol: str, signal_summary: str, score: int, action: str) -> str:
    return (
        "You help beginner investors skim a stock screener.\n\n"
        "Write exactly 1 or 2 short sentences (under 55 words total).\n"
        "Use simple everyday English only. Do NOT use abbreviations like RSI, MA, or numbers like 1.5x — "
        'say things like "momentum", "recent average price", "busier trading day".\n'
        "Do not give buy or sell orders. Do not promise returns. "
        'Tone: calm, like "worth keeping an eye on".\n\n'
        f"Stock: {symbol}\n"
        f"What the scanner noticed: {signal_summary}\n"
        f"Total score (more = more checks passed): {score}\n"
        f"Rough label (not advice): {action}\n\n"
        "Explain what this snapshot might mean in plain language."
    )


def _try_openai_insight(prompt: str) -> str | None:
    """Return model text or None if unavailable / failed."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.strip():
        return None
    try:
        from openai import OpenAI
    except ImportError:
        logger.debug("openai package not installed; using fallback insight")
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    try:
        client = OpenAI(api_key=key.strip())
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.45,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            return None
        # Strip markdown bullets / quotes the model might add
        raw = re.sub(r"^[\"']|[\"']$", "", raw)
        return _clamp_two_lines(raw)
    except Exception as exc:
        logger.debug("AI insight request failed: %s", exc)
        return None


def _fallback_stock_insight_plain(
    symbol: str,
    signal_summary: str,
    score: int,
    action: str,
    *,
    load_error: bool,
) -> str:
    """
    Template insight when the API is off or fails — simple English, under two lines.
    """
    if load_error:
        return _clamp_two_lines(
            "We couldn't load fresh data for this stock, so there's nothing useful to say yet. "
            "Try again later."
        )

    empty = not signal_summary.strip() or signal_summary in ("No signal", "—")
    if empty:
        if action == "AVOID":
            return _clamp_two_lines(
                f"Nothing much stands out for {symbol}. "
                "It is okay to wait until the picture looks clearer."
            )
        if action == "WATCH":
            return _clamp_two_lines(
                f"The picture for {symbol} is mixed. "
                "It may be worth watching, but there is no rush to decide."
            )
        return _clamp_two_lines(
            f"The picture for {symbol} is quiet. There is no rush to act on this name alone."
        )

    if action == "BUY":
        text = (
            f"{symbol} shows early bullish momentum with several checks in place: {signal_summary}. "
            "It is worth monitoring, but size any risk carefully."
        )
    elif action == "WATCH":
        text = (
            f"The setup for {symbol} is mixed: {signal_summary}. "
            "It is not a full green light yet, but it deserves a spot on your watchlist."
        )
    else:
        text = (
            f"The story for {symbol} is quiet: {signal_summary}. "
            "There is nothing strong enough here to chase on its own."
        )
    return _clamp_two_lines(text)


def generate_stock_insight(
    symbol: str,
    signal_summary: str,
    score: int,
    action: str,
    *,
    load_error: bool = False,
) -> str:
    """
    Short, plain-English blurb for a table row (signals + score + action).

    Uses OpenAI when ``OPENAI_API_KEY`` is set and the request succeeds; otherwise
    :func:`_fallback_stock_insight_plain`.

    Parameters
    ----------
    symbol
        Ticker.
    signal_summary
        Comma-separated signal titles, or ``No signal`` / unavailable text.
    score
        Total rule score.
    action
        BUY / WATCH / AVOID.
    load_error
        If True, skip the API and use the data-error fallback.

    Returns
    -------
    str
        One or two sentences, non-technical.
    """
    if load_error:
        return _fallback_stock_insight_plain(
            symbol, signal_summary, score, action, load_error=True
        )

    prompt = _build_insight_prompt(symbol, signal_summary, score, action)
    ai_text = _try_openai_insight(prompt)
    if ai_text:
        return ai_text

    return _fallback_stock_insight_plain(
        symbol, signal_summary, score, action, load_error=False
    )
