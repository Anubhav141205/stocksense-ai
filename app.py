"""
Streamlit UI: load data, compute indicators, show signals and explanations.
"""

from __future__ import annotations

import html
import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ai import explain_signals, generate_stock_insight
from data import StockDataError, fetch_stock_data, validate_ticker_quick
from indicators import add_indicators, latest_values
from scanner import opportunities_dataframe, recommendation_for_score, scan_universe
from signals import Signal, detect_signals, total_signal_score
from universe import NSE_SCANNER_SYMBOLS

logger = logging.getLogger(__name__)

# Navy + teal palette (readable on soft cool-gray background)
_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

    html, body, [class*="css"] div, .stMarkdown, label {
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
    }

    .stApp {
        background: linear-gradient(165deg, #eef2f7 0%, #e4eaf3 45%, #dfe8f2 100%);
    }

    /* Breathing room between major sections */
    main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.65);
        backdrop-filter: blur(8px);
    }

    h1 {
        color: #0d2137 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        color: #143d52 !important;
        font-weight: 600 !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255,255,255,0.5);
        border-radius: 12px;
        padding: 0.35rem;
        border: 1px solid rgba(13, 33, 55, 0.08);
    }

    [data-testid="stTabs"] button {
        border-radius: 10px !important;
        color: #0d2137 !important;
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(135deg, #0d3b66 0%, #1a6b6b 100%) !important;
        color: #f8fafc !important;
    }

    [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #0d3b66 0%, #2a9d8f 100%) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 12px rgba(13, 59, 102, 0.25);
    }

    [data-testid="baseButton-primary"]:hover {
        box-shadow: 0 4px 16px rgba(13, 59, 102, 0.35);
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(13, 33, 55, 0.07);
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        box-shadow:
            0 1px 2px rgba(13, 33, 55, 0.04),
            0 4px 16px rgba(13, 59, 102, 0.07),
            0 12px 28px rgba(13, 33, 55, 0.05);
    }

    /* Subtle depth on table / dataframe blocks */
    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        box-shadow:
            0 1px 3px rgba(13, 33, 55, 0.05),
            0 8px 24px rgba(13, 59, 102, 0.08);
        overflow: hidden;
        border: 1px solid rgba(13, 59, 102, 0.06);
    }

    .opps-hero {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248, 250, 252, 0.98) 100%);
        border: 1px solid rgba(13, 59, 102, 0.1);
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        margin: 0 0 1.75rem 0;
        box-shadow:
            0 1px 2px rgba(13, 33, 55, 0.04),
            0 6px 20px rgba(13, 59, 102, 0.08);
    }

    .opps-hero h3 {
        margin: 0 !important;
        color: #0d2137 !important;
    }

    /* Section dividers (use with section_divider()) */
    hr.ui-divider {
        border: none;
        height: 1px;
        background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(13, 59, 102, 0.14) 20%,
            rgba(13, 59, 102, 0.14) 80%,
            transparent 100%
        );
        margin: 2rem 0;
    }
    hr.ui-divider.loose {
        margin: 2.5rem 0;
    }

    /* Best opportunity spotlight — single prominent focal card */
    .best-opp-spotlight {
        margin: 0.5rem 0 2.75rem 0;
        padding: 0.35rem;
        border-radius: 28px;
        background: linear-gradient(
            145deg,
            rgba(13, 59, 102, 0.06) 0%,
            rgba(42, 157, 143, 0.09) 50%,
            rgba(13, 59, 102, 0.05) 100%
        );
        box-shadow:
            0 2px 4px rgba(13, 33, 55, 0.04),
            0 16px 48px rgba(13, 59, 102, 0.12),
            0 4px 12px rgba(13, 33, 55, 0.06);
    }
    .best-opp-card {
        background: linear-gradient(165deg, #ffffff 0%, #f1f5f9 55%, #ffffff 100%);
        border-radius: 24px;
        padding: 2rem 2.25rem 2.1rem 2.25rem;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.85),
            0 2px 6px rgba(13, 33, 55, 0.04),
            0 12px 36px rgba(13, 59, 102, 0.11),
            0 4px 14px rgba(13, 33, 55, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-left-width: 10px;
        border-left-style: solid;
        position: relative;
    }
    .best-opp-card.buy-accent { border-left-color: #059669; }
    .best-opp-card.watch-accent { border-left-color: #eab308; }
    .best-opp-card.avoid-accent { border-left-color: #ef4444; }
    .best-opp-ribbon {
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #64748b;
        margin: 0 0 0.6rem 0;
    }
    .best-opp-ticker {
        font-size: 2.35rem;
        font-weight: 800;
        color: #0d2137;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }
    .best-opp-sub {
        font-size: 1.05rem;
        font-weight: 600;
        color: #475569;
        margin: 0 0 1.35rem 0;
    }
    .best-opp-meta {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.85rem 1.1rem;
        margin-bottom: 0.25rem;
    }
    .best-opp-badge {
        display: inline-block;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 2px 6px rgba(13, 33, 55, 0.08);
    }
    .best-opp-badge.buy { background: #d1fae5; color: #065f46; }
    .best-opp-badge.watch { background: #fef9c3; color: #854d0e; }
    .best-opp-badge.avoid { background: #fee2e2; color: #991b1b; }
    .best-opp-score {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0d3b66;
    }
    .best-opp-insight {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #334155;
        margin: 1.15rem 0 0 0;
        padding-top: 1.15rem;
        border-top: 1px solid rgba(13, 59, 102, 0.12);
    }
</style>
"""


def inject_app_styles() -> None:
    """Apply global typography, background, and component styling."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def format_action_display(action: str) -> str:
    """Map raw action to emoji + label for tables and cards."""
    key = (action or "").strip().upper()
    labels = {"BUY": "🚀 BUY", "WATCH": "👀 WATCH", "AVOID": "⚠️ AVOID"}
    return labels.get(key, action or "—")


def add_action_emojis_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Action column with emoji-prefixed labels for demo display."""
    if df.empty or "Action" not in df.columns:
        return df
    out = df.copy()
    out["Action"] = out["Action"].astype(str).map(lambda x: format_action_display(x))
    return out


def style_opportunities_df(df: pd.DataFrame):
    """
    Color the Action column (green / yellow / red) for ``st.dataframe``.

    Falls back to the plain frame if styling is not supported.
    """

    def style_action_col(series: pd.Series) -> list[str]:
        styles: list[str] = []
        for v in series:
            vs = str(v)
            if vs.startswith("🚀"):
                styles.append(
                    "background-color: #d1fae5; color: #065f46; font-weight: 600; "
                    "border-radius: 6px;"
                )
            elif vs.startswith("👀"):
                styles.append(
                    "background-color: #fef9c3; color: #854d0e; font-weight: 600; "
                    "border-radius: 6px;"
                )
            elif vs.startswith("⚠️"):
                styles.append(
                    "background-color: #fecaca; color: #991b1b; font-weight: 600; "
                    "border-radius: 6px;"
                )
            else:
                styles.append("")
        return styles

    try:
        return df.style.apply(style_action_col, subset=["Action"], axis=0)
    except Exception:
        return df


def render_best_opportunity_section(full_df: pd.DataFrame) -> None:
    """
    Prominent metric + card for the highest-scoring row (first row after sort).
    """
    if full_df.empty:
        return

    row = full_df.iloc[0]
    stock = str(row["Stock"])
    score = int(row["Score"])
    action_raw = str(row["Action"]).strip().upper()
    insight = str(row["AI Insight"])
    action_disp = format_action_display(action_raw)

    if action_raw == "BUY":
        accent = "buy-accent"
        badge = "buy"
    elif action_raw == "WATCH":
        accent = "watch-accent"
        badge = "watch"
    else:
        accent = "avoid-accent"
        badge = "avoid"

    st.metric(
        label="Best Opportunity Today",
        value=stock,
        delta=f"Score {score} · {action_disp}",
    )

    safe_name = html.escape(stock)
    safe_insight = html.escape(insight)
    safe_badge = html.escape(action_disp)

    st.markdown(
        f'<div class="best-opp-wrap">'
        f'<div class="best-opp-card {accent}">'
        f'<p class="best-opp-kicker">Highest score this scan</p>'
        f'<p class="best-opp-name">{safe_name}</p>'
        f'<div class="best-opp-meta">'
        f'<span class="best-opp-badge {badge}">{safe_badge}</span>'
        f'<span class="best-opp-score">Score: {score}</span>'
        f"</div>"
        f'<p class="best-opp-insight"><strong>AI insight</strong><br/>{safe_insight}</p>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def run_fetch_and_store(symbol: str, period: str) -> None:
    """
    Load OHLCV data, compute indicators, and persist results in session state.

    On success, sets ``last_df``, ``last_symbol``, and ``last_period``.
    On failure, sets ``last_error`` and clears ``last_df``.

    Parameters
    ----------
    symbol
        Ticker string from the form.
    period
        yfinance period string (e.g. ``6mo``).
    """
    st.session_state.pop("last_error", None)
    quick_err = validate_ticker_quick(symbol)
    if quick_err:
        st.session_state["last_error"] = quick_err
        st.session_state.pop("last_df", None)
        return

    try:
        raw = fetch_stock_data(symbol, period=period)
        df = add_indicators(raw)
    except StockDataError as exc:
        st.session_state["last_error"] = str(exc)
        st.session_state.pop("last_df", None)
        st.session_state.pop("last_error_detail", None)
    except ValueError as exc:
        st.session_state["last_error"] = str(exc)
        st.session_state.pop("last_df", None)
        st.session_state.pop("last_error_detail", None)
    except Exception as exc:
        logger.exception("Unexpected error while loading data")
        st.session_state["last_error"] = "Something went wrong while loading data. Please try again."
        st.session_state["last_error_detail"] = str(exc)
        st.session_state.pop("last_df", None)
    else:
        st.session_state["last_df"] = df
        st.session_state["last_symbol"] = symbol.strip().upper()
        st.session_state["last_period"] = period
        st.session_state.pop("last_error_detail", None)


def _format_metric(value: float | None, decimals: int = 2) -> str:
    """Format a possibly missing float for ``st.metric`` display."""
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def render_metrics_header(symbol: str, period: str, lv: dict) -> None:
    """Show title row and four KPI metrics from ``latest_values``."""
    st.subheader(f"{symbol} — snapshot ({period})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Close", _format_metric(lv.get("close")))
    m2.metric("MA20", _format_metric(lv.get("ma20")))
    m3.metric("MA50", _format_metric(lv.get("ma50")))
    m4.metric("RSI(14)", _format_metric(lv.get("rsi14"), decimals=1))


def render_price_chart(df: pd.DataFrame) -> None:
    """Plot close and moving averages when enough non-NaN data exists."""
    need = ["close", "ma20", "ma50"]
    if not all(c in df.columns for c in need):
        return
    chart_df = df[need].dropna(how="any")
    if chart_df.empty:
        st.info("Not enough data to plot price and moving averages (need valid MA20/MA50 rows).")
        return
    st.subheader("Price & moving averages")
    st.line_chart(chart_df, height=320)


def render_signals_section(df: pd.DataFrame, symbol: str) -> None:
    """Show total score, action, AI insight, comma-separated titles, and per-signal detail."""
    signals_list = detect_signals(df)
    total = total_signal_score(signals_list)
    st.subheader("Signals")
    if not signals_list:
        st.info("No scoring conditions met on the latest bar (total score **0**).")
        return
    action = recommendation_for_score(total)
    summary = ", ".join(s.title for s in signals_list)
    a1, a2 = st.columns(2)
    with a1:
        st.metric("Total signal score", total)
    with a2:
        st.metric(
            "Action",
            format_action_display(action),
            help="BUY ≥4 · WATCH 2–3 · AVOID 0–1",
        )
    st.markdown("**AI insight**")
    st.caption("Uses OpenAI when `OPENAI_API_KEY` is set; otherwise a simple offline summary.")
    st.write(
        generate_stock_insight(
            symbol=symbol,
            signal_summary=summary,
            score=total,
            action=action,
            load_error=False,
        )
    )
    st.markdown("**Active signals (comma-separated):** " + summary)
    for sig, explanation in explain_signals(signals_list):
        _render_one_signal(sig, explanation)


def _render_one_signal(sig: Signal, explanation: str) -> None:
    """Render a single signal inside an expander."""
    with st.expander(sig.title, expanded=True):
        st.write(sig.detail)
        st.markdown("**Context**")
        st.write(explanation)


def render_raw_table(df: pd.DataFrame) -> None:
    """Show the last 10 rows of OHLCV + indicators."""
    show_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma20",
        "ma50",
        "rsi14",
        "volume_ma20",
    ]
    avail = [c for c in show_cols if c in df.columns]
    if not avail:
        st.caption("No columns available to display.")
        return
    tail = df[avail].tail(10)
    if tail.empty:
        st.caption("No rows to display.")
        return
    st.dataframe(tail, use_container_width=True)


def render_analysis_results() -> None:
    """Draw metrics, chart, signals, and table when session has a non-empty frame."""
    df = st.session_state.get("last_df")
    if df is None:
        return
    if df.empty:
        st.warning("Loaded data is empty; nothing to display.")
        return

    symbol = st.session_state.get("last_symbol", "")
    period = st.session_state.get("last_period", "")
    lv = latest_values(df)
    render_metrics_header(symbol, period, lv)
    render_price_chart(df)
    render_signals_section(df, symbol)
    with st.expander("Raw indicator table (last 10 rows)"):
        render_raw_table(df)


def render_top_opportunities_scanner() -> None:
    """
    Top Opportunities Scanner: NSE universe, scores, Stock/Signal/Score table, top 5.
    """
    st.subheader("Top Opportunities Scanner")
    st.caption(
        "NSE large-caps via Yahoo (`.NS`). "
        f"Universe ({len(NSE_SCANNER_SYMBOLS)}): {', '.join(NSE_SCANNER_SYMBOLS)}"
    )
    st.markdown(
        "**Scoring (latest bar, additive):** MA20 > MA50 → **+2** (trend bullish) · "
        "RSI 30–50 → **+1** (early momentum) · RSI < 30 → **+2** (oversold) · "
        "volume > 1.5× 20d avg → **+1** · close > MA20 → **+1**. "
        "Stocks are sorted by **total score** (desc)."
    )
    st.markdown(
        "**Action (from total score):** "
        "**BUY** if score ≥ 4 · **WATCH** if 2–3 · **AVOID** if 0–1. "
        "_Heuristic labels only — not financial advice._"
    )
    st.caption(
        "AI Insight: set environment variable `OPENAI_API_KEY` for OpenAI-generated blurbs; "
        "if unset or on API errors, a plain-English fallback is used."
    )

    period_options = ["1mo", "3mo", "6mo", "1y", "2y"]
    with st.form("scanner_form"):
        s_col1, s_col2 = st.columns([2, 1])
        with s_col1:
            scan_period = st.selectbox(
                "History period",
                options=period_options,
                index=2,
                key="nse_scan_period",
            )
        with s_col2:
            st.write("")
            run_scan = st.form_submit_button("Run scanner", type="primary")

    if run_scan:
        with st.spinner("Fetching data and scoring symbols…"):
            rows = scan_universe(NSE_SCANNER_SYMBOLS, period=scan_period)
        st.session_state["nse_scan_rows"] = rows

    rows = st.session_state.get("nse_scan_rows")
    if not rows:
        st.info("Choose a period and click **Run scanner** to load the universe.")
        return

    period_label = st.session_state.get("nse_scan_period", scan_period)
    full_df = opportunities_dataframe(rows)

    render_best_opportunity_section(full_df)

    st.markdown("<hr class='ui-divider'>", unsafe_allow_html=True)

    st.markdown(
        '<div class="opps-hero"><h3>🔥 Top opportunities (ranked)</h3>'
        f"<p style='margin:0.5rem 0 0 0;color:#3d5a73;font-size:0.95rem;'>"
        f"Period <strong>{period_label}</strong> · Action colors: "
        f"<span style='color:#065f46;font-weight:600;'>🚀 BUY</span> · "
        f"<span style='color:#854d0e;font-weight:600;'>👀 WATCH</span> · "
        f"<span style='color:#991b1b;font-weight:600;'>⚠️ AVOID</span></p></div>",
        unsafe_allow_html=True,
    )

    top_df = add_action_emojis_to_df(full_df.head(5).reset_index(drop=True))
    full_display = add_action_emojis_to_df(full_df.copy())
    styled_top = style_opportunities_df(top_df)
    styled_full = style_opportunities_df(full_display)

    _opp_columns = {
        "Stock": st.column_config.TextColumn("Stock", width="small"),
        "Signal": st.column_config.TextColumn("Signals", width="large"),
        "Score": st.column_config.NumberColumn("Score", format="%d"),
        "Action": st.column_config.TextColumn(
            "Action",
            width="medium",
            help="🚀 BUY (green) · 👀 WATCH (yellow) · ⚠️ AVOID (red).",
        ),
        "AI Insight": st.column_config.TextColumn(
            "AI Insight",
            width="large",
            help="OpenAI when API key set; otherwise template text.",
        ),
    }
    st.dataframe(styled_top, use_container_width=True, column_config=_opp_columns)

    st.markdown("<hr class='ui-divider loose'>", unsafe_allow_html=True)
    with st.expander("Full universe — sorted by score (includes Action)", expanded=False):
        st.dataframe(styled_full, use_container_width=True, column_config=_opp_columns)


def main() -> None:
    """Configure the page, form, and result / error areas."""
    st.set_page_config(
        page_title="Stock Analysis",
        layout="wide",
        page_icon="📈",
        initial_sidebar_state="collapsed",
    )
    inject_app_styles()
    st.title("Stock analysis")
    st.caption("Indicators and signals for education — not financial advice.")

    tab_single, tab_scanner = st.tabs(["Single stock", "Top opportunities"])

    with tab_single:
        with st.form("analysis_form"):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                symbol_input = st.text_input("Ticker", value="AAPL", max_chars=16)
            with col2:
                period = st.selectbox(
                    "Period",
                    options=["1mo", "3mo", "6mo", "1y", "2y"],
                    index=2,
                )
            with col3:
                st.write("")
                run = st.form_submit_button("Analyze", type="primary")

        symbol_input = (symbol_input or "").strip()

        if run:
            run_fetch_and_store(symbol_input, period)

        err = st.session_state.get("last_error")
        if err:
            st.error(err)
            detail = st.session_state.get("last_error_detail")
            if detail:
                with st.expander("Technical details"):
                    st.code(detail)

        render_analysis_results()

        if st.session_state.get("last_df") is None and not run:
            st.info("Enter a ticker and period, then click **Analyze** to load data.")

    with tab_scanner:
        render_top_opportunities_scanner()

    st.divider()
    st.caption("Data: Yahoo Finance via yfinance. Delays and errors may occur.")


main()
