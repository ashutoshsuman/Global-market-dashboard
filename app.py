import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Global Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide sidebar */
    [data-testid="collapsedControl"] { display: none; }

    /* Card styling */
    .chart-card {
        background: var(--background-color, #0e1117);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 10px;
        padding: 12px 16px 4px 16px;
        margin-bottom: 8px;
    }
    .chart-card h4 {
        margin: 0 0 4px 0;
        font-size: 1rem;
        color: inherit;
    }

    /* Tighter spacing */
    .block-container { padding-top: 1.5rem; padding-bottom: 0; }
    div[data-testid="stHorizontalBlock"] { gap: 0.5rem; }

    /* Section divider */
    .section-divider {
        border-top: 2px solid rgba(128,128,128,0.25);
        margin: 2rem 0 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Asset definitions
# ──────────────────────────────────────────────
ASSETS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "NIKKEI 225": "^N225",
    "HANG SENG": "^HSI",
    "SHANGHAI": "000001.SS",
    "GOLD": "GC=F",
    "CRUDE OIL": "CL=F",
}

DURATION_MAP = {
    "1M": 30,
    "2M": 60,
    "6M": 180,
    "1Y": 365,
    "3Y": 365 * 3,
    "5Y": 365 * 5,
}

INTERVAL_OPTIONS = ["4H", "1D", "1M", "1Y"]
INTERVAL_YF_MAP = {
    "4H": "1h",   # yfinance doesn't have 4h; use 1h as proxy
    "1D": "1d",
    "1M": "1mo",
    "1Y": "3mo",  # quarterly for yearly view
}

# Max intraday history yfinance allows is ~730 days for 1h
MAX_INTRADAY_DAYS = 729


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame | None:
    """Download data via yfinance with robust error handling."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None

        # Handle multi-index columns (yfinance sometimes returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure Close column exists
        if "Close" not in df.columns:
            # Try case-insensitive match
            close_cols = [c for c in df.columns if str(c).lower() == "close"]
            if close_cols:
                df.rename(columns={close_cols[0]: "Close"}, inplace=True)
            else:
                return None

        df = df[["Close"]].dropna()
        return df if not df.empty else None
    except Exception:
        return None


def resolve_dates(duration: str, manual_start: date, manual_end: date):
    """Return (start_str, end_str) based on quick duration or manual dates."""
    end_dt = datetime.combine(manual_end, datetime.min.time())
    if duration != "Custom":
        days = DURATION_MAP[duration]
        start_dt = end_dt - timedelta(days=days)
    else:
        start_dt = datetime.combine(manual_start, datetime.min.time())
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def clamp_start_for_intraday(start_str: str, interval: str) -> str:
    """Intraday intervals have limited history; clamp start date."""
    if interval in ("1h",):
        earliest = datetime.now() - timedelta(days=MAX_INTRADAY_DAYS)
        req = datetime.strptime(start_str, "%Y-%m-%d")
        if req < earliest:
            return earliest.strftime("%Y-%m-%d")
    return start_str


def make_chart(df: pd.DataFrame, title: str, color: str = "#00b4d8") -> go.Figure:
    """Build a Plotly line chart with range slider and crosshair."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        line=dict(color=color, width=1.8),
        name=title,
        hovertemplate="%{x|%b %d, %Y %H:%M}<br>%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=None,
        margin=dict(l=0, r=0, t=8, b=0),
        height=310,
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            type="date",
        ),
        yaxis=dict(title=None, side="right", tickformat=",.0f"),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
    return fig


PLOTLY_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "responsive": True,
}

CHART_COLORS = [
    "#00b4d8", "#f77f00", "#06d6a0", "#ef476f",
    "#8338ec", "#ffd166", "#118ab2", "#e63946",
    "#2a9d8f", "#264653", "#fb5607",
]


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("## 📈 Global Market Dashboard")
st.caption("Lightweight TradingView-style explorer  ·  Powered by Yahoo Finance")

# ══════════════════════════════════════════════
# SECTION 1 — Market Trends
# ══════════════════════════════════════════════
st.markdown("### Market Trends")

# ── Duration selector ──
dur_cols = st.columns([3, 2, 2])
with dur_cols[0]:
    durations = list(DURATION_MAP.keys()) + ["Custom"]
    quick_dur = st.radio("Quick Duration", durations, index=3, horizontal=True, label_visibility="collapsed")

with dur_cols[1]:
    default_start = date.today() - timedelta(days=365)
    manual_start = st.date_input("Start", value=default_start, key="trend_start")

with dur_cols[2]:
    manual_end = st.date_input("End", value=date.today(), key="trend_end")

start_str, end_str = resolve_dates(quick_dur, manual_start, manual_end)

# ── Asset selector ──
sel_cols = st.columns([1, 4])
with sel_cols[0]:
    select_all = st.checkbox("Select All Indices", value=True, key="trend_all")
with sel_cols[1]:
    if select_all:
        selected_assets = list(ASSETS.keys())
    else:
        selected_assets = st.multiselect(
            "Select assets",
            options=list(ASSETS.keys()),
            default=["NIFTY 50", "S&P 500", "GOLD"],
            label_visibility="collapsed",
        )

if not selected_assets:
    st.info("Select at least one asset to display charts.")
else:
    # Render 2-column grid of chart cards
    for row_idx in range(0, len(selected_assets), 2):
        cols = st.columns(2)
        for col_idx, col in enumerate(cols):
            asset_idx = row_idx + col_idx
            if asset_idx >= len(selected_assets):
                break
            asset_name = selected_assets[asset_idx]
            ticker = ASSETS[asset_name]
            color = CHART_COLORS[asset_idx % len(CHART_COLORS)]

            with col:
                st.markdown(f'<div class="chart-card"><h4>{asset_name}</h4></div>', unsafe_allow_html=True)

                interval_label = st.selectbox(
                    "Interval",
                    INTERVAL_OPTIONS,
                    index=1,
                    key=f"intv_{asset_name}",
                    label_visibility="collapsed",
                )
                yf_interval = INTERVAL_YF_MAP[interval_label]
                adj_start = clamp_start_for_intraday(start_str, yf_interval)

                with st.spinner(""):
                    df = fetch_data(ticker, adj_start, end_str, yf_interval)

                if df is not None and not df.empty:
                    fig = make_chart(df, asset_name, color)
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    st.warning(f"No data available for {asset_name} with interval {interval_label}.")

# ══════════════════════════════════════════════
# SECTION 2 — Correlation Analyzer
# ══════════════════════════════════════════════
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### Correlation Analyzer")

# ── Duration selector (correlation) ──
corr_dur_cols = st.columns([3, 2, 2])
with corr_dur_cols[0]:
    corr_dur = st.radio("Quick Duration ", durations, index=3, horizontal=True, key="corr_dur", label_visibility="collapsed")
with corr_dur_cols[1]:
    corr_start = st.date_input("Start", value=default_start, key="corr_start")
with corr_dur_cols[2]:
    corr_end = st.date_input("End", value=date.today(), key="corr_end")

corr_start_str, corr_end_str = resolve_dates(corr_dur, corr_start, corr_end)

# ── Asset selector (correlation) ──
corr_sel_cols = st.columns([1, 4])
with corr_sel_cols[0]:
    corr_all = st.checkbox("Select All (Correlation)", value=False, key="corr_all")
with corr_sel_cols[1]:
    if corr_all:
        corr_assets = list(ASSETS.keys())[:8]
    else:
        corr_assets = st.multiselect(
            "Select assets (max 8)",
            options=list(ASSETS.keys()),
            default=["NIFTY 50", "S&P 500", "GOLD", "CRUDE OIL"],
            max_selections=8,
            key="corr_multi",
            label_visibility="collapsed",
        )

if len(corr_assets) < 2:
    st.info("Select at least 2 assets for correlation analysis.")
else:
    # Fetch all close series & merge
    close_frames: dict[str, pd.Series] = {}
    with st.spinner("Fetching data for correlation…"):
        for name in corr_assets:
            df = fetch_data(ASSETS[name], corr_start_str, corr_end_str, "1d")
            if df is not None and not df.empty:
                close_frames[name] = df["Close"]

    if len(close_frames) < 2:
        st.warning("Not enough data to compute correlations.")
    else:
        combined = pd.DataFrame(close_frames).dropna()
        if combined.empty or len(combined) < 2:
            st.warning("Insufficient overlapping data for selected assets.")
        else:
            returns = combined.pct_change().dropna()
            corr_matrix = returns.corr()

            # 1 — Correlation table
            st.markdown("#### Correlation Matrix")
            st.dataframe(
                corr_matrix.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                use_container_width=True,
            )

            # 2 — Heatmap
            corr_cols = st.columns(2)
            with corr_cols[0]:
                st.markdown("#### Heatmap")
                heatmap_fig = px.imshow(
                    corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    color_continuous_scale="RdYlGn",
                    zmin=-1,
                    zmax=1,
                    text_auto=".2f",
                    aspect="equal",
                )
                heatmap_fig.update_layout(
                    margin=dict(l=0, r=0, t=24, b=0),
                    height=420,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11),
                )
                st.plotly_chart(heatmap_fig, use_container_width=True, config=PLOTLY_CONFIG)

            # 3 — Cumulative returns line chart
            with corr_cols[1]:
                st.markdown("#### Cumulative Returns")
                cum_returns = (1 + returns).cumprod() - 1
                ret_fig = go.Figure()
                for i, col in enumerate(cum_returns.columns):
                    ret_fig.add_trace(go.Scatter(
                        x=cum_returns.index,
                        y=cum_returns[col],
                        mode="lines",
                        name=col,
                        line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.6),
                        hovertemplate="%{x|%b %d, %Y}<br>%{y:.2%}<extra>" + col + "</extra>",
                    ))
                ret_fig.update_layout(
                    margin=dict(l=0, r=0, t=24, b=0),
                    height=420,
                    hovermode="x unified",
                    yaxis=dict(tickformat=".0%", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11),
                )
                ret_fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
                ret_fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.1)")
                st.plotly_chart(ret_fig, use_container_width=True, config=PLOTLY_CONFIG)

# ── Footer ──
st.markdown("---")
st.caption("Data from Yahoo Finance · Not financial advice · Built with Streamlit + Plotly")
