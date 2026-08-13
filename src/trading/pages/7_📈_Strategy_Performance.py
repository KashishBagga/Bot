#!/usr/bin/env python3
"""
Strategy Performance Dashboard (Page 7)
========================================
Institutional-grade per-strategy metrics including:
  - Global portfolio view (total R, regime, exposure)
  - Per-strategy KPI cards (Win Rate, Profit Factor, Expectancy, Sharpe, Kelly%)
  - Regime-split performance table (TREND vs RANGE vs COMPRESSION vs GAP)
  - Cumulative equity curves (Plotly, multi-strategy overlay)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import sys
import os

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from src.models.postgres_database import PostgresDatabase

st.set_page_config(
    page_title="Strategy Performance | Bot",
    page_icon="📈",
    layout="wide",
)

# ── Strategy registry meta (status + group) ──────────────────────────────────
STRATEGY_META = {
    "Structural_v3.2_RVOL1.0":         {"status": "🟢 Production", "group": "A — Trend Cont."},
    "Structural_v3.2_RVOL0.8":         {"status": "🔬 Research",   "group": "A — Trend Cont."},
    "EMA_Pullback_20_50_RVOL0.5":      {"status": "🟢 Production", "group": "A — Trend Cont."},
    "VWAP_Reversion_1.5ATR_RVOL1.0":  {"status": "🔬 Research",   "group": "B — S/R Reaction"},
    "PrevDay_Extremes_RVOL1.2":        {"status": "🟢 Production", "group": "C — Breakout"},
    "ORB_15m_RVOL1.2":                 {"status": "🟢 Production", "group": "C — Breakout"},
    "ORB_30m_RVOL1.2":                 {"status": "🔬 Research",   "group": "C — Breakout"},
    "ATR_Squeeze_RVOL1.5":             {"status": "🔬 Research",   "group": "D — Volatility"},
    "Geometry_v1.0_Score35":           {"status": "🟢 Production", "group": "B — S/R Reaction"},
    "Geometry_v1.0_Score50":           {"status": "🔬 Research",   "group": "B — S/R Reaction"},
    "OrderFlow_v1.0":                  {"status": "🟢 Production", "group": "A — Trend Cont."},
    "ChartPattern_v1.0_Conf55":        {"status": "🔬 Research",   "group": "C — Breakout"},
    "ChartPattern_v1.0_Conf40":        {"status": "🔬 Research",   "group": "C — Breakout"},
    "VWAP_Reclaim_v1.0":               {"status": "🔬 Research",   "group": "A — Trend Cont."},
    "CPR_v1.0":                        {"status": "🔬 Research",   "group": "B — S/R Reaction"},
    "Gap_v1.0":                        {"status": "🟢 Production", "group": "E — Gap"},
    "ORB_60m_IB_RVOL1.2":             {"status": "🔬 Research",   "group": "C — Breakout"},
    "VerticalSpread_v1.0":             {"status": "🔬 Research",   "group": "D — Volatility"},
    "Straddle_v1.0_VolCompression":    {"status": "🔬 Research",   "group": "D — Volatility"},
    "Strangle_v1.0_VolCompression":    {"status": "🔬 Research",   "group": "D — Volatility"},
    "OIWallReaction_v1.0":             {"status": "🟢 Production", "group": "B — S/R Reaction"},
    "PCRExtremeReversal_v1.0":         {"status": "🔬 Research",   "group": "B — S/R Reaction"},
    "IronCondor_v1.0":                 {"status": "🔬 Research",   "group": "D — Volatility"},
    "Butterfly_v1.0":                  {"status": "🔬 Research",   "group": "D — Volatility"},
}

# Regime primary buckets for regime-split table
REGIME_BUCKETS = {
    "STRONG_TREND_UP":   "Trend Up",
    "WEAK_TREND_UP":     "Trend Up",
    "STRONG_TREND_DOWN": "Trend Down",
    "WEAK_TREND_DOWN":   "Trend Down",
    "RANGE":             "Range",
    "COMPRESSION":       "Compression",
    "GAP_UP":            "Gap",
    "GAP_DOWN":          "Gap",
    "TREND_UP":          "Trend Up",
    "TREND_DOWN":        "Trend Down",
    "NEUTRAL":           "Range",
    "UNKNOWN":           "Unknown",
}

def bucket_regime(regime_str: str) -> str:
    if not regime_str:
        return "Unknown"
    for key, bucket in REGIME_BUCKETS.items():
        if regime_str.startswith(key):
            return bucket
    return regime_str.split("_")[0].title()


@st.cache_data(ttl=60)
def load_metrics(days: int):
    db = PostgresDatabase()
    return db.get_strategy_metrics(days=days)

@st.cache_data(ttl=60)
def load_equity(days: int):
    db = PostgresDatabase()
    return db.get_strategy_equity_curve(days=days)

@st.cache_data(ttl=60)
def load_daily_r(days: int):
    """Load total realized R per day."""
    db = PostgresDatabase()
    rows = db.get_strategy_equity_curve(days=days)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_time"]).dt.date
    return df


# ────────────────────────────────────────────────────────────────────────────
# Page header
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .kpi-title { font-size: 0.75rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
    .kpi-sub   { font-size: 0.75rem; color: #718096; margin-top: 0.2rem; }
    .positive  { color: #48bb78 !important; }
    .negative  { color: #fc8181 !important; }
    .neutral   { color: #a0aec0 !important; }
    .strategy-card {
        background: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .strategy-name { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }
    .badge-prod    { background: #276749; color: #c6f6d5; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; }
    .badge-res     { background: #2a4365; color: #bee3f8; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Strategy Performance")
st.caption("Institutional-grade analytics | Metrics update every 60 seconds")

# ────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")
    lookback_days = st.selectbox("Lookback Period", [7, 14, 30, 60, 90], index=1, key="lookback")
    show_research = st.checkbox("Show Research Strategies", value=True)
    min_trades    = st.slider("Min Trades Filter", 0, 20, 2, key="min_trades")

    st.divider()
    st.caption("⚠️ Sample size warning")
    st.caption("< 100 trades = pilot data only. Do not draw profitability conclusions.")

# ────────────────────────────────────────────────────────────────────────────
# Load data
# ────────────────────────────────────────────────────────────────────────────
metrics_raw  = load_metrics(lookback_days)
equity_rows  = load_equity(lookback_days)
daily_df     = load_daily_r(lookback_days)

metrics_df = pd.DataFrame(metrics_raw) if metrics_raw else pd.DataFrame()
equity_df  = pd.DataFrame(equity_rows) if equity_rows else pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────
# Section A — Global Portfolio View
# ────────────────────────────────────────────────────────────────────────────
st.subheader("🌐 Global Portfolio View")

if not metrics_df.empty:
    total_trades_all  = metrics_df["total_trades"].sum()
    total_r_all       = metrics_df["mean_r"].multiply(metrics_df["total_trades"]).sum()
    total_wins        = metrics_df["wins"].sum()
    overall_win_rate  = total_wins / total_trades_all if total_trades_all > 0 else 0.0
    total_gp          = metrics_df["gross_profit"].sum() if "gross_profit" in metrics_df else 0.0
    total_gl          = metrics_df["gross_loss"].sum()   if "gross_loss"   in metrics_df else 0.0
    portfolio_pf      = total_gp / total_gl if total_gl > 0 else 0.0
else:
    total_trades_all, total_r_all, overall_win_rate, portfolio_pf = 0, 0.0, 0.0, 0.0

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    r_color = "positive" if total_r_all > 0 else "negative"
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">Total Realized R</div>
        <div class="kpi-value {r_color}">{total_r_all:+.2f}R</div>
        <div class="kpi-sub">Last {lookback_days} days</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">Total Trades</div>
        <div class="kpi-value">{total_trades_all}</div>
        <div class="kpi-sub">Closed positions</div>
    </div>""", unsafe_allow_html=True)

with c3:
    wr_color = "positive" if overall_win_rate >= 0.5 else "negative"
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">Portfolio Win Rate</div>
        <div class="kpi-value {wr_color}">{overall_win_rate:.1%}</div>
        <div class="kpi-sub">Across all strategies</div>
    </div>""", unsafe_allow_html=True)

with c4:
    pf_color = "positive" if portfolio_pf >= 1.5 else ("neutral" if portfolio_pf >= 1.0 else "negative")
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">Portfolio Profit Factor</div>
        <div class="kpi-value {pf_color}">{portfolio_pf:.2f}</div>
        <div class="kpi-sub">Target: ≥ 1.5</div>
    </div>""", unsafe_allow_html=True)

with c5:
    prod_count = sum(1 for m in STRATEGY_META.values() if "Production" in m["status"])
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-title">Active Strategies</div>
        <div class="kpi-value">{prod_count}</div>
        <div class="kpi-sub">{len(STRATEGY_META) - prod_count} Research</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# Section B — Per-Strategy KPI Cards
# ────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Per-Strategy Metrics")

if metrics_df.empty:
    st.info("No closed trades found for the selected period. Metrics will populate as trades close.")
else:
    # Aggregate per strategy (sum across regimes)
    agg = (
        metrics_df.groupby("strategy")
        .agg(
            total_trades=("total_trades", "sum"),
            wins=("wins", "sum"),
            gross_profit=("gross_profit", "sum"),
            gross_loss=("gross_loss", "sum"),
            mean_r=("mean_r", "mean"),
            avg_hold_minutes=("avg_hold_minutes", "mean"),
        )
        .reset_index()
    )
    agg["win_rate"]      = agg["wins"] / agg["total_trades"].replace(0, pd.NA)
    agg["profit_factor"] = agg["gross_profit"] / agg["gross_loss"].replace(0, pd.NA)
    agg["expectancy"]    = agg["mean_r"]

    # Filter
    if not show_research:
        prod_strats = [k for k, v in STRATEGY_META.items() if "Production" in v["status"]]
        agg = agg[agg["strategy"].isin(prod_strats)]
    agg = agg[agg["total_trades"] >= min_trades]

    if agg.empty:
        st.warning("No strategies match the current filters. Try reducing Min Trades.")
    else:
        cols_per_row = 3
        strategies = agg.to_dict("records")
        for i in range(0, len(strategies), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, row in enumerate(strategies[i:i+cols_per_row]):
                strat   = row["strategy"]
                meta    = STRATEGY_META.get(strat, {"status": "🔬 Research", "group": "Unknown"})
                status  = meta["status"]
                group   = meta["group"]
                total   = int(row["total_trades"])
                win_r   = row.get("win_rate") or 0.0
                pf      = row.get("profit_factor") or 0.0
                exp     = row.get("expectancy") or 0.0
                hold    = row.get("avg_hold_minutes") or 0.0

                wr_col  = "#48bb78" if win_r >= 0.5 else "#fc8181"
                pf_col  = "#48bb78" if pf >= 1.5 else ("#f6ad55" if pf >= 1.0 else "#fc8181")
                ex_col  = "#48bb78" if exp > 0 else "#fc8181"

                sample_warn = " ⚠️ Low sample" if total < 30 else ""

                with cols[j]:
                    with st.expander(f"{strat}  {status}", expanded=False):
                        st.markdown(f"**Group**: {group}")
                        st.markdown(f"**Trades**: {total}{sample_warn}")

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Win Rate",      f"{win_r:.1%}",  delta=None)
                        m2.metric("Profit Factor", f"{pf:.2f}",     delta=None)
                        m3.metric("Expectancy",    f"{exp:+.3f}R",  delta=None)

                        st.caption(f"Avg Hold: {hold:.0f} min")

                        # Regime breakdown for this strategy
                        regime_rows = metrics_df[metrics_df["strategy"] == strat].copy()
                        if not regime_rows.empty:
                            regime_rows["bucket"] = regime_rows["market_regime"].apply(bucket_regime)
                            regime_tbl = regime_rows.groupby("bucket").agg(
                                trades=("total_trades", "sum"),
                                wins=("wins", "sum"),
                                exp=("mean_r", "mean"),
                            ).reset_index()
                            regime_tbl["win_rate"] = regime_tbl["wins"] / regime_tbl["trades"].replace(0, pd.NA)
                            regime_tbl = regime_tbl.rename(columns={"bucket": "Regime", "trades": "Trades", "exp": "Exp (R)", "win_rate": "Win %"})
                            regime_tbl["Win %"] = regime_tbl["Win %"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                            regime_tbl["Exp (R)"] = regime_tbl["Exp (R)"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
                            st.dataframe(regime_tbl[["Regime", "Trades", "Win %", "Exp (R)"]], hide_index=True, use_container_width=True)

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# Section C — Equity Curves
# ────────────────────────────────────────────────────────────────────────────
st.subheader("📈 Cumulative Equity Curves")

if equity_df.empty:
    st.info("No equity curve data available for the selected period.")
else:
    equity_df["entry_time"] = pd.to_datetime(equity_df["entry_time"])
    equity_df["cumulative_r"] = pd.to_numeric(equity_df["cumulative_r"], errors="coerce")

    # Strategy selector
    all_strats = sorted(equity_df["strategy"].unique())
    selected   = st.multiselect("Select strategies to display", all_strats, default=all_strats[:5])

    if selected:
        fig = go.Figure()

        palette = px.colors.qualitative.Set2
        for idx, strat in enumerate(selected):
            s_df = equity_df[equity_df["strategy"] == strat].sort_values("entry_time")
            color = palette[idx % len(palette)]
            meta  = STRATEGY_META.get(strat, {})
            is_prod = "Production" in meta.get("status", "")

            fig.add_trace(go.Scatter(
                x=s_df["entry_time"],
                y=s_df["cumulative_r"],
                mode="lines+markers",
                name=f"{'★ ' if is_prod else ''}{strat}",
                line=dict(width=2 if is_prod else 1, color=color, dash="solid" if is_prod else "dot"),
                marker=dict(size=4),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Time: %{x}<br>"
                    "Cumulative R: %{y:.2f}R<br>"
                    "<extra></extra>"
                ),
            ))

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(26,32,44,1)",
            paper_bgcolor="rgba(26,32,44,1)",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Cumulative R"),
            margin=dict(l=40, r=20, t=40, b=40),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Daily R bar chart ──
    if not daily_df.empty and "final_pnl_r" in daily_df.columns:
        st.subheader("📅 Daily PnL (All Strategies)")
        daily_agg = daily_df.groupby("entry_date")["final_pnl_r"].sum().reset_index()
        daily_agg.columns = ["Date", "Daily R"]
        bar_colors = ["#48bb78" if r >= 0 else "#fc8181" for r in daily_agg["Daily R"]]

        bar_fig = go.Figure(go.Bar(
            x=daily_agg["Date"].astype(str),
            y=daily_agg["Daily R"],
            marker_color=bar_colors,
            hovertemplate="<b>%{x}</b><br>Daily R: %{y:+.2f}R<extra></extra>",
        ))
        bar_fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
        bar_fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(26,32,44,1)",
            paper_bgcolor="rgba(26,32,44,1)",
            height=280,
            yaxis_title="R",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

st.divider()

# ────────────────────────────────────────────────────────────────────────────
# Section D — Full Metrics Table
# ────────────────────────────────────────────────────────────────────────────
st.subheader("📋 Full Metrics Table")
st.caption("One row per strategy × market regime. Exportable.")

if not metrics_df.empty:
    display_cols = [
        "strategy", "market_regime", "total_trades", "wins", "losses",
        "win_rate", "expectancy", "profit_factor", "sharpe", "kelly_pct",
        "avg_hold_minutes", "worst_r", "best_r",
    ]
    existing_cols = [c for c in display_cols if c in metrics_df.columns]
    tbl = metrics_df[existing_cols].copy()
    tbl = tbl[tbl["total_trades"] >= min_trades]

    # Format
    for col in ["win_rate"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    for col in ["expectancy", "worst_r", "best_r"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    for col in ["profit_factor", "sharpe", "kelly_pct"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")

    tbl.columns = [c.replace("_", " ").title() for c in tbl.columns]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    csv = metrics_df.to_csv(index=False)
    st.download_button("⬇️ Export Full Metrics CSV", csv, "strategy_metrics.csv", "text/csv")
else:
    st.info("No metrics data available. Trades will appear here once closed.")

st.divider()
st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S IST')} | Data from `trade_performance` table | Valid trades only")
