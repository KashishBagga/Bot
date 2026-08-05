#!/usr/bin/env python3
"""Market State — quick overview of what the system currently believes about
each symbol: bias, regime, volatility/efficiency, active S/R zones, and
in-progress/ready chart patterns.

Reads the market_state table, one row per symbol, overwritten every 5-min candle.
Uses st.fragment to refresh data seamlessly without full-page reloads.
"""
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.models.postgres_database import PostgresDatabase

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
STALE_AFTER = timedelta(minutes=10)
AUTO_REFRESH_SECONDS = 20

BIAS_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}


def format_dt(dt):
    if dt is None:
        return "N/A"
    if dt.tzinfo is not None:
        dt = dt.astimezone(KOLKATA_TZ)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


st.set_page_config(page_title="Market State", page_icon="📡", layout="wide")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
    }
    .zone-pill {
        padding: 6px 12px;
        border-radius: 8px;
        margin: 4px;
        display: inline-block;
        font-size: 0.9em;
    }
    .zone-supply { background-color: rgba(239, 68, 68, 0.10); border: 1px solid rgba(239, 68, 68, 0.35); }
    .zone-demand { background-color: rgba(34, 197, 94, 0.10); border: 1px solid rgba(34, 197, 94, 0.35); }
    .pattern-pill {
        background-color: rgba(6, 182, 212, 0.08);
        border: 1px solid rgba(6, 182, 212, 0.3);
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        display: block;
    }
    .stale-badge {
        background-color: rgba(234, 179, 8, 0.15);
        border: 1px solid rgba(234, 179, 8, 0.4);
        color: #eab308;
        padding: 2px 10px;
        border-radius: 8px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db():
    return PostgresDatabase()


db = get_db()


@st.cache_data(ttl=5)
def load_market_state():
    rows = []
    try:
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT symbol, updated_at, current_price, daily_bias, market_regime,
                           rvol, atr, move_efficiency, wickiness,
                           narrative_bias, narrative_confidence, zones, patterns
                    FROM market_state
                    ORDER BY symbol
                """)
                cols = [desc[0] for desc in cursor.description]
                rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
    except Exception as e:
        st.error(f"Failed to query database: {e}")
    return rows


st.title("📡 Market State — Current Overview")

@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def render_market_state():
    rows = load_market_state()
    now = datetime.now(KOLKATA_TZ)

    st.caption(f"Last loaded {now.strftime('%H:%M:%S')} IST · Real-time auto-refresh active")

    if not rows:
        st.info(
            "No market state recorded yet. This table is populated by the live trader "
            "every candle — run `./run_indian_trader.sh` during market hours to populate it."
        )
        return

    for row in rows:
        updated_at = row.get("updated_at")
        stale = updated_at is None or (now - updated_at.astimezone(KOLKATA_TZ)) > STALE_AFTER
        stale_badge = ' <span class="stale-badge">⚠️ stale</span>' if stale else ""

        bias = row.get("daily_bias") or "NEUTRAL"
        bias_emoji = BIAS_EMOJI.get(bias, "⚪")

        st.subheader(f"{row['symbol']}  ·  {row.get('current_price', 0.0):.2f}", anchor=False)
        st.markdown(
            f"**Updated:** {format_dt(updated_at)} IST{stale_badge}  ·  "
            f"**Regime:** `{row.get('market_regime', 'UNKNOWN')}`",
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Daily Bias", f"{bias_emoji} {bias}")
        narrative_bias = row.get("narrative_bias")
        narrative_conf = row.get("narrative_confidence")
        m2.metric(
            "Narrative Bias",
            narrative_bias or "N/A",
            delta=f"{narrative_conf*100:.0f}% conf" if narrative_conf is not None else None,
        )
        rvol = row.get("rvol")
        m3.metric("RVOL", f"{rvol:.2f}x" if rvol is not None else "N/A")
        atr = row.get("atr")
        m4.metric("ATR", f"{atr:.2f}" if atr is not None else "N/A")
        eff = row.get("move_efficiency")
        m5.metric("Move Efficiency", f"{eff:.2f}" if eff is not None else "N/A")

        z_col, p_col = st.columns(2)

        with z_col:
            st.markdown("**Active Supply/Demand Zones** *(scored: rejection count, RVOL at touch, freshness)*")
            zones = row.get("zones") or []
            if not zones:
                st.text("No active zones detected.")
            else:
                for z in zones:
                    css = "zone-supply" if z["type"] == "SUPPLY" else "zone-demand"
                    icon = "🔺" if z["type"] == "SUPPLY" else "🔻"
                    st.markdown(
                        f"<div class='zone-pill {css}'>{icon} <b>{z['type']}</b> @ {z['level']:.2f} "
                        f"— score {z['score']:.0f}, {z['rejection_count']} rejection(s), "
                        f"freshness {z['freshness']:.0f}</div>",
                        unsafe_allow_html=True,
                    )

        with p_col:
            st.markdown("**In-Progress / Ready Chart Patterns**")
            patterns = row.get("patterns") or []
            if not patterns:
                st.text("No chart patterns currently forming or ready.")
            else:
                for p in sorted(patterns, key=lambda x: x["confidence"], reverse=True):
                    dir_icon = "📈" if p["direction"] == "LONG" else ("📉" if p["direction"] == "SHORT" else "↔️")
                    targets_str = ", ".join(f"{t:.2f}" for t in p.get("targets", []))
                    st.markdown(
                        f"<div class='pattern-pill'>{dir_icon} <b>{p['type'].replace('_', ' ').title()}</b> "
                        f"— {p['state']} ({p['completion_pct']*100:.0f}% complete, "
                        f"confidence {p['confidence']*100:.0f}%)<br>"
                        f"Breakout: {p['breakout_level']:.2f} · Invalidation: {p['invalidation']:.2f} · "
                        f"Targets: {targets_str or 'N/A'}</div>",
                        unsafe_allow_html=True,
                    )

        st.write("---")

render_market_state()
