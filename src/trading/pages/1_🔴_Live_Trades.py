#!/usr/bin/env python3
"""Live Trades — currently-running real positions across all experiments.

Reads trade_performance rows with exit_time IS NULL. current_price /
unrealized_pnl_r / last_heartbeat_at are refreshed every 5-min candle.
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

# experiment_name -> category label.
EXPERIMENT_CATEGORIES = {
    "Structural_v3.2_RVOL1.0": "Structural",
    "Structural_v3.2_RVOL0.8": "Structural",
    "EMA_Pullback_20_50_RVOL0.5": "Pullback",
    "VWAP_Reversion_1.5ATR_RVOL1.0": "Reversion",
    "PrevDay_Extremes_RVOL1.2": "Prev-Day Levels",
    "ORB_15m_RVOL1.2": "Opening Range Breakout",
    "ORB_30m_RVOL1.2": "Opening Range Breakout",
    "ATR_Squeeze_RVOL1.5": "Volatility Breakout",
    "Geometry_v1.0_Score35": "Geometry / Confluence",
    "Geometry_v1.0_Score50": "Geometry / Confluence",
    "OrderFlow_v1.0": "Order Flow",
    "ChartPattern_v1.0_Conf55": "Chart Pattern",
    "ChartPattern_v1.0_Conf40": "Chart Pattern",
    "VWAP_Reclaim_v1.0": "VWAP Reclaim",
    "CPR_v1.0": "CPR (Pivot)",
    "Gap_v1.0": "Gap",
    "ORB_60m_IB_RVOL1.2": "Opening Range Breakout",
    "VerticalSpread_v1.0": "Vertical Spread",
    "Straddle_v1.0_VolCompression": "Straddle/Strangle",
    "Strangle_v1.0_VolCompression": "Straddle/Strangle",
}


def category_of(experiment_name: str) -> str:
    return EXPERIMENT_CATEGORIES.get(experiment_name, "Other")


def format_dt(dt):
    if dt is None:
        return "N/A"
    if dt.tzinfo is not None:
        dt = dt.astimezone(KOLKATA_TZ)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


st.set_page_config(page_title="Live Trades", page_icon="🔴", layout="wide")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
    }
    .factor-pill {
        background-color: rgba(6, 182, 212, 0.05);
        border: 1px solid rgba(6, 182, 212, 0.2);
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px;
        display: inline-block;
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
def load_live_positions():
    positions, cf_count = [], 0
    try:
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT trade_id, symbol, experiment_name, strategy, signal_type,
                           entry_price, entry_time, current_price, unrealized_pnl_r,
                           stop_loss, take_profit, tp1, initial_stop_loss, initial_take_profit,
                           mfe_r, mae_r, bars_held, stop_loss_distance,
                           last_heartbeat_at, diagnostics, features
                    FROM trade_performance
                    WHERE exit_time IS NULL
                    ORDER BY symbol, entry_time DESC
                """)
                cols = [desc[0] for desc in cursor.description]
                positions = [dict(zip(cols, row)) for row in cursor.fetchall()]

            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM counterfactual_results WHERE exit_time IS NULL")
                cf_count = cursor.fetchone()[0]
    except Exception as e:
        st.error(f"Failed to query database: {e}")
    return positions, cf_count


@st.cache_data(ttl=5)
def load_combo_positions():
    combos, cf_combo_count = [], 0
    try:
        with db._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT combo_id, symbol, experiment_name, combo_type, setup_type,
                           entry_time, underlying_entry_price, legs, net_premium_paid,
                           max_loss, max_profit, target_r, stop_r, current_pnl_r,
                           confidence, diagnostics
                    FROM combo_trades
                    WHERE exit_time IS NULL
                    ORDER BY symbol, entry_time DESC
                """)
                cols = [desc[0] for desc in cursor.description]
                combos = [dict(zip(cols, row)) for row in cursor.fetchall()]

            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM counterfactual_combo_results WHERE exit_time IS NULL")
                cf_combo_count = cursor.fetchone()[0]
    except Exception as e:
        st.error(f"Failed to query combo positions: {e}")
    return combos, cf_combo_count


# Render page titles outside fragment
st.title("🔴 Live — Currently Running Trades")

# Define category options
categories_all = list(set(EXPERIMENT_CATEGORIES.values()))
selected_category = st.radio(
    "Filter by category", ["All"] + categories_all, horizontal=True
)

st.write("---")

# Wrap live data block in a fragment
@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def render_live_trades(category_filter):
    positions, cf_count = load_live_positions()
    combos, cf_combo_count = load_combo_positions()
    now = datetime.now(KOLKATA_TZ)

    st.caption(f"Last loaded {now.strftime('%H:%M:%S')} IST · Real-time auto-refresh active")

    if not positions and not combos:
        st.info(
            f"No real trades currently running. "
            f"({cf_count + cf_combo_count} counterfactual/shadow position(s) active — see the EOD dashboard for those.)"
        )
        return

    # Filter data inside fragment
    if category_filter != "All":
        filtered_positions = [p for p in positions if category_of(p["experiment_name"]) == category_filter]
        filtered_combos = [c for c in combos if category_of(c["experiment_name"]) == category_filter]
    else:
        filtered_positions = positions
        filtered_combos = combos

    symbols_open = sorted(set(p["symbol"] for p in filtered_positions) | set(c["symbol"] for c in filtered_combos))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Open Positions (single-leg + combo)", f"{len(filtered_positions)} + {len(filtered_combos)}")
    col2.metric("Symbols With Open Trades", len(symbols_open))
    total_unrealized = sum(p.get("unrealized_pnl_r") or 0.0 for p in filtered_positions) + sum(c.get("current_pnl_r") or 0.0 for c in filtered_combos)
    col3.metric("Total Unrealized PnL", f"{total_unrealized:+.2f} R")
    categories_open = sorted(set(category_of(p["experiment_name"]) for p in filtered_positions) | set(category_of(c["experiment_name"]) for c in filtered_combos))
    col4.metric("Categories Represented", len(categories_open))

    st.write("---")

    for symbol in symbols_open:
        sym_positions = [p for p in filtered_positions if p["symbol"] == symbol]
        sym_combos = [c for c in filtered_combos if c["symbol"] == symbol]
        st.subheader(f"{symbol}  ·  {len(sym_positions)} single-leg + {len(sym_combos)} combo position(s)")

        for p in sym_positions:
            entry = p["entry_price"] or 0.0
            current = p["current_price"] if p["current_price"] is not None else entry
            pnl_r = p["unrealized_pnl_r"] or 0.0
            emoji = "🟢" if pnl_r >= 0 else "🔴"

            heartbeat = p.get("last_heartbeat_at")
            stale = heartbeat is None or (now - heartbeat.astimezone(KOLKATA_TZ)) > STALE_AFTER
            stale_badge = ' <span class="stale-badge">⚠️ stale</span>' if stale else ""

            cat = category_of(p["experiment_name"])
            diag = p.get("diagnostics") or {}
            opt_sym = diag.get("option_symbol") if isinstance(diag, dict) else None
            
            if opt_sym:
                title = f"{emoji} [{cat}] {p['experiment_name']} ({opt_sym}) | {p['signal_type']} | {pnl_r:+.2f} R"
            else:
                title = f"{emoji} [{cat}] {p['experiment_name']} | {p['signal_type']} | {pnl_r:+.2f} R"

            with st.expander(title, expanded=True):
                if opt_sym:
                    st.markdown(f"**Category:** {cat}  ·  **Strategy / Setup:** {p['strategy']}  ·  📦 **Option Contract:** `{opt_sym}`{stale_badge}", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Category:** {cat}  ·  **Strategy / Setup:** {p['strategy']}{stale_badge}", unsafe_allow_html=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Entry", f"{entry:.2f}")
                m2.metric("Current Market Price", f"{current:.2f}", delta=f"{current - entry:+.2f}")
                m3.metric("Unrealized PnL", f"{pnl_r:+.2f} R")
                m4.metric("Bars Held", p.get("bars_held") or 0)

                st.markdown("**Stop Loss**")
                sl_c1, sl_c2 = st.columns(2)
                sl_c1.write(f"Initial: `{p['initial_stop_loss']:.2f}`" if p.get('initial_stop_loss') else "Initial: N/A")
                trailed = p.get('stop_loss') != p.get('initial_stop_loss')
                sl_c2.write(f"Current: `{p['stop_loss']:.2f}`" + (" 🔒 trailed" if trailed else "") if p.get('stop_loss') else "Current: N/A")

                st.markdown("**Targets** *(every strategy sets a partial target at 1.5R alongside the final target)*")
                tp_c1, tp_c2 = st.columns(2)
                tp1 = p.get('tp1')
                tp_c1.write(f"🎯 Target 1 (partial, 1.5R): `{tp1:.2f}`" if tp1 else "🎯 Target 1: N/A")
                tp_final = p.get('take_profit')
                expanded_tp = p.get('take_profit') != p.get('initial_take_profit')
                tp_c2.write(
                    (f"🏁 Final Target: `{tp_final:.2f}`" + (" 📈 expanded" if expanded_tp else ""))
                    if tp_final else "🏁 Final Target: N/A"
                )

                diag_targets = p.get("diagnostics") or {}
                extra_targets = {
                    "Nearest Zone": diag_targets.get("tp_nearest_zone"),
                    "Measured Move (100%)": diag_targets.get("tp_measured_move"),
                    "Extended (161.8%)": diag_targets.get("tp_extended_1618"),
                }
                extra_targets = {k: v for k, v in extra_targets.items() if v is not None}
                if extra_targets:
                    st.caption("Other candidate targets (pattern-derived):")
                    tp_cols = st.columns(len(extra_targets))
                    for col, (label, val) in zip(tp_cols, extra_targets.items()):
                        col.write(f"{label}: `{val:.2f}`")

                st.markdown("**Excursion**")
                ex_c1, ex_c2 = st.columns(2)
                ex_c1.write(f"MFE (best so far): `{(p.get('mfe_r') or 0.0):+.2f}R`")
                ex_c2.write(f"MAE (worst so far): `{(p.get('mae_r') or 0.0):+.2f}R`")

                st.markdown("**Conditions at Entry**")
                diag = p.get("diagnostics") or p.get("features") or {}
                if not diag:
                    st.text("No diagnostic features recorded.")
                else:
                    pills = ""
                    for k, v in diag.items():
                        if isinstance(v, (dict, list)):
                            continue
                        pills += f"<span class='factor-pill'><b>{k.replace('_', ' ').upper()}:</b> {v}</span> "
                    st.markdown(pills, unsafe_allow_html=True)

                st.caption(
                    f"Entered {format_dt(p['entry_time'])} IST · "
                    f"Last heartbeat {format_dt(heartbeat) if heartbeat else 'N/A'} IST"
                )

        for c in sym_combos:
            pnl_r = c.get("current_pnl_r") or 0.0
            emoji = "🟢" if pnl_r >= 0 else "🔴"
            cat = category_of(c["experiment_name"])
            title = f"{emoji} [{cat}] {c['experiment_name']} | {c['combo_type']} | {pnl_r:+.2f} R"

            with st.expander(title, expanded=True):
                st.markdown(f"**Category:** {cat}  ·  **Combo type:** {c['combo_type']}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Underlying Entry", f"{c.get('underlying_entry_price', 0.0):.2f}")
                m2.metric("Net Premium Paid", f"{c.get('net_premium_paid', 0.0):.2f}")
                m3.metric("Current PnL", f"{pnl_r:+.2f} R")
                max_profit = c.get("max_profit")
                m4.metric("Max Loss / Max Profit", f"{c.get('max_loss', 0.0):.2f} / {max_profit:.2f}" if max_profit is not None else f"{c.get('max_loss', 0.0):.2f} / unbounded")

                st.markdown("**Legs**")
                legs = c.get("legs") or []
                if legs:
                    leg_cols = st.columns(len(legs))
                    for col, leg in zip(leg_cols, legs):
                        entry_p = leg.get("entry_premium")
                        exit_p = leg.get("exit_premium")
                        leg_pnl = (
                            (exit_p - entry_p) if leg["side"] == "BUY" else (entry_p - exit_p)
                        ) if entry_p is not None and exit_p is not None else None
                        col.write(
                            f"**{leg['side']} {leg['option_type']}** @ {leg['strike']:.0f}\n\n"
                            f"Entry: `{entry_p:.2f}`" + (f" → Now: `{exit_p:.2f}`" if exit_p is not None else "") +
                            (f"\n\nLeg PnL: `{leg_pnl:+.2f}`" if leg_pnl is not None else "")
                        )
                else:
                    st.text("No leg data recorded.")

                st.markdown(f"**Exit rules:** Target `{c.get('target_r')}R` · Stop `{c.get('stop_r')}R` · Session end (15:25 IST)")

                diag = c.get("diagnostics") or {}
                if diag:
                    st.markdown("**Conditions at Entry**")
                    pills = "".join(
                        f"<span class='factor-pill'><b>{k.replace('_', ' ').upper()}:</b> {v}</span> "
                        for k, v in diag.items() if not isinstance(v, (dict, list))
                    )
                    st.markdown(pills, unsafe_allow_html=True)

                st.caption(f"Entered {format_dt(c['entry_time'])} IST")

        st.write("---")

# Run the fragment
render_live_trades(selected_category)
