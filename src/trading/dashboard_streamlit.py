import os
import sys
import json
import logging
from datetime import datetime, date
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

# Path Injection
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.postgres_database import PostgresDatabase

kolkata_tz = ZoneInfo("Asia/Kolkata")

def format_dt(dt):
    if dt is None:
        return "N/A"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return dt
    if dt.tzinfo is not None:
        dt = dt.astimezone(kolkata_tz)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def event_ts_ist(ev):
    """Normalise an event's timestamp to a tz-aware IST datetime for correct
    ordering and latency deltas (some legacy rows were naive/mixed-tz)."""
    ts = ev.get("timestamp")
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return None
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=kolkata_tz)
    return ts.astimezone(kolkata_tz)


def price_label(row):
    """Distinguish index-level prices from combined-premium (combo) prices so the
    dashboard never presents a NIFTY level as if it were the option premium."""
    strat = str(row.get("strategy") or row.get("setup_type") or "")
    diag = row.get("diagnostics")
    is_combo = "STRADDLE" in strat or "STRANGLE" in strat or (isinstance(diag, dict) and "combo" in diag)
    return "combined premium ₹" if is_combo else "index level"

def format_event_description(event_type, payload):
    if not payload:
        return "No details provided"
        
    try:
        if isinstance(payload, str):
            payload = json.loads(payload)

        # ── Real trade_events shapes ──────────────────────────────────────────
        if event_type == "ENTRY":
            return (f"📥 **Entry** | Price: `{payload.get('entry_price', 0.0):.2f}` "
                    f"| SL: `{payload.get('stop_loss', 0.0):.2f}` "
                    f"| TP: `{payload.get('take_profit', 0.0):.2f}`")

        elif event_type == "SL_TRAIL":
            return (f"🛡️ **SL Trailed** | SL → `{payload.get('stop_loss', 0.0):.2f}` "
                    f"| Market: `{payload.get('current_price', 0.0):.2f}` "
                    f"| MFE: `{payload.get('mfe_r', 0.0):.2f}R` "
                    f"| MAE: `{payload.get('mae_r', 0.0):.2f}R`")

        elif event_type == "EXIT":
            pnl = payload.get('final_pnl_r', 0.0)
            pnl_str = f"{pnl:+.2f} R"
            return (f"🏁 **Exit** | Price: `{payload.get('exit_price', 0.0):.2f}` "
                    f"| Reason: `{payload.get('exit_reason')}` "
                    f"| PnL: **{pnl_str}** "
                    f"| Duration: `{payload.get('duration_minutes', 0.0):.1f} mins` "
                    f"| Bars: `{payload.get('bars_held')}`")

        # ── Newer execution_auditor shapes ────────────────────────────────────
        elif event_type == "SIGNAL_GENERATED":
            return f"🎯 **Signal Generated** | Direction: `{payload.get('signal')}` | Price: `{payload.get('price', 0.0):.2f}`"

        elif event_type == "STRIKE_SELECTED":
            return f"🎳 **Strike Selected** | Symbol: `{payload.get('symbol')}` | Strike: `{payload.get('strike')}` | Expiry: `{payload.get('expiry')}`"

        elif event_type == "PREMIUM_RETRIEVED":
            return f"💰 **Premium Retrieved** | LTP: `{payload.get('premium', 0.0):.2f}` | Bid: `{payload.get('bid', 0.0):.2f}` | Ask: `{payload.get('ask', 0.0):.2f}`"

        elif event_type in ("ORDER_SUBMITTED", "CF_SUBMITTED"):
            return f"📤 **Order Submitted** | Price: `{payload.get('price', 0.0):.2f}` | SL: `{payload.get('sl', 0.0):.2f}` | TP: `{payload.get('tp', 0.0):.2f}`"

        elif event_type in ("ORDER_FILLED", "CF_FILLED"):
            return f"✅ **Filled** | Price: `{payload.get('price', 0.0):.2f}`"

        elif event_type in ("SL_MODIFIED", "CF_SL_MODIFIED"):
            return (f"🛡️ **SL Modified** | `{payload.get('old_sl', 0.0):.2f}` → `{payload.get('new_sl', 0.0):.2f}` "
                    f"| Reason: `{payload.get('reason')}` | Market: `{payload.get('price', 0.0):.2f}`")

        elif event_type in ("TP_EXPANDED", "CF_TP_EXPANDED"):
            return f"📈 **TP Expanded** | `{payload.get('old_tp', 0.0):.2f}` → `{payload.get('new_tp', 0.0):.2f}` | Reason: `{payload.get('reason')}`"

        elif event_type in ("ORDER_EXITED", "CF_EXITED"):
            pnl = payload.get('pnl_r', 0.0)
            return (f"🏁 **Exited** | Price: `{payload.get('exit_price', 0.0):.2f}` "
                    f"| Reason: `{payload.get('exit_reason')}` "
                    f"| PnL: **{pnl:+.2f} R** "
                    f"| {payload.get('duration_minutes', 0.0):.1f} mins")

        return f"🔹 `{event_type}` | {json.dumps(payload)}"
    except Exception as e:
        return f"🔹 `{event_type}` | parse error: {e} | {payload}"

st.set_page_config(
    page_title="EOD Trading Analytics & Replay",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark theme support custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #0a0b0d;
    }
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
</style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def get_db():
    return PostgresDatabase()

db = get_db()

st.title("📊 Trading Analytics")


# ══════════════════════════════════════════════════════════════════════════
# LIVE MARKET STATUS  (what's happening now + current active trades)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=10)
def load_live():
    """Latest engine-computed market status + currently OPEN positions."""
    status, open_real, open_cf = [], [], []
    try:
        status = db.get_latest_market_status()
        open_real = db.get_open_positions()          # exit_time IS NULL
        open_cf = db.get_open_counterfactuals()       # exit_time IS NULL
    except Exception as e:
        st.error(f"Live query failed: {e}")
    return status, open_real, open_cf


def render_live():
    status, open_real, open_cf = load_live()
    st.header("🟢 Live Market Status")

    if not status:
        st.info("No live market status yet — start the trader (`./run_indian_trader.sh`) during market hours.")
    else:
        # price for each symbol (used for unrealized R on open directional trades)
        price_by_symbol = {}
        cols = st.columns(len(status))
        for col, row in zip(cols, status):
            p = row.get("payload") or {}
            if isinstance(p, str):
                try: p = json.loads(p)
                except Exception: p = {}
            sym = row.get("symbol", "?")
            price_by_symbol[sym] = p.get("price")
            mv = p.get("market_view") or {}
            with col:
                st.subheader(sym.replace("NSE:", "").replace("-INDEX", ""))
                st.metric("Price", f"{p.get('price', 0):.2f}")
                st.caption(f"as of {format_dt(row.get('timestamp'))}")
                st.write(f"**View:** {mv.get('dominant_direction', 'n/a')} "
                         f"(dir {mv.get('directional_score', 0):+.2f})")
                st.write(f"**Regime:** {mv.get('regime_label', p.get('regime', 'n/a'))}")
                st.write(f"**Bias:** {p.get('daily_bias', 'n/a')}  |  **RVOL:** {p.get('rvol') or 0:.2f}")
                pats = ", ".join(pt.get("name", "") for pt in mv.get("patterns", [])) or "none"
                st.write(f"**Patterns:** {pats}")
                st.caption(f"active real: {p.get('active_real_trades', 0)} | "
                           f"shadow: {p.get('active_counterfactuals', 0)}")

        # ── Active REAL trades (open) ──
        st.subheader(f"🎯 Active Trades ({len(open_real)})")
        if not open_real:
            st.caption("No open real positions.")
        for t in open_real:
            sym = t.get("symbol", "?")
            sig = t.get("signal_type") or t.get("setup_type") or "?"
            entry = t.get("entry_price") or 0.0
            sld = t.get("stop_loss_distance") or 0.0
            cur = price_by_symbol.get(sym)
            # Unrealized R for directional trades only (combos are premium-space).
            is_combo = str(sig) in ("STRADDLE", "STRANGLE")
            if cur and sld and not is_combo:
                if "CALL" in str(sig):
                    ur = (cur - entry) / sld
                else:
                    ur = (entry - cur) / sld
                ur_str = f"{ur:+.2f} R (live)"
            else:
                ur_str = "combo (premium-space)" if is_combo else "n/a"
            st.markdown(
                f"**{sym.replace('NSE:','')}** · `{sig}` · [{t.get('experiment_name','?')}] — "
                f"entry `{entry:.2f}`, SL `{t.get('stop_loss') or 0:.2f}`, TP `{t.get('take_profit') or 0:.2f}`, "
                f"bars `{t.get('bars_held', 0)}` · **{ur_str}**"
            )

        # ── Active shadow trades by experiment ──
        if open_cf:
            from collections import Counter
            by_exp = Counter(c.get("experiment_name", "?") for c in open_cf)
            st.subheader(f"👻 Active Shadow Trades ({len(open_cf)})")
            st.write(" · ".join(f"{k}: {v}" for k, v in sorted(by_exp.items())))

    st.divider()


render_live()

# Load available report dates (historical replay is optional; live works without it)
reports_dir = os.path.join(project_root, "reports")
dates = []
if os.path.exists(reports_dir):
    for f in os.listdir(reports_dir):
        if f.endswith(".json") and not f.startswith("daily_"):
            dates.append(f.replace(".json", ""))
dates.sort(reverse=True)

if not dates:
    st.info("📁 No end-of-day reports yet — showing live status only. Reports appear after 15:35 IST.")
    st.stop()

# Header layout
st.header("🕰️ Historical Session Replay")
selected_date = st.selectbox("Select Session Date", dates)

# Data querying
@st.cache_data(ttl=10)
def load_data(report_date):
    trades = []
    candidates = []
    events = []
    eod_report = None

    # Load EOD report JSON file
    report_file = os.path.join(project_root, "reports", f"{report_date}.json")
    if os.path.exists(report_file):
        try:
            with open(report_file, 'r') as f:
                eod_report = json.load(f)
        except Exception as e:
            pass

    # Query TimescaleDB
    try:
        with db._get_connection() as conn:
            # 1. Real Trades
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT trade_id, candidate_id, entry_time, exit_time, symbol, strategy, 
                           entry_price, exit_price, pnl, exit_reason, mfe_r, mae_r, final_pnl_r, 
                           bars_held, stop_loss, take_profit, experiment_name, diagnostics, features
                    FROM trade_performance
                    WHERE DATE(entry_time AT TIME ZONE 'Asia/Kolkata') = %s
                    ORDER BY entry_time ASC
                """, (report_date,))
                cols = [desc[0] for desc in cursor.description]
                trades = [dict(zip(cols, row)) for row in cursor.fetchall()]

            # 2. Candidate Opportunities
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT candidate_id, timestamp, symbol, signal_type, setup_type, 
                           rejection_reasons, primary_rejection_reason, entry_price, 
                           stop_loss, take_profit, exit_time, exit_price, mfe_r, mae_r, final_pnl_r, 
                           experiment_name, diagnostics
                    FROM counterfactual_results
                    WHERE DATE(timestamp AT TIME ZONE 'Asia/Kolkata') = %s
                    ORDER BY timestamp ASC
                """, (report_date,))
                cols = [desc[0] for desc in cursor.description]
                candidates = [dict(zip(cols, row)) for row in cursor.fetchall()]

            # 3. Trade lifecycle events (trade_events + counterfactual_trade_events)
            trade_evts = []
            cf_evts = []
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT event_id, trade_id, NULL AS candidate_id, timestamp, event_type, payload
                    FROM trade_events
                    WHERE DATE(timestamp AT TIME ZONE 'Asia/Kolkata') = %s
                    ORDER BY timestamp ASC
                """, (report_date,))
                cols = [desc[0] for desc in cursor.description]
                trade_evts = [dict(zip(cols, row)) for row in cursor.fetchall()]

            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT event_id, NULL AS trade_id, candidate_id, timestamp, event_type, payload
                    FROM counterfactual_trade_events
                    WHERE DATE(timestamp AT TIME ZONE 'Asia/Kolkata') = %s
                    ORDER BY timestamp ASC
                """, (report_date,))
                cols = [desc[0] for desc in cursor.description]
                cf_evts = [dict(zip(cols, row)) for row in cursor.fetchall()]

            exec_evts = []
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT event_id, trade_id, candidate_id, timestamp, event_type, payload
                        FROM execution_events
                        WHERE DATE(timestamp AT TIME ZONE 'Asia/Kolkata') = %s
                        ORDER BY timestamp ASC
                    """, (report_date,))
                    cols = [desc[0] for desc in cursor.description]
                    exec_evts = [dict(zip(cols, row)) for row in cursor.fetchall()]
            except Exception as ex_err:
                pass

            events = trade_evts + cf_evts + exec_evts

    except Exception as e:
        st.error(f"Failed to query database: {e}")

    return {
        "eod_report": eod_report,
        "trades": trades,
        "candidates": candidates,
        "events": events
    }

data = load_data(selected_date)

# Executive Metrics Row
eod = data["eod_report"] or {}
exec_summary = eod.get("sections", {}).get("executive_summary", {})
real_pnl = exec_summary.get("real", {}).get("total_pnl_r", 0.0)
win_rate = exec_summary.get("real", {}).get("win_rate", 0.0) * 100
expectancy = exec_summary.get("real", {}).get("expectancy", 0.0)
shadow_pnl = exec_summary.get("cf", {}).get("total_pnl_r", 0.0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Realized PnL", f"{real_pnl:+.2f} R", delta=f"{real_pnl:.2f} R" if real_pnl != 0 else None)
col2.metric("Win Rate", f"{win_rate:.0f}%")
col3.metric("Expectancy", f"{expectancy:.2f} R")
col4.metric("Shadow PnL (Counterfactual)", f"{shadow_pnl:+.2f} R")
col5.metric("Total Trades", len(data["trades"]))

st.write("---")

# Strategy Bifurcation Selector
all_strategies = sorted(list(set(
    [t["strategy"] for t in data["trades"]] + 
    [c["setup_type"] or c["strategy"] for c in data["candidates"] if c.get("setup_type") or c.get("strategy")]
)))
all_strategies = ["All"] + all_strategies

selected_strat = st.radio(
    "Filter Entire Session by Strategy",
    all_strategies,
    horizontal=True
)

# Filtering helper
filtered_trades = data["trades"]
filtered_candidates = data["candidates"]

if selected_strat != "All":
    filtered_trades = [t for t in data["trades"] if t["strategy"] == selected_strat]
    filtered_candidates = [c for c in data["candidates"] if (c.get("setup_type") or c.get("strategy_version")) == selected_strat]

# Only CLOSED trades are "realized". Open positions (exit_time IS NULL) have
# pnl 0.0 and belong in the Live panel, not counted here as +0.00 R winners.
closed_trades = [t for t in filtered_trades if t.get("exit_time") is not None]

tab1, tab2 = st.tabs([
    f"📈 Realized Positions ({len(closed_trades)})",
    f"👻 Counterfactual / Shadow Trades ({len(filtered_candidates)})"
])

# 1. Realized positions tab
with tab1:
    if not closed_trades:
        st.info("No realized (closed) trades for this date / filter.")
    else:
        for t in closed_trades:
            pnl_val = t["pnl"] or 0.0
            emoji = "🟢" if pnl_val >= 0 else "🔴"
            title = f"{emoji} {t['symbol']} | {t['strategy']} | {pnl_val:+.2f} R"
            
            with st.expander(title):
                m_col1, m_col2 = st.columns(2)
                
                with m_col1:
                    st.subheader("Milestones & Details")
                    st.write(f"⏱️ **Executed At:** {format_dt(t['entry_time'])}")
                    st.write(f"🛑 **Outcome/Exit Reason:** {t['exit_reason'] or 'OPEN'}")
                    st.write(f"💸 **PnL:** {pnl_val:+.2f} R")
                    st.write(f"🎯 **Levels ({price_label(t)}):** Entry: {t['entry_price']:.2f} | Exit: {t['exit_price'] or 0.0:.2f} | SL: {t['stop_loss'] or 0.0:.2f} | TP: {t['take_profit'] or 0.0:.2f}")
                    st.caption("P&L shown as R-multiples of the trade's defined risk (not rupees).")
                    st.write(f"📦 **Experiment / Version:** {t['experiment_name']}")

                with m_col2:
                    st.subheader("Attribution / Trigger Factors")
                    diag = t["diagnostics"] or t["features"] or {}
                    if not diag:
                        st.text("No diagnostic features recorded.")
                    else:
                        for k, v in diag.items():
                            if isinstance(v, (dict, list)):
                                continue
                            formatted_k = k.replace("_", " ").upper()
                            st.markdown(f"<div class='factor-pill'><b>{formatted_k}:</b> {v}</div>", unsafe_allow_html=True)
                
                st.subheader("Execution Latency Timeline")
                t_events = [e for e in data["events"] if e["trade_id"] == t["trade_id"]]
                # Sort chronologically on normalised IST timestamps (events come
                # from three tables and were previously unsorted + mixed-tz, making
                # latency deltas negative/meaningless). Drop rows with no timestamp.
                t_events = sorted((e for e in t_events if event_ts_ist(e)), key=event_ts_ist)
                if not t_events:
                    st.text("No audit trace events found for this trade.")
                else:
                    prev_t = None
                    for ev in t_events:
                        curr_t = event_ts_ist(ev)
                        if curr_t is None:
                            continue
                        latency = ""
                        if prev_t is not None:
                            diff_ms = int((curr_t - prev_t).total_seconds() * 1000)
                            latency = f"*(+{diff_ms}ms latency)*" if diff_ms >= 0 else ""
                        prev_t = curr_t
                        st.markdown(f"- **{curr_t.strftime('%H:%M:%S.%f')[:-3]}** {latency} &mdash; {format_event_description(ev['event_type'], ev['payload'])}")

# 2. Counterfactual missed opportunities tab
with tab2:
    if not filtered_candidates:
        st.info("No counterfactual signals match this strategy filter.")
    else:
        for c in filtered_candidates:
            pnl_val = c["final_pnl_r"] or 0.0
            title = f"👻 {c['symbol']} | {c['setup_type'] or c['strategy']} | Blocked: {c['primary_rejection_reason']} | {pnl_val:+.2f} R"
            
            with st.expander(title):
                m_col1, m_col2 = st.columns(2)
                
                with m_col1:
                    st.subheader("Milestones & Details")
                    st.write(f"⏱️ **Triggered At:** {format_dt(c['timestamp'])}")
                    st.write(f"🛑 **Primary Rejection:** {c['primary_rejection_reason']}")
                    st.write(f"⛔ **All Rejections:** {c['rejection_reasons']}")
                    st.write(f"💸 **Simulated Outcome:** {pnl_val:+.2f} R")
                    st.write(f"🎯 **Levels ({price_label(c)}):** Entry: {c['entry_price'] or 0.0:.2f} | Exit: {c['exit_price'] or 0.0:.2f} | SL: {c['stop_loss'] or 0.0:.2f} | TP: {c['take_profit'] or 0.0:.2f}")

                with m_col2:
                    st.subheader("Attribution / Trigger Factors")
                    diag = c["diagnostics"] or {}
                    if not diag:
                        st.text("No diagnostic features recorded.")
                    else:
                        for k, v in diag.items():
                            if isinstance(v, (dict, list)):
                                continue
                            formatted_k = k.replace("_", " ").upper()
                            st.markdown(f"<div class='factor-pill'><b>{formatted_k}:</b> {v}</div>", unsafe_allow_html=True)

                st.subheader("Execution Latency Timeline")
                t_events = [e for e in data["events"] if e["candidate_id"] == c["candidate_id"]]
                t_events = sorted((e for e in t_events if event_ts_ist(e)), key=event_ts_ist)
                if not t_events:
                    st.text("No audit trace events found for this signal.")
                else:
                    prev_t = None
                    for ev in t_events:
                        curr_t = event_ts_ist(ev)
                        if curr_t is None:
                            continue
                        latency = ""
                        if prev_t is not None:
                            diff_ms = int((curr_t - prev_t).total_seconds() * 1000)
                            latency = f"*(+{diff_ms}ms latency)*" if diff_ms >= 0 else ""
                        prev_t = curr_t
                        st.markdown(f"- **{curr_t.strftime('%H:%M:%S.%f')[:-3]}** {latency} &mdash; {format_event_description(ev['event_type'], ev['payload'])}")
