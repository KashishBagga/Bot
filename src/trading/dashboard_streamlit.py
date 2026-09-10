import os
import sys
import json
import logging
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

# Path Injection
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.postgres_database import PostgresDatabase

logger = logging.getLogger(__name__)

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

def adjust_shifted_timestamp(curr_t, ref_t):
    if curr_t is None or ref_t is None:
        return curr_t
    if curr_t.tzinfo is None or ref_t.tzinfo is None:
        return curr_t
    diff = (curr_t - ref_t).total_seconds()
    if diff > 18000:
        curr_t = curr_t - timedelta(hours=5, minutes=30)
    elif diff < -18000:
        curr_t = curr_t + timedelta(hours=5, minutes=30)
    return curr_t

def format_event_description(event_type, payload):
    if not payload:
        return "No details provided"
        
    try:
        if isinstance(payload, str):
            payload = json.loads(payload)

        def get_f(key, default=0.0):
            val = payload.get(key)
            if val is None:
                return default
            try:
                return float(val)
            except Exception:
                return default

        # ── Real trade_events shapes ──────────────────────────────────────────
        if event_type == "ENTRY":
            return (f"📥 **Entry** | Price: `{get_f('entry_price'):.2f}` "
                    f"| SL: `{get_f('stop_loss'):.2f}` "
                    f"| TP: `{get_f('take_profit'):.2f}`")

        elif event_type == "SL_TRAIL":
            return (f"🛡️ **SL Trailed** | SL → `{get_f('stop_loss'):.2f}` "
                    f"| Market: `{get_f('current_price'):.2f}` "
                    f"| MFE: `{get_f('mfe_r'):.2f}R` "
                    f"| MAE: `{get_f('mae_r'):.2f}R`")

        elif event_type == "EXIT":
            pnl = get_f('final_pnl_r')
            pnl_str = f"{pnl:+.2f} R"
            return (f"🏁 **Exit** | Price: `{get_f('exit_price'):.2f}` "
                    f"| Reason: `{payload.get('exit_reason')}` "
                    f"| PnL: **{pnl_str}** "
                    f"| Duration: `{get_f('duration_minutes'):.1f} mins` "
                    f"| Bars: `{payload.get('bars_held')}`")

        # ── Newer execution_auditor shapes ────────────────────────────────────
        elif event_type == "SIGNAL_GENERATED":
            return f"🎯 **Signal Generated** | Direction: `{payload.get('signal')}` | Price: `{get_f('price'):.2f}`"

        elif event_type == "STRIKE_SELECTED":
            return f"🎳 **Strike Selected** | Symbol: `{payload.get('symbol')}` | Strike: `{payload.get('strike')}` | Expiry: `{payload.get('expiry')}`"

        elif event_type == "PREMIUM_RETRIEVED":
            return f"💰 **Premium Retrieved** | LTP: `{get_f('premium'):.2f}` | Bid: `{get_f('bid'):.2f}` | Ask: `{get_f('ask'):.2f}`"

        elif event_type in ("ORDER_SUBMITTED", "CF_SUBMITTED"):
            return f"📤 **Order Submitted** | Price: `{get_f('price'):.2f}` | SL: `{get_f('sl'):.2f}` | TP: `{get_f('tp'):.2f}`"

        elif event_type in ("ORDER_FILLED", "CF_FILLED"):
            return f"✅ **Filled** | Price: `{get_f('price'):.2f}`"

        elif event_type in ("SL_MODIFIED", "CF_SL_MODIFIED"):
            return (f"🛡️ **SL Modified** | `{get_f('old_sl'):.2f}` → `{get_f('new_sl'):.2f}` "
                    f"| Reason: `{payload.get('reason')}` | Market: `{get_f('price'):.2f}`")

        elif event_type in ("TP_EXPANDED", "CF_TP_EXPANDED"):
            return f"📈 **TP Expanded** | `{get_f('old_tp'):.2f}` → `{get_f('new_tp'):.2f}` | Reason: `{payload.get('reason')}`"

        elif event_type in ("ORDER_EXITED", "CF_EXITED"):
            pnl = get_f('pnl_r')
            return (f"🏁 **Exited** | Price: `{get_f('exit_price'):.2f}` "
                    f"| Reason: `{payload.get('exit_reason')}` "
                    f"| PnL: **{pnl:+.2f} R** "
                    f"| {get_f('duration_minutes'):.1f} mins")

        return f"🔹 `{event_type}` | {json.dumps(payload)}"
    except Exception as e:
        return f"🔹 `{event_type}` | parse error: {e} | {payload}"

def format_combo_event_description(event_type, payload):
    """Combo (multi-leg) ENTRY/EXIT events share the ENTRY/EXIT event_type
    names with single-leg trade_events, but a different payload shape
    (combo_type/legs/net_premium_paid instead of entry_price/stop_loss) —
    reusing format_event_description() would render bogus 0.00 fields."""
    if not payload:
        return "No details provided"
    try:
        if isinstance(payload, str):
            payload = json.loads(payload)

        if event_type == "ENTRY":
            return (f"📥 **Combo Entry** | Type: `{payload.get('combo_type')}` "
                    f"| Net premium: `{payload.get('net_premium_paid', 0.0):.2f}` "
                    f"| Max loss: `{payload.get('max_loss', 0.0):.2f}`")
        elif event_type == "EXIT":
            pnl = payload.get('final_pnl_r') or 0.0
            return (f"🏁 **Combo Exit** | Reason: `{payload.get('exit_reason')}` "
                    f"| PnL: **{pnl:+.2f} R** "
                    f"| Duration: `{payload.get('duration_minutes', 0.0):.1f} mins`")
        return f"🔹 `{event_type}` | {json.dumps(payload)}"
    except Exception as e:
        return f"🔹 `{event_type}` | parse error: {e} | {payload}"

st.set_page_config(
    page_title="EOD Trading Analytics & Replay",
    page_icon="📊",
    layout="wide",
    # "expanded" (not "collapsed") so the multipage nav — including the new
    # Live Trades page in pages/ — is visible without the user hunting for it.
    initial_sidebar_state="expanded"
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

# Load available report dates
reports_dir = os.path.join(project_root, "reports")
dates = []
if os.path.exists(reports_dir):
    for f in os.listdir(reports_dir):
        if f.endswith(".json") and not f.startswith("daily_"):
            dates.append(f.replace(".json", ""))
dates.sort(reverse=True)

if not dates:
    st.warning("⚠️ No daily reports found in reports/ directory.")
    st.stop()

# Header layout
st.title("📊 Trading Session Analytics & Replay")
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
                      AND (valid IS NULL OR valid = TRUE)
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
                      AND (valid IS NULL OR valid = TRUE)
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

            # 4. Multi-leg combo trades (Butterfly/Straddle/IronCondor/etc) — a
            # separate table+shape from single-leg trades. Without this, every
            # combo trade the report/EOD-summary counts is invisible in the
            # per-trade drill-down below (see trade_review.py's same fix).
            combo_trades, combo_candidates = [], []
            combo_events = []
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT combo_id, entry_time, exit_time, symbol, experiment_name,
                               combo_type, setup_type, underlying_entry_price, underlying_exit_price,
                               legs, net_premium_paid, max_loss, max_profit, target_r, stop_r,
                               final_pnl_r, exit_reason, duration_minutes, diagnostics
                        FROM combo_trades
                        WHERE DATE(entry_time AT TIME ZONE 'Asia/Kolkata') = %s
                          AND (valid IS NULL OR valid = TRUE)
                        ORDER BY entry_time ASC
                    """, (report_date,))
                    cols = [desc[0] for desc in cursor.description]
                    combo_trades = [dict(zip(cols, row)) for row in cursor.fetchall()]

                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT combo_id, entry_time, exit_time, symbol, experiment_name,
                               combo_type, setup_type, rejection_reasons, primary_rejection_reason,
                               underlying_entry_price, underlying_exit_price, legs, net_premium_paid,
                               max_loss, max_profit, target_r, stop_r, final_pnl_r, exit_reason,
                               duration_minutes, diagnostics
                        FROM counterfactual_combo_results
                        WHERE DATE(entry_time AT TIME ZONE 'Asia/Kolkata') = %s
                          AND (valid IS NULL OR valid = TRUE)
                        ORDER BY entry_time ASC
                    """, (report_date,))
                    cols = [desc[0] for desc in cursor.description]
                    combo_candidates = [dict(zip(cols, row)) for row in cursor.fetchall()]

                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT event_id, combo_id, timestamp, event_type, payload
                        FROM combo_trade_events
                        WHERE DATE(timestamp AT TIME ZONE 'Asia/Kolkata') = %s
                        ORDER BY timestamp ASC
                    """, (report_date,))
                    cols = [desc[0] for desc in cursor.description]
                    combo_events = [dict(zip(cols, row)) for row in cursor.fetchall()]

                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT event_id, combo_id, timestamp, event_type, payload
                        FROM counterfactual_combo_events
                        WHERE DATE(timestamp AT TIME ZONE 'Asia/Kolkata') = %s
                        ORDER BY timestamp ASC
                    """, (report_date,))
                    cols = [desc[0] for desc in cursor.description]
                    combo_events += [dict(zip(cols, row)) for row in cursor.fetchall()]
            except Exception as combo_err:
                logger.warning(f"Combo trade query failed: {combo_err}")

    except Exception as e:
        st.error(f"Failed to query database: {e}")

    return {
        "eod_report": eod_report,
        "trades": trades,
        "candidates": candidates,
        "events": events,
        "combo_trades": combo_trades,
        "combo_candidates": combo_candidates,
        "combo_events": combo_events,
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
col5.metric("Total Trades", len(data["trades"]) + len(data["combo_trades"]))

st.write("---")

# Strategy Bifurcation Selector
all_strategies = sorted(list(set(
    [t["strategy"] for t in data["trades"]] +
    [c["setup_type"] or c["strategy"] for c in data["candidates"] if c.get("setup_type") or c.get("strategy")] +
    [c["setup_type"] or c["combo_type"] for c in data["combo_trades"]] +
    [c["setup_type"] or c["combo_type"] for c in data["combo_candidates"]]
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
filtered_combo_trades = data["combo_trades"]
filtered_combo_candidates = data["combo_candidates"]

if selected_strat != "All":
    filtered_trades = [t for t in data["trades"] if t["strategy"] == selected_strat]
    filtered_candidates = [c for c in data["candidates"] if (c["setup_type"] or c["strategy"]) == selected_strat]
    filtered_combo_trades = [c for c in data["combo_trades"] if (c["setup_type"] or c["combo_type"]) == selected_strat]
    filtered_combo_candidates = [c for c in data["combo_candidates"] if (c["setup_type"] or c["combo_type"]) == selected_strat]

tab1, tab2, tab3 = st.tabs([
    f"📈 Realized Positions ({len(filtered_trades) + len(filtered_combo_trades)})",
    f"👻 Counterfactual Missed Opportunities ({len(filtered_candidates) + len(filtered_combo_candidates)})",
    "🔮 Tomorrow's Outlook & Accuracy",
])

# 1. Realized positions tab
with tab1:
    if not filtered_trades:
        st.info("No realized trades match this strategy filter.")
    else:
        for t in filtered_trades:
            pnl_val = t["final_pnl_r"] or 0.0
            emoji = "🟢" if pnl_val >= 0 else "🔴"
            # Pre-filter events to find option contract symbol and construct timeline
            t_events = [e for e in data["events"] if e["trade_id"] == t["trade_id"] or (t.get("candidate_id") and e["candidate_id"] == t["candidate_id"])]
            
            # Heal shifted timestamps and sort timeline events chronologically
            ref_time = t["entry_time"]
            for ev in t_events:
                ev["timestamp"] = adjust_shifted_timestamp(ev["timestamp"], ref_time)
            t_events = sorted(t_events, key=lambda x: x["timestamp"])

            opt_sym = None
            for ev in t_events:
                pld = ev.get("payload") or {}
                if isinstance(pld, str):
                    try:
                        pld = json.loads(pld)
                    except Exception:
                        pld = {}
                cand_sym = pld.get("symbol") or pld.get("option_symbol")
                if cand_sym and "INDEX" not in str(cand_sym):
                    opt_sym = cand_sym
                    break

            opt_display = f" ({opt_sym})" if opt_sym else ""
            title = f"{emoji} {t['symbol']}{opt_display} | {t['strategy']} | {pnl_val:+.2f} R"

            with st.expander(title):
                m_col1, m_col2 = st.columns(2)
                
                with m_col1:
                    st.subheader("Milestones & Details")
                    st.write(f"⏱️ **Executed At:** {format_dt(t['entry_time'])}")
                    st.write(f"🛑 **Outcome/Exit Reason:** {t['exit_reason'] or 'OPEN'}")
                    st.write(f"💸 **PnL:** {pnl_val:+.2f} R")
                    st.write(f"🎯 **Target / Exit:** Entry: {t['entry_price']:.2f} | Exit: {t['exit_price'] or 0.0:.2f} | SL: {t['stop_loss'] or 0.0:.2f} | TP: {t['take_profit'] or 0.0:.2f}")
                    if opt_sym:
                        st.markdown(f"🎳 **Option Contract:** `{opt_sym}`")
                    else:
                        st.markdown("🎳 **Option Contract:** Index / Spot Only")
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
                if not t_events:
                    st.text("No audit trace events found for this trade.")
                else:
                    prev_t = None
                    for ev in t_events:
                        curr_t = ev["timestamp"]
                        curr_t_local = curr_t.astimezone(kolkata_tz) if curr_t.tzinfo else curr_t
                        latency = ""
                        if prev_t:
                            diff_ms = int((curr_t - prev_t).total_seconds() * 1000)
                            latency = f"*(+{diff_ms}ms latency)*"
                        prev_t = curr_t
                        
                        st.markdown(f"- **{curr_t_local.strftime('%H:%M:%S.%f')[:-3]}** {latency} &mdash; {format_event_description(ev['event_type'], ev['payload'])}")

    # Multi-leg combo trades (Butterfly/Straddle/IronCondor/etc) — same table
    # shape as the Live Trades page, rendered here for closed/historical combos.
    if filtered_combo_trades:
        st.write("---")
        st.caption("Multi-leg (combo) trades")
        for c in filtered_combo_trades:
            pnl_val = c["final_pnl_r"] or 0.0
            emoji = "🟢" if pnl_val >= 0 else "🔴"
            c_events = [e for e in data["combo_events"] if e["combo_id"] == c["combo_id"]]
            c_events = sorted(c_events, key=lambda x: x["timestamp"])

            title = f"{emoji} {c['symbol']} | {c['combo_type']} (combo) | {pnl_val:+.2f} R"

            with st.expander(title):
                m_col1, m_col2 = st.columns(2)

                with m_col1:
                    st.subheader("Milestones & Details")
                    st.write(f"⏱️ **Executed At:** {format_dt(c['entry_time'])}")
                    st.write(f"🛑 **Outcome/Exit Reason:** {c['exit_reason'] or 'OPEN'}")
                    st.write(f"💸 **PnL:** {pnl_val:+.2f} R")
                    st.write(
                        f"🎯 **Underlying:** Entry: {c['underlying_entry_price'] or 0.0:.2f} "
                        f"| Exit: {c['underlying_exit_price'] or 0.0:.2f}"
                    )
                    max_profit = c.get("max_profit")
                    st.write(
                        f"💰 **Net Premium:** {c['net_premium_paid'] or 0.0:.2f} "
                        f"| Max Loss: {c['max_loss'] or 0.0:.2f} "
                        f"| Max Profit: {f'{max_profit:.2f}' if max_profit is not None else 'unbounded'}"
                    )
                    legs = c.get("legs") or []
                    if legs:
                        leg_strs = ", ".join(
                            f"{leg['side']} {leg['option_type']} @ {leg['strike']:.0f}" for leg in legs
                        )
                        st.markdown(f"🦵 **Legs:** {leg_strs}")
                    st.write(f"📦 **Experiment / Version:** {c['experiment_name']}")

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

                st.subheader("Execution Timeline")
                if not c_events:
                    st.text("No audit trace events found for this combo trade.")
                else:
                    prev_t = None
                    for ev in c_events:
                        curr_t = ev["timestamp"]
                        curr_t_local = curr_t.astimezone(kolkata_tz) if curr_t.tzinfo else curr_t
                        latency = ""
                        if prev_t:
                            diff_ms = int((curr_t - prev_t).total_seconds() * 1000)
                            latency = f"*(+{diff_ms}ms latency)*"
                        prev_t = curr_t
                        st.markdown(f"- **{curr_t_local.strftime('%H:%M:%S.%f')[:-3]}** {latency} &mdash; {format_combo_event_description(ev['event_type'], ev['payload'])}")

# 2. Counterfactual missed opportunities tab
with tab2:
    if not filtered_candidates:
        st.info("No counterfactual signals match this strategy filter.")
    else:
        for c in filtered_candidates:
            pnl_val = c["final_pnl_r"] or 0.0
            # Pre-filter events to find option contract symbol and construct timeline
            t_events = [e for e in data["events"] if e["candidate_id"] == c["candidate_id"]]
            
            # Heal shifted timestamps and sort timeline events chronologically
            ref_time = c["timestamp"]
            for ev in t_events:
                ev["timestamp"] = adjust_shifted_timestamp(ev["timestamp"], ref_time)
            t_events = sorted(t_events, key=lambda x: x["timestamp"])

            opt_sym = None
            for ev in t_events:
                pld = ev.get("payload") or {}
                if isinstance(pld, str):
                    try:
                        pld = json.loads(pld)
                    except Exception:
                        pld = {}
                cand_sym = pld.get("symbol") or pld.get("option_symbol")
                if cand_sym and "INDEX" not in str(cand_sym):
                    opt_sym = cand_sym
                    break

            opt_display = f" ({opt_sym})" if opt_sym else ""
            title = f"👻 {c['symbol']}{opt_display} | {c['setup_type'] or c['strategy']} | Blocked: {c['primary_rejection_reason']} | {pnl_val:+.2f} R"

            with st.expander(title):
                m_col1, m_col2 = st.columns(2)
                
                with m_col1:
                    st.subheader("Milestones & Details")
                    st.write(f"⏱️ **Triggered At:** {format_dt(c['timestamp'])}")
                    st.write(f"🛑 **Primary Rejection:** {c['primary_rejection_reason']}")
                    st.write(f"⛔ **All Rejections:** {c['rejection_reasons']}")
                    st.write(f"💸 **Simulated Outcome:** {pnl_val:+.2f} R")
                    st.write(f"🎯 **Target / Exit:** Entry: {c['entry_price'] or 0.0:.2f} | Exit: {c['exit_price'] or 0.0:.2f} | SL: {c['stop_loss'] or 0.0:.2f} | TP: {c['take_profit'] or 0.0:.2f}")
                    if opt_sym:
                        st.markdown(f"🎳 **Option Contract:** `{opt_sym}`")
                    else:
                        st.markdown("🎳 **Option Contract:** Index / Spot Only")

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
                if not t_events:
                    st.text("No audit trace events found for this signal.")
                else:
                    prev_t = None
                    for ev in t_events:
                        curr_t = ev["timestamp"]
                        curr_t_local = curr_t.astimezone(kolkata_tz) if curr_t.tzinfo else curr_t
                        latency = ""
                        if prev_t:
                            diff_ms = int((curr_t - prev_t).total_seconds() * 1000)
                            latency = f"*(+{diff_ms}ms latency)*"
                        prev_t = curr_t

                        st.markdown(f"- **{curr_t_local.strftime('%H:%M:%S.%f')[:-3]}** {latency} &mdash; {format_event_description(ev['event_type'], ev['payload'])}")

    # Multi-leg combo counterfactuals (rejected combo setups replayed as shadow trades)
    if filtered_combo_candidates:
        st.write("---")
        st.caption("Multi-leg (combo) counterfactuals")
        for c in filtered_combo_candidates:
            pnl_val = c["final_pnl_r"] or 0.0
            c_events = [e for e in data["combo_events"] if e["combo_id"] == c["combo_id"]]
            c_events = sorted(c_events, key=lambda x: x["timestamp"])

            title = f"👻 {c['symbol']} | {c['combo_type']} (combo) | Blocked: {c['primary_rejection_reason']} | {pnl_val:+.2f} R"

            with st.expander(title):
                m_col1, m_col2 = st.columns(2)

                with m_col1:
                    st.subheader("Milestones & Details")
                    st.write(f"⏱️ **Triggered At:** {format_dt(c['entry_time'])}")
                    st.write(f"🛑 **Primary Rejection:** {c['primary_rejection_reason']}")
                    st.write(f"⛔ **All Rejections:** {c['rejection_reasons']}")
                    st.write(f"💸 **Simulated Outcome:** {pnl_val:+.2f} R")
                    st.write(
                        f"🎯 **Underlying:** Entry: {c['underlying_entry_price'] or 0.0:.2f} "
                        f"| Exit: {c['underlying_exit_price'] or 0.0:.2f}"
                    )
                    max_profit = c.get("max_profit")
                    st.write(
                        f"💰 **Net Premium:** {c['net_premium_paid'] or 0.0:.2f} "
                        f"| Max Loss: {c['max_loss'] or 0.0:.2f} "
                        f"| Max Profit: {f'{max_profit:.2f}' if max_profit is not None else 'unbounded'}"
                    )
                    legs = c.get("legs") or []
                    if legs:
                        leg_strs = ", ".join(
                            f"{leg['side']} {leg['option_type']} @ {leg['strike']:.0f}" for leg in legs
                        )
                        st.markdown(f"🦵 **Legs:** {leg_strs}")

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

                st.subheader("Execution Timeline")
                if not c_events:
                    st.text("No audit trace events found for this combo signal.")
                else:
                    prev_t = None
                    for ev in c_events:
                        curr_t = ev["timestamp"]
                        curr_t_local = curr_t.astimezone(kolkata_tz) if curr_t.tzinfo else curr_t
                        latency = ""
                        if prev_t:
                            diff_ms = int((curr_t - prev_t).total_seconds() * 1000)
                            latency = f"*(+{diff_ms}ms latency)*"
                        prev_t = curr_t
                        st.markdown(f"- **{curr_t_local.strftime('%H:%M:%S.%f')[:-3]}** {latency} &mdash; {format_combo_event_description(ev['event_type'], ev['payload'])}")

# 3. Tomorrow's Outlook & Accuracy tab — what the system expects next session,
# and whether the outlook it wrote for *this* session actually came true.
with tab3:
    outlook_sections = eod.get("sections", {})
    market_state = outlook_sections.get("market_state_outlook", {})
    tomorrow = outlook_sections.get("tomorrow_outlook", {})
    accuracy = outlook_sections.get("outlook_accuracy", {})

    st.subheader(f"What the system expects after {selected_date}")
    st.caption(
        "Scenario preparation, not prediction — same convention used in the report itself. "
        "No overnight/global-cues feed exists, so the gap call is a structural read from "
        "the session's own close."
    )

    obs = market_state.get("observations") or tomorrow.get("observations") or []
    if obs:
        for o in obs:
            st.markdown(f"- {o}")

    scenarios = market_state.get("scenarios") or tomorrow.get("scenarios") or []
    if scenarios:
        st.markdown("**Scenarios:**")
        for s in scenarios:
            st.markdown(f"**{s['name']} ({s['pct']}%)** — {s['desc']}")

    watch_levels = market_state.get("watch_levels") or tomorrow.get("watch_levels") or []
    if watch_levels:
        st.markdown("**Key levels to watch:**")
        st.dataframe(pd.DataFrame(watch_levels), use_container_width=True, hide_index=True)

    pref_col, avoid_col = st.columns(2)
    pref_col.success(f"**Prefer:** {market_state.get('prefer') or tomorrow.get('prefer') or '—'}")
    avoid_col.error(f"**Avoid:** {market_state.get('avoid') or tomorrow.get('avoid') or '—'}")

    playbooks = tomorrow.get("playbooks") or {}
    for key, pb in playbooks.items():
        gap = pb.get("gap_call") or {}
        with st.expander(f"{key.upper()} — Gap call & trade conditions"):
            st.markdown(
                f"**Gap call:** {gap.get('label', 'N/A')} — **{gap.get('likely_pct', '—')}% likely** "
                f"(vs. {gap.get('low_chance_label', 'N/A')}, {gap.get('low_chance_pct', '—')}%)"
            )
            st.caption(gap.get("reason", ""))
            res_t = pb.get("resistance_break_target")
            sup_t = pb.get("support_break_target")
            if res_t:
                st.markdown(f"- Resistance break target: **{res_t['level']}** ({res_t['distance_pct']}% away)")
            if sup_t:
                st.markdown(f"- Support break target: **{sup_t['level']}** ({sup_t['distance_pct']}% away)")
            trades = pb.get("trade_conditions") or []
            if trades:
                st.dataframe(pd.DataFrame(trades), use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader("Was yesterday's outlook right?")

    graded = accuracy.get("graded") or {}
    if not graded:
        st.info(
            f"No prior report was found to grade against {selected_date} "
            "(first day of data, or a gap in report history)."
        )
    else:
        gen_from = accuracy.get("generated_from_date")
        st.caption(f"Grading the outlook generated on {gen_from} against what actually happened on {selected_date}.")
        grade_rows = []
        for sym, r in graded.items():
            grade_rows.append({
                "Symbol": sym.upper(),
                "Gap call": f"{'✅' if r['gap_call_correct'] else '❌'} {r['gap_call_predicted']} → {r['gap_call_actual']}",
                "Bias": f"{'✅' if r['bias_correct'] else '❌'} {r['bias_predicted']} → {r['bias_actual']}",
                "Resistance target": (
                    f"{r['resistance_target']} ({'hit' if r['resistance_hit'] else 'missed'})"
                    if r.get("resistance_target") is not None else "—"
                ),
                "Support target": (
                    f"{r['support_target']} ({'hit' if r['support_hit'] else 'missed'})"
                    if r.get("support_target") is not None else "—"
                ),
                "Watch levels hit": f"{r['watch_levels_hit']}/{r['watch_levels_total']}",
            })
        st.dataframe(pd.DataFrame(grade_rows), use_container_width=True, hide_index=True)

    # Rolling hit-rate trend, queried directly (not just the single day's report)
    try:
        trailing_rows = db.get_outlook_accuracy_trailing(selected_date, days=60)
    except Exception as e:
        trailing_rows = []
        st.warning(f"Could not load rolling accuracy history: {e}")

    if trailing_rows:
        st.markdown("**Rolling accuracy trend (last 60 graded sessions):**")
        trend_df = pd.DataFrame(trailing_rows)
        trend_df["outlook_date"] = pd.to_datetime(trend_df["outlook_date"])
        summary = (
            trend_df.groupby("symbol")[["gap_call_correct", "bias_correct"]]
            .mean()
            .mul(100)
            .round(1)
            .rename(columns={"gap_call_correct": "Gap-call hit rate %", "bias_correct": "Bias hit rate %"})
        )
        st.dataframe(summary, use_container_width=True)

        for sym in trend_df["symbol"].unique():
            sym_df = trend_df[trend_df["symbol"] == sym].sort_values("outlook_date")
            chart_df = sym_df.set_index("outlook_date")[["gap_call_correct", "bias_correct"]].astype(float)
            chart_df = chart_df.rolling(window=10, min_periods=1).mean() * 100
            chart_df.columns = ["Gap-call hit rate % (10d rolling)", "Bias hit rate % (10d rolling)"]
            st.caption(f"{sym.upper()}")
            st.line_chart(chart_df)
    else:
        st.info("No accuracy history yet — it accumulates one row per graded session.")
