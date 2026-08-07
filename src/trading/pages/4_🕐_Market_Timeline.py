#!/usr/bin/env python3
"""
Market Timeline — hour-by-hour reconstruction of any trading day.

For a chosen date + symbol, shows:
  - Running PnL chart across the day
  - Hourly accordion panels: signals fired, trades entered/exited,
    market events (patterns, zone touches, breakouts, regime changes)
  - Bias/regime/RVOL context at each hour
"""
import os
import sys
import json
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

import streamlit as st
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.models.postgres_database import PostgresDatabase

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
MARKET_OPEN   = 9 * 60 + 15   # 09:15 in minutes
MARKET_CLOSE  = 15 * 60 + 30  # 15:30 in minutes
HOUR_WINDOWS  = [
    (9, 15, 10, 15),
    (10, 15, 11, 15),
    (11, 15, 12, 15),
    (12, 15, 13, 15),
    (13, 15, 14, 15),
    (14, 15, 15, 30),
]

SYMBOLS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
EVENT_EMOJI = {
    "PATTERN_CONFIRMED": "📐",
    "PATTERN_BREAKOUT": "💥",
    "PATTERN_FORMING": "🔍",
    "ZONE_TOUCH": "🎯",
    "REGIME_CHANGE": "🔄",
    "BOS": "📊",
    "GAP": "↕️",
}

# ─── helpers ──────────────────────────────────────────────────────────────────

def to_ist(dt):
    if dt is None:
        return None
    if hasattr(dt, "tzinfo") and dt.tzinfo:
        return dt.astimezone(KOLKATA_TZ)
    return dt


def fmt_t(dt):
    dt = to_ist(dt)
    return dt.strftime("%H:%M") if dt else "—"


def parse_json(v):
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v) if v else {}
    except Exception:
        return {}


def fmt_r(val):
    if val is None:
        return "—"
    return f"{float(val):+.2f}R"


# ─── CSS ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Market Timeline", page_icon="🕐", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px; padding: 24px 32px; margin-bottom: 24px;
  }
  .page-header h1 { color: #f8fafc; margin: 0; font-size: 1.8rem; font-weight: 700; }
  .page-header p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.9rem; }

  .hour-window {
    background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 18px; margin-bottom: 12px;
  }
  .hour-label { color: #a78bfa; font-weight: 700; font-size: 1.05rem; }

  .timeline-event {
    border-left: 2px solid rgba(139,92,246,0.4);
    padding: 5px 14px; margin: 4px 0;
    color: #cbd5e1; font-size: 0.82rem;
  }
  .timeline-event-time { color: #8b5cf6; font-weight: 600; margin-right: 8px; }

  .signal-row {
    border-left: 2px solid rgba(34,197,94,0.4);
    padding: 5px 14px; margin: 4px 0; font-size: 0.82rem; color: #cbd5e1;
  }
  .signal-row.rejected { border-left-color: rgba(239,68,68,0.4); }
  .signal-time { color: #22c55e; font-weight: 600; margin-right: 8px; }
  .signal-time.rejected { color: #f87171; }

  .trade-entry { background: rgba(34,197,94,0.06); border-left: 3px solid #22c55e; padding: 8px 14px; margin: 4px 0; border-radius: 4px; font-size: 0.83rem; }
  .trade-exit  { background: rgba(239,68,68,0.06); border-left: 3px solid #ef4444; padding: 8px 14px; margin: 4px 0; border-radius: 4px; font-size: 0.83rem; }

  .pill {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500; margin: 2px 2px;
  }
  .pill-info { background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.25); color: #22d3ee; }
  .pill-bias-bull { background: rgba(34,197,94,0.10); border: 1px solid rgba(34,197,94,0.3); color: #4ade80; }
  .pill-bias-bear { background: rgba(239,68,68,0.10); border: 1px solid rgba(239,68,68,0.3); color: #f87171; }
  .pill-neutral   { background: rgba(100,116,139,0.10); border: 1px solid rgba(100,116,139,0.3); color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ─── DB ────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    return PostgresDatabase()

db = get_db()

# ─── Header + controls ────────────────────────────────────────────────────────

st.markdown("""
<div class="page-header">
  <h1>🕐 Market Timeline</h1>
  <p>Hour-by-hour reconstruction of every signal, trade, and market event in a session</p>
</div>
""", unsafe_allow_html=True)

col_d, col_s = st.columns([2, 2])
with col_d:
    selected_date = st.date_input(
        "Trading Date", value=date.today() - timedelta(days=1),
        max_value=date.today(), key="tl_date"
    )
with col_s:
    symbol = st.selectbox("Symbol", SYMBOLS, key="tl_symbol")

date_str = selected_date.strftime("%Y-%m-%d")

# ─── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def load_signals_for_timeline(date_str, symbol):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT candidate_id, timestamp, accepted, setup_type, experiment_name,
                       rejection_reasons, score_breakdown, daily_bias, hourly_bias,
                       market_regime, entry_price, stop_loss, take_profit
                FROM signal_audit
                WHERE symbol = %s
                  AND timestamp AT TIME ZONE 'Asia/Kolkata' >= %s::date
                  AND timestamp AT TIME ZONE 'Asia/Kolkata' <  %s::date + interval '1 day'
                ORDER BY timestamp
            """, (symbol, date_str, date_str))
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


@st.cache_data(ttl=120, show_spinner=False)
def load_trades_for_timeline(date_str, symbol):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT trade_id, entry_time, exit_time, setup_type, experiment_name,
                       entry_price, exit_price, final_pnl_r, exit_reason,
                       stop_loss, take_profit, mfe_r, mae_r, diagnostics
                FROM trade_performance
                WHERE symbol = %s
                  AND entry_time AT TIME ZONE 'Asia/Kolkata' >= %s::date
                  AND entry_time AT TIME ZONE 'Asia/Kolkata' <  %s::date + interval '1 day'
                ORDER BY entry_time
            """, (symbol, date_str, date_str))
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


@st.cache_data(ttl=120, show_spinner=False)
def load_market_events_for_timeline(date_str, symbol):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, event_type, payload
                FROM market_events
                WHERE symbol = %s
                  AND timestamp AT TIME ZONE 'Asia/Kolkata' >= %s::date
                  AND timestamp AT TIME ZONE 'Asia/Kolkata' <  %s::date + interval '1 day'
                ORDER BY timestamp
            """, (symbol, date_str, date_str))
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


signals  = load_signals_for_timeline(date_str, symbol)
trades   = load_trades_for_timeline(date_str, symbol)
mevents  = load_market_events_for_timeline(date_str, symbol)

# ─── Running PnL chart ────────────────────────────────────────────────────────

if trades:
    st.markdown("### 📈 Cumulative PnL Across Session")
    pnl_data = []
    cum = 0.0
    for t in trades:
        p = t.get("final_pnl_r") or 0.0
        cum += p
        dt_ist = to_ist(t.get("exit_time") or t.get("entry_time"))
        pnl_data.append({
            "time": dt_ist.strftime("%H:%M") if dt_ist else "?",
            "trade": f"{t.get('setup_type','?')} {fmt_r(t.get('final_pnl_r'))}",
            "cumulative_pnl": round(cum, 3),
        })

    df_pnl = pd.DataFrame(pnl_data)
    st.line_chart(df_pnl.set_index("time")["cumulative_pnl"], use_container_width=True)
    st.caption(f"Total: **{cum:+.2f}R**  across {len(trades)} trade(s)")
else:
    st.info(f"No trades for {symbol} on {date_str}.")

st.markdown("---")

# ─── Bucket data into hourly windows ──────────────────────────────────────────

def in_window(dt, sh, sm, eh, em):
    dt = to_ist(dt)
    if dt is None:
        return False
    mins = dt.hour * 60 + dt.minute
    return (sh * 60 + sm) <= mins < (eh * 60 + em)


st.markdown("### 🕒 Hourly Windows")

for (sh, sm, eh, em) in HOUR_WINDOWS:
    label = f"{sh:02d}:{sm:02d} — {eh:02d}:{em:02d} IST"

    # Collect items in this window
    window_sigs   = [s for s in signals   if in_window(s.get("timestamp"), sh, sm, eh, em)]
    window_trades = [t for t in trades    if in_window(t.get("entry_time"), sh, sm, eh, em)
                                          or in_window(t.get("exit_time"),  sh, sm, eh, em)]
    window_events = [e for e in mevents   if in_window(e.get("timestamp"), sh, sm, eh, em)]

    accepted_w  = [s for s in window_sigs if s.get("accepted")]
    rejected_w  = [s for s in window_sigs if not s.get("accepted")]
    trade_pnl   = sum((t.get("final_pnl_r") or 0) for t in window_trades)

    # Count badge
    badge = ""
    if window_trades:
        badge += f"  📈 {len(window_trades)} trade(s) {fmt_r(trade_pnl) if window_trades else ''}"
    if accepted_w:
        badge += f"  ✅ {len(accepted_w)} signal(s)"
    if rejected_w:
        badge += f"  ❌ {len(rejected_w)} rejected"
    if window_events:
        badge += f"  ⚡ {len(window_events)} event(s)"

    has_content = window_sigs or window_trades or window_events

    with st.expander(f"🕒 **{label}**{badge}", expanded=(bool(window_trades) or bool(accepted_w))):

        if not has_content:
            st.markdown('<p style="color:#475569;font-size:0.85rem">No activity in this window.</p>', unsafe_allow_html=True)
            continue

        # Market events
        if window_events:
            st.markdown("**Market Events**")
            for ev in window_events:
                p = parse_json(ev.get("payload"))
                etype = ev["event_type"]
                emoji = EVENT_EMOJI.get(etype, "⚡")
                ts = fmt_t(ev["timestamp"])
                detail = ""
                if "pattern" in etype.lower():
                    detail = (
                        f"type={p.get('type','?')}  "
                        f"dir={p.get('direction','?')}  "
                        "conf=" + (f"{float(c):.2f}" if (c := p.get("confidence") if p.get("confidence") is not None else (p.get("delta") or {}).get("confidence_delta") if p.get("delta") else None) is not None else "?")
                        if isinstance(p.get("confidence"), float)
                        else f"state={p.get('state','?')}"
                    )
                elif etype == "REGIME_CHANGE":
                    detail = f"{p.get('from','?')} → {p.get('to','?')}"
                elif etype == "ZONE_TOUCH":
                    detail = f"zone @ {p.get('level','?')}  type={p.get('zone_type','?')}"
                else:
                    detail = str(p)[:60]
                st.markdown(
                    f'<div class="timeline-event"><span class="timeline-event-time">{ts}</span>'
                    f'{emoji} <b>{etype}</b> — {detail}</div>',
                    unsafe_allow_html=True
                )

        # Signals
        if window_sigs:
            st.markdown("**Signals Generated**")
            for sig in window_sigs:
                accepted = sig.get("accepted")
                reasons  = parse_json(sig.get("rejection_reasons")) or []
                ts = fmt_t(sig.get("timestamp"))
                cls = "" if accepted else "rejected"
                ts_cls = "signal-time" if accepted else "signal-time rejected"
                icon = "✅" if accepted else "❌"
                reason_str = f"  [{', '.join(reasons[:3])}]" if reasons else ""
                ep  = sig.get("entry_price")
                sl  = sig.get("stop_loss")
                bias = sig.get("daily_bias", "?")
                bias_pill_cls = "pill-bias-bull" if bias == "BULLISH" else ("pill-bias-bear" if bias == "BEARISH" else "pill-neutral")
                
                ep_str = f"₹{ep:.2f}" if ep is not None else "—"
                sl_str = f"₹{sl:.2f}" if sl is not None else "—"
                
                st.markdown(
                    f'<div class="signal-row {cls}">'
                    f'<span class="{ts_cls}">{ts}</span>'
                    f'{icon} <b>{sig.get("setup_type","?")}</b> [{sig.get("experiment_name","?")}]'
                    f'  Entry={ep_str}  SL={sl_str}{reason_str}'
                    f'  <span class="pill {bias_pill_cls}">{bias}</span>'
                    f'  <span class="pill pill-info">{sig.get("market_regime","?")}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Trades
        if window_trades:
            st.markdown("**Trades**")
            for t in window_trades:
                entry_ist = to_ist(t.get("entry_time"))
                exit_ist  = to_ist(t.get("exit_time"))
                pnl = t.get("final_pnl_r")
                icon = "🟢" if (pnl or 0) > 0 else ("🔴" if (pnl or 0) < 0 else "⚪")
                diag = parse_json(t.get("diagnostics")) or {}
                opt_sym = diag.get("option_symbol")
                opt_display = f" ({opt_sym})" if opt_sym else ""
                
                if entry_ist and in_window(t.get("entry_time"), sh, sm, eh, em):
                    ep = t.get("entry_price")
                    sl = t.get("stop_loss")
                    ep_str = f"₹{ep:,.2f}" if ep is not None else "—"
                    sl_str = f"₹{sl:,.2f}" if sl is not None else "—"
                    st.markdown(
                        f'<div class="trade-entry">⬆️ <b>ENTRY</b> {fmt_t(entry_ist)} | '
                        f'{t.get("setup_type","?")}{opt_display} [{t.get("experiment_name","?")}] | '
                        f'@ {ep_str}  SL={sl_str}</div>',
                        unsafe_allow_html=True
                    )
                if exit_ist and in_window(t.get("exit_time"), sh, sm, eh, em):
                    st.markdown(
                        f'<div class="trade-exit">⬇️ <b>EXIT</b> {fmt_t(exit_ist)} | '
                        f'{t.get("setup_type","?")}{opt_display} | '
                        f'{icon} {fmt_r(pnl)} | reason={t.get("exit_reason","?")} | '
                        f'MFE={fmt_r(t.get("mfe_r"))} MAE={fmt_r(t.get("mae_r"))}</div>',
                        unsafe_allow_html=True
                    )
