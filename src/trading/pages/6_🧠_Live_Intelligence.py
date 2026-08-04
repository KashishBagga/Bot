#!/usr/bin/env python3
"""
Live Intelligence — real-time trading cockpit.

Auto-refreshes every 20 seconds. Shows:
  - Active real trades + CF positions with live unrealized PnL
  - Current window: bias, regime, RVOL, zones, patterns per symbol
  - Session expectations: PnL used, capital deployed, limits remaining
  - Next expiry: time-to-expiry, straddle price, implied move
  - Quick option chain (ATM ±2 strikes)
"""
import os
import sys
import json
from datetime import datetime, date, timedelta, timezone
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.models.postgres_database import PostgresDatabase

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
AUTO_REFRESH_SECONDS = 20
STALE_AFTER = timedelta(minutes=10)
SYMBOLS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]

# ─── helpers ──────────────────────────────────────────────────────────────────

def now_ist():
    return datetime.now(KOLKATA_TZ)


def fmt_dt(dt):
    if dt is None:
        return "—"
    if hasattr(dt, "tzinfo") and dt.tzinfo:
        dt = dt.astimezone(KOLKATA_TZ)
    return dt.strftime("%H:%M:%S")


def fmt_r(val, default="—"):
    if val is None:
        return default
    return f"{float(val):+.2f}R"


def color_r(val):
    if val is None:
        return "⚪"
    return "🟢" if float(val) > 0 else ("🔴" if float(val) < 0 else "⚪")


def parse_json(v):
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v) if v else {}
    except Exception:
        return {}


def minutes_to_expiry(expiry_date_str: str) -> int:
    """Minutes until the expiry date (15:30 IST)."""
    try:
        exp_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
        exp_dt = datetime(exp_date.year, exp_date.month, exp_date.day, 15, 30, tzinfo=KOLKATA_TZ)
        delta = exp_dt - now_ist()
        return max(0, int(delta.total_seconds() / 60))
    except Exception:
        return -1


def is_stale(updated_at):
    if updated_at is None:
        return True
    if not hasattr(updated_at, "tzinfo") or updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - updated_at) > STALE_AFTER


def is_market_hours():
    n = now_ist()
    return (
        n.weekday() < 5
        and (n.hour > 9 or (n.hour == 9 and n.minute >= 15))
        and (n.hour < 15 or (n.hour == 15 and n.minute <= 30))
    )


def session_progress_pct():
    n = now_ist()
    total_mins = 375  # 09:15 to 15:30
    elapsed = (n.hour * 60 + n.minute) - (9 * 60 + 15)
    return max(0, min(100, elapsed / total_mins * 100))


# ─── CSS ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Live Intelligence", page_icon="🧠", layout="wide")
st.markdown(f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}">', unsafe_allow_html=True)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 16px; padding: 24px 32px; margin-bottom: 24px;
  }
  .page-header h1 { color: #f8fafc; margin: 0; font-size: 1.8rem; font-weight: 700; }
  .page-header p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.9rem; }

  .live-badge {
    display: inline-block; background: rgba(34,197,94,0.2);
    border: 1px solid rgba(34,197,94,0.5); border-radius: 20px;
    padding: 3px 12px; color: #4ade80; font-size: 0.78rem; font-weight: 600;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }

  .stat-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,41,59,0.9));
    border: 1px solid rgba(255,255,255,0.07); border-radius: 12px;
    padding: 18px 20px; text-align: center; height: 100%;
  }
  .stat-label { color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .stat-value { color: #f8fafc; font-size: 1.4rem; font-weight: 700; margin-top: 4px; }
  .stat-sub   { color: #94a3b8; font-size: 0.78rem; margin-top: 3px; }

  .trade-card-live {
    background: rgba(34,197,94,0.04); border: 1px solid rgba(34,197,94,0.2);
    border-radius: 12px; padding: 16px; margin-bottom: 10px;
  }
  .trade-card-cf {
    background: rgba(99,102,241,0.04); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px; padding: 16px; margin-bottom: 10px;
  }
  .trade-pnl-green { color: #4ade80; font-weight: 700; font-size: 1.1rem; }
  .trade-pnl-red   { color: #f87171; font-weight: 700; font-size: 1.1rem; }

  .market-card {
    background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 18px; height: 100%;
  }
  .market-card h4 { color: #e2e8f0; margin: 0 0 12px; font-size: 0.95rem; }

  .pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 500; margin: 2px;
  }
  .pill-bull { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.35); color: #4ade80; }
  .pill-bear { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.35); color: #f87171; }
  .pill-neu  { background: rgba(100,116,139,0.1); border: 1px solid rgba(100,116,139,0.3); color: #94a3b8; }
  .pill-info { background: rgba(6,182,212,0.08);  border: 1px solid rgba(6,182,212,0.25); color: #22d3ee; }
  .pill-warn { background: rgba(234,179,8,0.10);  border: 1px solid rgba(234,179,8,0.35); color: #facc15; }

  .zone-pill-supply { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); color: #f87171; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; margin: 3px; display: inline-block; }
  .zone-pill-demand { background: rgba(34,197,94,0.08);  border: 1px solid rgba(34,197,94,0.3); color: #4ade80; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; margin: 3px; display: inline-block; }

  .expiry-card {
    background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(234,179,8,0.04));
    border: 1px solid rgba(245,158,11,0.3); border-radius: 12px; padding: 18px;
  }

  .stale-warn {
    color: #facc15; background: rgba(234,179,8,0.08);
    border: 1px solid rgba(234,179,8,0.3); border-radius: 8px;
    padding: 6px 12px; font-size: 0.8rem; display: inline-block;
  }

  div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 12px;
  }
</style>
""", unsafe_allow_html=True)

# ─── DB ────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    return PostgresDatabase()

db = get_db()

# ─── Header ────────────────────────────────────────────────────────────────────

now = now_ist()
market_open = is_market_hours()
status_badge = '<span class="live-badge">🟢 LIVE</span>' if market_open else '🔴 Market Closed'

st.markdown(f"""
<div class="page-header">
  <h1>🧠 Live Intelligence &nbsp; {status_badge}</h1>
  <p>Real-time session cockpit — active trades, current window, session expectations, next expiry &nbsp;
     <small style="color:#475569">Auto-refreshes every {AUTO_REFRESH_SECONDS}s &nbsp;|&nbsp; {now.strftime("%H:%M:%S IST")}</small>
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Session progress ─────────────────────────────────────────────────────────

if market_open:
    prog = session_progress_pct()
    remaining_mins = int((375 - prog / 100 * 375))
    st.progress(prog / 100, text=f"Session: {prog:.0f}% complete  ·  ~{remaining_mins} min remaining  ·  {now.strftime('%H:%M IST')}")

# ─── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=20, show_spinner=False)
def load_open_positions():
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT trade_id, candidate_id, entry_time, strategy, symbol,
                       experiment_name, setup_type, signal_type,
                       entry_price, stop_loss, take_profit,
                       current_price, unrealized_pnl_r, last_heartbeat_at,
                       mfe_r, mae_r, position_size_inr, lots, diagnostics
                FROM trade_performance
                WHERE exit_time IS NULL AND (valid IS NULL OR valid = TRUE)
                ORDER BY entry_time
            """)
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


@st.cache_data(ttl=20, show_spinner=False)
def load_open_cf():
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT candidate_id, timestamp, symbol, experiment_name, setup_type,
                       entry_price, stop_loss, take_profit
                FROM counterfactual_results
                WHERE exit_time IS NULL AND (valid IS NULL OR valid = TRUE)
                ORDER BY timestamp
            """)
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


@st.cache_data(ttl=20, show_spinner=False)
def load_market_state():
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, updated_at, current_price, daily_bias, market_regime,
                       rvol, atr, move_efficiency, wickiness,
                       narrative_bias, narrative_confidence, zones, patterns
                FROM market_state
                ORDER BY symbol
            """)
            cols = [c.name for c in cur.description]
            return {r["symbol"]: r for r in (dict(zip(cols, row)) for row in cur.fetchall())}


@st.cache_data(ttl=30, show_spinner=False)
def load_realized_pnl_today():
    today = now_ist().date()
    return db.get_realized_r_today(today.strftime("%Y-%m-%d"))


@st.cache_data(ttl=60, show_spinner=False)
def load_latest_option_chain(symbol):
    return db.get_option_chain_snapshot(symbol)


open_trades  = load_open_positions()
open_cf      = load_open_cf()
market_state = load_market_state()
realized_pnl = load_realized_pnl_today()

# ─── Sections ──────────────────────────────────────────────────────────────────

# ─── Section 1: Active Trades ─────────────────────────────────────────────────
st.markdown("## 📈 Active Real Trades")

if not open_trades:
    st.info("No open real trades right now.")
else:
    for t in open_trades:
        upnl = t.get("unrealized_pnl_r")
        cp   = t.get("current_price")
        hb   = t.get("last_heartbeat_at")
        stale = is_stale(hb)
        sym_short = (t.get("symbol") or "").replace("NSE:", "").replace("-INDEX", "")
        upnl_cls = "trade-pnl-green" if (upnl or 0) >= 0 else "trade-pnl-red"

        with st.container():
            st.markdown('<div class="trade-card-live">', unsafe_allow_html=True)
            tc1, tc2, tc3, tc4 = st.columns([3, 2, 2, 2])
            with tc1:
                st.markdown(f"**{t.get('setup_type','?')} {t.get('signal_type','')}** — `{sym_short}`")
                st.markdown(f"<small>{t.get('experiment_name','?')} · Entry: {fmt_dt(t.get('entry_time'))}</small>", unsafe_allow_html=True)
            with tc2:
                st.markdown(f"Entry: **₹{t.get('entry_price',0):,.2f}**")
                st.markdown(f"SL: ₹{t.get('stop_loss',0):,.2f} | TP: ₹{t.get('take_profit',0):,.2f}")
            with tc3:
                if cp:
                    st.markdown(f"Current: **₹{cp:,.2f}**")
                st.markdown(
                    f'<span class="{upnl_cls}">{fmt_r(upnl)}</span>'
                    f'{"  ⚠️" if stale else ""}',
                    unsafe_allow_html=True
                )
                if stale:
                    st.markdown('<span class="stale-warn">⚠️ Stale (no heartbeat)</span>', unsafe_allow_html=True)
            with tc4:
                st.markdown(f"MFE: {fmt_r(t.get('mfe_r'))} | MAE: {fmt_r(t.get('mae_r'))}")
                cap = t.get('position_size_inr')
                lots = t.get('lots')
                if cap:
                    st.markdown(f"₹{cap:,.0f} deployed | {lots} lot(s)")
            st.markdown('</div>', unsafe_allow_html=True)

# ─── Section 2: Session Stats ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Session Overview")

total_unrealized = sum((t.get("unrealized_pnl_r") or 0) for t in open_trades)
total_deployed   = sum((t.get("position_size_inr") or 0) for t in open_trades)
session_pnl      = (realized_pnl or 0) + total_unrealized

sc1, sc2, sc3, sc4, sc5 = st.columns(5)
with sc1:
    st.markdown(f"""<div class="stat-card">
      <div class="stat-label">Realized PnL</div>
      <div class="stat-value" style="color:{'#22c55e' if (realized_pnl or 0) >= 0 else '#ef4444'}">{fmt_r(realized_pnl)}</div>
      <div class="stat-sub">Closed trades</div>
    </div>""", unsafe_allow_html=True)
with sc2:
    st.markdown(f"""<div class="stat-card">
      <div class="stat-label">Unrealized PnL</div>
      <div class="stat-value" style="color:{'#22c55e' if total_unrealized >= 0 else '#ef4444'}">{total_unrealized:+.2f}R</div>
      <div class="stat-sub">{len(open_trades)} open trade(s)</div>
    </div>""", unsafe_allow_html=True)
with sc3:
    st.markdown(f"""<div class="stat-card">
      <div class="stat-label">Session PnL</div>
      <div class="stat-value" style="color:{'#22c55e' if session_pnl >= 0 else '#ef4444'}">{session_pnl:+.2f}R</div>
      <div class="stat-sub">Realized + Unrealized</div>
    </div>""", unsafe_allow_html=True)
with sc4:
    st.markdown(f"""<div class="stat-card">
      <div class="stat-label">Capital Deployed</div>
      <div class="stat-value" style="font-size:1.1rem">₹{total_deployed:,.0f}</div>
      <div class="stat-sub">Across {len(open_trades)} position(s)</div>
    </div>""", unsafe_allow_html=True)
with sc5:
    st.markdown(f"""<div class="stat-card">
      <div class="stat-label">Shadow Positions</div>
      <div class="stat-value">{len(open_cf)}</div>
      <div class="stat-sub">CF trades running</div>
    </div>""", unsafe_allow_html=True)

# ─── Section 3: Current Market Window ─────────────────────────────────────────
st.markdown("---")
st.markdown("## 📡 Current Market Window")

if not market_state:
    st.info("No market_state data. The trading engine writes this every 5 min candle.")
else:
    ms_cols = st.columns(len(market_state))
    for idx, (sym, ms) in enumerate(market_state.items()):
        with ms_cols[idx]:
            sym_short = sym.replace("NSE:", "").replace("-INDEX", "")
            stale = is_stale(ms.get("updated_at"))
            bias = ms.get("daily_bias", "?")
            regime = ms.get("market_regime", "?")
            rvol = ms.get("rvol")
            atr  = ms.get("atr")
            eff  = ms.get("move_efficiency")
            wick = ms.get("wickiness")
            price = ms.get("current_price")
            narr_bias = ms.get("narrative_bias")
            narr_conf = ms.get("narrative_confidence")

            bias_cls = "pill-bull" if bias == "BULLISH" else ("pill-bear" if bias == "BEARISH" else "pill-neu")
            narr_cls = "pill-bull" if narr_bias == "BULLISH" else ("pill-bear" if narr_bias == "BEARISH" else "pill-neu")

            st.markdown(f'<div class="market-card">', unsafe_allow_html=True)
            st.markdown(f"#### {sym_short}  {'⚠️ STALE' if stale else '✅'}")
            if price:
                st.markdown(f"**₹{price:,.2f}**  <small>{fmt_dt(ms.get('updated_at'))}</small>", unsafe_allow_html=True)

            # Bias + regime pills
            st.markdown(
                f'<span class="pill {bias_cls}">{bias}</span>'
                f'<span class="pill pill-info">{regime}</span>',
                unsafe_allow_html=True
            )
            if narr_bias:
                st.markdown(
                    f'<span class="pill {narr_cls}">Narrative: {narr_bias} ({narr_conf:.0%})</span>' if narr_conf else f'<span class="pill {narr_cls}">Narrative: {narr_bias}</span>',
                    unsafe_allow_html=True
                )

            # Metrics
            m1, m2 = st.columns(2)
            with m1:
                if rvol is not None:
                    rvol_cls = "pill-bull" if rvol >= 1.0 else "pill-warn"
                    st.markdown(f'<span class="pill {rvol_cls}">RVOL {rvol:.2f}x</span>', unsafe_allow_html=True)
                if atr is not None:
                    st.markdown(f'<span class="pill pill-info">ATR {atr:.1f}</span>', unsafe_allow_html=True)
            with m2:
                if eff is not None:
                    st.markdown(f'<span class="pill pill-info">Eff {eff:.2f}</span>', unsafe_allow_html=True)
                if wick is not None:
                    wick_cls = "pill-warn" if wick > 0.4 else "pill-neu"
                    st.markdown(f'<span class="pill {wick_cls}">Wick {wick:.2f}</span>', unsafe_allow_html=True)

            # Zones
            zones = parse_json(ms.get("zones")) or []
            if zones:
                supply_z = [z for z in zones if z.get("type") == "SUPPLY"]
                demand_z = [z for z in zones if z.get("type") == "DEMAND"]
                if supply_z:
                    z_pills = " ".join(f'<span class="zone-pill-supply">R: {z["level"]:,.0f}</span>' for z in supply_z[:3])
                    st.markdown(z_pills, unsafe_allow_html=True)
                if demand_z:
                    z_pills = " ".join(f'<span class="zone-pill-demand">S: {z["level"]:,.0f}</span>' for z in demand_z[:3])
                    st.markdown(z_pills, unsafe_allow_html=True)

            # Active patterns
            patterns = parse_json(ms.get("patterns")) or []
            live_patterns = [p for p in patterns if p.get("state") in ("READY", "BREAKOUT", "CONFIRMED")]
            if live_patterns:
                st.markdown(f"**{len(live_patterns)} live pattern(s):**")
                for p in live_patterns[:3]:
                    st.markdown(
                        f'<span class="pill pill-warn">{p.get("type","?")} {p.get("direction","?")} '
                        f'{p.get("completion_pct",0)*100:.0f}% conf={p.get("confidence",0):.2f}</span>',
                        unsafe_allow_html=True
                    )

            st.markdown('</div>', unsafe_allow_html=True)

# ─── Section 4: Next Expiry ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## ⏱️ Next Expiry")

exp_cols = st.columns(len(SYMBOLS))
for idx, sym in enumerate(SYMBOLS):
    with exp_cols[idx]:
        sym_short = sym.replace("NSE:", "").replace("-INDEX", "")
        chain = load_latest_option_chain(sym)
        if chain:
            expiry_date = chain[0].get("expiry")
            mins_left = minutes_to_expiry(str(expiry_date)) if expiry_date else -1
            hours_left = mins_left // 60
            mins_rem = mins_left % 60

            ms = market_state.get(sym, {})
            ltp = ms.get("current_price")
            interval = 100 if "BANK" in sym else 50
            atm = round((ltp or 24500) / interval) * interval if ltp else None

            # Straddle from ATM CE + PE
            if atm:
                atm_ce = next((r for r in chain if r["strike"] == float(atm) and r["option_type"] == "CE"), {})
                atm_pe = next((r for r in chain if r["strike"] == float(atm) and r["option_type"] == "PE"), {})
                straddle = (atm_ce.get("ltp") or 0) + (atm_pe.get("ltp") or 0)
                impl_move = (straddle / ltp * 100) if ltp and straddle else None
            else:
                straddle = impl_move = None

            countdown_text = (
                f"⏰ {hours_left}h {mins_rem}m remaining"
                if mins_left >= 0
                else "Expired or unknown"
            )
            atm_row = (
                f'<div style="margin-top:8px;color:#94a3b8">ATM: <b style="color:#f8fafc">{int(atm)}</b>'
                f' | Straddle: <b style="color:#fbbf24">₹{straddle:.1f}</b></div>'
                if atm and straddle
                else ""
            )
            move_row = (
                f'<div style="color:#94a3b8">Implied move: <b style="color:#22d3ee">±{impl_move:.2f}%</b></div>'
                if impl_move
                else ""
            )
            st.markdown(f"""
<div class="expiry-card">
  <h4>📅 {sym_short} — Expiry</h4>
  <div style="font-size:1.2rem;font-weight:700;color:#f8fafc">{expiry_date}</div>
  <div style="color:#facc15;font-size:0.95rem;margin-top:4px">{countdown_text}</div>
  {atm_row}
  {move_row}
</div>""", unsafe_allow_html=True)
        else:
            st.info(f"No option data for {sym_short}. Start the Option Warehouse.")


# ─── Section 5: Shadow (CF) Positions Summary ─────────────────────────────────
if open_cf:
    st.markdown("---")
    st.markdown(f"## 👻 Open Shadow Positions ({len(open_cf)})")
    for cf in open_cf[:10]:
        sym_short = (cf.get("symbol") or "").replace("NSE:", "").replace("-INDEX", "")
        st.markdown(
            f'<div class="trade-card-cf">'
            f'<b>{cf.get("setup_type","?")} `{sym_short}`</b> [{cf.get("experiment_name","?")}] '
            f'Entry @ ₹{cf.get("entry_price",0):,.2f} | SL={cf.get("stop_loss",0):.2f}</div>',
            unsafe_allow_html=True
        )
    if len(open_cf) > 10:
        st.caption(f"+{len(open_cf)-10} more shadow positions not shown.")
