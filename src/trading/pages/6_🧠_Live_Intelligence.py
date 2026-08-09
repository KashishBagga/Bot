#!/usr/bin/env python3
"""
Live Intelligence — Actionable Trading Cockpit.

Uses st.fragment to refresh live dashboard components (trades, signals, state)
without browser flicker, page redirects, or resetting form inputs.
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

st.set_page_config(page_title="Actionable Trading Cockpit", page_icon="🧠", layout="wide")

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

# ─── Action Functions ──────────────────────────────────────────────────────────

def place_manual_trade(symbol, signal_type, entry_price, sl_price, tp_price, strategy="Manual"):
    now = datetime.now(timezone.utc)
    timestamp_str = int(now.timestamp())
    trade_id = f"trade_{symbol.replace(':', '_').replace('-', '_')}_Manual_{timestamp_str}"
    candidate_id = f"cand_{symbol.replace(':', '_').replace('-', '_')}_Manual_{timestamp_str}"
    
    option_contract = None
    lots = 1.0
    position_size_inr = 1000.0
    
    if "INDEX" in symbol:
        try:
            from src.core.options_execution_engine import OptionExecutionEngine
            from src.adapters.data.fyers_data_provider import FyersDataProvider
            dp = FyersDataProvider()
            engine = OptionExecutionEngine(db, dp, strike_policy="ATM")
            sig_m = {"symbol": symbol, "signal": signal_type}
            option_contract = engine.resolve(sig_m, entry_price)
        except Exception as e:
            st.warning(f"Could not resolve options premium (using default values): {e}")
            
        if option_contract:
            from src.core.options_mapper import OptionsMapper
            lot_size = OptionsMapper.get_lot_size(option_contract.symbol)
            premium = option_contract.premium or 100.0
            
            # Sizer calculation
            try:
                from src.core.position_sizer import PositionSizer
                sizer = PositionSizer(capital=100000.0)
                position_size_inr = sizer.get_position_size(
                    entry_price=entry_price,
                    stop_loss_price=sl_price,
                    strategy=strategy,
                    confidence=70.0,
                    regime_primary="UNKNOWN"
                )
            except Exception:
                position_size_inr = 1000.0
                
            if premium > 0 and lot_size > 0:
                lots = max(1, int(position_size_inr / (premium * lot_size)))

    trade_perf = {
        'trade_id': trade_id,
        'candidate_id': candidate_id,
        'entry_time': now,
        'exit_time': None,
        'strategy': strategy,
        'symbol': symbol,
        'entry_price': entry_price,
        'exit_price': None,
        'mfe': 0.0,
        'mae': 0.0,
        'pnl': None,
        'exit_reason': None,
        'features': {},
        'setup_type': strategy,
        'mfe_r': 0.0,
        'mae_r': 0.0,
        'max_closed_profit_r': 0.0,
        'final_pnl_r': None,
        'duration_minutes': 0.0,
        'bars_held': 0,
        'market_regime': 'UNKNOWN',
        'signal_logic_version': 'v3.2',
        'position_logic_version': 'v3.1',
        'risk_logic_version': 'v1.1',
        'stop_loss': sl_price,
        'take_profit': tp_price,
        'initial_stop_loss': sl_price,
        'initial_take_profit': tp_price,
        'highest_price': entry_price,
        'lowest_price': entry_price,
        'stop_loss_distance': abs(entry_price - sl_price) if (entry_price is not None and sl_price is not None) else 0.0,
        'signal_type': signal_type,
        'capture_rate': 0.0,
        'holding_efficiency': 0.0,
        'valid': True,
        'validation_errors': None,
        'confidence': 70.0,
        'diagnostics': {},
        'position_size_inr': position_size_inr,
        'lots': float(lots),
        'current_price': entry_price,
        'unrealized_pnl_r': 0.0,
        'last_heartbeat_at': now,
        'tp1': (entry_price + (abs(entry_price - sl_price) * 1.5) if signal_type == "BUY CALL" else entry_price - (abs(entry_price - sl_price) * 1.5)) if (entry_price is not None and sl_price is not None) else entry_price
    }
    
    try:
        db.save_trade_performance(trade_perf)
        # Save entry event
        event = {
            'event_id': f"evt_{timestamp_str}_{candidate_id}_entry",
            'trade_id': trade_id,
            'candidate_id': candidate_id,
            'timestamp': now,
            'event_type': 'ENTRY',
            'payload': {
                'entry_price': entry_price,
                'stop_loss': sl_price,
                'take_profit': tp_price
            }
        }
        db.save_trade_event(event)
        st.success(f"🚀 Paper trade entered successfully! ID: {trade_id}")
    except Exception as e:
        st.error(f"Failed to place manual trade: {e}")


def close_manual_trade(trade_id, current_price):
    now = datetime.now(timezone.utc)
    
    # Query position details
    try:
        with db._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT entry_price, stop_loss, stop_loss_distance, signal_type, symbol, strategy, entry_time
                    FROM trade_performance
                    WHERE trade_id = %s
                """, (trade_id,))
                row = cur.fetchone()
    except Exception as e:
        st.error(f"Error querying position details: {e}")
        return
        
    if not row:
        st.error("Position not found in database.")
        return
        
    entry_price, sl_price, sl_dist, signal_type, symbol, strategy, entry_time = row
    
    # Compute PnL
    pnl_r = 0.0
    if sl_dist and sl_dist > 0:
        if signal_type == "BUY CALL":
            pnl_r = (current_price - entry_price) / sl_dist
        else:
            pnl_r = (entry_price - current_price) / sl_dist
            
    # Calculate duration
    duration = (now - entry_time).total_seconds() / 60
    
    try:
        with db._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE trade_performance
                    SET exit_time = %s, exit_price = %s, pnl = %s, exit_reason = 'MANUAL_CLOSE',
                        final_pnl_r = %s, duration_minutes = %s, bars_held = %s,
                        current_price = %s, unrealized_pnl_r = 0.0, last_heartbeat_at = %s
                    WHERE trade_id = %s
                """, (now, current_price, pnl_r, pnl_r, duration, max(1, int(duration / 5)), current_price, now, trade_id))
            conn.commit()
            
        # Save exit event
        event = {
            'event_id': f"evt_{int(now.timestamp())}_{trade_id}_exit",
            'trade_id': trade_id,
            'candidate_id': None,
            'timestamp': now,
            'event_type': 'EXIT',
            'payload': {
                'exit_price': current_price,
                'exit_reason': 'MANUAL_CLOSE',
                'final_pnl_r': pnl_r,
                'duration_minutes': duration
            }
        }
        db.save_trade_event(event)
        st.success(f"🔴 Position {trade_id} manually closed at ₹{current_price:,.2f} ({pnl_r:+.2f}R)")
    except Exception as e:
        st.error(f"Failed to close manual trade: {e}")


# ─── Header ────────────────────────────────────────────────────────────────────

now = now_ist()
market_open = is_market_hours()
status_badge = '<span class="live-badge">🟢 LIVE</span>' if market_open else '🔴 Market Closed'

st.markdown(f"""
<div class="page-header">
  <h1>🧠 Actionable Trading Cockpit &nbsp; {status_badge}</h1>
  <p>Real-time session cockpit — active trades, current window, session expectations, next expiry &nbsp;
     <small style="color:#475569">Real-time auto-refresh active &nbsp;|&nbsp; {now.strftime("%H:%M:%S IST")}</small>
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Session progress ─────────────────────────────────────────────────────────

if market_open:
    prog = session_progress_pct()
    remaining_mins = int((375 - prog / 100 * 375))
    st.progress(prog / 100, text=f"Session: {prog:.0f}% complete  ·  ~{remaining_mins} min remaining  ·  {now.strftime('%H:%M IST')}")

# ─── Data loaders ─────────────────────────────────────────────────────────────

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
def load_open_combo():
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT combo_id, entry_time, symbol, experiment_name, combo_type,
                       setup_type, underlying_entry_price, legs, net_premium_paid,
                       max_loss, max_profit, target_r, stop_r, current_pnl_r,
                       confidence, diagnostics
                FROM combo_trades
                WHERE exit_time IS NULL
                ORDER BY entry_time DESC
            """)
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


@st.cache_data(ttl=20, show_spinner=False)
def load_open_cf():
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT candidate_id, timestamp, symbol, experiment_name, setup_type,
                       entry_price, stop_loss, take_profit, diagnostics
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


@st.cache_data(ttl=10, show_spinner=False)
def load_today_signals():
    today = now_ist().date()
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT candidate_id, timestamp, symbol, accepted, setup_type,
                       rejection_reasons, entry_price, stop_loss, take_profit, rr_ratio, experiment_name,
                       daily_bias, hourly_bias, market_regime
                FROM signal_audit
                WHERE timestamp AT TIME ZONE 'Asia/Kolkata' >= %s::date
                ORDER BY timestamp DESC LIMIT 15
            """, (today.strftime("%Y-%m-%d"),))
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


# ─── Tabs ──────────────────────────────────────────────────────────────────────

tab_cockpit, tab_market, tab_expiry = st.tabs([
    "🎮 Trading Cockpit",
    "📡 Market State",
    "⏱️ Expiry & Options"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRADING COCKPIT (Fully Actionable View)
# ══════════════════════════════════════════════════════════════════════════════
with tab_cockpit:
    
    # Fragment A: Active Trades & Metrics Row
    @st.fragment(run_every=AUTO_REFRESH_SECONDS)
    def render_active_positions_and_summary():
        open_trades  = load_open_positions()
        open_combo   = load_open_combo()
        open_cf      = load_open_cf()
        realized_pnl = load_realized_pnl_today()
        now = now_ist()

        st.markdown("## 📈 Active Real Positions")
        if not open_trades and not open_combo:
            st.info("No open real positions right now.")
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
                        diag = parse_json(t.get("diagnostics")) or {}
                        opt_sym = diag.get("option_symbol")
                        opt_display = f" ({opt_sym})" if opt_sym else ""
                        st.markdown(f"**{t.get('setup_type','?')} {t.get('signal_type','')}** — `{sym_short}`{opt_display}")
                        st.markdown(f"<small>{t.get('experiment_name','?')} · Entry: {fmt_dt(t.get('entry_time'))}</small>", unsafe_allow_html=True)
                    with tc2:
                        st.markdown(f"Entry: **₹{t.get('entry_price',0):,.2f}**")
                        st.markdown(f"SL: ₹{t.get('stop_loss',0):,.2f} | TP: ₹{t.get('take_profit',0):,.2f}")
                    with tc3:
                        if cp:
                            st.markdown(f"Current: **₹{cp:,.2f}**")
                        st.markdown(
                            f'<span class="{upnl_cls}">{fmt_r(upnl)}</span>',
                            unsafe_allow_html=True
                        )
                        if stale:
                            st.markdown('<span class="stale-warn">⚠️ Stale (no heartbeat)</span>', unsafe_allow_html=True)
                    with tc4:
                        if st.button("🔴 Close Manual", key=f"close_{t.get('trade_id')}"):
                            close_manual_trade(t.get('trade_id'), cp or t.get('entry_price'))
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            if open_combo:
                st.markdown("#### 🧩 Active Multi-Leg (Combo) Positions")
                for c in open_combo:
                    upnl = c.get("current_pnl_r")
                    sym_short = (c.get("symbol") or "").replace("NSE:", "").replace("-INDEX", "")
                    upnl_cls = "trade-pnl-green" if (upnl or 0) >= 0 else "trade-pnl-red"
                    with st.container():
                        st.markdown('<div class="trade-card-live">', unsafe_allow_html=True)
                        tc1, tc2, tc3 = st.columns([3, 2, 2])
                        with tc1:
                            st.markdown(f"**{c.get('combo_type','?')} {c.get('setup_type','')}** — `{sym_short}`")
                            st.markdown(f"<small>{c.get('experiment_name','?')} · Entry: {fmt_dt(c.get('entry_time'))}</small>", unsafe_allow_html=True)
                        with tc2:
                            st.markdown(f"Underlying entry: **₹{c.get('underlying_entry_price',0):,.2f}**")
                            st.markdown(f"Net premium: ₹{c.get('net_premium_paid',0):,.2f} | Max loss: ₹{c.get('max_loss',0):,.2f}")
                        with tc3:
                            st.markdown(f'<span class="{upnl_cls}">{fmt_r(upnl)}</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

        # ─── Executive Metrics Row
        st.markdown("---")
        st.markdown("## 📊 Session Summary")
        total_unrealized = sum((t.get("unrealized_pnl_r") or 0) for t in open_trades) + \
                           sum((c.get("current_pnl_r") or 0) for c in open_combo)
        total_deployed   = sum((t.get("position_size_inr") or 0) for t in open_trades) + \
                           sum((c.get("net_premium_paid") or 0) for c in open_combo)
        session_pnl      = (realized_pnl or 0) + total_unrealized
        total_open_positions = len(open_trades) + len(open_combo)

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        with sc1:
            st.metric("Realized PnL", fmt_r(realized_pnl))
        with sc2:
            st.metric("Unrealized PnL", f"{total_unrealized:+.2f}R", delta=f"{total_open_positions} open ({len(open_combo)} combo)")
        with sc3:
            st.metric("Session PnL", f"{session_pnl:+.2f}R")
        with sc4:
            st.metric("Capital Deployed", f"₹{total_deployed:,.0f}")
        with sc5:
            st.metric("Shadow Positions", len(open_cf))

    render_active_positions_and_summary()

    # ─── Order Placement Panel (Kept OUTSIDE fragments so inputs don't blink/reset)
    st.markdown("---")
    st.markdown("## 🚀 Place Manual Paper Trade")
    
    with st.expander("🛠️ Custom Order Placement Form", expanded=False):
        with st.form("manual_order_form"):
            o_sym = st.selectbox("Underlying Symbol", SYMBOLS)
            o_type = st.selectbox("Option Type", ["BUY CALL", "BUY PUT"])
            
            # Fetch latest price from market state as default
            latest_price = 0.0
            market_state = load_market_state()
            ms = market_state.get(o_sym, {})
            if ms:
                latest_price = ms.get("current_price") or 0.0
            
            o_entry = st.number_input("Index Entry Price", min_value=1.0, value=latest_price if latest_price > 0 else 24000.0, step=5.0)
            o_sl = st.number_input("Stop Loss Price", min_value=1.0, value=o_entry - 50.0 if o_type == "BUY CALL" else o_entry + 50.0, step=5.0)
            o_tp = st.number_input("Take Profit Price", min_value=1.0, value=o_entry + 100.0 if o_type == "BUY CALL" else o_entry - 100.0, step=5.0)
            
            o_strategy = st.text_input("Strategy Tag", value="Manual")
            
            submitted = st.form_submit_button("🚀 Submit Paper Trade")
            if submitted:
                if o_type == "BUY CALL" and o_sl >= o_entry:
                    st.error("For BUY CALL, Stop Loss must be BELOW entry price.")
                elif o_type == "BUY PUT" and o_sl <= o_entry:
                    st.error("For BUY PUT, Stop Loss must be ABOVE entry price.")
                else:
                    place_manual_trade(o_sym, o_type, o_entry, o_sl, o_tp, o_strategy)
                    st.rerun()

    # Fragment B: Signal Feed Override Center
    @st.fragment(run_every=AUTO_REFRESH_SECONDS)
    def render_signal_feed():
        today_signals = load_today_signals()
        st.markdown("---")
        st.markdown("## 📡 Today's Signal Feed & Override Center")
        if not today_signals:
            st.info("No signals generated today yet.")
        else:
            for idx, sig in enumerate(today_signals):
                accepted = sig.get("accepted")
                sym_short = sig.get("symbol").replace("NSE:", "").replace("-INDEX", "")
                reasons = parse_json(sig.get("rejection_reasons")) or []
                
                status_emoji = "🟢" if accepted else "🔴"
                status_text = "ACCEPTED (AUTOMATED)" if accepted else f"REJECTED: {', '.join(reasons)}"
                bg_color = "rgba(34,197,94,0.02)" if accepted else "rgba(239,68,68,0.02)"
                border_color = "rgba(34,197,94,0.2)" if accepted else "rgba(239,68,68,0.2)"
                
                ep = sig.get('entry_price')
                sl = sig.get('stop_loss')
                tp = sig.get('take_profit')
                ep_str = f"₹{ep:,.1f}" if ep is not None else "—"
                sl_str = f"₹{sl:,.1f}" if sl is not None else "—"
                tp_str = f"₹{tp:,.1f}" if tp is not None else "—"

                st.markdown(f"""
                <div style="background:{bg_color}; border: 1px solid {border_color}; border-radius:8px; padding:12px; margin-bottom:8px">
                    <div style="display:flex; justify-content:space-between; align-items:center">
                        <div>
                            <b>{sig.get('setup_type')}</b> | {sym_short} | <b>{status_emoji} {status_text}</b>
                            <br/>
                            <small style="color:#64748b">{fmt_dt(sig.get('timestamp'))} · exp: {sig.get('experiment_name')}</small>
                        </div>
                        <div>
                            Entry: <b>{ep_str}</b> | SL: <b>{sl_str}</b> | TP: <b>{tp_str}</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if not accepted:
                    col_ov1, col_ov2 = st.columns([7, 2])
                    with col_ov2:
                        if st.button("🚀 Override & Enter Real", key=f"override_{idx}_{sig.get('candidate_id')}"):
                            place_manual_trade(
                                sig.get("symbol"),
                                "BUY CALL" if "CALL" in str(sig.get("setup_type")).upper() or "BULL" in str(sig.get("daily_bias")).upper() else "BUY PUT",
                                sig.get("entry_price"),
                                sig.get("stop_loss"),
                                sig.get("take_profit"),
                                f"Override_{sig.get('setup_type')}"
                            )
                            st.rerun()

    render_signal_feed()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET STATE (System Indicators)
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    
    @st.fragment(run_every=AUTO_REFRESH_SECONDS)
    def render_market_state_tab():
        market_state = load_market_state()
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

    render_market_state_tab()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EXPIRY & OPTIONS (Option Analytics)
# ══════════════════════════════════════════════════════════════════════════════
with tab_expiry:
    
    @st.fragment(run_every=60)
    def render_expiry_tab():
        market_state = load_market_state()
        open_cf      = load_open_cf()
        st.markdown("## ⏱️ Next Expiry Status")
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

        # ─── Counterfactual Positions Summary
        if open_cf:
            st.markdown("---")
            st.markdown(f"## 👻 Open Shadow Positions ({len(open_cf)})")
            for cf in open_cf[:10]:
                sym_short = (cf.get("symbol") or "").replace("NSE:", "").replace("-INDEX", "")
                diag = parse_json(cf.get("diagnostics")) or {}
                opt_sym = diag.get("option_symbol")
                opt_display = f" ({opt_sym})" if opt_sym else ""
                st.markdown(
                    f'<div class="trade-card-cf">'
                    f'<b>{cf.get("setup_type","?")} `{sym_short}`{opt_display}</b> [{cf.get("experiment_name","?")}] '
                    f'Entry @ ₹{cf.get("entry_price",0):,.2f} | SL={cf.get("stop_loss",0):.2f}</div>',
                    unsafe_allow_html=True
                )
            if len(open_cf) > 10:
                st.caption(f"+{len(open_cf)-10} more shadow positions not shown.")

    render_expiry_tab()
