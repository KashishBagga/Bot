#!/usr/bin/env python3
"""
Daily Review — full forensics for any trading day.

Tabs:
  Summary        — session stats, experiment breakdown, market narrative
  All Trades     — every executed trade with full factor/diagnostic detail
  Rejected Signals — every rejected signal with score breakdown + CF outcome
  Risk Blocks    — signals that passed strategy filters but were risk-blocked
"""
import os
import sys
import json
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from src.models.postgres_database import PostgresDatabase

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
AUTO_REFRESH_SECONDS = 60


# ─── helpers ──────────────────────────────────────────────────────────────────

def fmt_dt(dt):
    if dt is None:
        return "—"
    if hasattr(dt, "tzinfo") and dt.tzinfo:
        dt = dt.astimezone(KOLKATA_TZ)
    return dt.strftime("%H:%M:%S")


def fmt_r(val, decimals=2):
    if val is None:
        return "—"
    return f"{float(val):+.{decimals}f}R"


def color_r(val):
    if val is None:
        return "⚪"
    return "🟢" if float(val) > 0 else ("🔴" if float(val) < 0 else "⚪")


def parse_json(v):
    if v is None:
        return {}
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return {}


# ─── CSS ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Daily Review", page_icon="📋", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px; padding: 24px 32px; margin-bottom: 24px;
  }
  .page-header h1 { color: #f8fafc; margin: 0; font-size: 1.8rem; font-weight: 700; }
  .page-header p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.9rem; }

  .stat-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(30,41,59,0.9) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 18px 20px; text-align: center;
  }
  .stat-label  { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .stat-value  { color: #f8fafc; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
  .stat-sub    { color: #94a3b8; font-size: 0.8rem; margin-top: 2px; }

  .trade-card {
    background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 20px; margin-bottom: 14px;
  }
  .trade-card-win  { border-left: 3px solid #22c55e; }
  .trade-card-loss { border-left: 3px solid #ef4444; }
  .trade-card-flat { border-left: 3px solid #64748b; }

  .pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 500; margin: 2px 3px;
  }
  .pill-pass { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: #4ade80; }
  .pill-fail { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.3);  color: #f87171; }
  .pill-info { background: rgba(6,182,212,0.08);  border: 1px solid rgba(6,182,212,0.25); color: #22d3ee; }
  .pill-warn { background: rgba(234,179,8,0.10);  border: 1px solid rgba(234,179,8,0.35); color: #facc15; }

  .event-row {
    border-left: 2px solid rgba(99,102,241,0.3); padding: 6px 14px;
    margin: 4px 0; color: #cbd5e1; font-size: 0.82rem;
  }
  .event-time { color: #6366f1; font-weight: 600; margin-right: 8px; }

  .rejected-card {
    background: rgba(239,68,68,0.04); border: 1px solid rgba(239,68,68,0.15);
    border-radius: 10px; padding: 16px; margin-bottom: 10px;
  }
  .block-card {
    background: rgba(234,179,8,0.04); border: 1px solid rgba(234,179,8,0.2);
    border-radius: 10px; padding: 14px; margin-bottom: 10px;
  }

  div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 14px;
  }
</style>
""", unsafe_allow_html=True)


# ─── DB connection ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_db():
    return PostgresDatabase()

db = get_db()


# ─── Date picker ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-header">
  <h1>📋 Daily Review</h1>
  <p>Full trade forensics — every decision the system made and why</p>
</div>
""", unsafe_allow_html=True)

col_date, col_sym = st.columns([2, 3])
with col_date:
    selected_date = st.date_input(
        "Trading Date",
        value=date.today() - timedelta(days=1),
        max_value=date.today(),
        key="review_date",
    )
date_str = selected_date.strftime("%Y-%m-%d")

# IST midnight bounds for the selected date
day_start = datetime(selected_date.year, selected_date.month, selected_date.day, 9, 15, tzinfo=KOLKATA_TZ)
day_end   = datetime(selected_date.year, selected_date.month, selected_date.day, 15, 45, tzinfo=KOLKATA_TZ)


# ─── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def load_trades(date_str: str):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT trade_id, candidate_id, entry_time, exit_time, strategy, symbol,
                       experiment_name, setup_type, signal_type,
                       entry_price, exit_price, stop_loss, take_profit,
                       initial_stop_loss, initial_take_profit,
                       final_pnl_r, mfe_r, mae_r, capture_rate,
                       duration_minutes, bars_held, exit_reason,
                       market_regime, (features->>'daily_bias') as daily_bias, diagnostics, features
                FROM trade_performance
                WHERE entry_time AT TIME ZONE 'Asia/Kolkata' >= %(ds)s::date
                  AND entry_time AT TIME ZONE 'Asia/Kolkata' <  %(ds)s::date + interval '1 day'
                  AND (valid IS NULL OR valid = TRUE)
                ORDER BY entry_time
            """, {"ds": date_str})
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


@st.cache_data(ttl=120, show_spinner=False)
def load_trade_events(date_str: str):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT te.trade_id, te.timestamp, te.event_type, te.payload
                FROM trade_events te
                JOIN trade_performance tp ON te.trade_id = tp.trade_id
                WHERE tp.entry_time AT TIME ZONE 'Asia/Kolkata' >= %(ds)s::date
                  AND tp.entry_time AT TIME ZONE 'Asia/Kolkata' <  %(ds)s::date + interval '1 day'
                ORDER BY te.timestamp
            """, {"ds": date_str})
            rows = cur.fetchall()
    result = {}
    for trade_id, ts, evt, payload in rows:
        result.setdefault(trade_id, []).append({
            "timestamp": ts, "event_type": evt, "payload": parse_json(payload)
        })
    return result


@st.cache_data(ttl=120, show_spinner=False)
def load_execution_events(date_str: str):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ee.trade_id, ee.candidate_id, ee.timestamp, ee.event_type, ee.payload
                FROM execution_events ee
                WHERE ee.timestamp AT TIME ZONE 'Asia/Kolkata' >= %(ds)s::date
                  AND ee.timestamp AT TIME ZONE 'Asia/Kolkata' <  %(ds)s::date + interval '1 day'
                ORDER BY ee.timestamp
            """, {"ds": date_str})
            rows = cur.fetchall()
    result = {}
    for trade_id, cand_id, ts, evt, payload in rows:
        key = trade_id or cand_id
        result.setdefault(key, []).append({
            "timestamp": ts, "event_type": evt, "payload": parse_json(payload)
        })
    return result


@st.cache_data(ttl=120, show_spinner=False)
def load_signal_audit(date_str: str):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT candidate_id, timestamp, symbol, accepted, experiment_name,
                       setup_type, rejection_reasons, score_breakdown,
                       daily_bias, hourly_bias, market_regime,
                       entry_price, stop_loss, take_profit, rr_ratio
                FROM signal_audit
                WHERE timestamp AT TIME ZONE 'Asia/Kolkata' >= %(ds)s::date
                  AND timestamp AT TIME ZONE 'Asia/Kolkata' <  %(ds)s::date + interval '1 day'
                ORDER BY timestamp
            """, {"ds": date_str})
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


@st.cache_data(ttl=120, show_spinner=False)
def load_cf_results(date_str: str):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT candidate_id, timestamp, symbol, experiment_name,
                       setup_type, rejection_reasons, primary_rejection_reason,
                       entry_price, stop_loss, take_profit,
                       final_pnl_r, mfe_r, mae_r, exit_reason, duration_minutes
                FROM counterfactual_results
                WHERE timestamp AT TIME ZONE 'Asia/Kolkata' >= %(ds)s::date
                  AND timestamp AT TIME ZONE 'Asia/Kolkata' <  %(ds)s::date + interval '1 day'
            """, {"ds": date_str})
            cols = [c.name for c in cur.description]
            return {r["candidate_id"]: r for r in (dict(zip(cols, row)) for row in cur.fetchall())}


@st.cache_data(ttl=120, show_spinner=False)
def load_risk_blocks(date_str: str):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT block_id, timestamp, symbol, experiment_name,
                       setup_type, signal_type, gate_reason,
                       entry_price, stop_loss, take_profit, rr_ratio
                FROM risk_governor_blocks
                WHERE timestamp AT TIME ZONE 'Asia/Kolkata' >= %(ds)s::date
                  AND timestamp AT TIME ZONE 'Asia/Kolkata' <  %(ds)s::date + interval '1 day'
                ORDER BY timestamp
            """, {"ds": date_str})
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


@st.cache_data(ttl=120, show_spinner=False)
def load_experiment_metrics(date_str: str):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT experiment_name, real_trades, cf_trades, wins, losses,
                       expectancy, total_pnl_r, avg_capture_rate
                FROM experiment_daily_metrics
                WHERE date = %s
                ORDER BY expectancy DESC
            """, (date_str,))
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def get_lot_size(symbol: str) -> int:
    if "BANKNIFTY" in symbol or "NIFTYBANK" in symbol:
        return 15
    elif "FINNIFTY" in symbol:
        return 40
    elif "NIFTY" in symbol:
        return 25
    return 1


@st.cache_data(ttl=120, show_spinner=False)
def load_combo_trades(date_str: str):
    with db._get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT combo_id, entry_time, exit_time, symbol,
                       experiment_name, strategy_id, version, combo_type, setup_type,
                       underlying_entry_price, underlying_exit_price, legs,
                       net_premium_paid, max_loss, max_profit, target_r, stop_r,
                       current_pnl_r, final_pnl_r, exit_reason, duration_minutes,
                       confidence, diagnostics, valid, validation_errors
                FROM combo_trades
                WHERE entry_time AT TIME ZONE 'Asia/Kolkata' >= %(ds)s::date
                  AND entry_time AT TIME ZONE 'Asia/Kolkata' <  %(ds)s::date + interval '1 day'
                  AND (valid IS NULL OR valid = TRUE)
                ORDER BY entry_time
            """, {"ds": date_str})
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def process_trade_metrics(trade):
    is_combo = trade.get("is_combo", False)
    symbol = trade.get("symbol") or "UNKNOWN"
    lot_size = get_lot_size(symbol)
    
    pnl_r = trade.get("final_pnl_r") or trade.get("pnl") or 0.0
    
    if is_combo:
        max_loss = trade.get("max_loss") or 0.0
        max_loss_inr = max_loss * lot_size
        realized_pnl_inr = pnl_r * max_loss_inr
        capital_deployed_inr = max_loss_inr
        is_win = pnl_r > 0
    else:
        entry_price = trade.get("entry_price") or 0.0
        initial_sl = trade.get("initial_stop_loss") or 0.0
        sl_distance = trade.get("stop_loss_distance") or abs(entry_price - initial_sl)
        diagnostics = parse_json(trade.get("diagnostics"))

        if sl_distance <= 0:
            features = parse_json(trade.get("features"))
            atr = features.get("atr") or diagnostics.get("atr") or 15.0
            sl_distance = 0.5 * atr

        lots = trade.get("lots") or diagnostics.get("lots") or 1
        option_premium = diagnostics.get("option_premium") or 100.0
        
        max_loss_premium_points = sl_distance * 0.5
        max_loss_inr = max_loss_premium_points * lot_size * lots
        realized_pnl_inr = pnl_r * max_loss_inr
        capital_deployed_inr = option_premium * lot_size * lots
        is_win = pnl_r > 0
        
    return {
        "max_loss_inr": round(max_loss_inr, 2),
        "realized_pnl_inr": round(realized_pnl_inr, 2),
        "capital_deployed_inr": round(capital_deployed_inr, 2),
        "capital_efficiency_percent": round((realized_pnl_inr / capital_deployed_inr * 100) if capital_deployed_inr > 0 else 0.0, 2),
        "is_win": is_win,
        "pnl_r": pnl_r
    }


# ─── Load all data ─────────────────────────────────────────────────────────────

trades        = load_trades(date_str)
combo_trades  = load_combo_trades(date_str)
trade_events  = load_trade_events(date_str)
exec_events   = load_execution_events(date_str)
signals       = load_signal_audit(date_str)
cf_results    = load_cf_results(date_str)
risk_blocks   = load_risk_blocks(date_str)
exp_metrics   = load_experiment_metrics(date_str)

rejected_signals = [s for s in signals if not s["accepted"]]
accepted_signals = [s for s in signals if s["accepted"]]


# ─── Tabs ──────────────────────────────────────────────────────────────────────

tab_summary, tab_trades, tab_reasoning, tab_rejected, tab_blocks = st.tabs([
    "📊 Summary",
    f"📈 All Trades ({len(trades)})",
    "🧠 Trade Reasoning & Triggers",
    f"❌ Rejected Signals ({len(rejected_signals)})",
    f"🛡️ Risk Blocks ({len(risk_blocks)})",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    wins   = [t for t in trades if (t.get("final_pnl_r") or 0) > 0]
    losses = [t for t in trades if (t.get("final_pnl_r") or 0) < 0]
    total_pnl = sum((t.get("final_pnl_r") or 0) for t in trades)
    win_rate  = len(wins) / len(trades) * 100 if trades else 0
    exp_val   = total_pnl / len(trades) if trades else 0
    cf_pnl    = sum((r.get("final_pnl_r") or 0) for r in cf_results.values())

    st.markdown(f"### Session: **{date_str}**")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-label">Real Trades</div>
          <div class="stat-value">{len(trades)}</div>
          <div class="stat-sub">{len(wins)}W / {len(losses)}L</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        pnl_color = "#22c55e" if total_pnl > 0 else ("#ef4444" if total_pnl < 0 else "#64748b")
        st.markdown(f"""<div class="stat-card">
          <div class="stat-label">Total PnL</div>
          <div class="stat-value" style="color:{pnl_color}">{total_pnl:+.2f}R</div>
          <div class="stat-sub">Expectancy {exp_val:+.2f}R</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-label">Win Rate</div>
          <div class="stat-value">{win_rate:.0f}%</div>
          <div class="stat-sub">{len(accepted_signals)} signals accepted</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-label">Rejected</div>
          <div class="stat-value">{len(rejected_signals)}</div>
          <div class="stat-sub">{len(risk_blocks)} risk blocks</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        cf_pnl_color = "#22c55e" if cf_pnl > 0 else ("#ef4444" if cf_pnl < 0 else "#64748b")
        st.markdown(f"""<div class="stat-card">
          <div class="stat-label">CF PnL ({len(cf_results)} shadows)</div>
          <div class="stat-value" style="color:{cf_pnl_color}">{cf_pnl:+.2f}R</div>
          <div class="stat-sub">Shadow portfolio</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        best = max(exp_metrics, key=lambda x: float(x.get("expectancy") if x.get("expectancy") is not None else -999.0), default=None)
        worst = min(exp_metrics, key=lambda x: float(x.get("expectancy") if x.get("expectancy") is not None else 999.0), default=None)
        best_name = (best["experiment_name"].split("_")[0] if best else "—")
        best_exp_val = best.get("expectancy") if best else None
        best_exp_str = f"{best_exp_val:+.2f}R" if best_exp_val is not None else "—"
        st.markdown(f"""<div class="stat-card">
          <div class="stat-label">Best Experiment</div>
          <div class="stat-value" style="font-size:1rem">{best_name}</div>
          <div class="stat-sub">{best_exp_str} expectancy</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Experiment breakdown
    if exp_metrics:
        st.markdown("### Experiment Breakdown")
        df = pd.DataFrame(exp_metrics)
        df["win_rate"] = df.apply(
            lambda r: f"{r['wins']/(r['real_trades'] or 1)*100:.0f}%" if r["real_trades"] else "—", axis=1
        )
        df["expectancy"] = df["expectancy"].apply(lambda x: f"{x:+.2f}R" if x is not None else "—")
        df["total_pnl_r"] = df["total_pnl_r"].apply(lambda x: f"{x:+.2f}R" if x is not None else "—")
        df["avg_capture_rate"] = df["avg_capture_rate"].apply(lambda x: f"{x*100:.0f}%" if x else "—")
        df = df.rename(columns={
            "experiment_name": "Experiment", "real_trades": "Trades",
            "cf_trades": "Shadow", "win_rate": "Win Rate",
            "expectancy": "Expectancy", "total_pnl_r": "Total PnL",
            "avg_capture_rate": "Capture",
        })
        st.dataframe(
            df[["Experiment", "Trades", "Shadow", "Win Rate", "Expectancy", "Total PnL", "Capture"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No experiment_daily_metrics for this date. They are written by the EOD auditor.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ALL TRADES
# ══════════════════════════════════════════════════════════════════════════════
with tab_trades:
    if not trades:
        st.info(f"No trades found for {date_str}.")
    else:
        for i, t in enumerate(trades, 1):
            pnl = t.get("final_pnl_r")
            card_cls = "trade-card-win" if (pnl or 0) > 0 else ("trade-card-loss" if (pnl or 0) < 0 else "trade-card-flat")
            symbol_short = (t.get("symbol") or "").replace("NSE:", "").replace("-INDEX", "")

            # Resolve events timeline early to extract option contract symbol
            events = trade_events.get(t.get("trade_id"), [])
            ex_evts = exec_events.get(t.get("trade_id"), []) or exec_events.get(t.get("candidate_id"), [])
            all_evts = events + ex_evts
            
            # Heal shifted timestamps
            ref_time = t.get("entry_time")
            for ev in all_evts:
                curr_t = ev.get("timestamp")
                if curr_t and ref_time and curr_t.tzinfo and ref_time.tzinfo:
                    diff = (curr_t - ref_time).total_seconds()
                    if diff > 18000:
                        ev["timestamp"] = curr_t - timedelta(hours=5, minutes=30)
                    elif diff < -18000:
                        ev["timestamp"] = curr_t + timedelta(hours=5, minutes=30)
            
            all_evts = sorted(all_evts, key=lambda e: e["timestamp"])

            opt_sym = None
            for ev in all_evts:
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

            with st.expander(
                f"{color_r(pnl)} Trade #{i} — {t.get('setup_type','?')} {t.get('signal_type','')} "
                f"`{symbol_short}`{opt_display} | {t.get('experiment_name','?')} | "
                f"**{fmt_r(pnl)}** | {fmt_dt(t.get('entry_time'))}",
                expanded=(i == 1),
            ):
                st.markdown(f'<div class="trade-card {card_cls}">', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    ep_val = t.get("entry_price")
                    ep_str = f"₹{ep_val:,.2f}" if ep_val is not None else "—"
                    ex_val = t.get("exit_price")
                    ex_str = f"₹{ex_val:,.2f}" if ex_val is not None else "OPEN"
                    st.metric("Entry Price", ep_str)
                    st.metric("Exit Price", ex_str)
                with col2:
                    sl  = t.get("initial_stop_loss") or t.get("stop_loss")
                    tp  = t.get("initial_take_profit") or t.get("take_profit")
                    st.metric("Stop Loss", f"₹{sl:,.2f}" if sl else "—")
                    st.metric("Take Profit", f"₹{tp:,.2f}" if tp else "—")
                with col3:
                    st.metric("PnL", fmt_r(pnl))
                    st.metric("MFE / MAE", f"{fmt_r(t.get('mfe_r'))} / {fmt_r(t.get('mae_r'))}")
                with col4:
                    cap = t.get("capture_rate")
                    st.metric("Capture Rate", f"{cap*100:.0f}%" if cap else "—")
                    st.metric("Duration", f"{t.get('duration_minutes', 0):.0f} min / {t.get('bars_held', 0)} bars")

                if opt_sym:
                    st.markdown(f"🎳 **Option Contract:** `{opt_sym}`")
                else:
                    st.markdown("🎳 **Option Contract:** Index / Spot Only")

                # Diagnostics / factors
                diag = parse_json(t.get("diagnostics"))
                feats = parse_json(t.get("features"))
                if diag or feats:
                    st.markdown("**Market Conditions at Entry**")
                    combined = {**feats, **diag}
                    pills = ""
                    key_fields = ["rvol", "atr", "bos_trend", "market_regime", "daily_bias",
                                  "move_efficiency", "wickiness", "lots", "position_size_inr"]
                    for k in key_fields:
                        v = combined.get(k)
                        if v is not None:
                            pills += f'<span class="pill pill-info">{k}: {v}</span>'
                    if pills:
                        st.markdown(pills, unsafe_allow_html=True)

                # Score breakdown from signal_audit
                sig_match = next((s for s in accepted_signals if s.get("candidate_id") == t.get("candidate_id")), None)
                if sig_match:
                    sb = parse_json(sig_match.get("score_breakdown"))
                    if sb:
                        st.markdown("**Signal Score Breakdown**")
                        pills = ""
                        for factor, val in sb.items():
                            pill_cls = "pill-pass" if val else "pill-fail"
                            pills += f'<span class="pill {pill_cls}">{factor}</span>'
                        st.markdown(pills, unsafe_allow_html=True)
                    bias_info = (
                        f"Daily bias: **{sig_match.get('daily_bias')}** | "
                        f"Hourly: **{sig_match.get('hourly_bias')}** | "
                        f"Regime: **{sig_match.get('market_regime')}**"
                    )
                    st.markdown(bias_info)

                # Trade event timeline
                if all_evts:
                    st.markdown("**Event Timeline**")
                    def safe_val(v, sign=False, prefix=""):
                        if v is None:
                            return "—"
                        try:
                            val = float(v)
                            s = f"{val:+.2f}" if sign else f"{val:.2f}"
                            return f"{prefix}{s}"
                        except Exception:
                            return "—"

                    for ev in all_evts:
                        p = ev.get("payload", {})
                        etype = ev["event_type"]
                        ts = fmt_dt(ev["timestamp"])
                        if etype == "ENTRY":
                            desc = f"Entry @ {safe_val(p.get('entry_price'), prefix='₹')}  SL={safe_val(p.get('stop_loss'))}  TP={safe_val(p.get('take_profit'))}"
                        elif etype == "EXIT":
                            desc = f"Exit @ {safe_val(p.get('exit_price'), prefix='₹')}  reason={p.get('exit_reason','?')}  PnL={safe_val(p.get('final_pnl_r'), sign=True)}R"
                        elif etype == "SL_TRAIL":
                            desc = f"SL trailed → {safe_val(p.get('stop_loss'))}  MFE={safe_val(p.get('mfe_r'))}R"
                        elif etype == "STRIKE_SELECTED":
                            desc = f"Strike: {p.get('symbol','?')}  expiry={p.get('expiry','?')}"
                        elif etype == "PREMIUM_RETRIEVED":
                            desc = f"Premium: {safe_val(p.get('premium'), prefix='₹')}  bid={safe_val(p.get('bid'))}  ask={safe_val(p.get('ask'))}"
                        else:
                            desc = str(p)[:80]
                        st.markdown(
                            f'<div class="event-row"><span class="event-time">{ts}</span>'
                            f'<b>{etype}</b> — {desc}</div>',
                            unsafe_allow_html=True
                        )

                # Exit reason
                er = t.get("exit_reason")
                if er:
                    st.markdown(f"**Exit Reason:** `{er}`")

                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADE REASONING & TRIGGERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_reasoning:
    # 1. Prepare and merge all trades for this day
    day_trades = []
    for t in trades:
        tc = dict(t)
        tc["is_combo"] = False
        day_trades.append(tc)
        
    for ct in combo_trades:
        ctc = dict(ct)
        ctc["is_combo"] = True
        ctc["trade_id"] = ctc.get("combo_id")
        ctc["strategy"] = ctc.get("combo_type")
        ctc["entry_price"] = ctc.get("underlying_entry_price")
        ctc["exit_price"] = ctc.get("underlying_exit_price")
        ctc["candidate_id"] = ctc.get("combo_id")
        day_trades.append(ctc)
        
    if not day_trades:
        st.info("No trades executed on this day.")
    else:
        # Match with signal audit features in memory
        for t in day_trades:
            t["features"] = parse_json(t.get("features"))
            t["diagnostics"] = parse_json(t.get("diagnostics"))
            
            cand_id = t.get("candidate_id")
            sig_match = None
            if cand_id:
                sig_match = next((s for s in signals if s.get("candidate_id") == cand_id), None)
            if not sig_match:
                entry_time = t.get("entry_time")
                symbol = t.get("symbol")
                if entry_time and symbol:
                    for s in signals:
                        if s.get("symbol") == symbol and abs((s.get("timestamp") - entry_time).total_seconds()) <= 300:
                            sig_match = s
                            break
            if sig_match:
                t["features"] = parse_json(sig_match.get("score_breakdown")) or t["features"]
                t["market_regime"] = sig_match.get("market_regime") or t.get("market_regime")
                t["daily_bias"] = sig_match.get("daily_bias")
                t["hourly_bias"] = sig_match.get("hourly_bias")
                t["setup_type"] = sig_match.get("setup_type") or t.get("setup_type")
                t["rejection_reasons"] = parse_json(sig_match.get("rejection_reasons"))
                
        # Calculate metrics for each trade
        metrics_dict = {}
        for t in day_trades:
            metrics_dict[t["trade_id"]] = process_trade_metrics(t)
            
        # Group by experiment
        experiments = {}
        for t in day_trades:
            exp = t.get("experiment_name") or "Default"
            if exp not in experiments:
                experiments[exp] = []
            experiments[exp].append(t)
            
        # Summary Table at the Top
        st.subheader("📊 Session Experiment Summary")
        summary_md = [
            "| Experiment Name | R-Multiple Ledger | Family | Trades | Win Rate | Total R-PnL | Total PnL (₹) | Capital Deployed | Capital Efficiency |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]
        
        for exp, exp_trades in sorted(experiments.items()):
            tot = len(exp_trades)
            wins = sum(1 for t in exp_trades if metrics_dict[t["trade_id"]]["is_win"])
            wr = f"{(wins / tot * 100):.1f}%" if tot > 0 else "0.0%"
            is_combo = exp_trades[0].get("is_combo", False)
            family = "Combo Spreads" if is_combo else "Directional"
            r_type = "Premium R" if is_combo else "Index R"
            
            tot_r = sum(metrics_dict[t["trade_id"]]["pnl_r"] for t in exp_trades)
            tot_pnl_inr = sum(metrics_dict[t["trade_id"]]["realized_pnl_inr"] for t in exp_trades)
            avg_cap = sum(metrics_dict[t["trade_id"]]["capital_deployed_inr"] for t in exp_trades) / tot if tot > 0 else 0.0
            avg_eff = sum(metrics_dict[t["trade_id"]]["capital_efficiency_percent"] for t in exp_trades) / tot if tot > 0 else 0.0
            
            summary_md.append(
                f"| **`{exp}`** | *{r_type}* | {family} | {tot} | {wr} | {tot_r:+.2f}R | **{tot_pnl_inr:+,.2f} ₹** | {avg_cap:,.2f} ₹ | {avg_eff:+.2f}% |"
            )
            
        st.markdown("\n".join(summary_md))
        st.markdown("---")
        
        # Deduplicated trigger signals block
        st.subheader("🧠 Deduplicated Trigger Performance")
        unique_signals = {}
        for t in day_trades:
            entry_time_str = str(t.get("entry_time"))
            symbol = t.get("symbol")
            key = (symbol, entry_time_str)
            if key not in unique_signals:
                unique_signals[key] = []
            unique_signals[key].append(t)
            
        dedup_list = [v[0] for v in unique_signals.values()]
        tot_dedup = len(dedup_list)
        wins_dedup = sum(1 for t in dedup_list if metrics_dict[t["trade_id"]]["is_win"])
        wr_dedup = f"{(wins_dedup / tot_dedup * 100):.1f}%" if tot_dedup > 0 else "0.0%"
        pnl_dedup = sum(metrics_dict[t["trade_id"]]["realized_pnl_inr"] for t in dedup_list)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Unique Signal Triggers", tot_dedup)
        with c2:
            st.metric("Deduplicated Win Rate", wr_dedup)
        with c3:
            st.metric("Realized ₹ PnL (Deduplicated)", f"₹{pnl_dedup:+,.2f}")
            
        st.markdown("---")
        
        # Strategies details grouped by family
        families = {
            "🎯 Directional Single-Leg Strategy Ledgers": [e for e, ts in experiments.items() if not ts[0].get("is_combo")],
            "⚖️ Options Combination Spread Strategy Ledgers": [e for e, ts in experiments.items() if ts[0].get("is_combo")]
        }
        
        for fam_name, exp_names in families.items():
            if not exp_names:
                continue
            st.markdown(f"## {fam_name}")
            
            for exp in sorted(exp_names):
                exp_trades = experiments[exp]
                st.markdown(f"### 🧪 Experiment: `{exp}`")
                
                for idx, t in enumerate(exp_trades, 1):
                    met = metrics_dict[t["trade_id"]]
                    pnl_r = met["pnl_r"]
                    real_pnl = met["realized_pnl_inr"]
                    cap_dep = met["capital_deployed_inr"]
                    cap_eff = met["capital_efficiency_percent"]
                    symbol_short = (t.get("symbol") or "").replace("NSE:", "").replace("-INDEX", "")
                    
                    icon = "🟢" if pnl_r > 0 else ("🔴" if pnl_r < 0 else "⚪")
                    header_str = (
                        f"{icon} Trade #{idx}: {symbol_short} {t.get('setup_type','?')} {t.get('signal_type') or ''} | "
                        f"{pnl_r:+.3f}R ({real_pnl:+,.2f} ₹) | Cap Deployed: {cap_dep:,.2f} ₹ | Eff: {cap_eff:+.2f}%"
                    )
                    
                    with st.expander(header_str):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Entry → Exit", f"{t.get('entry_price')} → {t.get('exit_price') or 'OPEN'}")
                            st.metric("Exit Reason", t.get("exit_reason") or "N/A")
                        with col2:
                            r_label = "R PnL (Premium-based)" if t.get("is_combo") else "R PnL (Index points-based)"
                            st.metric(r_label, f"{pnl_r:+.3f}R")
                            st.metric("Realized PnL (₹)", f"₹{real_pnl:+,.2f}")
                        with col3:
                            st.metric("Capital Deployed", f"₹{cap_dep:,.2f}")
                            st.metric("Capital Efficiency", f"{cap_eff:+.2f}%")
                        with col4:
                            st.metric("Stop Loss", f"₹{t.get('stop_loss') or '—'}")
                            st.metric("Take Profit", f"₹{t.get('take_profit') or '—'}")
                            
                        if t.get("is_combo"):
                            st.markdown(f"**Legs Structure:** `{format_legs(t.get('legs'))}`")
                            st.markdown(f"**Max Loss points:** `{t.get('max_loss')} pts` | **Max Profit:** `{t.get('max_profit') or 'Unlimited'}`")
                            st.caption(f"*Note: Realized ₹ P&L is calculated as R-Multiple * (Max Loss * Lot Size of {get_lot_size(t.get('symbol'))}).*")
                        else:
                            opt_symbol = t.get("option_symbol") or t["diagnostics"].get("option_symbol")
                            opt_premium = t.get("option_premium") or t["diagnostics"].get("option_premium")
                            lots = t.get("lots") or t["diagnostics"].get("lots") or 1
                            if opt_symbol:
                                st.markdown(f"**Option Resolved:** `{opt_symbol}` @ premium of `₹{opt_premium}` (Lots: {lots})")
                            st.caption(f"*Note: R is measured in underlying index points. Realized ₹ P&L is estimated using an ATM delta of 0.5: R * (Index SL Distance * 0.5 * Lot Size of {get_lot_size(t.get('symbol'))} * Lots).*")
                            
                        st.markdown("#### Strategy Forensics")
                        
                        trigger_text = ""
                        strategy = t.get("strategy")
                        diagnostics = t.get("diagnostics") or {}
                        features = t.get("features") or {}
                        zone_explanation = diagnostics.get("zone_explanation")
                        
                        if "Geometry" in exp and strategy == "TRENDLINE_RETEST":
                            trigger_text = (
                                f"The system detected a `TRENDLINE_RETEST` setup under the `GeometryStrategy` — "
                                f"a previously broken trendline (role reversal: old support/resistance retested "
                                f"from the other side) was retested and held. "
                                f"The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. "
                                f"Daily bias was '{diagnostics.get('narrative_bias', 'NEUTRAL')}' with confidence {diagnostics.get('bias_confidence', 0.5)}."
                            )
                        elif "Geometry" in exp:
                            zone_desc = zone_explanation or "confluence zone"
                            trigger_text = (
                                f"The system detected a `{strategy}` setup under the `GeometryStrategy`. "
                                f"Specifically, price hit the {zone_desc}. "
                                f"The system confirmed the reversal with a candle body of at least 40% of its range and close in the reversal direction. "
                                f"Daily bias was '{diagnostics.get('narrative_bias', 'NEUTRAL')}' with confidence {diagnostics.get('bias_confidence', 0.5)}."
                            )
                        elif "Structural" in exp:
                            rvol = features.get("rvol") or diagnostics.get("rvol") or "N/A"
                            daily_bias = features.get("daily_bias") or "N/A"
                            hourly_bias = features.get("hourly_bias") or "N/A"
                            rvol_threshold = features.get("rvol_threshold")
                            if rvol_threshold is None:
                                rvol_threshold = diagnostics.get("rvol_threshold", 0.8)
                            trigger_text = (
                                f"The frozen core `StructuralStrategy` (`EnhancedStrategyEngine` {t.get('version') or 'v3.2'}) triggered a `{strategy}` setup. "
                                f"This occurred under Daily Bias '{daily_bias}' and Hourly Bias '{hourly_bias}'. "
                                f"The trigger was validated by a Relative Volume (RVOL) of {rvol} (threshold >= {rvol_threshold}). "
                            )
                            if strategy == "SWEEP":
                                trigger_text += "Price swept liquidity at a major HTF structure zone (Supply/Demand) and printed a strong 5m rejection body."
                            elif strategy == "BREAKOUT":
                                trigger_text += "Price broke out of a key 5m Swing High/Low Break of Structure (BOS) level with high move efficiency and low wickiness."
                            elif strategy == "TRAP":
                                trigger_text += "Price attempted a breakout but failed to follow through (FFT), trapping breakout buyers/sellers and triggering a reversal fade."
                        elif "OrderFlow" in exp:
                            trigger_text = (
                                f"The `OrderFlowStrategy` v1.0 identified an institutional stop hunt (sweep) or pullback into an unmitigated Fair Value Gap (FVG) imbalance. "
                                f"The setup triggered when price swept stops at a high-value liquidity pool (PDH/PDL or EQH/EQL) and printed a confirmation reversal candle."
                            )
                        elif "VWAP_Reclaim" in exp:
                            trigger_text = (
                                f"The `VwapReclaimStrategy` triggered on a trend-continuation crossover. "
                                f"The 5m close crossed over the intraday VWAP line, clearing it by an ATR-scaled buffer to confirm momentum in the reclaim direction (continuation, not reversion)."
                            )
                        elif "EMA_Pullback" in exp:
                            trigger_text = (
                                f"The `EmaPullbackStrategy` triggered on a trend-continuation setup. "
                                f"Price pulled back to touch the 20 EMA, and then printed a green/red confirmation body in the direction of the macro EMA trend (bullish/bearish crossover)."
                            )
                        elif "VerticalSpread" in exp:
                            trigger_text = (
                                f"This is a debit vertical spread (`{strategy}`) — the opposite hypothesis to the "
                                f"credit-spread family. It requires a genuine directional move: trending, EMA-aligned "
                                f"conditions with high RVOL and high move efficiency, not the range-bound/low-RVOL "
                                f"conditions the credit-spread and volatility-combo strategies look for."
                            )
                        elif t.get("is_combo"):
                            trigger_text = (
                                f"This is an options combination spread strategy (`{strategy}`). "
                                f"It was triggered based on volatility conditions. Specifically, range-bound / sideways conditions "
                                f"(e.g., low RVOL, low move efficiency) led the system to believe that price would consolidate. "
                                f"The spread was structured using ITM, ATM, and OTM strikes to capture time decay (theta) or volatility expansion/contraction."
                            )
                        else:
                            trigger_text = f"This trade was triggered by strategy `{strategy}` under experiment `{exp}` based on default momentum/reversal rules."
                            
                        sl_tp_text = ""
                        if "Geometry" in exp and strategy == "TRENDLINE_RETEST":
                            sl_tp_text = (
                                f"The Stop Loss was set at the retest candle's low/high minus/plus an ATR buffer "
                                f"(no supply/demand zone involved — this is a pure trendline role-reversal setup). "
                                f"The Take Profit targets the nearest opposing structural level, capped at `3 * ATR` from entry."
                            )
                        elif "Geometry" in exp:
                            sl_tp_text = (
                                f"The Stop Loss was set at `band_low - 0.15 * ATR` (for longs) or `band_high + 0.15 * ATR` (for shorts) to protect against breakouts past the confluence zone. "
                                f"The Take Profit was set at the opposing composite level or trendline, capped at `3 * ATR` from entry."
                            )
                        elif "Structural" in exp:
                            if strategy == "SWEEP":
                                sl_tp_text = (
                                    f"Stop Loss was placed 1 tick beyond the sweep wick (the invalidation point of the sweep thesis). "
                                    f"Take Profit was set at the nearest opposing Supply/Demand zone level, capped at `5 * ATR` from entry."
                                )
                            elif strategy == "BREAKOUT":
                                sl_tp_text = (
                                    f"Stop Loss was set 0.3 * ATR below/above the broken structure level, with a minimum stop distance of `0.5 * ATR`. "
                                    f"Take Profit was set at the nearest opposing zone level or a fallback projection of 2.0 * risk distance."
                                )
                            elif strategy == "TRAP":
                                sl_tp_text = (
                                    f"Stop Loss was placed 1 tick beyond the breakout high/low (since a break past the trap high invalidates the trap thesis). "
                                    f"Take Profit was set at the opposing zone."
                                )
                        elif "OrderFlow" in exp:
                            sl_tp_text = (
                                f"Stop Loss was set at the swept level +/- `0.15 * ATR` buffer, floored at `0.5 * ATR` from entry. "
                                f"Take Profit was placed at the nearest opposing liquidity target or FVG imbalance."
                            )
                        elif "VWAP_Reclaim" in exp:
                            sl_tp_text = (
                                f"Stop Loss was set at `low/high - 0.15 * ATR`, floored at `0.5 * ATR` from entry. "
                                f"Take Profit was placed at the next opposing zone, floored at `2.0 * R` to ensure positive risk-reward."
                            )
                        elif "EMA_Pullback" in exp:
                            sl_tp_text = (
                                f"Stop Loss was set below/above the 50 EMA with a small buffer (`0.2 * ATR`), floored at `0.5 * ATR` from entry. "
                                f"Take Profit was projected to the nearest resistance or fallback R-multiple."
                            )
                        elif t.get("is_combo"):
                            sl_tp_text = (
                                f"For options combination spreads, the stop loss and take profit are defined in terms of net premium multiples. "
                                f"The target profit was set at `{t.get('target_r') or '1.5'}R` and the stop loss was set at `{t.get('stop_r') or '-0.5'}R` net debit/credit change."
                            )
                        else:
                            sl_tp_text = f"Stop Loss and Take Profit were placed according to standard risk parameters (ATR buffers and opposing structures)."
                            
                        exit_reason = t.get("exit_reason")
                        exit_text = f"The trade exited due to `{exit_reason}`. "
                        if exit_reason == "INITIAL_SL":
                            exit_text += "Price went immediately against the setup and hit the initial stop loss level, invalidating the structural thesis."
                        elif exit_reason == "TRAILING_SL":
                            exit_text += "Price initially moved in favor of the trade, allowing the system to trail the stop loss (e.g., lock in profits or reduce risk), and eventually hit the trailing stop."
                        elif exit_reason == "SESSION_END":
                            exit_text += "The position was closed at the market close (15:25 IST) as a paper-trading session requirement."
                        elif exit_reason == "TARGET_R" or exit_reason == "TP_EXPANSION" or exit_reason == "STOP_R":
                            exit_text += f"Price reached the target or stop R multiple ({pnl_r:+.2f}R realized)."
                            
                        st.markdown(f"**Why it triggered:** {trigger_text}")
                        st.markdown(f"**SL/TP Placement Logic:** {sl_tp_text}")
                        st.markdown(f"**Exit Behavior:** {exit_text}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REJECTED SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_rejected:
    if not rejected_signals:
        st.info(f"No rejected signals found for {date_str}.")
    else:
        # Group by rejection reason for a quick summary
        from collections import Counter
        all_reasons = []
        for s in rejected_signals:
            all_reasons.extend(parse_json(s.get("rejection_reasons")) or [])
        reason_counts = Counter(all_reasons).most_common(10)

        st.markdown("### Top Rejection Reasons")
        reason_cols = st.columns(min(len(reason_counts), 5))
        for idx, (reason, cnt) in enumerate(reason_counts[:5]):
            with reason_cols[idx]:
                st.metric(reason, cnt)

        st.markdown(f"---\n### All {len(rejected_signals)} Rejected Signals")

        for sig in rejected_signals:
            reasons = parse_json(sig.get("rejection_reasons")) or []
            sb = parse_json(sig.get("score_breakdown")) or {}
            sym_short = (sig.get("symbol") or "").replace("NSE:", "").replace("-INDEX", "")
            cf = cf_results.get(sig.get("candidate_id"))

            cf_badge = ""
            if cf:
                cfp = cf.get("final_pnl_r")
                cf_badge = f" → CF: {fmt_r(cfp)}" if cfp is not None else ""

            with st.expander(
                f"❌ {fmt_dt(sig.get('timestamp'))} | {sig.get('setup_type','?')} `{sym_short}` "
                f"[{sig.get('experiment_name','?')}]{cf_badge}",
                expanded=False,
            ):
                st.markdown('<div class="rejected-card">', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    ep = sig.get("entry_price")
                    sl = sig.get("stop_loss")
                    tp = sig.get("take_profit")
                    rr = sig.get("rr_ratio")
                    ep_str = f"{ep:.2f}" if ep is not None else "—"
                    sl_str = f"{sl:.2f}" if sl is not None else "—"
                    tp_str = f"{tp:.2f}" if tp is not None else "—"
                    rr_str = f"{rr:.2f}" if rr is not None else "—"
                    st.markdown(
                        f"**Proposed:** Entry `{ep_str}` | SL `{sl_str}` | TP `{tp_str}` | RR `{rr_str}`"
                    )
                    st.markdown(
                        f"**Context:** bias={sig.get('daily_bias')} | "
                        f"hourly={sig.get('hourly_bias')} | regime={sig.get('market_regime')}"
                    )
                with c2:
                    if cf:
                        cfp = cf.get("final_pnl_r")
                        st.markdown(
                            f"**Counterfactual outcome:** {color_r(cfp)} **{fmt_r(cfp)}** | "
                            f"MFE={fmt_r(cf.get('mfe_r'))} | MAE={fmt_r(cf.get('mae_r'))} | "
                            f"Exit: `{cf.get('exit_reason','?')}`"
                        )

                # Rejection reasons
                st.markdown("**Rejection Reasons:**")
                pills = " ".join(f'<span class="pill pill-fail">{r}</span>' for r in reasons)
                st.markdown(pills or "_(none recorded)_", unsafe_allow_html=True)

                # Score breakdown — pass (green) vs fail (red)
                if sb:
                    st.markdown("**Score Breakdown:**")
                    pills = ""
                    for factor, val in sb.items():
                        pill_cls = "pill-pass" if val else "pill-fail"
                        pills += f'<span class="pill {pill_cls}">{factor}</span>'
                    st.markdown(pills, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK BLOCKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_blocks:
    if not risk_blocks:
        st.info(f"No risk governor blocks for {date_str}.")
    else:
        from collections import Counter
        gate_counts = Counter(b["gate_reason"] for b in risk_blocks).most_common()

        st.markdown("### Gate Reason Breakdown")
        gate_cols = st.columns(min(len(gate_counts), 5))
        for idx, (reason, cnt) in enumerate(gate_counts[:5]):
            with gate_cols[idx % 5]:
                st.metric(reason.replace("_", " "), cnt)

        st.markdown("---")
        for blk in risk_blocks:
            sym_short = (blk.get("symbol") or "").replace("NSE:", "").replace("-INDEX", "")
            with st.expander(
                f"🛡️ {fmt_dt(blk.get('timestamp'))} | {blk.get('gate_reason')} | "
                f"`{sym_short}` [{blk.get('experiment_name','?')}]",
                expanded=False,
            ):
                st.markdown('<div class="block-card">', unsafe_allow_html=True)
                ep = blk.get("entry_price")
                sl = blk.get("stop_loss")
                tp = blk.get("take_profit")
                rr = blk.get("rr_ratio")
                ep_str = f"{ep:.2f}" if ep is not None else "—"
                sl_str = f"{sl:.2f}" if sl is not None else "—"
                tp_str = f"{tp:.2f}" if tp is not None else "—"
                rr_str = f"{rr:.2f}" if rr is not None else "—"
                st.markdown(
                    f"**Blocked Trade:** Entry `{ep_str}` | SL `{sl_str}` | TP `{tp_str}` | RR `{rr_str}`"
                )
                st.markdown(
                    f"**Setup:** {blk.get('setup_type','?')} | Signal: {blk.get('signal_type','?')} | "
                    f"Gate: **`{blk.get('gate_reason')}`**"
                )
                st.markdown('</div>', unsafe_allow_html=True)
