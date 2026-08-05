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
st.markdown(f'<meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}">', unsafe_allow_html=True)

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


# ─── Load all data ─────────────────────────────────────────────────────────────

trades        = load_trades(date_str)
trade_events  = load_trade_events(date_str)
exec_events   = load_execution_events(date_str)
signals       = load_signal_audit(date_str)
cf_results    = load_cf_results(date_str)
risk_blocks   = load_risk_blocks(date_str)
exp_metrics   = load_experiment_metrics(date_str)

rejected_signals = [s for s in signals if not s["accepted"]]
accepted_signals = [s for s in signals if s["accepted"]]


# ─── Tabs ──────────────────────────────────────────────────────────────────────

tab_summary, tab_trades, tab_rejected, tab_blocks = st.tabs([
    f"📊 Summary",
    f"📈 All Trades ({len(trades)})",
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

            with st.expander(
                f"{color_r(pnl)} Trade #{i} — {t.get('setup_type','?')} {t.get('signal_type','')} "
                f"`{symbol_short}` | {t.get('experiment_name','?')} | "
                f"**{fmt_r(pnl)}** | {fmt_dt(t.get('entry_time'))}",
                expanded=(i == 1),
            ):
                st.markdown(f'<div class="trade-card {card_cls}">', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Entry Price", f"₹{t.get('entry_price', 0):,.2f}")
                    st.metric("Exit Price",  f"₹{t.get('exit_price', 0):,.2f}" if t.get("exit_price") else "OPEN")
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
                events = trade_events.get(t.get("trade_id"), [])
                ex_evts = exec_events.get(t.get("trade_id"), []) or exec_events.get(t.get("candidate_id"), [])
                all_evts = sorted(events + ex_evts, key=lambda e: e["timestamp"])
                if all_evts:
                    st.markdown("**Event Timeline**")
                    for ev in all_evts:
                        p = ev.get("payload", {})
                        etype = ev["event_type"]
                        ts = fmt_dt(ev["timestamp"])
                        if etype == "ENTRY":
                            desc = f"Entry @ ₹{p.get('entry_price',0):.2f}  SL={p.get('stop_loss',0):.2f}  TP={p.get('take_profit',0):.2f}"
                        elif etype == "EXIT":
                            desc = f"Exit @ ₹{p.get('exit_price',0):.2f}  reason={p.get('exit_reason','?')}  PnL={p.get('final_pnl_r',0):+.2f}R"
                        elif etype == "SL_TRAIL":
                            desc = f"SL trailed → {p.get('stop_loss',0):.2f}  MFE={p.get('mfe_r',0):.2f}R"
                        elif etype == "STRIKE_SELECTED":
                            desc = f"Strike: {p.get('symbol','?')}  expiry={p.get('expiry','?')}"
                        elif etype == "PREMIUM_RETRIEVED":
                            desc = f"Premium: ₹{p.get('premium',0):.2f}  bid={p.get('bid',0):.2f}  ask={p.get('ask',0):.2f}"
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
                    st.markdown(
                        f"**Proposed:** Entry `{ep:.2f}` | SL `{sl:.2f}` | TP `{tp:.2f}` | RR `{rr:.2f}`"
                        if ep else "**Proposed:** prices not recorded"
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
                st.markdown(
                    f"**Blocked Trade:** Entry `{ep:.2f}` | SL `{sl:.2f}` | TP `{tp:.2f}` | RR `{rr:.2f}`"
                    if ep else "Prices not recorded"
                )
                st.markdown(
                    f"**Setup:** {blk.get('setup_type','?')} | Signal: {blk.get('signal_type','?')} | "
                    f"Gate: **`{blk.get('gate_reason')}`**"
                )
                st.markdown('</div>', unsafe_allow_html=True)
