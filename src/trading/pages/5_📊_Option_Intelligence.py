#!/usr/bin/env python3
"""
Option Intelligence Page.

Provides option chain tables, PCR metrics, max pain computations,
and Open Interest distribution charts.
Uses st.fragment to refresh data every 60 seconds without page flickers.
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

def parse_json(v):
    if v is None:
        return {}
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return {}


def is_market_hours():
    n = datetime.now(KOLKATA_TZ)
    return (
        n.weekday() < 5
        and (n.hour > 9 or (n.hour == 9 and n.minute >= 15))
        and (n.hour < 15 or (n.hour == 15 and n.minute <= 30))
    )


# ─── Option Chain Math ────────────────────────────────────────────────────────

def compute_pcr(chain_rows) -> float:
    """Put-Call Ratio based on total Open Interest."""
    total_calls = sum(float(r.get("oi") or 0.0) for r in chain_rows if r["option_type"] == "CE")
    total_puts  = sum(float(r.get("oi") or 0.0) for r in chain_rows if r["option_type"] == "PE")
    if total_calls == 0:
        return 1.0
    return total_puts / total_calls


def pcr_interpretation(pcr: float):
    if pcr > 1.25:
        return f"PCR={pcr:.2f} — Strong Bullish Bias (high put writing / support)", "pill-bias-bull"
    elif pcr < 0.75:
        return f"PCR={pcr:.2f} — Strong Bearish Bias (high call writing / resistance)", "pill-bias-bear"
    return f"PCR={pcr:.2f} — Neutral / Sideways Bias", "pill-neutral"


def compute_max_pain(chain_rows) -> float:
    """Calculates the strike price where option buyers lose the most money."""
    strikes = sorted(list(set(float(r["strike"]) for r in chain_rows)))
    if not strikes:
        return 0.0

    min_loss = float("inf")
    best_strike = strikes[0]

    # Map for fast lookup
    ce_map = {float(r["strike"]): float(r.get("oi") or 0.0) for r in chain_rows if r["option_type"] == "CE"}
    pe_map = {float(r["strike"]): float(r.get("oi") or 0.0) for r in chain_rows if r["option_type"] == "PE"}

    for candidate in strikes:
        total_loss = 0.0
        # Loss for call buyers if index expires at candidate
        for strike, oi in ce_map.items():
            if candidate > strike:
                total_loss += oi * (candidate - strike)
        # Loss for put buyers
        for strike, oi in pe_map.items():
            if candidate < strike:
                total_loss += oi * (strike - candidate)

        if total_loss < min_loss:
            min_loss = total_loss
            best_strike = candidate

    return best_strike


def build_chain_df(chain_rows, atm_strike) -> pd.DataFrame:
    """Builds a classic Option Chain layout (Calls on left, Puts on right)."""
    strikes = sorted(list(set(float(r["strike"]) for r in chain_rows)))
    
    result = []
    for s in strikes:
        c = next((r for r in chain_rows if r["strike"] == s and r["option_type"] == "CE"), {})
        p = next((r for r in chain_rows if r["strike"] == s and r["option_type"] == "PE"), {})
        
        result.append({
            "CE OI": int(c.get("oi") or 0),
            "CE OI Δ": int(c.get("oi_change") or 0),
            "CE Vol": int(c.get("volume") or 0),
            "CE LTP": round(float(c.get("ltp") or 0.0), 2),
            "Strike": int(s),
            "ATM": "◀ ATM" if int(s) == int(atm_strike) else "",
            "PE LTP": round(float(p.get("ltp") or 0.0), 2),
            "PE Vol": int(p.get("volume") or 0),
            "PE OI Δ": int(p.get("oi_change") or 0),
            "PE OI": int(p.get("oi") or 0),
        })
    return pd.DataFrame(result)


# ─── CSS ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Option Intelligence", page_icon="📊", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .page-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 16px; padding: 24px 32px; margin-bottom: 24px;
  }
  .page-header h1 { color: #f8fafc; margin: 0; font-size: 1.8rem; font-weight: 700; }
  .page-header p  { color: #94a3b8; margin: 4px 0 0; font-size: 0.9rem; }

  .kpi-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,41,59,0.9));
    border: 1px solid rgba(255,255,255,0.07); border-radius: 12px;
    padding: 18px 20px; text-align: center;
  }
  .kpi-label { color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .kpi-value { color: #f8fafc; font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
  .kpi-sub   { color: #94a3b8; font-size: 0.78rem; margin-top: 3px; }

  .pill {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 3px;
  }
  .pill-bias-bull { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.35); color: #4ade80; }
  .pill-bias-bear { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.35); color: #f87171; }
  .pill-neutral   { background: rgba(100,116,139,0.1); border: 1px solid rgba(100,116,139,0.3); color: #94a3b8; }
  .pill-info { background: rgba(6,182,212,0.08);  border: 1px solid rgba(6,182,212,0.25); color: #22d3ee; }

  .zone-card {
    padding: 10px 16px; border-radius: 8px; margin: 5px 0;
    display: flex; justify-content: space-between; align-items: center;
  }
  .zone-supply { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); }
  .zone-demand { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.3); }
  .zone-oi-res { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.3); }
  .zone-oi-sup { background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.3); }

  div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 14px;
  }

  .stale-warning {
    background: rgba(234,179,8,0.1); border: 1px solid rgba(234,179,8,0.4);
    border-radius: 8px; padding: 10px 16px; color: #facc15; font-size: 0.85rem;
    margin-bottom: 16px;
  }
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
  <h1>📊 Option Intelligence</h1>
  <p>Option chain, PCR, OI-based support & resistance, max pain, and trend inference</p>
</div>
""", unsafe_allow_html=True)

col_sym, col_mode = st.columns([2, 2])
with col_sym:
    symbol = st.selectbox("Underlying", SYMBOLS, key="oi_symbol")
with col_mode:
    view_mode = st.radio("Data Source", ["📦 Warehouse (historical)", "🔴 Live Fyers"], horizontal=True, key="oi_mode")

# ─── Load option chain data ────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def load_warehouse_chain(symbol):
    rows = db.get_option_chain_snapshot(symbol)
    return rows


def load_live_chain(symbol):
    """Fetch live option chain from Fyers (market hours only)."""
    try:
        from src.adapters.data.fyers_data_provider import FyersDataProvider
        dp = FyersDataProvider()
        ltp = dp.get_current_price(symbol)
        if not ltp:
            return None, None, []
        resolved = dp._find_active_expiry(symbol, ltp)
        if not resolved:
            return ltp, None, []
        expiry_str, expiry_date = resolved

        base = "BANKNIFTY" if "BANK" in symbol else "NIFTY"
        interval = 100 if "BANK" in symbol else 50
        atm = round(ltp / interval) * interval
        strikes = [int(atm + i * interval) for i in range(-6, 7)]

        rows = []
        client = dp.client
        for strike in strikes:
            for opt_type in ("CE", "PE"):
                sym = f"NSE:{base}{expiry_str}{strike}{opt_type}"
                depth = client.get_market_depth(sym)
                if depth:
                    rows.append({
                        "strike": float(strike),
                        "option_type": opt_type,
                        "ltp": depth.get("ltp", 0.0),
                        "bid": depth.get("bid", 0.0),
                        "ask": depth.get("ask", 0.0),
                        "volume": depth.get("volume", 0),
                        "oi": depth.get("oi", 0),
                        "oi_change": depth.get("oi_change", 0),
                        "expiry": expiry_date,
                    })
                import time; time.sleep(1.1)  # rate limit

        return ltp, expiry_date, rows
    except Exception as e:
        st.error(f"Live fetch error: {e}")
        return None, None, []


@st.cache_data(ttl=60, show_spinner=False)
def load_sr_zones_from_db(symbol):
    return db.get_sr_zones(symbol, active_only=True)


# ─── Fragmentized Dashboard ───────────────────────────────────────────────────

@st.fragment(run_every=60)
def render_option_intelligence(symbol, view_mode):
    chain_rows = []
    current_price = None
    expiry_date = None

    if view_mode.startswith("🔴"):
        if is_market_hours():
            with st.spinner("Fetching live option chain from Fyers..."):
                current_price, expiry_date, chain_rows = load_live_chain(symbol)
        else:
            st.markdown('<div class="stale-warning">⚠️ Market is closed. Showing warehouse data instead.</div>', unsafe_allow_html=True)
            chain_rows = load_warehouse_chain(symbol)
            if chain_rows:
                expiry_date = chain_rows[0].get("expiry")
    else:
        chain_rows = load_warehouse_chain(symbol)
        if chain_rows:
            expiry_date = chain_rows[0].get("expiry")
            last_time = max((r.get("time") for r in chain_rows if r.get("time")), default=None)
            if last_time:
                age_mins = (datetime.now(timezone.utc) - last_time.replace(tzinfo=timezone.utc if last_time.tzinfo is None else last_time.tzinfo)).total_seconds() / 60
                if age_mins > 10:
                    st.markdown(f'<div class="stale-warning">⚠️ Warehouse data is {age_mins:.0f} min old. Start the Option Warehouse for live OI data.</div>', unsafe_allow_html=True)

    sr_zones = load_sr_zones_from_db(symbol)

    if not chain_rows:
        st.warning(f"No option snapshot data found for {symbol}. Start the Option Warehouse (`python3 src/warehouse/option_warehouse.py`) during market hours.")
        return

    # ATM strike
    if current_price:
        interval = 100 if "BANK" in symbol else 50
        atm_strike = round(current_price / interval) * interval
    else:
        # Use strike with highest combined OI as proxy ATM
        oi_by_strike = {}
        for r in chain_rows:
            k = r["strike"]
            oi_by_strike[k] = oi_by_strike.get(k, 0) + (r.get("oi") or 0)
        atm_strike = max(oi_by_strike, key=oi_by_strike.get) if oi_by_strike else 0

    pcr = compute_pcr(chain_rows)
    max_pain = compute_max_pain(chain_rows)
    pcr_text, pcr_pill_cls = pcr_interpretation(pcr)

    # Straddle price (ATM CE + ATM PE LTP)
    atm_ce = next((r for r in chain_rows if r["strike"] == atm_strike and r["option_type"] == "CE"), {})
    atm_pe = next((r for r in chain_rows if r["strike"] == atm_strike and r["option_type"] == "PE"), {})
    straddle_price = (atm_ce.get("ltp") or 0.0) + (atm_pe.get("ltp") or 0.0)
    implied_move_pct = (straddle_price / (current_price or atm_strike or 1)) * 100 if straddle_price else None

    # Top OI strikes (resistance / support)
    ce_oi = [(r["strike"], r.get("oi") or 0) for r in chain_rows if r["option_type"] == "CE"]
    pe_oi = [(r["strike"], r.get("oi") or 0) for r in chain_rows if r["option_type"] == "PE"]
    top_ce = sorted(ce_oi, key=lambda x: x[1], reverse=True)[:3]  # highest call OI = resistance
    top_pe = sorted(pe_oi, key=lambda x: x[1], reverse=True)[:3]  # highest put OI = support

    # KPI Banner
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    sym_short = symbol.replace("NSE:", "").replace("-INDEX", "")

    with k1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Symbol</div>
          <div class="kpi-value" style="font-size:1.1rem">{sym_short}</div>
          <div class="kpi-sub">Expiry: {expiry_date or '—'}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        p = current_price or atm_strike
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Current / ATM</div>
          <div class="kpi-value">₹{p:,.0f}</div>
          <div class="kpi-sub">ATM strike: {atm_strike:,}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Put-Call Ratio</div>
          <div class="kpi-value">{pcr:.2f}</div>
          <div class="kpi-sub">{'Bullish' if pcr > 1.25 else ('Bearish' if pcr < 0.75 else 'Neutral')}</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Max Pain</div>
          <div class="kpi-value">{int(max_pain):,}</div>
          <div class="kpi-sub">Seller-optimal strike</div>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">Straddle Price</div>
          <div class="kpi-value">₹{straddle_price:.1f}</div>
          <div class="kpi-sub">Implied move: {f'{implied_move_pct:.2f}%' if implied_move_pct else '—'}</div>
        </div>""", unsafe_allow_html=True)
    with k6:
        top_res = top_ce[0][0] if top_ce else "—"
        top_sup = top_pe[0][0] if top_pe else "—"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">OI S/R</div>
          <div class="kpi-value" style="font-size:0.95rem">R: {top_res:,} / S: {top_sup:,}</div>
          <div class="kpi-sub">Highest OI strikes</div>
        </div>""", unsafe_allow_html=True)

    # PCR interpretation banner
    st.markdown(f'<span class="pill {pcr_pill_cls}">{pcr_text}</span>', unsafe_allow_html=True)

    if implied_move_pct:
        st.caption(
            f"Straddle price ₹{straddle_price:.1f} implies **±{implied_move_pct:.2f}%** "
            f"(±₹{straddle_price:.0f}) move by expiry for the straddle to be at breakeven."
        )

    st.markdown("---")

    tab_chain, tab_oi, tab_zones = st.tabs(["📋 Option Chain", "📈 OI Analysis", "🎯 S/R Zones"])

    with tab_chain:
        st.markdown("### Option Chain")
        df = build_chain_df(chain_rows, atm_strike)

        def style_chain(df):
            def highlight_row(row):
                if row["ATM"] == "◀ ATM":
                    return ["background-color: rgba(99,102,241,0.15)"] * len(row)
                if row["CE OI"] == df["CE OI"].max():
                    return ["color: #f87171" if c.startswith("CE") else "" for c in df.columns]
                if row["PE OI"] == df["PE OI"].max():
                    return ["color: #4ade80" if c.startswith("PE") else "" for c in df.columns]
                return [""] * len(row)
            return df.style.apply(highlight_row, axis=1)

        st.dataframe(style_chain(df), use_container_width=True, hide_index=True, height=450)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🔴 Top Call OI (Resistance)**")
            for strike, oi in top_ce:
                st.markdown(f'<div class="zone-card zone-oi-res"><span>Strike {int(strike):,}</span><span>{oi:,} OI</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown("**🟢 Top Put OI (Support)**")
            for strike, oi in top_pe:
                st.markdown(f'<div class="zone-card zone-oi-sup"><span>Strike {int(strike):,}</span><span>{oi:,} OI</span></div>', unsafe_allow_html=True)

    with tab_oi:
        st.markdown("### OI Distribution by Strike")
        df_oi = pd.DataFrame([
            {
                "Strike": int(r["strike"]),
                "CE OI": r.get("oi") or 0 if r["option_type"] == "CE" else 0,
                "PE OI": r.get("oi") or 0 if r["option_type"] == "PE" else 0,
            }
            for r in chain_rows
        ])
        df_oi = df_oi.groupby("Strike").sum().reset_index()
        df_oi = df_oi.set_index("Strike").sort_index()
        st.bar_chart(df_oi, use_container_width=True, height=320)

        st.markdown("### OI Change by Strike (Today)")
        df_delta = pd.DataFrame([
            {
                "Strike": int(r["strike"]),
                "CE OI Δ": r.get("oi_change") or 0 if r["option_type"] == "CE" else 0,
                "PE OI Δ": r.get("oi_change") or 0 if r["option_type"] == "PE" else 0,
            }
            for r in chain_rows
        ])
        df_delta = df_delta.groupby("Strike").sum().reset_index().set_index("Strike").sort_index()
        st.bar_chart(df_delta, use_container_width=True, height=280)

    with tab_zones:
        st.markdown("### 🎯 Persistent Support & Resistance Zones")
        if sr_zones:
            supply = [z for z in sr_zones if z["zone_type"] in ("SUPPLY", "OI_RESISTANCE")]
            demand = [z for z in sr_zones if z["zone_type"] in ("DEMAND", "OI_SUPPORT")]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**🔴 Supply / Resistance Zones ({len(supply)})**")
                for z in supply[:15]:
                    lo = z["price_low"]
                    hi = z["price_high"]
                    mid = (lo + hi) / 2
                    tc = z.get("touch_count", 1)
                    tf = z.get("timeframe", "h1")
                    source = "Option Chain (OI)" if tf == "options" else f"Price Action ({tf.upper()})"
                    st.markdown(
                        f'<div class="zone-card zone-supply">'
                        f'<span>₹{lo:,.1f} – ₹{hi:,.1f} <small>(mid {mid:,.0f})</small></span><br>'
                        f'<small style="color:#94a3b8">Source: {source} | Touches: {tc}</small></div>',
                        unsafe_allow_html=True
                    )
            with c2:
                st.markdown(f"**🟢 Demand / Support Zones ({len(demand)})**")
                for z in demand[:15]:
                    lo = z["price_low"]
                    hi = z["price_high"]
                    mid = (lo + hi) / 2
                    tc = z.get("touch_count", 1)
                    tf = z.get("timeframe", "h1")
                    source = "Option Chain (OI)" if tf == "options" else f"Price Action ({tf.upper()})"
                    st.markdown(
                        f'<div class="zone-card zone-demand">'
                        f'<span>₹{lo:,.1f} – ₹{hi:,.1f} <small>(mid {mid:,.0f})</small></span><br>'
                        f'<small style="color:#94a3b8">Source: {source} | Touches: {tc}</small></div>',
                        unsafe_allow_html=True
                    )

        if chain_rows:
            st.markdown("---")
            st.markdown("**📊 OI-Inferred Strike Zones (current chain)**")
            oi_zones = []
            for strike, oi_val in top_ce[:5]:
                oi_zones.append({"Type": "OI_RESISTANCE 🔴", "Strike": int(strike), "OI": f"{oi_val:,}", "Note": "Heavy call OI = likely cap"})
            for strike, oi_val in top_pe[:5]:
                oi_zones.append({"Type": "OI_SUPPORT 🟢", "Strike": int(strike), "OI": f"{oi_val:,}", "Note": "Heavy put OI = likely floor"})
            oi_zones.sort(key=lambda x: x["Strike"])
            if oi_zones:
                st.dataframe(pd.DataFrame(oi_zones), use_container_width=True, hide_index=True)

            mp = max_pain
            if mp:
                st.markdown(
                    f"**Max Pain Strike: `{int(mp):,}`** — "
                    f"This is the strike where the total payout to option buyers is minimized. "
                    f"Index has a statistical tendency to gravitate toward this level near expiry."
                )

# Execute the fragment
render_option_intelligence(symbol, view_mode)
