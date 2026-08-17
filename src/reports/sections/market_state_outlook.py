#!/usr/bin/env python3
"""Section 10 — Market State & Outlook.

Replaces Tomorrow's Outlook with a fact-backed synthesis of everything the
system already collects but never brought together in one place: VIX,
option-chain PCR/max-pain/OI walls, S/R zone touches, regime, and the
trailing multi-day/week performance trend — on top of the same
gap/trend-quality/close-position read Market Narrative already computes.

Every claim cites the number behind it. The "Tomorrow" block stays
scenario-based, not predictive ("if X, then watch Y"), same convention the
section it replaces used — it now also folds in PCR bias and OI-wall
proximity, not just close-position/trend-quality.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.reports.base_section import BaseSection
from src.reports.sections.tomorrow_outlook import TomorrowOutlookSection
from src.core.options_intelligence_engine import compute_pcr, compute_max_pain, top_oi_walls, pcr_bias

logger = logging.getLogger(__name__)

SYMBOLS = {"nifty": "NSE:NIFTY50-INDEX", "banknifty": "NSE:NIFTYBANK-INDEX"}
VIX_SYMBOL = "NSE:INDIAVIX-INDEX"
TRAILING_DAYS = 7


class MarketStateOutlookSection(BaseSection):
    section_id = "market_state_outlook"
    section_title = "Market State & Outlook"

    def compute(self) -> Dict[str, Any]:
        by_symbol = self.rolling.get("market_narrative", {}).get("by_symbol") or {}
        nifty_data = by_symbol.get("nifty")
        bn_data = by_symbol.get("banknifty")

        vix = self._vix_block()
        options = {
            "nifty": self._options_block("nifty"),
            "banknifty": self._options_block("banknifty"),
        }
        zones_today = self._zones_tested_today(nifty_data, bn_data)
        trailing = self._trailing_performance()

        best_signal = self._best_signal_today()
        scenarios, observations, watch_levels, avoid, prefer = TomorrowOutlookSection._build_scenarios(
            None, nifty_data, bn_data, best_signal
        )
        scenarios, observations, watch_levels = self._condition_on_options(
            scenarios, observations, watch_levels, options, nifty_data, bn_data
        )

        return {
            "nifty": nifty_data,
            "banknifty": bn_data,
            "vix": vix,
            "options": options,
            "zones_today": zones_today,
            "trailing": trailing,
            "best_signal_today": best_signal,
            "scenarios": scenarios,
            "observations": observations,
            "watch_levels": watch_levels,
            "avoid": avoid,
            "prefer": prefer,
        }

    # ── Data blocks ──────────────────────────────────────────────────────

    def _vix_block(self) -> Dict[str, Any]:
        """Today's VIX close + change vs prior close, plus a trailing week of
        closes for a simple up/down read on implied-vol regime."""
        try:
            dt = datetime.strptime(self.date_str, "%Y-%m-%d")
            end = dt + timedelta(days=1)
            start = dt - timedelta(days=TRAILING_DAYS + 5)  # pad for weekends/holidays
            df = self.data_provider.get_historical_data(VIX_SYMBOL, start, end, "D")
            if df is None or df.empty:
                return {"available": False}
            df = df[df.index.date <= dt.date()]
            if df.empty:
                return {"available": False}
            closes = df["close"].tail(TRAILING_DAYS + 1).tolist()
            today_close = float(closes[-1])
            prior_close = float(closes[-2]) if len(closes) >= 2 else today_close
            change_pct = (today_close - prior_close) / prior_close * 100 if prior_close else 0.0
            week_ago = float(closes[0])
            week_trend = "RISING" if today_close > week_ago * 1.05 else (
                "FALLING" if today_close < week_ago * 0.95 else "STABLE"
            )
            return {
                "available": True,
                "close": round(today_close, 2),
                "change_pct": round(change_pct, 2),
                "trailing_closes": [round(c, 2) for c in closes],
                "week_trend": week_trend,
            }
        except Exception as e:
            logger.warning(f"VIX block failed: {e}")
            return {"available": False}

    def _options_block(self, key: str) -> Dict[str, Any]:
        """End-of-day option-chain read: latest snapshot per strike/type today,
        fed through the same PCR/max-pain/OI-wall math the live strategies use
        (src.core.options_intelligence_engine), not re-derived."""
        underlying = SYMBOLS[key]
        rows = self._query(
            """
            SELECT DISTINCT ON (strike, option_type) strike, option_type, oi, time
            FROM option_snapshots
            WHERE underlying = %s AND DATE(time AT TIME ZONE 'Asia/Kolkata') = %s
            ORDER BY strike, option_type, time DESC
            """,
            (underlying, self.date_str),
        )
        if not rows:
            return {"available": False}
        chain_rows = [{"strike": float(r[0]), "option_type": r[1], "oi": r[2]} for r in rows]
        pcr = compute_pcr(chain_rows)
        return {
            "available": True,
            "pcr": round(pcr, 2) if pcr is not None else None,
            "pcr_bias": pcr_bias(pcr),
            "max_pain": compute_max_pain(chain_rows),
            "call_walls": [{"strike": w.strike, "oi": w.oi} for w in top_oi_walls(chain_rows, "CE")[:3]],
            "put_walls": [{"strike": w.strike, "oi": w.oi} for w in top_oi_walls(chain_rows, "PE")[:3]],
        }

    def _zones_tested_today(self, nifty_data, bn_data) -> List[Dict[str, Any]]:
        """S/R zones touched today, with whether today's session held or broke
        through them (compares the zone band against today's session high/low
        from Market Narrative — single source of truth, not re-pulled)."""
        rows = self._query(
            """
            SELECT symbol, zone_type, price_low, price_high, strength, touch_count
            FROM sr_zones
            WHERE DATE(last_tested AT TIME ZONE 'Asia/Kolkata') = %s AND active = TRUE
            ORDER BY strength DESC LIMIT 10
            """,
            (self.date_str,),
        )
        session_by_symbol = {SYMBOLS["nifty"]: nifty_data, SYMBOLS["banknifty"]: bn_data}
        result = []
        for symbol, zone_type, price_low, price_high, strength, touch_count in rows:
            session = session_by_symbol.get(symbol)
            outcome = "UNKNOWN"
            if session:
                mid = (float(price_low) + float(price_high)) / 2
                if session["low"] <= mid <= session["high"]:
                    outcome = "BROKEN" if (
                        (zone_type == "SUPPLY" and session["close"] > price_high)
                        or (zone_type == "DEMAND" and session["close"] < price_low)
                    ) else "HELD"
            result.append({
                "symbol": symbol, "zone_type": zone_type,
                "price_low": round(float(price_low), 1), "price_high": round(float(price_high), 1),
                "strength": round(float(strength), 1), "touch_count": touch_count,
                "outcome": outcome,
            })
        return result

    def _trailing_performance(self) -> Dict[str, Any]:
        """Trailing week of real-trade daily PnL + dominant regime — "how the
        past few days have traded", not just today."""
        daily_rows = self._query(
            """
            SELECT DATE(exit_time AT TIME ZONE 'Asia/Kolkata') AS d,
                   COALESCE(SUM(final_pnl_r), 0), COUNT(*)
            FROM trade_performance
            WHERE exit_time IS NOT NULL AND valid = TRUE
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata')
                  BETWEEN %s::date - %s AND %s::date
            GROUP BY d ORDER BY d
            """,
            (self.date_str, TRAILING_DAYS - 1, self.date_str),
        )
        regime_rows = self._query(
            """
            SELECT market_regime, COUNT(*) FROM trade_performance
            WHERE valid = TRUE AND market_regime IS NOT NULL
              AND DATE(entry_time AT TIME ZONE 'Asia/Kolkata')
                  BETWEEN %s::date - %s AND %s::date
            GROUP BY market_regime ORDER BY COUNT(*) DESC
            """,
            (self.date_str, TRAILING_DAYS - 1, self.date_str),
        )
        daily = [{"date": str(d), "pnl_r": round(float(p), 2), "trades": int(n)} for d, p, n in daily_rows]
        return {
            "daily": daily,
            "total_pnl_r": round(sum(d["pnl_r"] for d in daily), 2),
            "dominant_regime": regime_rows[0][0] if regime_rows else None,
            "regime_breakdown": [{"regime": r, "count": int(c)} for r, c in regime_rows],
        }

    def _best_signal_today(self) -> Optional[str]:
        rows = self._query(
            """
            SELECT signal_type, SUM(final_pnl_r) FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') = %s AND valid = TRUE
            GROUP BY signal_type ORDER BY SUM(final_pnl_r) DESC LIMIT 1
            """,
            (self.date_str,),
        )
        return rows[0][0] if rows else None

    def _condition_on_options(self, scenarios, observations, watch_levels, options, nifty_data, bn_data):
        """Layer PCR bias / OI-wall proximity onto the price-action-only
        scenarios from TomorrowOutlookSection — flags agreement/conflict
        rather than silently picking one signal over the other."""
        nifty_opts = options.get("nifty") or {}
        if nifty_opts.get("available") and nifty_data:
            close = nifty_data["close"]
            bias = nifty_opts["pcr_bias"]
            close_dir = "BULLISH" if nifty_data["close_position"] > 0.6 else (
                "BEARISH" if nifty_data["close_position"] < 0.4 else "NEUTRAL"
            )
            if bias in ("BULLISH", "BEARISH") and close_dir in ("BULLISH", "BEARISH"):
                if bias == close_dir:
                    observations.append(
                        f"NIFTY PCR {nifty_opts['pcr']} ({bias.lower()}) agrees with today's "
                        f"{close_dir.lower()} close position — same-direction confirmation, not two independent signals"
                    )
                else:
                    observations.append(
                        f"⚠️ NIFTY PCR {nifty_opts['pcr']} ({bias.lower()}) conflicts with today's "
                        f"{close_dir.lower()} close position — lower conviction on directional continuation"
                    )
            for wall in nifty_opts.get("call_walls", [])[:1]:
                if abs(wall["strike"] - close) / close < 0.01:
                    watch_levels.append({
                        "label": f"Call OI wall (resistance, OI={wall['oi']:,})",
                        "level": wall["strike"], "symbol": "NIFTY",
                    })
            for wall in nifty_opts.get("put_walls", [])[:1]:
                if abs(wall["strike"] - close) / close < 0.01:
                    watch_levels.append({
                        "label": f"Put OI wall (support, OI={wall['oi']:,})",
                        "level": wall["strike"], "symbol": "NIFTY",
                    })
        return scenarios, observations, watch_levels

    # ── Rendering ────────────────────────────────────────────────────────

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 10. Market State & Outlook\n"]

        # What happened today
        lines.append("### What happened today\n")
        vix = data.get("vix") or {}
        if vix.get("available"):
            arrow = "▲" if vix["change_pct"] > 0 else ("▼" if vix["change_pct"] < 0 else "→")
            lines.append(f"- **VIX:** {vix['close']} ({arrow} {vix['change_pct']:+.2f}%) — trailing week {vix['week_trend'].lower()}")
        for key, label in (("nifty", "NIFTY"), ("banknifty", "BANKNIFTY")):
            opts = (data.get("options") or {}).get(key) or {}
            if opts.get("available"):
                mp = f", max pain {opts['max_pain']}" if opts.get("max_pain") else ""
                lines.append(f"- **{label} PCR:** {opts['pcr']} ({opts['pcr_bias'].lower()}){mp}")
                if opts.get("call_walls"):
                    w = opts["call_walls"][0]
                    lines.append(f"  - Nearest call OI wall (resistance): {w['strike']} (OI {w['oi']:,})")
                if opts.get("put_walls"):
                    w = opts["put_walls"][0]
                    lines.append(f"  - Nearest put OI wall (support): {w['strike']} (OI {w['oi']:,})")

        zones = data.get("zones_today") or []
        if zones:
            lines.append("\n**S/R zones tested today:**\n")
            lines.append("| Symbol | Type | Band | Strength | Touches | Outcome |\n|---|---|---|---|---|---|")
            for z in zones[:6]:
                lines.append(
                    f"| {z['symbol']} | {z['zone_type']} | {z['price_low']}-{z['price_high']} "
                    f"| {z['strength']} | {z['touch_count']} | {z['outcome']} |"
                )

        # How the past week has traded
        trailing = data.get("trailing") or {}
        if trailing.get("daily"):
            lines.append("\n### How the past week has traded\n")
            lines.append(f"- **Trailing {len(trailing['daily'])}-day real PnL:** {trailing['total_pnl_r']:+.2f}R")
            if trailing.get("dominant_regime"):
                lines.append(f"- **Dominant regime:** {trailing['dominant_regime']}")
            lines.append("\n| Date | PnL | Trades |\n|---|---|---|")
            for d in trailing["daily"]:
                lines.append(f"| {d['date']} | {d['pnl_r']:+.2f}R | {d['trades']} |")

        # Tomorrow
        lines.append("\n### Tomorrow\n")
        lines.append(
            "> *This is scenario preparation, not prediction. "
            "The goal is to know what you're watching and why.*\n"
        )
        obs = data.get("observations", [])
        if obs:
            lines.append("**Observations:**")
            for o in obs:
                lines.append(f"- {o}")
            lines.append("")
        scenarios = data.get("scenarios", [])
        if scenarios:
            lines.append("**Scenarios:**\n")
            for s in scenarios:
                lines.append(f"**{s['name']} ({s['pct']}%)**  \n{s['desc']}\n")
        watch = data.get("watch_levels", [])
        if watch:
            lines.append("**Key Levels to Watch:**\n")
            lines.append("| Symbol | Level | Label |\n|---|---|---|")
            for w in watch:
                lines.append(f"| {w['symbol']} | {w['level']} | {w['label']} |")
            lines.append("")
        lines.append(f"**Prefer:** {data.get('prefer', '—')}  \n")
        lines.append(f"**Avoid:** {data.get('avoid', '—')}\n")

        return "\n".join(lines)
