#!/usr/bin/env python3
"""Section 10 — Tomorrow's Outlook.

Scenarios-based, not predictive. Derived from:
  - Today's close position and trend quality (from Fyers OHLC)
  - Session OHLC for structural levels
  - CF pattern (bias from today's dominant direction)

Wording is always conditional: "If X, then watch Y."
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

NIFTY = "NSE:NIFTY50-INDEX"
BANKNIFTY = "NSE:NIFTYBANK-INDEX"


class TomorrowOutlookSection(BaseSection):
    section_id = "tomorrow_outlook"
    section_title = "Tomorrow's Outlook"

    def compute(self) -> Dict[str, Any]:
        dt = datetime.strptime(self.date_str, "%Y-%m-%d")
        start = dt - timedelta(days=5)
        end = dt + timedelta(days=1)

        nifty_data, bn_data = None, None
        try:
            nifty_df = self.data_provider.get_historical_data(NIFTY, start, end, "D")
            bn_df = self.data_provider.get_historical_data(BANKNIFTY, start, end, "D")
            if nifty_df is not None and not nifty_df.empty:
                today_rows = nifty_df[nifty_df.index.date == dt.date()]
                if not today_rows.empty:
                    row = today_rows.iloc[-1]
                    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                    close_pos = (c - l) / (h - l + 0.01)
                    trend_q = abs(c - o) / (h - l + 0.01)
                    nifty_data = {
                        "open": o, "high": h, "low": l, "close": c,
                        "close_position": round(close_pos, 2),
                        "trend_quality": round(trend_q, 2),
                    }
            if bn_df is not None and not bn_df.empty:
                today_rows = bn_df[bn_df.index.date == dt.date()]
                if not today_rows.empty:
                    row = today_rows.iloc[-1]
                    o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                    close_pos = (c - l) / (h - l + 0.01)
                    trend_q = abs(c - o) / (h - l + 0.01)
                    bn_data = {
                        "open": o, "high": h, "low": l, "close": c,
                        "close_position": round(close_pos, 2),
                        "trend_quality": round(trend_q, 2),
                    }
        except Exception as e:
            logger.warning(f"Tomorrow outlook API fetch failed: {e}")

        # Dominant bias from today's CFs
        bias_rows = self._query(
            """
            SELECT signal_type, SUM(final_pnl_r), COUNT(*)
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND valid = TRUE
            GROUP BY signal_type
            ORDER BY SUM(final_pnl_r) DESC LIMIT 1
            """,
            (self.date_str,),
        )
        best_signal = bias_rows[0][0] if bias_rows else None

        scenarios, observations, watch_levels, avoid, prefer = (
            self._build_scenarios(nifty_data, bn_data, best_signal)
        )

        return {
            "nifty": nifty_data,
            "banknifty": bn_data,
            "best_signal_today": best_signal,
            "scenarios": scenarios,
            "observations": observations,
            "watch_levels": watch_levels,
            "avoid": avoid,
            "prefer": prefer,
        }

    def _build_scenarios(self, nifty, bn, best_signal):
        observations = []
        scenarios = []
        watch_levels = []
        avoid = "Opening breakout trades before 09:45 — market needs time to show direction"
        prefer = "Pullback entries once opening range (09:15–09:45) is established"

        ref = nifty or bn
        if ref is None:
            observations = ["Market data unavailable — structural analysis based on CF patterns only"]
            if best_signal == "BUY CALL":
                observations.append("CF patterns showed bullish edge today")
                scenarios = [
                    {"name": "Continuation", "pct": 55, "desc": "Gap-up or pullback-continuation if structure holds"},
                    {"name": "Reversal", "pct": 45, "desc": "Mean reversion if today's move was extended"},
                ]
            elif best_signal == "BUY PUT":
                observations.append("CF patterns showed bearish edge today")
                scenarios = [
                    {"name": "Continuation", "pct": 55, "desc": "Further downside if bearish structure holds"},
                    {"name": "Bounce", "pct": 45, "desc": "Dead-cat bounce or reversal from oversold levels"},
                ]
            else:
                scenarios = [
                    {"name": "Range", "pct": 50, "desc": "Continuation of ranging behaviour"},
                    {"name": "Breakout", "pct": 50, "desc": "Directional breakout once catalyst emerges"},
                ]
            return scenarios, observations, watch_levels, avoid, prefer

        cp = ref["close_position"]
        tq = ref["trend_quality"]
        h = ref["high"]
        l = ref["low"]
        c = ref["close"]

        # Observations
        if cp > 0.80:
            observations.append(f"Strong close near session highs ({cp*100:.0f}% of range)")
        elif cp < 0.20:
            observations.append(f"Weak close near session lows ({cp*100:.0f}% of range)")
        else:
            observations.append(f"Closed at {cp*100:.0f}% of session range — neutral positioning")

        if tq > 0.60:
            direction = "bullish" if c > ref["open"] else "bearish"
            observations.append(f"High trend quality ({tq*100:.0f}%) — {direction} conviction today")
        else:
            observations.append(f"Low trend quality ({tq*100:.0f}%) — choppy session")

        # Watch levels from today's OHLC
        if nifty:
            watch_levels += [
                {"label": "Today's High (supply)", "level": round(nifty["high"], 0), "symbol": "NIFTY"},
                {"label": "Today's Low (demand)", "level": round(nifty["low"], 0), "symbol": "NIFTY"},
                {"label": "Today's Close", "level": round(nifty["close"], 0), "symbol": "NIFTY"},
            ]
        if bn:
            watch_levels += [
                {"label": "Today's High (supply)", "level": round(bn["high"], 0), "symbol": "BANKNIFTY"},
                {"label": "Today's Low (demand)", "level": round(bn["low"], 0), "symbol": "BANKNIFTY"},
            ]

        # Scenario probabilities based on close position and trend quality
        if cp > 0.75 and tq > 0.55:
            scenarios = [
                {"name": "Scenario A — Pullback continuation", "pct": 60,
                 "desc": f"Pullback to {round(h*0.97,0)}-area, then continuation higher"},
                {"name": "Scenario B — Gap reversal", "pct": 40,
                 "desc": "Gap-up fade or early reversal if overnight catalyst is absent"},
            ]
            prefer = "Wait for first pullback after open, then enter on structure"
            avoid = "Chasing gap-up opens or buying into extended price action"
        elif cp < 0.25 and tq > 0.55:
            scenarios = [
                {"name": "Scenario A — Continuation lower", "pct": 60,
                 "desc": f"Break below today's low {round(l,0)} opens next support zone"},
                {"name": "Scenario B — Dead-cat bounce", "pct": 40,
                 "desc": "Short-covering bounce that fails at today's close or midpoint"},
            ]
            prefer = "Short bounces into resistance rather than buying dips"
            avoid = "Counter-trend CALL buying unless clear institutional absorption is visible"
        else:
            scenarios = [
                {"name": "Scenario A — Range continuation", "pct": 50,
                 "desc": f"Fade extremes within {round(l,0)}-{round(h,0)} until breakout"},
                {"name": "Scenario B — Directional breakout", "pct": 50,
                 "desc": "Clean RVOL breakout above/below today's range triggers thesis"},
            ]
            prefer = "Wait for RVOL confirmation before entering directional setups"
            avoid = "Low-RVOL breakout trades in the first 30 minutes"

        return scenarios, observations, watch_levels, avoid, prefer

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 10. Tomorrow's Outlook\n"]
        lines.append(
            "> *This is scenario preparation, not prediction. "
            "The goal is to know what you're watching and why.*\n"
        )

        obs = data.get("observations", [])
        if obs:
            lines.append("**Observations from today:**")
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
