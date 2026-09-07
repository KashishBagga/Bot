#!/usr/bin/env python3
"""Section 11 — Tomorrow's Outlook.

Scenarios-based, not predictive. Derived from:
  - Today's close position and trend quality (reused from the Market Narrative
    section's intraday snapshot — see rolling["market_narrative"]["by_symbol"])
  - Session OHLC for structural levels
  - Daily-timeframe supply/demand zones (ZoneEngine) for break-and-target levels
  - CF pattern (bias from today's dominant direction)

There is no overnight/global-cues data feed (no GIFT Nifty/SGX), so the gap
call below is a structural probability read from today's own close, not a
live overnight prediction. Wording is always conditional: "If X, then watch Y."
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.reports.base_section import BaseSection
from src.core.zone_engine import ZoneEngine, Zone

logger = logging.getLogger(__name__)

SYMBOLS = {"nifty": "NSE:NIFTY50-INDEX", "banknifty": "NSE:NIFTYBANK-INDEX"}


class TomorrowOutlookSection(BaseSection):
    section_id = "tomorrow_outlook"
    section_title = "Tomorrow's Outlook"

    def compute(self) -> Dict[str, Any]:
        by_symbol = self.rolling.get("market_narrative", {}).get("by_symbol") or {}
        nifty_data = by_symbol.get("nifty")
        bn_data = by_symbol.get("banknifty")

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

        playbooks = {}
        for key, session in (("nifty", nifty_data), ("banknifty", bn_data)):
            pb = self._build_playbook(key, session)
            if pb:
                playbooks[key] = pb

        return {
            "nifty": nifty_data,
            "banknifty": bn_data,
            "best_signal_today": best_signal,
            "scenarios": scenarios,
            "observations": observations,
            "watch_levels": watch_levels,
            "avoid": avoid,
            "prefer": prefer,
            "playbooks": playbooks,
        }

    # ── Structural playbook: gap call + zone targets + conditional trades ──

    def _build_playbook(self, key: str, session: Optional[dict]) -> Optional[Dict[str, Any]]:
        if not session:
            return None

        cp = session["close_position"]
        tq = session["trend_quality"]
        today_high = session["high"]
        today_low = session["low"]
        today_close = session["close"]

        gap_call = self._gap_call(cp, tq)

        daily_df = self._fetch_daily(SYMBOLS[key])
        res_target, sup_target = self._zone_targets(daily_df, today_close)
        weekly = self._weekly_view(daily_df)

        trades = self._trade_conditions(
            key, today_high, today_low, today_close, res_target, sup_target, gap_call
        )

        return {
            "gap_call": gap_call,
            "resistance_break_target": res_target,
            "support_break_target": sup_target,
            "weekly": weekly,
            "trade_conditions": trades,
        }

    @staticmethod
    def _gap_call(cp: float, tq: float) -> Dict[str, Any]:
        """Structural (not overnight-data-driven) read on tomorrow's likely open.
        No SGX/GIFT-Nifty feed exists in this system — this is a probability
        call from today's own close, framed as likely/low-chance, not a forecast."""
        if cp > 0.75 and tq > 0.55:
            return {
                "label": "Gap-up / higher open",
                "likely_pct": 55,
                "low_chance_label": "Flat or gap-down open",
                "low_chance_pct": 45,
                "reason": f"Strong close near highs ({cp*100:.0f}% of range) with high trend quality ({tq*100:.0f}%)",
            }
        if cp < 0.25 and tq > 0.55:
            return {
                "label": "Gap-down / lower open",
                "likely_pct": 55,
                "low_chance_label": "Flat or gap-up open",
                "low_chance_pct": 45,
                "reason": f"Weak close near lows ({cp*100:.0f}% of range) with high trend quality ({tq*100:.0f}%)",
            }
        return {
            "label": "Flat / normal open (within yesterday's range)",
            "likely_pct": 60,
            "low_chance_label": "Gap (either direction)",
            "low_chance_pct": 40,
            "reason": f"Neutral close ({cp*100:.0f}% of range) or choppy session (trend quality {tq*100:.0f}%) — no strong directional conviction into the close",
        }

    def _fetch_daily(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            dt = datetime.strptime(self.date_str, "%Y-%m-%d")
            df = self.data_provider.get_historical_data(
                symbol, dt - timedelta(days=150), dt + timedelta(days=1), "D"
            )
            return df
        except Exception as e:
            logger.warning(f"[tomorrow_outlook] daily fetch failed for {symbol}: {e}")
            return None

    @staticmethod
    def _zone_targets(daily_df: Optional[pd.DataFrame], close: float):
        """Nearest supply zone above close (break-of-high target) and nearest
        demand zone below close (break-of-low target), from daily-timeframe
        ZoneEngine zones."""
        if daily_df is None or len(daily_df) < 30:
            return None, None
        try:
            zones: List[Zone] = ZoneEngine().detect_zones(daily_df, timeframe="d1")
        except Exception as e:
            logger.warning(f"[tomorrow_outlook] zone detection failed: {e}")
            return None, None

        above = [z for z in zones if z.zone_type == "SUPPLY" and z.level > close]
        below = [z for z in zones if z.zone_type == "DEMAND" and z.level < close]

        res_target = None
        if above:
            z = min(above, key=lambda z: z.level)
            res_target = {"level": round(z.level, 0), "score": z.score,
                          "distance_pct": round((z.level - close) / close * 100, 2)}

        sup_target = None
        if below:
            z = max(below, key=lambda z: z.level)
            sup_target = {"level": round(z.level, 0), "score": z.score,
                          "distance_pct": round((close - z.level) / close * 100, 2)}

        return res_target, sup_target

    @staticmethod
    def _weekly_view(daily_df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Coarser 2-day/weekly framing from the last 5 daily candles: weekly
        range and a bias read using the same close-position/trend-quality
        heuristic as the single-day view, applied to the week's OHLC."""
        if daily_df is None or len(daily_df) < 5:
            return None
        week = daily_df.tail(5)
        w_high = float(week["high"].max())
        w_low = float(week["low"].min())
        w_open = float(week.iloc[0]["open"])
        w_close = float(week.iloc[-1]["close"])
        rng = w_high - w_low + 0.01
        cp = (w_close - w_low) / rng
        tq = abs(w_close - w_open) / rng

        if cp > 0.70 and tq > 0.5:
            bias = "Bullish — week closed strong near highs"
        elif cp < 0.30 and tq > 0.5:
            bias = "Bearish — week closed weak near lows"
        else:
            bias = "Range-bound — no clear weekly directional edge yet"

        return {
            "week_high": round(w_high, 0),
            "week_low": round(w_low, 0),
            "week_close": round(w_close, 0),
            "bias": bias,
        }

    @staticmethod
    def _trade_conditions(key, today_high, today_low, today_close, res_target, sup_target, gap_call):
        sym = key.upper()
        rows = []

        res_desc = (f"toward {res_target['level']} ({res_target['distance_pct']}% away, "
                    f"zone score {res_target['score']})") if res_target else "toward the next unfilled supply zone"
        sup_desc = (f"toward {sup_target['level']} ({sup_target['distance_pct']}% away, "
                    f"zone score {sup_target['score']})") if sup_target else "toward the next unfilled demand zone"

        rows.append({
            "condition": f"{sym} opens above today's high ({round(today_high,0)}) and the first 15-min candle "
                          f"closes above it — confirms gap-up holding, not fading",
            "trade": f"BUY CALL on the first pullback toward {round(today_high,0)}",
            "stop_loss": f"Below {round(today_high,0)} (structure invalidation)",
            "target": res_desc,
        })
        rows.append({
            "condition": f"{sym} opens below today's low ({round(today_low,0)}) and the first 15-min candle "
                          f"closes below it — confirms gap-down holding, not fading",
            "trade": f"BUY PUT on the first pullback toward {round(today_low,0)}",
            "stop_loss": f"Above {round(today_low,0)} (structure invalidation)",
            "target": sup_desc,
        })
        rows.append({
            "condition": f"{sym} opens within today's range ({round(today_low,0)}–{round(today_high,0)}) — flat/normal open",
            "trade": "No trade until the opening range (09:15–09:45) breaks with RVOL confirmation, "
                     "then trade in the breakout direction",
            "stop_loss": "Opposite side of the opening range",
            "target": f"{res_desc} on an upside break, {sup_desc} on a downside break",
        })
        rows.append({
            "condition": f"{sym} gaps (either direction) but reclaims today's close ({round(today_close,0)}) "
                          f"within the first 30 minutes — the gap is failing",
            "trade": "Fade the gap: trade back toward today's close/midpoint instead of the gap direction",
            "stop_loss": "Beyond the gap extreme",
            "target": f"Today's close {round(today_close,0)}, then session midpoint",
        })
        return rows

    # ── Existing single-day scenario logic (unchanged) ──────────────────────

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
            "> *This is scenario preparation, not prediction. There is no overnight/global-cues "
            "feed in this system, so the gap call below is a structural read from today's own "
            "close — not a live overnight forecast. The goal is to know what you're watching, "
            "why, and what you'd actually do about it.*\n"
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

        playbooks = data.get("playbooks", {})
        for key, pb in playbooks.items():
            sym = key.upper()
            gap = pb["gap_call"]
            lines.append(f"\n### {sym} — Tomorrow's Playbook\n")
            lines.append(
                f"**Gap call:** {gap['label']} — **{gap['likely_pct']}% likely** "
                f"(vs. {gap['low_chance_label']}, {gap['low_chance_pct']}%)  \n"
                f"*Reason:* {gap['reason']}\n"
            )

            res_t = pb.get("resistance_break_target")
            sup_t = pb.get("support_break_target")
            if res_t or sup_t:
                lines.append("**Break-and-target levels:**\n")
                if res_t:
                    lines.append(f"- If today's high breaks and holds → next supply zone at "
                                  f"**{res_t['level']}** ({res_t['distance_pct']}% away, zone score {res_t['score']})")
                if sup_t:
                    lines.append(f"- If today's low breaks and holds → next demand zone at "
                                  f"**{sup_t['level']}** ({sup_t['distance_pct']}% away, zone score {sup_t['score']})")
                lines.append("")

            trades = pb.get("trade_conditions", [])
            if trades:
                lines.append("**If this happens → take this trade:**\n")
                lines.append("| Condition | Trade | Stop Loss | Target |\n|---|---|---|---|")
                for t in trades:
                    lines.append(f"| {t['condition']} | {t['trade']} | {t['stop_loss']} | {t['target']} |")
                lines.append("")

            weekly = pb.get("weekly")
            if weekly:
                lines.append(
                    f"**Week-ahead framing:** {weekly['bias']}  \n"
                    f"Weekly range: {weekly['week_low']}–{weekly['week_high']}, last close {weekly['week_close']}. "
                    f"A break of this range (not just today's) is the higher-conviction 2–5 day signal.\n"
                )

        return "\n".join(lines)
