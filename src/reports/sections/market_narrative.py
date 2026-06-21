#!/usr/bin/env python3
"""Section 2 — Market Narrative."""

import logging
from datetime import datetime, timedelta, time as dtime
from typing import Any, Dict

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

SYMBOLS = {"nifty": "NSE:NIFTY50-INDEX", "banknifty": "NSE:NIFTYBANK-INDEX"}
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


class MarketNarrativeSection(BaseSection):
    section_id = "market_narrative"
    section_title = "Market Narrative"

    def compute(self) -> Dict[str, Any]:
        dt = datetime.strptime(self.date_str, "%Y-%m-%d")
        start = datetime.combine(dt.date(), MARKET_OPEN)
        end = datetime.combine(dt.date(), MARKET_CLOSE)

        nifty_5m = None
        bn_5m = None
        try:
            nifty_5m = self.data_provider.get_historical_data(
                SYMBOLS["nifty"], start - timedelta(days=1), end, "5"
            )
            bn_5m = self.data_provider.get_historical_data(
                SYMBOLS["banknifty"], start - timedelta(days=1), end, "5"
            )
        except Exception as e:
            logger.warning(f"Intraday fetch failed: {e}")

        # Dominant regime from trade_performance (if available) or infer from intraday
        regime_rows = self._query(
            """
            SELECT market_regime, COUNT(*) FROM trade_performance
            WHERE DATE(entry_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND market_regime IS NOT NULL
              AND valid = TRUE
            GROUP BY market_regime ORDER BY COUNT(*) DESC LIMIT 1
            """,
            (self.date_str,),
        )
        regime = regime_rows[0][0] if regime_rows else None

        narrative = self._build_narrative(nifty_5m, bn_5m, dt)
        narrative["regime_from_cf"] = regime
        return narrative

    def _build_narrative(self, nifty_5m, bn_5m, dt) -> dict:
        result = {
            "gap": {"direction": "Unknown", "pct": 0.0},
            "opening_range_pts": None,
            "trend_quality": None,
            "close_position": None,
            "session_high": None,
            "session_low": None,
            "session_open": None,
            "session_close": None,
            "bullets": [],
            "regime_from_cf": "UNKNOWN",
        }

        try:
            df = nifty_5m
            if df is None or df.empty:
                return result

            # Filter to today's session
            today = dt.date()
            session = df[df.index.date == today].copy()
            if session.empty:
                return result

            session_open = float(session.iloc[0]["open"])
            session_close = float(session.iloc[-1]["close"])
            session_high = float(session["high"].max())
            session_low = float(session["low"].min())

            # Previous day's close
            prev_day = df[df.index.date < today]
            prev_close = float(prev_day.iloc[-1]["close"]) if not prev_day.empty else session_open
            gap_pct = (session_open - prev_close) / prev_close * 100

            # Opening range (first 30 min = 6 x 5-min candles)
            opening_range = session.iloc[:6]
            or_high = float(opening_range["high"].max())
            or_low = float(opening_range["low"].min())
            or_range = or_high - or_low

            trend_quality = abs(session_close - session_open) / (session_high - session_low + 0.01)
            close_position = (session_close - session_low) / (session_high - session_low + 0.01)

            # Build narrative bullets
            bullets = []
            gap_dir = "up" if gap_pct > 0.15 else "down" if gap_pct < -0.15 else "flat"
            if gap_dir == "up":
                bullets.append(f"Gap-up open (+{gap_pct:.2f}%)")
            elif gap_dir == "down":
                bullets.append(f"Gap-down open ({gap_pct:.2f}%)")
            else:
                bullets.append("Flat open (gap < 0.15%)")

            if trend_quality > 0.60:
                direction = "bullish" if session_close > session_open else "bearish"
                bullets.append(f"Strong directional drive — {direction} trend day")
            elif trend_quality < 0.30:
                bullets.append("Choppy session — low directional conviction")
            else:
                bullets.append("Mixed session — partial trend with consolidation")

            if close_position > 0.80:
                bullets.append("Closed near session highs — strength into close")
            elif close_position < 0.20:
                bullets.append("Closed near session lows — weakness into close")
            else:
                bullets.append(f"Closed at {close_position*100:.0f}% of session range")

            result.update({
                "gap": {"direction": gap_dir.title(), "pct": round(gap_pct, 2)},
                "opening_range_pts": round(or_range, 2),
                "trend_quality": round(trend_quality, 2),
                "close_position": round(close_position, 2),
                "session_high": round(session_high, 2),
                "session_low": round(session_low, 2),
                "session_open": round(session_open, 2),
                "session_close": round(session_close, 2),
                "prev_close": round(prev_close, 2),
                "bullets": bullets,
            })

        except Exception as e:
            logger.warning(f"Narrative build failed: {e}")

        return result

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 2. Market Narrative\n"]

        if data.get("session_open") is None:
            lines.append("*Market data unavailable — API offline during report generation.*\n")
            return "\n".join(lines)

        gap = data["gap"]
        tq = data.get("trend_quality")
        cp = data.get("close_position")

        tq_str = f"{tq*10:.1f}/10" if tq is not None else "N/A"
        cp_pct = f"{cp*100:.0f}%" if cp is not None else "N/A"

        lines.append(
            f"| Metric | Value |\n|---|---|\n"
            f"| Gap | {gap['direction']} ({gap['pct']:+.2f}%) |\n"
            f"| Opening Range | {data.get('opening_range_pts', 'N/A')} pts |\n"
            f"| Trend Quality | {tq_str} |\n"
            f"| Close Position | {cp_pct} of session range |\n"
            f"| Session | {data.get('session_low','N/A')} — {data.get('session_high','N/A')} |\n"
            f"| Regime (CF-inferred) | **{data.get('regime_from_cf', 'N/A')}** |\n"
        )

        bullets = data.get("bullets", [])
        if bullets:
            lines.append("\n**What happened:**")
            for b in bullets:
                lines.append(f"- {b}")
        lines.append("")
        return "\n".join(lines)
