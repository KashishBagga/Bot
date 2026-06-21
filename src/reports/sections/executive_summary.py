#!/usr/bin/env python3
"""Section 1 — Executive Summary."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

SYMBOLS = {
    "nifty": "NSE:NIFTY50-INDEX",
    "banknifty": "NSE:NIFTYBANK-INDEX",
    "vix": "NSE:INDIA VIX-INDEX",
}


class ExecutiveSummarySection(BaseSection):
    section_id = "executive_summary"
    section_title = "Executive Summary"

    def compute(self) -> Dict[str, Any]:
        market = self._fetch_market(self.date_str)
        real = self._fetch_real(self.date_str)
        cf = self._fetch_cf(self.date_str)
        experiments = self._fetch_experiments(self.date_str)
        return {
            "date": self.date_str,
            "market": market,
            "real": real,
            "cf": cf,
            "experiments": experiments,
        }

    # ── Market data ─────────────────────────────────────────────────────────

    def _fetch_market(self, date_str: str) -> dict:
        result = {"nifty": None, "banknifty": None, "vix": None, "day_type": "Unknown"}
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            start = dt - timedelta(days=3)
            end = dt + timedelta(days=1)

            def _ohlc(sym: str) -> Optional[dict]:
                df = self.data_provider.get_historical_data(sym, start, end, "D")
                if df is None or df.empty:
                    return None
                # Filter to date
                df_today = df[df.index.date == dt.date()]
                df_prev = df[df.index.date < dt.date()]
                if df_today.empty:
                    return None
                row = df_today.iloc[-1]
                prev_close = float(df_prev.iloc[-1]["close"]) if not df_prev.empty else float(row["open"])
                change_pct = (float(row["close"]) - prev_close) / prev_close * 100
                gap_pct = (float(row["open"]) - prev_close) / prev_close * 100
                close_pos = (float(row["close"]) - float(row["low"])) / (float(row["high"]) - float(row["low"]) + 0.01)
                trend_q = abs(float(row["close"]) - float(row["open"])) / (float(row["high"]) - float(row["low"]) + 0.01)
                return {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "prev_close": prev_close,
                    "change_pct": round(change_pct, 2),
                    "gap_pct": round(gap_pct, 2),
                    "close_position": round(close_pos, 2),
                    "trend_quality": round(trend_q, 2),
                }

            result["nifty"] = _ohlc(SYMBOLS["nifty"])
            result["banknifty"] = _ohlc(SYMBOLS["banknifty"])

            # VIX
            try:
                vix_val = self.data_provider.get_current_price(SYMBOLS["vix"])
                result["vix"] = {"value": vix_val}
            except Exception:
                result["vix"] = {"value": None}

            # Day type classification
            bn = result["banknifty"]
            if bn:
                if bn["trend_quality"] > 0.65 and bn["change_pct"] > 0.5:
                    result["day_type"] = "Trend Day ↑"
                elif bn["trend_quality"] > 0.65 and bn["change_pct"] < -0.5:
                    result["day_type"] = "Trend Day ↓"
                elif abs(bn["gap_pct"]) > 0.4:
                    result["day_type"] = "Gap Day"
                elif bn["trend_quality"] < 0.3:
                    result["day_type"] = "Choppy / Ranging"
                else:
                    result["day_type"] = "Balanced Day"
        except Exception as e:
            logger.warning(f"Market data fetch failed: {e}")
        return result

    # ── DB queries ───────────────────────────────────────────────────────────

    def _fetch_real(self, date_str: str) -> dict:
        rows = self._query(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN final_pnl_r > 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN final_pnl_r <= 0 THEN 1 ELSE 0 END),
                   COALESCE(SUM(final_pnl_r), 0),
                   COALESCE(AVG(final_pnl_r), 0)
            FROM trade_performance
            WHERE DATE(entry_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND valid = TRUE
            """,
            (date_str,),
        )
        r = rows[0]
        n = int(r[0] or 0)
        wins = int(r[1] or 0)
        losses = int(r[2] or 0)
        return {
            "trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / n, 2) if n > 0 else 0.0,
            "total_pnl_r": round(float(r[3]), 2),
            "expectancy": round(float(r[4]), 2),
        }

    def _fetch_cf(self, date_str: str) -> dict:
        rows = self._query(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN final_pnl_r > 0 THEN 1 ELSE 0 END),
                   COALESCE(AVG(final_pnl_r), 0),
                   COALESCE(SUM(final_pnl_r), 0)
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND valid = TRUE
            """,
            (date_str,),
        )
        r = rows[0]
        n = int(r[0] or 0)
        pos = int(r[1] or 0)
        return {
            "total": n,
            "positive": pos,
            "negative": n - pos,
            "positive_rate": round(pos / n, 2) if n > 0 else 0.0,
            "expectancy": round(float(r[2]), 2),
            "total_pnl_r": round(float(r[3]), 2),
        }

    def _fetch_experiments(self, date_str: str) -> list:
        rows = self._query(
            """
            SELECT experiment_name, expectancy, total_pnl_r, cf_trades, real_trades
            FROM experiment_daily_metrics
            WHERE date = %s
            ORDER BY expectancy DESC NULLS LAST
            """,
            (date_str,),
        )
        return [
            {
                "name": r[0],
                "expectancy": round(float(r[1] or 0), 2),
                "total_pnl_r": round(float(r[2] or 0), 2),
                "cf_trades": int(r[3] or 0),
                "real_trades": int(r[4] or 0),
            }
            for r in rows
        ]

    # ── Markdown ─────────────────────────────────────────────────────────────

    def render_md(self, data: Dict[str, Any]) -> str:
        m = data["market"]
        r = data["real"]
        cf = data["cf"]
        exps = data["experiments"]

        lines = ["\n---\n\n## 1. Executive Summary\n"]

        # Market
        nifty = m.get("nifty")
        bn = m.get("banknifty")
        vix = m.get("vix")

        if nifty or bn:
            day_type = m.get("day_type", "Unknown")
            n_chg = f"{nifty['change_pct']:+.2f}%" if nifty else "N/A"
            bn_chg = f"{bn['change_pct']:+.2f}%" if bn else "N/A"
            vix_val = f"{vix['value']:.1f}" if vix and vix.get("value") else "N/A"
            lines.append(
                f"**Session:** {day_type}  \n"
                f"**NIFTY50:** {n_chg}  |  **BANKNIFTY:** {bn_chg}  |  **VIX:** {vix_val}\n"
            )
        else:
            lines.append("**Market data:** Unavailable\n")

        # Real trades
        lines.append("\n### Real Trades\n")
        lines.append(
            f"| Trades | Wins | Losses | Win Rate | PnL | Expectancy |\n"
            f"|---|---|---|---|---|---|\n"
            f"| {r['trades']} | {r['wins']} | {r['losses']} | "
            f"{self._pct(r['win_rate'])} | {self._pnl_str(r['total_pnl_r'])} | "
            f"**{self._pnl_str(r['expectancy'])}** |\n"
        )
        if r["trades"] == 0:
            lines.append("*No live trades executed today.*\n")

        # Counterfactuals
        lines.append("\n### Counterfactual Research\n")
        lines.append(
            f"| Shadow Trades | Positive | Negative | Pos Rate | Expectancy | Total PnL |\n"
            f"|---|---|---|---|---|---|\n"
            f"| {cf['total']} | {cf['positive']} | {cf['negative']} | "
            f"{self._pct(cf['positive_rate'])} | **{self._pnl_str(cf['expectancy'])}** | "
            f"{self._pnl_str(cf['total_pnl_r'])} |\n"
        )

        # Experiment rankings
        if exps:
            best = exps[0]
            lines.append(
                f"\n**Best Experiment:** {best['name']} "
                f"({self._pnl_str(best['expectancy'])} expectancy)\n"
            )
            if len(exps) > 1:
                worst = exps[-1]
                lines.append(
                    f"**Worst Experiment:** {worst['name']} "
                    f"({self._pnl_str(worst['expectancy'])} expectancy)\n"
                )

        return "\n".join(lines)
