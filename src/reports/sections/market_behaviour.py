#!/usr/bin/env python3
"""Section 9 — Market Behaviour Score."""

import logging
from typing import Any, Dict, List

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)


class MarketBehaviourSection(BaseSection):
    section_id = "market_behaviour"
    section_title = "Market Behaviour"

    def compute(self) -> Dict[str, Any]:
        # CF exit distribution by setup type and outcome
        rows = self._query(
            """
            SELECT setup_type, exit_reason,
                   COUNT(*) as n,
                   COALESCE(AVG(final_pnl_r), 0) as avg_pnl,
                   COALESCE(AVG(mfe_r), 0) as avg_mfe
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND valid = TRUE
            GROUP BY setup_type, exit_reason
            """,
            (self.date_str,),
        )

        # Hourly breakdown (by exit hour)
        hourly_rows = self._query(
            """
            SELECT EXTRACT(HOUR FROM exit_time AT TIME ZONE 'Asia/Kolkata')::int as hr,
                   COUNT(*) as trades,
                   COALESCE(AVG(final_pnl_r), 0) as avg_pnl,
                   SUM(CASE WHEN final_pnl_r > 0 THEN 1 ELSE 0 END) as wins
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND valid = TRUE
            GROUP BY hr
            ORDER BY hr
            """,
            (self.date_str,),
        )

        hourly = [
            {
                "hour": int(r[0]),
                "trades": int(r[1]),
                "avg_pnl": round(float(r[2]), 2),
                "wins": int(r[3]),
                "win_rate": round(int(r[3]) / int(r[1]), 2) if int(r[1]) > 0 else 0.0,
            }
            for r in hourly_rows
        ]

        # Aggregate metrics per setup type
        setup_stats: Dict[str, Dict] = {}
        total = 0
        initial_sl_count = 0
        for r in rows:
            setup, reason, n, avg_pnl, avg_mfe = r[0], r[1], int(r[2]), float(r[3]), float(r[4])
            total += n
            if reason == "INITIAL_SL":
                initial_sl_count += n
            if setup not in setup_stats:
                setup_stats[setup] = {"total": 0, "wins": 0, "total_pnl": 0.0}
            setup_stats[setup]["total"] += n
            if avg_pnl > 0:
                setup_stats[setup]["wins"] += n
            setup_stats[setup]["total_pnl"] += avg_pnl * n

        choppiness = round(initial_sl_count / total, 2) if total > 0 else 0.0

        setup_summary = {}
        for setup, stats in setup_stats.items():
            t = stats["total"]
            setup_summary[setup] = {
                "total": t,
                "success_rate": round(stats["wins"] / t, 2) if t > 0 else 0.0,
                "avg_pnl": round(stats["total_pnl"] / t, 2) if t > 0 else 0.0,
            }

        # Score out of 10
        breakout_rate = setup_summary.get("BREAKOUT", {}).get("success_rate", 0)
        sweep_rate = setup_summary.get("SWEEP", {}).get("success_rate", 0)
        trend_score = max(0, min(10, int(breakout_rate * 12)))
        mean_rev_score = 10 - trend_score

        return {
            "total_cf": total,
            "choppiness": choppiness,
            "initial_sl_rate": choppiness,
            "setup_summary": setup_summary,
            "trend_score": trend_score,
            "mean_reversion_score": mean_rev_score,
            "breakout_success_pct": breakout_rate,
            "sweep_success_pct": sweep_rate,
            "hourly_performance": hourly,
        }

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 9. Market Behaviour\n"]

        if data.get("total_cf", 0) == 0:
            lines.append("*No CF data available.*\n")
            return "\n".join(lines)

        chop_pct = f"{data['choppiness']*100:.0f}%"
        lines.append(
            f"| Metric | Score |\n|---|---|\n"
            f"| Trend Quality | {data['trend_score']}/10 |\n"
            f"| Mean Reversion | {data['mean_reversion_score']}/10 |\n"
            f"| Breakout Success | {data['breakout_success_pct']*100:.0f}% |\n"
            f"| Sweep Success | {data['sweep_success_pct']*100:.0f}% |\n"
            f"| Choppiness (INITIAL_SL rate) | {chop_pct} |\n"
        )

        # Setup breakdown
        if data.get("setup_summary"):
            lines.append("\n**Per Setup Type:**\n")
            lines.append("| Setup | Trades | Success | Avg PnL |\n|---|---|---|---|")
            for setup, stats in sorted(data["setup_summary"].items()):
                lines.append(
                    f"| {setup} | {stats['total']} | {stats['success_rate']*100:.0f}% | "
                    f"{self._pnl_str(stats['avg_pnl'])} |"
                )
            lines.append("")

        # Hourly heatmap (text)
        hp = data.get("hourly_performance", [])
        if hp:
            lines.append("\n**Hourly Performance:**\n")
            lines.append("| Hour | Trades | Avg PnL | Win Rate |\n|---|---|---|---|")
            for h in hp:
                lines.append(
                    f"| {h['hour']:02d}:xx | {h['trades']} | "
                    f"{self._pnl_str(h['avg_pnl'])} | {h['win_rate']*100:.0f}% |"
                )
            lines.append("")

        return "\n".join(lines)
