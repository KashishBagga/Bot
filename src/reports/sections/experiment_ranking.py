#!/usr/bin/env python3
"""Section 7 — Experiment Ranking."""

import logging
from typing import Any, Dict

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)


class ExperimentRankingSection(BaseSection):
    section_id = "experiment_ranking"
    section_title = "Experiment Ranking"

    def compute(self) -> Dict[str, Any]:
        # Today
        today_rows = self._query(
            """
            SELECT experiment_name, expectancy, total_pnl_r, cf_trades,
                   real_trades, wins, losses, avg_mfe, avg_mae, avg_capture_rate,
                   avg_holding_eff, config_hash
            FROM experiment_daily_metrics
            WHERE date = %s
            ORDER BY expectancy DESC NULLS LAST
            """,
            (self.date_str,),
        )

        # Rolling 5-day average per experiment
        roll_rows = self._query(
            """
            SELECT experiment_name,
                   ROUND(AVG(expectancy)::numeric, 2),
                   SUM(cf_trades)
            FROM experiment_daily_metrics
            WHERE date BETWEEN %s::date - 4 AND %s::date
            GROUP BY experiment_name
            """,
            (self.date_str, self.date_str),
        )
        roll_map = {r[0]: {"expectancy_5d": float(r[1] or 0), "cf_5d": int(r[2] or 0)} for r in roll_rows}

        rankings = []
        for i, r in enumerate(today_rows, 1):
            name = r[0]
            roll = roll_map.get(name, {})
            rankings.append({
                "rank": i,
                "name": name,
                "today_expectancy": round(float(r[1] or 0), 2),
                "today_pnl_r": round(float(r[2] or 0), 2),
                "today_cf": int(r[3] or 0),
                "today_real": int(r[4] or 0),
                "wins": int(r[5] or 0),
                "losses": int(r[6] or 0),
                "avg_mfe": round(float(r[7] or 0), 2),
                "avg_mae": round(float(r[8] or 0), 2),
                "avg_capture": round(float(r[9] or 0), 2) if r[9] else None,
                "avg_hold_eff": round(float(r[10] or 0), 3),
                "config_hash": r[11],
                "expectancy_5d": roll.get("expectancy_5d"),
                "cf_5d": roll.get("cf_5d"),
            })

        return {"rankings": rankings}

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 7. Experiment Ranking\n"]
        rankings = data.get("rankings", [])

        if not rankings:
            lines.append("*No experiment metrics for today.*\n")
            return "\n".join(lines)

        for e in rankings:
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(e["rank"], f"#{e['rank']}")
            exp_5d_str = (
                f" *(5d avg: {self._pnl_str(e['expectancy_5d'])})*"
                if e.get("expectancy_5d") is not None else ""
            )
            cap_str = f"{e['avg_capture']*100:.0f}%" if e["avg_capture"] else "N/A"
            lines += [
                f"\n### {medal} {e['name']}\n",
                f"| Metric | Today | 5-Day Rolling |",
                f"|---|---|---|",
                f"| Expectancy | **{self._pnl_str(e['today_expectancy'])}** | {self._pnl_str(e.get('expectancy_5d'))} |",
                f"| Total PnL | {self._pnl_str(e['today_pnl_r'])} | — |",
                f"| CF Trades | {e['today_cf']} | {e.get('cf_5d', '—')} |",
                f"| Real Trades | {e['today_real']} | — |",
                f"| Avg MFE | {self._pnl_str(e['avg_mfe'])} | — |",
                f"| Avg MAE | {self._pnl_str(e['avg_mae'])} | — |",
                f"| Avg Capture | {cap_str} | — |",
                "",
            ]

        return "\n".join(lines)
