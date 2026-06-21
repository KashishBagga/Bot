#!/usr/bin/env python3
"""Section 12 — Strategy Health Score.

Answers the most important question after a bad day:
  "Should I change anything?"

Scores:
  Execution    — Did the system run without errors?
  Data Quality — Were there data fetch issues?
  Strategy     — Is the rolling expectancy trending in the right direction?
  Research     — How much evidence do we have?

Overall recommendation: No changes / Monitor / Investigate
"""

import logging
import os
from typing import Any, Dict

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

LOG_DIR = "logs"
CONFIDENCE_PER_25 = 1  # Each 25 CFs = 1 star (max 5)


class StrategyHealthSection(BaseSection):
    section_id = "strategy_health"
    section_title = "Strategy Health Score"

    def compute(self) -> Dict[str, Any]:
        execution = self._score_execution()
        data_quality = self._score_data_quality()
        strategy = self._score_strategy()
        research = self._score_research()

        # Overall recommendation
        scores = [execution["score"], data_quality["score"], strategy["score"]]
        avg = sum(scores) / len(scores)
        if avg >= 8:
            recommendation = "✅ No strategy changes. System is performing as designed."
        elif avg >= 6:
            recommendation = "🟡 Monitor. No changes today, but review Research Queue."
        else:
            recommendation = "⚠️ Investigate. One or more system components need attention."

        # Equity curve (cumulative PnL over CF history for Streamlit)
        equity_rows = self._query(
            """
            SELECT DATE(exit_time AT TIME ZONE 'Asia/Kolkata') as dt,
                   SUM(final_pnl_r) as daily_pnl
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND valid = TRUE
            GROUP BY dt ORDER BY dt
            """,
            (),
        )
        cumulative = 0.0
        equity_curve = []
        for row in equity_rows:
            cumulative += float(row[1] or 0)
            equity_curve.append({"date": str(row[0]), "cumulative_pnl": round(cumulative, 2)})

        return {
            "execution": execution,
            "data_quality": data_quality,
            "strategy": strategy,
            "research": research,
            "recommendation": recommendation,
            "equity_curve": equity_curve,
        }

    # ── Sub-scores ──────────────────────────────────────────────────────────

    def _score_execution(self) -> dict:
        log_path = os.path.join(
            LOG_DIR, f"paper_trading_{self.date_str}.log"
        )
        loop_errors = 0
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    content = f.read()
                loop_errors = content.count("Error in market loop")
            except Exception:
                pass

        score = 10 if loop_errors == 0 else max(4, 10 - loop_errors * 2)
        return {
            "score": score,
            "loop_errors": loop_errors,
            "detail": "No crashes" if loop_errors == 0 else f"{loop_errors} loop error(s) — check logs",
        }

    def _score_data_quality(self) -> dict:
        log_path = os.path.join(
            LOG_DIR, f"paper_trading_{self.date_str}.log"
        )
        fetch_errors = 0
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    content = f.read()
                fetch_errors = content.count("Could not fetch complete MTF data")
            except Exception:
                pass

        score = 10 if fetch_errors == 0 else max(5, 10 - fetch_errors)
        return {
            "score": score,
            "fetch_errors": fetch_errors,
            "detail": (
                "All MTF data fetched cleanly"
                if fetch_errors == 0
                else f"{fetch_errors} MTF fetch warning(s) — transient API hiccup"
            ),
        }

    def _score_strategy(self) -> dict:
        # 5-day and 20-day rolling expectancy trend
        rows = self._query(
            """
            SELECT DATE(exit_time AT TIME ZONE 'Asia/Kolkata') as dt,
                   AVG(final_pnl_r) as exp
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') BETWEEN %s::date - 19 AND %s::date
              AND valid = TRUE
            GROUP BY dt
            ORDER BY dt
            """,
            (self.date_str, self.date_str),
        )
        daily_exps = [float(r[1]) for r in rows]
        recent_5d = daily_exps[-5:] if len(daily_exps) >= 5 else daily_exps
        avg_20d = sum(daily_exps) / len(daily_exps) if daily_exps else 0.0
        avg_5d = sum(recent_5d) / len(recent_5d) if recent_5d else 0.0

        if avg_20d > 0.5:
            score = 9
            trend = "Strongly positive"
        elif avg_20d > 0:
            score = 7
            trend = "Positive"
        elif avg_20d > -0.5:
            score = 5
            trend = "Neutral — marginal"
        else:
            score = 3
            trend = "Negative — investigate"

        # Trending up vs down recently?
        if avg_5d > avg_20d + 0.2:
            trend += " (improving)"
        elif avg_5d < avg_20d - 0.2:
            trend += " (deteriorating)"

        return {
            "score": score,
            "avg_20d_expectancy": round(avg_20d, 2),
            "avg_5d_expectancy": round(avg_5d, 2),
            "trend": trend,
            "detail": f"20d avg: {avg_20d:+.2f}R, 5d avg: {avg_5d:+.2f}R",
        }

    def _score_research(self) -> dict:
        total_cf = self.rolling.get("total_cf_count", 0)
        cf_5d = self.rolling.get("cf_last_5d", 0)

        stars = min(5, total_cf // 25)
        if stars >= 4:
            confidence = "High"
        elif stars >= 3:
            confidence = "Medium"
        elif stars >= 2:
            confidence = "Low-Medium"
        else:
            confidence = "Low"

        needed_for_high = max(0, 100 - total_cf)
        return {
            "stars": stars,
            "confidence": confidence,
            "total_cf": total_cf,
            "cf_last_5d": cf_5d,
            "detail": (
                f"Based on {total_cf} cumulative shadow trades."
                if needed_for_high <= 0
                else f"Need {needed_for_high} more CF trades for HIGH confidence."
            ),
        }

    # ── Markdown ─────────────────────────────────────────────────────────────

    def render_md(self, data: Dict[str, Any]) -> str:
        ex = data["execution"]
        dq = data["data_quality"]
        st = data["strategy"]
        rs = data["research"]

        lines = ["\n---\n\n## 12. Strategy Health Score\n"]
        lines.append(
            f"| Dimension | Score | Detail |\n|---|---|---|\n"
            f"| Execution | **{ex['score']}/10** | {ex['detail']} |\n"
            f"| Data Quality | **{dq['score']}/10** | {dq['detail']} |\n"
            f"| Strategy Confidence | **{st['score']}/10** | {st['trend']} |\n"
            f"| Research Confidence | **{self._stars(rs['stars'])}** | {rs['detail']} |\n"
        )

        lines.append(
            f"\n**Overall Recommendation:**  \n{data['recommendation']}\n"
        )

        # Calming note for losing days
        avg = (ex["score"] + dq["score"] + st["score"]) / 3
        if avg >= 8:
            lines.append(
                "\n> 💡 *High execution score on a losing day means the system is working. "
                "Losses are part of the edge. Do not change the strategy based on one day.*\n"
            )

        return "\n".join(lines)
