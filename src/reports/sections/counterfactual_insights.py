#!/usr/bin/env python3
"""Section 6 — Counterfactual Insights (Filter Attribution).

Shows today's performance per rejection filter alongside rolling 5d and 20d stats.
Verdicts are based on rolling evidence, never single-day conclusions.
"""

import logging
from typing import Any, Dict

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

# Thresholds for verdict (based on rolling expectancy)
OVER_FILTERED_THRESHOLD = 0.50   # Rolling exp above this → investigate
GOOD_FILTER_THRESHOLD = -0.50    # Rolling exp below this → excellent filter


class CounterfactualInsightsSection(BaseSection):
    section_id = "counterfactual_insights"
    section_title = "Counterfactual Insights"

    def compute(self) -> Dict[str, Any]:
        # Today's per-filter stats
        today_rows = self._query(
            """
            SELECT primary_rejection_reason,
                   COUNT(*) as trades,
                   COALESCE(AVG(final_pnl_r), 0) as expectancy,
                   COALESCE(AVG(mfe_r), 0) as avg_mfe,
                   SUM(CASE WHEN final_pnl_r > 0 THEN 1 ELSE 0 END) as wins
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND valid = TRUE
            GROUP BY primary_rejection_reason
            ORDER BY COUNT(*) DESC
            """,
            (self.date_str,),
        )

        filter_5d = self.rolling.get("filter_5d", {})
        filter_20d = self.rolling.get("filter_20d", {})

        filters = []
        for r in today_rows:
            fname = r[0] or "UNKNOWN"
            today_exp = float(r[2])
            today_count = int(r[1])
            today_wins = int(r[4])

            r5 = filter_5d.get(fname, {})
            r20 = filter_20d.get(fname, {})

            exp_5d = r5.get("expectancy")
            exp_20d = r20.get("expectancy")
            count_5d = r5.get("count", 0)
            count_20d = r20.get("count", 0)

            # Use rolling evidence for verdict — single-day is noise
            reference_exp = exp_20d if exp_20d is not None else exp_5d
            reference_count = count_20d if count_20d else count_5d

            if reference_exp is None or reference_count < 10:
                verdict = "insufficient_data"
                verdict_text = "📊 Insufficient data — keep monitoring"
            elif reference_exp > OVER_FILTERED_THRESHOLD:
                verdict = "over_filtered"
                verdict_text = "⚠️ Over-filtered — rolling expectancy positive, worth investigating"
            elif reference_exp < GOOD_FILTER_THRESHOLD:
                verdict = "good_filter"
                verdict_text = "✅ Excellent filter — rolling expectancy negative"
            else:
                verdict = "neutral"
                verdict_text = "➡️ Neutral — keep monitoring"

            filters.append({
                "name": fname,
                "today_count": today_count,
                "today_expectancy": round(today_exp, 2),
                "today_wins": today_wins,
                "today_win_rate": round(today_wins / today_count, 2) if today_count > 0 else 0.0,
                "avg_mfe": round(float(r[3]), 2),
                "rolling_5d_count": count_5d,
                "rolling_5d_expectancy": round(exp_5d, 2) if exp_5d is not None else None,
                "rolling_20d_count": count_20d,
                "rolling_20d_expectancy": round(exp_20d, 2) if exp_20d is not None else None,
                "verdict": verdict,
                "verdict_text": verdict_text,
            })

        return {"filters": filters, "count": len(filters)}

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 6. Counterfactual Insights\n"]
        filters = data.get("filters", [])

        if not filters:
            lines.append("*No CF exits today.*\n")
            return "\n".join(lines)

        lines.append(
            "> *Verdicts are based on rolling 20-day evidence. "
            "Single-day results are shown for context only.*\n"
        )

        # Table
        lines.append(
            "| Filter | Today (n) | Today Exp | 5d Exp | 20d Exp | Verdict |\n"
            "|---|---|---|---|---|---|"
        )
        for f in filters:
            exp_5d = self._pnl_str(f["rolling_5d_expectancy"]) if f["rolling_5d_expectancy"] is not None else "—"
            exp_20d = self._pnl_str(f["rolling_20d_expectancy"]) if f["rolling_20d_expectancy"] is not None else "—"
            lines.append(
                f"| `{f['name']}` | {f['today_count']} | {self._pnl_str(f['today_expectancy'])} "
                f"| {exp_5d} | {exp_20d} | {f['verdict_text']} |"
            )

        lines.append("")

        # Highlight actionable findings
        actionable = [f for f in filters if f["verdict"] in ("over_filtered", "good_filter")]
        if actionable:
            lines.append("\n**Key findings:**")
            for f in actionable:
                count = f["rolling_20d_count"] or f["rolling_5d_count"]
                exp = f["rolling_20d_expectancy"] or f["rolling_5d_expectancy"]
                lines.append(
                    f"- **{f['name']}**: {f['verdict_text']}  \n"
                    f"  Rolling expectancy {self._pnl_str(exp)} over {count} shadow trades."
                )
        lines.append("")
        return "\n".join(lines)
