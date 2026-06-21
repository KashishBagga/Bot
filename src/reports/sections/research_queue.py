#!/usr/bin/env python3
"""Section 11 — Research Queue.

Replaces "Action Items" with a hypothesis-driven priority list.
Every recommendation is backed by rolling evidence.
Nothing changes immediately — everything becomes a hypothesis first.

Confidence thresholds (rolling expectancy + sample count):
  HIGH:   20d count ≥ 100 AND rolling exp clearly directional
  MEDIUM: 20d count ≥ 40
  LOW:    < 40 samples — interesting but not actionable yet
"""

import logging
from typing import Any, Dict, List

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

MIN_SAMPLES_HIGH = 100
MIN_SAMPLES_MEDIUM = 40
EXPECTANCY_THRESHOLD = 0.50


class ResearchQueueSection(BaseSection):
    section_id = "research_queue"
    section_title = "Research Queue"

    def compute(self) -> Dict[str, Any]:
        queue: List[Dict] = []

        filter_20d = self.rolling.get("filter_20d", {})
        filter_5d = self.rolling.get("filter_5d", {})
        woodchopper_5d = self.rolling.get("woodchopper_5d", [])

        # ── Hypothesis 1: Over-filtered rejections ────────────────────────
        for fname, stats in filter_20d.items():
            count = stats.get("count", 0)
            exp = stats.get("expectancy", 0)

            if exp < EXPECTANCY_THRESHOLD:
                continue  # Filter is doing its job

            if count >= MIN_SAMPLES_HIGH:
                confidence = "High"
                stars = 4
            elif count >= MIN_SAMPLES_MEDIUM:
                confidence = "Medium"
                stars = 3
            else:
                confidence = "Low"
                stars = 2

            queue.append({
                "priority": stars,
                "title": f"Investigate `{fname}` filter threshold",
                "description": (
                    f"Shadow trades rejected for `{fname}` show rolling expectancy "
                    f"{exp:+.2f}R over {count} trades. "
                    f"This suggests the filter may be too strict."
                ),
                "evidence_stars": stars,
                "confidence": confidence,
                "rolling_expectancy": round(exp, 2),
                "sample_count": count,
                "status": "Needs experiment",
                "next_step": f"Create Experiment with {fname} threshold loosened by 20%, run for 5 days.",
            })

        # ── Hypothesis 2: Woodchopper patterns ───────────────────────────
        destructive_choppers = [
            w for w in woodchopper_5d
            if w["net_pnl"] < -5 and w["attempts"] >= 8
        ]
        if destructive_choppers:
            worst = sorted(destructive_choppers, key=lambda x: x["net_pnl"])[0]
            queue.append({
                "priority": 4,
                "title": "Investigate repeated thesis entries (Woodchopper)",
                "description": (
                    f"{worst['symbol']} {worst['setup_type']} {worst['signal']}: "
                    f"{worst['attempts']} entries over 5 days, net {worst['net_pnl']:+.2f}R. "
                    "Repeated failures on same structural level."
                ),
                "evidence_stars": 3,
                "confidence": "Medium",
                "rolling_expectancy": round(worst["net_pnl"] / max(worst["attempts"], 1), 2),
                "sample_count": worst["attempts"],
                "status": "Needs experiment",
                "next_step": "Implement `max_attempts_per_level = 2` in thesis deduplication. Shadow test.",
            })

        # ── Hypothesis 3: SESSION_END exits (if significant) ─────────────
        session_end_rows = self._query(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN exit_reason = 'SESSION_END' THEN 1 ELSE 0 END) as session_ends,
                   COALESCE(AVG(CASE WHEN exit_reason = 'SESSION_END' THEN final_pnl_r END), 0) as avg_pnl
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') BETWEEN %s::date - 19 AND %s::date
              AND valid = TRUE
            """,
            (self.date_str, self.date_str),
        )
        if session_end_rows:
            row = session_end_rows[0]
            total = int(row[0] or 0)
            se_count = int(row[1] or 0)
            se_avg = float(row[2] or 0)
            if total > 0 and se_count / total > 0.25:
                queue.append({
                    "priority": 2,
                    "title": "High SESSION_END exit rate — consider wider TP targets",
                    "description": (
                        f"{se_count}/{total} trades ({se_count/total*100:.0f}%) "
                        f"exiting at session end over 20 days (avg PnL: {se_avg:+.2f}R). "
                        "Trades are being closed by time, not by structure."
                    ),
                    "evidence_stars": 2,
                    "confidence": "Medium" if se_count > 20 else "Low",
                    "rolling_expectancy": round(se_avg, 2),
                    "sample_count": se_count,
                    "status": "Monitor",
                    "next_step": "Observe for 5 more days. If SESSION_END > 30%, investigate TP zone selection.",
                })

        # Sort by priority descending
        queue.sort(key=lambda x: x["priority"], reverse=True)
        return {"queue": queue}

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 11. Research Queue\n"]
        queue = data.get("queue", [])

        lines.append(
            "> *Every item here is a hypothesis, not a decision. "
            "No strategy changes until experiments confirm the hypothesis.*\n"
        )

        if not queue:
            lines.append("*No new research items. Strategy looks healthy.*\n")
            return "\n".join(lines)

        for i, item in enumerate(queue, 1):
            stars = self._stars(item["evidence_stars"])
            lines += [
                f"\n### Priority {i} — {item['title']}\n",
                f"| Field | Value |\n|---|---|",
                f"| Evidence | {stars} ({item['confidence']} confidence) |",
                f"| Rolling Expectancy | {self._pnl_str(item['rolling_expectancy'])} |",
                f"| Sample Count | {item['sample_count']} trades |",
                f"| Status | **{item['status']}** |",
                f"\n{item['description']}\n",
                f"**Next step:** *{item['next_step']}*\n",
            ]

        return "\n".join(lines)
