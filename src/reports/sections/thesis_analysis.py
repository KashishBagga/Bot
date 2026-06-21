#!/usr/bin/env python3
"""Section 8 — Thesis Analysis (Woodchopper Detection)."""

import logging
from typing import Any, Dict

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

WOODCHOPPER_ATTEMPTS = 5      # Minimum re-entries on same level to flag
WOODCHOPPER_LOSS_THRESHOLD = -3.0   # Net loss (R) to classify as destructive


class ThesisAnalysisSection(BaseSection):
    section_id = "thesis_analysis"
    section_title = "Thesis Analysis"

    def compute(self) -> Dict[str, Any]:
        # Today's thesis performance grouped by (symbol, setup, direction)
        today_rows = self._query(
            """
            SELECT symbol, setup_type, signal_type,
                   COUNT(*) as attempts,
                   COALESCE(SUM(final_pnl_r), 0) as net_pnl,
                   COALESCE(AVG(final_pnl_r), 0) as avg_pnl,
                   MIN(timestamp) as first_entry,
                   MAX(exit_time) as last_exit
            FROM counterfactual_results
            WHERE DATE(timestamp AT TIME ZONE 'Asia/Kolkata') = %s
              AND exit_time IS NOT NULL
              AND valid = TRUE
            GROUP BY symbol, setup_type, signal_type
            ORDER BY attempts DESC
            """,
            (self.date_str,),
        )

        # Cross-reference with 5-day rolling woodchopper data
        woodchopper_5d = self.rolling.get("woodchopper_5d", [])
        choppers_5d = {(w["symbol"], w["setup_type"], w["signal"]) for w in woodchopper_5d}

        theses = []
        for r in today_rows:
            symbol, setup, signal = r[0], r[1], r[2]
            attempts = int(r[3])
            net_pnl = float(r[4])
            avg_pnl = float(r[5])
            recurring_5d = (symbol, setup, signal) in choppers_5d

            if attempts >= WOODCHOPPER_ATTEMPTS and net_pnl < WOODCHOPPER_LOSS_THRESHOLD:
                verdict = "woodchopper_destructive"
                verdict_text = "🔴 Destructive chop — repeated losses, consider max_attempts"
                priority = 5
            elif attempts >= WOODCHOPPER_ATTEMPTS and net_pnl > 0:
                verdict = "woodchopper_profitable"
                verdict_text = "🟡 High-frequency profitable — absorbing chop to catch runner"
                priority = 2
            elif attempts >= WOODCHOPPER_ATTEMPTS:
                verdict = "woodchopper_neutral"
                verdict_text = "🟠 Repeated entries, marginal outcome — monitor"
                priority = 3
            else:
                verdict = "normal"
                verdict_text = "✅ Normal thesis frequency"
                priority = 0

            theses.append({
                "symbol": symbol,
                "setup": setup,
                "signal": signal,
                "attempts_today": attempts,
                "net_pnl_today": round(net_pnl, 2),
                "avg_pnl_today": round(avg_pnl, 2),
                "verdict": verdict,
                "verdict_text": verdict_text,
                "recurring_5d": recurring_5d,
                "priority": priority,
            })

        theses.sort(key=lambda x: x["priority"], reverse=True)
        flagged = [t for t in theses if t["verdict"] != "normal"]
        return {"theses": theses, "flagged": flagged}

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 8. Thesis Analysis\n"]
        theses = data.get("theses", [])
        flagged = data.get("flagged", [])

        if not theses:
            lines.append("*No CF thesis data today.*\n")
            return "\n".join(lines)

        # Summary table of all theses
        lines.append(
            "| Symbol | Setup | Direction | Attempts | Net PnL | Verdict |\n"
            "|---|---|---|---|---|---|"
        )
        for t in theses:
            sym_short = t["symbol"].replace("NSE:", "").replace("-INDEX", "")
            lines.append(
                f"| {sym_short} | {t['setup']} | {t['signal']} "
                f"| {t['attempts_today']} | {self._pnl_str(t['net_pnl_today'])} "
                f"| {t['verdict_text']} |"
            )
        lines.append("")

        # Detailed breakdown of flagged theses
        if flagged:
            lines.append("\n**⚠️ Flagged Theses:**\n")
            for t in flagged:
                recurring_note = (
                    " *(also flagged in 5-day rolling data)*"
                    if t["recurring_5d"] else ""
                )
                lines += [
                    f"\n**{t['symbol']} — {t['setup']} {t['signal']}**{recurring_note}  ",
                    f"Attempts today: **{t['attempts_today']}** | Net: {self._pnl_str(t['net_pnl_today'])}  ",
                    f"Verdict: {t['verdict_text']}  ",
                    "",
                ]
            if any(t["verdict"] == "woodchopper_destructive" for t in flagged):
                lines.append(
                    "> 💡 **Research suggestion:** If this persists across 3+ days, "
                    "add `max_attempts_per_level` to `StructuralStrategy.thesis_key()`. "
                    "Do not change strategy yet — first verify in Research Queue.\n"
                )
        else:
            lines.append("*No destructive or high-frequency thesis patterns today.*\n")

        return "\n".join(lines)
