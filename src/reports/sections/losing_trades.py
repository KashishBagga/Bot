#!/usr/bin/env python3
"""Section 4 — Losing Trades."""

import logging
from typing import Any, Dict

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

# Filters whose presence on a losing trade implies it was avoidable
AVOIDANCE_FILTERS = {"HIGH_WICKINESS", "LOW_EFFICIENCY", "BIAS_MISMATCH"}


class LosingTradesSection(BaseSection):
    section_id = "losing_trades"
    section_title = "Losing Trades"

    def compute(self) -> Dict[str, Any]:
        rows = self._query(
            """
            SELECT t.trade_id, t.experiment_name, t.symbol, t.signal_type,
                   t.setup_type, t.entry_price, t.exit_price,
                   t.final_pnl_r, t.mfe_r, t.exit_reason, t.bars_held,
                   c.rejection_reasons
            FROM trade_performance t
            LEFT JOIN counterfactual_results c ON c.candidate_id = t.candidate_id
            WHERE DATE(t.entry_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND t.final_pnl_r < 0
              AND t.valid = TRUE
            ORDER BY t.final_pnl_r ASC
            """,
            (self.date_str,),
        )

        losers = []
        for r in rows:
            rejection = r[11] or []
            if isinstance(rejection, str):
                import json
                rejection = json.loads(rejection)

            avoidable = bool(set(rejection) & AVOIDANCE_FILTERS)
            avoidance_reason = None
            if avoidable:
                matching = set(rejection) & AVOIDANCE_FILTERS
                avoidance_reason = ", ".join(sorted(matching))

            mfe = float(r[8] or 0)
            exit_reason = r[9] or ""
            if mfe > 0.5 and exit_reason == "TRAILING_SL":
                failure_note = "Trade had MFE but trailing SL trailed back. Possible tight trail."
            elif mfe == 0 and exit_reason == "INITIAL_SL":
                failure_note = "Never moved in our favour — thesis was immediately invalidated."
            elif exit_reason == "SESSION_END":
                failure_note = "Held overnight / to close without thesis playing out."
            else:
                failure_note = "Standard loss."

            losers.append({
                "trade_id": r[0],
                "experiment": r[1],
                "symbol": r[2],
                "signal": r[3],
                "setup": r[4],
                "entry": round(float(r[5]), 2),
                "exit": round(float(r[6]), 2),
                "pnl_r": round(float(r[7]), 2),
                "mfe_r": round(mfe, 2),
                "exit_reason": exit_reason,
                "bars_held": int(r[10] or 0),
                "avoidable": avoidable,
                "avoidance_reason": avoidance_reason,
                "failure_note": failure_note,
            })

        return {"losers": losers, "count": len(losers)}

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 4. Losing Trades\n"]
        losers = data.get("losers", [])

        if not losers:
            lines.append("*No losing real trades today.*\n")
            return "\n".join(lines)

        avoidable_count = sum(1 for l in losers if l["avoidable"])
        lines.append(
            f"*{len(losers)} loss(es) today. {avoidable_count} potentially avoidable.*\n"
        )

        for i, t in enumerate(losers, 1):
            avoid_str = (
                f"**⚠️ Potentially avoidable** — filter signal present: `{t['avoidance_reason']}`"
                if t["avoidable"]
                else "**✅ Valid loss** — not avoidable with current filters"
            )
            lines += [
                f"\n### Loss #{i} — {t['setup']} `{t['symbol']}`\n",
                f"| Field | Value |\n|---|---|",
                f"| Entry → Exit | {t['entry']} → {t['exit']} |",
                f"| **PnL** | **🔴 {self._pnl_str(t['pnl_r'])}** |",
                f"| MFE | {self._pnl_str(t['mfe_r'])} |",
                f"| Exit Reason | {t['exit_reason']} |",
                f"| Held | {t['bars_held']} bars |",
                f"\n**Avoidable?** {avoid_str}  \n**What happened:** {t['failure_note']}\n",
            ]

        return "\n".join(lines)
