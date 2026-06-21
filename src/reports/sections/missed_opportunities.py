#!/usr/bin/env python3
"""Section 5 — Missed Opportunities.

Shows the largest MFE counterfactual trades that were rejected by the live
strategy — i.e., where the filter cost the most potential profit.
"""

import logging
from typing import Any, Dict

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

MFE_THRESHOLD = 3.0   # Minimum MFE (R) to qualify as a "missed opportunity"
TOP_N = 5


class MissedOpportunitiesSection(BaseSection):
    section_id = "missed_opportunities"
    section_title = "Missed Opportunities"

    def compute(self) -> Dict[str, Any]:
        rows = self._query(
            """
            SELECT candidate_id, symbol, signal_type, setup_type,
                   primary_rejection_reason, rejection_reasons,
                   entry_price, stop_loss, stop_loss_distance,
                   mfe_r, mae_r, final_pnl_r, capture_rate,
                   exit_reason, experiment_name
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND mfe_r >= %s
              AND valid = TRUE
            ORDER BY mfe_r DESC
            LIMIT %s
            """,
            (self.date_str, MFE_THRESHOLD, TOP_N),
        )

        opps = []
        for r in rows:
            mfe_r = float(r[9] or 0)
            final_pnl = float(r[11] or 0)
            # Priority: how large was the missed edge?
            priority = min(5, int(mfe_r / 5) + 1)

            # Compute what the blocking parameter was
            primary = r[4] or "UNKNOWN"
            blocking_hint = _blocking_param_hint(primary, r[5])

            opps.append({
                "candidate_id": r[0],
                "symbol": r[1],
                "signal": r[2],
                "setup": r[3],
                "primary_rejection": primary,
                "rejection_reasons": r[5],
                "entry_price": round(float(r[6]), 2),
                "sl": round(float(r[7]), 2),
                "sl_dist": round(float(r[8] or 0), 2),
                "mfe_r": round(mfe_r, 2),
                "mae_r": round(float(r[10] or 0), 2),
                "final_pnl_r": round(final_pnl, 2),
                "capture_rate": float(r[12]) if r[12] is not None else None,
                "exit_reason": r[13],
                "experiment": r[14],
                "priority_stars": priority,
                "blocking_hint": blocking_hint,
            })

        return {"opportunities": opps, "count": len(opps)}

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 5. Missed Opportunities\n"]
        opps = data.get("opportunities", [])

        if not opps:
            lines.append(
                f"*No rejected CF trades with MFE ≥ {MFE_THRESHOLD}R today.*  \n"
                "*Filters are not blocking significant opportunities.*\n"
            )
            return "\n".join(lines)

        lines.append(
            f"*Top {len(opps)} rejected trades where price moved {MFE_THRESHOLD}R+ in our favour.*\n"
        )

        for i, o in enumerate(opps, 1):
            cap_str = f"{o['capture_rate']*100:.0f}%" if o["capture_rate"] else "N/A"
            lines += [
                f"\n### #{i} — {o['setup']} {o['signal']} `{o['symbol']}`\n",
                f"| Field | Value |\n|---|---|",
                f"| Candidate | `{o['candidate_id'][-30:]}` |",
                f"| Experiment | {o['experiment']} |",
                f"| Rejected for | **{o['primary_rejection']}** |",
                f"| Blocking param | {o['blocking_hint']} |",
                f"| Entry | {o['entry_price']} |",
                f"| **MFE (max possible)** | **+{o['mfe_r']:.2f}R** |",
                f"| CF PnL achieved | {self._pnl_str(o['final_pnl_r'])} |",
                f"| Capture | {cap_str} |",
                f"| Exit reason | {o['exit_reason']} |",
                f"| Priority | {self._stars(o['priority_stars'])} |",
                "",
            ]

        return "\n".join(lines)


def _blocking_param_hint(primary_reason: str, all_reasons) -> str:
    """Return a human-readable description of what parameter blocked this trade."""
    hints = {
        "LOW_RVOL": "RVOL below threshold (e.g. 0.44 vs 1.0)",
        "BIAS_MISMATCH": "Signal direction opposite to HTF bias",
        "LOW_EFFICIENCY": "Move efficiency below minimum",
        "HIGH_WICKINESS": "Excessive candle wicks indicating absorption",
        "LOW_RR": "Risk:Reward ratio below 1.5",
        "NO_TARGET_ZONE": "No supply/demand zone ahead to set TP",
        "TIME_FILTER": "Signal generated outside allowed session hours",
        "TP_CAPPED": "TP was capped at 5× ATR (no realistic zone found)",
    }
    return hints.get(primary_reason, primary_reason)
