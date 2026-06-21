#!/usr/bin/env python3
"""Section 3 — Trade Review (real trades)."""

import logging
from typing import Any, Dict, List

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

RATINGS = [
    (4.0, "A+", "Elite execution"),
    (2.5, "A", "Strong trade"),
    (1.0, "B", "Good trade"),
    (0.0, "C", "Marginal winner"),
    (float("-inf"), "D", "Loser"),
]


def _rate(pnl_r: float, capture: float) -> tuple[str, str]:
    for threshold, grade, desc in RATINGS:
        if pnl_r >= threshold:
            return grade, desc
    return "D", "Loser"


class TradeReviewSection(BaseSection):
    section_id = "trade_review"
    section_title = "Trade Review"

    def compute(self) -> Dict[str, Any]:
        rows = self._query(
            """
            SELECT trade_id, experiment_name, symbol, signal_type, setup_type,
                   entry_price, exit_price, stop_loss, take_profit,
                   final_pnl_r, mfe_r, mae_r, exit_reason,
                   bars_held, duration_minutes, capture_rate, holding_efficiency,
                   entry_time, exit_time
            FROM trade_performance
            WHERE DATE(entry_time AT TIME ZONE 'Asia/Kolkata') = %s
              AND valid = TRUE
            ORDER BY entry_time
            """,
            (self.date_str,),
        )

        trades = []
        for r in rows:
            trade_id = r[0]
            pnl_r = float(r[9] or 0)
            capture = float(r[15]) if r[15] is not None else None
            grade, grade_desc = _rate(pnl_r, capture)

            # Count TP expansion events
            tp_events = self._query(
                """
                SELECT COUNT(*) FROM trade_events
                WHERE trade_id = %s AND event_type = 'TP_EXPANSION'
                """,
                (trade_id,),
            )
            tp_count = int(tp_events[0][0]) if tp_events else 0

            # Qualitative note
            exit_reason = r[12] or ""
            if pnl_r > 2.0 and tp_count > 0:
                note = f"TP expansion captured continuation ({tp_count} expansions)"
            elif pnl_r > 0 and exit_reason == "TRAILING_SL":
                note = "Trailing SL locked in profits correctly"
            elif exit_reason == "INITIAL_SL":
                note = "Stopped at initial SL — thesis failed immediately"
            elif exit_reason == "SESSION_END":
                note = "Held through close — consider whether session exit is optimal"
            else:
                note = "Standard exit"

            trades.append({
                "trade_id": trade_id,
                "experiment": r[1],
                "symbol": r[2],
                "signal": r[3],
                "setup": r[4],
                "entry": round(float(r[5]), 2),
                "exit": round(float(r[6]), 2),
                "sl": round(float(r[7]), 2),
                "tp": round(float(r[8]), 2),
                "pnl_r": round(pnl_r, 2),
                "mfe_r": round(float(r[10] or 0), 2),
                "mae_r": round(float(r[11] or 0), 2),
                "exit_reason": exit_reason,
                "bars_held": int(r[13] or 0),
                "duration_min": round(float(r[14] or 0), 1),
                "capture_rate": round(capture, 2) if capture else None,
                "holding_eff": round(float(r[16] or 0), 3),
                "tp_expansions": tp_count,
                "grade": grade,
                "grade_desc": grade_desc,
                "note": note,
            })

        return {"trades": trades, "count": len(trades)}

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 3. Trade Review\n"]
        trades = data.get("trades", [])

        if not trades:
            lines.append("*No live trades executed today.*\n")
            return "\n".join(lines)

        lines.append(f"*{len(trades)} live trade(s) executed today.*\n")

        for i, t in enumerate(trades, 1):
            cap_str = f"{t['capture_rate']*100:.0f}%" if t["capture_rate"] else "N/A"
            pnl_emoji = "🟢" if t["pnl_r"] > 0 else "🔴"
            lines += [
                f"\n### Trade #{i} — {t['setup']} {t['signal']} `{t['symbol']}`\n",
                f"| Field | Value |\n|---|---|",
                f"| Experiment | {t['experiment']} |",
                f"| Entry → Exit | {t['entry']} → {t['exit']} |",
                f"| SL / TP | {t['sl']} / {t['tp']} |",
                f"| **PnL** | **{pnl_emoji} {self._pnl_str(t['pnl_r'])}** |",
                f"| MFE / MAE | {self._pnl_str(t['mfe_r'])} / {self._pnl_str(t['mae_r'])} |",
                f"| Capture | {cap_str} |",
                f"| Hold | {t['bars_held']} bars ({t['duration_min']:.0f} min) |",
                f"| TP Expansions | {t['tp_expansions']} |",
                f"| Exit Reason | {t['exit_reason']} |",
                f"| **Grade** | **{t['grade']} — {t['grade_desc']}** |",
                f"\n> {t['note']}\n",
            ]

        return "\n".join(lines)
