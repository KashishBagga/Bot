#!/usr/bin/env python3
"""Section — Outlook Accuracy.

Grades yesterday's "Tomorrow's Outlook" (src.reports.sections.tomorrow_outlook)
against what actually happened today, and persists the grade to
outlook_accuracy so a rolling hit-rate can be tracked over time
(src.models.postgres_database.save_outlook_accuracy /
get_outlook_accuracy_trailing).

Graded per symbol (nifty/banknifty), each independently — one missing
ingredient (no prior report, no playbook, no data today) skips that symbol
rather than failing the whole section:
  - gap_call:  predicted open type (gap-up/gap-down/flat) vs today's actual
               open relative to yesterday's high/low.
  - bias:      predicted close-position/trend-quality bias (BULLISH/BEARISH/
               RANGE, same thresholds TomorrowOutlookSection._build_scenarios
               uses) vs today's actual bias.
  - targets:   whether today's session reached the predicted
               resistance/support break-target zone level.
  - watch levels: how many of yesterday's watch levels fell within today's
               actual high/low.

This does not re-run any strategy logic — it only reads yesterday's already
-written report JSON and today's already-computed market_narrative session
data (shared via `rolling`), so there's no way for this section's grading to
drift from what the outlook sections actually said.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"
LOOKBACK_CALENDAR_DAYS = 6  # covers weekends/holidays to find the prior report
SYMBOLS = ("nifty", "banknifty")


class OutlookAccuracySection(BaseSection):
    section_id = "outlook_accuracy"
    section_title = "Outlook Accuracy"

    def compute(self) -> Dict[str, Any]:
        prev_json = self._load_prior_report()
        by_symbol = self.rolling.get("market_narrative", {}).get("by_symbol") or {}

        graded: Dict[str, Any] = {}
        if prev_json:
            prev_outlook = prev_json.get("sections", {}).get("tomorrow_outlook", {})
            prev_date = prev_json.get("date")
            playbooks = prev_outlook.get("playbooks", {}) or {}
            watch_levels = prev_outlook.get("watch_levels", []) or []

            for key in SYMBOLS:
                yesterday_session = prev_outlook.get(key)
                today_session = by_symbol.get(key)
                playbook = playbooks.get(key)
                if not (yesterday_session and today_session and playbook):
                    continue
                result = self._grade_symbol(
                    key, yesterday_session, today_session, playbook, watch_levels
                )
                result["generated_from_date"] = prev_date
                graded[key] = result
                self._persist(key, prev_date, result)

        trailing = self._trailing_stats()

        return {
            "generated_from_date": prev_json.get("date") if prev_json else None,
            "graded": graded,
            "trailing": trailing,
        }

    # ── Grading ──────────────────────────────────────────────────────────

    @staticmethod
    def _classify_bias(session: Dict[str, Any]) -> str:
        """Same thresholds as TomorrowOutlookSection._build_scenarios."""
        cp = session.get("close_position", 0.5)
        tq = session.get("trend_quality", 0.0)
        if cp > 0.75 and tq > 0.55:
            return "BULLISH"
        if cp < 0.25 and tq > 0.55:
            return "BEARISH"
        return "RANGE"

    @staticmethod
    def _classify_gap_label(label: str) -> str:
        low = label.lower()
        if "gap" in low and "up" in low:
            return "GAP_UP"
        if "gap" in low and "down" in low:
            return "GAP_DOWN"
        return "FLAT"

    def _grade_symbol(
        self,
        key: str,
        yesterday_session: Dict[str, Any],
        today_session: Dict[str, Any],
        playbook: Dict[str, Any],
        watch_levels: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # Gap call
        y_high, y_low = yesterday_session["high"], yesterday_session["low"]
        actual_open = today_session["open"]
        if actual_open > y_high:
            gap_actual = "GAP_UP"
        elif actual_open < y_low:
            gap_actual = "GAP_DOWN"
        else:
            gap_actual = "FLAT"
        gap_call = playbook.get("gap_call") or {}
        gap_predicted = self._classify_gap_label(gap_call.get("label", ""))
        gap_correct = gap_predicted == gap_actual

        # Bias
        bias_predicted = self._classify_bias(yesterday_session)
        bias_actual = self._classify_bias(today_session)
        bias_correct = bias_predicted == bias_actual

        # Break-and-target zone levels
        today_high, today_low = today_session["high"], today_session["low"]
        res_target = playbook.get("resistance_break_target")
        sup_target = playbook.get("support_break_target")
        resistance_hit = bool(res_target) and today_high >= res_target["level"]
        support_hit = bool(sup_target) and today_low <= sup_target["level"]

        # Watch levels touched (label carries the symbol name, e.g. "NIFTY"/"BANKNIFTY")
        sym_upper = key.upper()
        relevant_levels = [w for w in watch_levels if w.get("symbol", "").upper() == sym_upper]
        hits = sum(1 for w in relevant_levels if today_low <= w["level"] <= today_high)

        return {
            "gap_call_predicted": gap_predicted,
            "gap_call_actual": gap_actual,
            "gap_call_correct": gap_correct,
            "gap_call_label": gap_call.get("label"),
            "bias_predicted": bias_predicted,
            "bias_actual": bias_actual,
            "bias_correct": bias_correct,
            "resistance_target": res_target["level"] if res_target else None,
            "resistance_hit": resistance_hit if res_target else None,
            "support_target": sup_target["level"] if sup_target else None,
            "support_hit": support_hit if sup_target else None,
            "watch_levels_hit": hits,
            "watch_levels_total": len(relevant_levels),
        }

    def _persist(self, key: str, prev_date: Optional[str], result: Dict[str, Any]) -> None:
        try:
            self.db.save_outlook_accuracy({
                "outlook_date": self.date_str,
                "symbol": key,
                "generated_from_date": prev_date,
                "gap_call_predicted": result["gap_call_predicted"],
                "gap_call_actual": result["gap_call_actual"],
                "gap_call_correct": result["gap_call_correct"],
                "bias_predicted": result["bias_predicted"],
                "bias_actual": result["bias_actual"],
                "bias_correct": result["bias_correct"],
                "resistance_target": result["resistance_target"],
                "resistance_hit": result["resistance_hit"],
                "support_target": result["support_target"],
                "support_hit": result["support_hit"],
                "watch_levels_hit": result["watch_levels_hit"],
                "watch_levels_total": result["watch_levels_total"],
                "details": result,
            })
        except Exception as e:
            logger.warning(f"[outlook_accuracy] persist failed for {key}: {e}")

    # ── Prior report lookup ──────────────────────────────────────────────

    def _load_prior_report(self) -> Optional[Dict[str, Any]]:
        """Walk back from date_str to find the most recent existing report
        JSON (handles weekends/holidays where no report was generated)."""
        from datetime import datetime, timedelta

        dt = datetime.strptime(self.date_str, "%Y-%m-%d")
        for i in range(1, LOOKBACK_CALENDAR_DAYS + 1):
            candidate = (dt - timedelta(days=i)).strftime("%Y-%m-%d")
            path = os.path.join(REPORTS_DIR, f"{candidate}.json")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"[outlook_accuracy] failed to load {path}: {e}")
                    return None
        return None

    # ── Rolling accuracy ─────────────────────────────────────────────────

    def _trailing_stats(self, days: int = 20) -> Dict[str, Any]:
        rows = self.db.get_outlook_accuracy_trailing(self.date_str, days=days)
        if not rows:
            return {"days": 0, "by_symbol": {}}

        by_symbol: Dict[str, Any] = {}
        for key in SYMBOLS:
            sym_rows = [r for r in rows if r["symbol"] == key]
            if not sym_rows:
                continue
            gap_graded = [r for r in sym_rows if r["gap_call_correct"] is not None]
            bias_graded = [r for r in sym_rows if r["bias_correct"] is not None]
            target_rows = [r for r in sym_rows if r["resistance_hit"] is not None or r["support_hit"] is not None]
            target_hits = sum(
                1 for r in target_rows
                if r.get("resistance_hit") or r.get("support_hit")
            )
            by_symbol[key] = {
                "n": len(sym_rows),
                "gap_call_hit_rate": round(100 * sum(r["gap_call_correct"] for r in gap_graded) / len(gap_graded), 1)
                if gap_graded else None,
                "bias_hit_rate": round(100 * sum(r["bias_correct"] for r in bias_graded) / len(bias_graded), 1)
                if bias_graded else None,
                "target_hit_rate": round(100 * target_hits / len(target_rows), 1) if target_rows else None,
            }
        return {"days": days, "by_symbol": by_symbol}

    # ── Rendering ────────────────────────────────────────────────────────

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## Outlook Accuracy — Yesterday's Predictions vs Today\n"]
        lines.append(
            "> *Scores the Tomorrow's Outlook / Market State & Outlook sections "
            "written for this date against what actually happened, so the "
            "forecast isn't made and forgotten.*\n"
        )

        graded = data.get("graded") or {}
        prev_date = data.get("generated_from_date")
        if not graded:
            lines.append(f"⚠️ No prior outlook found to grade (looked back from {self.date_str}).\n")
        else:
            lines.append(f"**Graded against outlook generated on {prev_date}:**\n")
            lines.append(
                "| Symbol | Gap Call (pred → actual) | Bias (pred → actual) | Target hit | Watch levels hit |\n"
                "|---|---|---|---|---|"
            )
            for key, r in graded.items():
                gap_mark = "✅" if r["gap_call_correct"] else "❌"
                bias_mark = "✅" if r["bias_correct"] else "❌"
                target_bits = []
                if r["resistance_target"] is not None:
                    target_bits.append(f"R {r['resistance_target']}: {'✅' if r['resistance_hit'] else '❌'}")
                if r["support_target"] is not None:
                    target_bits.append(f"S {r['support_target']}: {'✅' if r['support_hit'] else '❌'}")
                target_str = "; ".join(target_bits) if target_bits else "—"
                lines.append(
                    f"| {key.upper()} | {gap_mark} {r['gap_call_predicted']} → {r['gap_call_actual']} "
                    f"| {bias_mark} {r['bias_predicted']} → {r['bias_actual']} "
                    f"| {target_str} | {r['watch_levels_hit']}/{r['watch_levels_total']} |"
                )
            lines.append("")

        trailing = data.get("trailing") or {}
        by_symbol = trailing.get("by_symbol") or {}
        if by_symbol:
            def _pct(v):
                return f"{v}%" if v is not None else "—"

            lines.append(f"**Rolling {trailing['days']}-day hit rate:**\n")
            lines.append("| Symbol | Days graded | Gap-call hit rate | Bias hit rate | Target hit rate |\n|---|---|---|---|---|")
            for key, s in by_symbol.items():
                lines.append(
                    f"| {key.upper()} | {s['n']} | {_pct(s['gap_call_hit_rate'])} "
                    f"| {_pct(s['bias_hit_rate'])} | {_pct(s['target_hit_rate'])} |"
                )

        return "\n".join(lines)
