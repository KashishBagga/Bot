#!/usr/bin/env python3
"""Section 13 — Strategic Outlook (Day / Week / Month).

Answers: "based on past trends, what's realistically likely next?"

This is a trend READ, not a forecast — every other section in this report
(strategy_health.py, counterfactual_insights.py) treats rolling expectancy
the same conservative way: describe what the data shows, tier confidence by
sample size, and say "insufficient data" outright when it is, rather than
force a number the sample can't support. This section does the same thing,
just synthesized into three explicit horizons instead of one.

Data sources (reused query shapes, not reinvented):
  - System-wide expectancy trend (5d/20d/60d): same query shape as
    strategy_health.py._score_strategy(), against counterfactual_results.
  - Per-experiment trend (5d/20d/60d): same query shape as
    experiment_ranking.py, against experiment_daily_metrics (covers both
    single-leg and combo experiments via experiment_name).
  - Regime color: same query shape as
    market_state_outlook.py._trailing_performance(), against
    trade_performance (real trades only — market_regime isn't persisted on
    any counterfactual table today, a known gap, not fixed here).
"""

import logging
from typing import Any, Dict, List, Optional

from src.reports.base_section import BaseSection

logger = logging.getLogger(__name__)

# Same thresholds research_queue.py already uses — reused, not reinvented.
MIN_SAMPLES_HIGH = 100
MIN_SAMPLES_MEDIUM = 40

# Same trend-direction threshold strategy_health.py already uses.
TREND_DELTA = 0.2

REGIME_TRAILING_DAYS = 7
MONTH_MIN_TRADING_DAYS = 40  # ~2 calendar months of trading days

DISCLAIMER = (
    "This describes the trend in past data, not a prediction of future results. "
    "Treat it as a prioritized watch-list, not a forecast — and treat "
    "'insufficient data' as a real answer, not a gap to explain away."
)


def _confidence_tier(count: int) -> str:
    if count >= MIN_SAMPLES_HIGH:
        return "High"
    if count >= MIN_SAMPLES_MEDIUM:
        return "Medium"
    return "Low"


def _trend_label(recent: float, older: float) -> str:
    if recent > older + TREND_DELTA:
        return "improving"
    if recent < older - TREND_DELTA:
        return "declining"
    return "stable"


class StrategicOutlookSection(BaseSection):
    section_id = "strategic_outlook"
    section_title = "Strategic Outlook (Day / Week / Month)"

    def compute(self) -> Dict[str, Any]:
        trading_days_available = self._trading_days_available()

        system_5d, system_20d, system_60d = self._system_trend()
        experiments = self._experiment_trend()
        regime_note = self._regime_note()

        month_insufficient = trading_days_available < MONTH_MIN_TRADING_DAYS

        return {
            "trading_days_available": trading_days_available,
            "day_ahead": {
                "system_trend": {
                    "avg_5d": system_5d, "avg_20d": system_20d,
                    "label": _trend_label(system_5d, system_20d),
                },
                "hot_experiments": self._top_experiments(experiments, "5d", "20d", improving=True),
                "cold_experiments": self._top_experiments(experiments, "5d", "20d", improving=False),
                "regime_note": regime_note,
            },
            "week_ahead": {
                "system_trend": {
                    "avg_20d": system_20d, "avg_60d": system_60d,
                    "label": _trend_label(system_20d, system_60d),
                },
                "hot_experiments": self._top_experiments(experiments, "20d", "60d", improving=True),
                "cold_experiments": self._top_experiments(experiments, "20d", "60d", improving=False),
            },
            "month_ahead": {
                "insufficient_history": month_insufficient,
                "system_trend": None if month_insufficient else {
                    "avg_20d": system_20d, "avg_60d": system_60d,
                    "label": _trend_label(system_20d, system_60d),
                },
                "hot_experiments": [] if month_insufficient else self._top_experiments(experiments, "20d", "60d", improving=True),
                "cold_experiments": [] if month_insufficient else self._top_experiments(experiments, "20d", "60d", improving=False),
            },
        }

    # ── Data blocks ──────────────────────────────────────────────────────

    def _trading_days_available(self) -> int:
        rows = self._query(
            """
            SELECT COUNT(DISTINCT DATE(exit_time AT TIME ZONE 'Asia/Kolkata'))
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL AND valid = TRUE
            """,
            (),
        )
        return int(rows[0][0]) if rows and rows[0][0] is not None else 0

    def _system_trend(self) -> tuple:
        """Same query shape as strategy_health.py._score_strategy(), extended
        to a third 60-day window."""
        rows = self._query(
            """
            SELECT DATE(exit_time AT TIME ZONE 'Asia/Kolkata') as dt,
                   AVG(final_pnl_r) as exp
            FROM counterfactual_results
            WHERE exit_time IS NOT NULL
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') BETWEEN %s::date - 59 AND %s::date
              AND valid = TRUE
            GROUP BY dt
            ORDER BY dt
            """,
            (self.date_str, self.date_str),
        )
        daily_exps = [float(r[1]) for r in rows]
        last_5d = daily_exps[-5:] if len(daily_exps) >= 5 else daily_exps
        last_20d = daily_exps[-20:] if len(daily_exps) >= 20 else daily_exps
        avg_5d = sum(last_5d) / len(last_5d) if last_5d else 0.0
        avg_20d = sum(last_20d) / len(last_20d) if last_20d else 0.0
        avg_60d = sum(daily_exps) / len(daily_exps) if daily_exps else 0.0
        return round(avg_5d, 3), round(avg_20d, 3), round(avg_60d, 3)

    def _experiment_trend(self) -> List[Dict[str, Any]]:
        """Same query shape as experiment_ranking.py's rolling pull, against
        experiment_daily_metrics, at 5d/20d/60d windows."""
        windows = {"5d": 4, "20d": 19, "60d": 59}
        by_experiment: Dict[str, Dict[str, Any]] = {}

        for label, offset in windows.items():
            rows = self._query(
                """
                SELECT experiment_name,
                       AVG(expectancy),
                       SUM(cf_trades)
                FROM experiment_daily_metrics
                WHERE date BETWEEN %s::date - %s AND %s::date
                GROUP BY experiment_name
                """,
                (self.date_str, offset, self.date_str),
            )
            for name, exp, cf_trades in rows:
                entry = by_experiment.setdefault(name, {"name": name})
                entry[f"expectancy_{label}"] = round(float(exp or 0.0), 3)
                entry[f"cf_trades_{label}"] = int(cf_trades or 0)

        return list(by_experiment.values())

    def _regime_note(self) -> Dict[str, Any]:
        """Same query shape as market_state_outlook.py._trailing_performance(),
        extended to group by market_regime — real trades only, thin sample,
        directional color for "right now", not a scored trend."""
        rows = self._query(
            """
            SELECT market_regime, COUNT(*), COALESCE(SUM(final_pnl_r), 0)
            FROM trade_performance
            WHERE exit_time IS NOT NULL AND valid = TRUE
              AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata')
                  BETWEEN %s::date - %s AND %s::date
            GROUP BY market_regime
            ORDER BY COUNT(*) DESC
            """,
            (self.date_str, REGIME_TRAILING_DAYS, self.date_str),
        )
        breakdown = [
            {"regime": r[0] or "UNKNOWN", "trades": int(r[1]), "total_pnl_r": round(float(r[2]), 2)}
            for r in rows
        ]
        total_trades = sum(b["trades"] for b in breakdown)
        return {
            "trailing_days": REGIME_TRAILING_DAYS,
            "total_real_trades": total_trades,
            "breakdown": breakdown,
            "note": "Real-trade sample, thin — directional only, not confidence-scored.",
        }

    def _top_experiments(
        self, experiments: List[Dict[str, Any]], recent_key: str, older_key: str, improving: bool, limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Top N experiments by (recent_key - older_key) expectancy delta that
        clear at least Low-Medium confidence — below that, they don't appear
        here at all (shown separately as insufficient data, not ranked)."""
        candidates = []
        for e in experiments:
            recent = e.get(f"expectancy_{recent_key}")
            older = e.get(f"expectancy_{older_key}")
            cf_trades = e.get(f"cf_trades_{older_key}", 0)
            if recent is None or older is None:
                continue
            tier = _confidence_tier(cf_trades)
            if tier == "Low":
                continue
            delta = recent - older
            candidates.append({
                "name": e["name"], "expectancy_recent": recent, "expectancy_older": older,
                "delta": round(delta, 3), "confidence": tier, "cf_trades": cf_trades,
            })
        candidates.sort(key=lambda c: c["delta"], reverse=improving)
        return [c for c in candidates if (c["delta"] > 0) == improving][:limit]

    # ── Markdown ─────────────────────────────────────────────────────────

    def render_md(self, data: Dict[str, Any]) -> str:
        lines = ["\n---\n\n## 13. Strategic Outlook (Day / Week / Month)\n"]
        lines.append(f"*Based on {data['trading_days_available']} trading days of counterfactual history.*\n")

        lines.append(self._render_horizon("Day Ahead", data["day_ahead"], show_regime=True))
        lines.append(self._render_horizon("Week Ahead", data["week_ahead"]))

        month = data["month_ahead"]
        if month["insufficient_history"]:
            lines.append(
                f"\n### Month Ahead\n\n"
                f"⚠️ *Insufficient history for a month-ahead read "
                f"({data['trading_days_available']} trading days available, "
                f"need ≥{MONTH_MIN_TRADING_DAYS}). Not shown rather than guessed at.*\n"
            )
        else:
            lines.append(self._render_horizon("Month Ahead", month))

        lines.append(f"\n> 💡 *{DISCLAIMER}*\n")
        return "\n".join(lines)

    def _render_horizon(self, title: str, block: Dict[str, Any], show_regime: bool = False) -> str:
        lines = [f"\n### {title}\n"]
        st = block.get("system_trend")
        if st:
            keys = [k for k in st if k.startswith("avg_")]
            trend_str = ", ".join(f"{k.replace('avg_', '')}={self._pnl_str(st[k])}" for k in keys)
            lines.append(f"System expectancy trend: **{st['label'].upper()}** ({trend_str})\n")

        for label, key in (("🔥 Hot experiments", "hot_experiments"), ("🧊 Cold experiments", "cold_experiments")):
            items = block.get(key, [])
            lines.append(f"\n**{label}:**\n")
            if not items:
                lines.append("*None clearing confidence threshold.*\n")
                continue
            lines.append("| Experiment | Δ Expectancy | Confidence | Sample |\n|---|---|---|---|\n")
            for it in items:
                lines.append(
                    f"| {it['name']} | {self._pnl_str(it['delta'])} | {it['confidence']} | {it['cf_trades']} CF |\n"
                )

        if show_regime:
            rn = block.get("regime_note", {})
            lines.append(f"\n**Regime color (trailing {rn.get('trailing_days', REGIME_TRAILING_DAYS)}d, real trades only):**\n")
            breakdown = rn.get("breakdown", [])
            if not breakdown:
                lines.append("*No real trades in this window.*\n")
            else:
                lines.append("| Regime | Trades | Total PnL |\n|---|---|---|\n")
                for b in breakdown:
                    lines.append(f"| {b['regime']} | {b['trades']} | {self._pnl_str(b['total_pnl_r'])} |\n")
            lines.append(f"\n*{rn.get('note', '')}*\n")

        return "".join(lines)
