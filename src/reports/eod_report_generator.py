#!/usr/bin/env python3
"""
EODReportGenerator — orchestrates all 12 report sections.
==========================================================

Usage:
    gen = EODReportGenerator(db, data_provider)
    md_path, json_path = gen.generate("2026-06-19")

Completely standalone. Called by:
  - generate_report.py (CLI)
  - indian_trader.py (automatic 15:35 trigger)
  - Future: backtests, Streamlit refresh
"""

import json
import logging
import os
from datetime import datetime, timezone

from src.models.postgres_database import PostgresDatabase
from src.adapters.data.fyers_data_provider import FyersDataProvider
from src.reports.renderers.markdown_renderer import MarkdownRenderer
from src.reports.renderers.json_renderer import JsonRenderer
from src.reports.sections.executive_summary import ExecutiveSummarySection
from src.reports.sections.market_narrative import MarketNarrativeSection
from src.reports.sections.trade_review import TradeReviewSection
from src.reports.sections.losing_trades import LosingTradesSection
from src.reports.sections.missed_opportunities import MissedOpportunitiesSection
from src.reports.sections.counterfactual_insights import CounterfactualInsightsSection
from src.reports.sections.experiment_ranking import ExperimentRankingSection
from src.reports.sections.thesis_analysis import ThesisAnalysisSection
from src.reports.sections.market_behaviour import MarketBehaviourSection
from src.reports.sections.tomorrow_outlook import TomorrowOutlookSection
from src.reports.sections.research_queue import ResearchQueueSection
from src.reports.sections.strategy_health import StrategyHealthSection

logger = logging.getLogger(__name__)

REPORTS_DIR = "reports"


class EODReportGenerator:
    """Orchestrates all report sections and writes output files."""

    def __init__(self, db: PostgresDatabase, data_provider: FyersDataProvider):
        self.db = db
        self.data_provider = data_provider
        self.md_renderer = MarkdownRenderer()
        self.json_renderer = JsonRenderer()

    def generate(self, date_str: str) -> tuple[str, str]:
        """
        Generate the EOD report for a given date.

        Returns:
            (md_path, json_path) — absolute paths of the written files.
        """
        logger.info(f"📝 Generating EOD report for {date_str}…")
        os.makedirs(REPORTS_DIR, exist_ok=True)

        generated_at = datetime.now(tz=timezone.utc).astimezone(
            __import__("pytz").timezone("Asia/Kolkata")
        ).strftime("%Y-%m-%d %H:%M:%S")

        # 1. Pre-fetch rolling stats once (shared across all sections)
        rolling = self._fetch_rolling_data(date_str)
        logger.info(
            f"   Rolling data: 5d filters={len(rolling['filter_5d'])}, "
            f"20d filters={len(rolling['filter_20d'])}, "
            f"total CF={rolling['total_cf_count']}"
        )

        # 2. Define ordered section list
        section_classes = [
            ExecutiveSummarySection,
            MarketNarrativeSection,
            TradeReviewSection,
            LosingTradesSection,
            MissedOpportunitiesSection,
            CounterfactualInsightsSection,
            ExperimentRankingSection,
            ThesisAnalysisSection,
            MarketBehaviourSection,
            TomorrowOutlookSection,
            ResearchQueueSection,
            StrategyHealthSection,
        ]

        # 3. Compute + render each section
        section_mds = []
        section_jsons = {}

        for SectionClass in section_classes:
            section = SectionClass(
                db=self.db,
                data_provider=self.data_provider,
                date_str=date_str,
                rolling=rolling,
            )
            data = section.safe_compute()
            md = section.safe_render_md(data)
            js = section.render_json(data)
            section_mds.append(md)
            section_jsons[section.section_id] = js
            logger.info(f"   ✓ {section.section_id}")

        # 4. Assemble and write Markdown
        full_md = self.md_renderer.assemble(date_str, section_mds, generated_at)
        md_path = os.path.join(REPORTS_DIR, f"{date_str}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_md)

        # 5. Assemble and write JSON
        json_payload = self.json_renderer.assemble(date_str, section_jsons, generated_at)
        json_path = os.path.join(REPORTS_DIR, f"{date_str}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.json_renderer.to_string(json_payload))

        logger.info(f"📄 Report written: {md_path}")
        logger.info(f"📊 Analytics written: {json_path}")
        return md_path, json_path

    # ── Rolling data pre-fetch ──────────────────────────────────────────────

    def _fetch_rolling_data(self, date_str: str) -> dict:
        """Pre-fetch 5d/20d stats once, shared by all sections."""
        with self.db._get_connection() as conn:
            with conn.cursor() as cur:
                # 5-day filter attribution
                cur.execute(
                    """
                    SELECT primary_rejection_reason,
                           COUNT(*), COALESCE(AVG(final_pnl_r), 0)
                    FROM counterfactual_results
                    WHERE exit_time IS NOT NULL
                      AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata')
                          BETWEEN %s::date - 4 AND %s::date
                      AND valid = TRUE
                    GROUP BY primary_rejection_reason
                    """,
                    (date_str, date_str),
                )
                filter_5d = {
                    r[0]: {"count": int(r[1]), "expectancy": float(r[2])}
                    for r in cur.fetchall()
                    if r[0]
                }

                # 20-day filter attribution
                cur.execute(
                    """
                    SELECT primary_rejection_reason,
                           COUNT(*), COALESCE(AVG(final_pnl_r), 0)
                    FROM counterfactual_results
                    WHERE exit_time IS NOT NULL
                      AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata')
                          BETWEEN %s::date - 19 AND %s::date
                      AND valid = TRUE
                    GROUP BY primary_rejection_reason
                    """,
                    (date_str, date_str),
                )
                filter_20d = {
                    r[0]: {"count": int(r[1]), "expectancy": float(r[2])}
                    for r in cur.fetchall()
                    if r[0]
                }

                # Cumulative CF count (all time, up to date_str)
                cur.execute(
                    """
                    SELECT COUNT(*) FROM counterfactual_results
                    WHERE exit_time IS NOT NULL
                      AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') <= %s::date
                      AND valid = TRUE
                    """,
                    (date_str,),
                )
                total_cf = int(cur.fetchone()[0] or 0)

                # CF last 5 days
                cur.execute(
                    """
                    SELECT COUNT(*) FROM counterfactual_results
                    WHERE exit_time IS NOT NULL
                      AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata')
                          BETWEEN %s::date - 4 AND %s::date
                      AND valid = TRUE
                    """,
                    (date_str, date_str),
                )
                cf_last_5d = int(cur.fetchone()[0] or 0)

                # 5-day woodchopper patterns (high-frequency, destructive)
                cur.execute(
                    """
                    SELECT symbol, setup_type, signal_type,
                           COUNT(*) as attempts,
                           COALESCE(SUM(final_pnl_r), 0) as net_pnl
                    FROM counterfactual_results
                    WHERE exit_time IS NOT NULL
                      AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata')
                          BETWEEN %s::date - 4 AND %s::date
                      AND valid = TRUE
                    GROUP BY symbol, setup_type, signal_type
                    HAVING COUNT(*) >= 5
                    ORDER BY COUNT(*) DESC
                    """,
                    (date_str, date_str),
                )
                woodchopper_5d = [
                    {
                        "symbol": r[0],
                        "setup_type": r[1],
                        "signal": r[2],
                        "attempts": int(r[3]),
                        "net_pnl": float(r[4]),
                    }
                    for r in cur.fetchall()
                ]

        return {
            "filter_5d": filter_5d,
            "filter_20d": filter_20d,
            "total_cf_count": total_cf,
            "cf_last_5d": cf_last_5d,
            "woodchopper_5d": woodchopper_5d,
        }
