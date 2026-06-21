#!/usr/bin/env python3
"""
CLI entrypoint for EOD Trading Journal generation.

Usage:
    python generate_report.py                          # today
    python generate_report.py --date 2026-06-19       # specific date
    python generate_report.py --backfill 2026-06-16 2026-06-20  # range
"""

import argparse
import logging
import sys
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def daterange(start: date, end: date):
    """Yield each date from start to end inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate EOD Trading Journal report(s).",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to generate report for (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--backfill",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Generate reports for a date range: --backfill 2026-06-16 2026-06-20",
    )
    args = parser.parse_args()

    # Lazy import so CLI startup is fast even if API is slow to initialise
    from src.models.postgres_database import PostgresDatabase
    from src.adapters.data.fyers_data_provider import FyersDataProvider
    from src.reports.eod_report_generator import EODReportGenerator

    db = PostgresDatabase()
    try:
        data_provider = FyersDataProvider()
    except Exception as e:
        logger.warning(f"⚠️ Fyers data provider failed to initialise: {e}")
        logger.warning("   Market data sections will show N/A — report will still generate.")
        data_provider = None

    gen = EODReportGenerator(db, data_provider)

    if args.backfill:
        try:
            start = date.fromisoformat(args.backfill[0])
            end = date.fromisoformat(args.backfill[1])
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            sys.exit(1)
        dates = list(daterange(start, end))
        logger.info(f"📅 Backfilling {len(dates)} reports: {start} → {end}")
        for d in dates:
            date_str = d.strftime("%Y-%m-%d")
            try:
                md_path, json_path = gen.generate(date_str)
                print(f"✓ {date_str}: {md_path}")
            except Exception as e:
                logger.error(f"✗ {date_str}: {e}")
    else:
        date_str = args.date or date.today().strftime("%Y-%m-%d")
        try:
            md_path, json_path = gen.generate(date_str)
            print(f"\n✅ Report ready: {md_path}")
            print(f"   Analytics:    {json_path}")
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
