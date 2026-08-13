#!/usr/bin/env python3
"""
Option Warehouse Background Service (Priority 6)
==============================================
Captures raw option chain snapshots (ATM ±5 strikes) every ~60 seconds.

Uses the Fyers `depth` endpoint to record REAL open interest (oi, pdoi, oi_change),
not the quotes endpoint which only returns LTP/volume with no OI.

Rate-limit budget: 1 depth call per second  ×  ~22 symbols per underlying
→ ~22s per underlying.  Two underlyings ⇒ ~47s total per cycle.
The loop sleeps for the remainder of the 60s interval.
"""

import time
import logging
import asyncio
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional

# Path Injection
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.adapters.data.fyers_data_provider import FyersDataProvider
from src.models.postgres_database import PostgresDatabase

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OptionWarehouse")


class OptionWarehouse:
    """Service that captures real option chain snapshots including open interest."""

    # How many strikes above/below ATM to capture
    DEPTH = 5

    def __init__(self, symbols: List[str]):
        self.underlyings = symbols
        self.data_provider = FyersDataProvider()
        self.db = PostgresDatabase()
        self.tz = ZoneInfo("Asia/Kolkata")
        self.interval = 60          # target cycle interval in seconds
        self.depth_sleep = 1.2      # seconds between depth calls (rate-limit buffer)

        self.stats = {
            "expected": 0,
            "received": 0,
            "zeros": 0,
            "errors": 0,
            "latency_ms": [],
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_option_symbols(
        self, underlying: str, ltp: float, expiry_str: str
    ) -> List[Dict]:
        """Return a list of metadata dicts for ATM ±DEPTH strikes (CE + PE)."""
        base = "BANKNIFTY" if "BANK" in underlying else "NIFTY"
        interval = 100 if "BANK" in underlying else 50
        atm = round(ltp / interval) * interval
        strikes = [int(atm + i * interval) for i in range(-self.DEPTH, self.DEPTH + 1)]

        result = []
        for strike in strikes:
            for opt_type in ("CE", "PE"):
                result.append({
                    "fyers_symbol": f"NSE:{base}{expiry_str}{strike}{opt_type}",
                    "strike": float(strike),
                    "option_type": opt_type,
                })
        return result

    async def _fetch_chain_with_real_oi(
        self, underlying: str, ltp: float, expiry_str: str, expiry_date: str
    ) -> List[Dict]:
        """Fetch depth for each option symbol, returning snapshot rows with real OI."""
        from datetime import timezone
        symbols_meta = self._build_option_symbols(underlying, ltp, expiry_str)
        snapshots = []
        now_dt = datetime.now(timezone.utc)
        client = self.data_provider.client

        for meta in symbols_meta:
            sym = meta["fyers_symbol"]
            self.stats["expected"] += 1
            t0 = time.time()

            try:
                depth = client.get_market_depth(sym)
                latency = (time.time() - t0) * 1000
                self.stats["latency_ms"].append(latency)

                if depth is None:
                    logger.warning(f"⚠️  No depth for {sym}")
                    self.stats["errors"] += 1
                    await asyncio.sleep(self.depth_sleep)
                    continue

                self.stats["received"] += 1

                if depth.get("ltp", 0) == 0:
                    self.stats["zeros"] += 1
                    logger.debug(f"⚠️  Zero LTP for {sym}")

                snapshots.append({
                    "time": now_dt,
                    "underlying": underlying,
                    "strike": meta["strike"],
                    "expiry": expiry_date,
                    "option_type": meta["option_type"],
                    "ltp": depth.get("ltp", 0.0),
                    "bid": depth.get("bid", 0.0),
                    "ask": depth.get("ask", 0.0),
                    "volume": depth.get("volume", 0),
                    "oi": depth.get("oi", 0),
                    "oi_change": depth.get("oi_change", 0),
                })

            except Exception as e:
                logger.error(f"❌ Error fetching depth for {sym}: {e}")
                self.stats["errors"] += 1

            # Stay inside rate-limit budget
            await asyncio.sleep(self.depth_sleep)

        return snapshots

    # ──────────────────────────────────────────────────────────────────────────
    # Main Loop
    # ──────────────────────────────────────────────────────────────────────────

    async def run(self):
        """Main loop — captures option chain snapshots during market hours."""
        logger.info(
            f"🚀 Option Warehouse (real OI) started for: {self.underlyings}  "
            f"(interval={self.interval}s, depth={self.DEPTH} strikes)"
        )

        while True:
            cycle_start = time.time()
            try:
                now = datetime.now(self.tz)
                is_market_hours = (
                    now.weekday() < 5
                    and (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
                    and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
                )

                if is_market_hours:
                    for underlying in self.underlyings:
                        try:
                            ltp = self.data_provider.get_current_price(underlying)
                            if not ltp:
                                logger.warning(f"⚠️  Cannot get LTP for {underlying}")
                                continue

                            resolved = self.data_provider._find_active_expiry(underlying, ltp)
                            if not resolved:
                                logger.warning(f"⚠️  Cannot resolve expiry for {underlying}")
                                continue

                            expiry_str, expiry_date = resolved
                            logger.info(
                                f"📡 Fetching {underlying} @ {ltp:.0f}  expiry={expiry_str}  "
                                f"strikes=ATM±{self.DEPTH}"
                            )

                            snapshots = await self._fetch_chain_with_real_oi(
                                underlying, ltp, expiry_str, expiry_date
                            )

                            if snapshots:
                                try:
                                    self.db.save_option_snapshots(snapshots)
                                    logger.info(
                                        f"📊 Saved {len(snapshots)} snapshots for {underlying}  "
                                        f"(oi range: {min(s['oi'] for s in snapshots):,}–"
                                        f"{max(s['oi'] for s in snapshots):,})"
                                    )
                                except Exception as db_err:
                                    # A whole cycle's snapshots failing to persist starves
                                    # every downstream OI-dependent strategy (OI_Scalping's
                                    # 10-min lookback especially) — track it in stats so it
                                    # shows up in get_stats(), not just a per-cycle error log.
                                    self.stats["errors"] += 1
                                    logger.error(f"❌ DB insert error for {underlying}: {db_err}")

                        except Exception as e:
                            logger.error(f"❌ Error processing {underlying}: {e}")

                # Sleep for the remainder of the interval
                elapsed = time.time() - cycle_start
                sleep_for = max(0.0, self.interval - elapsed)
                logger.debug(f"Cycle took {elapsed:.1f}s — sleeping {sleep_for:.1f}s")
                await asyncio.sleep(sleep_for)

            except Exception as e:
                logger.error(f"❌ Outer loop error: {e}")
                await asyncio.sleep(10)

    # ──────────────────────────────────────────────────────────────────────────
    # Health
    # ──────────────────────────────────────────────────────────────────────────

    def get_health_report(self) -> Dict[str, Any]:
        expected = self.stats["expected"]
        received = self.stats["received"]
        missing_pct = ((expected - received) / expected * 100) if expected else 0.0
        avg_latency = (
            sum(self.stats["latency_ms"]) / len(self.stats["latency_ms"])
            if self.stats["latency_ms"]
            else 0.0
        )
        return {
            "expected": expected,
            "received": received,
            "zeros": self.stats["zeros"],
            "errors": self.stats["errors"],
            "missing_pct": round(missing_pct, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "health_status": (
                "HEALTHY" if missing_pct < 5 and avg_latency < 2000 else "DEGRADED"
            ),
        }


if __name__ == "__main__":
    symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
    warehouse = OptionWarehouse(symbols)
    asyncio.run(warehouse.run())
