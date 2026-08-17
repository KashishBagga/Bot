#!/usr/bin/env python3
"""
Pre-Market Data Collector (v1.0)
=================================
Background service that collects pre-open indicative prices and first-5-minute
opening data for NIFTY and BANKNIFTY each trading day.

Schedule:
  09:00–09:14 IST  — Poll Fyers indicative prices every 2 minutes, updating
                     in-memory PreMarketData continuously.
  09:15 IST        — Freeze final snapshot to DB (preopen_price = last polled value).
  09:20 IST        — Collect first-5-minute bar data (OpeningData) and upsert.

Failure handling:
  API failures within the window retry on next 2-min tick.
  If the entire 09:00–09:14 window fails → is_available=False → GapStrategy
  falls back to standard (today's open vs prev_close) gap detection.

NEVER blocks trading. All DB writes are best-effort with exception handling.
"""

import logging
import time
import threading
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


class PreMarketCollector:
    """
    Daemon thread that wakes on market days to collect pre-market and opening data.
    Designed to run alongside OptionWarehouse in a background thread pool.
    """

    # How often to poll during the pre-open window (seconds)
    POLL_INTERVAL_SECONDS = 120   # every 2 minutes

    def __init__(self, symbols: List[str]):
        """
        symbols: underlying symbols to track (e.g. ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX'])
        """
        self.symbols = symbols
        self._stop_event = threading.Event()
        self._latest_premarket = {}  # symbol → dict with latest pre-open data

        # Lazy-import to avoid circular imports at module load
        self._data_provider = None
        self._db = None

    def _get_provider(self):
        if self._data_provider is None:
            from src.adapters.data.fyers_data_provider import FyersDataProvider
            self._data_provider = FyersDataProvider()
        return self._data_provider

    def _get_db(self):
        if self._db is None:
            from src.models.postgres_database import PostgresDatabase
            self._db = PostgresDatabase()
        return self._db

    # ── Core data fetch ────────────────────────────────────────────────────────

    def _fetch_indicative_price(self, symbol: str) -> Optional[float]:
        """Fetch pre-open indicative price from Fyers quotes endpoint."""
        try:
            provider = self._get_provider()
            price = provider.get_current_price(symbol)
            return price if price else None
        except Exception as e:
            logger.warning(f"[PreMarketCollector] indicative price fetch failed for {symbol}: {e}")
            return None

    def _fetch_prev_day_ohlc(self, symbol: str, today: date):
        """Fetch previous trading day OHLCV."""
        try:
            provider = self._get_provider()
            start = datetime.combine(today - timedelta(days=5), datetime.min.time(), tzinfo=IST)
            end = datetime.combine(today, datetime.min.time(), tzinfo=IST)
            df = provider.get_historical_data(symbol, start, end, "D")
            if df is None or len(df) < 2:
                return None
            # Last row might be today (partial) — use second-to-last
            row = df.iloc[-2] if df.index[-1].date() == today else df.iloc[-1]
            return {
                "pdh":        float(row["high"]),
                "pdl":        float(row["low"]),
                "prev_close": float(row["close"]),
            }
        except Exception as e:
            logger.warning(f"[PreMarketCollector] prev-day OHLC fetch failed for {symbol}: {e}")
            return None

    def _fetch_first_5m_bar(self, symbol: str, today: date):
        """Fetch first 5-minute bar after 9:15 IST."""
        try:
            provider = self._get_provider()
            start = datetime.combine(today - timedelta(days=2), datetime.min.time(), tzinfo=IST)
            end = datetime.combine(today, datetime.min.time(), tzinfo=IST)
            df = provider.get_historical_data(symbol, start, end, "5")
            if df is None or len(df) < 1:
                return None
            today_bars = df[df.index.date == today]
            if today_bars.empty:
                return None
            first_bar = today_bars.iloc[0]
            return {
                "open":   float(first_bar["open"]),
                "high":   float(first_bar["high"]),
                "low":    float(first_bar["low"]),
                "close":  float(first_bar["close"]),
                "volume": float(first_bar.get("volume", 0)),
            }
        except Exception as e:
            logger.warning(f"[PreMarketCollector] first-5m bar fetch failed for {symbol}: {e}")
            return None

    # ── Gap classification ────────────────────────────────────────────────────

    @staticmethod
    def _classify_gap(gap_pct: float):
        direction = "UP" if gap_pct > 0.15 else ("DOWN" if gap_pct < -0.15 else "FLAT")
        abs_gap = abs(gap_pct)
        magnitude = "LARGE" if abs_gap > 1.0 else ("MEDIUM" if abs_gap > 0.5 else "SMALL")
        return direction, magnitude

    # ── Pre-open polling loop (09:00–09:14) ───────────────────────────────────

    def _run_preopen_window(self, today: date):
        """Poll indicative prices from 09:00 to 09:14 IST every 2 minutes."""
        window_start = datetime(today.year, today.month, today.day, 9, 0, 0, tzinfo=IST)
        window_end   = datetime(today.year, today.month, today.day, 9, 14, 0, tzinfo=IST)
        freeze_time  = datetime(today.year, today.month, today.day, 9, 15, 0, tzinfo=IST)

        # Pre-fetch previous day data (stable, fetch once)
        prev_ohlc = {}
        for sym in self.symbols:
            ohlc = self._fetch_prev_day_ohlc(sym, today)
            if ohlc:
                prev_ohlc[sym] = ohlc
                logger.info(f"[PreMarketCollector] {sym} prev OHLC: {ohlc}")
            else:
                logger.warning(f"[PreMarketCollector] {sym} prev OHLC unavailable — will skip premarket for this symbol")

        if not prev_ohlc:
            logger.warning("[PreMarketCollector] No prev-day data for any symbol — aborting pre-open window")
            return

        success_count = {sym: 0 for sym in self.symbols}
        last_price    = {sym: None for sym in self.symbols}

        now_ist = datetime.now(IST)

        # Sleep until 09:00 if early
        if now_ist < window_start:
            sleep_secs = (window_start - now_ist).total_seconds()
            logger.info(f"[PreMarketCollector] Sleeping {sleep_secs:.0f}s until 09:00 IST")
            time.sleep(max(sleep_secs, 0))

        # Poll loop: every POLL_INTERVAL_SECONDS until 09:14
        while not self._stop_event.is_set():
            now_ist = datetime.now(IST)
            if now_ist >= window_end:
                break

            for sym in self.symbols:
                if sym not in prev_ohlc:
                    continue
                price = self._fetch_indicative_price(sym)
                if price is not None and price > 0:
                    last_price[sym] = price
                    success_count[sym] += 1

            time.sleep(self.POLL_INTERVAL_SECONDS)

        # ── 09:15 IST: freeze final snapshot ─────────────────────────────────
        now_ist = datetime.now(IST)
        if now_ist < freeze_time:
            sleep_secs = (freeze_time - now_ist).total_seconds()
            time.sleep(max(sleep_secs, 0))

        db = self._get_db()
        for sym in self.symbols:
            if sym not in prev_ohlc or last_price.get(sym) is None:
                logger.warning(f"[PreMarketCollector] {sym} — no indicative price captured; is_available=False")
                continue

            ohlc = prev_ohlc[sym]
            preopen_price = last_price[sym]
            gap_pct = (preopen_price - ohlc["prev_close"]) / ohlc["prev_close"] * 100.0
            direction, magnitude = self._classify_gap(gap_pct)

            data = {
                "gap_pct":        round(gap_pct, 4),
                "gap_direction":  direction,
                "gap_magnitude":  magnitude,
                "pdh":            ohlc["pdh"],
                "pdl":            ohlc["pdl"],
                "prev_close":     ohlc["prev_close"],
                "preopen_price":  preopen_price,
                "captured_at":    datetime.now(IST).isoformat(),
                # Opening fields left None — filled at 09:20
                "first_5m_direction": None,
                "first_5m_rvol":      None,
                "opening_vol_ratio":  None,
            }
            db.save_premarket_snapshot(sym, today, data)
            logger.info(
                f"[PreMarketCollector] ✅ {sym} premarket frozen at 09:15: "
                f"gap={gap_pct:+.2f}% ({direction}/{magnitude})"
            )

    # ── First-5-minute opening data (09:20) ───────────────────────────────────

    def _run_opening_collection(self, today: date):
        """Collect first 5-min bar data and upsert to DB at 09:20 IST."""
        target_time = datetime(today.year, today.month, today.day, 9, 20, 0, tzinfo=IST)
        now_ist = datetime.now(IST)
        if now_ist < target_time:
            sleep_secs = (target_time - now_ist).total_seconds()
            logger.info(f"[PreMarketCollector] Sleeping {sleep_secs:.0f}s until 09:20 IST (opening data)")
            time.sleep(max(sleep_secs, 0))

        db = self._get_db()
        for sym in self.symbols:
            bar = self._fetch_first_5m_bar(sym, today)
            if bar is None:
                logger.warning(f"[PreMarketCollector] {sym} first-5m bar unavailable at 09:20")
                continue

            # Direction of first bar
            first_direction = (
                "UP"   if bar["close"] > bar["open"] * 1.001 else
                "DOWN" if bar["close"] < bar["open"] * 0.999 else "FLAT"
            )

            # Rough RVOL: compare bar volume to stored data if available
            # (simplified — full RVOL needs 5-day history of first bars)
            first_5m_rvol = 1.0  # placeholder; proper RVOL added in v2

            data = {
                "first_5m_direction": first_direction,
                "first_5m_rvol":      first_5m_rvol,
                "opening_vol_ratio":  first_5m_rvol,
            }
            db.save_premarket_snapshot(sym, today, data)
            logger.info(
                f"[PreMarketCollector] ✅ {sym} opening data: "
                f"direction={first_direction} volume={bar['volume']:.0f}"
            )

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self):
        """
        Main loop — runs continuously as a daemon thread.
        Each trading day: pre-open window → 09:15 freeze → 09:20 opening data.
        Sleeps the rest of the day.
        """
        logger.info("[PreMarketCollector] Started")

        while not self._stop_event.is_set():
            now_ist = datetime.now(IST)
            today   = now_ist.date()

            # Only run on weekdays
            if now_ist.weekday() < 5:   # Mon=0 … Fri=4
                try:
                    self._run_preopen_window(today)
                    self._run_opening_collection(today)
                except Exception as e:
                    logger.error(f"[PreMarketCollector] Unhandled error: {e}", exc_info=True)

            # Sleep until next 08:58 IST (2 min before pre-open window)
            tomorrow = today + timedelta(days=1)
            next_run = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 8, 58, 0, tzinfo=IST)
            sleep_secs = (next_run - datetime.now(IST)).total_seconds()
            if sleep_secs > 0:
                logger.info(f"[PreMarketCollector] Next cycle in {sleep_secs/3600:.1f}h")
                # Sleep in small increments so stop_event is checked
                while sleep_secs > 0 and not self._stop_event.is_set():
                    time.sleep(min(sleep_secs, 60))
                    sleep_secs -= 60

    def stop(self):
        self._stop_event.set()
        logger.info("[PreMarketCollector] Stop requested")
