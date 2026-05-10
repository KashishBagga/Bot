#!/usr/bin/env python3
"""
NSE F&O Expiry & Event Blackout Manager
========================================
Prevents new trade entries during high-risk windows:
  1. NSE Weekly F&O expiry days  (Thursday)
  2. Monthly F&O expiry          (last Thursday each month)
  3. RBI MPC policy dates        (hard-coded upcoming dates)
  4. Budget / government events  (hard-coded)
  5. Configurable pre/post window in minutes around each event.

Usage:
    from src.core.expiry_blackout import ExpiryBlackoutManager
    manager = ExpiryBlackoutManager()
    blocked, reason = manager.is_blackout()
    if blocked:
        logger.info(f"Skipping entry: {reason}")
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, List
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

# ── Window (minutes) blocked BEFORE and AFTER each event ─────────────────────
PRE_EXPIRY_MINS  = 30    # block 30 min before expiry market-open
POST_EXPIRY_MINS = 30    # block 30 min after expiry market-open

PRE_EVENT_MINS   = 60    # for RBI / Budget (1 hour before)
POST_EVENT_MINS  = 60    # for RBI / Budget (1 hour after)

# ── Hard-coded high-impact event dates (IST, 24h format) ─────────────────────
# Format: (YYYY, MM, DD, HH, MM)  — use 09:15 for market-open events
RBI_MPC_DATES: List[Tuple] = [
    # RBI MPC announcement dates 2025-2026 (approximate — update quarterly)
    (2025,  4,  9, 10,  0),
    (2025,  6,  6, 10,  0),
    (2025,  8,  6, 10,  0),
    (2025, 10,  8, 10,  0),
    (2025, 12,  5, 10,  0),
    (2026,  2,  7, 10,  0),
    (2026,  4,  9, 10,  0),
    (2026,  6,  5, 10,  0),
]

BUDGET_DATES: List[Tuple] = [
    (2026,  2,  1,  9, 15),   # Union Budget 2026-27
]

# Additional one-off event dates (e.g., election results, US Fed meet)
CUSTOM_EVENT_DATES: List[Tuple] = [
    # Add custom high-impact dates here
]


class ExpiryBlackoutManager:
    """
    Determines whether the current moment falls inside a blackout window.
    Can be combined with a LunchHourFilter.
    """

    def __init__(
        self,
        pre_expiry_mins:  int = PRE_EXPIRY_MINS,
        post_expiry_mins: int = POST_EXPIRY_MINS,
        pre_event_mins:   int = PRE_EVENT_MINS,
        post_event_mins:  int = POST_EVENT_MINS,
    ):
        self.pre_expiry  = timedelta(minutes=pre_expiry_mins)
        self.post_expiry = timedelta(minutes=post_expiry_mins)
        self.pre_event   = timedelta(minutes=pre_event_mins)
        self.post_event  = timedelta(minutes=post_event_mins)

    # ── Public API ────────────────────────────────────────────────────────────

    def is_blackout(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Main entry point. Returns (is_blocked, reason).
        Call once per trading cycle before processing signals.
        """
        if now is None:
            now = datetime.now(IST)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=IST)

        # 1. Lunch-hour filter (11:30 – 13:30 IST)
        blocked, reason = self._check_lunch_hour(now)
        if blocked:
            return True, reason

        # 2. Weekly F&O expiry (all Thursdays)
        blocked, reason = self._check_weekly_expiry(now)
        if blocked:
            return True, reason

        # 3. Monthly F&O expiry (last Thursday of month)
        blocked, reason = self._check_monthly_expiry(now)
        if blocked:
            return True, reason

        # 4. RBI MPC dates
        for event_tuple in RBI_MPC_DATES:
            blocked, reason = self._check_event_window(now, event_tuple, "RBI MPC")
            if blocked:
                return True, reason

        # 5. Budget dates
        for event_tuple in BUDGET_DATES:
            blocked, reason = self._check_event_window(now, event_tuple, "Budget")
            if blocked:
                return True, reason

        # 6. Custom events
        for event_tuple in CUSTOM_EVENT_DATES:
            blocked, reason = self._check_event_window(now, event_tuple, "CustomEvent")
            if blocked:
                return True, reason

        return False, ""

    def next_blackout(self, now: Optional[datetime] = None) -> Optional[str]:
        """Return a human-readable description of the next upcoming blackout."""
        if now is None:
            now = datetime.now(IST)
        # Scan next 7 days in 30-minute increments
        for delta_minutes in range(0, 7 * 24 * 60, 30):
            future = now + timedelta(minutes=delta_minutes)
            blocked, reason = self.is_blackout(future)
            if blocked:
                return f"{future.strftime('%Y-%m-%d %H:%M')} IST — {reason}"
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _check_lunch_hour(now: datetime) -> Tuple[bool, str]:
        """Block 11:30 – 13:30 IST (low liquidity)."""
        t = now.time()
        from datetime import time
        if time(11, 30) <= t <= time(13, 30):
            return True, "Lunch-hour blackout (11:30–13:30 IST)"
        return False, ""

    def _check_weekly_expiry(self, now: datetime) -> Tuple[bool, str]:
        """
        Every Thursday is a weekly Nifty/BankNifty expiry.
        Block PRE_EXPIRY_MINS before market open to POST_EXPIRY_MINS after.
        """
        if now.weekday() != 3:   # 3 = Thursday
            return False, ""

        # Market open 09:15 IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        window_start = market_open - self.pre_expiry
        window_end   = market_open + self.post_expiry

        if window_start <= now <= window_end:
            return True, f"Weekly F&O expiry blackout ({window_start.strftime('%H:%M')}–{window_end.strftime('%H:%M')} IST)"
        return False, ""

    def _check_monthly_expiry(self, now: datetime) -> Tuple[bool, str]:
        """
        Last Thursday of each calendar month = monthly F&O expiry.
        Extend the block to the entire trading day ± 60 min.
        """
        if now.weekday() != 3:
            return False, ""

        if not self._is_last_thursday(now.date()):
            return False, ""

        # Whole trading day block
        day_start = now.replace(hour=9, minute=15, second=0, microsecond=0) - timedelta(hours=1)
        day_end   = now.replace(hour=15, minute=30, second=0, microsecond=0) + timedelta(hours=1)

        if day_start <= now <= day_end:
            return True, f"Monthly F&O expiry blackout ({now.strftime('%Y-%m-%d')})"
        return False, ""

    def _check_event_window(
        self, now: datetime, event_tuple: Tuple, label: str
    ) -> Tuple[bool, str]:
        year, month, day, hour, minute = event_tuple
        event_time = datetime(year, month, day, hour, minute, tzinfo=IST)
        window_start = event_time - self.pre_event
        window_end   = event_time + self.post_event

        if window_start <= now <= window_end:
            return True, (
                f"{label} event blackout "
                f"({window_start.strftime('%Y-%m-%d %H:%M')}–"
                f"{window_end.strftime('%H:%M')} IST)"
            )
        return False, ""

    @staticmethod
    def _is_last_thursday(d: date) -> bool:
        """Return True if `d` is the last Thursday of its month."""
        if d.weekday() != 3:
            return False
        next_thursday = d + timedelta(weeks=1)
        return next_thursday.month != d.month


# ── Module-level singleton ────────────────────────────────────────────────────
expiry_blackout = ExpiryBlackoutManager()
