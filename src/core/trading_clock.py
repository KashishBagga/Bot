#!/usr/bin/env python3
"""
Trading Clock Utility (MKE Stage 1 Context)
===========================================
Translates calendar datetime differences into discrete market time (bar counts).
Insulates structure analysis from overnight gaps and weekend pauses.
"""

from datetime import datetime, time, timedelta
import logging

logger = logging.getLogger(__name__)


class TradingClock:
    """Calculates trading bar counts between datetimes based on standard Indian market hours."""

    MARKET_START = time(9, 15)
    MARKET_END = time(15, 30)
    BARS_PER_DAY = 75  # 375 minutes / 5 minutes = 75 bars

    @staticmethod
    def bars_between(start_dt: datetime, end_dt: datetime, interval_minutes: int = 5) -> int:
        """
        Calculates the number of trading bars between two timestamps.
        Excludes weekends and non-market hours.
        """
        if start_dt > end_dt:
            return 0

        # Work strictly with dates and times
        d1, d2 = start_dt.date(), end_dt.date()
        
        # Calculate full trading days (excluding weekends)
        full_days = 0
        curr_date = d1 + timedelta(days=1)
        while curr_date < d2:
            if curr_date.weekday() < 5:  # Monday - Friday
                full_days += 1
            curr_date += timedelta(days=1)

        total_bars = full_days * TradingClock.BARS_PER_DAY

        # Calculate partial day bars
        t1 = start_dt.time()
        t2 = end_dt.time()

        # Clamp times to market hours
        def clamp_time(t: time) -> time:
            if t < TradingClock.MARKET_START:
                return TradingClock.MARKET_START
            if t > TradingClock.MARKET_END:
                return TradingClock.MARKET_END
            return t

        t1_clamped = clamp_time(t1)
        t2_clamped = clamp_time(t2)

        def minutes_from_open(t: time) -> int:
            return (t.hour - 9) * 60 + (t.minute - 15)

        if d1 == d2:
            # Same day
            if start_dt.weekday() >= 5:  # Weekend
                return 0
            m1 = minutes_from_open(t1_clamped)
            m2 = minutes_from_open(t2_clamped)
            return max(0, (m2 - m1) // interval_minutes)
        else:
            # Different days
            # Start day bars from start_time to close
            start_day_bars = 0
            if start_dt.weekday() < 5:
                m1 = minutes_from_open(t1_clamped)
                start_day_bars = max(0, (TradingClock.BARS_PER_DAY * interval_minutes - m1) // interval_minutes)

            # End day bars from open to end_time
            end_day_bars = 0
            if end_dt.weekday() < 5:
                m2 = minutes_from_open(t2_clamped)
                end_day_bars = max(0, m2 // interval_minutes)

            return total_bars + start_day_bars + end_day_bars
