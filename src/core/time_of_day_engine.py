#!/usr/bin/env python3
"""
Time of Day Engine (MKE Stage 1 Context)
========================================
Manages the TimeOfDayProfile for the Indian equity market session (09:15 to 15:30 IST).
Provides U-curve volume and ATR normalization parameters statelessly in O(1) lookup.
"""

from datetime import datetime, time, timedelta
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class TimeOfDayProfileSlot:
    """Represents a single 5-minute interval's market profile slot."""
    def __init__(
        self,
        avg_volume_factor: float,
        avg_atr_factor: float,
        avg_range_factor: float,
        avg_wickiness: float,
        avg_efficiency: float
    ):
        self.avg_volume_factor = avg_volume_factor
        self.avg_atr_factor = avg_atr_factor
        self.avg_range_factor = avg_range_factor
        self.avg_wickiness = avg_wickiness
        self.avg_efficiency = avg_efficiency


class TimeOfDayEngine:
    """
    Computes and caches typical intraday U-curve profiles for volume, ATR, and range.
    Insulates execution from database calls during the live loops.
    """
    required_history = 1

    def __init__(self, use_bootstrap: bool = True):
        self.profile: Dict[time, TimeOfDayProfileSlot] = {}
        if use_bootstrap:
            self._bootstrap_default_profile()

    def _bootstrap_default_profile(self):
        """
        Generates a standard U-curve profile for Indian market hours (09:15 to 15:30 IST).
        Volume and ATR are highest at the open and close, flat in the afternoon.
        """
        start_dt = datetime(2026, 1, 1, 9, 15)
        end_dt = datetime(2026, 1, 1, 15, 30)
        curr_dt = start_dt

        while curr_dt < end_dt:  # 75 bar-open slots: 09:15 → 15:25 (15:30 is bar close, not open)
            t = curr_dt.time()
            minutes_from_open = (curr_dt.hour - 9) * 60 + (curr_dt.minute - 15)
            
            # Simple mathematical U-curve:
            # high at 0 mins (open) and 375 mins (close), minimum around 180 mins (mid-day)
            # volume U-curve:
            if minutes_from_open < 60:  # First hour: high volume
                vol_factor = 2.5 - 1.5 * (minutes_from_open / 60.0)
                atr_factor = 2.0 - 1.0 * (minutes_from_open / 60.0)
            elif minutes_from_open > 300:  # Last hour: rising volume
                close_fraction = (minutes_from_open - 300) / 75.0
                vol_factor = 1.0 + 1.2 * close_fraction
                atr_factor = 1.0 + 0.8 * close_fraction
            else:  # Mid-day: low, stable volume
                vol_factor = 0.6 + 0.4 * abs(minutes_from_open - 180) / 120.0
                vol_factor = min(vol_factor, 1.0)
                atr_factor = 0.8 + 0.2 * abs(minutes_from_open - 180) / 120.0
                atr_factor = min(atr_factor, 1.0)

            self.profile[t] = TimeOfDayProfileSlot(
                avg_volume_factor=round(vol_factor, 3),
                avg_atr_factor=round(atr_factor, 3),
                avg_range_factor=round(atr_factor * 0.9, 3),
                avg_wickiness=0.25,
                avg_efficiency=0.55
            )
            curr_dt += timedelta(minutes=5)

        logger.info(f"✅ TimeOfDayEngine initialized with {len(self.profile)} slots.")

    def lookup(self, timestamp: datetime) -> TimeOfDayProfileSlot:
        """O(1) lookup for profile features at any given timestamp."""
        t = timestamp.time()
        # Round time to nearest 5 minutes
        minute = (t.minute // 5) * 5
        rounded_time = time(t.hour, minute)
        
        if rounded_time in self.profile:
            return self.profile[rounded_time]
        
        # Fallback if outside standard market hours (return neutral/standard slot)
        return TimeOfDayProfileSlot(
            avg_volume_factor=1.0,
            avg_atr_factor=1.0,
            avg_range_factor=1.0,
            avg_wickiness=0.25,
            avg_efficiency=0.55
        )
