#!/usr/bin/env python3
"""
Timezone Utilities for Production Systems
Centralized timezone handling to prevent naive datetime issues
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Optional

logger = logging.getLogger(__name__)

class TimezoneManager:
    """Centralized timezone management for trading systems"""
    
    def __init__(self, timezone: str = "Asia/Kolkata"):
        self.timezone = timezone
        self.tz = ZoneInfo(timezone)
        
    def now(self) -> datetime:
        """Get current timezone-aware datetime"""
        return datetime.now(tz=self.tz)
    
    def now_kolkata(self) -> datetime:
        """Get current IST time (alias for now)"""
        return self.now()
    
    def to_naive(self, dt: datetime) -> datetime:
        """Convert timezone-aware datetime to naive (for DB storage)"""
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt
    
    def to_aware(self, dt: datetime) -> datetime:
        """Convert naive datetime to timezone-aware"""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.tz)
        return dt
    
    def is_market_hours(self, dt: Optional[datetime] = None) -> bool:
        """Check if given time is within market hours (9:15 AM - 3:30 PM IST)"""
        if dt is None:
            dt = self.now()
        
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = self.to_aware(dt)
        
        # Convert to IST if needed
        dt_ist = dt.astimezone(self.tz)
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if dt_ist.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check market hours (9:15 AM - 3:30 PM IST)
        market_start = dt_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = dt_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= dt_ist <= market_end
    
    def get_market_open_time(self, dt: Optional[datetime] = None) -> datetime:
        """Get next market open time"""
        if dt is None:
            dt = self.now()
        
        if dt.tzinfo is None:
            dt = self.to_aware(dt)
        
        dt_ist = dt.astimezone(self.tz)
        
        # Set to 9:15 AM today
        market_open = dt_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        
        # If market already closed today, get next trading day
        if dt_ist > market_open.replace(hour=15, minute=30):
            # Move to next day
            from datetime import timedelta
            market_open += timedelta(days=1)
            
            # Skip weekends
            while market_open.weekday() >= 5:
                market_open += timedelta(days=1)
        
        return market_open
    
    def get_market_close_time(self, dt: Optional[datetime] = None) -> datetime:
        """Get market close time for given date"""
        if dt is None:
            dt = self.now()
        
        if dt.tzinfo is None:
            dt = self.to_aware(dt)
        
        dt_ist = dt.astimezone(self.tz)
        
        # Set to 3:30 PM today
        market_close = dt_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_close
    
    def format_datetime(self, dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
        """Format datetime with timezone info"""
        if dt is None:
            dt = self.now()
        
        if dt.tzinfo is None:
            dt = self.to_aware(dt)
        
        return dt.strftime(format_str)

# Global timezone manager instance
timezone_manager = TimezoneManager()

# Convenience functions
def now() -> datetime:
    """Get current timezone-aware datetime"""
    return timezone_manager.now()

def now_kolkata() -> datetime:
    """Get current IST time"""
    return timezone_manager.now_kolkata()

def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """Check if given time is within market hours"""
    return timezone_manager.is_market_hours(dt)

def format_datetime(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Format datetime with timezone info"""
    return timezone_manager.format_datetime(dt, format_str)
