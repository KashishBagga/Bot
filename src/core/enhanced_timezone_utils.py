#!/usr/bin/env python3
"""
Enhanced Timezone-Awareness with Market Sessions
Comprehensive market session management with pre-open, auction, and post-close periods
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List
from enum import Enum
import requests
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class MarketState(Enum):
    PREOPEN = "PREOPEN"
    OPEN = "OPEN"
    AUCTION = "AUCTION"
    CLOSED = "CLOSED"
    HOLIDAY = "HOLIDAY"

@dataclass
class MarketSession:
    """Market session configuration"""
    name: str
    start_time: str  # HH:MM format
    end_time: str    # HH:MM format
    state: MarketState

class EnhancedTimezoneManager:
    """Enhanced timezone management with market sessions and holiday handling"""
    
    def __init__(self, timezone: str = "Asia/Kolkata"):
        self.timezone = timezone
        self.tz = ZoneInfo(timezone)
        
        # Market sessions for NSE
        self.market_sessions = {
            'NSE': [
                MarketSession("Pre-Open", "09:00", "09:15", MarketState.PREOPEN),
                MarketSession("Normal Trading", "09:15", "15:30", MarketState.OPEN),
                MarketSession("Post-Close", "15:30", "16:00", MarketState.CLOSED),
            ],
            'BSE': [
                MarketSession("Pre-Open", "09:00", "09:15", MarketState.PREOPEN),
                MarketSession("Normal Trading", "09:15", "15:30", MarketState.OPEN),
                MarketSession("Post-Close", "15:30", "16:00", MarketState.CLOSED),
            ]
        }
        
        # Holiday management
        self.holiday_manager = HolidayManager()
        
    def now(self) -> datetime:
        """Get current timezone-aware datetime"""
        return datetime.now(tz=self.tz)
    
    def now_isoformat(self) -> str:
        """Get current timezone-aware datetime in ISO format"""
        return self.now().isoformat()
    
    def is_market_open(self, timestamp: Optional[datetime] = None, 
                      symbol: str = "NSE") -> MarketState:
        """Check market state for given timestamp and symbol"""
        if timestamp is None:
            timestamp = self.now()
        
        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.tz)
        
        # Convert to IST
        timestamp_ist = timestamp.astimezone(self.tz)
        
        # Check if it's a holiday
        if self.holiday_manager.is_holiday(timestamp_ist):
            return MarketState.HOLIDAY
        
        # Check if it's a weekday
        if timestamp_ist.weekday() >= 5:  # Saturday or Sunday
            return MarketState.CLOSED
        
        # Get market sessions for symbol
        sessions = self.market_sessions.get(symbol, self.market_sessions['NSE'])
        
        # Check current time against sessions
        current_time = timestamp_ist.time()
        
        for session in sessions:
            start_time = datetime.strptime(session.start_time, "%H:%M").time()
            end_time = datetime.strptime(session.end_time, "%H:%M").time()
            
            if start_time <= current_time < end_time:
                return session.state
        
        return MarketState.CLOSED
    
    def get_next_market_open(self, timestamp: Optional[datetime] = None) -> datetime:
        """Get next market open time"""
        if timestamp is None:
            timestamp = self.now()
        
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.tz)
        
        timestamp_ist = timestamp.astimezone(self.tz)
        
        # Set to 9:15 AM today
        market_open = timestamp_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        
        # If market already closed today, get next trading day
        if timestamp_ist > market_open.replace(hour=15, minute=30):
            market_open += timedelta(days=1)
        
        # Skip weekends and holidays
        while market_open.weekday() >= 5 or self.holiday_manager.is_holiday(market_open):
            market_open += timedelta(days=1)
        
        return market_open
    
    def get_market_close(self, timestamp: Optional[datetime] = None) -> datetime:
        """Get market close time for given date"""
        if timestamp is None:
            timestamp = self.now()
        
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.tz)
        
        timestamp_ist = timestamp.astimezone(self.tz)
        
        # Set to 3:30 PM
        market_close = timestamp_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_close
    
    def format_datetime(self, dt: Optional[datetime] = None, 
                       format_str: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
        """Format datetime with timezone info"""
        if dt is None:
            dt = self.now()
        
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        
        return dt.strftime(format_str)

class HolidayManager:
    """Enhanced holiday management with API integration"""
    
    def __init__(self):
        self.tz = ZoneInfo('Asia/Kolkata')
        self.holidays_cache = set()
        self.last_update = None
        self.cache_duration = timedelta(days=30)
        
        # Fallback holidays for 2024-2025
        self.fallback_holidays = {
            '2024-01-26', '2024-03-08', '2024-03-29', '2024-04-11',
            '2024-04-17', '2024-05-01', '2024-06-17', '2024-08-15',
            '2024-08-26', '2024-10-02', '2024-10-12', '2024-11-01',
            '2024-11-15', '2024-12-25',
            '2025-01-26', '2025-03-08', '2025-03-29', '2025-04-11',
            '2025-04-17', '2025-05-01', '2025-06-17', '2025-08-15',
            '2025-08-26', '2025-10-02', '2025-10-12', '2025-11-01',
            '2025-11-15', '2025-12-25'
        }
    
    def is_holiday(self, date: datetime = None) -> bool:
        """Check if given date is a market holiday"""
        if date is None:
            date = datetime.now(self.tz)
        
        # Ensure timezone-aware
        if date.tzinfo is None:
            date = date.replace(tzinfo=self.tz)
        
        date_str = date.strftime('%Y-%m-%d')
        
        # Check cache first
        if self._is_cache_valid():
            return date_str in self.holidays_cache
        
        # Update cache
        self._update_holiday_cache()
        
        # Check fallback if cache update failed
        if not self.holidays_cache:
            return date_str in self.fallback_holidays
        
        return date_str in self.holidays_cache
    
    def _is_cache_valid(self) -> bool:
        """Check if holiday cache is valid"""
        if self.last_update is None:
            return False
        
        return datetime.now(self.tz) - self.last_update < self.cache_duration
    
    def _update_holiday_cache(self):
        """Update holiday cache from API"""
        try:
            # Try to get holidays from NSE API
            current_year = datetime.now(self.tz).year
            holidays = self._fetch_holidays_from_api(current_year)
            
            if holidays:
                self.holidays_cache = holidays
                self.last_update = datetime.now(self.tz)
                logger.info(f"✅ Updated holiday cache with {len(holidays)} holidays")
            else:
                # Use fallback
                self.holidays_cache = self.fallback_holidays
                self.last_update = datetime.now(self.tz)
                logger.warning("⚠️ Using fallback holiday list")
                
        except Exception as e:
            logger.error(f"❌ Failed to update holiday cache: {e}")
            self.holidays_cache = self.fallback_holidays
            self.last_update = datetime.now(self.tz)
    
    def _fetch_holidays_from_api(self, year: int) -> set:
        """Fetch holidays from NSE API"""
        try:
            # NSE holiday API endpoint
            url = f"https://www.nseindia.com/api/holiday-master?type=trading"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            holidays = set()
            
            for holiday in data.get('data', []):
                if holiday.get('year') == year:
                    date_str = holiday.get('tradingDate', '')
                    if date_str:
                        holidays.add(date_str)
            
            return holidays
            
        except Exception as e:
            logger.error(f"❌ API holiday fetch failed: {e}")
            return set()

# Global enhanced timezone manager instance
enhanced_timezone_manager = EnhancedTimezoneManager()

# Convenience functions
def now() -> datetime:
    """Get current timezone-aware datetime"""
    return enhanced_timezone_manager.now()

def now_isoformat() -> str:
    """Get current timezone-aware datetime in ISO format"""
    return enhanced_timezone_manager.now_isoformat()

def is_market_open(timestamp: Optional[datetime] = None, symbol: str = "NSE") -> MarketState:
    """Check market state for given timestamp and symbol"""
    return enhanced_timezone_manager.is_market_open(timestamp, symbol)

def format_datetime(dt: Optional[datetime] = None, format_str: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """Format datetime with timezone info"""
    return enhanced_timezone_manager.format_datetime(dt, format_str)
