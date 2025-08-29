#!/usr/bin/env python3
"""
Unit tests for Live Paper Trading System
Tests critical functions: open, close, exposure, position sizing
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from live_paper_trading import LivePaperTradingSystem, PaperTrade


class TestLivePaperTradingSystem(unittest.TestCase):
    """Test cases for Live Paper Trading System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trading_system = LivePaperTradingSystem(
            initial_capital=100000.0,
            max_risk_per_trade=0.02,
            confidence_cutoff=40.0,
            exposure_limit=0.6,
            max_daily_loss_pct=0.03,
            commission_bps=1.0,
            slippage_bps=5.0,
            symbols=['NSE:NIFTY50-INDEX'],
            data_provider='fyers',
            stop_loss_pct=-30.0,
            take_profit_pct=25.0,
            time_stop_minutes=30,
            verbose=False
        )
        
        # Mock data manager
        self.trading_system.data_manager = Mock()
        self.trading_system.data_manager.get_underlying_price.return_value = 20000.0
        
        # Mock database
        self.trading_system.db = Mock()
        
        # Mock strategies
        self.trading_system.strategies = {
            'test_strategy': Mock()
        }
    
    def test_safe_pct(self):
        """Test safe percentage calculation."""
        # Test normal case
        result = self.trading_system.safe_pct(50, 100)
        self.assertEqual(result, 50.0)
        
        # Test division by zero
        result = self.trading_system.safe_pct(50, 0)
        self.assertEqual(result, 0.0)
        
        # Test negative values
        result = self.trading_system.safe_pct(-25, 100)
        self.assertEqual(result, -25.0)
    
    def test_calculate_dynamic_position_size(self):
        """Test dynamic position size calculation."""
        # Test normal case
        risk_amount = 2000.0  # 2% of 100000
        confidence = 75.0
        entry_price = 100.0
        lot_size = 50
        
        position_size = self.trading_system._calculate_dynamic_position_size(
            risk_amount, confidence, entry_price, lot_size
        )
        
        # Expected: (2000 * 1.5) / (100 * 50) = 3000 / 5000 = 0.6 lots
        # But minimum is 1 lot
        self.assertEqual(position_size, 1)
        
        # Test with higher confidence
        position_size = self.trading_system._calculate_dynamic_position_size(
            risk_amount, 90.0, entry_price, lot_size
        )
        # Expected: (2000 * 1.8) / (100 * 50) = 3600 / 5000 = 0.72 lots
        # But minimum is 1 lot
        self.assertEqual(position_size, 1)
    
    def test_check_exposure_limits(self):
        """Test exposure limit checking."""
        # Test within limits
        result = self.trading_system._check_exposure_limits(
            'NSE:NIFTY50-INDEX', 10000.0, {'NSE:NIFTY50-INDEX': 20000.0}
        )
        self.assertTrue(result)
        
        # Test exceeding total exposure
        result = self.trading_system._check_exposure_limits(
            'NSE:NIFTY50-INDEX', 70000.0, {'NSE:NIFTY50-INDEX': 20000.0}
        )
        self.assertFalse(result)
        
        # Test with zero equity
        result = self.trading_system._check_exposure_limits(
            'NSE:NIFTY50-INDEX', 10000.0, {}
        )
        self.assertFalse(result)
    
    def test_commission_calculation(self):
        """Test commission calculation."""
        # Test commission calculation
        amount = 10000.0
        commission = self.trading_system._commission_amount(amount)
        
        # Expected: 10000 * 0.0001 = 1.0
        self.assertEqual(commission, 1.0)
        
        # Test with zero amount
        commission = self.trading_system._commission_amount(0.0)
        self.assertEqual(commission, 0.0)
    
    def test_slippage_calculation(self):
        """Test slippage calculation."""
        # Test buy slippage
        buy_price = self.trading_system._apply_slippage(100.0, is_buy=True)
        # Expected: 100 * (1 + 0.0005) = 100.05
        self.assertGreater(buy_price, 100.0)
        
        # Test sell slippage
        sell_price = self.trading_system._apply_slippage(100.0, is_buy=False)
        # Expected: 100 * (1 - 0.0005) = 99.95
        self.assertLess(sell_price, 100.0)
    
    def test_timezone_handling(self):
        """Test timezone-aware datetime handling."""
        # Test now_kolkata
        kolkata_time = self.trading_system.now_kolkata()
        self.assertIsNotNone(kolkata_time.tzinfo)
        self.assertEqual(str(kolkata_time.tzinfo), 'Asia/Kolkata')


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 