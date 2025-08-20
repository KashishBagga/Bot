#!/usr/bin/env python3
"""
Enhanced Rejected Signals System
Tracks rejected signals with real P&L calculations from strategies
Separate tables for live trading and backtesting rejected signals
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import math

class EnhancedRejectedSignals:
    """Enhanced system for tracking rejected signals with real P&L calculation"""
    
    _tables_setup = False  # Class variable to track if tables have been set up
    
    def __init__(self, db_path: str = "trading_signals.db"):
        self.db_path = db_path
        self.setup_enhanced_tables()
    
    def setup_enhanced_tables(self):
        """Create enhanced rejected signals tables for both live trading and backtesting"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create enhanced rejected signals table for LIVE TRADING
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rejected_signals_live (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Basic signal info
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_attempted TEXT NOT NULL,  -- BUY CALL, BUY PUT, etc.
                    rejection_reason TEXT NOT NULL,
                    rejection_category TEXT,  -- NO_TRADE, LOW_CONFIDENCE, ERROR, RISK_LIMIT
                    
                    -- Market data at signal time
                    price REAL NOT NULL,
                    confidence TEXT,
                    confidence_score INTEGER,
                    
                    -- All indicator values for analysis
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    ema_9 REAL,
                    ema_21 REAL,
                    ema_20 REAL,
                    ema_50 REAL,
                    atr REAL,
                    supertrend REAL,
                    supertrend_direction INTEGER,
                    bb_upper REAL,
                    bb_lower REAL,
                    bb_middle REAL,
                    volume REAL,
                    
                    -- Strategy-specific indicators (JSON)
                    strategy_indicators TEXT,  -- JSON with all strategy-specific data
                    
                    -- Trade parameters that would have been used
                    stop_loss REAL,
                    target REAL,
                    target2 REAL,
                    target3 REAL,
                    trade_type TEXT,
                    
                    -- REAL PERFORMANCE DATA (calculated by strategy with future data)
                    outcome TEXT,  -- Win, Loss, Pending
                    pnl REAL,     -- Actual P&L calculated by strategy
                    targets_hit INTEGER,
                    stoploss_count INTEGER,
                    exit_time TEXT,
                    failure_reason TEXT,
                    
                    -- Analysis data
                    reasoning TEXT,
                    market_condition TEXT,
                    volatility_at_signal REAL,
                    
                    -- Metadata
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    future_data_available BOOLEAN DEFAULT 0  -- Whether future data was available for P&L calc
                )
            ''')
            
            # Create enhanced rejected signals table for BACKTESTING
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rejected_signals_backtest (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    
                    -- Basic signal info
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_attempted TEXT NOT NULL,  -- BUY CALL, BUY PUT, etc.
                    rejection_reason TEXT NOT NULL,
                    rejection_category TEXT,  -- NO_TRADE, LOW_CONFIDENCE, ERROR, RISK_LIMIT
                    
                    -- Market data at signal time
                    price REAL NOT NULL,
                    confidence TEXT,
                    confidence_score INTEGER,
                    
                    -- All indicator values for analysis
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    ema_9 REAL,
                    ema_21 REAL,
                    ema_20 REAL,
                    ema_50 REAL,
                    atr REAL,
                    supertrend REAL,
                    supertrend_direction INTEGER,
                    bb_upper REAL,
                    bb_lower REAL,
                    bb_middle REAL,
                    volume REAL,
                    
                    -- Strategy-specific indicators (JSON)
                    strategy_indicators TEXT,  -- JSON with all strategy-specific data
                    
                    -- Trade parameters that would have been used
                    stop_loss REAL,
                    target REAL,
                    target2 REAL,
                    target3 REAL,
                    trade_type TEXT,
                    
                    -- REAL PERFORMANCE DATA (calculated by strategy with future data)
                    outcome TEXT,  -- Win, Loss, Pending
                    pnl REAL,     -- Actual P&L calculated by strategy
                    targets_hit INTEGER,
                    stoploss_count INTEGER,
                    exit_time TEXT,
                    failure_reason TEXT,
                    
                    -- Analysis data
                    reasoning TEXT,
                    market_condition TEXT,
                    volatility_at_signal REAL,
                    
                    -- Backtesting specific fields
                    backtest_run_id TEXT,  -- Identifier for the backtest run
                    backtest_parameters TEXT,  -- JSON with backtest parameters
                    
                    -- Metadata
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    future_data_available BOOLEAN DEFAULT 1  -- Always true for backtesting
                )
            ''')
            
            conn.commit()
            conn.close()
            
            # Only print success message once per session
            if not EnhancedRejectedSignals._tables_setup:
                print("✅ Enhanced rejected signals tables created successfully")
                print("  • rejected_signals_live: For live trading rejected signals")
                print("  • rejected_signals_backtest: For backtesting rejected signals")
                EnhancedRejectedSignals._tables_setup = True
            
        except Exception as e:
            print(f"❌ Error setting up enhanced rejected signals tables: {e}")
    
    def log_rejected_signal(self, signal_data: Dict[str, Any], 
                           future_data: Optional[pd.DataFrame] = None,
                           is_backtest: bool = False,
                           backtest_run_id: str = None,
                           backtest_parameters: Dict[str, Any] = None):
        """
        Log a rejected signal with complete analysis including real P&L from strategy.
        
        Args:
            signal_data: Dictionary containing signal information with real performance data
            future_data: DataFrame with future price data (not used, kept for compatibility)
            is_backtest: Whether this is from backtesting (True) or live trading (False)
            backtest_run_id: Unique identifier for the backtest run
            backtest_parameters: Parameters used in the backtest
        """
        try:
            # Use real performance data from strategy (not hypothetical calculations)
            real_performance = {
                "outcome": signal_data.get('outcome', 'Pending'),
                "pnl": signal_data.get('pnl', 0.0),
                "targets_hit": signal_data.get('targets_hit', 0),
                "stoploss_count": signal_data.get('stoploss_count', 0),
                "exit_time": signal_data.get('exit_time', ''),
                "failure_reason": signal_data.get('failure_reason', ''),
                "future_data_available": True  # Always true for backtesting
            }
            
            # Extract all indicator values
            strategy_indicators = {
                'crossover_strength': signal_data.get('crossover_strength', 0),
                'momentum': signal_data.get('momentum', ''),
                'body_ratio': signal_data.get('body_ratio', 0),
                'price_position': signal_data.get('price_position', 0),
                'volume_ratio': signal_data.get('volume_ratio', 0),
                'candle_size': signal_data.get('candle_size', 0)
            }
            
            # Determine rejection category
            rejection_reason = signal_data.get('rejection_reason', 'Unknown')
            if 'low confidence' in rejection_reason.lower() or 'confidence' in rejection_reason.lower():
                rejection_category = 'LOW_CONFIDENCE'
            elif 'no trade' in rejection_reason.lower():
                rejection_category = 'NO_TRADE'
            elif 'error' in rejection_reason.lower():
                rejection_category = 'ERROR'
            elif 'risk' in rejection_reason.lower():
                rejection_category = 'RISK_LIMIT'
            else:
                rejection_category = 'OTHER'
            
            # Choose table based on whether it's backtesting or live trading
            table_name = "rejected_signals_backtest" if is_backtest else "rejected_signals_live"
            
            # Insert into appropriate table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if is_backtest:
                # Insert into backtesting table with real performance data
                payload = {
                    'timestamp': signal_data.get('timestamp', datetime.now().isoformat()),
                    'strategy': signal_data.get('strategy', 'Unknown'),
                    'symbol': signal_data.get('symbol', 'Unknown'),
                    'signal_attempted': signal_data.get('signal', signal_data.get('signal_attempted', 'UNKNOWN')),
                    'rejection_reason': rejection_reason,
                    'rejection_category': rejection_category,
                    'price': signal_data.get('price', 0),
                    'confidence': signal_data.get('confidence', 'Unknown'),
                    'confidence_score': signal_data.get('confidence_score', 0),
                    'rsi': signal_data.get('rsi', 0),
                    'macd': signal_data.get('macd', 0),
                    'macd_signal': signal_data.get('macd_signal', 0),
                    'macd_histogram': signal_data.get('macd_histogram', 0),
                    'ema_9': signal_data.get('ema_9', 0),
                    'ema_21': signal_data.get('ema_21', 0),
                    'ema_20': signal_data.get('ema_20', 0),
                    'ema_50': signal_data.get('ema_50', 0),
                    'atr': signal_data.get('atr', 0),
                    'supertrend': signal_data.get('supertrend', 0),
                    'supertrend_direction': signal_data.get('supertrend_direction', 0),
                    'bb_upper': signal_data.get('bb_upper', 0),
                    'bb_lower': signal_data.get('bb_lower', 0),
                    'bb_middle': signal_data.get('bb_middle', 0),
                    'volume': signal_data.get('volume', 0),
                    'strategy_indicators': json.dumps(strategy_indicators),
                    'stop_loss': signal_data.get('stop_loss', 0),
                    'target': signal_data.get('target', 0),
                    'target2': signal_data.get('target2', 0),
                    'target3': signal_data.get('target3', 0),
                    'trade_type': signal_data.get('trade_type', 'Intraday'),
                    'outcome': real_performance['outcome'],
                    'pnl': real_performance['pnl'],
                    'targets_hit': real_performance['targets_hit'],
                    'stoploss_count': real_performance['stoploss_count'],
                    'exit_time': real_performance['exit_time'],
                    'failure_reason': real_performance['failure_reason'],
                    'reasoning': signal_data.get('reasoning', signal_data.get('reason', '')),
                    'market_condition': signal_data.get('market_condition', 'Unknown'),
                    'volatility_at_signal': signal_data.get('volatility_at_signal', signal_data.get('atr', 0)),
                    'backtest_run_id': backtest_run_id or f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'backtest_parameters': json.dumps(backtest_parameters or {})
                }
                cursor.execute(f'''
                    INSERT INTO {table_name} (
                        timestamp, strategy, symbol, signal_attempted, rejection_reason, rejection_category,
                        price, confidence, confidence_score,
                        rsi, macd, macd_signal, macd_histogram, ema_9, ema_21, ema_20, ema_50, atr,
                        supertrend, supertrend_direction, bb_upper, bb_lower, bb_middle, volume,
                        strategy_indicators, stop_loss, target, target2, target3, trade_type,
                        outcome, pnl, targets_hit, stoploss_count, exit_time, failure_reason,
                        reasoning, market_condition, volatility_at_signal, 
                        backtest_run_id, backtest_parameters
                    ) VALUES (
                        :timestamp, :strategy, :symbol, :signal_attempted, :rejection_reason, :rejection_category,
                        :price, :confidence, :confidence_score,
                        :rsi, :macd, :macd_signal, :macd_histogram, :ema_9, :ema_21, :ema_20, :ema_50, :atr,
                        :supertrend, :supertrend_direction, :bb_upper, :bb_lower, :bb_middle, :volume,
                        :strategy_indicators, :stop_loss, :target, :target2, :target3, :trade_type,
                        :outcome, :pnl, :targets_hit, :stoploss_count, :exit_time, :failure_reason,
                        :reasoning, :market_condition, :volatility_at_signal, :backtest_run_id, :backtest_parameters
                    )
                ''', payload)
            else:
                # Insert into live trading table (similar structure but without backtest fields)
                cursor.execute(f'''
                    INSERT INTO {table_name} (
                        timestamp, strategy, symbol, signal_attempted, rejection_reason, rejection_category,
                        price, confidence, confidence_score,
                        rsi, macd, macd_signal, macd_histogram, ema_9, ema_21, ema_20, ema_50, atr,
                        supertrend, supertrend_direction, bb_upper, bb_lower, bb_middle, volume,
                        strategy_indicators, stop_loss, target, target2, target3, trade_type,
                        outcome, pnl, targets_hit, stoploss_count, exit_time, failure_reason,
                        reasoning, market_condition, volatility_at_signal, future_data_available
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data.get('timestamp', datetime.now().isoformat()),
                    signal_data.get('strategy', 'Unknown'),
                    signal_data.get('symbol', 'Unknown'),
                    signal_data.get('signal', signal_data.get('signal_attempted', 'UNKNOWN')),
                    rejection_reason,
                    rejection_category,
                    
                    # Market data
                    signal_data.get('price', 0),
                    signal_data.get('confidence', 'Unknown'),
                    signal_data.get('confidence_score', 0),
                    
                    # Indicators
                    signal_data.get('rsi', 0),
                    signal_data.get('macd', 0),
                    signal_data.get('macd_signal', 0),
                    signal_data.get('macd_histogram', 0),
                    signal_data.get('ema_9', 0),
                    signal_data.get('ema_21', 0),
                    signal_data.get('ema_20', 0),
                    signal_data.get('ema_50', 0),
                    signal_data.get('atr', 0),
                    signal_data.get('supertrend', 0),
                    signal_data.get('supertrend_direction', 0),
                    signal_data.get('bb_upper', 0),
                    signal_data.get('bb_lower', 0),
                    signal_data.get('bb_middle', 0),
                    signal_data.get('volume', 0),
                    
                    # Strategy-specific data
                    json.dumps(strategy_indicators),
                    
                    # Trade parameters
                    signal_data.get('stop_loss', 0),
                    signal_data.get('target', 0),
                    signal_data.get('target2', 0),
                    signal_data.get('target3', 0),
                    signal_data.get('trade_type', 'Intraday'),
                    
                    # Real performance data from strategy
                    real_performance['outcome'],
                    real_performance['pnl'],
                    real_performance['targets_hit'],
                    real_performance['stoploss_count'],
                    real_performance['exit_time'],
                    real_performance['failure_reason'],
                    
                    # Analysis data
                    signal_data.get('reasoning', signal_data.get('reason', '')),
                    signal_data.get('market_condition', 'Unknown'),
                    signal_data.get('volatility_at_signal', signal_data.get('atr', 0)),  # Using ATR as volatility proxy
                    real_performance['future_data_available']
                ))
            
            conn.commit()
            conn.close()
            
            # Only log summary for non-backtest or first few signals to reduce noise
            if not is_backtest or (is_backtest and signal_data.get('_log_summary', False)):
                source = "Backtest" if is_backtest else "Live"
                pnl_summary = f"P&L: ₹{real_performance['pnl']:.2f}" if real_performance['pnl'] != 0 else "P&L: ₹0.00"
                print(f"📋 {source} rejected signal logged: {signal_data.get('strategy', 'Unknown')} - {rejection_reason} ({pnl_summary})")
            
        except Exception as e:
            print(f"❌ Error logging rejected signal: {e}")
    
    def get_missed_opportunities_report(self, days: int = 7, min_pnl: float = 50.0, 
                                      is_backtest: bool = False) -> Dict[str, Any]:
        """
        Generate a report of missed opportunities (rejected signals that would have been profitable).
        
        Args:
            days: Number of days to look back
            min_pnl: Minimum P&L to consider as missed opportunity
            is_backtest: Whether to analyze backtest data (True) or live data (False)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            table_name = "rejected_signals_backtest" if is_backtest else "rejected_signals_live"
            
            # Get profitable rejected signals
            cursor.execute(f'''
                SELECT strategy, symbol, signal_attempted, rejection_reason, rejection_category,
                       price, confidence_score, pnl, targets_hit,
                       outcome, timestamp
                FROM {table_name}
                WHERE timestamp >= datetime('now', '-{days} days')
                  AND pnl >= ?
                  AND outcome = 'Win'
                  AND future_data_available = 1
                ORDER BY pnl DESC
            ''', (min_pnl,))
            
            missed_opportunities = cursor.fetchall()
            
            # Calculate totals
            total_missed_pnl = sum(row[7] for row in missed_opportunities)  # pnl
            
            # Group by rejection category
            category_analysis = {}
            for row in missed_opportunities:
                category = row[4]  # rejection_category
                if category not in category_analysis:
                    category_analysis[category] = {'count': 0, 'total_pnl': 0}
                category_analysis[category]['count'] += 1
                category_analysis[category]['total_pnl'] += row[7]
            
            # Group by strategy
            strategy_analysis = {}
            for row in missed_opportunities:
                strategy = row[0]
                if strategy not in strategy_analysis:
                    strategy_analysis[strategy] = {'count': 0, 'total_pnl': 0}
                strategy_analysis[strategy]['count'] += 1
                strategy_analysis[strategy]['total_pnl'] += row[7]
            
            conn.close()
            
            return {
                'total_missed_opportunities': len(missed_opportunities),
                'total_missed_pnl': total_missed_pnl,
                'avg_missed_pnl': total_missed_pnl / len(missed_opportunities) if missed_opportunities else 0,
                'category_breakdown': category_analysis,
                'strategy_breakdown': strategy_analysis,
                'top_missed_trades': missed_opportunities[:10]  # Top 10 biggest missed opportunities
            }
            
        except Exception as e:
            print(f"❌ Error generating missed opportunities report: {e}")
            return {}
    
    def get_rejection_analysis(self, days: int = 7, is_backtest: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive analysis of rejection patterns with P&L impact.
        
        Args:
            days: Number of days to analyze
            is_backtest: Whether to analyze backtest data (True) or live data (False)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            table_name = "rejected_signals_backtest" if is_backtest else "rejected_signals_live"
            
            # Overall rejection stats
            cursor.execute(f'''
                SELECT 
                    COUNT(*) as total_rejected,
                    SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as would_have_won,
                    SUM(CASE WHEN outcome = 'Loss' THEN 1 ELSE 0 END) as would_have_lost,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as total_missed_profits,
                    SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as total_avoided_losses
                FROM {table_name}
                WHERE timestamp >= datetime('now', '-{days} days')
                  AND future_data_available = 1
            ''')
            
            overall_stats = cursor.fetchone()
            
            # Rejection category analysis
            cursor.execute(f'''
                SELECT 
                    rejection_category,
                    COUNT(*) as count,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as profitable_count,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as loss_count
                FROM {table_name}
                WHERE timestamp >= datetime('now', '-{days} days')
                  AND future_data_available = 1
                GROUP BY rejection_category
                ORDER BY total_pnl DESC
            ''')
            
            category_stats = cursor.fetchall()
            
            conn.close()
            
            return {
                'period_days': days,
                'source': 'backtest' if is_backtest else 'live',
                'overall_stats': {
                    'total_rejected': overall_stats[0] or 0,
                    'would_have_won': overall_stats[1] or 0,
                    'would_have_lost': overall_stats[2] or 0,
                    'total_pnl': overall_stats[3] or 0.0,
                    'avg_pnl': overall_stats[4] or 0.0,
                    'total_missed_profits': overall_stats[5] or 0.0,
                    'total_avoided_losses': overall_stats[6] or 0.0,
                    'rejection_efficiency': ((overall_stats[6] or 0) / ((overall_stats[5] or 0) + (overall_stats[6] or 0))) * 100 if ((overall_stats[5] or 0) + (overall_stats[6] or 0)) > 0 else 0
                },
                'category_analysis': [
                    {
                        'category': row[0],
                        'count': row[1],
                        'total_pnl': row[2],
                        'avg_pnl': row[3],
                        'profitable_count': row[4],
                        'loss_count': row[5],
                        'win_rate': (row[4] / row[1]) * 100 if row[1] > 0 else 0
                    } for row in category_stats
                ]
            }
            
        except Exception as e:
            print(f"❌ Error generating rejection analysis: {e}")
            return {}

# Convenience functions for easy integration
def log_rejected_signal_live(signal_data: Dict[str, Any], 
                           future_data: Optional[pd.DataFrame] = None,
                           db_path: str = "trading_signals.db"):
    """
    Convenience function to log a rejected signal from LIVE TRADING with real P&L calculation.
    """
    enhanced_system = EnhancedRejectedSignals(db_path)
    enhanced_system.log_rejected_signal(signal_data, future_data, is_backtest=False)

def log_rejected_signal_backtest(signal_data: Dict[str, Any], 
                               future_data: Optional[pd.DataFrame] = None,
                               backtest_run_id: str = None,
                               backtest_parameters: Dict[str, Any] = None,
                               db_path: str = "trading_signals.db"):
    """
    Convenience function to log a rejected signal from BACKTESTING with real P&L calculation.
    """
    enhanced_system = EnhancedRejectedSignals(db_path)
    enhanced_system.log_rejected_signal(signal_data, future_data, is_backtest=True, 
                                      backtest_run_id=backtest_run_id, 
                                      backtest_parameters=backtest_parameters)

if __name__ == "__main__":
    # Test the enhanced system
    print("🧪 Testing Enhanced Rejected Signals System with Real Performance Data")
    
    enhanced_system = EnhancedRejectedSignals()
    
    # Test with sample live trading data
    sample_live_signal = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'ema_crossover',
        'symbol': 'BANKNIFTY',
        'signal_attempted': 'BUY CALL',
        'rejection_reason': 'Low confidence: 45 < 60 threshold',
        'price': 45000.0,
        'confidence': 'Low',
        'confidence_score': 45,
        'rsi': 55.5,
        'macd': 25.3,
        'macd_signal': 20.1,
        'ema_20': 44950.0,
        'atr': 120.0,
        'stop_loss': 120,
        'target': 180,
        'target2': 240,
        'target3': 300,
        'reasoning': 'EMA crossover but low confidence due to weak momentum',
        'outcome': 'Win',
        'pnl': 180.0,
        'targets_hit': 1,
        'stoploss_count': 0,
        'exit_time': '2025-01-11 10:30:00',
        'failure_reason': ''
    }
    
    # Test with sample backtest data
    sample_backtest_signal = {
        'timestamp': datetime.now().isoformat(),
        'strategy': 'supertrend_macd_rsi_ema',
        'symbol': 'NIFTY50',
        'signal_attempted': 'BUY PUT',
        'rejection_reason': 'No trade signal generated',
        'price': 24500.0,
        'confidence': 'Unknown',
        'confidence_score': 0,
        'rsi': 35.2,
        'macd': -15.8,
        'macd_signal': -10.3,
        'ema_20': 24520.0,
        'atr': 80.0,
        'stop_loss': 80,
        'target': 120,
        'target2': 160,
        'target3': 200,
        'reasoning': 'No clear signal pattern detected',
        'outcome': 'Pending',
        'pnl': 0.0,
        'targets_hit': 0,
        'stoploss_count': 0,
        'exit_time': '',
        'failure_reason': ''
    }
    
    print("✅ Sample live trading rejected signal logged")
    enhanced_system.log_rejected_signal(sample_live_signal, is_backtest=False)
    
    print("✅ Sample backtest rejected signal logged")
    enhanced_system.log_rejected_signal(sample_backtest_signal, is_backtest=True, 
                                      backtest_run_id="test_backtest_001",
                                      backtest_parameters={"timeframe": "5min", "days": 5})
    
    print("\n📊 Enhanced Rejected Signals System with Real Performance Data ready!")
    print("✅ rejected_signals_live: For live trading rejected signals")
    print("✅ rejected_signals_backtest: For backtesting rejected signals")
    print("Use log_rejected_signal_live() and log_rejected_signal_backtest() functions.")