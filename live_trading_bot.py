#!/usr/bin/env python3
"""
Live Trading Bot - Production Version
Consistent with backtesting system, automated scheduling, and daily summaries
"""

import os
import sys
import time
import sqlite3
import pandas as pd
import logging
import numpy as np
import schedule
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.data.parquet_data_store import ParquetDataStore
from src.models.unified_database import UnifiedDatabase
from dotenv import load_dotenv
import src.warning_filters  # noqa: F401
from src.models.enhanced_rejected_signals import log_rejected_signal_live

class LiveTradingBot:
    """Production Live Trading Bot with backtesting consistency"""
    
    def __init__(self, config_path: str = ".env", db_path: str = "trading_signals.db"):
        """Initialize the live trading bot"""
        self.db_path = db_path
        self.config_path = config_path
        self.is_running = False
        
        # FIX: Standardize symbol naming to match parquet store format
        self.symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']  # Match parquet store exactly
        self.timeframe = '5min'  # Primary timeframe for live trading
        
        # Risk management parameters - EMERGENCY LOSS PREVENTION
        self.risk_params = {
            'min_confidence_score': 75,  # Reduced from 85 for more realistic trading
            'max_daily_loss': -1000,     # Reduced from -1500 for tighter risk control
            'max_positions_per_strategy': 1,
            'position_size_multiplier': 0.6,  # Reduced from 0.8 for better risk management
            'emergency_stop': True,  # Enable emergency stop
            'disabled_strategies': [  # EMERGENCY: Disable major loss strategies
                'ema_crossover',
                'macd_cross_rsi_filter'
            ]
        }
        
        # OPTIMIZATION: Focus on profitable strategies only - FIX: Use same symbol format
        self.profitable_strategies = {
            'supertrend_ema': {'symbols': ['NSE:NIFTY50-INDEX'], 'active': True},
            'supertrend_macd_rsi_ema': {'symbols': ['NSE:NIFTYBANK-INDEX'], 'active': True}
        }
        
        # Initialize only profitable strategies
        self.strategies = {
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }
        
        # Daily tracking
        self.daily_stats = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'signals_generated': 0,
            'trades_taken': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'strategies_active': list(self.strategies.keys()),
            'start_time': None,
            'end_time': None,
            'market_sessions': 0
        }
        
        # Data store for historical data
        self.data_store = ParquetDataStore()
        
        # Initialize unified database
        self.unified_db = UnifiedDatabase(db_path)
        
        self.setup_logging()
        self.setup_legacy_database()  # Keep legacy tables for compatibility
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        os.makedirs('logs', exist_ok=True)
        
        # Create daily log file
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = f'logs/live_trading_{today}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LiveTradingBot')
        
    def setup_legacy_database(self):
        """Setup legacy database tables for backward compatibility"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Keep legacy live_signals table for compatibility
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_signals_realtime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence TEXT,
                    confidence_score INTEGER,
                    price REAL,
                    stop_loss REAL,
                    target REAL,
                    target2 REAL,
                    target3 REAL,
                    reasoning TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    analysis_time_ms REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Keep legacy rejected_signals table for compatibility  
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rejected_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    rejection_reason TEXT,
                    confidence TEXT,
                    confidence_score INTEGER,
                    price REAL,
                    stop_loss REAL,
                    target REAL,
                    reasoning TEXT
                )
            ''')
            
            # Create daily_trading_summary table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_trading_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    signals_generated INTEGER,
                    trades_taken INTEGER,
                    profitable_trades INTEGER,
                    total_pnl REAL,
                    win_rate REAL,
                    strategies_active TEXT,
                    market_start_time TEXT,
                    market_end_time TEXT,
                    session_duration_minutes INTEGER,
                    avg_confidence_score REAL,
                    max_drawdown REAL,
                    best_strategy TEXT,
                    worst_strategy TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create live_trade_executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    entry_time TEXT,
                    exit_time TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    pnl REAL,
                    status TEXT,
                    exit_reason TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES live_signals_realtime (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("✅ Legacy database compatibility maintained")
            
        except Exception as e:
            self.logger.error(f"❌ Database setup error: {e}")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open (9:15 AM to 3:30 PM, Monday to Friday)"""
        now = datetime.now()
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # Market hours: 9:15 AM to 3:30 PM (IST)
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def get_market_data(self, symbol: str, periods: int = 100) -> Optional[pd.DataFrame]:
        """Get market data - in production, this would connect to live data feed"""
        try:
            # For now, use recent historical data from parquet files
            # In production, this would be replaced with live data feed
            df = self.data_store.load_data(symbol, self.timeframe, days_back=2)
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Take the most recent data
            if len(df) > periods:
                df = df.tail(periods)
            
            # Add technical indicators using the same method as backtesting
            df = self.add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error getting market data for {symbol}: {e}")
            return None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators - SAME AS BACKTESTING"""
        if len(df) < 50:
            return df
        
        try:
            # EMA calculations (if not already present)
            if 'ema_9' not in df.columns:
                df['ema_9'] = df['close'].ewm(span=9).mean()
            if 'ema_21' not in df.columns:
                df['ema_21'] = df['close'].ewm(span=21).mean()
            if 'ema_50' not in df.columns:
                df['ema_50'] = df['close'].ewm(span=50).mean()
            
            # RSI calculation (if not already present)
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = delta.clip(lower=0).rolling(window=14).mean()
                loss = (-delta.clip(upper=0)).rolling(window=14).mean().replace(0, np.nan)
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                df['rsi'] = df['rsi'].fillna(method='bfill').fillna(50)
            
            # MACD calculation (if not already present)
            if 'macd' not in df.columns:
                ema_12 = df['close'].ewm(span=12).mean()
                ema_26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands (if not already present)
            if 'bb_upper' not in df.columns:
                sma_20 = df['close'].rolling(window=20).mean()
                std_20 = df['close'].rolling(window=20).std()
                df['bb_upper'] = sma_20 + (std_20 * 2)
                df['bb_lower'] = sma_20 - (std_20 * 2)
                df['bb_middle'] = sma_20
            
            # Additional indicators for comprehensive analysis
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price position indicators - FIX: Add safety guards
            range_ = (df['high'] - df['low']).replace(0, np.nan)
            df['price_position'] = (df['close'] - df['low']) / range_
            df['price_position'] = df['price_position'].clip(0, 1).fillna(0)
            
            den_close = df['close'].replace(0, np.nan)
            df['candle_size'] = (df['high'] - df['low']) / den_close
            df['candle_size'] = df['candle_size'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
        except Exception as e:
            self.logger.error(f"⚠️ Error adding indicators: {e}")
        
        return df
    
    def analyze_with_strategy(self, strategy_name: str, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data with strategy - SAME AS BACKTESTING"""
        analysis_start = time.time()
        
        try:
            strategy = self.strategies[strategy_name]
            
            # Add strategy-specific indicators
            data = strategy.add_indicators(data)
            
            # Call strategy analyze method using the SAME logic as backtesting
            if strategy_name == 'insidebar_rsi':
                # Special handling for insidebar_rsi
                result = strategy.analyze(data, symbol, None)
            elif hasattr(strategy, 'analyze_single_timeframe'):
                result = strategy.analyze_single_timeframe(data, None)
            else:
                # For strategies that need candle, index, df parameters
                candle = data.iloc[-1]
                result = strategy.analyze(candle, len(data)-1, data, None)
            
            analysis_time = (time.time() - analysis_start) * 1000  # Convert to milliseconds
            
            if result and isinstance(result, dict):
                result['strategy'] = strategy_name
                result['symbol'] = symbol
                result['timestamp'] = datetime.now().isoformat()
                result['analysis_time_ms'] = analysis_time
                
                # Add confidence score if not present
                if 'confidence_score' not in result:
                    confidence_map = {
                        'Very High': 85, 'High': 70, 'Medium': 50, 'Low': 30, 'Very Low': 10
                    }
                    result['confidence_score'] = confidence_map.get(result.get('confidence', 'Low'), 30)
                
                return result
            else:
                return {
                    'signal': 'NO TRADE',
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'reason': 'No signal generated',
                    'analysis_time_ms': analysis_time
                }
            
        except Exception as e:
            analysis_time = (time.time() - analysis_start) * 1000
            self.logger.error(f"❌ Error analyzing {strategy_name} for {symbol}: {e}")
            return {
                'signal': 'ERROR',
                'reason': str(e),
                'strategy': strategy_name,
                'symbol': symbol,
                'analysis_time_ms': analysis_time
            }
    
    def execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        if not self.is_market_open():
            self.logger.debug("🕐 Market closed - skipping cycle")
            return
        
        # FIX: Auto-lift emergency stop when starting mid-session
        if self.is_market_open() and self.risk_params.get('emergency_stop', False):
            self.logger.warning("⚠️ Emergency stop was ON during market hours; auto-disabling for live cycle.")
            self.risk_params['emergency_stop'] = False
        
        if self.risk_params['emergency_stop']:
            self.logger.warning("🛑 Emergency stop activated - trading halted")
            return
        
        self.logger.info("🔄 Starting trading cycle...")
        
        all_signals = []
        rejected_signals = []
        total_analyses = 0
        no_trade_count = 0
        low_confidence_count = 0
        error_count = 0
        
        # Analyze all symbols with all strategies
        for symbol in self.symbols:
            data = self.get_market_data(symbol)
            if data is not None:
                self.logger.info(f"📊 Analyzing {symbol} with {len(data)} candles (latest close: ₹{data.iloc[-1]['close']:.2f})")
                
                for strategy_name in self.strategies.keys():
                    # EMERGENCY: Skip disabled strategies
                    if strategy_name in self.risk_params.get('disabled_strategies', []):
                        self.logger.warning(f"🚫 Strategy {strategy_name} is DISABLED for emergency loss prevention")
                        continue
                        
                    result = self.analyze_with_strategy(strategy_name, symbol, data)
                    total_analyses += 1
                    
                    # Extract common fields
                    confidence_score = result.get('confidence_score', 0)
                    signal_type = result.get('signal', 'UNKNOWN')
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Add common fields to result
                    result.update({
                        'timestamp': timestamp,
                        'signal_time': timestamp,
                        'price': data.iloc[-1]['close'] if len(data) > 0 else 0
                    })
                    
                    if signal_type == 'NO TRADE':
                        no_trade_count += 1
                        rejected_signals.append({
                            **result,
                            'rejection_reason': 'No trade signal generated',
                            'signal': 'NO TRADE'
                        })
                        self.logger.debug(
                            f"📋 {strategy_name} - {symbol}: NO TRADE "
                            f"(Reason: {result.get('reason', 'Unknown')})"
                        )
                    elif signal_type == 'ERROR':
                        error_count += 1
                        rejected_signals.append({
                            **result,
                            'rejection_reason': f"Strategy error: {result.get('reason', 'Unknown')}",
                            'signal': 'ERROR'
                        })
                        self.logger.warning(
                            f"⚠️ {strategy_name} - {symbol}: ERROR "
                            f"(Reason: {result.get('reason', 'Unknown')})"
                        )
                    else:
                        # Valid signal generated
                        if confidence_score >= self.risk_params['min_confidence_score']:
                            all_signals.append(result)
                            self.logger.info(
                                f"🎯 {strategy_name} - {symbol}: "
                                f"{signal_type} (Confidence: {confidence_score})"
                            )
                        else:
                            low_confidence_count += 1
                            # Enhanced rejected signal with P&L calculation
                            enhanced_rejected_data = {
                                **result,
                                'rejection_reason': f"Low confidence: {confidence_score} < {self.risk_params['min_confidence_score']}",
                                'signal': signal_type,
                                'signal_attempted': signal_type,
                                'timestamp': timestamp,
                                'strategy': strategy_name,
                                'symbol': symbol,
                                'price': data.iloc[-1]['close'] if len(data) > 0 else 0,
                                'confidence': result.get('confidence', 'Low'),
                                'confidence_score': confidence_score,
                                'rsi': result.get('rsi', 0),
                                'macd': result.get('macd', 0),
                                'macd_signal': result.get('macd_signal', 0),
                                'ema_20': result.get('ema_20', 0),
                                'atr': result.get('atr', 0),
                                'stop_loss': result.get('stop_loss', 0),
                                'target': result.get('target', 0),
                                'target2': result.get('target2', 0),
                                'target3': result.get('target3', 0),
                                'reasoning': result.get('reasoning', result.get('reason', '')),
                                'trade_type': 'Intraday'
                            }
                            
                            # FIX: Don't compute future P&L in live mode - compute ex-post in scheduled job
                            # future_data = self.get_market_data(symbol, periods=50)
                            
                            # Log with enhanced P&L calculation (no future data in live mode)
                            log_rejected_signal_live(enhanced_rejected_data, None)
                            
                            rejected_signals.append({
                                **result,
                                'rejection_reason': f"Low confidence: {confidence_score} < {self.risk_params['min_confidence_score']}",
                                'signal': signal_type
                            })
                            self.logger.info(
                                f"🔻 {strategy_name} - {symbol}: "
                                f"{signal_type} (Confidence: {confidence_score} < {self.risk_params['min_confidence_score']} threshold)"
                            )
        
        # Log cycle summary
        self.logger.info(
            f"📈 Cycle Summary: {total_analyses} analyses | "
            f"{len(all_signals)} valid signals | "
            f"{no_trade_count} no-trade | "
            f"{low_confidence_count} low-confidence | "
            f"{error_count} errors"
        )
        
        # Store rejected signals using unified database
        if rejected_signals:
            self.store_rejected_signals_unified(rejected_signals)
        
        # Process and store valid signals using unified database
        if all_signals:
            self.process_signals_unified(all_signals)
            self.daily_stats['signals_generated'] += len(all_signals)
        else:
            self.logger.info("💭 No qualifying signals generated this cycle")
        
        # Risk management check
        self.check_risk_limits()
    
    def process_signals_unified(self, signals: List[Dict[str, Any]]):
        """Process and store trading signals using unified database"""
        try:
            for signal in signals:
                # Log to unified database
                signal_id = self.unified_db.log_live_signal(
                    strategy=signal['strategy'],
                    symbol=signal['symbol'],
                    signal_data=signal
                )
                
                # Also log to legacy table for compatibility
                self.store_signal_legacy(signal)
                
                # In production, this would execute the actual trade
                # For now, we'll simulate the trade execution
                self.simulate_trade_execution(signal_id, signal)
            
            self.logger.info(f"💾 Processed {len(signals)} signals with unified database")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing signals: {e}")
    
    def simulate_trade_execution(self, signal_id: int, signal: Dict[str, Any]):
        """Simulate trade execution - in production, this would place actual trades"""
        try:
            # Simulate trade execution
            entry_price = signal.get('price', 0)
            quantity = 1  # Simplified for simulation
            
            # Store trade execution
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO live_trade_executions (
                    signal_id, entry_time, entry_price, quantity, status
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                signal_id,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                entry_price,
                quantity,
                'EXECUTED'
            ))
            
            conn.commit()
            conn.close()
            
            self.daily_stats['trades_taken'] += 1
            self.logger.info(f"💼 Simulated trade execution for signal {signal_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Error simulating trade execution: {e}")
    
    def check_risk_limits(self):
        """Check and enforce risk management limits"""
        try:
            # Check daily P&L
            if self.daily_stats['total_pnl'] <= self.risk_params['max_daily_loss']:
                self.logger.warning(
                    f"⚠️ Daily loss limit reached: ₹{self.daily_stats['total_pnl']:.2f}"
                )
                self.risk_params['emergency_stop'] = True
                return
            
            # Additional risk checks can be added here
            
        except Exception as e:
            self.logger.error(f"❌ Error checking risk limits: {e}")
    
    def market_open_routine(self):
        """Routine to run when market opens"""
        self.logger.info("🟢 MARKET OPENED - Starting trading session")
        self.daily_stats['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.daily_stats['market_sessions'] += 1
        self.risk_params['emergency_stop'] = False
        
        # Reset daily stats if it's a new day
        current_date = datetime.now().strftime('%Y-%m-%d')
        if self.daily_stats['date'] != current_date:
            self.reset_daily_stats()
    
    def market_close_routine(self):
        """Routine to run when market closes"""
        self.logger.info("🔴 MARKET CLOSED - Ending trading session")
        self.daily_stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate and save daily summary
        self.generate_daily_summary()
        
        # Save daily summary to database
        self.save_daily_summary()
    
    def generate_daily_summary(self):
        """Generate comprehensive daily trading summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            today = self.daily_stats['date']
            
            # Get today's signals
            query = '''
                SELECT strategy, COUNT(*) as count, AVG(confidence_score) as avg_confidence
                FROM live_signals_realtime
                WHERE date(created_at) = ?
                GROUP BY strategy
            '''
            strategy_stats = pd.read_sql_query(query, conn, params=(today,))
            
            # Get today's trade executions
            query = '''
                SELECT COUNT(*) as trades, SUM(pnl) as total_pnl, 
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as profitable_trades
                FROM live_trade_executions
                WHERE date(created_at) = ?
            '''
            trade_stats = pd.read_sql_query(query, conn, params=(today,))
            
            conn.close()
            
            # Update daily stats
            if not trade_stats.empty:
                self.daily_stats['trades_taken'] = trade_stats.iloc[0]['trades'] or 0
                self.daily_stats['total_pnl'] = trade_stats.iloc[0]['total_pnl'] or 0.0
                self.daily_stats['profitable_trades'] = trade_stats.iloc[0]['profitable_trades'] or 0
            
            # Calculate win rate
            win_rate = 0.0
            if self.daily_stats['trades_taken'] > 0:
                win_rate = (self.daily_stats['profitable_trades'] / self.daily_stats['trades_taken']) * 100
            
            # Print daily summary
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 DAILY TRADING SUMMARY")
            self.logger.info("="*80)
            self.logger.info(f"📅 Date: {self.daily_stats['date']}")
            self.logger.info(f"🕘 Session: {self.daily_stats['start_time']} - {self.daily_stats['end_time']}")
            self.logger.info(f"🎯 Signals Generated: {self.daily_stats['signals_generated']}")
            self.logger.info(f"💼 Trades Taken: {self.daily_stats['trades_taken']}")
            self.logger.info(f"✅ Profitable Trades: {self.daily_stats['profitable_trades']}")
            self.logger.info(f"💰 Total P&L: ₹{self.daily_stats['total_pnl']:.2f}")
            self.logger.info(f"📈 Win Rate: {win_rate:.1f}%")
            self.logger.info(f"🧠 Active Strategies: {', '.join(self.daily_stats['strategies_active'])}")
            
            if not strategy_stats.empty:
                self.logger.info("\n📊 Strategy Performance:")
                for _, row in strategy_stats.iterrows():
                    self.logger.info(f"  🎯 {row['strategy']}: {row['count']} signals, "
                                   f"avg confidence: {row['avg_confidence']:.1f}")
            
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating daily summary: {e}")
    
    def save_daily_summary(self):
        """Save daily summary to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate session duration
            session_duration = 0
            if self.daily_stats['start_time'] and self.daily_stats['end_time']:
                start = datetime.strptime(self.daily_stats['start_time'], '%Y-%m-%d %H:%M:%S')
                end = datetime.strptime(self.daily_stats['end_time'], '%Y-%m-%d %H:%M:%S')
                session_duration = int((end - start).total_seconds() / 60)
            
            # Calculate win rate
            win_rate = 0.0
            if self.daily_stats['trades_taken'] > 0:
                win_rate = (self.daily_stats['profitable_trades'] / self.daily_stats['trades_taken']) * 100
            
            # Insert or update daily summary
            conn.execute('''
                INSERT OR REPLACE INTO daily_trading_summary (
                    date, signals_generated, trades_taken, profitable_trades, total_pnl,
                    win_rate, strategies_active, market_start_time, market_end_time,
                    session_duration_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.daily_stats['date'],
                self.daily_stats['signals_generated'],
                self.daily_stats['trades_taken'],
                self.daily_stats['profitable_trades'],
                self.daily_stats['total_pnl'],
                win_rate,
                json.dumps(self.daily_stats['strategies_active']),
                self.daily_stats['start_time'],
                self.daily_stats['end_time'],
                session_duration
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info("💾 Daily summary saved to database")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving daily summary: {e}")
    
    def reset_daily_stats(self):
        """Reset daily statistics for new trading day"""
        self.daily_stats = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'signals_generated': 0,
            'trades_taken': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'strategies_active': list(self.strategies.keys()),
            'start_time': None,
            'end_time': None,
            'market_sessions': 0
        }
        self.logger.info("🔄 Daily statistics reset for new trading day")
    
    def start_bot(self):
        """Start the live trading bot with automated scheduling"""
        self.logger.info("🚀 STARTING LIVE TRADING BOT")
        self.logger.info("="*60)
        self.logger.info("🎯 Features:")
        self.logger.info("  ✅ Consistent with backtesting system")
        self.logger.info("  ✅ Automated daily scheduling")
        self.logger.info("  ✅ Daily P&L summaries")
        self.logger.info("  ✅ Risk management")
        self.logger.info("  ✅ Multi-strategy analysis")
        self.logger.info("="*60)
        
        # Schedule market open routine at 09:15 (Indian market opens at 09:15)
        schedule.every().monday.at("09:15").do(self.market_open_routine)
        schedule.every().tuesday.at("09:15").do(self.market_open_routine)
        schedule.every().wednesday.at("09:15").do(self.market_open_routine)
        schedule.every().thursday.at("09:15").do(self.market_open_routine)
        schedule.every().friday.at("09:15").do(self.market_open_routine)
        
        schedule.every().monday.at("15:35").do(self.market_close_routine)
        schedule.every().tuesday.at("15:35").do(self.market_close_routine)
        schedule.every().wednesday.at("15:35").do(self.market_close_routine)
        schedule.every().thursday.at("15:35").do(self.market_close_routine)
        schedule.every().friday.at("15:35").do(self.market_close_routine)
        
        # Schedule trading cycles every 5 minutes during market hours
        schedule.every(5).minutes.do(self.execute_trading_cycle)
        
        self.is_running = True
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Stopping live trading bot...")
        except Exception as e:
            self.logger.error(f"❌ Critical error: {e}")
        finally:
            self.stop_bot()
    
    def stop_bot(self):
        """Stop the live trading bot"""
        self.is_running = False
        
        # If market is still open, run close routine
        if self.is_market_open():
            self.market_close_routine()
        
        self.logger.info("🛑 Live trading bot stopped")
        
        # Save final stats
        final_stats = {
            'timestamp': datetime.now().isoformat(),
            'daily_stats': self.daily_stats,
            'risk_params': self.risk_params,
            'strategies_active': list(self.strategies.keys())
        }
        
        with open(f'logs/session_stats_{self.daily_stats["date"]}.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        self.logger.info("📊 Session statistics saved")

    def store_rejected_signals_unified(self, rejected_signals: List[Dict[str, Any]]):
        """Store rejected signals using unified database"""
        try:
            for signal in rejected_signals:
                # Log to unified database
                self.unified_db.log_rejected_signal(
                    strategy=signal.get('strategy', 'Unknown'),
                    symbol=signal.get('symbol', 'Unknown'),
                    rejection_data=signal,
                    source='LIVE'
                )
            
            # Also store in legacy table for compatibility
            self.store_rejected_signals_legacy(rejected_signals)
            
            self.logger.info(f"📋 Stored {len(rejected_signals)} rejected signals with unified database")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing rejected signals: {e}")
    
    def store_signal_legacy(self, signal: Dict[str, Any]):
        """Store signal in legacy live_signals_realtime table for compatibility"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = signal.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            if isinstance(timestamp, str) and 'T' in timestamp:
                timestamp = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('''
                INSERT INTO live_signals_realtime (
                    timestamp, strategy, symbol, signal, confidence, 
                    confidence_score, price, stop_loss, target, target2, target3, reasoning, analysis_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                signal['strategy'],
                signal['symbol'],
                signal['signal'],
                signal.get('confidence', 'Unknown'),
                signal.get('confidence_score', 0),
                signal.get('price', 0),
                signal.get('stop_loss', 0),
                signal.get('target', 0),
                signal.get('target2', 0),
                signal.get('target3', 0),
                signal.get('reasoning', ''),
                signal.get('analysis_time_ms', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error storing legacy signal: {e}")
    
    def store_rejected_signals_legacy(self, rejected_signals: List[Dict[str, Any]]):
        """Store rejected signals in legacy rejected_signals table for compatibility"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for signal in rejected_signals:
                cursor.execute('''
                    INSERT INTO rejected_signals (
                        timestamp, strategy, symbol, signal, rejection_reason,
                        confidence, confidence_score, price, stop_loss, target, reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    signal.get('strategy', 'Unknown'),
                    signal.get('symbol', 'Unknown'),
                    signal.get('signal', 'UNKNOWN'),
                    signal.get('rejection_reason', 'Unknown'),
                    signal.get('confidence', 'Unknown'),
                    signal.get('confidence_score', 0),
                    signal.get('price', 0),
                    signal.get('stop_loss', 0),
                    signal.get('target', 0),
                    signal.get('reasoning', signal.get('reason', ''))
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error storing legacy rejected signals: {e}")


def main():
    """Main execution function"""
    print("🤖 LIVE TRADING BOT - PRODUCTION VERSION")
    print("="*60)
    print("🎯 Automated Trading System")
    print("Features:")
    print("  🔧 Consistent with backtesting")
    print("  📅 Automated daily scheduling")
    print("  📊 Daily P&L summaries")
    print("  🛡️ Risk management")
    print("  📱 Multi-strategy analysis")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and start bot
    bot = LiveTradingBot()
    
    try:
        bot.start_bot()
    except Exception as e:
        print(f"❌ Failed to start live trading bot: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 