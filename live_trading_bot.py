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
import threading
import logging
import numpy as np
import schedule
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.strategies.insidebar_rsi import InsidebarRsi
from src.strategies.ema_crossover import EmaCrossover
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.models.backtesting_summary import BacktestingSummary
from src.data.parquet_data_store import ParquetDataStore
from dotenv import load_dotenv

class LiveTradingBot:
    """Production Live Trading Bot with backtesting consistency"""
    
    def __init__(self, config_path: str = ".env", db_path: str = "trading_signals.db"):
        """Initialize the live trading bot"""
        self.db_path = db_path
        self.config_path = config_path
        self.is_running = False
        self.symbols = ['NSE_NIFTYBANK_INDEX', 'NSE_NIFTY50_INDEX']  # Updated to match data directory names
        self.timeframe = '5min'  # Primary timeframe for live trading
        
        # Initialize strategies exactly as in backtesting
        self.strategies = {
            'insidebar_rsi': InsidebarRsi(),
            'ema_crossover': EmaCrossover(),
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
        
        # Risk management parameters
        self.risk_params = {
            'min_confidence_score': 60,
            'max_daily_loss': -5000,
            'max_positions_per_strategy': 2,
            'position_size_multiplier': 1.0,
            'emergency_stop': False
        }
        
        # Data store for historical data
        self.data_store = ParquetDataStore()
        
        self.setup_logging()
        self.setup_database()
        
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
        
    def setup_database(self):
        """Setup database tables for live trading"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create live_signals table (consistent with backtesting)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_signals (
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
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
                    FOREIGN KEY (signal_id) REFERENCES live_signals (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("✅ Database setup completed")
            
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
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
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
            
            # Price position indicators
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['candle_size'] = (df['high'] - df['low']) / df['close']
            
        except Exception as e:
            self.logger.error(f"⚠️ Error adding indicators: {e}")
        
        return df
    
    def analyze_with_strategy(self, strategy_name: str, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data with strategy - SAME AS BACKTESTING"""
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
            
            if result and isinstance(result, dict):
                result['strategy'] = strategy_name
                result['symbol'] = symbol
                result['timestamp'] = datetime.now().isoformat()
                
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
                    'reason': 'No signal generated'
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing {strategy_name} for {symbol}: {e}")
            return {
                'signal': 'ERROR',
                'reason': str(e),
                'strategy': strategy_name,
                'symbol': symbol
            }
    
    def execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        if not self.is_market_open():
            self.logger.debug("🕐 Market closed - skipping cycle")
            return
        
        if self.risk_params['emergency_stop']:
            self.logger.warning("🛑 Emergency stop activated - trading halted")
            return
        
        self.logger.info("🔄 Starting trading cycle...")
        
        all_signals = []
        
        # Analyze all symbols with all strategies
        for symbol in self.symbols:
            data = self.get_market_data(symbol)
            if data is not None:
                for strategy_name in self.strategies.keys():
                    result = self.analyze_with_strategy(strategy_name, symbol, data)
                    
                    if result['signal'] not in ['NO TRADE', 'ERROR']:
                        confidence_score = result.get('confidence_score', 0)
                        
                        if confidence_score >= self.risk_params['min_confidence_score']:
                            all_signals.append(result)
                            self.logger.info(
                                f"🎯 {result['strategy']} - {result['symbol']}: "
                                f"{result['signal']} (Confidence: {confidence_score})"
                            )
        
        # Process and store signals
        if all_signals:
            self.process_signals(all_signals)
            self.daily_stats['signals_generated'] += len(all_signals)
        
        # Risk management check
        self.check_risk_limits()
    
    def process_signals(self, signals: List[Dict[str, Any]]):
        """Process and store trading signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            for signal in signals:
                timestamp = signal.get('timestamp', datetime.now().isoformat())
                if isinstance(timestamp, str) and 'T' in timestamp:
                    timestamp = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                
                # Store signal in live_signals table
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO live_signals (
                        timestamp, strategy, symbol, signal, confidence, 
                        confidence_score, price, stop_loss, target, target2, target3, reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    signal.get('reasoning', '')
                ))
                
                signal_id = cursor.lastrowid
                
                # In production, this would execute the actual trade
                # For now, we'll simulate the trade execution
                self.simulate_trade_execution(signal_id, signal)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"💾 Processed {len(signals)} signals")
            
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
                FROM live_signals
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
        
        # Schedule market routines
        schedule.every().monday.at("09:00").do(self.market_open_routine)
        schedule.every().tuesday.at("09:00").do(self.market_open_routine)
        schedule.every().wednesday.at("09:00").do(self.market_open_routine)
        schedule.every().thursday.at("09:00").do(self.market_open_routine)
        schedule.every().friday.at("09:00").do(self.market_open_routine)
        
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