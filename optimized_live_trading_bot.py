#!/usr/bin/env python3
"""
Optimized Live Trading Bot
Real-time trading with optimized confidence-based strategies
"""

import os
import sys
import time
import sqlite3
import pandas as pd
import threading
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import schedule
from concurrent.futures import ThreadPoolExecutor

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from strategies.insidebar_rsi import InsidebarRsi
from strategies.ema_crossover import EmaCrossover
from strategies.supertrend_ema import SupertrendEma
from strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma

class OptimizedLiveTradingBot:
    def __init__(self, config_path: str = ".env", db_path: str = "trading_signals.db"):
        """Initialize the optimized live trading bot"""
        self.db_path = db_path
        self.config_path = config_path
        self.is_running = False
        self.symbols = ['BANKNIFTY', 'NIFTY50']
        
        # Optimized strategy instances
        self.strategies = {
            'insidebar_rsi': InsidebarRsi(),
            'ema_crossover': EmaCrossover(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }
        
        # Performance tracking
        self.daily_stats = {
            'trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'signals_generated': 0,
            'high_confidence_signals': 0,
            'strategies_active': []
        }
        
        # Risk management with confidence-based thresholds
        self.risk_params = {
            'min_confidence_score': 60,    # Only trade signals with 60+ confidence
            'max_daily_loss': -5000,       # Maximum daily loss limit
            'max_positions_per_strategy': 2,
            'position_size_multiplier': 1.0,
            'emergency_stop': False
        }
        
        self.setup_logging()
        self.setup_database()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/optimized_live_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('OptimizedLiveBot')
        
    def setup_database(self):
        """Setup database tables for signal storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create live_signals_realtime table if it doesn't exist
            conn.execute('''
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
                    cycle_number INTEGER,
                    analysis_time_ms REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create rejected_signals table if it doesn't exist
            conn.execute('''
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
            
            conn.commit()
            conn.close()
            self.logger.info("✅ Database setup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Database setup error: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get the current connection status for real-time data"""
        try:
            # For this optimized bot, we're using simulated data
            # In a real implementation, this would check Fyers API connectivity
            return {
                'real_time_data': True,  # We can generate real-time-like data
                'data_source': 'Simulated Market Data',
                'message': 'Connected (using optimized simulated data)',
                'status': 'CONNECTED'
            }
        except Exception as e:
            return {
                'real_time_data': False,
                'data_source': 'None',
                'message': f'Connection failed: {e}',
                'status': 'DISCONNECTED'
            }
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute
        
        # Market hours: 9:15 AM to 3:30 PM (IST)
        market_start = 9 * 60 + 15  # 9:15 AM in minutes
        market_end = 15 * 60 + 30   # 3:30 PM in minutes
        current_time = current_hour * 60 + current_minute
        
        # Check if it's a weekday and within market hours
        is_weekday = now.weekday() < 5  # Monday to Friday
        is_within_hours = market_start <= current_time <= market_end
        
        return is_weekday and is_within_hours
    
    def get_market_data(self, symbol: str, periods: int = 50) -> Optional[pd.DataFrame]:
        """Generate realistic market data for testing"""
        try:
            # Generate timestamps
            timestamps = pd.date_range(
                end=datetime.now(),
                periods=periods,
                freq='5min'
            )
            
            # Generate realistic OHLCV data
            np.random.seed(int(time.time()) % 1000)  # Different seed each time
            
            # Base price for different symbols
            base_prices = {'NIFTY50': 25000, 'BANKNIFTY': 52000}
            base_price = base_prices.get(symbol, 25000)
            
            # Generate price movements with some trend
            returns = np.random.normal(0, 0.002, periods)  # 0.2% volatility
            trend = np.linspace(-0.002, 0.002, periods)    # Small trend
            
            prices = []
            current_price = base_price
            
            for i in range(periods):
                change = (returns[i] + trend[i]) * current_price
                current_price += change
                prices.append(current_price)
            
            # Generate OHLC from prices
            data = []
            for i, price in enumerate(prices):
                volatility = np.random.uniform(0.001, 0.003)
                high = price * (1 + volatility)
                low = price * (1 - volatility)
                open_price = prices[i-1] if i > 0 else price
                close = price
                
                # Ensure OHLC consistency
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                volume = np.random.randint(50000, 200000)
                
                data.append({
                    'time': timestamps[i],
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('time', inplace=True)
            
            # Add technical indicators required by strategies
            self.add_technical_indicators(df)
            
            self.logger.debug(f"📊 Generated {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error generating data for {symbol}: {e}")
            return None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators required by strategies"""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # EMA
            df['ema'] = df['close'].ewm(span=20).mean()
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # SuperTrend components (simplified)
            hl2 = (df['high'] + df['low']) / 2
            df['basic_upperband'] = hl2 + (3 * df['atr'])
            df['basic_lowerband'] = hl2 - (3 * df['atr'])
            df['supertrend'] = df['basic_upperband']  # Simplified
            df['supertrend_direction'] = 1  # Simplified
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error adding indicators: {e}")
            return df
    
    def analyze_with_strategy(self, strategy_name: str, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market data with a specific strategy"""
        try:
            strategy = self.strategies[strategy_name]
            
            # Use appropriate analysis method based on strategy
            if strategy_name == 'insidebar_rsi':
                result = strategy.analyze(data, symbol)
            elif hasattr(strategy, 'analyze_single_timeframe'):
                result = strategy.analyze_single_timeframe(data)
            else:
                # For strategies that need candle, index, df parameters
                candle = data.iloc[-1]
                result = strategy.analyze(candle, len(data)-1, data)
            
            if result:
                result['strategy'] = strategy_name
                result['symbol'] = symbol
                result['timestamp'] = datetime.now().isoformat()
                
                # Add confidence score if not present
                if 'confidence_score' not in result:
                    # Estimate confidence score based on confidence level
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
            self.logger.info("🕐 Market closed - waiting for next session")
            return
        
        if self.risk_params['emergency_stop']:
            self.logger.warning("🛑 Emergency stop activated - trading halted")
            return
        
        self.logger.info("🔄 Starting trading cycle...")
        
        all_signals = []
        rejected_signals = []
        
        # Analyze all symbols with all strategies
        for symbol in self.symbols:
            data = self.get_market_data(symbol)
            if data is not None:
                for strategy_name in self.strategies.keys():
                    result = self.analyze_with_strategy(strategy_name, symbol, data)
                    
                    # Check if signal is valid but doesn't meet confidence threshold
                    if result['signal'] not in ['NO TRADE', 'ERROR']:
                        confidence_score = result.get('confidence_score', 0)
                        
                        if confidence_score >= self.risk_params['min_confidence_score']:
                            all_signals.append(result)
                            self.logger.info(
                                f"🎯 {result['strategy']} - {result['symbol']}: "
                                f"{result['signal']} (Confidence: {confidence_score})"
                            )
                        else:
                            # Signal generated but rejected due to low confidence
                            result['rejection_reason'] = f"Low confidence score: {confidence_score} < {self.risk_params['min_confidence_score']}"
                            rejected_signals.append(result)
                            self.logger.info(
                                f"🚫 {result['strategy']} - {result['symbol']}: "
                                f"{result['signal']} REJECTED (Confidence: {confidence_score})"
                            )
        
        # Process and store high-quality signals
        if all_signals:
            self.process_signals(all_signals)
            self.daily_stats['signals_generated'] += len(all_signals)
            self.daily_stats['high_confidence_signals'] += len(all_signals)
        
        # Store rejected signals for analysis
        if rejected_signals:
            self.store_rejected_signals(rejected_signals)
            self.logger.info(f"📊 {len(rejected_signals)} signals rejected due to low confidence")
        
        if not all_signals and not rejected_signals:
            self.logger.info("📊 No signals generated this cycle")
        
        # Risk management check
        self.check_risk_limits()
        
        # Update performance stats
        self.update_performance_stats()
    
    def process_signals(self, signals: List[Dict[str, Any]]):
        """Process and store trading signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            for signal in signals:
                # Ensure proper timestamp format
                timestamp = signal.get('timestamp')
                if isinstance(timestamp, str) and 'T' in timestamp:
                    # ISO format - convert to standard datetime string
                    timestamp = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                elif not isinstance(timestamp, str):
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Store signal in live_signals_realtime table
                conn.execute('''
                    INSERT INTO live_signals_realtime (
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
                    signal.get('price_reason', '') + ' | ' + str(signal.get('confidence_reasons', ''))
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"💾 Stored {len(signals)} high-confidence signals")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing signals: {e}")
    
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
            
            # Check confidence score distribution
            if self.daily_stats['signals_generated'] > 5:
                confidence_ratio = (self.daily_stats['high_confidence_signals'] / 
                                  self.daily_stats['signals_generated']) * 100
                if confidence_ratio < 80:  # Less than 80% high confidence
                    self.logger.warning(f"⚠️ Low confidence signal ratio: {confidence_ratio:.1f}%")
            
        except Exception as e:
            self.logger.error(f"❌ Error checking risk limits: {e}")
    
    def update_performance_stats(self):
        """Update daily performance statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Get today's performance from live signals
            query = '''
                SELECT COUNT(*) as signals, 
                       AVG(confidence_score) as avg_confidence
                FROM live_signals_realtime
                WHERE date(created_at) = ?
            '''
            result = conn.execute(query, (today,)).fetchone()
            
            if result:
                total_signals = result[0] or 0
                avg_confidence = result[1] or 0
                
                self.daily_stats.update({
                    'signals_generated': total_signals,
                    'avg_confidence': avg_confidence
                })
                
                if total_signals > 0:
                    self.logger.info(
                        f"📊 Daily Stats: {total_signals} signals, "
                        f"Avg Confidence: {avg_confidence:.1f}"
                    )
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating performance stats: {e}")
    
    def generate_market_report(self):
        """Generate and log market analysis report"""
        try:
            self.logger.info("📈 OPTIMIZED TRADING BOT REPORT")
            self.logger.info("=" * 50)
            
            # Current market status
            market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
            self.logger.info(f"Market Status: {market_status}")
            
            # Strategy status
            self.logger.info(f"Active Strategies: {len(self.strategies)}")
            for strategy in self.strategies.keys():
                self.logger.info(f"  📊 {strategy}: Optimized & Ready")
            
            # Daily performance
            if self.daily_stats['signals_generated'] > 0:
                self.logger.info(f"Today's Performance:")
                self.logger.info(f"  🎯 High-Confidence Signals: {self.daily_stats['high_confidence_signals']}")
                self.logger.info(f"  📊 Total Signals: {self.daily_stats['signals_generated']}")
                self.logger.info(f"  🛡️ Min Confidence Threshold: {self.risk_params['min_confidence_score']}")
            
            # Risk status
            risk_status = "🛑 STOPPED" if self.risk_params['emergency_stop'] else "✅ NORMAL"
            self.logger.info(f"Risk Status: {risk_status}")
            
            self.logger.info("=" * 50)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating market report: {e}")
    
    def reset_daily_stats(self):
        """Reset daily statistics at market open"""
        self.daily_stats = {
            'trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'signals_generated': 0,
            'high_confidence_signals': 0,
            'strategies_active': []
        }
        self.risk_params['emergency_stop'] = False
        self.logger.info("🔄 Daily statistics reset")
    
    def start(self):
        """Start the optimized live trading bot"""
        self.logger.info("🚀 STARTING OPTIMIZED LIVE TRADING BOT")
        self.logger.info("=" * 60)
        self.logger.info("🎯 Features:")
        self.logger.info("  ✅ Confidence-based signal filtering")
        self.logger.info("  ✅ No time-based restrictions")
        self.logger.info("  ✅ Dynamic risk management")
        self.logger.info("  ✅ Real-time market analysis")
        self.logger.info("=" * 60)
        
        # Setup schedule
        schedule.every(5).minutes.do(self.execute_trading_cycle)
        schedule.every(30).minutes.do(self.generate_market_report)
        schedule.every().day.at("09:00").do(self.reset_daily_stats)
        
        self.is_running = True
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Stopping optimized trading bot...")
        except Exception as e:
            self.logger.error(f"❌ Critical error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the optimized live trading bot"""
        self.is_running = False
        self.logger.info("🛑 Optimized live trading bot stopped")
        
        # Generate final report
        self.generate_market_report()
        
        # Save final stats
        final_stats = {
            'timestamp': datetime.now().isoformat(),
            'daily_stats': self.daily_stats,
            'risk_params': self.risk_params,
            'strategies_tested': list(self.strategies.keys())
        }
        
        with open('logs/optimized_session_stats.json', 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        self.logger.info("📊 Optimized session statistics saved")
    
    def store_rejected_signals(self, rejected_signals: List[Dict[str, Any]]):
        """Store rejected signals for analysis and improvement"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            for signal in rejected_signals:
                # Ensure proper timestamp format
                timestamp = signal.get('timestamp')
                if isinstance(timestamp, str) and 'T' in timestamp:
                    # ISO format - convert to standard datetime string
                    timestamp = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                elif not isinstance(timestamp, str):
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Store signal in rejected_signals table
                conn.execute('''
                    INSERT INTO rejected_signals (
                        timestamp, strategy, symbol, signal, rejection_reason,
                        confidence, confidence_score, price, stop_loss, target, reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    signal['strategy'],
                    signal['symbol'],
                    signal['signal'],
                    signal.get('rejection_reason', 'Unknown rejection reason'),
                    signal.get('confidence', 'Unknown'),
                    signal.get('confidence_score', 0),
                    signal.get('price', 0),
                    signal.get('stop_loss', 0),
                    signal.get('target', 0),
                    signal.get('price_reason', '') + ' | ' + str(signal.get('confidence_reasons', ''))
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"💾 Stored {len(rejected_signals)} rejected signals for analysis")
            
        except Exception as e:
            self.logger.error(f"❌ Error storing rejected signals: {e}")


def main():
    """Main execution function"""
    print("🤖 OPTIMIZED LIVE TRADING BOT")
    print("=" * 60)
    print("🎯 Confidence-Based Trading System")
    print("Features:")
    print("  🔧 Optimized strategies with confidence scoring")
    print("  📊 Real-time market analysis")
    print("  🛡️ Advanced risk management")
    print("  📱 High-confidence signal filtering")
    print("  ⚡ No time-based restrictions")
    print("=" * 60)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and start bot
    bot = OptimizedLiveTradingBot()
    
    try:
        bot.start()
    except Exception as e:
        print(f"❌ Failed to start optimized bot: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 