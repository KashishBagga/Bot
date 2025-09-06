#!/usr/bin/env python3
"""
Replay/Backfill Engine
Feed historical tick/option data into the system at 10x speed to test performance and catch edge cases
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import pandas as pd
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ReplayConfig:
    """Configuration for replay/backfill"""
    speed_multiplier: float = 10.0  # 10x speed
    start_date: datetime = None
    end_date: datetime = None
    symbols: List[str] = None
    strategies: List[str] = None
    enable_trading: bool = False  # Set to True to actually execute trades
    max_trades: int = 1000
    stop_on_error: bool = True

@dataclass
class ReplayResult:
    """Result of replay/backfill operation"""
    total_events: int
    total_trades: int
    total_pnl: float
    execution_time: float
    errors: List[str]
    performance_metrics: Dict
    edge_cases_found: List[str]

class ReplayBackfillEngine:
    """Engine for replaying historical data at high speed"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.replay_config = ReplayConfig()
        self.replay_results = []
        self.is_replaying = False
        self.replay_thread = None
        self.performance_metrics = {}
        
    def configure_replay(self, config: ReplayConfig):
        """Configure replay parameters"""
        self.replay_config = config
        logger.info(f"🎬 Replay configured: {config.speed_multiplier}x speed, "
                   f"Trading: {'Enabled' if config.enable_trading else 'Disabled'}")
    
    def start_replay(self, historical_data: Dict[str, pd.DataFrame], config: ReplayConfig = None):
        """Start replaying historical data"""
        if self.is_replaying:
            logger.warning("⚠️ Replay already in progress")
            return
        
        if config:
            self.configure_replay(config)
        
        logger.info("🎬 Starting historical data replay...")
        self.is_replaying = True
        
        # Start replay in separate thread
        self.replay_thread = threading.Thread(
            target=self._replay_historical_data, 
            args=(historical_data,), 
            daemon=True
        )
        self.replay_thread.start()
        
        logger.info("✅ Replay started")
    
    def stop_replay(self):
        """Stop replay operation"""
        if not self.is_replaying:
            return
        
        logger.info("🛑 Stopping replay...")
        self.is_replaying = False
        
        if self.replay_thread:
            self.replay_thread.join(timeout=10)
        
        logger.info("✅ Replay stopped")
    
    def _replay_historical_data(self, historical_data: Dict[str, pd.DataFrame]):
        """Replay historical data at configured speed"""
        try:
            start_time = time.time()
            total_events = 0
            total_trades = 0
            total_pnl = 0.0
            errors = []
            edge_cases = []
            
            logger.info(f"🎬 Replaying {len(historical_data)} symbols at {self.replay_config.speed_multiplier}x speed")
            
            # Process each symbol's data
            for symbol, data in historical_data.items():
                if not self.is_replaying:
                    break
                
                logger.info(f"📊 Processing {symbol}: {len(data)} data points")
                
                # Sort data by timestamp
                data = data.sort_values('timestamp').reset_index(drop=True)
                
                # Process data points
                for i, row in data.iterrows():
                    if not self.is_replaying:
                        break
                    
                    try:
                        # Simulate real-time data processing
                        self._process_historical_data_point(symbol, row)
                        total_events += 1
                        
                        # Check for edge cases
                        edge_case = self._check_edge_cases(symbol, row)
                        if edge_case:
                            edge_cases.append(edge_case)
                        
                        # Simulate time passage
                        if i < len(data) - 1:
                            next_timestamp = data.iloc[i + 1]['timestamp']
                            time_diff = (next_timestamp - row['timestamp']).total_seconds()
                            sleep_time = time_diff / self.replay_config.speed_multiplier
                            
                            if sleep_time > 0:
                                time.sleep(min(sleep_time, 0.1))  # Cap at 100ms
                        
                        # Check if we should execute trades
                        if self.replay_config.enable_trading and total_events % 100 == 0:
                            trades_executed = self._execute_replay_trades()
                            total_trades += trades_executed
                        
                        # Log progress
                        if total_events % 1000 == 0:
                            logger.info(f"📊 Processed {total_events} events, {total_trades} trades")
                        
                    except Exception as e:
                        error_msg = f"Error processing {symbol} at {row['timestamp']}: {e}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        
                        if self.replay_config.stop_on_error:
                            raise e
            
            # Calculate final results
            execution_time = time.time() - start_time
            total_pnl = self._calculate_total_pnl()
            
            # Create replay result
            result = ReplayResult(
                total_events=total_events,
                total_trades=total_trades,
                total_pnl=total_pnl,
                execution_time=execution_time,
                errors=errors,
                performance_metrics=self._get_performance_metrics(),
                edge_cases_found=edge_cases
            )
            
            self.replay_results.append(result)
            
            # Generate replay report
            self._generate_replay_report(result)
            
        except Exception as e:
            logger.error(f"❌ Error in replay: {e}")
            self.is_replaying = False
    
    def _process_historical_data_point(self, symbol: str, row: pd.Series):
        """Process a single historical data point"""
        try:
            # Update price cache with historical data
            self.trading_system.price_cache[symbol] = (row['close'], time.time())
            
            # Generate signals if market is open
            if self._is_market_open_historical(row['timestamp']):
                # Get recent data for signal generation
                recent_data = self._get_recent_data_for_symbol(symbol, row['timestamp'])
                if recent_data is not None and not recent_data.empty:
                    # Generate signals
                    signals = self.trading_system._generate_signals(recent_data)
                    
                    # Process signals
                    for signal in signals:
                        self._process_replay_signal(signal, row['close'])
            
        except Exception as e:
            logger.error(f"Error processing data point for {symbol}: {e}")
    
    def _get_recent_data_for_symbol(self, symbol: str, timestamp: datetime) -> Optional[pd.DataFrame]:
        """Get recent data for signal generation"""
        try:
            # In a real implementation, you'd fetch recent historical data
            # For now, create a simple DataFrame
            data = pd.DataFrame({
                'timestamp': [timestamp - timedelta(minutes=i) for i in range(100, 0, -1)],
                'open': [100.0] * 100,
                'high': [101.0] * 100,
                'low': [99.0] * 100,
                'close': [100.0] * 100,
                'volume': [1000] * 100
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting recent data for {symbol}: {e}")
            return None
    
    def _process_replay_signal(self, signal: Dict, current_price: float):
        """Process a signal during replay"""
        try:
            # Log signal
            logger.debug(f"📡 Replay Signal: {signal['strategy']} {signal['signal']} at ₹{current_price:.2f}")
            
            # If trading is enabled, execute the signal
            if self.replay_config.enable_trading:
                # Select option contract
                option_contract = self.trading_system._select_option_contract(signal, current_price)
                if option_contract:
                    # Execute trade
                    trade_id = self.trading_system._open_paper_trade(
                        signal, option_contract, option_contract.last, 
                        self.trading_system.now_kolkata()
                    )
                    
                    if trade_id:
                        logger.info(f"✅ Replay Trade Executed: {trade_id[:8]}...")
            
        except Exception as e:
            logger.error(f"Error processing replay signal: {e}")
    
    def _execute_replay_trades(self) -> int:
        """Execute trades during replay"""
        try:
            trades_executed = 0
            
            # Check for trade exits
            current_prices = {}
            for trade in self.trading_system.open_trades.values():
                current_prices[trade.contract_symbol] = trade.entry_price  # Simplified
            
            closed_trades = self.trading_system._check_trade_exits(
                current_prices, self.trading_system.now_kolkata()
            )
            
            trades_executed = len(closed_trades)
            
            return trades_executed
            
        except Exception as e:
            logger.error(f"Error executing replay trades: {e}")
            return 0
    
    def _check_edge_cases(self, symbol: str, row: pd.Series) -> Optional[str]:
        """Check for edge cases in historical data"""
        try:
            edge_cases = []
            
            # Check for extreme price movements
            if hasattr(row, 'high') and hasattr(row, 'low'):
                price_range = (row['high'] - row['low']) / row['close']
                if price_range > 0.1:  # 10% price range
                    edge_cases.append(f"Extreme price movement: {price_range:.1%} range")
            
            # Check for zero volume
            if hasattr(row, 'volume') and row['volume'] == 0:
                edge_cases.append("Zero volume detected")
            
            # Check for missing data
            if pd.isna(row['close']):
                edge_cases.append("Missing close price")
            
            # Check for negative prices
            if row['close'] <= 0:
                edge_cases.append("Negative or zero price")
            
            return edge_cases[0] if edge_cases else None
            
        except Exception as e:
            logger.error(f"Error checking edge cases: {e}")
            return None
    
    def _is_market_open_historical(self, timestamp: datetime) -> bool:
        """Check if market was open at historical timestamp"""
        try:
            # Simple market hours check (9:15 AM - 3:30 PM IST, weekdays)
            if timestamp.weekday() >= 5:  # Weekend
                return False
            
            market_start = timestamp.replace(hour=9, minute=15, second=0, microsecond=0)
            market_end = timestamp.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_start <= timestamp <= market_end
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
    
    def _calculate_total_pnl(self) -> float:
        """Calculate total PnL from replay"""
        try:
            total_pnl = 0.0
            
            # Sum PnL from closed trades
            for trade in self.trading_system.closed_trades:
                if trade.pnl is not None:
                    total_pnl += trade.pnl
            
            return total_pnl
            
        except Exception as e:
            logger.error(f"Error calculating total PnL: {e}")
            return 0.0
    
    def _get_performance_metrics(self) -> Dict:
        """Get performance metrics during replay"""
        try:
            return {
                'signals_generated': self.trading_system.total_signals_generated,
                'trades_executed': self.trading_system.total_trades_executed,
                'trades_closed': self.trading_system.total_trades_closed,
                'winning_trades': self.trading_system.winning_trades,
                'losing_trades': self.trading_system.losing_trades,
                'win_rate': (self.trading_system.winning_trades / max(self.trading_system.total_trades_closed, 1)) * 100,
                'max_drawdown': self.trading_system.max_drawdown,
                'api_calls': self.trading_system.performance_stats.get('api_calls_made', 0),
                'api_failures': self.trading_system.performance_stats.get('api_failures', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _generate_replay_report(self, result: ReplayResult):
        """Generate comprehensive replay report"""
        try:
            logger.info("=" * 80)
            logger.info("🎬 REPLAY/BACKFILL REPORT")
            logger.info("=" * 80)
            logger.info(f"📊 Total Events Processed: {result.total_events:,}")
            logger.info(f"📈 Total Trades Executed: {result.total_trades}")
            logger.info(f"💰 Total PnL: ₹{result.total_pnl:+,.2f}")
            logger.info(f"⏱️ Execution Time: {result.execution_time:.2f} seconds")
            logger.info(f"🚀 Effective Speed: {result.total_events / result.execution_time:.1f} events/second")
            logger.info(f"❌ Errors: {len(result.errors)}")
            logger.info(f"🔍 Edge Cases Found: {len(result.edge_cases_found)}")
            
            # Performance metrics
            if result.performance_metrics:
                logger.info("\n📊 PERFORMANCE METRICS:")
                for metric, value in result.performance_metrics.items():
                    logger.info(f"   {metric}: {value}")
            
            # Edge cases
            if result.edge_cases_found:
                logger.info("\n🔍 EDGE CASES FOUND:")
                for edge_case in result.edge_cases_found[:10]:  # Show first 10
                    logger.info(f"   - {edge_case}")
                if len(result.edge_cases_found) > 10:
                    logger.info(f"   ... and {len(result.edge_cases_found) - 10} more")
            
            # Errors
            if result.errors:
                logger.info("\n❌ ERRORS:")
                for error in result.errors[:5]:  # Show first 5
                    logger.info(f"   - {error}")
                if len(result.errors) > 5:
                    logger.info(f"   ... and {len(result.errors) - 5} more")
            
            # Save detailed report
            filename = f"replay_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
            
            logger.info(f"\n📄 Detailed report saved to: {filename}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error generating replay report: {e}")
    
    def get_replay_status(self) -> Dict:
        """Get current replay status"""
        return {
            'is_replaying': self.is_replaying,
            'config': asdict(self.replay_config),
            'results_count': len(self.replay_results),
            'last_result': asdict(self.replay_results[-1]) if self.replay_results else None
        }
    
    def create_sample_historical_data(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """Create sample historical data for testing"""
        try:
            historical_data = {}
            
            for symbol in symbols:
                # Create sample data
                dates = pd.date_range(
                    start=datetime.now() - timedelta(days=days),
                    end=datetime.now(),
                    freq='5T'  # 5-minute intervals
                )
                
                # Generate realistic price data
                base_price = 100.0
                prices = []
                current_price = base_price
                
                for i, date in enumerate(dates):
                    # Random walk with some trend
                    change = (i % 100 - 50) * 0.01 + (i % 20 - 10) * 0.005
                    current_price += change
                    current_price = max(current_price, base_price * 0.8)  # Floor
                    current_price = min(current_price, base_price * 1.2)  # Ceiling
                    
                    prices.append({
                        'timestamp': date,
                        'open': current_price,
                        'high': current_price * 1.01,
                        'low': current_price * 0.99,
                        'close': current_price,
                        'volume': 1000 + (i % 100) * 10
                    })
                
                historical_data[symbol] = pd.DataFrame(prices)
                logger.info(f"📊 Created {len(prices)} data points for {symbol}")
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error creating sample historical data: {e}")
            return {}

def main():
    """Main function to run replay/backfill engine"""
    try:
        from live_paper_trading import LivePaperTradingSystem
        
        # Initialize trading system
        logger.info("🚀 Initializing trading system for replay/backfill...")
        trading_system = LivePaperTradingSystem(initial_capital=100000)
        
        # Initialize replay engine
        replay_engine = ReplayBackfillEngine(trading_system)
        
        # Configure replay
        config = ReplayConfig(
            speed_multiplier=10.0,
            enable_trading=False,  # Set to True to execute trades
            max_trades=100,
            stop_on_error=False
        )
        replay_engine.configure_replay(config)
        
        # Create sample historical data
        symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
        historical_data = replay_engine.create_sample_historical_data(symbols, days=7)
        
        if historical_data:
            # Start replay
            replay_engine.start_replay(historical_data)
            
            logger.info("🎬 Replay started - press Ctrl+C to stop")
            
            try:
                while replay_engine.is_replaying:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Stopping replay...")
                replay_engine.stop_replay()
                logger.info("✅ Replay stopped")
        else:
            logger.error("❌ Failed to create historical data")
        
    except Exception as e:
        logger.error(f"❌ Replay/backfill failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
