#!/usr/bin/env python3
"""
Enhanced Indian Trader with Enhanced Database Integration
"""

import sys
import os
import time
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/enhanced_indian_trading.log'),
        logging.StreamHandler()
    ]
)

# Suppress urllib3 debug logs
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('src.api.fyers').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class EnhancedIndianTrader:
    """Enhanced Indian trader with comprehensive database integration"""
    
    def __init__(self):
        self.symbols = [
            "NSE:NIFTY50-INDEX",
            "NSE:NIFTYBANK-INDEX", 
            "NSE:FINNIFTY-INDEX"
        ]
        self.is_running = False
        self.start_time = None
        
        # Initialize systems
        self._initialize_systems()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_systems(self):
        """Initialize all trading systems"""
        try:
            # Import required modules
            from src.api.fyers import FyersClient
            from src.core.enhanced_strategy_engine import EnhancedStrategyEngine
            from src.core.enhanced_real_time_manager import EnhancedRealTimeDataManager
            from src.models.enhanced_database import EnhancedTradingDatabase
            from src.core.risk_manager import RiskManager
            from src.monitoring.system_monitor import SystemMonitor
            
            # Initialize components
            self.data_provider = FyersClient()
            self.strategy_engine = EnhancedStrategyEngine(self.symbols)
            self.real_time_data = EnhancedRealTimeDataManager(self.data_provider, self.symbols)
            self.database = EnhancedTradingDatabase("data/enhanced_trading.db")
            self.risk_manager = RiskManager()
            self.system_monitor = SystemMonitor()
            
            logger.info("✅ All systems initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize systems: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self.stop_trading()
    
    def start_trading(self):
        """Start the enhanced trading system"""
        if self.is_running:
            logger.warning("⚠️ Trading system is already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info("🚀 Enhanced Indian Trading System Started")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"🗄️ Database: Enhanced structure with market separation")
        logger.info(f"📡 Data Source: WebSocket + REST API")
        logger.info(f"🛡️ Risk Management: Active")
        logger.info(f"📈 System Monitoring: Active")
        
        try:
            # Start WebSocket
            self.real_time_data.start_websocket()
            logger.info("📡 WebSocket started")
            
            # Main trading loop
            self._trading_loop()
            
        except Exception as e:
            logger.error(f"❌ Trading loop error: {e}")
        finally:
            self.stop_trading()
    
    def _trading_loop(self):
        """Main trading loop with enhanced features"""
        cycle_count = 0
        
        while self.is_running:
            try:
                cycle_count += 1
                logger.info(f"🔄 Trading Cycle #{cycle_count}")
                
                # Get real-time data
                current_prices = self.real_time_data.get_current_prices()
                logger.info(f"📊 Current prices: {len(current_prices)} symbols")
                
                # Get historical data for all symbols
                historical_data = {}
                for symbol in self.symbols:
                    hist_data = self.real_time_data.get_historical_data(symbol, days=30)
                    if hist_data is not None:
                        historical_data[symbol] = hist_data
                
                if len(historical_data) > 0:
                    logger.info(f"📈 Historical data: {len(historical_data)} symbols")
                    
                    # Generate signals for all symbols
                    all_signals = self.strategy_engine.generate_signals_for_all_symbols(
                        historical_data, current_prices
                    )
                    
                    if all_signals:
                        logger.info(f"🎯 Generated {len(all_signals)} signals")
                        
                        # Process signals with enhanced database
                        self._process_signals_enhanced(all_signals, current_prices)
                    else:
                        logger.info("⏳ No signals generated this cycle")
                else:
                    logger.warning("⚠️ No historical data available")
                
                # Update system health
                self.system_monitor.check_system_health()
                
                # Sleep between cycles
                time.sleep(5)  # 5-second cycles for enhanced responsiveness
                
            except Exception as e:
                logger.error(f"❌ Trading cycle error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _process_signals_enhanced(self, signals: List[Dict], current_prices: Dict[str, float]):
        """Process signals with enhanced database integration"""
        for signal in signals:
            try:
                symbol = signal['symbol']
                strategy = signal['strategy']
                signal_type = signal['signal']
                confidence = signal['confidence']
                price = current_prices.get(symbol, 0.0)
                
                # Generate unique signal ID
                signal_id = f"{strategy}_{symbol}_{int(time.time())}_{signal_type}"
                
                # Risk management check
                if not self.risk_manager.should_execute_signal(signal, current_prices):
                    # Save rejected signal
                    self.database.save_rejected_signal(
                        market="indian",
                        signal_id=signal_id,
                        symbol=symbol,
                        strategy=strategy,
                        signal_type=signal_type,
                        confidence=confidence,
                        price=price,
                        timestamp=datetime.now().isoformat(),
                        rejection_reason="RISK_MANAGEMENT",
                        indicator_values=signal.get('indicator_values', {})
                    )
                    logger.info(f"🚫 Signal rejected by risk management: {symbol} {signal_type}")
                    continue
                
                # Save entry signal
                self.database.save_entry_signal(
                    market="indian",
                    signal_id=signal_id,
                    symbol=symbol,
                    strategy=strategy,
                    signal_type=signal_type,
                    confidence=confidence,
                    price=price,
                    timestamp=datetime.now().isoformat(),
                    timeframe=signal.get('timeframe', '1h'),
                    strength=signal.get('strength', 'MEDIUM'),
                    indicator_values=signal.get('indicator_values', {}),
                    market_condition=signal.get('market_condition', 'UNKNOWN'),
                    volatility=signal.get('volatility', 0.0),
                    position_size=signal.get('position_size', 100.0),
                    stop_loss_price=signal.get('stop_loss_price', 0.0),
                    take_profit_price=signal.get('take_profit_price', 0.0)
                )
                
                # Save trade
                self.database.save_trade(
                    market="indian",
                    symbol=symbol,
                    trade_id=signal_id,
                    strategy=strategy,
                    signal_type=signal_type,
                    entry_price=price,
                    quantity=signal.get('position_size', 100.0),
                    entry_time=datetime.now().isoformat(),
                    stop_loss_price=signal.get('stop_loss_price', 0.0),
                    take_profit_price=signal.get('take_profit_price', 0.0),
                    indicator_values=signal.get('indicator_values', {})
                )
                
                logger.info(f"✅ Signal executed: {symbol} {signal_type} @ {price} (Confidence: {confidence:.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Error processing signal: {e}")
    
    def stop_trading(self):
        """Stop the trading system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        try:
            # Stop WebSocket
            self.real_time_data.stop_websocket()
            logger.info("📡 WebSocket stopped")
            
            # Update daily summary
            if self.start_time:
                duration = datetime.now() - self.start_time
                self._update_daily_summary(duration)
            
            logger.info("🛑 Enhanced Indian Trading System Stopped")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
    
    def _update_daily_summary(self, duration: timedelta):
        """Update daily summary with session statistics"""
        try:
            # Get market statistics
            stats = self.database.get_market_statistics("indian")
            
            # Update daily summary
            self.database.update_daily_summary_kwargs(
                market="indian",
                date=datetime.now().strftime("%Y-%m-%d"),
                total_signals=stats.get('total_signals', 0),
                executed_signals=stats.get('executed_signals', 0),
                rejected_signals=stats.get('rejected_signals', 0),
                total_trades=stats.get('open_trades', 0) + stats.get('closed_trades', 0),
                open_trades=stats.get('open_trades', 0),
                closed_trades=stats.get('closed_trades', 0),
                total_pnl=stats.get('total_pnl', 0.0),
                realized_pnl=stats.get('total_pnl', 0.0),
                unrealized_pnl=0.0,
                win_rate=0.0,
                avg_trade_duration=duration.total_seconds() / 60.0,
                max_drawdown=0.0,
                volatility=0.02
            )
            
            logger.info("📊 Daily summary updated")
            
        except Exception as e:
            logger.error(f"❌ Failed to update daily summary: {e}")

def main():
    """Main function"""
    trader = EnhancedIndianTrader()
    
    try:
        trader.start_trading()
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        trader.stop_trading()

if __name__ == "__main__":
    main()
