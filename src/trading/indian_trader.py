#!/usr/bin/env python3
"""
Enhanced Indian Trading System with Signal Execution Tracking
FIXED VERSION: Removed cooldown logic, signal limiting, and optimized WebSocket
"""

import sys
import os
import time
import signal
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.core.enhanced_strategy_engine import EnhancedStrategyEngine
from src.adapters.market_factory import MarketFactory
from src.adapters.market_interface import MarketType
from src.models.consolidated_database import ConsolidatedTradingDatabase, initialize_connection_pools
from src.core.error_handler import error_handler, handle_errors
from src.core.technical_indicators import calculate_all_indicators, validate_indicators
from risk_config import risk_config

# Configure logging with absolute path
import logging

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(project_root, "logs/indian/indian_trading.log")),
        logging.StreamHandler()
    ],
    force=True
)

# Suppress urllib3 debug logs
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
# Suppress fyers API debug logs
logging.getLogger("src.api.fyers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class EnhancedIndianTrader:
    def __init__(self, capital: float = 50000):
        self.capital = capital
        self.tz = pytz.timezone('Asia/Kolkata')
        
        # Risk management from config (COOLDOWN REMOVED)
        self.risk_config = risk_config
        self.max_positions_per_symbol = risk_config.get("max_positions_per_symbol", 3)
        self.max_total_positions = risk_config.get("max_total_positions", 15)
        self.daily_loss_limit = 0.15  # 15%
        self.emergency_stop_loss = risk_config.get_emergency_stop_loss()
        
        # Trading state (LAST_TRADE_TIME REMOVED)
        self.open_trades = {}
        self.daily_pnl = 0.0
        self.start_time = datetime.now(self.tz)
        
        # Initialize systems
        self._initialize_systems()
        
        # Control flags
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_systems(self):
        """Initialize all trading systems."""
        try:
            # Initialize connection pools
            initialize_connection_pools()
            
            # Initialize database
            self.db = ConsolidatedTradingDatabase("data/indian_trading.db")
            
            # Initialize symbols
            self.symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX", "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ"]
            
            # Initialize market data provider
            self.data_provider = MarketFactory.create_market(MarketType.INDIAN_STOCKS)
            
            
            # Load existing open trades from database
            self._load_open_trades_from_db()
            
            logger.info("✅ All systems initialized successfully")
            logger.info("📡 WebSocket disabled for optimal performance - using REST API for real-time data")
            
            # Initialize strategy engine
            self.strategy_engine = EnhancedStrategyEngine(self.symbols)
            
            logger.info("✅ All systems initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize systems: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"🛑 Received signal {signum}, shutting down gracefully...")
        self._stop_event.set()
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            return self.data_provider.get_current_price(symbol)
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[object]:
        """Get historical data for a symbol."""
        try:
            end_date = datetime.now(self.tz)
            start_date = end_date - timedelta(days=days)
            return self.data_provider.get_historical_data(symbol, start_date, end_date, "1h")
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def _check_risk_limits(self, symbol: str, signal: Dict) -> tuple[bool, str]:
        """Check risk management limits and return (allowed, reason). COOLDOWN LOGIC REMOVED."""
        try:
            # Check if risk management is enabled
            if not self.risk_config.is_risk_management_enabled():
                return True, "Risk management disabled"
            
            # Check daily loss limit
            if self.daily_pnl <= -self.capital * self.daily_loss_limit:
                return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"
            
            # Check emergency stop loss
            if self.daily_pnl <= -self.capital * self.emergency_stop_loss:
                return False, f"Emergency stop loss triggered: {self.daily_pnl:.2f}"
            
            # Check position limits per symbol
            symbol_positions = sum(1 for trade in self.open_trades.values() if trade.symbol == symbol)
            if symbol_positions >= self.max_positions_per_symbol:
                return False, f"Max positions per symbol reached: {symbol_positions}/{self.max_positions_per_symbol}"
            
            # Check total position limit
            if len(self.open_trades) >= self.max_total_positions:
                return False, f"Max total positions reached: {len(self.open_trades)}/{self.max_total_positions}"
            
            # Check capital availability
            available_capital = self.capital - sum(trade.position_size for trade in self.open_trades.values())
            if available_capital <= 0:
                return False, f"Insufficient capital: {available_capital:.2f}"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            logger.error(f"Error in risk checks: {e}")
            return False, f"Risk check error: {e}"

    def _get_rejection_reason(self, signal: Dict, entry_price: float) -> str:
        """Get the reason why a signal was rejected. COOLDOWN LOGIC REMOVED."""
        try:
            symbol = signal["symbol"]
            
            # Check if risk management is enabled
            if not self.risk_config.is_risk_management_enabled():
                return "Risk management disabled - should not be rejected"
            
            # Check position limits
            symbol_positions = sum(1 for trade in self.open_trades.values() if trade.symbol == symbol)
            if symbol_positions >= self.max_positions_per_symbol:
                return f"Max positions per symbol reached: {symbol_positions}/{self.max_positions_per_symbol}"
            
            # Check risk limits
            allowed, reason = self._check_risk_limits(symbol, signal)
            if not allowed:
                return reason
            
            return "Unknown rejection reason"
        except Exception as e:
            return f"Error determining rejection reason: {e}"
    
    def _open_trade(self, signal: Dict, entry_price: float, timestamp: datetime) -> Optional[str]:
        """Open a new trade with comprehensive risk checks. LAST_TRADE_TIME TRACKING REMOVED."""
        try:
            symbol = signal['symbol']
            
            # Check risk limits
            allowed, reason = self._check_risk_limits(symbol, signal)
            if not allowed:
                # Update signal as rejected with reason
                if signal.get("signal_id"):
                    self.db.update_signal_execution_status(signal["signal_id"], False, reason)
                logger.debug(f"❌ Trade rejected for {symbol}: {reason}")
                return None
            
            # Calculate position size
            position_size = min(self.capital * 0.1, 5000)  # 10% of capital or max 5000
            
            # Calculate stop loss and take profit
            if signal['signal'] == 'BUY CALL':
                stop_loss_price = entry_price * (1 - 0.03)  # 3% stop loss
                take_profit_price = entry_price * (1 + 0.05)  # 5% take profit
            else:  # BUY PUT
                stop_loss_price = entry_price * (1 + 0.03)  # 3% stop loss
                take_profit_price = entry_price * (1 - 0.05)  # 5% take profit
            
            # Generate trade ID
            trade_id = f"{symbol}_{int(timestamp.timestamp())}"
            
            # Create trade object
            trade = type('Trade', (), {
                'trade_id': trade_id,
                'symbol': symbol,
                'strategy': signal['strategy'],
                'signal': signal['signal'],
                'entry_price': entry_price,
                'position_size': position_size,
                'entry_time': timestamp,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'status': 'open'
            })()
            
            # Store trade
            self.open_trades[trade_id] = trade
            
            # Save to database
            self.db.save_open_trade(trade_id, "indian", symbol, signal["strategy"], signal["signal"], 
                                   entry_price, position_size, timestamp, 
                                   stop_loss_price, take_profit_price)
            
            # Update signal as executed
            if signal.get("signal_id"):
                self.db.update_signal_execution_status(signal["signal_id"], True, "Trade executed successfully")
            
            logger.info(f"✅ Opened trade: {trade_id} - {symbol} {signal['signal']} @ {entry_price:.2f}")
            return trade_id
            
        except Exception as e:
            logger.error(f"Error opening trade: {e}")
            # Update signal as rejected with error reason
            if signal.get("signal_id"):
                self.db.update_signal_execution_status(signal["signal_id"], False, f"Trade opening error: {e}")
            return None
    
    def _process_signals(self):
        """Process trading signals with execution tracking. SIGNAL LIMITING REMOVED."""
        try:
            # Get current prices using batch request
            try:
                current_prices = self.data_provider.get_current_prices_batch(self.symbols)
                # Filter out None values
                current_prices = {symbol: price for symbol, price in current_prices.items() if price is not None}
                
                if not current_prices:
                    logger.warning("No current prices available")
                    return
                    
                logger.debug(f"✅ Got prices for {len(current_prices)} symbols: {list(current_prices.keys())}")
                
            except Exception as e:
                logger.error(f"Error getting batch prices: {e}")
                return
            
            # Get historical data and calculate indicators
            historical_data = {}
            for symbol in self.symbols:
                data = self.get_historical_data(symbol, 30)
                if data is not None and len(data) > 50:
                    # Calculate technical indicators
                    data_with_indicators = calculate_all_indicators(data)
                    
                    # Validate indicators
                    if validate_indicators(data_with_indicators):
                        historical_data[symbol] = data_with_indicators
                        logger.debug(f"✅ Calculated indicators for {symbol}")
                    else:
                        logger.warning(f"⚠️ Invalid indicators for {symbol}")
                else:
                    logger.warning(f"⚠️ Insufficient data for {symbol}: {len(data) if data is not None else 0} candles")
            
            if not historical_data:
                logger.warning("No historical data with valid indicators available")
                return
            
            # Generate signals
            signals = self.strategy_engine.generate_signals_for_all_symbols(historical_data, current_prices)
            
            # Save all signals to database and track their IDs
            for signal in signals:
                try:
                    signal_id = self.db.save_signal(
                        market="indian",
                        symbol=signal["symbol"],
                        strategy=signal["strategy"],
                        signal=signal["signal"],
                        confidence=signal["confidence"],
                        price=signal["price"],
                        timestamp=signal["timestamp"],
                        timeframe=signal["timeframe"],
                        strength=signal.get("strength"),
                        confirmed=signal.get("confirmed", False)
                    )
                    # Store signal ID for execution tracking
                    signal["signal_id"] = signal_id
                except Exception as e:
                    logger.error(f"Failed to save signal: {e}")
                    signal["signal_id"] = None
            
            # Process ALL signals (signal limiting removed)
            logger.info(f"🎯 Processing {len(signals)} signals (no artificial limits)")
            for signal in signals:
                if self._stop_event.is_set():
                    break
                
                self._process_signal(signal, current_prices)
                
        except Exception as e:
            error_handler.handle_error(e, {'context': 'process_signals'})
    
    def _process_signal(self, signal: Dict, current_prices: Dict[str, float]):
        """Process a single signal with execution tracking."""
        try:
            symbol = signal['symbol']
            if symbol not in current_prices:
                # Update signal as rejected
                if signal.get("signal_id"):
                    self.db.update_signal_execution_status(signal["signal_id"], False, "No current price available")
                return
            
            entry_price = current_prices[symbol]
            
            # Open trade (with all risk checks)
            trade_id = self._open_trade(signal, entry_price, datetime.now(self.tz))
            if trade_id:
                logger.info(f"✅ Signal executed: {signal['strategy']} {signal['signal']} for {symbol} @ {entry_price:.2f}")
            else:
                logger.debug(f"❌ Signal rejected: {signal['strategy']} {signal['signal']} for {symbol}")
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'process_signal', 'signal': signal})
    
    
    def _update_open_trades(self):
        """Update open trades with current prices and check exit conditions."""
        if not self.open_trades:
            return
        
        try:
            # Get current prices for all open trade symbols
            symbols = list(self.open_trades.keys())
            current_prices = self.data_provider.get_current_prices_batch(symbols)
            
            if not current_prices:
                logger.warning("No current prices available for trade monitoring")
                return
            
            trades_to_close = []
            
            for trade_id, trade in self.open_trades.items():
                symbol = trade.symbol
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                entry_price = trade.entry_price
                position_size = trade.quantity
                
                # Calculate P&L
                if trade.signal == 'BUY CALL':
                    pnl = (current_price - entry_price) * position_size
                else:  # BUY PUT
                    pnl = (entry_price - current_price) * position_size
                
                # Check exit conditions
                exit_reason = None
                exit_price = current_price
                
                # Stop-loss: 2% loss
                if pnl <= -abs(entry_price * position_size * 0.02):
                    exit_reason = "STOP_LOSS"
                
                # Take-profit: 3% gain
                elif pnl >= abs(entry_price * position_size * 0.03):
                    exit_reason = "TAKE_PROFIT"
                
                # Time-based exit: 1 hour
                elif (datetime.now(self.tz) - trade.entry_time).total_seconds() > 3600:
                    exit_reason = "TIME_EXIT"
                
                if exit_reason:
                    trades_to_close.append((trade_id, trade, exit_price, exit_reason, pnl))
            
            # Close trades that met exit conditions
            for trade_id, trade, exit_price, exit_reason, pnl in trades_to_close:
                self._close_trade(trade_id, exit_price, exit_reason, pnl)
                
        except Exception as e:
            logger.error(f"Error updating open trades: {e}")

    def _check_exit_conditions(self, trade, current_price: float, current_time: datetime) -> Optional[str]:
        """Check if trade should be closed."""
        try:
            # Time-based exit (24 hours)
            if (current_time - trade.entry_time).total_seconds() > 86400:
                return "TIME_EXIT"
            
            # Stop loss
            if trade.signal == 'BUY CALL' and current_price <= trade.stop_loss_price:
                return "STOP_LOSS"
            elif trade.signal == 'BUY PUT' and current_price >= trade.stop_loss_price:
                return "STOP_LOSS"
            
            # Take profit
            if trade.signal == 'BUY CALL' and current_price >= trade.take_profit_price:
                return "TARGET_HIT"
            elif trade.signal == 'BUY PUT' and current_price <= trade.take_profit_price:
                return "TARGET_HIT"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def _close_trade(self, trade_id: str, exit_price: float, exit_reason: str):
        """Close a trade and calculate P&L."""
        try:
            trade = self.open_trades.pop(trade_id, None)
            if not trade:
                return
            
            # Calculate P&L
            if trade.signal == 'BUY CALL':
                pnl = (exit_price - trade.entry_price) * (trade.position_size / trade.entry_price)
            else:  # BUY PUT
                pnl = (trade.entry_price - exit_price) * (trade.position_size / trade.entry_price)
            
            # Update daily P&L
            self.daily_pnl += pnl
            
            # Save to database
            self.db.close_trade(trade_id, "indian", exit_price, datetime.now(self.tz), exit_reason, pnl)
            
            logger.info(f"🔒 Closed trade: {trade_id} - {trade.symbol} @ {exit_price:.2f} | P&L: {pnl:.2f} | Reason: {exit_reason}")
            
        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
    
    def run(self):
        """Main trading loop."""
        logger.info(f"🚀 Starting Enhanced Indian Trader with ₹{self.capital:,.2f} capital")
        logger.info("🔧 OPTIMIZATIONS: Cooldown removed, Signal limiting removed, WebSocket disabled")
        
        try:
            while not self._stop_event.is_set():
                # Process signals
                self._process_signals()
                
                # Update open trades
                self._update_open_trades()
                
                # Wait before next cycle
                time.sleep(10)  # 10-second cycle
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading error: {e}")
        finally:
            self._stop_event.set()
            logger.info("🏁 Trading system shutdown complete")

    def _load_open_trades_from_db(self):
        """Load existing open trades from database."""
        try:
            open_trades = self.db.get_open_trades("indian")
            for trade in open_trades:
                trade_id = trade[1]  # trade_id is at index 1
                symbol = trade[3]    # symbol is at index 3
                strategy = trade[4]  # strategy is at index 4
                signal = trade[5]    # signal is at index 5
                entry_price = trade[6]  # entry_price is at index 6
                quantity = trade[7]  # quantity is at index 7
                entry_time = datetime.fromisoformat(trade[8])  # entry_time is at index 8
                stop_loss_price = trade[9]  # stop_loss_price is at index 9
                take_profit_price = trade[10]  # take_profit_price is at index 10
                
                # Create trade object
                trade_obj = type("Trade", (), {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "strategy": strategy,
                    "signal": signal,
                    "entry_price": entry_price,
                    "position_size": quantity,
                    "entry_time": entry_time,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                    "status": "open"
                })()
                
                self.open_trades[trade_id] = trade_obj
            
            logger.info(f"📊 Loaded {len(self.open_trades)} open trades from database")
            
        except Exception as e:
            logger.error(f"Error loading open trades from database: {e}")
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Indian Trading System')
    parser.add_argument('--market', default='indian', help='Market to trade')
    parser.add_argument('--capital', type=float, default=50000, help='Starting capital')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run trader
    trader = EnhancedIndianTrader(capital=args.capital)
    trader.run()

if __name__ == "__main__":
    main()

