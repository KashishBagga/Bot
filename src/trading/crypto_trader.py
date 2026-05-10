#!/usr/bin/env python3
"""
WORKING Optimized Modular Trading System - FIXED VERSION
Removed cooldown logic, signal limiting, and optimized performance
"""

import sys
import os
import time
import signal
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.adapters.market_factory import MarketFactory
from src.adapters.market_interface import MarketType
from src.core.enhanced_strategy_engine import EnhancedStrategyEngine
from src.models.consolidated_database import ConsolidatedTradingDatabase, initialize_connection_pools
from src.core.error_handler import error_handler
from src.core.technical_indicators import calculate_all_indicators, validate_indicators
from risk_config import risk_config
# ── New intelligence modules ──────────────────────────────────────────────────
from src.core.atr_risk_manager import ATRRiskManager
from src.core.position_sizer import PositionSizer
from src.core.expiry_blackout import ExpiryBlackoutManager

# Configure logging with absolute path
import logging

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(project_root, "logs/crypto/crypto_trading.log")),
        logging.StreamHandler()
    ],
    force=True
)

# Suppress debug logs
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@dataclass
class OptimizedTrade:
    """Optimized trade data structure."""
    trade_id: str
    symbol: str
    strategy: str
    signal: str
    entry_price: float
    position_size: float
    entry_time: datetime
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

class WorkingOptimizedModularTradingSystem:
    """WORKING Optimized modular trading system with proper risk management. FIXED VERSION."""
    
    def __init__(self, 
                 market_type: MarketType,
                 initial_capital: float = 20000.0,
                 max_risk_per_trade: float = 0.02,
                 confidence_cutoff: float = 25.0,
                 exposure_limit: float = 0.6,
                 symbols: List[str] = None,
                 verbose: bool = False):
        """Initialize the WORKING trading system."""
        
        self.market_type = market_type
        self.initial_capital = initial_capital
        self.cash = float(initial_capital)
        self.max_risk_per_trade = max_risk_per_trade
        self.confidence_cutoff = confidence_cutoff
        self.exposure_limit = exposure_limit
        self.verbose = verbose
        
        # CRITICAL FIX: Risk management parameters (COOLDOWN REMOVED)
        self.risk_config = risk_config
        self.stop_loss_percent = 0.03  # 3% stop loss
        self.take_profit_percent = 0.05  # 5% take profit
        self.max_positions_per_symbol = risk_config.get("max_positions_per_symbol", 3)
        self.max_total_positions = risk_config.get("max_total_positions", 15)
        self.daily_loss_limit = risk_config.get("daily_loss_limit", 1.0)
        self.emergency_stop_loss = risk_config.get("emergency_stop_loss", 0.20)
        
        # Initialize enhanced systems
        self._initialize_systems()
        
        # Initialize market
        self.market = MarketFactory.create_market(market_type)
        self.tz = ZoneInfo(self.market.config.timezone)
        
        # Set default symbols
        if symbols is None:
            symbols = self._get_default_symbols()
        self.symbols = symbols
        
        # Initialize data provider
        self.data_provider = self._create_data_provider()
        
        # Initialize strategy engine
        self.strategy_engine = EnhancedStrategyEngine(self.symbols)
        
        # Trading state (LAST_TRADE_TIME REMOVED)
        self.open_trades: Dict[str, OptimizedTrade] = {}
        self.daily_pnl = 0.0
        self.start_time = datetime.now(self.tz)
        
        # Control flags
        self._stop_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"✅ WORKING Optimized Trading System initialized")
        logger.info(f"🔧 OPTIMIZATIONS: Cooldown removed, Signal limiting removed")
        logger.info(f"💰 Capital: ${self.initial_capital:,.2f}")
        logger.info(f"📊 Symbols: {len(self.symbols)} symbols")
        
    def _initialize_systems(self):
        """Initialize enhanced systems."""
        try:
            # Initialize connection pools
            initialize_connection_pools()
            
            # Initialize database
            self.db = ConsolidatedTradingDatabase("data/trading.db")

            # ── New intelligence modules ───────────────────────────────
            self.atr_risk       = ATRRiskManager()
            self.position_sizer = PositionSizer(capital=self.initial_capital)
            self.blackout       = ExpiryBlackoutManager()
            self._last_hist: dict = {}  # historical data cache for ATR fallback

            logger.info("✅ Enhanced systems initialized")

            # Load existing open trades from database
            self._load_open_trades_from_db()

            logger.info("✅ Enhanced systems initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced systems: {e}")
            raise
    
    def _get_default_symbols(self) -> List[str]:
        """Get default symbols based on market type."""
        if self.market_type == MarketType.CRYPTO:
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]
        elif self.market_type == MarketType.INDIAN_STOCKS:
            return ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX", "NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ"]
        else:
            return ["BTCUSDT", "ETHUSDT"]  # Default fallback
    
    def _create_data_provider(self):
        """Create data provider for the market."""
        return MarketFactory.create_market(self.market_type)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"�� Received signal {signum}, shutting down gracefully...")
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
    
    def _check_risk_limits(self, symbol: str) -> bool:
        """
        Check if we can open a new position for the symbol. COOLDOWN LOGIC REMOVED.
        """
        try:
            current_time = datetime.now(self.tz)
            
            # Check daily loss limit
            if self.daily_pnl <= -self.initial_capital * self.daily_loss_limit:
                logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
                return False
            
            # Check emergency stop loss
            if self.daily_pnl <= -self.initial_capital * self.emergency_stop_loss:
                logger.error(f"Emergency stop loss triggered: {self.daily_pnl:.2f}")
                return False
            
            # Check position limits per symbol
            symbol_positions = sum(1 for trade in self.open_trades.values() if trade.symbol == symbol)
            if symbol_positions >= self.max_positions_per_symbol:
                logger.debug(f"Max positions per symbol reached for {symbol}: {symbol_positions}")
                return False
            
            # Check total position limit
            if len(self.open_trades) >= self.max_total_positions:
                logger.debug(f"Max total positions reached: {len(self.open_trades)}")
                return False
            
            # Check capital availability - FIXED CALCULATION
            used_capital = sum(getattr(trade, "position_size", 1000.0) for trade in self.open_trades.values())
            available_capital = self.initial_capital - used_capital
            min_position_size = min(self.initial_capital * 0.1, 5000)  # Minimum position size
            
            if available_capital < min_position_size:
                logger.warning(f"Insufficient capital: {available_capital:.2f} (need {min_position_size:.2f})")
                return False
            
            return True
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'check_risk_limits', 'symbol': symbol})
            return False

    def _get_rejection_reason(self, signal: Dict, entry_price: float) -> str:
        """Get the reason why a signal was rejected. COOLDOWN LOGIC REMOVED."""
        try:
            symbol = signal["symbol"]
            
            # Check position limits
            symbol_positions = sum(1 for trade in self.open_trades.values() if trade.symbol == symbol)
            if symbol_positions >= self.max_positions_per_symbol:
                return f"Max positions per symbol reached: {symbol_positions}/{self.max_positions_per_symbol}"
            
            # Check risk limits
            if not self._check_risk_limits(symbol):
                return "Risk limits exceeded"
            
            return "Unknown rejection reason"
        except Exception as e:
            return f"Error determining rejection reason: {e}"

    def _open_trade(self, signal: Dict, entry_price: float) -> Optional[str]:
        """Open a new trade. LAST_TRADE_TIME TRACKING REMOVED."""
        try:
            symbol = signal['symbol']
            
            # Check risk limits
            if not self._check_risk_limits(symbol):
                logger.debug(f"❌ Trade rejected for {symbol}: Risk limits exceeded")
                return None
            
            # ── ATR-based SL / TP ───────────────────────────────────────
            symbol_data = self._last_hist.get(symbol)
            atr = float(signal.get('atr', 0))
            if atr == 0 and symbol_data is not None:
                atr = self.atr_risk.compute_atr(symbol_data)
            if atr == 0:
                atr = entry_price * 0.01

            # Generate stable trade_id before ATR registration
            timestamp = datetime.now(self.tz)
            trade_id  = f"{symbol}_{int(timestamp.timestamp())}"

            atr_levels = self.atr_risk.register_trade(
                trade_id=trade_id,
                direction=signal['signal'],
                entry_price=entry_price,
                atr=atr,
            )
            stop_loss_price   = atr_levels.stop_loss
            take_profit_price = atr_levels.take_profit_1

            # ── Risk-based position sizing (Kelly + regime) ─────────────
            deployed = sum(getattr(t, 'position_size', 0) for t in self.open_trades.values())
            position_size = self.position_sizer.get_position_size(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                strategy=signal['strategy'],
                confidence=signal.get('confidence', 70.0),
                regime=signal.get('regime', 'UNKNOWN'),
                deployed_capital=deployed,
            )
            if position_size <= 0:
                logger.debug(f"❌ Position size 0 for {symbol} — capital limit")
                return None
            
            # Create trade object
            trade = OptimizedTrade(
                trade_id=trade_id,
                symbol=symbol,
                strategy=signal['strategy'],
                signal=signal['signal'],
                entry_price=entry_price,
                position_size=position_size,
                entry_time=timestamp,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
            
            # Store trade
            self.open_trades[trade_id] = trade
            
            # Save to database
            self.db.save_open_trade(trade_id, "crypto", symbol, signal["strategy"], signal["signal"], 
                                   entry_price, position_size, timestamp, 
                                   stop_loss_price, take_profit_price)
            
            logger.info(f"✅ Opened trade: {trade_id} - {symbol} {signal['signal']} @ ${entry_price:.2f}")
            return trade_id
            
        except Exception as e:
            logger.error(f"Error opening trade: {e}")
            return None
    
    def _process_signals(self):
        """Process trading signals."""
        try:
            # ── Expiry / event blackout ───────────────────────────────
            is_blocked, block_reason = self.blackout.is_blackout()
            if is_blocked:
                logger.info(f"🚫 Blackout active — skipping: {block_reason}")
                return
            # Get current prices
            current_prices = {}
            for symbol in self.symbols:
                price = self.get_current_price(symbol)
                if price:
                    current_prices[symbol] = price
            
            if not current_prices:
                logger.warning("No current prices available")
                return
                
            logger.debug(f"✅ Got prices for {len(current_prices)} symbols")
            
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

            # Cache for ATR fallback inside _open_trade
            self._last_hist = historical_data
            
            # Generate signals
            signals = self.strategy_engine.generate_signals_for_all_symbols(historical_data, current_prices)
            
            # Process ALL signals (signal limiting removed)
            logger.info(f"🎯 Processing {len(signals)} signals (no artificial limits)")
            for signal in signals:
                if self._stop_event.is_set():
                    break
                
                self._process_signal(signal, current_prices)
                
        except Exception as e:
            error_handler.handle_error(e, {'context': 'process_signals'})
    
    def _process_signal(self, signal: Dict, current_prices: Dict[str, float]):
        """Process a single signal."""
        try:
            symbol = signal['symbol']
            if symbol not in current_prices:
                return
            
            entry_price = current_prices[symbol]
            
            # Open trade (with all risk checks)
            trade_id = self._open_trade(signal, entry_price)
            if trade_id:
                logger.info(f"✅ Signal executed: {signal['strategy']} {signal['signal']} for {symbol} @ ${entry_price:.2f}")
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
            
            for trade_id, trade in list(self.open_trades.items()):
                symbol = trade.symbol
                if symbol not in current_prices:
                    continue

                current_price = current_prices[symbol]
                entry_price   = trade.entry_price
                position_size = getattr(trade, 'position_size', 1000.0)

                # ── ATR trailing stop (replaces flat 2%/3% logic) ────────
                exit_reason = self.atr_risk.check_exit(trade_id, current_price)

                if exit_reason is None:
                    elapsed = (datetime.now(self.tz) - trade.entry_time).total_seconds()
                    if elapsed > 4 * 3600:
                        exit_reason = "TIME_EXIT"

                if exit_reason:
                    if trade.signal in ['BUY', 'BUY CALL']:
                        pnl = (current_price - entry_price) * (position_size / entry_price)
                    else:
                        pnl = (entry_price - current_price) * (position_size / entry_price)
                    trades_to_close.append((trade_id, trade, current_price, exit_reason, pnl))
            
            # Close trades that met exit conditions
            for trade_id, trade, exit_price, exit_reason, pnl in trades_to_close:
                self._close_trade(trade_id, exit_price, exit_reason, pnl)
                
        except Exception as e:
            logger.error(f"Error updating open trades: {e}")

    def _check_exit_conditions(self, trade: OptimizedTrade, current_price: float, current_time: datetime) -> Optional[str]:
        """Check if trade should be closed."""
        try:
            # Time-based exit (24 hours)
            if (current_time - trade.entry_time).total_seconds() > 86400:
                return "TIME_EXIT"
            
            # Stop loss
            if trade.signal in ['BUY', 'BUY CALL'] and current_price <= trade.stop_loss_price:
                return "STOP_LOSS"
            elif trade.signal in ['SELL', 'BUY PUT'] and current_price >= trade.stop_loss_price:
                return "STOP_LOSS"
            
            # Take profit
            if trade.signal in ['BUY', 'BUY CALL'] and current_price >= trade.take_profit_price:
                return "TARGET_HIT"
            elif trade.signal in ['SELL', 'BUY PUT'] and current_price <= trade.take_profit_price:
                return "TARGET_HIT"
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None
    
    def _close_trade(self, trade_id: str, exit_price: float, exit_reason: str, pnl: float = None):
        """Close a trade — updates Kelly model and ATR registry."""
        try:
            trade = self.open_trades.pop(trade_id, None)
            if not trade:
                return

            if pnl is None:
                if trade.signal in ['BUY', 'BUY CALL']:
                    pnl = (exit_price - trade.entry_price) * (trade.position_size / trade.entry_price)
                else:
                    pnl = (trade.entry_price - exit_price) * (trade.position_size / trade.entry_price)

            self.daily_pnl += pnl

            # Feed into Kelly position sizer
            self.position_sizer.record_trade_result(trade.strategy, pnl)
            # Purge ATR registry
            self.atr_risk.remove_trade(trade_id)

            self.db.close_trade(trade_id, "crypto", exit_price, datetime.now(self.tz), exit_reason, pnl)

            icon = "💹" if pnl < 0 else "💸"
            logger.info(
                f"{icon} Closed [{exit_reason}]: {trade_id} — "
                f"{trade.symbol} @ ${exit_price:.2f} | P&L: ${pnl:.2f}"
            )

        except Exception as e:
            logger.error(f"Error closing trade {trade_id}: {e}")
    
    def run(self):
        """Main trading loop."""
        logger.info(f"🚀 Starting WORKING Optimized Crypto Trader with ${self.initial_capital:,.2f} capital")
        logger.info("🔧 OPTIMIZATIONS: Cooldown removed, Signal limiting removed")
        
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
        if not hasattr(self, 'open_trades'):
            self.open_trades = {}
        """Load existing open trades from database."""
        try:
            open_trades = self.db.get_open_trades("crypto")
            for trade in open_trades:
                trade_id = trade[1]  # trade_id column
                # Create a simple trade object with necessary attributes
                class TradeObject:
                    def __init__(self, data):
                        self.trade_id = data[1]
                        self.symbol = data[3]
                        self.strategy = data[4]
                        self.signal = data[5]
                        self.entry_price = data[6]
                        self.quantity = data[7]
                        self.position_size = data[7]  # Use quantity as position_size
                        self.entry_time = datetime.fromisoformat(data[8])
                        self.stop_loss_price = data[9] if data[9] else 0
                        self.take_profit_price = data[10] if data[10] else 0
                
                self.open_trades[trade_id] = TradeObject(trade)
            
            logger.info(f"📊 Loaded {len(self.open_trades)} open trades from database")
            
        except Exception as e:
            logger.error(f"Error loading open trades: {e}")
            self.open_trades = {}

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='WORKING Optimized Modular Trading System')
    parser.add_argument('--market', choices=['crypto', 'indian'], default='crypto', help='Market to trade')
    parser.add_argument('--capital', type=float, default=20000, help='Starting capital')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine market type
    market_type = MarketType.CRYPTO if args.market == 'crypto' else MarketType.INDIAN_STOCKS
    
    # Create and run trader
    trader = WorkingOptimizedModularTradingSystem(
        market_type=market_type,
        initial_capital=args.capital,
        verbose=args.verbose
    )
    trader.run()

if __name__ == "__main__":
    main()

    def _load_open_trades_from_db(self):
        """Load existing open trades from database."""
        if not hasattr(self, 'open_trades'):
            self.open_trades = {}
        """Load existing open trades from database."""
        try:
            open_trades = self.db.get_open_trades("crypto")
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
                trade_obj = type('Trade', (), {
                    'trade_id': trade_id,
                    'symbol': symbol,
                    'strategy': strategy,
                    'signal': signal,
                    'entry_price': entry_price,
                    'position_size': quantity,
                    'entry_time': entry_time,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'status': 'open'
                })()
                
                self.open_trades[trade_id] = trade_obj
            
            logger.info(f"📊 Loaded {len(self.open_trades)} open trades from database")
            
        except Exception as e:
            logger.error(f"Error loading open trades from database: {e}")

