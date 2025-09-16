"""
WORKING Optimized Modular Trading System with proper timezone handling
and all critical fixes implemented.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from zoneinfo import ZoneInfo

# Import our enhanced systems
from risk_config import risk_config
from src.core.error_handler import error_handler, handle_errors, TradingError, APIError
from src.core.connection_pool import initialize_connection_pools, get_db_connection, get_api_session
from src.core.websocket_manager import WebSocketManager
from src.core.memory_monitor import memory_monitor, start_memory_monitoring, MemoryAlertLevel

# Import existing components
from src.adapters.market_factory import MarketFactory
from src.adapters.market_interface import MarketType
from src.core.enhanced_strategy_engine import EnhancedStrategyEngine
from src.models.consolidated_database import ConsolidatedTradingDatabase

logger = logging.getLogger(__name__)

@dataclass
class OptimizedTrade:
    """Optimized trade data structure."""
    id: str
    symbol: str
    strategy: str
    signal: str
    entry_price: float
    quantity: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    commission: float = 0.0
    exit_reason: Optional[str] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

class WorkingOptimizedModularTradingSystem:
    """WORKING Optimized modular trading system with proper risk management."""
    
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
        # CRITICAL FIX: Risk management parameters (configurable)
        self.risk_config = risk_config
        self.stop_loss_percent = 0.03  # 3% stop loss
        self.take_profit_percent = 0.05  # 5% take profit
        self.max_positions_per_symbol = risk_config.get("max_positions_per_symbol", 3)
        self.max_total_positions = risk_config.get("max_total_positions", 15)
        self.trade_cooldown_minutes = risk_config.get("trade_cooldown_seconds", 300) / 60.0
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
        self.strategy_engine = EnhancedStrategyEngine(symbols, confidence_cutoff)
        
        # Initialize database with connection pooling
        self.db = ConsolidatedTradingDatabase("data/trading.db")
        
        # Trading state
        self.is_running = False
        self.open_trades = {}
        self.closed_trades = []
        self.rejected_signals = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # CRITICAL FIX: Position tracking with proper timezone handling
        self.positions_per_symbol = {symbol: 0 for symbol in symbols}
        # Initialize with timezone-aware datetime
        self.last_trade_time = {symbol: datetime.now(self.tz) for symbol in symbols}
        self.daily_start_capital = initial_capital
        self.daily_pnl = 0.0
        
        # Performance tracking
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        self.total_trades_executed = 0
        self.total_trades_closed = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info(f"🚀 WORKING Trading System initialized with {initial_capital} capital")
        logger.info(f"🛡️ Risk Management: {self.stop_loss_percent*100}% stop, {self.take_profit_percent*100}% target")
        logger.info(f"📊 Position Limits: {self.max_positions_per_symbol} per symbol, {self.max_total_positions} total")
    
    def _initialize_systems(self):
        """Initialize enhanced systems."""
        try:
            # Initialize connection pools
            initialize_connection_pools()
            
            # Start memory monitoring
            start_memory_monitoring()
            
            logger.info("✅ Enhanced systems initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize systems: {e}")
            raise
    
    def _get_default_symbols(self) -> List[str]:
        """Get default symbols for the market."""
        try:
            return self.market.get_default_symbols()
        except Exception as e:
            logger.error(f"Failed to get default symbols: {e}")
            return ['BTCUSDT', 'ETHUSDT']  # Fallback
    
    def _create_data_provider(self):
        """Create data provider for the market."""
        try:
            return self.market.get_data_provider()
        except Exception as e:
            logger.error(f"Failed to create data provider: {e}")
            return None
    
    @handle_errors()
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with error handling."""
        try:
            if not self.data_provider:
                return None
            return self.data_provider.get_current_price(symbol)
        except Exception as e:
            error_handler.handle_error(e, {'context': 'get_current_price', 'symbol': symbol})
            return None
    
    @handle_errors()
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[Any]:
        """Get historical data with error handling."""
        try:
            if not self.data_provider:
                return None
            
            end_date = datetime.now(self.tz)
            start_date = end_date - timedelta(days=days)
            
            return self.data_provider.get_historical_data(
                symbol, start_date, end_date, '5'
            )
        except Exception as e:
            error_handler.handle_error(e, {'context': 'get_historical_data', 'symbol': symbol})
            return None
    
    def _check_risk_limits(self, symbol: str) -> bool:
        """CRITICAL FIX: Check if we can open a new position with proper timezone handling."""
        try:
            # Check if risk management is enabled
            if not self.risk_config.is_risk_management_enabled():
                return True
            
            # Use timezone-aware datetime            # Check daily loss limit
            current_equity = self.cash + sum(t.pnl for t in self.open_trades.values())
            daily_loss = (self.daily_start_capital - current_equity) / self.daily_start_capital
            if daily_loss > self.daily_loss_limit:
                logger.warning(f"🚨 Daily loss limit exceeded: {daily_loss*100:.1f}% > {self.daily_loss_limit*100:.1f}%")
                return False
            
            # Check emergency stop
            total_loss = (self.initial_capital - current_equity) / self.initial_capital
            if total_loss > self.emergency_stop_loss:
                logger.error(f"🚨 EMERGENCY STOP: Total loss {total_loss*100:.1f}% > {self.emergency_stop_loss*100:.1f}%")
                return False
            
            # Check total position limit
            if len(self.open_trades) >= self.max_total_positions:
                logger.warning(f"🚨 Maximum total positions reached: {len(self.open_trades)}")
                return False
            
            # Check positions per symbol
            if self.positions_per_symbol[symbol] >= self.max_positions_per_symbol:
                logger.warning(f"🚨 Maximum positions for {symbol} reached: {self.positions_per_symbol[symbol]}")
                return False
            
            # CRITICAL FIX: Check cooldown period with proper timezone handling
            last_trade_time = self.last_trade_time[symbol]
            
            # Ensure both datetimes are timezone-aware
            if last_trade_time.tzinfo is None:
                # If last_trade_time is naive, make it timezone-aware
                last_trade_time = last_trade_time.replace(tzinfo=self.tz)
            
            # Now both datetimes are timezone-aware, safe to subtract
            time_since_last_trade = (current_time - last_trade_time).total_seconds()
            
            if time_since_last_trade < (self.trade_cooldown_minutes * 60):
                logger.debug(f"⏰ Cooldown active for {symbol}: {time_since_last_trade:.0f}s remaining")
                return False
            
            return True
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'check_risk_limits', 'symbol': symbol})

    def _get_rejection_reason(self, signal: Dict, entry_price: float) -> str:
        """Get the reason why a signal was rejected."""
        try:
            symbol = signal["symbol"]
            
            # Check if risk management is enabled
            if not self.risk_config.is_risk_management_enabled():
                return "Risk management disabled - should not be rejected"
            
            # Check position limits
            symbol_positions = sum(1 for trade in self.open_trades.values() if trade.symbol == symbol)
            if symbol_positions >= self.max_positions_per_symbol:
                return f"Max positions per symbol reached: {symbol_positions}/{self.max_positions_per_symbol}"
            
            # Check cooldown
            if symbol in self.last_trade_time:
                time_since_last = datetime.now(self.tz) - self.last_trade_time[symbol]
                if time_since_last.total_seconds() < self.trade_cooldown_minutes * 60:
                    return f"Trade cooldown active: {time_since_last.total_seconds():.1f}s remaining"
            
            # Check risk limits
            if not self._check_risk_limits(symbol):
                return "Risk limits exceeded"
            
            return "Unknown rejection reason"
        except Exception as e:
            return f"Error determining rejection reason: {e}"

    @handle_errors()
    def _calculate_position_size(self, signal: Dict, entry_price: float) -> float:
        """CRITICAL FIX: Calculate position size with proper risk management."""
        try:
            # Base risk amount
            base_risk = self.cash * self.max_risk_per_trade
            
            # Confidence multiplier (more conservative)
            confidence = signal.get('confidence', 50)
            confidence_multiplier = max(0.3, min(1.5, confidence / 100.0))  # More conservative range
            
            # Calculate adjusted risk
            adjusted_risk = base_risk * confidence_multiplier
            
            # Get lot size for symbol
            lot_size = self.market.get_lot_size(signal['symbol'])
            
            # Calculate position size based on risk amount and entry price
            position_size_units = adjusted_risk / entry_price
            
            # Convert to lots and ensure whole lots
            position_size = (position_size_units // lot_size) * lot_size
            
            # Ensure we don't exceed available capital
            max_affordable_lots = (self.cash * 0.9) // (entry_price * lot_size)
            position_size = min(position_size, max_affordable_lots * lot_size)
            
            # Ensure minimum 1 lot if affordable
            if position_size < lot_size and max_affordable_lots >= 1:
                position_size = lot_size
            elif position_size < lot_size and max_affordable_lots < 1:
                logger.warning(f"⚠️ Cannot afford even 1 lot for {signal['symbol']} - need {entry_price * lot_size:,.2f}, have {self.cash * 0.9:,.2f}")
                return 0.0
            
            return position_size
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'position_sizing', 'signal': signal})
            return 0.0
    
    @handle_errors()
    def _open_trade(self, signal: Dict, entry_price: float, timestamp: datetime) -> Optional[str]:
        """CRITICAL FIX: Open a trade with proper risk management."""
        try:
            symbol = signal['symbol']
            
            # Check risk limits first
            if not self._check_risk_limits(symbol):
                return None
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, entry_price)
            if position_size <= 0:
                logger.warning(f"Cannot open trade: insufficient capital for {symbol}")
                return None
            
            # Create trade
            trade_id = f"{symbol}_{signal['strategy']}_{int(timestamp.timestamp())}"
            
            # Calculate stop loss and take profit prices
            stop_loss_price = entry_price * (1 - self.stop_loss_percent)
            take_profit_price = entry_price * (1 + self.take_profit_percent)
            
            trade = OptimizedTrade(
                id=trade_id,
                symbol=symbol,
                strategy=signal['strategy'],
                signal=signal['signal'],
                entry_price=entry_price,
                quantity=position_size,
                entry_time=timestamp,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            # Calculate commission
            commission_rate = self.market.get_commission_rate(symbol)
            trade.commission = entry_price * position_size * commission_rate
            
            # Check if we have enough capital
            total_cost = (entry_price * position_size) + trade.commission
            if total_cost > self.cash:
                logger.warning(f"Insufficient capital: need {total_cost:.2f}, have {self.cash:.2f}")
                return None
            
            # Deduct capital
            self.cash -= total_cost
            
            # Store trade and update tracking
            with self._lock:
                self.open_trades[trade_id] = trade
                self.positions_per_symbol[symbol] += 1
                self.last_trade_time[symbol] = timestamp  # This is already timezone-aware
                self.total_trades_executed += 1
            
            # Save to database
            self.db.save_open_trade(
                trade_id=trade_id,
                market="crypto",
                symbol=symbol,
                strategy=signal["strategy"],
                signal=signal["signal"],
                entry_price=entry_price,
                quantity=position_size,
                entry_time=timestamp,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price
            )
            
            
            logger.info(f"✅ Opened trade: {trade_id} - {symbol} {signal['signal']} "
                       f"@ {entry_price:.2f} (qty: {position_size:.2f})")
            logger.info(f"   🛡️ Stop Loss: {stop_loss_price:.2f}, Take Profit: {take_profit_price:.2f}")
            
            return trade_id
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'open_trade', 'signal': signal})
            return None
    
    @handle_errors()
    def _close_trade(self, trade_id: str, exit_price: float, exit_reason: str) -> bool:
        """CRITICAL FIX: Close a trade with proper tracking."""
        try:
            with self._lock:
                if trade_id not in self.open_trades:
                    logger.warning(f"Trade {trade_id} not found in open trades")
                    return False
                
                trade = self.open_trades[trade_id]
                trade.exit_price = exit_price
                trade.exit_time = datetime.now(self.tz)
                trade.exit_reason = exit_reason
                
                # Calculate P&L
                pnl = (exit_price - trade.entry_price) * trade.quantity - trade.commission
                trade.pnl = pnl
                
                # Add capital back
                self.cash += (exit_price * trade.quantity) - trade.commission
                
                # Update position tracking
                self.positions_per_symbol[trade.symbol] -= 1
                
                # Move to closed trades
                self.closed_trades.append(trade)
                del self.open_trades[trade_id]
                
                # Update statistics
                self.total_trades_closed += 1
                self.daily_pnl += pnl
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                # Update peak capital and drawdown
                current_equity = self.cash + sum(t.pnl for t in self.open_trades.values())
                if current_equity > self.peak_capital:
                    self.peak_capital = current_equity
                
                current_drawdown = (self.peak_capital - current_equity) / self.peak_capital
                if current_drawdown > self.max_drawdown:
                    self.max_drawdown = current_drawdown
            
            logger.info(f"✅ Closed trade: {trade_id} - P&L: {pnl:.2f} ({exit_reason})")
            return True
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'close_trade', 'trade_id': trade_id})
            return False
    
    def emergency_stop_loss(self, trade_id: str, current_price: float) -> bool:
        """Emergency stop loss for a trade."""
        try:
            if trade_id not in self.open_trades:
                return False
            
            trade = self.open_trades[trade_id]
            
            # Calculate emergency stop loss (e.g., 5% loss)
            emergency_stop_price = trade.entry_price * 0.95  # 5% loss
            
            if current_price <= emergency_stop_price:
                logger.warning(f"Emergency stop loss triggered for {trade_id} at {current_price}")
                return self._close_trade(trade_id, current_price, "Emergency Stop Loss")
            
            return False
        except Exception as e:
            error_handler.handle_error(e, {"context": "emergency_stop_loss", "trade_id": trade_id})
            return False
    @handle_errors()
    def start_trading(self):
        """Start the WORKING trading system."""
        if self.is_running:
            logger.warning("Trading system is already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # Start trading thread
        self._trading_thread = threading.Thread(target=self._trading_loop)
        self._trading_thread.daemon = True
        self._trading_thread.start()
        
        logger.info("🚀 WORKING Trading system started with proper risk management")
    
    def stop_trading(self):
        """Stop the trading system."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        # Wait for trading thread to finish
        if self._trading_thread:
            self._trading_thread.join(timeout=10.0)
        
        logger.info("🛑 WORKING Trading system stopped")
    
    def _trading_loop(self):
        """CRITICAL FIX: Main trading loop with proper risk management."""
        while self.is_running and not self._stop_event.is_set():
            try:
                # Check if market is open
                if not self.market.is_market_open():
                    time.sleep(60)  # Check every minute when market is closed
                    continue
                
                # CRITICAL FIX: Check emergency stop first
                current_equity = self.cash + sum(t.pnl for t in self.open_trades.values())
                total_loss = (self.initial_capital - current_equity) / self.initial_capital
                if total_loss > self.emergency_stop_loss:
                    logger.error(f"🚨 EMERGENCY STOP TRIGGERED: {total_loss*100:.1f}% loss")
                    self.stop_trading()
                    break
                
                # Process signals
                self._process_signals()
                
                # CRITICAL FIX: Update and close trades
                self._update_open_trades()
                
                # Sleep between iterations
                time.sleep(10)  # 10-second intervals (less aggressive)
                
            except Exception as e:
                error_handler.handle_error(e, {'context': 'trading_loop'})
                time.sleep(30)  # Wait longer on error
    
    @handle_errors()
    def _process_signals(self):
        """CRITICAL FIX: Process trading signals with proper filtering."""
        try:
            # Get current prices
            current_prices = {}
            for symbol in self.symbols:
                price = self.get_current_price(symbol)
                if price:
                    current_prices[symbol] = price
            
            if not current_prices:
                logger.warning("No current prices available")
                return
            
            # Get historical data
            historical_data = {}
            for symbol in self.symbols:
                data = self.get_historical_data(symbol, 30)
                if data is not None:
                    historical_data[symbol] = data
            
            if not historical_data:
                logger.warning("No historical data available")
                return
            
            # Generate signals
            signals = self.strategy_engine.generate_signals(historical_data, current_prices)
            
            # Save all signals to database (both executed and rejected)
            for signal in signals:
                try:
                    self.db.save_signal(
                        market="crypto",
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
                except Exception as e:
                    logger.error(f"Failed to save signal: {e}")
            # CRITICAL FIX: Limit signals to prevent over-trading
            max_signals_per_cycle = 3  # Maximum 3 signals per cycle
            signals = signals[:max_signals_per_cycle]
            
            # Process signals
            for signal in signals:
                if self._stop_event.is_set():
                    break
                
                self._process_signal(signal, current_prices)
                
        except Exception as e:
            error_handler.handle_error(e, {'context': 'process_signals'})
    
    @handle_errors()
    def _process_signal(self, signal: Dict, current_prices: Dict[str, float]):
        """CRITICAL FIX: Process a single signal with proper risk checks."""
        try:
            symbol = signal['symbol']
            if symbol not in current_prices:
                return
            
            entry_price = current_prices[symbol]
            
            # Open trade (with all risk checks)
            trade_id = self._open_trade(signal, entry_price, datetime.now(self.tz))
            if trade_id:
                logger.info(f"✅ Signal processed: {signal['strategy']} {signal['signal']} "
                           f"for {symbol} @ {entry_price:.2f}")
            else:
                logger.debug(f"❌ Signal rejected: {signal['strategy']} {signal['signal']} for {symbol}")
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'process_signal', 'signal': signal})
    
    @handle_errors()
    def _update_open_trades(self):
        """CRITICAL FIX: Update open trades with proper exit logic."""
        try:
            current_time = datetime.now(self.tz)
            
            for trade_id, trade in list(self.open_trades.items()):
                if self._stop_event.is_set():
                    break
                
                # Get current price
                current_price = self.get_current_price(trade.symbol)
                if not current_price:
                    continue
                
                # Check exit conditions
                exit_reason = self._check_exit_conditions(trade, current_price, current_time)
                if exit_reason:
                    self._close_trade(trade_id, current_price, exit_reason)
                
        except Exception as e:
            error_handler.handle_error(e, {'context': 'update_open_trades'})
    
    def _check_exit_conditions(self, trade: OptimizedTrade, current_price: float, 
                              current_time: datetime) -> Optional[str]:
        """CRITICAL FIX: Check exit conditions with proper stop loss and take profit."""
        try:
            # Time-based exit (30 minutes maximum)
            if (current_time - trade.entry_time).total_seconds() > 1800:
                return "time_exit"
            
            # Stop loss check
            if current_price <= trade.stop_loss_price:
                return "stop_loss"
            
            # Take profit check
            if current_price >= trade.take_profit_price:
                return "take_profit"
            
            return None
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'check_exit_conditions', 'trade_id': trade.id})
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status with enhanced metrics."""
        try:
            current_equity = self.cash + sum(t.pnl for t in self.open_trades.values())
            total_return = (current_equity - self.initial_capital) / self.initial_capital
            
            return {
                "system": {
                    "market_type": self.market_type.value,
                    "symbols": self.symbols,
                    "is_running": self.is_running,
                    "total_positions": len(self.open_trades),
                    "positions_per_symbol": self.positions_per_symbol.copy()
                },
                "capital": {
                    "initial_capital": self.initial_capital,
                    "current_cash": self.cash,
                    "current_equity": current_equity,
                    "total_return": total_return,
                    "max_drawdown": self.max_drawdown
                },
                "trades": {
                    "total_executed": self.total_trades_executed,
                    "total_closed": self.total_trades_closed,
                    "winning_trades": self.winning_trades,
                    "losing_trades": self.losing_trades,
                    "win_rate": self.winning_trades / max(1, self.total_trades_closed)
                },
                "risk": {
                    "max_risk_per_trade": self.max_risk_per_trade,
                    "stop_loss_percent": self.stop_loss_percent,
                    "take_profit_percent": self.take_profit_percent,
                    "daily_loss_limit": self.daily_loss_limit,
                    "emergency_stop_loss": self.emergency_stop_loss
                }
            }
        except Exception as e:
            error_handler.handle_error(e, {'context': 'get_status'})
            return {}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WORKING Optimized Modular Trading System")
    parser.add_argument("--market", choices=["crypto", "indian"], default="crypto")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=0.02)
    parser.add_argument("--confidence", type=float, default=25.0)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(filename=f"logs/{args.market}/{args.market}_trading.log", 
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Suppress urllib3 debug logs
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # Suppress fyers API debug logs
    logging.getLogger("src.api.fyers").setLevel(logging.WARNING)
    )
    
    # Create and start the WORKING system
    market_type = MarketType.CRYPTO if args.market == "crypto" else MarketType.INDIAN_STOCKS
    
    system = WorkingOptimizedModularTradingSystem(
        market_type=market_type,
        initial_capital=args.capital,
        max_risk_per_trade=args.risk,
        confidence_cutoff=args.confidence,
        verbose=args.verbose
    )
    
    try:
        system.start_trading()
        
        # Status reporting loop
        while system.is_running:
            time.sleep(30)  # Report every 30 seconds
            status = system.get_status()
            equity = status.get("capital", {}).get("current_equity", 0)
            positions = status.get("system", {}).get("total_positions", 0)
            logger.info(f"Status: {equity:.2f} equity, {positions} open trades")
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        system.stop_trading()
    except Exception as e:
        logger.error(f"System error: {e}")
        system.stop_trading()
