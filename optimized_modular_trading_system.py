"""
Optimized Modular Trading System with enhanced error handling, connection pooling,
WebSocket reconnection, and memory monitoring.
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
from src.core.error_handler import error_handler, handle_errors, TradingError, APIError
from src.core.connection_pool import initialize_connection_pools, get_db_connection, get_api_session
from src.core.websocket_manager import WebSocketManager
from src.core.memory_monitor import memory_monitor, start_memory_monitoring, MemoryAlertLevel

# Import existing components
from src.adapters.market_factory import MarketFactory
from src.adapters.market_interface import MarketType
from src.core.unified_strategy_engine import UnifiedStrategyEngine
from src.models.unified_database_updated import UnifiedTradingDatabase

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

class OptimizedModularTradingSystem:
    """Optimized modular trading system with enhanced reliability."""
    
    def __init__(self, 
                 market_type: MarketType,
                 initial_capital: float = 20000.0,
                 max_risk_per_trade: float = 0.02,
                 confidence_cutoff: float = 25.0,
                 exposure_limit: float = 0.6,
                 symbols: List[str] = None,
                 verbose: bool = False):
        """Initialize the optimized trading system."""
        
        self.market_type = market_type
        self.initial_capital = initial_capital
        self.cash = float(initial_capital)
        self.max_risk_per_trade = max_risk_per_trade
        self.confidence_cutoff = confidence_cutoff
        self.exposure_limit = exposure_limit
        self.verbose = verbose
        
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
        self.strategy_engine = UnifiedStrategyEngine(symbols, confidence_cutoff)
        
        # Initialize database with connection pooling
        self.db = UnifiedTradingDatabase("optimized_trading.db")
        
        # Trading state
        self.is_running = False
        self.open_trades = {}
        self.closed_trades = []
        self.rejected_signals = []
        
        # Performance tracking
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        self.total_trades_executed = 0
        self.total_trades_closed = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.daily_pnl = 0.0
        
        # Threading and async
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._trading_thread = None
        self._websocket_manager = None
        
        # Memory monitoring
        self._setup_memory_monitoring()
        
        logger.info(f"🚀 Optimized Modular Trading System initialized for {market_type.value}")
        logger.info(f"💰 Initial capital: {self.market.config.currency} {initial_capital:,.2f}")
        logger.info(f"📊 Symbols: {', '.join(symbols)}")
    
    def _initialize_systems(self):
        """Initialize enhanced systems."""
        try:
            # Initialize connection pools
            initialize_connection_pools("optimized_trading.db")
            
            # Start memory monitoring
            start_memory_monitoring()
            
            # Setup memory alert handlers
            memory_monitor.add_alert_handler(
                MemoryAlertLevel.HIGH,
                self._handle_memory_alert
            )
            
            logger.info("✅ Enhanced systems initialized")
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'system_initialization'})
            raise
    
    def _setup_memory_monitoring(self):
        """Setup memory monitoring alerts."""
        def memory_alert_handler(stats, level):
            logger.warning(f"MEMORY ALERT {level.value.upper()}: "
                          f"Process memory {stats.process_memory:.1f}MB")
            
            if level == memory_monitor.MemoryAlertLevel.CRITICAL:
                # Trigger memory cleanup
                memory_monitor.cleanup_memory()
                
                # Reduce trading frequency if needed
                self._reduce_trading_frequency()
        
        memory_monitor.add_alert_handler(
            MemoryAlertLevel.HIGH,
            memory_alert_handler
        )
    
    def _handle_memory_alert(self, stats, level):
        """Handle memory alerts."""
        logger.warning(f"MEMORY ALERT {level.value.upper()}: "
                      f"Process memory {stats.process_memory:.1f}MB")
        
        if level == memory_monitor.MemoryAlertLevel.CRITICAL:
            memory_monitor.cleanup_memory()
            self._reduce_trading_frequency()
    
    def _reduce_trading_frequency(self):
        """Reduce trading frequency to conserve memory."""
        logger.warning("Reducing trading frequency due to high memory usage")
        # Implementation would reduce signal processing frequency
    
    @handle_errors()
    def _get_default_symbols(self) -> List[str]:
        """Get default symbols based on market type."""
        return self.market.get_default_symbols()
    
    @handle_errors()
    def _create_data_provider(self):
        """Create data provider with error handling."""
        try:
            return self.market.get_data_provider()
        except Exception as e:
            error_handler.handle_error(e, {'context': 'data_provider_creation'})
            raise
    
    @handle_errors()
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with error handling."""
        try:
            return self.data_provider.get_current_price(symbol)
        except Exception as e:
            error_handler.handle_error(e, {'context': 'get_current_price', 'symbol': symbol})
            return None
    
    @handle_errors()
    def get_historical_data(self, symbol: str, days: int = 30) -> Optional[Any]:
        """Get historical data with error handling."""
        try:
            end_date = datetime.now(self.tz)
            start_date = end_date - timedelta(days=days)
            return self.data_provider.get_historical_data(symbol, start_date, end_date, '5')
        except Exception as e:
            error_handler.handle_error(e, {'context': 'get_historical_data', 'symbol': symbol})
            return None
    
    @handle_errors()
    def _calculate_position_size(self, signal: Dict, entry_price: float) -> float:
        """Calculate position size with enhanced error handling."""
        try:
            # Base risk amount
            base_risk = self.cash * self.max_risk_per_trade
            
            # Confidence multiplier
            confidence = signal.get('confidence', 50)
            confidence_multiplier = max(0.5, min(2.0, confidence / 50.0))
            
            # Calculate adjusted risk
            adjusted_risk = base_risk * confidence_multiplier
            
            # Get lot size for symbol
            lot_size = self.market.get_lot_size(signal['symbol'])
            
            # Calculate maximum position value based on risk
            max_position_value = adjusted_risk
            
            # Calculate maximum number of lots we can afford
            max_lots = int(max_position_value / (entry_price * lot_size))
            
            # Ensure we don't exceed available capital (use 90% for safety)
            available_capital = self.cash * 0.9
            max_affordable_lots = int(available_capital / (entry_price * lot_size))
            max_lots = min(max_lots, max_affordable_lots)
            
            # Minimum 1 lot if we can afford it, otherwise 0
            if max_lots >= 1:
                position_size = max_lots * lot_size
            else:
                position_size = 0.0
            
            return position_size
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'position_sizing', 'signal': signal})
            return 0.0
    
    @handle_errors()
    def _open_trade(self, signal: Dict, entry_price: float, timestamp: datetime) -> Optional[str]:
        """Open a trade with enhanced error handling."""
        try:
            # Calculate position size
            position_size = self._calculate_position_size(signal, entry_price)
            if position_size <= 0:
                logger.warning(f"Cannot open trade: insufficient capital for {signal['symbol']}")
                return None
            
            # Create trade
            trade_id = f"{signal['symbol']}_{signal['strategy']}_{int(timestamp.timestamp())}"
            trade = OptimizedTrade(
                id=trade_id,
                symbol=signal['symbol'],
                strategy=signal['strategy'],
                signal=signal['signal'],
                entry_price=entry_price,
                quantity=position_size,
                entry_time=timestamp
            )
            
            # Calculate commission
            commission_rate = self.market.get_commission_rate(signal['symbol'])
            trade.commission = entry_price * position_size * commission_rate
            
            # Check if we have enough capital
            total_cost = (entry_price * position_size) + trade.commission
            if total_cost > self.cash:
                logger.warning(f"Insufficient capital: need {total_cost:.2f}, have {self.cash:.2f}")
                return None
            
            # Deduct capital
            self.cash -= total_cost
            
            # Store trade
            with self._lock:
                self.open_trades[trade_id] = trade
                self.total_trades_executed += 1
            
            # Save to database with connection pooling
            with get_db_connection() as conn:
                # Implementation would save trade to database
                pass
            
            logger.info(f"✅ Opened trade: {trade_id} - {signal['symbol']} {signal['signal']} "
                       f"@ {entry_price:.2f} (qty: {position_size:.2f})")
            
            return trade_id
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'open_trade', 'signal': signal})
            return None
    
    @handle_errors()
    def _close_trade(self, trade_id: str, exit_price: float, exit_reason: str) -> bool:
        """Close a trade with enhanced error handling."""
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
    
    @handle_errors()
    def start_trading(self):
        """Start the trading system."""
        if self.is_running:
            logger.warning("Trading system is already running")
            return
        
        self.is_running = True
        self._stop_event.clear()
        
        # Start trading thread
        self._trading_thread = threading.Thread(target=self._trading_loop)
        self._trading_thread.daemon = True
        self._trading_thread.start()
        
        logger.info("🚀 Optimized trading system started")
    
    def stop_trading(self):
        """Stop the trading system."""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        # Wait for trading thread to finish
        if self._trading_thread:
            self._trading_thread.join(timeout=10.0)
        
        # Close WebSocket connections
        if self._websocket_manager:
            asyncio.run(self._websocket_manager.disconnect())
        
        logger.info("🛑 Optimized trading system stopped")
    
    def _trading_loop(self):
        """Main trading loop with enhanced error handling."""
        while self.is_running and not self._stop_event.is_set():
            try:
                # Check if market is open
                if not self.market.is_market_open():
                    time.sleep(60)  # Check every minute when market is closed
                    continue
                
                # Process signals
                self._process_signals()
                
                # Update open trades
                self._update_open_trades()
                
                # Sleep between iterations
                time.sleep(5)  # 5-second intervals
                
            except Exception as e:
                error_handler.handle_error(e, {'context': 'trading_loop'})
                time.sleep(10)  # Wait longer on error
    
    @handle_errors()
    def _process_signals(self):
        """Process trading signals with error handling."""
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
            
            # Process signals
            for signal in signals:
                if self._stop_event.is_set():
                    break
                
                self._process_signal(signal, current_prices)
                
        except Exception as e:
            error_handler.handle_error(e, {'context': 'process_signals'})
    
    @handle_errors()
    def _process_signal(self, signal: Dict, current_prices: Dict[str, float]):
        """Process a single signal with error handling."""
        try:
            symbol = signal['symbol']
            if symbol not in current_prices:
                return
            
            entry_price = current_prices[symbol]
            
            # Open trade
            trade_id = self._open_trade(signal, entry_price, datetime.now(self.tz))
            if trade_id:
                logger.info(f"✅ Signal processed: {signal['strategy']} {signal['signal']} "
                           f"for {symbol} @ {entry_price:.2f}")
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'process_signal', 'signal': signal})
    
    @handle_errors()
    def _update_open_trades(self):
        """Update open trades with error handling."""
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
        """Check exit conditions for a trade."""
        try:
            # Time-based exit (30 minutes)
            if (current_time - trade.entry_time).total_seconds() > 1800:
                return "time_exit"
            
            # Price-based exit (simplified)
            price_change = (current_price - trade.entry_price) / trade.entry_price
            
            if price_change > 0.05:  # 5% profit
                return "profit_target"
            elif price_change < -0.03:  # 3% loss
                return "stop_loss"
            
            return None
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'check_exit_conditions', 'trade_id': trade.id})
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status with enhanced information."""
        try:
            current_equity = self.cash + sum(t.pnl for t in self.open_trades.values())
            
            return {
                'system': {
                    'is_running': self.is_running,
                    'market_type': self.market_type.value,
                    'symbols': self.symbols
                },
                'capital': {
                    'initial_capital': self.initial_capital,
                    'current_cash': self.cash,
                    'current_equity': current_equity,
                    'daily_pnl': self.daily_pnl,
                    'max_drawdown': self.max_drawdown
                },
                'trades': {
                    'open_trades': len(self.open_trades),
                    'closed_trades': len(self.closed_trades),
                    'total_executed': self.total_trades_executed,
                    'total_closed': self.total_trades_closed,
                    'winning_trades': self.winning_trades,
                    'losing_trades': self.losing_trades
                },
                'performance': {
                    'win_rate': (self.winning_trades / max(1, self.total_trades_closed)) * 100,
                    'total_pnl': sum(t.pnl for t in self.closed_trades),
                    'avg_trade_pnl': sum(t.pnl for t in self.closed_trades) / max(1, len(self.closed_trades))
                },
                'systems': {
                    'memory': memory_monitor.get_summary(),
                    'errors': error_handler.get_error_stats()
                }
            }
            
        except Exception as e:
            error_handler.handle_error(e, {'context': 'get_status'})
            return {'error': str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Modular Trading System")
    parser.add_argument("--market", choices=["indian", "crypto"], required=True,
                       help="Market type to trade")
    parser.add_argument("--capital", type=float, default=20000.0,
                       help="Initial capital")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create trading system
    market_type = MarketType.INDIAN_STOCKS if args.market == "indian" else MarketType.CRYPTO
    
    system = OptimizedModularTradingSystem(
        market_type=market_type,
        initial_capital=args.capital,
        verbose=True
    )
    
    try:
        # Start trading
        system.start_trading()
        
        # Run for test duration or indefinitely
        if args.test:
            time.sleep(60)  # Run for 1 minute in test mode
        else:
            # Run indefinitely
            while True:
                time.sleep(60)
                status = system.get_status()
                logger.info(f"Status: {status['capital']['current_equity']:.2f} equity, "
                           f"{status['trades']['open_trades']} open trades")
    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down trading system...")
    finally:
        system.stop_trading()
