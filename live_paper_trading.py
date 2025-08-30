#!/usr/bin/env python3
"""
Live Paper Trading System
Uses real-time broker data to simulate options trading
"""

import os
import sys
import time
import gc
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import sqlite3
from pathlib import Path
import uuid
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from http.server import HTTPServer, BaseHTTPRequestHandler

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.unified_database_updated import UnifiedDatabase, UnifiedTradingDatabase
from src.models.option_contract import OptionContract, OptionChain, OptionType, StrikeSelection
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.data.local_data_loader import LocalDataLoader
from src.data.realtime_data_manager import RealTimeDataManager
from src.execution.broker_execution import PaperBrokerAPI
from src.core.option_signal_mapper import OptionSignalMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """Represents a paper trade."""
    id: str
    timestamp: datetime
    contract_symbol: str
    underlying: str
    strategy: str
    signal_type: str
    entry_price: float
    quantity: int
    lot_size: int
    strike: float
    expiry: datetime
    option_type: str
    status: str = 'OPEN'
    commission: float = 0.0
    confidence: float = 0.0
    reasoning: str = ''
    stop_loss: Optional[float] = None
    target1: Optional[float] = None
    target2: Optional[float] = None
    target3: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    entry_value: Optional[float] = None # Added for consolidation
    entry_commission: Optional[float] = None # Added for consolidation
    entry_time: Optional[datetime] = None # Added for proper time tracking


@dataclass
class RejectedSignal:
    """Represents a rejected signal with reasoning."""
    id: str
    timestamp: datetime
    strategy: str
    signal_type: str
    underlying: str
    price: float
    confidence: float
    reasoning: str
    rejection_reason: str


def make_health_handler(trading_system):
    """Return a BaseHTTPRequestHandler subclass that has access to `trading_system`."""
    class _HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/health':
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            try:
                # Use Kolkata time for uptime/timestamps
                now = datetime.now(tz=trading_system.tz)
                equity = trading_system._equity({})
                exposure = trading_system._current_total_exposure({})

                uptime = 0
                if getattr(trading_system, 'session_start', None):
                    # session_start should be tz-aware
                    ss = trading_system.session_start
                    if ss.tzinfo is None:
                        ss = ss.replace(tzinfo=ZoneInfo("UTC")).astimezone(trading_system.tz)
                    uptime = (now - ss).total_seconds()

                health_data = {
                    'status': 'healthy' if trading_system.is_running else 'stopped',
                    'timestamp': now.isoformat(),
                    'cash': trading_system.cash,
                    'equity': equity,
                    'exposure': exposure,
                    'open_trades': len(trading_system.open_trades),
                    'closed_trades': len(trading_system.closed_trades),
                    'daily_pnl': trading_system.daily_pnl,
                    'max_drawdown': trading_system.max_drawdown,
                    'uptime_seconds': uptime
                }
                self.wfile.write(json.dumps(health_data, indent=2).encode())
            except Exception as e:
                logger.exception("Health endpoint error: %s", e)
                error_data = {'status': 'error', 'message': str(e)}
                self.wfile.write(json.dumps(error_data).encode())

        # suppress console logging per-request (optional)
        def log_message(self, format, *args):
            return

    return _HealthHandler


class HealthMetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for health and metrics endpoint."""
    
    def __init__(self, trading_system, *args, **kwargs):
        self.trading_system = trading_system
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get current metrics
            try:
                equity = self.trading_system._equity({})
                exposure = self.trading_system._current_total_exposure({})
                
                health_data = {
                    'status': 'healthy' if self.trading_system.is_running else 'stopped',
                    'timestamp': datetime.now().isoformat(),
                    'cash': self.trading_system.cash,
                    'equity': equity,
                    'exposure': exposure,
                    'open_trades': len(self.trading_system.open_trades),
                    'closed_trades': len(self.trading_system.closed_trades),
                    'daily_pnl': self.trading_system.daily_pnl,
                    'max_drawdown': self.trading_system.max_drawdown,
                    'uptime': (datetime.now() - self.trading_system.session_start).total_seconds() if self.trading_system.session_start else 0
                }
                
                self.wfile.write(json.dumps(health_data, indent=2).encode())
            except Exception as e:
                error_data = {'status': 'error', 'message': str(e)}
                self.wfile.write(json.dumps(error_data).encode())
        else:
            self.send_response(404)
            self.end_headers()

class LivePaperTradingSystem:
    def __init__(self, initial_capital: float = 20000.0, max_risk_per_trade: float = 0.02,
                 confidence_cutoff: float = 40.0, exposure_limit: float = 0.6,
                 max_daily_loss_pct: float = 0.03, commission_bps: float = 1.0,
                 slippage_bps: float = 5.0, symbols: List[str] = None,
                 data_provider: str = 'fyers', stop_loss_pct: float = -30.0,
                 take_profit_pct: float = 25.0, time_stop_minutes: int = 30,
                 verbose: bool = False):
        """Initialize the live paper trading system."""
        self.initial_capital = initial_capital
        
        # Proper capital accounting
        self.cash = float(initial_capital)
        self.margin_used = 0.0
        self.fees_paid = 0.0
        
        # Legacy fields (deprecated - use cash instead)
        self.current_capital = initial_capital
        
        # Performance tracking
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        
        self.max_risk_per_trade = max_risk_per_trade
        self.confidence_cutoff = confidence_cutoff
        self.exposure_limit = exposure_limit
        self.max_daily_loss_pct = max_daily_loss_pct
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.symbols = symbols or ['NSE:NIFTY50-INDEX']
        self.data_provider = data_provider
        
        # Configurable exit rules
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.time_stop_minutes = time_stop_minutes
        
        # Logging control
        self.verbose = verbose
        
        # Timezone handling
        self.tz = ZoneInfo("Asia/Kolkata")
        
        # Health monitoring
        self.health_server = None
        self.health_port = 8081  # Changed from 8080 to avoid conflicts
        
        # Trading state
        self.is_running = False
        self.open_trades = {}
        self.closed_trades = []
        self.rejected_signals = []  # Initialize rejected signals list
        self.total_signals_generated = 0
        self.total_signals_rejected = 0
        self.total_trades_executed = 0
        self.total_trades_closed = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        self.eod_exit_triggered = False  # Flag to block new trades after 15:20
        self.session_start = None  # Will be set in start_trading()
        self._last_metrics_count = 0  # Track last metrics update
        self._signal_dedupe_cache = {}  # Persistent signal deduplication cache
        
        # Data management
        self.data_manager = None
        self.data_cache = {}
        self.cache_duration = 300  # 5 minutes cache for historical data
        
        # Price caching for API rate limiting
        self.price_cache = {}
        self.price_cache_duration = 5  # 5 seconds cache for live prices
        
        # Last bar timestamps for signal deduplication
        self._last_bar_ts = {}
        
        # Open position tracking for deduplication
        self._open_keys = set()
        
        # Database
        self.db = UnifiedTradingDatabase("unified_trading.db")
        
        # Threading safety
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._trading_thread = None
        
        # Initialize data provider
        self._initialize_data_provider()
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Reset trading state to ensure clean start
        self._reset_trading_state()
        
        logger.info(f"🚀 Live Paper Trading System initialized with ₹{initial_capital:,.2f} capital")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")
        logger.info(f"🎯 Strategies: {', '.join(self._get_available_strategies())}")
        logger.info(f"📛 Risk: {max_risk_per_trade*100:.1f}% per trade, {exposure_limit*100:.1f}% max exposure")
        logger.info(f"📉 Daily loss limit: {max_daily_loss_pct*100:.1f}%")

        # Strategy-specific deduplication TTLs (in seconds)
        self.strategy_dedupe_ttls = {
            'ema_crossover_enhanced': 300,  # 5 minutes
            'supertrend_ema': 180,          # 3 minutes  
            'supertrend_macd_rsi_ema': 240  # 4 minutes
        }
        
        # Default TTL for unknown strategies
        self.default_dedupe_ttl = 300  # 5 minutes

        # Performance metrics update tracking
        self._last_metrics_count = 0
        self._last_metrics_time = None
        self._metrics_update_interval = 900  # 15 minutes

    def _equity(self, last_prices: Dict[str, float] = None) -> float:
        """Calculate mark-to-market equity."""
        if last_prices is None:
            last_prices = {}
        
        mtm = 0.0
        for trade in self.open_trades.values():
            # Use last price if available, otherwise use entry price
            lp = last_prices.get(trade.contract_symbol, trade.entry_price)
            mtm += (lp - trade.entry_price) * trade.quantity
        
        return self.cash + mtm

    def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            # Use the unified strategy engine instead of individual strategies
            from src.core.unified_strategy_engine import UnifiedStrategyEngine
            self.strategy_engine = UnifiedStrategyEngine(self.symbols, self.confidence_cutoff)
            logger.info(f"✅ Unified Strategy Engine initialized with {len(self.symbols)} symbols")
        except Exception as e:
            logger.error(f"❌ Error initializing strategy engine: {e}")
            raise

    def _reset_trading_state(self):
        """Reset trading state to clear any phantom data."""
        logger.info("🔄 Resetting trading state...")
        
        # Reset capital accounting
        self.cash = float(self.initial_capital)
        self.margin_used = 0.0
        self.fees_paid = 0.0
        
        # Reset performance tracking
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        
        # Clear all trades
        self.open_trades.clear()
        self.closed_trades.clear()
        self.rejected_signals.clear()
        self._open_keys.clear()
        
        # Reset performance metrics
        self.total_signals_generated = 0
        self.total_signals_rejected = 0
        self.total_trades_executed = 0
        self.total_trades_closed = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.eod_exit_triggered = False
        self.session_start = None
        
        # Reset daily limits
        self.daily_pnl = 0.0
        self.daily_loss_limit_hit = False
        
        # Clear caches
        self.data_cache.clear()
        self.price_cache.clear()
        self._last_bar_ts.clear()
        
        logger.info("✅ Trading state reset complete")

    def _initialize_data_provider(self):
        """Initialize the data provider."""
        # Ensure database is initialized
        logger.info("🗄️ Initializing database tables...")
        self.db.init_database()
        
        # Initialize Fyers client with automated authentication
        if self.data_provider == 'fyers':
            try:
                logger.info("🔐 Initializing Fyers with automated authentication...")
                
                # Run automated authentication script
                import subprocess
                import sys
                result = subprocess.run([sys.executable, 'automated_fyers_auth.py'], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    raise Exception(f"Authentication failed: {result.stderr}")
                
                logger.info("✅ Using Fyers live data with fresh token")
                
                # Initialize Fyers client
                from src.api.fyers import FyersClient
                self.data_manager = FyersClient()
                
                # Initialize the client with the access token
                if not self.data_manager.initialize_client():
                    raise Exception("Failed to initialize Fyers client")
                
                self.data_provider = self.data_manager
                
                # Initialize WebSocket for real-time data
                try:
                    from src.api.fyers_websocket import FyersWebSocketManager
                    from src.config.settings import FYERS_CLIENT_ID
                    
                    # Get access token from Fyers client
                    access_token = self.data_manager.access_token
                    if access_token:
                        self.websocket_manager = FyersWebSocketManager(access_token, FYERS_CLIENT_ID)
                        
                        # Add callbacks for real-time data
                        self.websocket_manager.add_price_callback(self._on_price_update)
                        self.websocket_manager.add_candle_callback(self._on_candle_update)
                        
                        # Connect to WebSocket
                        if self.websocket_manager.connect(self.symbols):
                            logger.info("🔌 WebSocket connected for real-time data streaming")
                        else:
                            logger.warning("⚠️ WebSocket connection failed, falling back to REST API")
                            self.websocket_manager = None
                    else:
                        logger.warning("⚠️ No access token available for WebSocket")
                        self.websocket_manager = None
                        
                except ImportError as e:
                    logger.warning(f"⚠️ WebSocket not available: {e}, using REST API only")
                    self.websocket_manager = None
                except Exception as e:
                    logger.warning(f"⚠️ WebSocket initialization failed: {e}, using REST API only")
                    self.websocket_manager = None
                
            except Exception as e:
                logger.error(f"❌ Fyers initialization failed: {e}")
                logger.error("❌ System requires live data from Fyers to operate")
                raise Exception(f"Fyers initialization failed: {e}")
        else:
            logger.error("❌ Only 'fyers' data provider is supported for live trading")
            raise Exception("Only 'fyers' data provider is supported for live trading")
    
    def _on_price_update(self, symbol: str, price: float, timestamp):
        """Callback for real-time price updates"""
        try:
            # Update price cache
            self.price_cache[symbol] = (price, time.time())
            
            # Update live prices for exposure calculation
            if hasattr(self, 'live_prices'):
                self.live_prices[symbol] = price
            
            if self.verbose:
                logger.debug(f"📈 Real-time price update: {symbol} = ₹{price:,.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error in price update callback: {e}")
    
    def _on_candle_update(self, symbol: str, candles):
        """Callback for real-time candle updates"""
        try:
            if candles and len(candles) > 0:
                latest_candle = candles[-1]
                if self.verbose:
                    logger.debug(f"📊 Real-time candle update: {symbol} = ₹{latest_candle[4]:,.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Error in candle update callback: {e}")
    
    def _get_price_cached(self, symbol: str, ttl: int = 5) -> Optional[float]:
        """Canonical cached price helper. Stores timestamps as epoch seconds."""
        now = time.time()
        cached = self.price_cache.get(symbol)
        if cached:
            price, ts = cached
            if now - ts < ttl:
                if self.verbose:
                    logger.debug(f"Using cached price for {symbol}: {price} (age {now-ts:.1f}s)")
                return price

        # Try WebSocket first for real-time data
        if hasattr(self, 'websocket_manager') and self.websocket_manager and self.websocket_manager.is_healthy():
            live_price = self.websocket_manager.get_live_price(symbol)
            if live_price is not None and live_price > 0:
                self.price_cache[symbol] = (live_price, now)
                if self.verbose:
                    logger.debug(f"📡 WebSocket price for {symbol}: {live_price}")
                return live_price

        # Fallback to REST API
        try:
            price = self.data_manager.get_underlying_price(symbol)
            if price is not None and price > 0:
                self.price_cache[symbol] = (price, now)
                if self.verbose:
                    logger.debug(f"🌐 REST API price for {symbol}: {price}")
                return price
        except Exception as e:
            logger.debug(f"Error fetching price for {symbol}: {e}")
        return None

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to price with symmetric calculation."""
        bps = float(self.slippage_bps) / 10000.0
        if is_buy:
            return price * (1.0 + bps)
        else:
            return price * max(0.0, (1.0 - bps))

    def _commission_amount(self, notional: float) -> float:
        """Calculate commission amount."""
        return notional * (self.commission_bps / 10000)

    def _current_total_exposure(self, last_prices: Dict[str, float] = None) -> float:
        """Calculate current total exposure using market value of positions."""
        with self._lock:
            if last_prices is None:
                last_prices = {}
            
            # Compute market value of open positions using current option LTP if available,
            # otherwise fall back to entry price
            total_market_value = 0.0
            for trade in self.open_trades.values():
                # Try to get current option price, fallback to entry price
                ltp = last_prices.get(trade.contract_symbol, trade.entry_price)
                total_market_value += ltp * trade.quantity

            # Calculate NAV (Net Asset Value)
            nav = self.cash + total_market_value
            if nav <= 0:
                return float('inf')
            
            exposure = total_market_value / nav
            
            logger.debug(f"🔍 Exposure: Market Value: ₹{total_market_value:,.2f}, Cash: ₹{self.cash:,.2f}, NAV: ₹{nav:,.2f}, Exposure: {exposure:.2%}")
            
            return exposure

    def _execute_with_lock(self, operation, *args, **kwargs):
        """Execute an operation with proper lock handling."""
        with self._lock:
            return operation(*args, **kwargs)

    def _should_open_trade(self, signal: Dict) -> Tuple[bool, str]:
        """Check if we should open a trade. Returns (should_open, reason)."""
        # Respect daily stop
        if self.daily_loss_limit_hit:
            return False, "Daily loss limit breached"

        # Confidence threshold
        if float(signal.get('confidence', 0.0)) < self.confidence_cutoff:
            return False, f"Confidence {signal.get('confidence', 0.0)} below threshold {self.confidence_cutoff}"

        # Exposure check disabled for 1 lot trading
        # current_exposure = self._current_total_exposure()
        # logger.debug(f"Current exposure: {current_exposure:.2%}, Limit: {self.exposure_limit:.2%}")
        # 
        # if current_exposure >= self.exposure_limit:
        #     return False, f"Exposure {current_exposure:.2%} at limit {self.exposure_limit:.2%}"

        return True, "Signal accepted"

    def _select_option_contract(self, signal: Dict, current_price: float) -> Optional[Any]:
        """Select option contract using REAL option chain data from Fyers."""
        try:
            symbol = signal['symbol']
            signal_type = signal['signal']
            
            # Get REAL option chain data from Fyers
            option_chain = self.data_provider.get_option_chain(symbol)
            
            # Accumulate options data for backtesting
            self._accumulate_options_data(symbol, option_chain)
            
            if not option_chain:
                logger.warning(f"⚠️ Could not fetch real option chain for {symbol}, using fallback")
                return self._create_fallback_option_contract(signal, current_price)
            
            # Use real option data
            atm_strike = option_chain['atm_strike']
            expiry_date = option_chain['expiry_date']
            
            # Determine option type and get real premium
            if 'CALL' in signal_type:
                option_type = OptionType.CALL
                if 'CE' in option_chain['options']:
                    real_premium = option_chain['options']['CE']['ltp']
                    option_symbol = option_chain['options']['CE']['symbol']
                else:
                    logger.warning(f"⚠️ No CE option found in real data for {symbol}")
                    return self._create_fallback_option_contract(signal, current_price)
            elif 'PUT' in signal_type:
                option_type = OptionType.PUT
                if 'PE' in option_chain['options']:
                    real_premium = option_chain['options']['PE']['ltp']
                    option_symbol = option_chain['options']['PE']['symbol']
                else:
                    logger.warning(f"⚠️ No PE option found in real data for {symbol}")
                    return self._create_fallback_option_contract(signal, current_price)
            else:
                logger.error(f"❌ Invalid signal type: {signal_type}")
                return None
            
            # Get lot size based on symbol
            if 'NIFTY50' in symbol:
                lot_size = 50
            elif 'NIFTYBANK' in symbol:
                lot_size = 25
            else:
                lot_size = 50  # Default
            
            # Create contract with REAL data
            contract = OptionContract(
                symbol=option_symbol,
                underlying=symbol,
                strike=atm_strike,
                expiry=datetime.strptime(expiry_date, '%Y-%m-%d'),
                option_type=option_type,
                lot_size=lot_size,
                bid=option_chain['options'].get('CE' if option_type == OptionType.CALL else 'PE', {}).get('bid', real_premium * 0.95),
                ask=option_chain['options'].get('CE' if option_type == OptionType.CALL else 'PE', {}).get('ask', real_premium * 1.05),
                last=real_premium,
                volume=option_chain['options'].get('CE' if option_type == OptionType.CALL else 'PE', {}).get('volume', 1000),
                open_interest=option_chain['options'].get('CE' if option_type == OptionType.CALL else 'PE', {}).get('oi', 5000),
                implied_volatility=0.25,
                delta=0.5 if option_type == OptionType.CALL else -0.5,
                gamma=0.01,
                theta=-real_premium * 0.1,
                vega=real_premium * 0.5
            )
            
            logger.info(f"✅ Using REAL option data: {option_type.value} Strike ₹{atm_strike:,.0f}, Premium ₹{real_premium:.2f}")
            return contract

        except Exception as e:
            logger.error(f"❌ Error selecting option contract: {e}")
            return self._create_fallback_option_contract(signal, current_price)

    def _create_fallback_option_contract(self, signal: Dict, current_price: float) -> Optional[Any]:
        """Create fallback option contract when real data is unavailable."""
        try:
            symbol = signal['symbol']
            signal_type = signal['signal']
            
            # Calculate ATM strike
            atm_strike = round(current_price / 50) * 50
            
            # Determine option type
            if 'CALL' in signal_type:
                option_type = OptionType.CALL
                # Use more realistic premium calculation for ATM options
                premium = current_price * 0.008  # 0.8% of underlying for ATM options
            elif 'PUT' in signal_type:
                option_type = OptionType.PUT
                # Use more realistic premium calculation for ATM options
                premium = current_price * 0.008  # 0.8% of underlying for ATM options
            else:
                logger.error(f"❌ Invalid signal type: {signal_type}")
                return None
            
            # Get lot size based on symbol
            if 'NIFTY50' in symbol:
                lot_size = 50
            elif 'NIFTYBANK' in symbol:
                lot_size = 25
            else:
                lot_size = 50  # Default
            
            # Create contract with fallback data
            contract = OptionContract(
                symbol=f"{symbol.replace(':', '')}{self.now_kolkata().strftime('%d%m%y')}{atm_strike}{'CE' if option_type == OptionType.CALL else 'PE'}",
                underlying=symbol,
                strike=atm_strike,
                expiry=self.now_kolkata() + timedelta(days=7),  # Weekly expiry
                option_type=option_type,
                lot_size=lot_size,
                bid=premium * 0.95,  # 5% spread
                ask=premium * 1.05,
                last=premium,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                delta=0.5 if option_type == OptionType.CALL else -0.5,
                gamma=0.01,
                theta=-premium * 0.1,
                vega=premium * 0.5
            )
            
            logger.warning(f"⚠️ Created FALLBACK option contract: {option_type.value} Strike ₹{atm_strike:,.0f}, Premium ₹{premium:.2f}")
            return contract

        except Exception as e:
            logger.error(f"❌ Error creating fallback option contract: {e}")
            return None

    def _open_paper_trade(self, signal: Dict, option_contract: Any, entry_price: float, timestamp: datetime, last_prices: Dict[str, float] = None) -> Optional[str]:
        """Open a new paper trade with proper cash accounting and consolidation."""
        with self._lock:
            try:
                # Calculate position size with proper risk management
                risk_amount = self.cash * self.max_risk_per_trade
                confidence = float(signal.get('confidence', 50))
                confidence_multiplier = min(max(confidence / 50.0, 0.5), 1.5)
                adjusted_risk = risk_amount * confidence_multiplier

                premium_per_lot = entry_price * option_contract.lot_size
                
                # Always use exactly 1 lot per trade for simplicity
                lots = 1
                quantity_shares = lots * option_contract.lot_size
                
                # Check if we can afford 1 lot
                available_capital = self.cash * 0.9
                if premium_per_lot > available_capital:
                    logger.info(f"⚠️ Cannot afford 1 lot for {signal['strategy']} - need ₹{premium_per_lot:,.2f}, have ₹{available_capital:,.2f}")
                    
                    # Log capital rejection to database
                    rejection_data = {
                        'timestamp': timestamp,
                        'symbol': signal['symbol'],
                        'strategy': signal['strategy'],
                        'signal_type': signal['signal'],
                        'confidence': signal.get('confidence', 0),
                        'required_capital': premium_per_lot,
                        'available_capital': available_capital,
                        'capital_shortfall': premium_per_lot - available_capital,
                        'option_premium': entry_price,
                        'lot_size': option_contract.lot_size,
                        'total_cost_per_lot': premium_per_lot
                    }
                    self.db.save_capital_rejection_log(rejection_data)
                    
                    return None

                # Apply slippage and calculate costs
                exec_price = self._apply_slippage(entry_price, is_buy=True)
                entry_value = exec_price * quantity_shares
                entry_commission = self._commission_amount(entry_value)

                # Check position limits
                if len(self.open_trades) >= 8:
                    logger.warning(f"🚫 Max total positions reached (8)")
                    return None

                symbol = signal['symbol']
                symbol_positions = sum(1 for t in self.open_trades.values() if t.underlying == symbol)
                if symbol_positions >= 3:
                    logger.warning(f"🚫 Max positions per symbol reached (3) for {symbol}")
                    return None

                # Check for duplicate positions
                key = (signal['strategy'], symbol, 'CALL' if 'CALL' in signal['signal'] else 'PUT')
                if key in self._open_keys:
                    logger.warning(f"🚫 Position already open for {key}")
                    return None

                # Check exposure limits
                if not self._check_exposure_limits(signal['symbol'], entry_value, last_prices):
                    logger.warning(f"🚫 Exposure limit exceeded for {signal['symbol']}")
                    return None

                # CASH BOOKKEEPING: Deduct entry_value + commission
                total_cash_out = entry_value + entry_commission
                if total_cash_out > self.cash:
                    logger.warning(f"🚫 Insufficient cash: need ₹{total_cash_out:,.2f}, have ₹{self.cash:,.2f}")
                    return None

                self.cash -= total_cash_out
                self.fees_paid += entry_commission

                # Get contract symbol for proper consolidation
                contract_sym = option_contract.symbol

                # Check for duplicate positions by contract symbol
                key = self._make_open_key(signal['strategy'], contract_sym, option_contract.option_type.value)
                if key in self._open_keys:
                    logger.warning(f"🚫 Position already open for {key}")
                    return None

                # Consolidation: If contract already exists, aggregate
                for trade in self.open_trades.values():
                    if (trade.contract_symbol == contract_sym and 
                        trade.status == 'OPEN' and 
                        trade.signal_type == signal['signal']):
                        
                        # Weighted average entry price
                        old_qty = trade.quantity
                        new_qty = old_qty + quantity_shares
                        trade.entry_price = (trade.entry_price * old_qty + exec_price * quantity_shares) / new_qty
                        trade.quantity = new_qty
                        
                        # Store original entry_value/commission cumulatively
                        trade.entry_value = getattr(trade, 'entry_value', trade.entry_price * old_qty) + entry_value
                        trade.entry_commission = getattr(trade, 'entry_commission', 0.0) + entry_commission
                        
                        logger.info(f"🔄 Consolidated trade {trade.id[:8]}: {lots} lots added, new qty: {new_qty}")
                        return trade.id

                # Create new trade
                trade_id = str(uuid.uuid4())
                trade = PaperTrade(
                    id=trade_id,
                    timestamp=timestamp,
                    contract_symbol=contract_sym,
                    underlying=option_contract.underlying,
                    strategy=signal['strategy'],
                    signal_type=signal['signal'],
                    entry_price=exec_price,
                    quantity=quantity_shares,
                    lot_size=option_contract.lot_size,
                    strike=option_contract.strike,
                    expiry=option_contract.expiry,
                    option_type=option_contract.option_type.value,
                    status='OPEN',
                    commission=entry_commission,
                    confidence=signal.get('confidence', 0),
                    reasoning=signal.get('reasoning', ''),
                    stop_loss=signal.get('stop_loss'),
                    target1=signal.get('target1'),
                    target2=signal.get('target2'),
                    target3=signal.get('target3'),
                    entry_time=timestamp  # Set entry time for proper time tracking
                )
                
                # Add bookkeeping fields
                trade.entry_value = entry_value
                trade.entry_commission = entry_commission

                # Store trade
                self.open_trades[trade_id] = trade
                self._open_keys.add(key)

                self.total_trades_executed += 1

                # Log structured trade event
                self._log_trade_event('trade_opened', {
                    'id': trade_id,
                    'strategy': signal['strategy'],
                    'contract_symbol': trade.contract_symbol,
                    'entry_price': exec_price,
                    'quantity': trade.quantity
                })

                logger.info(f"✅ Opened trade {trade_id[:8]}... | {signal['signal']} {signal['strategy']}")
                logger.info(f"   Symbol: {signal['symbol']} | Lots: {lots} | Shares: {quantity_shares} | Price: ₹{exec_price:.2f}")
                logger.info(f"   Cash Out: ₹{total_cash_out:,.2f} | Commission: ₹{entry_commission:.2f}")
                logger.info(f"   Remaining Cash: ₹{self.cash:,.2f}")

                # Update drawdown and peak capital using full equity calculation
                # Build complete last_prices map for accurate equity calculation
                full_last_prices = {}
                for existing_trade in self.open_trades.values():
                    full_last_prices[existing_trade.contract_symbol] = existing_trade.entry_price
                full_last_prices[trade.contract_symbol] = exec_price
                
                current_equity = self._equity(full_last_prices)
                if current_equity > self.peak_capital:
                    self.peak_capital = current_equity
                elif self.peak_capital > 0:
                    drawdown_pct = self.safe_pct(self.peak_capital - current_equity, self.peak_capital)
                    self.max_drawdown = max(self.max_drawdown, drawdown_pct)

                return trade_id

            except Exception as e:
                logger.error(f"❌ Error opening trade: {e}")
                return None
        
        # Outside lock: save to database
        try:
            self.db.save_open_option_position(trade)
        except Exception as e:
            logger.exception("Failed to save trade to database")
    
    def _calculate_dynamic_position_size(self, signal: Dict, entry_price: float) -> int:
        """Calculate position size with dynamic lot sizing based on confidence and volatility."""
        try:
            # Base risk amount (2% of capital)
            base_risk = self.cash * self.max_risk_per_trade
            
            # Confidence multiplier (0.5 to 2.0 based on confidence)
            confidence = signal.get('confidence', 50)
            confidence_multiplier = max(0.5, min(2.0, confidence / 50.0))
            
            # Volatility adjustment (if we have ATR data)
            volatility_multiplier = 1.0
            if 'atr' in signal:
                # Adjust based on ATR - higher volatility = smaller position
                atr = signal['atr']
                if atr and atr > 0:
                    volatility_multiplier = max(0.5, min(1.5, 1.0 / (atr / entry_price)))
            
            # Calculate adjusted risk
            adjusted_risk = base_risk * confidence_multiplier * volatility_multiplier
            
            # For options, use a more conservative risk calculation
            # Risk per unit should be the option premium (entry_price)
            risk_per_unit = entry_price
            
            if risk_per_unit <= 0:
                return 0
            
            # Calculate position size (how many option contracts we can buy)
            position_size = int(adjusted_risk / risk_per_unit)
            
            # Limit position size to reasonable values for options trading
            max_position = 10  # Maximum 10 contracts
            min_position = 1   # Minimum 1 contract
            
            position_size = max(min_position, min(max_position, position_size))
            
            # Add debugging
            logger.info(f"🔍 DEBUG: Position size calculation:")
            logger.info(f"   Base risk: ₹{base_risk:,.2f}")
            logger.info(f"   Confidence: {confidence}, Multiplier: {confidence_multiplier}")
            logger.info(f"   Entry price: ₹{entry_price:,.2f}")
            logger.info(f"   Risk per unit: ₹{risk_per_unit:,.2f}")
            logger.info(f"   Raw position size: {int(adjusted_risk / risk_per_unit)}")
            logger.info(f"   Final position size: {position_size}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            return 0
    
    def _check_exposure_limits(self, symbol: str, new_notional: float, last_prices: Dict[str, float] = None) -> bool:
        """Check exposure limits using equity-based calculation."""
        # DISABLED: Always allow trades for 1 lot trading
        return True

    def _close_paper_trade(self, trade_id: str, exit_price: float,
                          exit_reason: str, timestamp: datetime) -> Optional[PaperTrade]:
        """Close a paper trade with proper P&L and cash accounting.
        Returns the closed PaperTrade object or None.
        Does DB update outside the critical lock.
        """
        # Normalize timestamp to Kolkata tz
        if timestamp.tzinfo is None:
            # assume timestamp is local/UTC depending on your source; here we treat naive as UTC
            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC")).astimezone(self.tz)
        else:
            timestamp = timestamp.astimezone(self.tz)

        # We'll prepare db_payload to update DB after releasing lock
        db_payload = None
        closed_trade = None

        with self._lock:
            if trade_id not in self.open_trades:
                return None

            trade = self.open_trades[trade_id]

            # Determine whether this is effectively a sell (closing a long) or buy (closing a short)
            # Convention: quantity > 0 => long (we sold to close), quantity < 0 => short (we bought to close)
            is_buy = False
            if getattr(trade, 'quantity', 0) < 0:
                # we are closing a short by buying
                is_buy = True

            # Apply slippage with correct direction (closing a long => we sell, so is_buy=False)
            exec_price = self._apply_slippage(exit_price, is_buy=is_buy)

            # Compute commissions / values
            exit_value = exec_price * abs(trade.quantity)
            exit_commission = self._commission_amount(exit_value)

            # entry_value might be stored as cumulative cost; fallback to entry_price * abs(q)
            entry_value = getattr(trade, 'entry_value', trade.entry_price * abs(trade.quantity))
            entry_commission = getattr(trade, 'entry_commission', 0.0)

            # P&L: (exit - entry) * signed_quantity - fees
            # Using abs(quantity) where appropriate but preserving direction via sign for P&L
            signed_qty = trade.quantity  # positive long, negative short
            pnl = (exec_price - trade.entry_price) * signed_qty - (entry_commission + exit_commission)

            # Safe returns calculation using absolute invested amount
            invested_abs = abs(trade.entry_price * signed_qty)
            returns_pct = self.safe_pct(pnl, invested_abs)

            # CASH bookkeeping:
            # For a long: cash was reduced by entry_value at open; on close we add exit proceeds (exec_price * qty) minus exit commission
            # For a short (if supported), logic depends on how you accounted margin; ensure symmetry with open logic.
            self.cash += (exec_price * signed_qty) - exit_commission
            self.fees_paid += exit_commission

            # Update trade fields
            trade.exit_price = exec_price
            trade.exit_time = timestamp
            trade.pnl = pnl
            trade.exit_reason = exit_reason
            trade.status = 'CLOSED'

            # Move trade to closed_trades and remove from open_trades/_open_keys
            self.closed_trades.append(trade)
            del self.open_trades[trade_id]
            key = self._make_open_key(trade.strategy, trade.contract_symbol, trade.option_type)
            self._open_keys.discard(key)

            # update daily pnl in-memory
            self.daily_pnl += pnl

            # prepare DB payload (update outside lock)
            db_payload = (trade_id, 'CLOSED', pnl, exit_reason)

            closed_trade = trade

        # outside lock: update DB and compute drawdown using fresh equity snapshot
        try:
            # Update DB; tolerate DB errors
            try:
                self.db.update_option_position_status(*db_payload)
            except Exception:
                logger.exception("Failed to update DB for closed trade %s", trade_id)

            # Update peak / drawdown using a best-effort current prices map
            last_prices = {}
            for t in self.open_trades.values():
                # try to get real LTP if available; fallback to entry
                ltp = None
                try:
                    ltp = self._get_option_ltp(t.contract_symbol)
                except Exception:
                    ltp = None
                last_prices[t.contract_symbol] = ltp or t.entry_price

            # compute equity
            equity = self._equity(last_prices)
            if equity > self.peak_capital:
                self.peak_capital = equity
            elif self.peak_capital > 0:
                drawdown_pct = self.safe_pct(self.peak_capital - equity, self.peak_capital)
                self.max_drawdown = max(self.max_drawdown, drawdown_pct)

            # Check daily loss limit
            if self.daily_pnl < -(self.initial_capital * self.max_daily_loss_pct):
                self.daily_loss_limit_hit = True
                logger.warning(f"🚫 Daily loss limit breached: PnL={self.daily_pnl:.2f}")

            logger.info(f"🔒 Closed paper trade: {trade_id} | P&L: ₹{pnl:+.2f} ({returns_pct:+.2f}%)")
            logger.info(f"   Cash: ₹{self.cash:,.2f} | Fees Paid: ₹{self.fees_paid:,.2f}")

            # Log structured trade event
            self._log_trade_event('trade_closed', {
                'id': trade_id,
                'strategy': trade.strategy,
                'contract_symbol': trade.contract_symbol,
                'exit_price': exec_price,
                'quantity': trade.quantity,
                'pnl': pnl,
                'exit_reason': exit_reason
            })

        except Exception:
            logger.exception("Unexpected error while finalizing close for trade %s", trade_id)

        return closed_trade

    def _close_all_trades(self, reason: str, timestamp: datetime, current_prices: Dict[str, float]) -> List[PaperTrade]:
        """Close all open trades with the given reason. Returns list of closed trades."""
        closed_trades = []
        for trade_id in list(self.open_trades.keys()):
            trade = self.open_trades[trade_id]
            ltp = current_prices.get(trade.contract_symbol, trade.entry_price)
            result = self._close_paper_trade(trade_id, ltp, reason, timestamp)
            if result:
                closed_trades.append(result)
        return closed_trades

    def _check_trade_exits(self, current_prices: Dict[str, float], timestamp: datetime) -> List[Dict]:
        """Check for trade exits with proper exit strategies."""
        closed_trades = []
        
        # Check if EOD exit was triggered
        if self.eod_exit_triggered:
            logger.info("🕐 EOD exit triggered, closing all positions.")
            closed_trades.extend(self._close_all_trades('EOD Exit', timestamp, current_prices))
            return closed_trades
        
        # Check for EOD exit (15:20 IST)
        IST = ZoneInfo("Asia/Kolkata")
        ist_time = timestamp.astimezone(IST) if timestamp.tzinfo else timestamp.replace(tzinfo=ZoneInfo("UTC")).astimezone(IST)
        if ist_time.hour == 15 and ist_time.minute >= 20:
            if not self.eod_exit_triggered:
                logger.info("🕐 EOD exit time - closing all positions and blocking new trades")
                self.eod_exit_triggered = True
            
            closed_trades.extend(self._close_all_trades('EOD Exit', timestamp, current_prices))
            return closed_trades
        
        # Check individual trade exits
        for trade_id, trade in list(self.open_trades.items()):
            if trade.contract_symbol not in current_prices:
                continue
            
            current_price = current_prices[trade.contract_symbol]
            exit_reason = None
            
            # Calculate price change percentage with safe division
            price_change_pct = self.safe_pct(current_price - trade.entry_price, trade.entry_price)
            
            # Stop Loss (configurable)
            if price_change_pct <= self.stop_loss_pct:
                exit_reason = f'Stop Loss ({self.stop_loss_pct:.0f}%)'
            
            # Take Profit (configurable)
            elif price_change_pct >= self.take_profit_pct:
                exit_reason = f'Take Profit ({self.take_profit_pct:.0f}%)'
            
            # Time-based exit (configurable)
            elif hasattr(trade, 'entry_time') and trade.entry_time and (timestamp - trade.entry_time).total_seconds() >= self.time_stop_minutes * 60:
                exit_reason = f'Time Stop ({self.time_stop_minutes}min)'
            
            # Close the trade if exit condition met
            if exit_reason:
                logger.info(f"🔒 Exit condition met for {trade_id[:8]}: {exit_reason} (Price: ₹{current_price:.2f}, Change: {price_change_pct:+.1f}%)")
                result = self._close_paper_trade(trade_id, current_price, exit_reason, timestamp)
                if result:
                    closed_trades.append(result)
        
        return closed_trades

    def _generate_signals(self, index_data: pd.DataFrame) -> List[Dict]:
        """Generate trading signals using unified strategy engine."""
        try:
            # Prepare data for unified engine - handle multiple symbols
            current_prices = {}
            data_dict = {}
            
            # Get data for all symbols
            for symbol in self.symbols:
                recent_data = self._get_recent_index_data(symbol)
                if recent_data is not None and not recent_data.empty:
                    data_dict[symbol] = recent_data
                    current_prices[symbol] = recent_data['close'].iloc[-1]
            
            if not data_dict:
                logger.debug("⚠️ No data available for signal generation")
                return []
            
            # Use unified strategy engine
            signals = self.strategy_engine.generate_signals(data_dict, current_prices)
            
            # Apply deduplication
            current_time = time.time()
            deduplicated_signals = []
            
            for signal in signals:
                # Create unique signal identifier
                signal_key = f"{signal['strategy']}_{signal['signal']}_{signal.get('current_price', 0):.0f}"
                
                # Skip if we've already seen this signal recently
                if signal_key in self._signal_dedupe_cache:
                    continue
                
                self._signal_dedupe_cache[signal_key] = current_time
                deduplicated_signals.append(signal)
            
            # Limit signals to prevent overwhelming the system
            max_signals_per_cycle = 5
            if len(deduplicated_signals) > max_signals_per_cycle:
                # Sort by confidence and take top signals
                deduplicated_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                deduplicated_signals = deduplicated_signals[:max_signals_per_cycle]
                logger.info(f"📊 Limited signals to top {max_signals_per_cycle} by confidence")
            
            if deduplicated_signals:
                logger.info(f"📊 Generated {len(deduplicated_signals)} signals")
            
            return deduplicated_signals
            
        except Exception as e:
            logger.error(f"❌ Error generating signals with unified engine: {e}")
            return []

    def _log_rejected_signal(self, signal: Dict, reason: str):
        """Log rejected signal with detailed reasoning for debugging."""
        try:
            rejected_signal = RejectedSignal(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                strategy=signal['strategy'],
                signal_type=signal['signal'],
                underlying=signal['symbol'],
                price=signal['price'],
                confidence=signal.get('confidence', 0),
                reasoning=signal.get('reasoning', ''),
                rejection_reason=reason
            )
            
            self.rejected_signals.append(rejected_signal)
            self.db.save_rejected_signal(rejected_signal)
            
            self.total_signals_rejected += 1
            
            logger.info(f"🚫 Rejected signal: {signal['strategy']} {signal['signal']} | Reason: {reason}")
            
        except Exception as e:
            logger.error(f"❌ Error logging rejected signal: {e}")
    
    def _log_signal_generation(self, symbol: str, signals: List[Dict]):
        """Log signal generation with proper timestamp."""
        if not signals:
            return
        
        self.total_signals_generated += len(signals)
        
        for signal in signals:
            # Use signal timestamp if available, otherwise use current time
            signal_timestamp = signal.get('timestamp', datetime.now())
            
            # Save to database with proper timestamp
            self.db.save_trading_signal({
                'symbol': symbol,
                'strategy': signal['strategy'],
                'signal_type': signal['signal'],
                'confidence': signal.get('confidence', 0),
                'reasoning': signal.get('reasoning', ''),
                'timestamp': signal_timestamp,
                'created_at': datetime.now()
            })
            
            logger.info(f"📡 Signal: {signal['strategy']} {signal['signal']} {symbol} "
                       f"(Confidence: {signal.get('confidence', 0):.1f})")
    
    def _get_current_option_price(self, contract_symbol: str, fallback_price: float = None) -> float:
        """Get current option price with consistent fallback logic."""
        try:
            # Try to get live LTP first
            ltp = self._get_option_ltp(contract_symbol)
            if ltp and ltp > 0:
                return ltp
        except Exception:
            pass
        
        # Fallback to provided price or entry price from open trade
        if fallback_price and fallback_price > 0:
            return fallback_price
        
        # Last resort: get entry price from open trade
        for trade in self.open_trades.values():
            if trade.contract_symbol == contract_symbol:
                return trade.entry_price
        
        # Ultimate fallback
        return 0.0

    def _update_performance_metrics(self):
        """Update performance metrics with proper equity calculation."""
        try:
            # Get current prices for equity calculation using unified method
            current_prices = {}
            for trade in self.open_trades.values():
                current_prices[trade.contract_symbol] = self._get_current_option_price(trade.contract_symbol)
            
            # Calculate proper equity (cash + unrealized P&L)
            equity = self._equity(current_prices)
            
            # Calculate unrealized P&L from open trades using unified price sourcing
            unrealized_pnl = 0.0
            for trade in self.open_trades.values():
                current_price = self._get_current_option_price(trade.contract_symbol)
                if current_price > 0:
                    current_value = current_price * trade.quantity
                    entry_value = trade.entry_price * trade.quantity
                    unrealized_pnl += (current_value - entry_value)
            
            # Calculate realized P&L from closed trades
            realized_pnl = sum(trade.pnl for trade in self.closed_trades if trade.pnl is not None)
            
            # Update peak capital and drawdown using equity
            if equity > self.peak_capital:
                self.peak_capital = equity
            elif self.peak_capital > 0:
                drawdown_pct = self.safe_pct(self.peak_capital - equity, self.peak_capital)
                self.max_drawdown = max(self.max_drawdown, drawdown_pct)
            
            # Calculate win rate from closed trades
            total_closed_trades = len(self.closed_trades)
            if total_closed_trades > 0:
                winning_trades = sum(1 for trade in self.closed_trades if trade.pnl and trade.pnl > 0)
                win_rate = (winning_trades / total_closed_trades) * 100
            else:
                win_rate = 0.0
            
            # Log structured P&L update
            self._log_pnl_update(realized_pnl, unrealized_pnl, equity)
            
            # Log performance summary
            logger.info(f"📊 Performance Update:")
            logger.info(f"   Cash: ₹{self.cash:,.2f} | Equity: ₹{equity:,.2f}")
            logger.info(f"   Realized P&L: ₹{realized_pnl:+,.2f} | Unrealized P&L: ₹{unrealized_pnl:+,.2f}")
            logger.info(f"   Total Return: {self.safe_pct(equity - self.initial_capital, self.initial_capital):+.2f}%")
            logger.info(f"   Max Drawdown: {self.max_drawdown:.2f}% | Win Rate: {win_rate:.1f}%")
            logger.info(f"   Open Trades: {len(self.open_trades)} | Closed Trades: {total_closed_trades}")
            
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {e}")
    
    def _generate_session_report(self) -> Dict:
        """Generate comprehensive session report with proper equity calculation."""
        try:
            # Get current prices for equity calculation
            current_prices = {}
            for trade in self.open_trades.values():
                current_prices[trade.contract_symbol] = trade.entry_price
            
            # Calculate proper equity
            equity = self._equity(current_prices)
            realized_pnl = sum(trade.pnl for trade in self.closed_trades if trade.pnl is not None)
            unrealized_pnl = equity - self.cash
            
            # Calculate metrics
            total_closed_trades = len(self.closed_trades)
            if total_closed_trades > 0:
                winning_trades = sum(1 for trade in self.closed_trades if trade.pnl and trade.pnl > 0)
                win_rate = (winning_trades / total_closed_trades) * 100
                avg_pnl = realized_pnl / total_closed_trades
            else:
                winning_trades = 0
                win_rate = 0.0
                avg_pnl = 0.0
            
            # Strategy performance
            strategy_performance = {}
            for trade in self.closed_trades:
                strategy = trade.strategy
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'trades': 0, 'pnl': 0.0, 'wins': 0}
                strategy_performance[strategy]['trades'] += 1
                strategy_performance[strategy]['pnl'] += trade.pnl or 0.0
                if trade.pnl and trade.pnl > 0:
                    strategy_performance[strategy]['wins'] += 1
            
            return {
                'session_start': getattr(self, 'session_start', self.now_kolkata()),
                'session_end': self.now_kolkata(),
                'total_signals_generated': self.total_signals_generated,
                'total_signals_rejected': self.total_signals_rejected,
                'total_trades_executed': self.total_trades_executed,
                'total_trades_closed': total_closed_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_closed_trades - winning_trades,
                'win_rate': win_rate,
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': realized_pnl + unrealized_pnl,
                'avg_pnl': avg_pnl,
                'initial_capital': self.initial_capital,
                'final_cash': self.cash,
                'final_equity': equity,
                'return_pct': ((equity - self.initial_capital) / self.initial_capital * 100),
                'max_drawdown': self.max_drawdown,
                'fees_paid': self.fees_paid,
                'strategy_performance': strategy_performance,
                'open_trades': len(self.open_trades)
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating session report: {e}")
            return {}

    def _trading_loop(self):
        """Main trading loop with optimized logging."""
        logger.info("🔄 Trading loop started - entering main loop")
        
        while not self._stop_event.is_set():
            try:
                current_time = self.now_kolkata()
                
                # Reduced logging frequency - only log every 5 minutes
                if not hasattr(self, '_last_status_log') or (current_time - self._last_status_log).total_seconds() > 300:
                    logger.info(f"🔄 Trading loop iteration at {current_time.strftime('%H:%M:%S')}")
                    self._last_status_log = current_time
                
                if self._is_market_open():
                    # Only log market status every 5 minutes
                    if not hasattr(self, '_last_market_log') or (current_time - self._last_market_log).total_seconds() > 300:
                        logger.debug(f"📊 Market is open - processing signals at {current_time.strftime('%H:%M:%S')}")
                        self._last_market_log = current_time
                    
                    # Process signals for each symbol
                    for symbol in self.symbols:
                        try:
                            # Get live price with reduced logging
                            index_price = self._get_price_cached(symbol)
                            if not index_price:
                                logger.debug(f"⚠️ No price data for {symbol}")
                                continue
                            
                            # Only log price updates every 2 minutes
                            if not hasattr(self, f'_last_price_log_{symbol}') or (current_time - getattr(self, f'_last_price_log_{symbol}', current_time)).total_seconds() > 120:
                                logger.debug(f"📊 Live price for {symbol}: ₹{index_price:,.2f}")
                                setattr(self, f'_last_price_log_{symbol}', current_time)
                            
                            # Get recent data for signal generation
                            recent_data = self._get_recent_index_data(symbol)
                            if recent_data is None:
                                continue
                            
                            # Generate signals with reduced logging
                            signals = self._generate_signals(recent_data)
                            if signals:
                                logger.debug(f"📊 Generated {len(signals)} signals for {symbol}")
                                
                                # Process signals
                                for signal in signals:
                                    self._process_signal(signal, index_price)
                            
                        except Exception as e:
                            logger.error(f"❌ Error processing {symbol}: {e}")
                    
                    # Update performance metrics less frequently
                    if len(self.closed_trades) % 5 == 0 and len(self.closed_trades) != self._last_metrics_count:
                        self._update_performance_metrics()
                        self._last_metrics_count = len(self.closed_trades)
                    
                    # Check trade exits
                    self._check_trade_exits()
                    
                    # Log status less frequently
                    if not hasattr(self, '_last_status_update') or (current_time - self._last_status_update).total_seconds() > 60:
                        logger.debug(f"📊 Status: Cash: ₹{self.cash:,.2f}, Equity: ₹{self._equity():,.2f}, Exposure: {self._current_total_exposure():.1%}, Open Trades: {len(self.open_trades)}")
                        self._last_status_update = current_time
                
                # Sleep for trading interval
                time.sleep(self.trading_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(self.trading_interval)
        
        logger.info("🛑 Stop event received, exiting trading loop")

    def _is_market_open(self, current_time: Optional[datetime] = None) -> bool:
        """Check if market is open (market timezone = Asia/Kolkata)."""
        if current_time is None:
            current_time = self.now_kolkata()
        else:
            current_time = current_time.astimezone(self.tz) if current_time.tzinfo else current_time.replace(tzinfo=self.tz)

        if current_time.weekday() >= 5:  # Saturday/Sunday
            logger.debug(f"❌ Market closed: Weekend ({current_time.strftime('%A')})")
            return False
            
        market_start = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_open = market_start <= current_time <= market_end
        if not is_open:
            logger.debug(f"❌ Market closed: {current_time.strftime('%H:%M:%S')} (Market: {market_start.strftime('%H:%M')}-{market_end.strftime('%H:%M')})")
        else:
            logger.debug(f"✅ Market open: {current_time.strftime('%H:%M:%S')}")
        
        return is_open

    def _get_recent_index_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get recent index data for signal generation - only on new candles."""
        try:
            # Get historical data for indicators
            data = self.data_manager.get_historical_data(
                symbol=symbol,
                resolution="5",  # 5-minute data
                date_format=1,
                range_from=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                range_to=datetime.now().strftime('%Y-%m-%d'),
                cont_flag=1
            )
            
            if not data or 'candles' not in data or len(data['candles']) == 0:
                logger.warning(f"⚠️ No historical data available for {symbol}")
                return None
            
            # Convert to DataFrame with timezone handling
            df = pd.DataFrame(data['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert(ZoneInfo("Asia/Kolkata"))
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Check if we have new candle data
            latest_timestamp = df['timestamp'].iloc[-1]
            last_processed = self._last_bar_ts.get(symbol, pd.Timestamp(0, tz=ZoneInfo("Asia/Kolkata")))
            
            # In test mode or if no new candle, still allow signal generation but log it
            if latest_timestamp <= last_processed:
                logger.debug(f"⏭️ No new candle data for {symbol} - using existing data for signal generation")
                # Don't return None, continue with existing data
            
            # Update last processed timestamp
            self._last_bar_ts[symbol] = latest_timestamp
            
            # First, get current live price using cache
            try:
                current_price = self._get_price_cached(symbol)
                if not current_price or current_price <= 0:
                    logger.error(f"❌ Could not get live price for {symbol}")
                    return None
                
                logger.info(f"📊 Live price for {symbol}: ₹{current_price:,.2f}")
                
            except Exception as e:
                logger.error(f"❌ Failed to get live price for {symbol}: {e}")
                return None
            
            # Add technical indicators
            from src.core.indicators import add_technical_indicators
            df = add_technical_indicators(df)
            
            # Add current price as latest row
            current_row = pd.DataFrame([{
                'timestamp': self.now_kolkata(),
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'volume': 0
            }])
            
            df = pd.concat([df, current_row], ignore_index=True)
            
            logger.info(f"📊 Loaded {len(df)} candles for {symbol} (latest: {latest_timestamp})")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting recent index data for {symbol}: {e}")
            return None

    def _fetch_option_ltps_concurrent(self, contract_symbols: List[str]) -> Dict[str, float]:
        """Fetch option LTPs concurrently for better performance."""
        results = {}
        
        def fetch_single_ltp(contract_symbol: str) -> Tuple[str, Optional[float]]:
            try:
                ltp = self._get_option_ltp(contract_symbol)
                return contract_symbol, ltp
            except Exception as e:
                logger.debug(f"Failed to fetch LTP for {contract_symbol}: {e}")
                return contract_symbol, None
        
        # Use ThreadPoolExecutor for concurrent fetching
        with ThreadPoolExecutor(max_workers=min(10, len(contract_symbols))) as executor:
            future_to_symbol = {executor.submit(fetch_single_ltp, symbol): symbol 
                              for symbol in contract_symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, ltp = future.result()
                if ltp is not None:
                    results[symbol] = ltp
        
        return results

    def _get_option_ltp(self, contract_symbol: str) -> Optional[float]:
        """Get real option LTP from data provider."""
        try:
            # Try to get option price from data provider
            if hasattr(self.data_manager, 'get_option_price'):
                option_price = self.data_manager.get_option_price(contract_symbol)
                if option_price and option_price > 0:
                    return option_price
            
            # Fallback: try to get from option chain
            if hasattr(self.data_manager, 'get_option_chain'):
                # Extract underlying and strike from contract symbol
                # This is a simplified approach - you may need to parse the symbol properly
                option_chain = self.data_manager.get_option_chain(contract_symbol)
                if option_chain and 'ltp' in option_chain:
                    return option_chain['ltp']
            
            # Last resort: use entry price
            return None
            
        except Exception as e:
            logger.debug(f"Could not fetch option LTP for {contract_symbol}: {e}")
            return None

    def start_health_server(self):
        """Start health monitoring server."""
        try:
            handler_cls = make_health_handler(self)
            self.health_server = HTTPServer(('localhost', self.health_port), handler_cls)
            health_thread = threading.Thread(target=self.health_server.serve_forever, 
                                           daemon=True)
            health_thread.start()
            logger.info(f"🏥 Health server started on port {self.health_port}")
        except OSError as e:
            if "Address already in use" in str(e):
                logger.warning(f"⚠️ Health server port {self.health_port} in use, skipping health server")
            else:
                logger.warning(f"⚠️ Failed to start health server: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to start health server: {e}")

    def stop_health_server(self):
        """Stop health monitoring server."""
        if self.health_server:
            self.health_server.shutdown()
            self.health_server.server_close()
            logger.info("🏥 Health server stopped")

    def start_trading(self):
        """Start live paper trading."""
        logger.info("🚀 Starting live paper trading...")
        self.is_running = True
        self.session_start = datetime.now(self.tz) # Set session start time with timezone
        
        # Start health server
        self.start_health_server()
        
        # Start trading thread
        logger.info("🔄 Creating trading thread...")
        self._trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._trading_thread.start()
        
        # Wait a moment to ensure thread starts
        time.sleep(1)
        
        if self._trading_thread.is_alive():
            logger.info("✅ Trading thread started successfully")
        else:
            logger.error("❌ Trading thread failed to start")
            return False
        
        logger.info("✅ Live paper trading started successfully")

    def stop_trading(self):
        """Stop live paper trading and close all positions."""
        logger.info("🛑 Stopping live paper trading...")
        self.is_running = False
        self._stop_event.set()

        # Close all open trades using contract_symbol -> price mapping
        with self._lock:
            logger.info(f"🔒 Closing {len(self.open_trades)} open positions...")
            
            for trade_id, trade in list(self.open_trades.items()):
                # Try to get current price from cache, fallback to entry price (no arbitrary haircut)
                price = self._get_price_cached(trade.contract_symbol, ttl=5) or trade.entry_price
                self._close_paper_trade(trade_id, price, 'Trading Stopped', datetime.now())

        # Wait for trading thread to finish
        if self._trading_thread and self._trading_thread.is_alive():
            self._trading_thread.join(timeout=10)
            if self._trading_thread.is_alive():
                logger.warning("⚠️ Trading thread did not stop gracefully")

        logger.info("✅ Live paper trading stopped")
        
        # Stop health server
        self.stop_health_server()
        
        # Flush pending database updates
        self._flush_pending_db_updates()
    
    def _print_session_summary(self, report: Dict):
        """Print comprehensive session summary with proper equity breakdown."""
        if not report:
            logger.warning("⚠️ No session report available")
            return
        
        logger.info("=" * 80)
        logger.info("📊 LIVE PAPER TRADING SESSION SUMMARY")
        logger.info("=" * 80)
        
        # Session info
        logger.info(f"🕐 Session Duration: {report.get('session_end', datetime.now()) - report.get('session_start', datetime.now())}")
        logger.info(f"💰 Initial Capital: ₹{report.get('initial_capital', 0):,.2f}")
        logger.info(f"💵 Final Cash: ₹{report.get('final_cash', 0):,.2f}")
        logger.info(f"📈 Final Equity: ₹{report.get('final_equity', 0):,.2f}")
        logger.info(f"📊 Total Return: {report.get('return_pct', 0):+.2f}%")
        
        # P&L breakdown
        logger.info(f"✅ Realized P&L: ₹{report.get('realized_pnl', 0):+,.2f}")
        logger.info(f"📊 Unrealized P&L: ₹{report.get('unrealized_pnl', 0):+,.2f}")
        logger.info(f"💸 Total P&L: ₹{report.get('total_pnl', 0):+,.2f}")
        logger.info(f"💳 Fees Paid: ₹{report.get('fees_paid', 0):,.2f}")
        
        # Trading metrics
        logger.info(f"📈 Max Drawdown: {report.get('max_drawdown', 0):.2f}%")
        logger.info(f"🎯 Win Rate: {report.get('win_rate', 0):.1f}%")
        logger.info(f"📊 Closed Trades: {report.get('total_trades_closed', 0)}")
        logger.info(f"🔓 Open Trades: {report.get('open_trades', 0)}")
        
        # Signal metrics
        logger.info(f"📡 Signals Generated: {report.get('total_signals_generated', 0)}")
        logger.info(f"🚫 Signals Rejected: {report.get('total_signals_rejected', 0)}")
        logger.info(f"✅ Trades Executed: {report.get('total_trades_executed', 0)}")
        
        # Strategy performance
        strategy_perf = report.get('strategy_performance', {})
        if strategy_perf:
            logger.info("\n🎯 Strategy Performance:")
            for strategy, perf in strategy_perf.items():
                win_rate = (perf['wins'] / perf['trades'] * 100) if perf['trades'] > 0 else 0
                logger.info(f"   {strategy}: {perf['trades']} trades, ₹{perf['pnl']:+,.2f} P&L, {win_rate:.1f}% win rate")
        
        logger.info("=" * 80)

    def _flush_pending_db_updates(self):
        """Flush any pending database updates before shutdown."""
        try:
            # Force database connection to commit any pending transactions
            if hasattr(self.db, '_conn') and self.db._conn:
                self.db._conn.commit()
                logger.info("✅ Database updates flushed successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to flush database updates: {e}")

    def _log_trade_event(self, event_type: str, trade_data: Dict):
        """Log trade events in structured JSON format for analysis."""
        try:
            log_entry = {
                'timestamp': self.now_kolkata().isoformat(),
                'event_type': event_type,
                'trade_id': trade_data.get('id', ''),
                'strategy': trade_data.get('strategy', ''),
                'symbol': trade_data.get('contract_symbol', ''),
                'price': trade_data.get('entry_price', trade_data.get('exit_price', 0)),
                'quantity': trade_data.get('quantity', 0),
                'pnl': trade_data.get('pnl', 0),
                'cash': self.cash,
                'equity': self._equity({}),
                'exposure': self._current_total_exposure()
            }
            
            # Log as JSON for easy parsing
            logger.info(f"📊 TRADE_EVENT: {json.dumps(log_entry)}")
            
        except Exception as e:
            logger.error(f"❌ Error logging trade event: {e}")

    def _log_pnl_update(self, realized_pnl: float, unrealized_pnl: float, equity: float):
        """Log P&L updates in structured format."""
        try:
            log_entry = {
                'timestamp': self.now_kolkata().isoformat(),
                'event_type': 'pnl_update',
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': realized_pnl + unrealized_pnl,
                'equity': equity,
                'cash': self.cash,
                'max_drawdown': self.max_drawdown,
                'open_trades': len(self.open_trades),
                'closed_trades': len(self.closed_trades)
            }
            
            logger.info(f"📈 PNL_UPDATE: {json.dumps(log_entry)}")
            
        except Exception as e:
            logger.error(f"❌ Error logging P&L update: {e}")

    def _accumulate_options_data(self, symbol: str, option_chain: Dict = None):
        """Accumulate options data for historical analysis using real Fyers Option Chain API."""
        try:
            # Get raw option chain data from Fyers API
            raw_option_chain = self.data_manager.get_option_chain(symbol)
            
            if raw_option_chain:
                # Save raw data to database
                self.database.save_raw_options_chain(raw_option_chain)
                
                # Log key metrics
                options_chain = raw_option_chain.get('optionsChain', [])
                call_oi = raw_option_chain.get('callOi', 0)
                put_oi = raw_option_chain.get('putOi', 0)
                indiavix = raw_option_chain.get('indiavixData', {}).get('ltp', 0)
                
                logger.info(f"📊 Accumulated raw option chain for {symbol}")
                logger.info(f"📈 Call OI: {call_oi:,}, Put OI: {put_oi:,}")
                logger.info(f"📊 India VIX: {indiavix:.2f}")
                logger.info(f"📋 Total Options: {len(options_chain)}")
                
                # Count real strikes
                real_strikes = set()
                for option in options_chain:
                    if option.get('option_type') in ['CE', 'PE']:
                        strike = option.get('strike_price', -1)
                        if strike > 0:
                            real_strikes.add(strike)
                
                real_strikes = sorted(list(real_strikes))
                logger.info(f"🎯 Real Strikes: {len(real_strikes)} strikes")
                if real_strikes:
                    logger.info(f"📊 Strike Range: {real_strikes[0]} - {real_strikes[-1]}")
                
            else:
                logger.warning(f"⚠️ Could not fetch raw option chain for {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error accumulating options data for {symbol}: {e}")

    def accumulate_options_data_continuously(self, symbols: List[str] = None, interval_seconds: int = 60):
        """Continuously accumulate options data for all major indexes."""
        if symbols is None:
            symbols = ['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX', 'NSE:FINNIFTY-INDEX']
        
        logger.info(f"🚀 Starting continuous options data accumulation for {len(symbols)} symbols")
        logger.info(f"📊 Symbols: {', '.join(symbols)}")
        logger.info(f"⏰ Interval: {interval_seconds} seconds")
        
        try:
            while not self._stop_event.is_set():
                start_time = time.time()
                
                for symbol in symbols:
                    try:
                        # Get option chain data
                        option_chain = self.data_provider.get_option_chain(symbol)
                        
                        if option_chain:
                            # Accumulate the data
                            self._accumulate_options_data(symbol, option_chain)
                            logger.debug(f"✅ Accumulated options data for {symbol}")
                        else:
                            logger.warning(f"⚠️ Could not fetch options data for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"❌ Error accumulating options data for {symbol}: {e}")
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0, interval_seconds - elapsed)
                
                if sleep_time > 0:
                    logger.debug(f"💤 Sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            logger.info("🛑 Options data accumulation stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in options data accumulation: {e}")
        finally:
            logger.info("✅ Options data accumulation completed")

    def get_accumulated_options_summary(self) -> Dict:
        """Get summary of accumulated options data."""
        try:
            return self.db.get_options_data_summary()
        except Exception as e:
            logger.error(f"❌ Error getting options data summary: {e}")
            return {}

    def _get_available_strategies(self):
        """Get list of available strategies."""
        return ['ema_crossover_enhanced', 'supertrend_ema', 'supertrend_macd_rsi_ema']

    def safe_pct(self, value: float, denominator: float) -> float:
        """Safely calculate percentage with division by zero protection."""
        if denominator == 0:
            return 0.0
        return (value / denominator) * 100

    def now_kolkata(self) -> datetime:
        """Get current time in Kolkata timezone."""
        return datetime.now(tz=self.tz)

    def _make_open_key(self, strategy: str, contract_sym: str, option_type: str) -> Tuple[str, str, str]:
        """Create canonical open key for position tracking."""
        return (strategy, contract_sym, option_type)


def main():
    parser = argparse.ArgumentParser(description='Live Paper Trading System')
    parser.add_argument('--symbols', nargs='+', default=['NSE:NIFTY50-INDEX'], help='Trading symbols')
    parser.add_argument('--capital', type=float, default=20000.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Max risk per trade')
    parser.add_argument('--confidence_cutoff', type=float, default=40.0,
                       help='Minimum confidence score to execute trades (default: 40.0)')
    parser.add_argument('--exposure_limit', type=float, default=0.6,
                       help='Maximum portfolio exposure (default: 0.6)')
    parser.add_argument('--max_daily_loss_pct', type=float, default=0.03,
                       help='Maximum daily loss percentage (default: 0.03)')
    parser.add_argument('--commission_bps', type=float, default=1.0,
                       help='Commission in basis points (default: 1.0)')
    parser.add_argument('--slippage_bps', type=float, default=5.0,
                       help='Slippage in basis points (default: 5.0)')
    parser.add_argument('--stop_loss_pct', type=float, default=-30.0,
                       help='Stop loss percentage (default: -30.0)')
    parser.add_argument('--take_profit_pct', type=float, default=25.0,
                       help='Take profit percentage (default: 25.0)')
    parser.add_argument('--time_stop_minutes', type=int, default=30,
                       help='Time-based exit in minutes (default: 30)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--data_provider', choices=['paper', 'fyers'], default='fyers', help='Data provider')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode for a few minutes')
    parser.add_argument('--accumulate_options', action='store_true', help='Run in options data accumulation mode only')
    parser.add_argument('--options_interval', type=int, default=60, help='Options data accumulation interval in seconds')
    parser.add_argument('--options_symbols', nargs='+', default=['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX', 'NSE:FINNIFTY-INDEX'], help='Symbols for options data accumulation')

    args = parser.parse_args()

    # Initialize trading system
    trading_system = LivePaperTradingSystem(
        symbols=args.symbols,
        initial_capital=args.capital,
        max_risk_per_trade=args.risk,
        confidence_cutoff=args.confidence_cutoff,
        exposure_limit=args.exposure_limit,
        max_daily_loss_pct=args.max_daily_loss_pct,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        time_stop_minutes=args.time_stop_minutes,
        verbose=args.verbose,
        data_provider=args.data_provider
    )

    try:
        if args.accumulate_options:
            # Run in options data accumulation mode only
            logger.info("🚀 Starting options data accumulation mode")
            trading_system.accumulate_options_data_continuously(
                symbols=args.options_symbols,
                interval_seconds=args.options_interval
            )
        else:
            # Run normal trading mode
            if args.test_mode:
                logger.info("🧪 Running in test mode for a few minutes")
                trading_system.start_trading()
                time.sleep(300)  # Run for 5 minutes
                trading_system.stop_trading()
            else:
                logger.info("🚀 Starting live paper trading")
                trading_system.start_trading()
                
                # Keep running until interrupted
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("🛑 Stopping trading system...")
                    trading_system.stop_trading()
                    
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
        trading_system.stop_trading()
        raise


if __name__ == "__main__":
    main() 