#!/usr/bin/env python3
"""
Risk Management System
Comprehensive risk controls for live options trading
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.option_contract import OptionContract, OptionType
from src.execution.broker_execution import OrderSide, OrderType

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskConfig:
    """Risk configuration parameters."""
    # Capital limits
    initial_capital: float = 100000.0
    max_capital_utilization: float = 0.8  # 80%
    max_daily_loss_pct: float = 0.03  # 3%
    max_weekly_loss_pct: float = 0.10  # 10%
    max_monthly_loss_pct: float = 0.20  # 20%
    
    # Position limits
    max_positions_per_symbol: int = 3
    max_total_positions: int = 10
    max_lots_per_trade: int = 2
    max_lots_per_symbol: int = 5
    
    # Risk per trade
    max_risk_per_trade_pct: float = 0.02  # 2%
    max_risk_per_symbol_pct: float = 0.05  # 5%
    
    # Margin limits
    max_margin_utilization: float = 0.7  # 70%
    min_margin_buffer: float = 50000.0  # ₹50k buffer
    
    # Time limits
    max_holding_days: int = 5
    trading_start_time: str = "09:15"
    trading_end_time: str = "15:30"
    
    # Volatility limits
    max_iv_percentile: float = 90.0  # Don't trade when IV > 90th percentile
    min_iv_percentile: float = 10.0  # Don't trade when IV < 10th percentile
    
    # Liquidity limits
    min_option_volume: int = 100
    min_option_oi: int = 1000
    max_bid_ask_spread_pct: float = 0.10  # 10%


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    current_capital: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    total_positions: int
    margin_utilization: float
    max_drawdown: float
    current_drawdown: float
    risk_level: RiskLevel
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, config: RiskConfig):
        """Initialize risk manager."""
        self.config = config
        self.current_capital = config.initial_capital
        self.peak_capital = config.initial_capital
        
        # Tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.daily_loss_limit_hit = False
        self.weekly_loss_limit_hit = False
        self.monthly_loss_limit_hit = False
        
        # Position tracking
        self.positions = {}  # symbol -> list of positions
        self.position_history = []
        
        # Risk alerts
        self.risk_alerts = []
        self.risk_callbacks = []
        
        # Threading
        self.lock = threading.Lock()
        self.running = False
        self.monitor_thread = None
        
        logger.info("🛡️ Risk Manager initialized")
        logger.info(f"💰 Initial Capital: ₹{config.initial_capital:,.2f}")
        logger.info(f"📊 Max Daily Loss: {config.max_daily_loss_pct*100:.1f}%")
        logger.info(f"📈 Max Positions: {config.max_total_positions}")
    
    def start_monitoring(self):
        """Start risk monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🔄 Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("🛑 Risk monitoring stopped")
    
    def add_risk_callback(self, callback: Callable[[RiskMetrics], None]):
        """Add risk alert callback."""
        self.risk_callbacks.append(callback)
    
    def check_can_place_order(self, contract: OptionContract, quantity: int, 
                            side: OrderSide, price: Optional[float] = None) -> Tuple[bool, str]:
        """Check if order can be placed based on risk rules."""
        with self.lock:
            # 1. Check if trading is allowed
            if not self._is_trading_allowed():
                return False, "Trading not allowed at this time"
            
            # 2. Check daily loss limit
            if self.daily_loss_limit_hit:
                return False, "Daily loss limit hit"
            
            # 3. Check weekly loss limit
            if self.weekly_loss_limit_hit:
                return False, "Weekly loss limit hit"
            
            # 4. Check monthly loss limit
            if self.monthly_loss_limit_hit:
                return False, "Monthly loss limit hit"
            
            # 5. Check position limits
            if not self._check_position_limits(contract, quantity):
                return False, "Position limits exceeded"
            
            # 6. Check risk per trade
            if not self._check_risk_per_trade(contract, quantity, price):
                return False, "Risk per trade exceeded"
            
            # 7. Check margin requirements
            if not self._check_margin_requirements(contract, quantity, side, price):
                return False, "Insufficient margin"
            
            # 8. Check liquidity
            if not self._check_liquidity(contract):
                return False, "Insufficient liquidity"
            
            return True, "Order allowed"
    
    def record_trade(self, contract: OptionContract, quantity: int, side: OrderSide,
                    price: float, timestamp: datetime = None):
        """Record a completed trade."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            # Calculate P&L impact
            if side == OrderSide.BUY:
                pnl_impact = -price * quantity  # Cost
            else:
                pnl_impact = price * quantity  # Revenue
            
            # Update P&L
            self.daily_pnl += pnl_impact
            self.weekly_pnl += pnl_impact
            self.monthly_pnl += pnl_impact
            
            # Update capital
            self.current_capital += pnl_impact
            
            # Update peak capital
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
            
            # Record position
            position = {
                'contract': contract,
                'quantity': quantity,
                'side': side,
                'price': price,
                'timestamp': timestamp,
                'pnl_impact': pnl_impact
            }
            
            if contract.symbol not in self.positions:
                self.positions[contract.symbol] = []
            
            self.positions[contract.symbol].append(position)
            self.position_history.append(position)
            
            # Check loss limits
            self._check_loss_limits()
            
            logger.info(f"📊 Trade recorded: {side.value} {quantity} {contract.symbol} @ ₹{price:.2f}")
            logger.info(f"💰 Daily P&L: ₹{self.daily_pnl:+,.2f}, Capital: ₹{self.current_capital:,.2f}")
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        with self.lock:
            # Calculate drawdown
            if self.peak_capital > 0:
                current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            else:
                current_drawdown = 0.0
            
            # Determine risk level
            risk_level = self._calculate_risk_level()
            
            return RiskMetrics(
                current_capital=self.current_capital,
                daily_pnl=self.daily_pnl,
                weekly_pnl=self.weekly_pnl,
                monthly_pnl=self.monthly_pnl,
                total_positions=len(self.positions),
                margin_utilization=self._calculate_margin_utilization(),
                max_drawdown=self._calculate_max_drawdown(),
                current_drawdown=current_drawdown,
                risk_level=risk_level
            )
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of trading day)."""
        with self.lock:
            self.daily_pnl = 0.0
            self.daily_loss_limit_hit = False
            logger.info("🔄 Daily risk metrics reset")
    
    def reset_weekly_metrics(self):
        """Reset weekly metrics (call at start of week)."""
        with self.lock:
            self.weekly_pnl = 0.0
            self.weekly_loss_limit_hit = False
            logger.info("🔄 Weekly risk metrics reset")
    
    def reset_monthly_metrics(self):
        """Reset monthly metrics (call at start of month)."""
        with self.lock:
            self.monthly_pnl = 0.0
            self.monthly_loss_limit_hit = False
            logger.info("🔄 Monthly risk metrics reset")
    
    def _is_trading_allowed(self) -> bool:
        """Check if trading is allowed at current time."""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # Check if within trading hours
        if not (self.config.trading_start_time <= current_time <= self.config.trading_end_time):
            return False
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        return True
    
    def _check_position_limits(self, contract: OptionContract, quantity: int) -> bool:
        """Check position limits."""
        # Check total positions
        if len(self.positions) >= self.config.max_total_positions:
            return False
        
        # Check positions per symbol
        symbol_positions = len(self.positions.get(contract.symbol, []))
        if symbol_positions >= self.config.max_positions_per_symbol:
            return False
        
        # Check lots per trade
        lots = quantity / contract.lot_size
        if lots > self.config.max_lots_per_trade:
            return False
        
        # Check lots per symbol
        total_lots_symbol = sum(
            pos['quantity'] / pos['contract'].lot_size 
            for pos in self.positions.get(contract.symbol, [])
        )
        if total_lots_symbol + lots > self.config.max_lots_per_symbol:
            return False
        
        return True
    
    def _check_risk_per_trade(self, contract: OptionContract, quantity: int, price: Optional[float]) -> bool:
        """Check risk per trade limits."""
        if price is None:
            price = contract.ask if contract.ask > 0 else contract.last
        
        # Calculate trade value
        trade_value = price * quantity
        
        # Check risk per trade
        max_risk_amount = self.current_capital * self.config.max_risk_per_trade_pct
        if trade_value > max_risk_amount:
            return False
        
        # Check risk per symbol
        symbol_positions = self.positions.get(contract.symbol, [])
        symbol_risk = sum(pos['price'] * pos['quantity'] for pos in symbol_positions)
        max_symbol_risk = self.current_capital * self.config.max_risk_per_symbol_pct
        
        if symbol_risk + trade_value > max_symbol_risk:
            return False
        
        return True
    
    def _check_margin_requirements(self, contract: OptionContract, quantity: int, 
                                 side: OrderSide, price: Optional[float]) -> bool:
        """Check margin requirements."""
        if price is None:
            price = contract.ask if contract.ask > 0 else contract.last
        
        # Calculate required margin
        if side == OrderSide.BUY:
            required_margin = price * quantity
        else:
            # For selling options, use strike-based margin
            required_margin = contract.strike * quantity * 0.15  # 15% margin
        
        # Check available margin
        available_margin = self.current_capital * self.config.max_margin_utilization
        if required_margin > available_margin:
            return False
        
        # Check margin buffer
        if self.current_capital - required_margin < self.config.min_margin_buffer:
            return False
        
        return True
    
    def _check_liquidity(self, contract: OptionContract) -> bool:
        """Check liquidity requirements."""
        # Check volume
        if contract.volume < self.config.min_option_volume:
            return False
        
        # Check open interest
        if contract.open_interest < self.config.min_option_oi:
            return False
        
        # Check bid-ask spread
        if contract.ask > 0 and contract.bid > 0:
            spread_pct = (contract.ask - contract.bid) / contract.bid
            if spread_pct > self.config.max_bid_ask_spread_pct:
                return False
        
        return True
    
    def _check_loss_limits(self):
        """Check loss limits and trigger alerts."""
        # Daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.config.initial_capital
        if daily_loss_pct >= self.config.max_daily_loss_pct and not self.daily_loss_limit_hit:
            self.daily_loss_limit_hit = True
            self._trigger_risk_alert("Daily loss limit hit", RiskLevel.CRITICAL)
        
        # Weekly loss limit
        weekly_loss_pct = abs(self.weekly_pnl) / self.config.initial_capital
        if weekly_loss_pct >= self.config.max_weekly_loss_pct and not self.weekly_loss_limit_hit:
            self.weekly_loss_limit_hit = True
            self._trigger_risk_alert("Weekly loss limit hit", RiskLevel.HIGH)
        
        # Monthly loss limit
        monthly_loss_pct = abs(self.monthly_pnl) / self.config.initial_capital
        if monthly_loss_pct >= self.config.max_monthly_loss_pct and not self.monthly_loss_limit_hit:
            self.monthly_loss_limit_hit = True
            self._trigger_risk_alert("Monthly loss limit hit", RiskLevel.HIGH)
    
    def _calculate_risk_level(self) -> RiskLevel:
        """Calculate current risk level."""
        # Calculate various risk metrics
        daily_loss_pct = abs(self.daily_pnl) / self.config.initial_capital
        weekly_loss_pct = abs(self.weekly_pnl) / self.config.initial_capital
        monthly_loss_pct = abs(self.monthly_pnl) / self.config.initial_capital
        
        # Determine risk level based on multiple factors
        if (daily_loss_pct >= self.config.max_daily_loss_pct * 0.8 or
            weekly_loss_pct >= self.config.max_weekly_loss_pct * 0.8 or
            monthly_loss_pct >= self.config.max_monthly_loss_pct * 0.8):
            return RiskLevel.CRITICAL
        elif (daily_loss_pct >= self.config.max_daily_loss_pct * 0.6 or
              weekly_loss_pct >= self.config.max_weekly_loss_pct * 0.6):
            return RiskLevel.HIGH
        elif (daily_loss_pct >= self.config.max_daily_loss_pct * 0.4 or
              weekly_loss_pct >= self.config.max_weekly_loss_pct * 0.4):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_margin_utilization(self) -> float:
        """Calculate current margin utilization."""
        # This would need to be implemented based on actual margin data from broker
        # For now, return a simplified calculation
        total_position_value = sum(
            pos['price'] * pos['quantity'] 
            for positions in self.positions.values() 
            for pos in positions
        )
        
        if self.current_capital > 0:
            return total_position_value / self.current_capital
        return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if self.peak_capital > 0:
            return (self.peak_capital - self.current_capital) / self.peak_capital
        return 0.0
    
    def _trigger_risk_alert(self, message: str, level: RiskLevel):
        """Trigger a risk alert."""
        alert = {
            'message': message,
            'level': level,
            'timestamp': datetime.now(),
            'metrics': self.get_risk_metrics()
        }
        
        self.risk_alerts.append(alert)
        
        # Call risk callbacks
        for callback in self.risk_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"❌ Error in risk callback: {e}")
        
        logger.warning(f"🚨 Risk Alert ({level.value}): {message}")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check risk metrics periodically
                metrics = self.get_risk_metrics()
                
                # Trigger alerts based on risk level
                if metrics.risk_level == RiskLevel.CRITICAL:
                    self._trigger_risk_alert("Critical risk level reached", RiskLevel.CRITICAL)
                elif metrics.risk_level == RiskLevel.HIGH:
                    self._trigger_risk_alert("High risk level reached", RiskLevel.HIGH)
                
                # Check for position holding time limits
                self._check_holding_time_limits()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in risk monitoring loop: {e}")
                time.sleep(60)
    
    def _check_holding_time_limits(self):
        """Check if positions have exceeded holding time limits."""
        now = datetime.now()
        
        for symbol, positions in self.positions.items():
            for position in positions:
                holding_days = (now - position['timestamp']).days
                
                if holding_days >= self.config.max_holding_days:
                    self._trigger_risk_alert(
                        f"Position {symbol} exceeded max holding time ({holding_days} days)",
                        RiskLevel.MEDIUM
                    )
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report."""
        metrics = self.get_risk_metrics()
        
        return {
            'timestamp': datetime.now(),
            'risk_metrics': metrics,
            'config': self.config,
            'positions': {
                symbol: len(positions) for symbol, positions in self.positions.items()
            },
            'alerts': self.risk_alerts[-10:],  # Last 10 alerts
            'daily_loss_limit_hit': self.daily_loss_limit_hit,
            'weekly_loss_limit_hit': self.weekly_loss_limit_hit,
            'monthly_loss_limit_hit': self.monthly_loss_limit_hit
        }


# Example usage
if __name__ == "__main__":
    # Create risk configuration
    config = RiskConfig(
        initial_capital=100000.0,
        max_daily_loss_pct=0.03,
        max_positions_per_symbol=3,
        max_total_positions=10
    )
    
    # Create risk manager
    risk_manager = RiskManager(config)
    
    # Add risk callback
    def risk_alert_callback(alert):
    
    risk_manager.add_risk_callback(risk_alert_callback)
    
    # Start monitoring
    risk_manager.start_monitoring()
    
    # Simulate some trades
    from src.models.option_contract import OptionContract, OptionType
    from datetime import datetime, timedelta
    
    contract = OptionContract(
        symbol="NIFTY25AUG25000CE",
        underlying="NSE:NIFTY50-INDEX",
        strike=25000,
        expiry=datetime.now() + timedelta(days=7),
        option_type=OptionType.CALL,
        lot_size=50,
        bid=100,
        ask=110,
        last=105,
        volume=1000,
        open_interest=5000
    )
    
    # Check if we can place an order
    can_place, reason = risk_manager.check_can_place_order(
        contract, 50, OrderSide.BUY, 110
    )
    
    if not can_place:
    
    # Record a trade
    risk_manager.record_trade(contract, 50, OrderSide.BUY, 110)
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics()
    
    # Get risk report
    report = risk_manager.get_risk_report()
    
    # Stop monitoring
    risk_manager.stop_monitoring() 