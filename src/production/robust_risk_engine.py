#!/usr/bin/env python3
"""
Robust Risk Engine (Portfolio-level)
MUST #4: Portfolio-level constraints and circuit breakers
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from typing import Any
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class CircuitBreakerStatus(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    TRIGGERED = "TRIGGERED"
    HALTED = "HALTED"

@dataclass
class PortfolioConstraints:
    """Portfolio-level risk constraints"""
    max_portfolio_exposure: float = 0.60  # 60%
    max_daily_drawdown: float = 0.03  # 3%
    max_single_position: float = 0.10  # 10%
    max_sector_exposure: float = 0.30  # 30%
    max_correlation_exposure: float = 0.50  # 50%
    max_consecutive_losses: int = 5
    max_api_failure_rate: float = 0.10  # 10% in 5 minutes
    max_slippage_spike: float = 0.02  # 2% sudden spike
    max_latency_spike: float = 5.0  # 5 seconds

@dataclass
class StrategyAllocation:
    """Strategy allocation constraints"""
    strategy_name: str
    max_allocation: float  # Maximum % of portfolio
    current_allocation: float
    min_allocation: float  # Minimum % of portfolio
    performance_threshold: float  # Sharpe ratio threshold
    max_drawdown_threshold: float

@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""
    status: CircuitBreakerStatus
    trigger_reason: Optional[str]
    trigger_timestamp: Optional[datetime]
    consecutive_losses: int
    api_failure_count: int
    api_failure_window_start: datetime
    last_slippage_check: datetime
    last_latency_check: datetime

class RobustRiskEngine:
    """Robust portfolio-level risk engine with circuit breakers"""
    
    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints
        self.portfolio_value = 0.0
        self.positions = {}
        self.daily_pnl = 0.0
        self.peak_portfolio_value = 0.0
        self.consecutive_losses = 0
        self.trade_history = []
        self.strategy_allocations = {}
        self.circuit_breaker = CircuitBreakerState(
            status=CircuitBreakerStatus.NORMAL,
            trigger_reason=None,
            trigger_timestamp=None,
            consecutive_losses=0,
            api_failure_count=0,
            api_failure_window_start=datetime.now(),
            last_slippage_check=datetime.now(),
            last_latency_check=datetime.now()
        )
        
        # Performance tracking
        self.api_failures = []
        self.slippage_history = []
        self.latency_history = []
        
    def check_portfolio_risk(self, new_signal: Dict[str, Any], current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """Check portfolio-level risk constraints"""
        try:
            # Update portfolio value
            self._update_portfolio_value(current_prices)
            
            # Check circuit breaker status
            if self.circuit_breaker.status == CircuitBreakerStatus.TRIGGERED:
                return False, f"Circuit breaker triggered: {self.circuit_breaker.trigger_reason}"
            
            # Check daily drawdown limit
            if self.daily_pnl < -self.constraints.max_daily_drawdown * self.peak_portfolio_value:
                self._trigger_circuit_breaker("Daily drawdown limit exceeded")
                return False, "Daily drawdown limit exceeded"
            
            # Check portfolio exposure
            total_exposure = self._calculate_total_exposure()
            if total_exposure > self.constraints.max_portfolio_exposure:
                return False, f"Portfolio exposure too high: {total_exposure:.1%}"
            
            # Check single position limit
            signal_symbol = new_signal.get('symbol')
            signal_quantity = new_signal.get('position_size', 0)
            signal_price = current_prices.get(signal_symbol, 0)
            signal_value = signal_quantity * signal_price
            
            if signal_value > self.constraints.max_single_position * self.portfolio_value:
                return False, f"Single position too large: {signal_value:,.2f}"
            
            # Check sector exposure
            sector_exposure = self._calculate_sector_exposure()
            for sector, exposure in sector_exposure.items():
                if exposure > self.constraints.max_sector_exposure:
                    return False, f"Sector exposure too high: {sector} {exposure:.1%}"
            
            # Check correlation exposure
            correlation_exposure = self._calculate_correlation_exposure()
            if correlation_exposure > self.constraints.max_correlation_exposure:
                return False, f"Correlation exposure too high: {correlation_exposure:.1%}"
            
            # Check consecutive losses
            if self.consecutive_losses >= self.constraints.max_consecutive_losses:
                self._trigger_circuit_breaker("Too many consecutive losses")
                return False, "Too many consecutive losses"
            
            # Check strategy allocation
            strategy_name = new_signal.get('strategy', 'unknown')
            if not self._check_strategy_allocation(strategy_name, signal_value):
                return False, f"Strategy allocation limit exceeded: {strategy_name}"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            logger.error(f"❌ Portfolio risk check failed: {e}")
            return False, f"Risk check error: {e}"
    
    def _update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update portfolio value and positions"""
        try:
            # Update position prices
            for symbol, position in self.positions.items():
                if symbol in current_prices:
                    position['current_price'] = current_prices[symbol]
                    position['market_value'] = position['quantity'] * current_prices[symbol]
                    position['unrealized_pnl'] = (current_prices[symbol] - position['entry_price']) * position['quantity']
            
            # Calculate total portfolio value
            total_market_value = sum(pos['market_value'] for pos in self.positions.values())
            self.portfolio_value = total_market_value
            
            # Update peak portfolio value
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
                
        except Exception as e:
            logger.error(f"❌ Portfolio value update failed: {e}")
    
    def _calculate_total_exposure(self) -> float:
        """Calculate total portfolio exposure"""
        if self.portfolio_value == 0:
            return 0.0
        
        total_exposure = sum(pos['market_value'] for pos in self.positions.values())
        return total_exposure / self.portfolio_value
    
    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure"""
        sector_exposure = {}
        
        for symbol, position in self.positions.items():
            # Determine sector from symbol
            if 'NIFTY' in symbol:
                sector = 'INDEX'
            elif 'BANK' in symbol:
                sector = 'BANKING'
            elif 'FIN' in symbol:
                sector = 'FINANCIAL'
            else:
                sector = 'OTHER'
            
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position['market_value']
        
        # Convert to percentages
        if self.portfolio_value > 0:
            for sector in sector_exposure:
                sector_exposure[sector] /= self.portfolio_value
        
        return sector_exposure
    
    def _calculate_correlation_exposure(self) -> float:
        """Calculate correlation exposure (simplified)"""
        if len(self.positions) < 2:
            return 0.0
        
        # Simplified correlation calculation
        # In real implementation, calculate actual correlations
        symbols = list(self.positions.keys())
        
        # Check for highly correlated symbols
        correlated_pairs = 0
        total_pairs = 0
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                total_pairs += 1
                # Simplified correlation check
                if self._are_symbols_correlated(symbol1, symbol2):
                    correlated_pairs += 1
        
        return correlated_pairs / total_pairs if total_pairs > 0 else 0.0
    
    def _are_symbols_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are highly correlated"""
        # Simplified correlation check
        # In real implementation, use actual correlation data
        
        # Same sector = high correlation
        if ('NIFTY' in symbol1 and 'NIFTY' in symbol2) or \
           ('BANK' in symbol1 and 'BANK' in symbol2):
            return True
        
        return False
    
    def _check_strategy_allocation(self, strategy_name: str, signal_value: float) -> bool:
        """Check strategy allocation limits"""
        if strategy_name not in self.strategy_allocations:
            # Initialize strategy allocation
            self.strategy_allocations[strategy_name] = StrategyAllocation(
                strategy_name=strategy_name,
                max_allocation=0.20,  # 20% max allocation
                current_allocation=0.0,
                min_allocation=0.05,  # 5% min allocation
                performance_threshold=1.0,  # Sharpe ratio threshold
                max_drawdown_threshold=0.10  # 10% max drawdown
            )
        
        strategy = self.strategy_allocations[strategy_name]
        
        # Calculate new allocation
        new_allocation = (strategy.current_allocation * self.portfolio_value + signal_value) / self.portfolio_value
        
        return new_allocation <= strategy.max_allocation
    
    def add_position(self, symbol: str, quantity: float, price: float, strategy: str):
        """Add position to portfolio"""
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': price,
            'current_price': price,
            'market_value': quantity * price,
            'unrealized_pnl': 0.0,
            'strategy': strategy,
            'timestamp': datetime.now()
        }
        
        # Update strategy allocation
        if strategy in self.strategy_allocations:
            self.strategy_allocations[strategy].current_allocation += quantity * price
    
    def add_trade_result(self, symbol: str, pnl: float, strategy: str):
        """Add trade result and update risk metrics"""
        trade = {
            'symbol': symbol,
            'pnl': pnl,
            'strategy': strategy,
            'timestamp': datetime.now(),
            'is_win': pnl > 0
        }
        
        self.trade_history.append(trade)
        self.daily_pnl += pnl
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update strategy performance
        self._update_strategy_performance(strategy, pnl)
        
        logger.info(f"📈 Trade result: {symbol} P&L: ₹{pnl:,.2f}, Consecutive losses: {self.consecutive_losses}")
    
    def _update_strategy_performance(self, strategy_name: str, pnl: float):
        """Update strategy performance metrics"""
        if strategy_name not in self.strategy_allocations:
            return
        
        strategy = self.strategy_allocations[strategy_name]
        
        # Get recent trades for this strategy
        recent_trades = [t for t in self.trade_history[-20:] if t['strategy'] == strategy_name]
        
        if len(recent_trades) >= 10:
            # Calculate Sharpe ratio
            returns = [t['pnl'] for t in recent_trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
            
            # Update allocation based on performance
            if sharpe_ratio > strategy.performance_threshold and max_drawdown < strategy.max_drawdown_threshold:
                # Increase allocation
                strategy.max_allocation = min(0.30, strategy.max_allocation * 1.1)
            elif sharpe_ratio < 0.5 or max_drawdown > strategy.max_drawdown_threshold:
                # Decrease allocation
                strategy.max_allocation = max(0.05, strategy.max_allocation * 0.9)
    
    def record_api_failure(self):
        """Record API failure for circuit breaker"""
        current_time = datetime.now()
        self.api_failures.append(current_time)
        
        # Reset window if needed
        if current_time - self.circuit_breaker.api_failure_window_start > timedelta(minutes=5):
            self.circuit_breaker.api_failure_window_start = current_time
            self.circuit_breaker.api_failure_count = 0
        
        self.circuit_breaker.api_failure_count += 1
        
        # Check API failure rate
        recent_failures = [f for f in self.api_failures 
                          if current_time - f <= timedelta(minutes=5)]
        
        if len(recent_failures) > 10:  # More than 10 failures in 5 minutes
            failure_rate = len(recent_failures) / 300  # 5 minutes = 300 seconds
            if failure_rate > self.constraints.max_api_failure_rate:
                self._trigger_circuit_breaker(f"API failure rate too high: {failure_rate:.1%}")
    
    def record_slippage(self, expected_price: float, actual_price: float):
        """Record slippage for circuit breaker"""
        slippage = abs(actual_price - expected_price) / expected_price
        self.slippage_history.append({
            'timestamp': datetime.now(),
            'slippage': slippage
        })
        
        # Check for slippage spike
        if slippage > self.constraints.max_slippage_spike:
            self._trigger_circuit_breaker(f"Slippage spike detected: {slippage:.1%}")
    
    def record_latency(self, latency: float):
        """Record latency for circuit breaker"""
        self.latency_history.append({
            'timestamp': datetime.now(),
            'latency': latency
        })
        
        # Check for latency spike
        if latency > self.constraints.max_latency_spike:
            self._trigger_circuit_breaker(f"Latency spike detected: {latency:.1f}s")
    
    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker"""
        self.circuit_breaker.status = CircuitBreakerStatus.TRIGGERED
        self.circuit_breaker.trigger_reason = reason
        self.circuit_breaker.trigger_timestamp = datetime.now()
        
        logger.error(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
        
        # Send alert
        self._send_circuit_breaker_alert(reason)
    
    def _send_circuit_breaker_alert(self, reason: str):
        """Send circuit breaker alert"""
        # In real implementation, send to alerting system
        logger.critical(f"🚨 CRITICAL ALERT: Circuit breaker triggered - {reason}")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention required)"""
        self.circuit_breaker.status = CircuitBreakerStatus.NORMAL
        self.circuit_breaker.trigger_reason = None
        self.circuit_breaker.trigger_timestamp = None
        self.consecutive_losses = 0
        
        logger.info("✅ Circuit breaker reset")
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.api_failures = []
        self.slippage_history = []
        self.latency_history = []
        
        logger.info("🔄 Daily risk metrics reset")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report"""
        try:
            # Calculate risk metrics
            total_exposure = self._calculate_total_exposure()
            sector_exposure = self._calculate_sector_exposure()
            correlation_exposure = self._calculate_correlation_exposure()
            
            # Calculate recent performance
            recent_trades = self.trade_history[-20:] if len(self.trade_history) >= 20 else self.trade_history
            win_rate = sum(1 for t in recent_trades if t['is_win']) / len(recent_trades) if recent_trades else 0
            
            # Calculate strategy performance
            strategy_performance = {}
            for strategy_name, allocation in self.strategy_allocations.items():
                strategy_trades = [t for t in recent_trades if t['strategy'] == strategy_name]
                if strategy_trades:
                    strategy_pnl = sum(t['pnl'] for t in strategy_trades)
                    strategy_win_rate = sum(1 for t in strategy_trades if t['is_win']) / len(strategy_trades)
                    strategy_performance[strategy_name] = {
                        'pnl': strategy_pnl,
                        'win_rate': strategy_win_rate,
                        'allocation': allocation.current_allocation,
                        'max_allocation': allocation.max_allocation
                    }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'daily_pnl': self.daily_pnl,
                'total_exposure': total_exposure,
                'sector_exposure': sector_exposure,
                'correlation_exposure': correlation_exposure,
                'consecutive_losses': self.consecutive_losses,
                'win_rate': win_rate,
                'circuit_breaker': {
                    'status': self.circuit_breaker.status.value,
                    'trigger_reason': self.circuit_breaker.trigger_reason,
                    'trigger_timestamp': self.circuit_breaker.trigger_timestamp.isoformat() if self.circuit_breaker.trigger_timestamp else None
                },
                'strategy_performance': strategy_performance,
                'risk_constraints': {
                    'max_portfolio_exposure': self.constraints.max_portfolio_exposure,
                    'max_daily_drawdown': self.constraints.max_daily_drawdown,
                    'max_single_position': self.constraints.max_single_position,
                    'max_sector_exposure': self.constraints.max_sector_exposure,
                    'max_correlation_exposure': self.constraints.max_correlation_exposure,
                    'max_consecutive_losses': self.constraints.max_consecutive_losses
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Risk report generation failed: {e}")
            return {}

def main():
    """Main function for testing"""
    constraints = PortfolioConstraints(
        max_portfolio_exposure=0.60,
        max_daily_drawdown=0.03,
        max_single_position=0.10,
        max_sector_exposure=0.30,
        max_correlation_exposure=0.50,
        max_consecutive_losses=5
    )
    
    risk_engine = RobustRiskEngine(constraints)
    
    # Add some positions
    risk_engine.add_position("NSE:NIFTY50-INDEX", 100, 19500, "ema_strategy")
    risk_engine.add_position("NSE:NIFTYBANK-INDEX", 50, 45000, "supertrend_strategy")
    
    # Update prices
    current_prices = {
        "NSE:NIFTY50-INDEX": 19600,
        "NSE:NIFTYBANK-INDEX": 45100
    }
    
    # Test risk check
    test_signal = {
        'symbol': 'NSE:FINNIFTY-INDEX',
        'position_size': 100,
        'strategy': 'ema_strategy'
    }
    
    can_trade, reason = risk_engine.check_portfolio_risk(test_signal, current_prices)
    print(f"Can trade: {can_trade}, Reason: {reason}")
    
    # Test circuit breaker
    for i in range(6):
        risk_engine.add_trade_result("NSE:NIFTY50-INDEX", -1000, "ema_strategy")
    
    can_trade, reason = risk_engine.check_portfolio_risk(test_signal, current_prices)
    print(f"After losses - Can trade: {can_trade}, Reason: {reason}")
    
    # Get risk report
    report = risk_engine.get_risk_report()
    print(f"Risk report: {json.dumps(report, indent=2, default=str)}")

if __name__ == "__main__":
    main()
