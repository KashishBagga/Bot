#!/usr/bin/env python3
"""
Advanced Risk Management System
Portfolio-level risk controls, correlation analysis, and circuit breakers
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from src.core.timezone_utils import timezone_manager, now, now_kolkata
from dataclasses import dataclass
from enum import Enum
import json
import threading
from zoneinfo import ZoneInfo
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_daily_loss: float = 0.05  # 5%
    max_portfolio_exposure: float = 0.8  # 80%
    max_single_position: float = 0.2  # 20%
    max_correlation_exposure: float = 0.6  # 60%
    max_sector_exposure: float = 0.4  # 40%
    max_volatility_exposure: float = 0.3  # 30%
    min_win_rate: float = 0.4  # 40%
    max_consecutive_losses: int = 5
    circuit_breaker_threshold: float = 0.1  # 10%

@dataclass
class PositionRisk:
    """Position risk metrics"""
    symbol: str
    quantity: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    risk_exposure: float
    volatility: float
    beta: float
    var_95: float  # Value at Risk 95%

@dataclass
class PortfolioRisk:
    """Portfolio risk metrics"""
    total_value: float
    total_exposure: float
    diversification_ratio: float
    correlation_risk: float
    concentration_risk: float
    sector_concentration: Dict[str, float]
    portfolio_var_95: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    risk_level: RiskLevel

class AdvancedRiskManager:
    """Advanced risk management system with portfolio-level controls"""
    
    def __init__(self, risk_limits: RiskLimits = None):
        # Thread safety
        self._lock = threading.RLock()
        self.tz = ZoneInfo('Asia/Kolkata')
        self.risk_limits = risk_limits or RiskLimits()
        self.positions = {}
        self.trade_history = []
        self.daily_pnl = 0.0
        self.peak_portfolio_value = 0.0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.risk_alerts = []
        self.alert_timestamps = {}  # For rate limiting
        self.alert_cooldown = 300  # 5 minutes
        
        # Risk monitoring
        self.last_risk_check = now()
        self.risk_check_interval = int(os.getenv('RISK_CHECK_INTERVAL', '60'))  # 1 minute
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', '0.05'))
        self.max_portfolio_exposure = float(os.getenv('MAX_PORTFOLIO_EXPOSURE', '0.8'))
        self.max_single_position = float(os.getenv('MAX_SINGLE_POSITION', '0.2'))
        

    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if alert should be sent (rate limiting)"""
        current_time = now()
        last_alert = self.alert_timestamps.get(alert_type)
        
        if last_alert is None:
            self.alert_timestamps[alert_type] = current_time
            return True
        
        time_diff = (current_time - last_alert).total_seconds()
        if time_diff >= self.alert_cooldown:
            self.alert_timestamps[alert_type] = current_time
            return True
        
        return False
    
    def _add_risk_alert(self, alert_type: str, message: str, level: RiskLevel):
        """Add risk alert with rate limiting"""
        if self._should_send_alert(alert_type):
            alert = {
                'timestamp': now(),
                'type': alert_type,
                'message': message,
                'level': level.value
            }
            self.risk_alerts.append(alert)
            logger.warning(f"🚨 Risk Alert: {message}")
    
    def _persist_risk_event(self, event_type: str, data: dict):
        """Persist critical risk events to logs"""
        event = {
            'timestamp': now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        logger.critical(f"RISK_EVENT: {json.dumps(event)}")

    def add_position(self, symbol: str, quantity: float, price: float, timestamp: datetime):
        """Add position to risk tracking"""
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': price,
            'current_price': price,
            'timestamp': timestamp,
            'unrealized_pnl': 0.0
        }
        
        logger.info(f"📊 Position added: {symbol} {quantity} @ {price}")
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update position with current price"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position['current_price'] = current_price
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
    
    def add_trade_result(self, symbol: str, pnl: float, timestamp: datetime):
        """Add trade result to history"""
        trade = {
            'symbol': symbol,
            'pnl': pnl,
            'timestamp': timestamp,
            'is_win': pnl > 0
        }
        
        self.trade_history.append(trade)
        self.daily_pnl += pnl
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        logger.info(f"📈 Trade result: {symbol} P&L: ₹{pnl:,.2f}")
    
    def check_risk_limits(self, new_signal: Dict[str, Any], current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """Check if new signal violates risk limits"""
        try:
            # Update current prices
            for symbol in self.positions:
                if symbol in current_prices:
                    self.update_position_price(symbol, current_prices[symbol])
            
            # Check daily loss limit
            if self.daily_pnl < -self.risk_limits.max_daily_loss * self.peak_portfolio_value:
                return False, f"Daily loss limit exceeded: {self.daily_pnl:,.2f}"
            
            # Check consecutive losses
            if self.consecutive_losses >= self.risk_limits.max_consecutive_losses:
                return False, f"Too many consecutive losses: {self.consecutive_losses}"
            
            # Check win rate
            recent_trades = self.trade_history[-20:]  # Last 20 trades
            if len(recent_trades) >= 10:
                win_rate = sum(1 for t in recent_trades if t['is_win']) / len(recent_trades)
                if win_rate < self.risk_limits.min_win_rate:
                    return False, f"Win rate too low: {win_rate:.1%}"
            
            # Check portfolio exposure
            portfolio_risk = self.calculate_portfolio_risk(current_prices)
            if portfolio_risk.total_exposure > self.risk_limits.max_portfolio_exposure:
                return False, f"Portfolio exposure too high: {portfolio_risk.total_exposure:.1%}"
            
            # Check single position limit
            signal_symbol = new_signal.get('symbol')
            signal_quantity = new_signal.get('position_size', 0)
            signal_price = current_prices.get(signal_symbol, 0)
            signal_value = signal_quantity * signal_price
            
            if signal_value > self.risk_limits.max_single_position * portfolio_risk.total_value:
                return False, f"Single position too large: {signal_value:,.2f}"
            
            # Check correlation risk
            if portfolio_risk.correlation_risk > self.risk_limits.max_correlation_exposure:
                return False, f"Correlation risk too high: {portfolio_risk.correlation_risk:.1%}"
            
            # Check sector concentration
            for sector, exposure in portfolio_risk.sector_concentration.items():
                if exposure > self.risk_limits.max_sector_exposure:
                    return False, f"Sector concentration too high: {sector} {exposure:.1%}"
            
            # Check circuit breaker
            if self.circuit_breaker_active:
                return False, "Circuit breaker active"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            logger.error(f"❌ Error checking risk limits: {e}")
            return False, f"Risk check error: {e}"
    
    def calculate_portfolio_risk(self, current_prices: Dict[str, float]) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Update all position prices
            for symbol in self.positions:
                if symbol in current_prices:
                    self.update_position_price(symbol, current_prices[symbol])
            
            # Calculate total portfolio value
            total_value = 0.0
            total_exposure = 0.0
            position_risks = []
            
            for symbol, position in self.positions.items():
                current_price = position['current_price']
                quantity = position['quantity']
                market_value = abs(quantity * current_price)
                
                total_value += market_value
                total_exposure += market_value
                
                # Calculate position risk metrics
                volatility = self._estimate_volatility(symbol)
                beta = self._estimate_beta(symbol)
                var_95 = self._calculate_var_95(position['unrealized_pnl'], volatility)
                
                position_risk = PositionRisk(
                    symbol=symbol,
                    quantity=quantity,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=position['unrealized_pnl'],
                    risk_exposure=market_value / total_value if total_value > 0 else 0,
                    volatility=volatility,
                    beta=beta,
                    var_95=var_95
                )
                
                position_risks.append(position_risk)
            
            # Calculate portfolio metrics
            diversification_ratio = self._calculate_diversification_ratio(position_risks)
            correlation_risk = self._calculate_correlation_risk(position_risks)
            concentration_risk = self._calculate_concentration_risk(position_risks)
            sector_concentration = self._calculate_sector_concentration(position_risks)
            
            # Calculate portfolio VaR and volatility
            portfolio_var_95 = self._calculate_portfolio_var(position_risks)
            portfolio_volatility = self._calculate_portfolio_volatility(position_risks)
            
            # Calculate Sharpe ratio
            returns = [p.unrealized_pnl for p in position_risks]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown()
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                total_exposure, correlation_risk, concentration_risk, max_drawdown
            )
            
            return PortfolioRisk(
                total_value=total_value,
                total_exposure=total_exposure,
                diversification_ratio=diversification_ratio,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                sector_concentration=sector_concentration,
                portfolio_var_95=portfolio_var_95,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating portfolio risk: {e}")
            return PortfolioRisk(0, 0, 0, 0, 0, {}, 0, 0, 0, 0, RiskLevel.CRITICAL)
    
    def _estimate_volatility(self, symbol: str) -> float:
        """Estimate volatility for symbol"""
        # In real implementation, calculate from historical data
        # For now, use mock values
        volatility_map = {
            "NSE:NIFTY50-INDEX": 0.02,
            "NSE:NIFTYBANK-INDEX": 0.025,
            "NSE:FINNIFTY-INDEX": 0.03
        }
        return volatility_map.get(symbol, 0.02)
    
    def _estimate_beta(self, symbol: str) -> float:
        """Estimate beta for symbol"""
        # In real implementation, calculate from historical data
        beta_map = {
            "NSE:NIFTY50-INDEX": 1.0,
            "NSE:NIFTYBANK-INDEX": 1.2,
            "NSE:FINNIFTY-INDEX": 1.1
        }
        return beta_map.get(symbol, 1.0)
    
    def _calculate_var_95(self, pnl: float, volatility: float) -> float:
        """Calculate Value at Risk 95%"""
        # Simplified VaR calculation
        return pnl - 1.645 * volatility * abs(pnl)
    
    def _calculate_diversification_ratio(self, position_risks: List[PositionRisk]) -> float:
        """Calculate diversification ratio"""
        if not position_risks:
            return 0.0
        
        # Simplified diversification ratio
        num_positions = len(position_risks)
        return min(1.0, num_positions / 10.0)  # Max diversification at 10 positions
    
    def _calculate_correlation_risk(self, position_risks: List[PositionRisk]) -> float:
        """Calculate correlation risk"""
        if len(position_risks) < 2:
            return 0.0
        
        # Simplified correlation risk calculation
        # In real implementation, calculate actual correlations
        return 0.3  # Mock value
    
    def _calculate_concentration_risk(self, position_risks: List[PositionRisk]) -> float:
        """Calculate concentration risk"""
        if not position_risks:
            return 0.0
        
        # Calculate Herfindahl index
        total_value = sum(p.market_value for p in position_risks)
        if total_value == 0:
            return 0.0
        
        herfindahl = sum((p.market_value / total_value) ** 2 for p in position_risks)
        return herfindahl
    
    def _calculate_sector_concentration(self, position_risks: List[PositionRisk]) -> Dict[str, float]:
        """Calculate sector concentration"""
        sector_exposure = {}
        total_value = sum(p.market_value for p in position_risks)
        
        if total_value == 0:
            return {}
        
        for position in position_risks:
            # Determine sector from symbol
            symbol = position.symbol
            if 'NIFTY' in symbol:
                sector = 'INDEX'
            elif 'BANK' in symbol:
                sector = 'BANKING'
            elif 'FIN' in symbol:
                sector = 'FINANCIAL'
            else:
                sector = 'OTHER'
            
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position.market_value
        
        # Convert to percentages
        for sector in sector_exposure:
            sector_exposure[sector] /= total_value
        
        return sector_exposure
    
    def _calculate_portfolio_var(self, position_risks: List[PositionRisk]) -> float:
        """Calculate portfolio Value at Risk"""
        if not position_risks:
            return 0.0
        
        # Simplified portfolio VaR
        individual_vars = [p.var_95 for p in position_risks]
        return sum(individual_vars)
    
    def _calculate_portfolio_volatility(self, position_risks: List[PositionRisk]) -> float:
        """Calculate portfolio volatility"""
        if not position_risks:
            return 0.0
        
        # Simplified portfolio volatility calculation
        volatilities = [p.volatility for p in position_risks]
        weights = [p.risk_exposure for p in position_risks]
        
        # Weighted average volatility
        return sum(v * w for v, w in zip(volatilities, weights))
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.trade_history:
            return 0.0
        
        # Calculate running portfolio value
        portfolio_values = []
        running_value = self.peak_portfolio_value
        
        for trade in self.trade_history:
            running_value += trade['pnl']
            portfolio_values.append(running_value)
            self.peak_portfolio_value = max(self.peak_portfolio_value, running_value)
        
        if not portfolio_values:
            return 0.0
        
        # Calculate drawdown
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _determine_risk_level(self, total_exposure: float, correlation_risk: float, 
                            concentration_risk: float, max_drawdown: float) -> RiskLevel:
        """Determine overall risk level"""
        risk_score = 0
        
        # Exposure risk
        if total_exposure > 0.8:
            risk_score += 3
        elif total_exposure > 0.6:
            risk_score += 2
        elif total_exposure > 0.4:
            risk_score += 1
        
        # Correlation risk
        if correlation_risk > 0.7:
            risk_score += 3
        elif correlation_risk > 0.5:
            risk_score += 2
        elif correlation_risk > 0.3:
            risk_score += 1
        
        # Concentration risk
        if concentration_risk > 0.5:
            risk_score += 3
        elif concentration_risk > 0.3:
            risk_score += 2
        elif concentration_risk > 0.2:
            risk_score += 1
        
        # Drawdown risk
        if max_drawdown > 0.2:
            risk_score += 3
        elif max_drawdown > 0.1:
            risk_score += 2
        elif max_drawdown > 0.05:
            risk_score += 1
        
        # Determine risk level
        if risk_score >= 8:
            return RiskLevel.CRITICAL
        elif risk_score >= 6:
            return RiskLevel.HIGH
        elif risk_score >= 3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be activated"""
        try:
            # Check daily loss
            if self.daily_pnl < -self.risk_limits.circuit_breaker_threshold * self.peak_portfolio_value:
                self.circuit_breaker_active = True
                self.risk_alerts.append({
                    'timestamp': now(),
                    'type': 'CIRCUIT_BREAKER',
                    'message': f'Circuit breaker activated: Daily loss {self.daily_pnl:,.2f}'
                })
                logger.error(f"🚨 Circuit breaker activated: Daily loss {self.daily_pnl:,.2f}")
                return True
            
            # Check consecutive losses
            if self.consecutive_losses >= self.risk_limits.max_consecutive_losses:
                self.circuit_breaker_active = True
                self.risk_alerts.append({
                    'timestamp': now(),
                    'type': 'CIRCUIT_BREAKER',
                    'message': f'Circuit breaker activated: {self.consecutive_losses} consecutive losses'
                })
                logger.error(f"🚨 Circuit breaker activated: {self.consecutive_losses} consecutive losses")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking circuit breaker: {e}")
            return False
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.risk_alerts = []
        self.alert_timestamps = {}  # For rate limiting
        self.alert_cooldown = 300  # 5 minutes
        logger.info("🔄 Daily risk metrics reset")
    
    def get_risk_report(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            portfolio_risk = self.calculate_portfolio_risk(current_prices)
            
            report = {
                'timestamp': now().isoformat(),
                'portfolio_risk': {
                    'total_value': portfolio_risk.total_value,
                    'total_exposure': portfolio_risk.total_exposure,
                    'diversification_ratio': portfolio_risk.diversification_ratio,
                    'correlation_risk': portfolio_risk.correlation_risk,
                    'concentration_risk': portfolio_risk.concentration_risk,
                    'sector_concentration': portfolio_risk.sector_concentration,
                    'portfolio_var_95': portfolio_risk.portfolio_var_95,
                    'portfolio_volatility': portfolio_risk.portfolio_volatility,
                    'sharpe_ratio': portfolio_risk.sharpe_ratio,
                    'max_drawdown': portfolio_risk.max_drawdown,
                    'risk_level': portfolio_risk.risk_level.value
                },
                'daily_metrics': {
                    'daily_pnl': self.daily_pnl,
                    'consecutive_losses': self.consecutive_losses,
                    'circuit_breaker_active': self.circuit_breaker_active
                },
                'risk_limits': {
                    'max_daily_loss': self.risk_limits.max_daily_loss,
                    'max_portfolio_exposure': self.risk_limits.max_portfolio_exposure,
                    'max_single_position': self.risk_limits.max_single_position,
                    'max_correlation_exposure': self.risk_limits.max_correlation_exposure,
                    'max_sector_exposure': self.risk_limits.max_sector_exposure
                },
                'alerts': self.risk_alerts[-10:]  # Last 10 alerts
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating risk report: {e}")
            return {}

def main():
    """Main function for testing"""
    # Create risk manager
    risk_limits = RiskLimits(
        max_daily_loss=0.05,
        max_portfolio_exposure=0.8,
        max_single_position=0.2,
        max_correlation_exposure=0.6,
        max_sector_exposure=0.4
    )
    
    risk_manager = AdvancedRiskManager(risk_limits)
    
    # Add some positions
    risk_manager.add_position("NSE:NIFTY50-INDEX", 100, 19500, now())
    risk_manager.add_position("NSE:NIFTYBANK-INDEX", 50, 45000, now())
    
    # Update prices
    current_prices = {
        "NSE:NIFTY50-INDEX": 19600,
        "NSE:NIFTYBANK-INDEX": 45100
    }
    
    # Check risk
    test_signal = {
        'symbol': 'NSE:FINNIFTY-INDEX',
        'position_size': 100,
        'confidence': 75
    }
    
    can_trade, reason = risk_manager.check_risk_limits(test_signal, current_prices)
    print(f"Can trade: {can_trade}, Reason: {reason}")
    
    # Generate risk report
    report = risk_manager.get_risk_report(current_prices)
    print(f"Risk report: {json.dumps(report, indent=2, default=str)}")

if __name__ == "__main__":
    main()
