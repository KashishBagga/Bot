#!/usr/bin/env python3
"""
Enhanced Risk Management with Portfolio-Level Controls
Comprehensive risk management with realistic position sizing and portfolio optimization
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    portfolio_value: float
    total_exposure: float
    exposure_percentage: float
    daily_pnl: float
    daily_drawdown: float
    max_drawdown: float
    var_95: float
    var_99: float
    sharpe_ratio: float
    max_position_size: float
    correlation_risk: float

@dataclass
class PositionRisk:
    """Risk metrics for individual position"""
    symbol: str
    position_size: float
    position_value: float
    risk_per_trade: float
    stop_loss_distance: float
    expected_return: float
    volatility: float
    correlation_with_portfolio: float
    margin_requirement: float

class EnhancedRiskManager:
    """Enhanced risk manager with portfolio-level controls"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limits = {
            'max_portfolio_exposure': 0.6,  # 60% of capital
            'max_daily_drawdown': 0.03,     # 3% daily drawdown
            'max_position_size': 0.1,       # 10% per position
            'max_correlation': 0.7,         # 70% correlation limit
            'max_daily_loss': 0.02,         # 2% daily loss limit
            'min_win_rate': 0.4,            # 40% minimum win rate
            'max_consecutive_losses': 5,    # 5 consecutive losses
            'circuit_breaker_threshold': 0.05  # 5% circuit breaker
        }
        
        self.positions = {}
        self.daily_pnl_history = []
        self.trade_history = []
        self.circuit_breaker_active = False
        self.consecutive_losses = 0
        self.last_reset_date = datetime.now().date()
        
    def calculate_portfolio_risk(self, positions: Dict[str, PositionRisk]) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            total_value = sum(pos.position_value for pos in positions.values())
            total_exposure = sum(abs(pos.position_value) for pos in positions.values())
            exposure_percentage = total_exposure / self.current_capital if self.current_capital > 0 else 0
            
            # Calculate daily P&L
            daily_pnl = self._calculate_daily_pnl(positions)
            
            # Calculate drawdown
            daily_drawdown = self._calculate_daily_drawdown()
            max_drawdown = self._calculate_max_drawdown()
            
            # Calculate VaR
            var_95, var_99 = self._calculate_var(positions)
            
            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio()
            
            # Calculate correlation risk
            correlation_risk = self._calculate_correlation_risk(positions)
            
            # Calculate max position size
            max_position_size = max((pos.position_value / self.current_capital for pos in positions.values()), default=0)
            
            return RiskMetrics(
                portfolio_value=total_value,
                total_exposure=total_exposure,
                exposure_percentage=exposure_percentage,
                daily_pnl=daily_pnl,
                daily_drawdown=daily_drawdown,
                max_drawdown=max_drawdown,
                var_95=var_95,
                var_99=var_99,
                sharpe_ratio=sharpe_ratio,
                max_position_size=max_position_size,
                correlation_risk=correlation_risk
            )
            
        except Exception as e:
            logger.error(f"❌ Portfolio risk calculation failed: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def check_risk_limits(self, new_position: PositionRisk, 
                         existing_positions: Dict[str, PositionRisk]) -> Tuple[bool, str]:
        """Check if new position violates risk limits"""
        try:
            # Check portfolio exposure limit
            total_exposure = sum(abs(pos.position_value) for pos in existing_positions.values())
            new_total_exposure = total_exposure + abs(new_position.position_value)
            
            if new_total_exposure / self.current_capital > self.risk_limits['max_portfolio_exposure']:
                return False, f"Portfolio exposure limit exceeded: {new_total_exposure/self.current_capital:.1%}"
            
            # Check position size limit
            position_size_ratio = abs(new_position.position_value) / self.current_capital
            if position_size_ratio > self.risk_limits['max_position_size']:
                return False, f"Position size limit exceeded: {position_size_ratio:.1%}"
            
            # Check correlation limit
            if new_position.correlation_with_portfolio > self.risk_limits['max_correlation']:
                return False, f"Correlation limit exceeded: {new_position.correlation_with_portfolio:.1%}"
            
            # Check daily loss limit
            if self._get_daily_pnl() < -self.risk_limits['max_daily_loss'] * self.current_capital:
                return False, f"Daily loss limit exceeded: {self._get_daily_pnl():.2f}"
            
            # Check circuit breaker
            if self.circuit_breaker_active:
                return False, "Circuit breaker is active"
            
            # Check consecutive losses
            if self.consecutive_losses >= self.risk_limits['max_consecutive_losses']:
                return False, f"Too many consecutive losses: {self.consecutive_losses}"
            
            return True, "Risk limits OK"
            
        except Exception as e:
            logger.error(f"❌ Risk limit check failed: {e}")
            return False, f"Risk check error: {e}"
    
    def calculate_optimal_position_size(self, symbol: str, entry_price: float,
                                      stop_loss_price: float, expected_return: float,
                                      volatility: float, confidence: float) -> float:
        """Calculate optimal position size using Kelly Criterion and risk management"""
        try:
            # Calculate risk per trade
            risk_per_trade = abs(entry_price - stop_loss_price) / entry_price
            
            # Calculate expected value
            win_probability = confidence
            loss_probability = 1 - confidence
            avg_win = expected_return * entry_price
            avg_loss = risk_per_trade * entry_price
            
            expected_value = win_probability * avg_win - loss_probability * avg_loss
            
            # Kelly Criterion
            if avg_loss > 0:
                kelly_fraction = (win_probability * avg_win - loss_probability * avg_loss) / avg_win
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.1  # Default 10%
            
            # Adjust for volatility
            volatility_adjustment = 1 / (1 + volatility)
            adjusted_fraction = kelly_fraction * volatility_adjustment
            
            # Adjust for confidence
            confidence_adjustment = confidence ** 2  # Square to reduce impact of low confidence
            final_fraction = adjusted_fraction * confidence_adjustment
            
            # Calculate position size
            position_value = self.current_capital * final_fraction
            
            # Apply risk limits
            max_position_value = self.current_capital * self.risk_limits['max_position_size']
            position_value = min(position_value, max_position_value)
            
            # Calculate quantity
            quantity = position_value / entry_price
            
            logger.info(f"📊 Position sizing for {symbol}:")
            logger.info(f"   Kelly fraction: {kelly_fraction:.3f}")
            logger.info(f"   Volatility adjustment: {volatility_adjustment:.3f}")
            logger.info(f"   Confidence adjustment: {confidence_adjustment:.3f}")
            logger.info(f"   Final fraction: {final_fraction:.3f}")
            logger.info(f"   Position value: {position_value:.2f}")
            logger.info(f"   Quantity: {quantity:.2f}")
            
            return quantity
            
        except Exception as e:
            logger.error(f"❌ Position sizing calculation failed: {e}")
            return 0.0
    
    def calculate_portfolio_optimization(self, candidates: List[PositionRisk]) -> Dict[str, float]:
        """Calculate optimal portfolio allocation using mean-variance optimization"""
        try:
            if not candidates:
                return {}
            
            # Prepare data for optimization
            n_assets = len(candidates)
            expected_returns = np.array([cand.expected_return for cand in candidates])
            volatilities = np.array([cand.volatility for cand in candidates])
            
            # Create correlation matrix (simplified)
            correlation_matrix = np.eye(n_assets)
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    correlation_matrix[i, j] = 0.3  # Default correlation
                    correlation_matrix[j, i] = 0.3
            
            # Create covariance matrix
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            
            # Objective function (negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
                if portfolio_volatility == 0:
                    return -portfolio_return
                return -(portfolio_return / portfolio_volatility)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds (0 to max position size)
            bounds = [(0, self.risk_limits['max_position_size']) for _ in range(n_assets)]
            
            # Initial guess
            x0 = np.ones(n_assets) / n_assets
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = result.x
                allocation = {}
                for i, candidate in enumerate(candidates):
                    if optimal_weights[i] > 0.01:  # Only include weights > 1%
                        allocation[candidate.symbol] = optimal_weights[i]
                
                logger.info(f"✅ Portfolio optimization completed")
                logger.info(f"   Optimal allocation: {allocation}")
                
                return allocation
            else:
                logger.warning(f"⚠️ Portfolio optimization failed: {result.message}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Portfolio optimization failed: {e}")
            return {}
    
    def check_circuit_breaker(self, current_pnl: float) -> bool:
        """Check if circuit breaker should be triggered"""
        try:
            # Check daily drawdown
            daily_drawdown = abs(current_pnl) / self.current_capital
            if daily_drawdown > self.risk_limits['circuit_breaker_threshold']:
                self.circuit_breaker_active = True
                logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: Daily drawdown {daily_drawdown:.1%}")
                return True
            
            # Check consecutive losses
            if self.consecutive_losses >= self.risk_limits['max_consecutive_losses']:
                self.circuit_breaker_active = True
                logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {self.consecutive_losses} consecutive losses")
                return True
            
            # Check win rate
            if len(self.trade_history) >= 10:
                recent_trades = self.trade_history[-10:]
                win_rate = sum(1 for trade in recent_trades if trade['pnl'] > 0) / len(recent_trades)
                if win_rate < self.risk_limits['min_win_rate']:
                    self.circuit_breaker_active = True
                    logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: Win rate {win_rate:.1%}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Circuit breaker check failed: {e}")
            return False
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention required)"""
        try:
            self.circuit_breaker_active = False
            self.consecutive_losses = 0
            self.last_reset_date = datetime.now().date()
            logger.info("✅ Circuit breaker reset")
            
        except Exception as e:
            logger.error(f"❌ Circuit breaker reset failed: {e}")
    
    def update_trade_result(self, trade_id: str, pnl: float, exit_reason: str):
        """Update trade result and risk metrics"""
        try:
            # Add to trade history
            trade_record = {
                'trade_id': trade_id,
                'pnl': pnl,
                'exit_reason': exit_reason,
                'timestamp': datetime.now().isoformat()
            }
            self.trade_history.append(trade_record)
            
            # Update consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Update capital
            self.current_capital += pnl
            
            # Update daily P&L
            self.daily_pnl_history.append(pnl)
            
            # Keep only last 30 days
            if len(self.daily_pnl_history) > 30:
                self.daily_pnl_history = self.daily_pnl_history[-30:]
            
            logger.info(f"📊 Trade result updated: {trade_id}, P&L: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Trade result update failed: {e}")
    
    def _calculate_daily_pnl(self, positions: Dict[str, PositionRisk]) -> float:
        """Calculate daily P&L from positions"""
        try:
            # This would typically calculate unrealized P&L
            # For now, return 0 as we don't have current prices
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Daily P&L calculation failed: {e}")
            return 0.0
    
    def _calculate_daily_drawdown(self) -> float:
        """Calculate daily drawdown"""
        try:
            if not self.daily_pnl_history:
                return 0.0
            
            # Calculate running maximum
            running_max = 0.0
            max_drawdown = 0.0
            
            for pnl in self.daily_pnl_history:
                running_max = max(running_max, pnl)
                drawdown = running_max - pnl
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown / self.current_capital if self.current_capital > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Daily drawdown calculation failed: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            if not self.daily_pnl_history:
                return 0.0
            
            # Calculate cumulative returns
            cumulative_returns = np.cumsum(self.daily_pnl_history)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdown
            drawdown = running_max - cumulative_returns
            
            return np.max(drawdown) / self.current_capital if self.current_capital > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Max drawdown calculation failed: {e}")
            return 0.0
    
    def _calculate_var(self, positions: Dict[str, PositionRisk]) -> Tuple[float, float]:
        """Calculate Value at Risk"""
        try:
            if not self.daily_pnl_history:
                return 0.0, 0.0
            
            # Calculate returns
            returns = np.array(self.daily_pnl_history)
            
            # Calculate VaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            return var_95, var_99
            
        except Exception as e:
            logger.error(f"❌ VaR calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if not self.daily_pnl_history or len(self.daily_pnl_history) < 2:
                return 0.0
            
            returns = np.array(self.daily_pnl_history)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            # Annualized Sharpe ratio
            sharpe = (mean_return / std_return) * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            logger.error(f"❌ Sharpe ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self, positions: Dict[str, PositionRisk]) -> float:
        """Calculate correlation risk in portfolio"""
        try:
            if len(positions) < 2:
                return 0.0
            
            # Calculate weighted average correlation
            correlations = [pos.correlation_with_portfolio for pos in positions.values()]
            weights = [abs(pos.position_value) for pos in positions.values()]
            
            if sum(weights) == 0:
                return 0.0
            
            weighted_correlation = sum(c * w for c, w in zip(correlations, weights)) / sum(weights)
            return weighted_correlation
            
        except Exception as e:
            logger.error(f"❌ Correlation risk calculation failed: {e}")
            return 0.0
    
    def _get_daily_pnl(self) -> float:
        """Get daily P&L"""
        try:
            if not self.daily_pnl_history:
                return 0.0
            
            # Return today's P&L (last entry)
            return self.daily_pnl_history[-1]
            
        except Exception as e:
            logger.error(f"❌ Daily P&L retrieval failed: {e}")
            return 0.0
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report"""
        try:
            return {
                'current_capital': self.current_capital,
                'initial_capital': self.initial_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
                'consecutive_losses': self.consecutive_losses,
                'circuit_breaker_active': self.circuit_breaker_active,
                'daily_pnl': self._get_daily_pnl(),
                'max_drawdown': self._calculate_max_drawdown(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'total_trades': len(self.trade_history),
                'win_rate': sum(1 for trade in self.trade_history if trade['pnl'] > 0) / len(self.trade_history) if self.trade_history else 0,
                'risk_limits': self.risk_limits,
                'last_reset_date': self.last_reset_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Risk report generation failed: {e}")
            return {}

# Global enhanced risk manager instance
enhanced_risk_manager = EnhancedRiskManager()

# Convenience functions
def calculate_portfolio_risk(positions: Dict[str, PositionRisk]) -> RiskMetrics:
    """Calculate portfolio risk metrics"""
    return enhanced_risk_manager.calculate_portfolio_risk(positions)

def check_risk_limits(new_position: PositionRisk, existing_positions: Dict[str, PositionRisk]) -> Tuple[bool, str]:
    """Check risk limits for new position"""
    return enhanced_risk_manager.check_risk_limits(new_position, existing_positions)

def calculate_optimal_position_size(symbol: str, entry_price: float, stop_loss_price: float,
                                  expected_return: float, volatility: float, confidence: float) -> float:
    """Calculate optimal position size"""
    return enhanced_risk_manager.calculate_optimal_position_size(
        symbol, entry_price, stop_loss_price, expected_return, volatility, confidence
    )

def check_circuit_breaker(current_pnl: float) -> bool:
    """Check circuit breaker"""
    return enhanced_risk_manager.check_circuit_breaker(current_pnl)

def get_risk_report() -> Dict[str, Any]:
    """Get risk report"""
    return enhanced_risk_manager.get_risk_report()
