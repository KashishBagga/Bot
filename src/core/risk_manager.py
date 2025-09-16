"""
Advanced Risk Management System
==============================
Implements portfolio-level risk management with position sizing,
correlation analysis, drawdown protection, and volatility adjustments
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    position_size: float
    risk_per_trade: float
    correlation_risk: float
    volatility_risk: float
    drawdown_risk: float
    overall_risk: RiskLevel

@dataclass
class PortfolioRisk:
    """Portfolio-level risk assessment"""
    total_exposure: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%
    sharpe_ratio: float
    correlation_matrix: pd.DataFrame
    risk_level: RiskLevel

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, max_portfolio_risk: float = 0.02, max_position_risk: float = 0.01):
        self.max_portfolio_risk = max_portfolio_risk  # 2% max portfolio risk
        self.max_position_risk = max_position_risk    # 1% max position risk
        self.max_correlation = 0.7                    # Max correlation between positions
        self.max_drawdown = 0.15                      # 15% max drawdown
        self.volatility_lookback = 20                 # Days for volatility calculation
        
        # Risk limits
        self.daily_loss_limit = 0.05                  # 5% daily loss limit
        self.weekly_loss_limit = 0.10                 # 10% weekly loss limit
        self.monthly_loss_limit = 0.20                # 20% monthly loss limit
        
        logger.info("🛡️ Risk Manager initialized")
    
    def calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate historical volatility."""
        if len(prices) < period:
            return 0.0
        
        returns = prices.pct_change().dropna()
        if len(returns) < period:
            return 0.0
        
        return returns.tail(period).std() * np.sqrt(252)  # Annualized
    
    def calculate_correlation_matrix(self, price_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate correlation matrix between symbols."""
        if len(price_data) < 2:
            return pd.DataFrame()
        
        # Align all price series
        aligned_data = {}
        for symbol, prices in price_data.items():
            if len(prices) > 0:
                aligned_data[symbol] = prices
        
        if len(aligned_data) < 2:
            return pd.DataFrame()
        
        # Create DataFrame and calculate correlations
        df = pd.DataFrame(aligned_data)
        returns = df.pct_change().dropna()
        
        if len(returns) < 10:  # Need minimum data
            return pd.DataFrame()
        
        return returns.corr()
    
    def calculate_position_risk(self, symbol: str, position_size: float, 
                              entry_price: float, stop_loss: float,
                              volatility: float, correlation_risk: float = 0.0) -> RiskMetrics:
        """Calculate risk metrics for a single position."""
        
        # Position risk (risk per trade)
        risk_per_trade = abs(entry_price - stop_loss) / entry_price * position_size
        
        # Volatility risk adjustment
        volatility_risk = min(volatility * 2, 1.0)  # Cap at 100%
        
        # Correlation risk
        correlation_risk = min(correlation_risk, 1.0)
        
        # Drawdown risk (based on position size and volatility)
        drawdown_risk = position_size * volatility_risk
        
        # Overall risk calculation
        overall_risk_score = (risk_per_trade + volatility_risk + correlation_risk + drawdown_risk) / 4
        
        # Classify risk level
        if overall_risk_score < 0.3:
            risk_level = RiskLevel.LOW
        elif overall_risk_score < 0.6:
            risk_level = RiskLevel.MEDIUM
        elif overall_risk_score < 0.8:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        return RiskMetrics(
            position_size=position_size,
            risk_per_trade=risk_per_trade,
            correlation_risk=correlation_risk,
            volatility_risk=volatility_risk,
            drawdown_risk=drawdown_risk,
            overall_risk=risk_level
        )
    
    def calculate_portfolio_risk(self, positions: Dict[str, Dict], 
                               price_data: Dict[str, pd.Series]) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics."""
        
        if not positions:
            return PortfolioRisk(
                total_exposure=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                sharpe_ratio=0.0,
                correlation_matrix=pd.DataFrame(),
                risk_level=RiskLevel.LOW
            )
        
        # Calculate total exposure
        total_exposure = sum(pos.get('position_size', 0) for pos in positions.values())
        
        # Calculate correlation matrix
        correlation_matrix = self.calculate_correlation_matrix(price_data)
        
        # Calculate portfolio volatility
        portfolio_volatility = 0.0
        if correlation_matrix is not None and not correlation_matrix.empty:
            # Simplified portfolio volatility calculation
            individual_volatilities = []
            for symbol in positions.keys():
                if symbol in price_data:
                    vol = self.calculate_volatility(price_data[symbol])
                    individual_volatilities.append(vol)
            
            if individual_volatilities:
                portfolio_volatility = np.mean(individual_volatilities)
        
        # Calculate Value at Risk (95%)
        var_95 = total_exposure * portfolio_volatility * 1.645  # 95% VaR
        
        # Calculate maximum drawdown
        max_drawdown = min(total_exposure * 0.5, self.max_drawdown)  # Simplified
        
        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = 0.0  # Would need returns data for accurate calculation
        
        # Determine overall risk level
        risk_score = (total_exposure + portfolio_volatility + max_drawdown) / 3
        
        if risk_score < 0.3:
            risk_level = RiskLevel.LOW
        elif risk_score < 0.6:
            risk_level = RiskLevel.MEDIUM
        elif risk_score < 0.8:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        return PortfolioRisk(
            total_exposure=total_exposure,
            max_drawdown=max_drawdown,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            correlation_matrix=correlation_matrix,
            risk_level=risk_level
        )
    
    def check_risk_limits(self, new_position: Dict, existing_positions: Dict[str, Dict],
                         price_data: Dict[str, pd.Series]) -> Tuple[bool, str]:
        """Check if a new position violates risk limits."""
        
        symbol = new_position.get('symbol')
        position_size = new_position.get('position_size', 0)
        entry_price = new_position.get('price', 0)
        stop_loss = new_position.get('stop_loss_price', 0)
        
        # Check position size limit
        if position_size > self.max_position_risk:
            return False, f"Position size {position_size:.2f} exceeds limit {self.max_position_risk:.2f}"
        
        # Check portfolio exposure limit
        total_exposure = sum(pos.get('position_size', 0) for pos in existing_positions.values())
        total_exposure += position_size
        
        if total_exposure > self.max_portfolio_risk:
            return False, f"Total exposure {total_exposure:.2f} exceeds limit {self.max_portfolio_risk:.2f}"
        
        # Check correlation limits
        if symbol in existing_positions:
            # Calculate correlation with existing positions
            correlation_matrix = self.calculate_correlation_matrix(price_data)
            
            if not correlation_matrix.empty and symbol in correlation_matrix.index:
                max_correlation = 0.0
                for existing_symbol in existing_positions.keys():
                    if existing_symbol in correlation_matrix.columns:
                        corr = abs(correlation_matrix.loc[symbol, existing_symbol])
                        max_correlation = max(max_correlation, corr)
                
                if max_correlation > self.max_correlation:
                    return False, f"Correlation {max_correlation:.2f} exceeds limit {self.max_correlation:.2f}"
        
        # Check volatility limits
        if symbol in price_data:
            volatility = self.calculate_volatility(price_data[symbol])
            if volatility > 0.5:  # 50% annualized volatility limit
                return False, f"Volatility {volatility:.2f} exceeds limit 0.50"
        
        return True, "Risk limits satisfied"
    
    def adjust_position_size(self, symbol: str, base_position_size: float,
                           volatility: float, correlation_risk: float,
                           portfolio_exposure: float) -> float:
        """Adjust position size based on risk factors."""
        
        adjusted_size = base_position_size
        
        # Volatility adjustment (reduce size for high volatility)
        if volatility > 0.3:
            volatility_factor = max(0.5, 1.0 - (volatility - 0.3) * 2)
            adjusted_size *= volatility_factor
        
        # Correlation adjustment (reduce size for high correlation)
        if correlation_risk > 0.5:
            correlation_factor = max(0.7, 1.0 - correlation_risk)
            adjusted_size *= correlation_factor
        
        # Portfolio exposure adjustment
        if portfolio_exposure > self.max_portfolio_risk * 0.8:
            exposure_factor = max(0.5, 1.0 - (portfolio_exposure - self.max_portfolio_risk * 0.8) * 2)
            adjusted_size *= exposure_factor
        
        # Ensure minimum and maximum bounds
        adjusted_size = max(0.1, min(adjusted_size, self.max_position_risk))
        
        return round(adjusted_size, 2)
    
    def get_risk_summary(self, positions: Dict[str, Dict], 
                        price_data: Dict[str, pd.Series]) -> Dict:
        """Get comprehensive risk summary."""
        
        portfolio_risk = self.calculate_portfolio_risk(positions, price_data)
        
        # Count positions by risk level
        risk_counts = {level.value: 0 for level in RiskLevel}
        
        for symbol, position in positions.items():
            if symbol in price_data:
                volatility = self.calculate_volatility(price_data[symbol])
                position_risk = self.calculate_position_risk(
                    symbol, position.get('position_size', 0),
                    position.get('price', 0), position.get('stop_loss_price', 0),
                    volatility
                )
                risk_counts[position_risk.overall_risk.value] += 1
        
        return {
            'portfolio_risk_level': portfolio_risk.risk_level.value,
            'total_exposure': portfolio_risk.total_exposure,
            'max_drawdown': portfolio_risk.max_drawdown,
            'var_95': portfolio_risk.var_95,
            'position_count': len(positions),
            'risk_distribution': risk_counts,
            'correlation_matrix': portfolio_risk.correlation_matrix.to_dict() if not portfolio_risk.correlation_matrix.empty else {}
        }
    def should_execute_signal(self, signal: Dict[str, Any], current_prices: Dict[str, float]) -> bool:
        """Check if signal should be executed based on risk limits"""
        try:
            # Basic risk checks
            confidence = signal.get('confidence', 0)
            if confidence < 50:  # Minimum confidence threshold
                return False
            
            # Check position size
            position_size = signal.get('position_size', 0)
            if position_size <= 0:
                return False
            
            # Check if we have price data
            symbol = signal.get('symbol')
            if symbol not in current_prices:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking signal execution: {e}")
            return False

