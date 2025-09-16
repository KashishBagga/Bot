#!/usr/bin/env python3
"""
Per-Strategy Capital Allocation & Optimizer
HIGH PRIORITY #4: Dynamic capital allocation across strategies
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyStatus(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DISABLED = "DISABLED"
    UNDERPERFORMING = "UNDERPERFORMING"

@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    current_drawdown: float
    volatility: float
    var_95: float
    last_updated: datetime

@dataclass
class StrategyAllocation:
    """Strategy allocation configuration"""
    strategy_name: str
    current_allocation: float  # Current % of portfolio
    target_allocation: float   # Target % of portfolio
    min_allocation: float      # Minimum % of portfolio
    max_allocation: float      # Maximum % of portfolio
    performance_threshold: float  # Sharpe ratio threshold
    drawdown_threshold: float     # Max drawdown threshold
    rebalance_threshold: float    # Allocation change threshold
    last_rebalance: datetime
    status: StrategyStatus

class CapitalEfficiencyOptimizer:
    """Capital efficiency optimizer for strategy allocation"""
    
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.strategy_metrics = {}
        self.strategy_allocations = {}
        self.performance_history = []
        self.rebalance_history = []
        
        # Optimization parameters
        self.min_allocation = 0.05  # 5% minimum allocation
        self.max_allocation = 0.40  # 40% maximum allocation
        self.rebalance_threshold = 0.05  # 5% change threshold
        self.performance_window = 30  # 30 days performance window
        self.min_trades_for_evaluation = 10  # Minimum trades for evaluation
        
        # Risk parameters
        self.max_portfolio_volatility = 0.20  # 20% max portfolio volatility
        self.max_correlation = 0.70  # 70% max correlation between strategies
        self.target_sharpe_ratio = 1.5  # Target Sharpe ratio
        
    def add_strategy(self, strategy_name: str, initial_allocation: float = 0.20):
        """Add strategy to optimizer"""
        if strategy_name in self.strategy_allocations:
            logger.warning(f"⚠️ Strategy {strategy_name} already exists")
            return
        
        self.strategy_allocations[strategy_name] = StrategyAllocation(
            strategy_name=strategy_name,
            current_allocation=initial_allocation,
            target_allocation=initial_allocation,
            min_allocation=self.min_allocation,
            max_allocation=self.max_allocation,
            performance_threshold=1.0,  # Sharpe ratio threshold
            drawdown_threshold=0.15,    # 15% max drawdown
            rebalance_threshold=self.rebalance_threshold,
            last_rebalance=datetime.now(),
            status=StrategyStatus.ACTIVE
        )
        
        logger.info(f"✅ Strategy added: {strategy_name} with {initial_allocation:.1%} allocation")
    
    def update_strategy_metrics(self, strategy_name: str, trades: List[Dict[str, Any]]):
        """Update strategy performance metrics"""
        try:
            if strategy_name not in self.strategy_allocations:
                logger.error(f"❌ Strategy {strategy_name} not found")
                return
            
            if len(trades) < self.min_trades_for_evaluation:
                logger.warning(f"⚠️ Insufficient trades for {strategy_name}: {len(trades)}")
                return
            
            # Calculate metrics
            metrics = self._calculate_strategy_metrics(strategy_name, trades)
            self.strategy_metrics[strategy_name] = metrics
            
            # Store performance history
            self.performance_history.append({
                'strategy_name': strategy_name,
                'timestamp': datetime.now(),
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'total_pnl': metrics.total_pnl
            })
            
            logger.info(f"📊 Metrics updated for {strategy_name}: Sharpe={metrics.sharpe_ratio:.2f}, Win Rate={metrics.win_rate:.1%}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update metrics for {strategy_name}: {e}")
    
    def _calculate_strategy_metrics(self, strategy_name: str, trades: List[Dict[str, Any]]) -> StrategyMetrics:
        """Calculate comprehensive strategy metrics"""
        try:
            # Extract P&L data
            pnl_data = [trade['pnl'] for trade in trades if 'pnl' in trade]
            
            if not pnl_data:
                raise ValueError("No P&L data available")
            
            # Basic metrics
            total_trades = len(pnl_data)
            winning_trades = sum(1 for pnl in pnl_data if pnl > 0)
            losing_trades = sum(1 for pnl in pnl_data if pnl < 0)
            total_pnl = sum(pnl_data)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Average win/loss
            wins = [pnl for pnl in pnl_data if pnl > 0]
            losses = [pnl for pnl in pnl_data if pnl < 0]
            average_win = np.mean(wins) if wins else 0
            average_loss = np.mean(losses) if losses else 0
            
            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Risk metrics
            returns = np.array(pnl_data)
            volatility = np.std(returns) if len(returns) > 1 else 0
            
            # Sharpe ratio (assuming risk-free rate = 0)
            sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns if r < 0]
            downside_volatility = np.std(downside_returns) if len(downside_returns) > 1 else 0
            sortino_ratio = np.mean(returns) / downside_volatility if downside_volatility > 0 else 0
            
            # Drawdown calculation
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
            
            # Max drawdown duration
            max_drawdown_duration = self._calculate_max_drawdown_duration(drawdowns)
            
            # Current drawdown
            current_drawdown = abs(drawdowns[-1]) if len(drawdowns) > 0 else 0
            
            # VaR 95%
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            
            return StrategyMetrics(
                strategy_name=strategy_name,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                total_pnl=total_pnl,
                win_rate=win_rate,
                average_win=average_win,
                average_loss=average_loss,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                current_drawdown=current_drawdown,
                volatility=volatility,
                var_95=var_95,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Metrics calculation failed for {strategy_name}: {e}")
            raise
    
    def _calculate_max_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """Calculate maximum drawdown duration"""
        try:
            if len(drawdowns) == 0:
                return 0
            
            max_duration = 0
            current_duration = 0
            
            for drawdown in drawdowns:
                if drawdown < 0:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0
            
            return max_duration
            
        except Exception as e:
            logger.error(f"❌ Max drawdown duration calculation failed: {e}")
            return 0
    
    def optimize_allocations(self) -> Dict[str, float]:
        """Optimize strategy allocations based on performance"""
        try:
            logger.info("🔍 Optimizing strategy allocations")
            
            # Check if we have enough data
            if not self.strategy_metrics:
                logger.warning("⚠️ No strategy metrics available for optimization")
                return {}
            
            # Calculate new allocations
            new_allocations = self._calculate_optimal_allocations()
            
            # Check if rebalancing is needed
            rebalance_needed = self._check_rebalance_needed(new_allocations)
            
            if rebalance_needed:
                # Update allocations
                for strategy_name, new_allocation in new_allocations.items():
                    if strategy_name in self.strategy_allocations:
                        old_allocation = self.strategy_allocations[strategy_name].current_allocation
                        self.strategy_allocations[strategy_name].current_allocation = new_allocation
                        self.strategy_allocations[strategy_name].target_allocation = new_allocation
                        self.strategy_allocations[strategy_name].last_rebalance = datetime.now()
                        
                        logger.info(f"🔄 {strategy_name}: {old_allocation:.1%} → {new_allocation:.1%}")
                
                # Record rebalance
                self.rebalance_history.append({
                    'timestamp': datetime.now(),
                    'allocations': new_allocations.copy(),
                    'reason': 'Performance optimization'
                })
                
                logger.info("✅ Strategy allocations optimized")
            else:
                logger.info("ℹ️ No rebalancing needed")
            
            return new_allocations
            
        except Exception as e:
            logger.error(f"❌ Allocation optimization failed: {e}")
            return {}
    
    def _calculate_optimal_allocations(self) -> Dict[str, float]:
        """Calculate optimal allocations using performance-based weighting"""
        try:
            # Get performance scores for each strategy
            performance_scores = {}
            
            for strategy_name, metrics in self.strategy_metrics.items():
                if strategy_name not in self.strategy_allocations:
                    continue
                
                allocation = self.strategy_allocations[strategy_name]
                
                # Skip if strategy is disabled or underperforming
                if allocation.status in [StrategyStatus.DISABLED, StrategyStatus.UNDERPERFORMING]:
                    performance_scores[strategy_name] = 0.0
                    continue
                
                # Calculate performance score
                score = self._calculate_performance_score(metrics, allocation)
                performance_scores[strategy_name] = score
            
            # Normalize scores to allocations
            total_score = sum(performance_scores.values())
            if total_score == 0:
                # Equal allocation if no performance data
                num_strategies = len(self.strategy_allocations)
                return {name: 1.0 / num_strategies for name in self.strategy_allocations.keys()}
            
            # Calculate allocations
            allocations = {}
            for strategy_name, score in performance_scores.items():
                raw_allocation = score / total_score
                
                # Apply constraints
                min_alloc = self.strategy_allocations[strategy_name].min_allocation
                max_alloc = self.strategy_allocations[strategy_name].max_allocation
                
                constrained_allocation = max(min_alloc, min(max_alloc, raw_allocation))
                allocations[strategy_name] = constrained_allocation
            
            # Normalize to ensure total = 1.0
            total_allocation = sum(allocations.values())
            if total_allocation > 0:
                allocations = {name: alloc / total_allocation for name, alloc in allocations.items()}
            
            return allocations
            
        except Exception as e:
            logger.error(f"❌ Optimal allocation calculation failed: {e}")
            return {}
    
    def _calculate_performance_score(self, metrics: StrategyMetrics, allocation: StrategyAllocation) -> float:
        """Calculate performance score for strategy"""
        try:
            # Base score from Sharpe ratio
            sharpe_score = max(0, metrics.sharpe_ratio / self.target_sharpe_ratio)
            
            # Win rate bonus
            win_rate_bonus = max(0, (metrics.win_rate - 0.5) * 2)  # Bonus for >50% win rate
            
            # Drawdown penalty
            drawdown_penalty = max(0, metrics.current_drawdown / allocation.drawdown_threshold)
            
            # Volatility penalty
            volatility_penalty = max(0, metrics.volatility / self.max_portfolio_volatility)
            
            # Profit factor bonus
            profit_factor_bonus = min(1.0, metrics.profit_factor / 2.0)  # Bonus for >2.0 profit factor
            
            # Calculate final score
            score = (sharpe_score + win_rate_bonus + profit_factor_bonus - 
                    drawdown_penalty - volatility_penalty)
            
            return max(0, score)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"❌ Performance score calculation failed: {e}")
            return 0.0
    
    def _check_rebalance_needed(self, new_allocations: Dict[str, float]) -> bool:
        """Check if rebalancing is needed"""
        try:
            for strategy_name, new_allocation in new_allocations.items():
                if strategy_name in self.strategy_allocations:
                    current_allocation = self.strategy_allocations[strategy_name].current_allocation
                    change = abs(new_allocation - current_allocation)
                    
                    if change >= self.rebalance_threshold:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Rebalance check failed: {e}")
            return False
    
    def get_strategy_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive strategy performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_capital': self.total_capital,
                'strategies': {},
                'portfolio_metrics': {},
                'allocation_summary': {}
            }
            
            # Strategy details
            for strategy_name, allocation in self.strategy_allocations.items():
                metrics = self.strategy_metrics.get(strategy_name)
                
                strategy_info = {
                    'allocation': {
                        'current': allocation.current_allocation,
                        'target': allocation.target_allocation,
                        'min': allocation.min_allocation,
                        'max': allocation.max_allocation,
                        'capital': allocation.current_allocation * self.total_capital
                    },
                    'status': allocation.status.value,
                    'last_rebalance': allocation.last_rebalance.isoformat()
                }
                
                if metrics:
                    strategy_info['performance'] = {
                        'total_trades': metrics.total_trades,
                        'win_rate': metrics.win_rate,
                        'total_pnl': metrics.total_pnl,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'sortino_ratio': metrics.sortino_ratio,
                        'max_drawdown': metrics.max_drawdown,
                        'current_drawdown': metrics.current_drawdown,
                        'volatility': metrics.volatility,
                        'profit_factor': metrics.profit_factor,
                        'var_95': metrics.var_95
                    }
                
                report['strategies'][strategy_name] = strategy_info
            
            # Portfolio metrics
            if self.strategy_metrics:
                portfolio_metrics = self._calculate_portfolio_metrics()
                report['portfolio_metrics'] = portfolio_metrics
            
            # Allocation summary
            total_allocated = sum(alloc.current_allocation for alloc in self.strategy_allocations.values())
            report['allocation_summary'] = {
                'total_allocated': total_allocated,
                'unallocated': 1.0 - total_allocated,
                'num_strategies': len(self.strategy_allocations),
                'active_strategies': len([a for a in self.strategy_allocations.values() if a.status == StrategyStatus.ACTIVE])
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Performance report generation failed: {e}")
            return {}
    
    def _calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio-level metrics"""
        try:
            if not self.strategy_metrics:
                return {}
            
            # Weighted portfolio metrics
            total_pnl = 0.0
            total_volatility = 0.0
            total_sharpe = 0.0
            total_weight = 0.0
            
            for strategy_name, metrics in self.strategy_metrics.items():
                if strategy_name in self.strategy_allocations:
                    weight = self.strategy_allocations[strategy_name].current_allocation
                    total_pnl += metrics.total_pnl * weight
                    total_volatility += metrics.volatility * weight
                    total_sharpe += metrics.sharpe_ratio * weight
                    total_weight += weight
            
            if total_weight > 0:
                return {
                    'weighted_pnl': total_pnl,
                    'weighted_volatility': total_volatility,
                    'weighted_sharpe': total_sharpe,
                    'total_weight': total_weight
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Portfolio metrics calculation failed: {e}")
            return {}
    
    def disable_strategy(self, strategy_name: str, reason: str = "Manual disable"):
        """Disable strategy"""
        if strategy_name in self.strategy_allocations:
            self.strategy_allocations[strategy_name].status = StrategyStatus.DISABLED
            logger.info(f"🚫 Strategy disabled: {strategy_name} - {reason}")
    
    def enable_strategy(self, strategy_name: str):
        """Enable strategy"""
        if strategy_name in self.strategy_allocations:
            self.strategy_allocations[strategy_name].status = StrategyStatus.ACTIVE
            logger.info(f"✅ Strategy enabled: {strategy_name}")

def main():
    """Main function for testing"""
    # Create optimizer
    optimizer = CapitalEfficiencyOptimizer(total_capital=100000)
    
    # Add strategies
    optimizer.add_strategy("EMA_Strategy", initial_allocation=0.30)
    optimizer.add_strategy("Supertrend_Strategy", initial_allocation=0.25)
    optimizer.add_strategy("MACD_Strategy", initial_allocation=0.20)
    optimizer.add_strategy("RSI_Strategy", initial_allocation=0.25)
    
    # Generate mock trade data
    strategies = ["EMA_Strategy", "Supertrend_Strategy", "MACD_Strategy", "RSI_Strategy"]
    
    for strategy in strategies:
        # Generate 50 trades with different performance characteristics
        trades = []
        for i in range(50):
            if strategy == "EMA_Strategy":
                # Good performance
                pnl = np.random.normal(100, 200)
            elif strategy == "Supertrend_Strategy":
                # Moderate performance
                pnl = np.random.normal(50, 150)
            elif strategy == "MACD_Strategy":
                # Poor performance
                pnl = np.random.normal(-20, 100)
            else:  # RSI_Strategy
                # Very good performance
                pnl = np.random.normal(150, 180)
            
            trades.append({'pnl': pnl})
        
        # Update metrics
        optimizer.update_strategy_metrics(strategy, trades)
    
    # Optimize allocations
    new_allocations = optimizer.optimize_allocations()
    print(f"🔄 New allocations: {new_allocations}")
    
    # Get performance report
    report = optimizer.get_strategy_performance_report()
    print(f"📊 Performance report: {json.dumps(report, indent=2, default=str)}")

if __name__ == "__main__":
    main()
