#!/usr/bin/env python3
"""
Advanced Analytics and Reporting System
Comprehensive analytics, reporting, and business intelligence
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int

@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float
    treynor_ratio: float

@dataclass
class AttributionAnalysis:
    """Performance attribution analysis"""
    total_attribution: float
    strategy_attribution: Dict[str, float]
    market_attribution: float
    timing_attribution: float
    selection_attribution: float
    interaction_attribution: float

class AdvancedAnalyticsReporting:
    """Advanced analytics and reporting system"""
    
    def __init__(self):
        self.performance_data = {}
        self.risk_data = {}
        self.attribution_data = {}
        
    def calculate_performance_metrics(self, returns: pd.Series, 
                                    benchmark_returns: pd.Series = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + returns.mean()) ** 252 - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252)
            sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade statistics
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            
            win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0
            avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
            
            # Profit factor
            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                total_trades=len(returns),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades)
            )
            
            logger.info(f"✅ Performance metrics calculated - Sharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Performance metrics calculation failed: {e}")
            return None
    
    def calculate_risk_metrics(self, returns: pd.Series, 
                             benchmark_returns: pd.Series = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Beta and correlation
            beta = 0
            correlation = 0
            tracking_error = 0
            information_ratio = 0
            treynor_ratio = 0
            
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                # Align returns
                aligned_returns = returns.align(benchmark_returns, join='inner')
                returns_aligned = aligned_returns[0]
                benchmark_aligned = aligned_returns[1]
                
                # Calculate beta
                covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
                benchmark_variance = np.var(benchmark_aligned)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Correlation
                correlation = np.corrcoef(returns_aligned, benchmark_aligned)[0, 1]
                
                # Tracking error
                excess_returns = returns_aligned - benchmark_aligned
                tracking_error = excess_returns.std() * np.sqrt(252)
                
                # Information ratio
                information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0
                
                # Treynor ratio
                risk_free_rate = 0.02  # 2% risk-free rate
                treynor_ratio = (returns.mean() * 252 - risk_free_rate) / beta if beta != 0 else 0
            
            metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                beta=beta,
                correlation=correlation,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                treynor_ratio=treynor_ratio
            )
            
            logger.info(f"✅ Risk metrics calculated - VaR 95%: {var_95:.2%}, Beta: {beta:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Risk metrics calculation failed: {e}")
            return None
    
    def perform_attribution_analysis(self, strategy_returns: Dict[str, pd.Series], 
                                   benchmark_returns: pd.Series) -> AttributionAnalysis:
        """Perform performance attribution analysis"""
        try:
            # Calculate total portfolio returns
            total_returns = pd.Series(0, index=benchmark_returns.index)
            strategy_attribution = {}
            
            for strategy_name, returns in strategy_returns.items():
                # Align returns
                aligned_returns = returns.align(benchmark_returns, join='inner')
                strategy_aligned = aligned_returns[0]
                benchmark_aligned = aligned_returns[1]
                
                # Calculate strategy attribution
                strategy_attribution[strategy_name] = strategy_aligned.mean() * 252
                total_returns += strategy_aligned
            
            # Calculate total attribution
            total_attribution = total_returns.mean() * 252
            
            # Market attribution (beta-adjusted)
            market_attribution = benchmark_returns.mean() * 252
            
            # Timing attribution (interaction between strategy and market)
            timing_attribution = 0.0  # Simplified calculation
            
            # Selection attribution (stock picking)
            selection_attribution = total_attribution - market_attribution - timing_attribution
            
            # Interaction attribution
            interaction_attribution = 0.0  # Simplified calculation
            
            attribution = AttributionAnalysis(
                total_attribution=total_attribution,
                strategy_attribution=strategy_attribution,
                market_attribution=market_attribution,
                timing_attribution=timing_attribution,
                selection_attribution=selection_attribution,
                interaction_attribution=interaction_attribution
            )
            
            logger.info(f"✅ Attribution analysis completed - Total: {total_attribution:.2%}")
            
            return attribution
            
        except Exception as e:
            logger.error(f"❌ Attribution analysis failed: {e}")
            return None
    
    def detect_market_regimes(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regimes using advanced techniques"""
        try:
            # Calculate regime indicators
            price_data['returns'] = price_data['close'].pct_change()
            price_data['volatility'] = price_data['returns'].rolling(20).std()
            price_data['momentum'] = price_data['close'].pct_change(20)
            price_data['trend'] = price_data['close'].rolling(50).mean()
            
            # Regime detection using multiple indicators
            current_volatility = price_data['volatility'].iloc[-1]
            current_momentum = price_data['momentum'].iloc[-1]
            current_trend = price_data['trend'].iloc[-1]
            current_price = price_data['close'].iloc[-1]
            
            # Regime classification
            if current_volatility > price_data['volatility'].quantile(0.8):
                regime = 'HIGH_VOLATILITY'
            elif current_volatility < price_data['volatility'].quantile(0.2):
                regime = 'LOW_VOLATILITY'
            elif abs(current_momentum) > 0.05:
                regime = 'TRENDING'
            elif abs(current_price - current_trend) / current_trend < 0.02:
                regime = 'SIDEWAYS'
            else:
                regime = 'TRANSITIONAL'
            
            # Calculate regime probabilities
            regime_probabilities = {
                'HIGH_VOLATILITY': 0.2,
                'LOW_VOLATILITY': 0.2,
                'TRENDING': 0.3,
                'SIDEWAYS': 0.2,
                'TRANSITIONAL': 0.1
            }
            
            # Adjust probabilities based on current indicators
            if regime == 'HIGH_VOLATILITY':
                regime_probabilities['HIGH_VOLATILITY'] = 0.6
            elif regime == 'TRENDING':
                regime_probabilities['TRENDING'] = 0.5
            
            # Normalize probabilities
            total_prob = sum(regime_probabilities.values())
            regime_probabilities = {k: v/total_prob for k, v in regime_probabilities.items()}
            
            result = {
                'current_regime': regime,
                'regime_probabilities': regime_probabilities,
                'indicators': {
                    'volatility': current_volatility,
                    'momentum': current_momentum,
                    'trend_deviation': (current_price - current_trend) / current_trend
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Market regime detected: {regime}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {e}")
            return {}
    
    def generate_performance_report(self, returns: pd.Series, 
                                  benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Calculate metrics
            performance_metrics = self.calculate_performance_metrics(returns, benchmark_returns)
            risk_metrics = self.calculate_risk_metrics(returns, benchmark_returns)
            
            if not performance_metrics or not risk_metrics:
                return {'error': 'Failed to calculate metrics'}
            
            # Generate report
            report = {
                'report_date': datetime.now().isoformat(),
                'performance_metrics': {
                    'total_return': performance_metrics.total_return,
                    'annualized_return': performance_metrics.annualized_return,
                    'volatility': performance_metrics.volatility,
                    'sharpe_ratio': performance_metrics.sharpe_ratio,
                    'sortino_ratio': performance_metrics.sortino_ratio,
                    'max_drawdown': performance_metrics.max_drawdown,
                    'calmar_ratio': performance_metrics.calmar_ratio,
                    'win_rate': performance_metrics.win_rate,
                    'profit_factor': performance_metrics.profit_factor,
                    'total_trades': performance_metrics.total_trades
                },
                'risk_metrics': {
                    'var_95': risk_metrics.var_95,
                    'var_99': risk_metrics.var_99,
                    'cvar_95': risk_metrics.cvar_95,
                    'cvar_99': risk_metrics.cvar_99,
                    'beta': risk_metrics.beta,
                    'correlation': risk_metrics.correlation,
                    'tracking_error': risk_metrics.tracking_error,
                    'information_ratio': risk_metrics.information_ratio,
                    'treynor_ratio': risk_metrics.treynor_ratio
                },
                'summary': {
                    'risk_adjusted_return': performance_metrics.sharpe_ratio,
                    'downside_protection': risk_metrics.var_95,
                    'consistency': performance_metrics.win_rate,
                    'efficiency': performance_metrics.calmar_ratio
                }
            }
            
            logger.info("✅ Performance report generated")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Performance report generation failed: {e}")
            return {'error': str(e)}
    
    def create_visualizations(self, data: pd.DataFrame, save_path: str = None) -> Dict[str, str]:
        """Create performance visualizations"""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            visualizations = {}
            
            # 1. Equity curve
            plt.figure(figsize=(12, 6))
            cumulative_returns = (1 + data['returns']).cumprod()
            plt.plot(cumulative_returns.index, cumulative_returns.values)
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.grid(True)
            
            if save_path:
                equity_path = f"{save_path}/equity_curve.png"
                plt.savefig(equity_path, dpi=300, bbox_inches='tight')
                visualizations['equity_curve'] = equity_path
            
            plt.close()
            
            # 2. Drawdown chart
            plt.figure(figsize=(12, 6))
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            plt.title('Drawdown Chart')
            plt.xlabel('Date')
            plt.ylabel('Drawdown')
            plt.grid(True)
            
            if save_path:
                drawdown_path = f"{save_path}/drawdown_chart.png"
                plt.savefig(drawdown_path, dpi=300, bbox_inches='tight')
                visualizations['drawdown_chart'] = drawdown_path
            
            plt.close()
            
            # 3. Returns distribution
            plt.figure(figsize=(10, 6))
            plt.hist(data['returns'], bins=50, alpha=0.7, edgecolor='black')
            plt.title('Returns Distribution')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            if save_path:
                returns_path = f"{save_path}/returns_distribution.png"
                plt.savefig(returns_path, dpi=300, bbox_inches='tight')
                visualizations['returns_distribution'] = returns_path
            
            plt.close()
            
            # 4. Rolling Sharpe ratio
            plt.figure(figsize=(12, 6))
            rolling_sharpe = data['returns'].rolling(252).mean() / data['returns'].rolling(252).std() * np.sqrt(252)
            plt.plot(rolling_sharpe.index, rolling_sharpe.values)
            plt.title('Rolling Sharpe Ratio (252 days)')
            plt.xlabel('Date')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True)
            
            if save_path:
                sharpe_path = f"{save_path}/rolling_sharpe.png"
                plt.savefig(sharpe_path, dpi=300, bbox_inches='tight')
                visualizations['rolling_sharpe'] = sharpe_path
            
            plt.close()
            
            logger.info(f"✅ Created {len(visualizations)} visualizations")
            
            return visualizations
            
        except Exception as e:
            logger.error(f"❌ Visualization creation failed: {e}")
            return {}
    
    def run_comprehensive_analysis(self, data: pd.DataFrame, 
                                 benchmark_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Run comprehensive analytics analysis"""
        try:
            logger.info("🚀 Starting comprehensive analytics analysis...")
            
            # Prepare data
            returns = data['close'].pct_change().dropna()
            benchmark_returns = benchmark_data['close'].pct_change().dropna() if benchmark_data is not None else None
            
            # Generate performance report
            performance_report = self.generate_performance_report(returns, benchmark_returns)
            
            # Detect market regimes
            market_regimes = self.detect_market_regimes(data)
            
            # Create visualizations
            visualizations = self.create_visualizations(data)
            
            # Compile results
            results = {
                'analysis_date': datetime.now().isoformat(),
                'performance_report': performance_report,
                'market_regimes': market_regimes,
                'visualizations': visualizations,
                'data_summary': {
                    'total_observations': len(data),
                    'date_range': {
                        'start': data.index[0].isoformat(),
                        'end': data.index[-1].isoformat()
                    },
                    'data_quality': {
                        'missing_values': data.isnull().sum().sum(),
                        'duplicate_values': data.duplicated().sum()
                    }
                }
            }
            
            logger.info("✅ Comprehensive analytics analysis completed")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis failed: {e}")
            return {'error': str(e)}

def main():
    """Run advanced analytics and reporting"""
    analytics = AdvancedAnalyticsReporting()
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    np.random.seed(42)
    
    # Generate price data with trend and volatility
    returns = np.random.normal(0.0005, 0.02, 1000)  # 0.05% daily return, 2% volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'returns': returns
    }, index=dates)
    
    # Generate benchmark data
    benchmark_returns = np.random.normal(0.0003, 0.015, 1000)  # 0.03% daily return, 1.5% volatility
    benchmark_prices = 100 * np.exp(np.cumsum(benchmark_returns))
    
    benchmark_data = pd.DataFrame({
        'close': benchmark_prices,
        'returns': benchmark_returns
    }, index=dates)
    
    # Run analysis
    results = analytics.run_comprehensive_analysis(data, benchmark_data)
    
    print("\n" + "="*80)
    print("📊 ADVANCED ANALYTICS AND REPORTING RESULTS")
    print("="*80)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    
    print(f"\n📈 PERFORMANCE METRICS:")
    perf_metrics = results['performance_report']['performance_metrics']
    for metric, value in perf_metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    print(f"\n⚠️ RISK METRICS:")
    risk_metrics = results['performance_report']['risk_metrics']
    for metric, value in risk_metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    print(f"\n🎯 MARKET REGIMES:")
    regimes = results['market_regimes']
    print(f"   Current Regime: {regimes['current_regime']}")
    print(f"   Regime Probabilities:")
    for regime, prob in regimes['regime_probabilities'].items():
        print(f"     {regime}: {prob:.1%}")
    
    print(f"\n📊 DATA SUMMARY:")
    summary = results['data_summary']
    print(f"   Total Observations: {summary['total_observations']}")
    print(f"   Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"   Missing Values: {summary['data_quality']['missing_values']}")
    print(f"   Duplicate Values: {summary['data_quality']['duplicate_values']}")
    
    print(f"\n📈 VISUALIZATIONS CREATED: {len(results['visualizations'])}")
    for viz_name, path in results['visualizations'].items():
        print(f"   {viz_name}: {path}")
    
    print("\n" + "="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
