#!/usr/bin/env python3
"""
AI-Driven Trade Review System
Generates daily trade reports in plain English with ML analytics
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
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeMetrics:
    """Trade performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    avg_trade_duration: float
    volatility: float

@dataclass
class StrategyPerformance:
    """Strategy-specific performance metrics"""
    strategy_name: str
    trades_count: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    avg_confidence: float
    execution_rate: float

@dataclass
class RiskExposure:
    """Risk exposure analysis"""
    total_exposure: float
    max_single_position: float
    correlation_risk: float
    volatility_risk: float
    concentration_risk: float
    sector_exposure: Dict[str, float]

class AITradeReview:
    """AI-driven trade review and analysis system"""
    
    def __init__(self):
        self.database = None
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize components"""
        try:
            from src.models.enhanced_database import EnhancedTradingDatabase
            self.database = EnhancedTradingDatabase("data/enhanced_trading.db")
            logger.info("✅ AI Trade Review system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI Trade Review: {e}")
            raise
    
    def generate_daily_report(self, date: str = None) -> Dict[str, Any]:
        """Generate comprehensive daily trade report"""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"📊 Generating AI trade report for {date}")
        
        try:
            # Get daily data
            daily_summary = self.database.get_daily_summary(date, "indian")
            entry_signals = self.database.get_entry_signals("indian", limit=100)
            exit_signals = self.database.get_exit_signals("indian", limit=100)
            
            # Calculate metrics
            trade_metrics = self._calculate_trade_metrics(entry_signals, exit_signals)
            strategy_performance = self._analyze_strategy_performance(entry_signals, exit_signals)
            risk_exposure = self._analyze_risk_exposure(entry_signals)
            market_conditions = self._analyze_market_conditions(entry_signals)
            
            # Generate AI insights
            ai_insights = self._generate_ai_insights(trade_metrics, strategy_performance, risk_exposure, market_conditions)
            
            # Create plain English report
            report = self._create_plain_english_report(
                date, trade_metrics, strategy_performance, risk_exposure, market_conditions, ai_insights
            )
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating daily report: {e}")
            return {}
    
    def _calculate_trade_metrics(self, entry_signals: List[Dict], exit_signals: List[Dict]) -> TradeMetrics:
        """Calculate comprehensive trade metrics"""
        try:
            # Basic metrics
            total_trades = len(entry_signals)
            winning_trades = sum(1 for signal in exit_signals if signal.get('pnl', 0) > 0)
            losing_trades = sum(1 for signal in exit_signals if signal.get('pnl', 0) < 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # P&L metrics
            pnls = [signal.get('pnl', 0) for signal in exit_signals if signal.get('pnl') is not None]
            total_pnl = sum(pnls)
            
            wins = [pnl for pnl in pnls if pnl > 0]
            losses = [pnl for pnl in pnls if pnl < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            max_win = max(wins) if wins else 0
            max_loss = min(losses) if losses else 0
            
            # Risk metrics
            sharpe_ratio = self._calculate_sharpe_ratio(pnls)
            max_drawdown = self._calculate_max_drawdown(pnls)
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
            
            # Duration metrics
            durations = [signal.get('duration_minutes', 0) for signal in exit_signals if signal.get('duration_minutes')]
            avg_trade_duration = np.mean(durations) if durations else 0
            
            # Volatility
            volatility = np.std(pnls) if len(pnls) > 1 else 0
            
            return TradeMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_win=max_win,
                max_loss=max_loss,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                profit_factor=profit_factor,
                avg_trade_duration=avg_trade_duration,
                volatility=volatility
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating trade metrics: {e}")
            return TradeMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _analyze_strategy_performance(self, entry_signals: List[Dict], exit_signals: List[Dict]) -> List[StrategyPerformance]:
        """Analyze performance by strategy"""
        try:
            strategy_stats = {}
            
            # Group signals by strategy
            for signal in entry_signals:
                strategy = signal.get('strategy', 'unknown')
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        'trades': [],
                        'confidences': [],
                        'executed': 0
                    }
                
                strategy_stats[strategy]['trades'].append(signal)
                strategy_stats[strategy]['confidences'].append(signal.get('confidence', 0))
                if signal.get('executed', False):
                    strategy_stats[strategy]['executed'] += 1
            
            # Calculate performance for each strategy
            strategy_performance = []
            for strategy, stats in strategy_stats.items():
                trades_count = len(stats['trades'])
                executed_count = stats['executed']
                execution_rate = (executed_count / trades_count * 100) if trades_count > 0 else 0
                avg_confidence = np.mean(stats['confidences']) if stats['confidences'] else 0
                
                # Get P&L for this strategy
                strategy_exits = [s for s in exit_signals if s.get('strategy') == strategy]
                strategy_pnls = [s.get('pnl', 0) for s in strategy_exits if s.get('pnl') is not None]
                
                win_rate = (sum(1 for pnl in strategy_pnls if pnl > 0) / len(strategy_pnls) * 100) if strategy_pnls else 0
                total_pnl = sum(strategy_pnls)
                sharpe_ratio = self._calculate_sharpe_ratio(strategy_pnls)
                max_drawdown = self._calculate_max_drawdown(strategy_pnls)
                
                strategy_performance.append(StrategyPerformance(
                    strategy_name=strategy,
                    trades_count=trades_count,
                    win_rate=win_rate,
                    total_pnl=total_pnl,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    avg_confidence=avg_confidence,
                    execution_rate=execution_rate
                ))
            
            return strategy_performance
            
        except Exception as e:
            logger.error(f"❌ Error analyzing strategy performance: {e}")
            return []
    
    def _analyze_risk_exposure(self, entry_signals: List[Dict]) -> RiskExposure:
        """Analyze risk exposure across portfolio"""
        try:
            # Calculate total exposure
            total_exposure = sum(signal.get('position_size', 0) for signal in entry_signals)
            
            # Max single position
            position_sizes = [signal.get('position_size', 0) for signal in entry_signals]
            max_single_position = max(position_sizes) if position_sizes else 0
            
            # Correlation risk (simplified - based on symbol diversity)
            symbols = [signal.get('symbol') for signal in entry_signals]
            unique_symbols = len(set(symbols))
            correlation_risk = 1 - (unique_symbols / len(symbols)) if symbols else 0
            
            # Volatility risk
            volatilities = [signal.get('volatility', 0.02) for signal in entry_signals]
            volatility_risk = np.mean(volatilities) if volatilities else 0.02
            
            # Concentration risk
            symbol_counts = {}
            for symbol in symbols:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            max_concentration = max(symbol_counts.values()) if symbol_counts else 0
            concentration_risk = max_concentration / len(symbols) if symbols else 0
            
            # Sector exposure (simplified)
            sector_exposure = {
                'NIFTY': sum(1 for s in symbols if 'NIFTY' in s) / len(symbols) if symbols else 0,
                'BANK': sum(1 for s in symbols if 'BANK' in s) / len(symbols) if symbols else 0,
                'FIN': sum(1 for s in symbols if 'FIN' in s) / len(symbols) if symbols else 0
            }
            
            return RiskExposure(
                total_exposure=total_exposure,
                max_single_position=max_single_position,
                correlation_risk=correlation_risk,
                volatility_risk=volatility_risk,
                concentration_risk=concentration_risk,
                sector_exposure=sector_exposure
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing risk exposure: {e}")
            return RiskExposure(0, 0, 0, 0, 0, {})
    
    def _analyze_market_conditions(self, entry_signals: List[Dict]) -> Dict[str, Any]:
        """Analyze market conditions during trading"""
        try:
            # Market condition distribution
            conditions = [signal.get('market_condition', 'UNKNOWN') for signal in entry_signals]
            condition_counts = {}
            for condition in conditions:
                condition_counts[condition] = condition_counts.get(condition, 0) + 1
            
            # Signal type distribution
            signal_types = [signal.get('signal_type', 'UNKNOWN') for signal in entry_signals]
            buy_signals = sum(1 for st in signal_types if 'BUY' in st)
            sell_signals = sum(1 for st in signal_types if 'SELL' in st)
            
            # Confidence distribution
            confidences = [signal.get('confidence', 0) for signal in entry_signals]
            avg_confidence = np.mean(confidences) if confidences else 0
            confidence_std = np.std(confidences) if len(confidences) > 1 else 0
            
            return {
                'market_conditions': condition_counts,
                'signal_distribution': {'buy': buy_signals, 'sell': sell_signals},
                'avg_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'total_signals': len(entry_signals)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing market conditions: {e}")
            return {}
    
    def _generate_ai_insights(self, trade_metrics: TradeMetrics, strategy_performance: List[StrategyPerformance], 
                            risk_exposure: RiskExposure, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights and recommendations"""
        try:
            insights = {
                'performance_insights': [],
                'risk_insights': [],
                'strategy_insights': [],
                'recommendations': []
            }
            
            # Performance insights
            if trade_metrics.win_rate > 60:
                insights['performance_insights'].append("🎯 Excellent win rate! Your strategy is performing well above average.")
            elif trade_metrics.win_rate > 50:
                insights['performance_insights'].append("✅ Good win rate. Room for improvement but solid performance.")
            else:
                insights['performance_insights'].append("⚠️ Win rate below 50%. Consider reviewing your entry criteria.")
            
            if trade_metrics.sharpe_ratio > 1.5:
                insights['performance_insights'].append("📈 Outstanding risk-adjusted returns! Your strategy is highly efficient.")
            elif trade_metrics.sharpe_ratio > 1.0:
                insights['performance_insights'].append("📊 Good risk-adjusted returns. Strategy is performing well.")
            else:
                insights['performance_insights'].append("📉 Risk-adjusted returns need improvement. Consider reducing volatility.")
            
            # Risk insights
            if risk_exposure.concentration_risk > 0.5:
                insights['risk_insights'].append("🚨 High concentration risk! Too many trades in single symbols.")
            elif risk_exposure.concentration_risk > 0.3:
                insights['risk_insights'].append("⚠️ Moderate concentration risk. Consider diversifying more.")
            else:
                insights['risk_insights'].append("✅ Good diversification. Risk is well spread across symbols.")
            
            if risk_exposure.volatility_risk > 0.05:
                insights['risk_insights'].append("📊 High volatility environment. Consider reducing position sizes.")
            else:
                insights['risk_insights'].append("✅ Low volatility environment. Normal position sizing appropriate.")
            
            # Strategy insights
            best_strategy = max(strategy_performance, key=lambda x: x.total_pnl) if strategy_performance else None
            worst_strategy = min(strategy_performance, key=lambda x: x.total_pnl) if strategy_performance else None
            
            if best_strategy:
                insights['strategy_insights'].append(f"🏆 Best performing strategy: {best_strategy.strategy_name} (P&L: ₹{best_strategy.total_pnl:,.2f})")
            
            if worst_strategy and worst_strategy.total_pnl < 0:
                insights['strategy_insights'].append(f"📉 Strategy needs review: {worst_strategy.strategy_name} (P&L: ₹{worst_strategy.total_pnl:,.2f})")
            
            # Recommendations
            if trade_metrics.max_drawdown > 0.1:
                insights['recommendations'].append("🛡️ Implement stricter stop-losses to reduce maximum drawdown.")
            
            if risk_exposure.correlation_risk > 0.7:
                insights['recommendations'].append("🔄 Add more diverse symbols to reduce correlation risk.")
            
            if market_conditions.get('avg_confidence', 0) < 60:
                insights['recommendations'].append("🎯 Improve signal quality by raising confidence thresholds.")
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generating AI insights: {e}")
            return {'performance_insights': [], 'risk_insights': [], 'strategy_insights': [], 'recommendations': []}
    
    def _create_plain_english_report(self, date: str, trade_metrics: TradeMetrics, 
                                   strategy_performance: List[StrategyPerformance], 
                                   risk_exposure: RiskExposure, market_conditions: Dict[str, Any],
                                   ai_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create plain English daily report"""
        
        report = {
            'date': date,
            'summary': f"Daily Trading Report for {date}",
            'executive_summary': self._create_executive_summary(trade_metrics, strategy_performance),
            'performance_analysis': self._create_performance_analysis(trade_metrics),
            'strategy_breakdown': self._create_strategy_breakdown(strategy_performance),
            'risk_assessment': self._create_risk_assessment(risk_exposure),
            'market_conditions': self._create_market_conditions_analysis(market_conditions),
            'ai_insights': ai_insights,
            'recommendations': self._create_recommendations(ai_insights),
            'next_day_focus': self._create_next_day_focus(trade_metrics, strategy_performance, risk_exposure)
        }
        
        return report
    
    def _create_executive_summary(self, trade_metrics: TradeMetrics, strategy_performance: List[StrategyPerformance]) -> str:
        """Create executive summary"""
        total_pnl = trade_metrics.total_pnl
        win_rate = trade_metrics.win_rate
        total_trades = trade_metrics.total_trades
        
        if total_pnl > 0:
            pnl_status = f"profitable with ₹{total_pnl:,.2f} gains"
        else:
            pnl_status = f"loss of ₹{abs(total_pnl):,.2f}"
        
        return f"Today's trading session was {pnl_status} across {total_trades} trades with a {win_rate:.1f}% win rate. " \
               f"The best performing strategy was {max(strategy_performance, key=lambda x: x.total_pnl).strategy_name if strategy_performance else 'N/A'}."
    
    def _create_performance_analysis(self, trade_metrics: TradeMetrics) -> str:
        """Create performance analysis"""
        return f"Performance Analysis: Average win was ₹{trade_metrics.avg_win:,.2f} while average loss was ₹{abs(trade_metrics.avg_loss):,.2f}. " \
               f"Maximum drawdown was {trade_metrics.max_drawdown*100:.1f}% and Sharpe ratio was {trade_metrics.sharpe_ratio:.2f}. " \
               f"Average trade duration was {trade_metrics.avg_trade_duration:.1f} minutes."
    
    def _create_strategy_breakdown(self, strategy_performance: List[StrategyPerformance]) -> str:
        """Create strategy breakdown"""
        if not strategy_performance:
            return "No strategy performance data available."
        
        breakdown = "Strategy Performance: "
        for strategy in strategy_performance:
            breakdown += f"{strategy.strategy_name} generated ₹{strategy.total_pnl:,.2f} with {strategy.win_rate:.1f}% win rate. "
        
        return breakdown
    
    def _create_risk_assessment(self, risk_exposure: RiskExposure) -> str:
        """Create risk assessment"""
        return f"Risk Assessment: Total exposure was ₹{risk_exposure.total_exposure:,.2f} with maximum single position of ₹{risk_exposure.max_single_position:,.2f}. " \
               f"Concentration risk was {risk_exposure.concentration_risk*100:.1f}% and correlation risk was {risk_exposure.correlation_risk*100:.1f}%."
    
    def _create_market_conditions_analysis(self, market_conditions: Dict[str, Any]) -> str:
        """Create market conditions analysis"""
        if not market_conditions:
            return "No market conditions data available."
        
        signal_dist = market_conditions.get('signal_distribution', {})
        avg_confidence = market_conditions.get('avg_confidence', 0)
        
        return f"Market Conditions: {signal_dist.get('buy', 0)} buy signals vs {signal_dist.get('sell', 0)} sell signals. " \
               f"Average signal confidence was {avg_confidence:.1f}%."
    
    def _create_recommendations(self, ai_insights: Dict[str, Any]) -> str:
        """Create recommendations"""
        recommendations = ai_insights.get('recommendations', [])
        if not recommendations:
            return "No specific recommendations for today. Continue current strategy."
        
        return "Recommendations: " + " ".join(recommendations)
    
    def _create_next_day_focus(self, trade_metrics: TradeMetrics, strategy_performance: List[StrategyPerformance], 
                              risk_exposure: RiskExposure) -> str:
        """Create next day focus"""
        if trade_metrics.win_rate < 50:
            return "Focus on improving signal quality and entry criteria."
        elif risk_exposure.concentration_risk > 0.5:
            return "Focus on diversifying positions across more symbols."
        elif trade_metrics.max_drawdown > 0.1:
            return "Focus on implementing stricter risk management."
        else:
            return "Continue current strategy with minor optimizations."
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for simplicity
        return mean_return / std_return
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(min(drawdown)) if len(drawdown) > 0 else 0.0
    
    def display_report(self, report: Dict[str, Any]):
        """Display the AI trade report"""
        print("\n" + "="*80)
        print(f"🤖 AI-DRIVEN DAILY TRADE REPORT - {report['date']}")
        print("="*80)
        
        print(f"\n📊 EXECUTIVE SUMMARY")
        print("-" * 40)
        print(report['executive_summary'])
        
        print(f"\n📈 PERFORMANCE ANALYSIS")
        print("-" * 40)
        print(report['performance_analysis'])
        
        print(f"\n🎯 STRATEGY BREAKDOWN")
        print("-" * 40)
        print(report['strategy_breakdown'])
        
        print(f"\n��️ RISK ASSESSMENT")
        print("-" * 40)
        print(report['risk_assessment'])
        
        print(f"\n🌊 MARKET CONDITIONS")
        print("-" * 40)
        print(report['market_conditions'])
        
        print(f"\n🧠 AI INSIGHTS")
        print("-" * 40)
        for category, insights in report['ai_insights'].items():
            if insights:
                print(f"\n{category.replace('_', ' ').title()}:")
                for insight in insights:
                    print(f"  {insight}")
        
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 40)
        print(report['recommendations'])
        
        print(f"\n🎯 NEXT DAY FOCUS")
        print("-" * 40)
        print(report['next_day_focus'])
        
        print("\n" + "="*80)
        print(f"📊 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

def main():
    """Main function"""
    ai_review = AITradeReview()
    
    try:
        # Generate today's report
        report = ai_review.generate_daily_report()
        
        if report:
            ai_review.display_report(report)
        else:
            print("❌ No data available for report generation")
        
    except Exception as e:
        logger.error(f"❌ AI Trade Review error: {e}")

if __name__ == "__main__":
    main()
