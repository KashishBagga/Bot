#!/usr/bin/env python3
"""
PnL Attribution Engine
Break down PnL per signal, strategy, symbol for optimization insights
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PnLAttribution:
    """PnL attribution for a specific dimension"""
    dimension: str  # 'strategy', 'symbol', 'signal_type', 'time_period'
    value: str
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_pnl: float
    max_pnl: float
    min_pnl: float
    total_volume: float
    sharpe_ratio: float
    max_drawdown: float

class PnLAttributionEngine:
    """Engine for PnL attribution and analysis"""
    
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.attribution_data = []
        self.daily_attributions = defaultdict(list)
        self.performance_metrics = {}
        
    def analyze_pnl_attribution(self, trades: List, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """Analyze PnL attribution across multiple dimensions"""
        try:
            logger.info("📊 Analyzing PnL attribution...")
            
            # Filter trades by date range if provided
            if start_date or end_date:
                filtered_trades = self._filter_trades_by_date(trades, start_date, end_date)
            else:
                filtered_trades = trades
            
            if not filtered_trades:
                logger.warning("⚠️ No trades found for attribution analysis")
                return {}
            
            # Analyze by different dimensions
            attribution_results = {
                'by_strategy': self._analyze_by_strategy(filtered_trades),
                'by_symbol': self._analyze_by_symbol(filtered_trades),
                'by_signal_type': self._analyze_by_signal_type(filtered_trades),
                'by_time_period': self._analyze_by_time_period(filtered_trades),
                'by_confidence': self._analyze_by_confidence(filtered_trades),
                'correlation_analysis': self._analyze_correlations(filtered_trades),
                'performance_insights': self._generate_performance_insights(filtered_trades)
            }
            
            # Store attribution data
            self.attribution_data = attribution_results
            
            # Generate summary report
            self._generate_attribution_report(attribution_results)
            
            return attribution_results
            
        except Exception as e:
            logger.error(f"❌ Error analyzing PnL attribution: {e}")
            return {}
    
    def _filter_trades_by_date(self, trades: List, start_date: datetime, end_date: datetime) -> List:
        """Filter trades by date range"""
        filtered = []
        for trade in trades:
            trade_date = trade.exit_time if hasattr(trade, 'exit_time') and trade.exit_time else trade.timestamp
            if start_date and trade_date < start_date:
                continue
            if end_date and trade_date > end_date:
                continue
            filtered.append(trade)
        return filtered
    
    def _analyze_by_strategy(self, trades: List) -> Dict[str, PnLAttribution]:
        """Analyze PnL by trading strategy"""
        strategy_stats = defaultdict(lambda: {
            'pnl': 0.0, 'trades': 0, 'wins': 0, 'losses': 0, 'volumes': 0.0,
            'pnls': [], 'volumes': []
        })
        
        for trade in trades:
            strategy = trade.strategy
            pnl = trade.pnl or 0.0
            volume = trade.entry_price * trade.quantity
            
            strategy_stats[strategy]['pnl'] += pnl
            strategy_stats[strategy]['trades'] += 1
            strategy_stats[strategy]['volumes'] += volume
            strategy_stats[strategy]['pnls'].append(pnl)
            strategy_stats[strategy]['volumes'].append(volume)
            
            if pnl > 0:
                strategy_stats[strategy]['wins'] += 1
            elif pnl < 0:
                strategy_stats[strategy]['losses'] += 1
        
        # Convert to PnLAttribution objects
        attributions = {}
        for strategy, stats in strategy_stats.items():
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            avg_pnl = stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            attributions[strategy] = PnLAttribution(
                dimension='strategy',
                value=strategy,
                total_pnl=stats['pnl'],
                total_trades=stats['trades'],
                winning_trades=stats['wins'],
                losing_trades=stats['losses'],
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                max_pnl=max(stats['pnls']) if stats['pnls'] else 0,
                min_pnl=min(stats['pnls']) if stats['pnls'] else 0,
                total_volume=stats['volumes'],
                sharpe_ratio=self._calculate_sharpe_ratio(stats['pnls']),
                max_drawdown=self._calculate_max_drawdown(stats['pnls'])
            )
        
        return attributions
    
    def _analyze_by_symbol(self, trades: List) -> Dict[str, PnLAttribution]:
        """Analyze PnL by underlying symbol"""
        symbol_stats = defaultdict(lambda: {
            'pnl': 0.0, 'trades': 0, 'wins': 0, 'losses': 0, 'volumes': 0.0,
            'pnls': [], 'volumes': []
        })
        
        for trade in trades:
            symbol = trade.underlying
            pnl = trade.pnl or 0.0
            volume = trade.entry_price * trade.quantity
            
            symbol_stats[symbol]['pnl'] += pnl
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['volumes'] += volume
            symbol_stats[symbol]['pnls'].append(pnl)
            symbol_stats[symbol]['volumes'].append(volume)
            
            if pnl > 0:
                symbol_stats[symbol]['wins'] += 1
            elif pnl < 0:
                symbol_stats[symbol]['losses'] += 1
        
        # Convert to PnLAttribution objects
        attributions = {}
        for symbol, stats in symbol_stats.items():
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            avg_pnl = stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            attributions[symbol] = PnLAttribution(
                dimension='symbol',
                value=symbol,
                total_pnl=stats['pnl'],
                total_trades=stats['trades'],
                winning_trades=stats['wins'],
                losing_trades=stats['losses'],
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                max_pnl=max(stats['pnls']) if stats['pnls'] else 0,
                min_pnl=min(stats['pnls']) if stats['pnls'] else 0,
                total_volume=stats['volumes'],
                sharpe_ratio=self._calculate_sharpe_ratio(stats['pnls']),
                max_drawdown=self._calculate_max_drawdown(stats['pnls'])
            )
        
        return attributions
    
    def _analyze_by_signal_type(self, trades: List) -> Dict[str, PnLAttribution]:
        """Analyze PnL by signal type (BUY CALL, SELL PUT, etc.)"""
        signal_stats = defaultdict(lambda: {
            'pnl': 0.0, 'trades': 0, 'wins': 0, 'losses': 0, 'volumes': 0.0,
            'pnls': [], 'volumes': []
        })
        
        for trade in trades:
            signal_type = trade.signal_type
            pnl = trade.pnl or 0.0
            volume = trade.entry_price * trade.quantity
            
            signal_stats[signal_type]['pnl'] += pnl
            signal_stats[signal_type]['trades'] += 1
            signal_stats[signal_type]['volumes'] += volume
            signal_stats[signal_type]['pnls'].append(pnl)
            signal_stats[signal_type]['volumes'].append(volume)
            
            if pnl > 0:
                signal_stats[signal_type]['wins'] += 1
            elif pnl < 0:
                signal_stats[signal_type]['losses'] += 1
        
        # Convert to PnLAttribution objects
        attributions = {}
        for signal_type, stats in signal_stats.items():
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            avg_pnl = stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            attributions[signal_type] = PnLAttribution(
                dimension='signal_type',
                value=signal_type,
                total_pnl=stats['pnl'],
                total_trades=stats['trades'],
                winning_trades=stats['wins'],
                losing_trades=stats['losses'],
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                max_pnl=max(stats['pnls']) if stats['pnls'] else 0,
                min_pnl=min(stats['pnls']) if stats['pnls'] else 0,
                total_volume=stats['volumes'],
                sharpe_ratio=self._calculate_sharpe_ratio(stats['pnls']),
                max_drawdown=self._calculate_max_drawdown(stats['pnls'])
            )
        
        return attributions
    
    def _analyze_by_time_period(self, trades: List) -> Dict[str, PnLAttribution]:
        """Analyze PnL by time periods (hour, day, week)"""
        time_stats = defaultdict(lambda: {
            'pnl': 0.0, 'trades': 0, 'wins': 0, 'losses': 0, 'volumes': 0.0,
            'pnls': [], 'volumes': []
        })
        
        for trade in trades:
            trade_time = trade.exit_time if hasattr(trade, 'exit_time') and trade.exit_time else trade.timestamp
            
            # Group by hour
            hour_key = f"{trade_time.hour:02d}:00"
            pnl = trade.pnl or 0.0
            volume = trade.entry_price * trade.quantity
            
            time_stats[hour_key]['pnl'] += pnl
            time_stats[hour_key]['trades'] += 1
            time_stats[hour_key]['volumes'] += volume
            time_stats[hour_key]['pnls'].append(pnl)
            time_stats[hour_key]['volumes'].append(volume)
            
            if pnl > 0:
                time_stats[hour_key]['wins'] += 1
            elif pnl < 0:
                time_stats[hour_key]['losses'] += 1
        
        # Convert to PnLAttribution objects
        attributions = {}
        for time_period, stats in time_stats.items():
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            avg_pnl = stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            attributions[time_period] = PnLAttribution(
                dimension='time_period',
                value=time_period,
                total_pnl=stats['pnl'],
                total_trades=stats['trades'],
                winning_trades=stats['wins'],
                losing_trades=stats['losses'],
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                max_pnl=max(stats['pnls']) if stats['pnls'] else 0,
                min_pnl=min(stats['pnls']) if stats['pnls'] else 0,
                total_volume=stats['volumes'],
                sharpe_ratio=self._calculate_sharpe_ratio(stats['pnls']),
                max_drawdown=self._calculate_max_drawdown(stats['pnls'])
            )
        
        return attributions
    
    def _analyze_by_confidence(self, trades: List) -> Dict[str, PnLAttribution]:
        """Analyze PnL by confidence levels"""
        confidence_stats = defaultdict(lambda: {
            'pnl': 0.0, 'trades': 0, 'wins': 0, 'losses': 0, 'volumes': 0.0,
            'pnls': [], 'volumes': []
        })
        
        for trade in trades:
            confidence = getattr(trade, 'confidence', 50)
            confidence_bucket = self._get_confidence_bucket(confidence)
            pnl = trade.pnl or 0.0
            volume = trade.entry_price * trade.quantity
            
            confidence_stats[confidence_bucket]['pnl'] += pnl
            confidence_stats[confidence_bucket]['trades'] += 1
            confidence_stats[confidence_bucket]['volumes'] += volume
            confidence_stats[confidence_bucket]['pnls'].append(pnl)
            confidence_stats[confidence_bucket]['volumes'].append(volume)
            
            if pnl > 0:
                confidence_stats[confidence_bucket]['wins'] += 1
            elif pnl < 0:
                confidence_stats[confidence_bucket]['losses'] += 1
        
        # Convert to PnLAttribution objects
        attributions = {}
        for confidence_bucket, stats in confidence_stats.items():
            win_rate = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            avg_pnl = stats['pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            
            attributions[confidence_bucket] = PnLAttribution(
                dimension='confidence',
                value=confidence_bucket,
                total_pnl=stats['pnl'],
                total_trades=stats['trades'],
                winning_trades=stats['wins'],
                losing_trades=stats['losses'],
                win_rate=win_rate,
                avg_pnl=avg_pnl,
                max_pnl=max(stats['pnls']) if stats['pnls'] else 0,
                min_pnl=min(stats['pnls']) if stats['pnls'] else 0,
                total_volume=stats['volumes'],
                sharpe_ratio=self._calculate_sharpe_ratio(stats['pnls']),
                max_drawdown=self._calculate_max_drawdown(stats['pnls'])
            )
        
        return attributions
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for grouping"""
        if confidence >= 80:
            return "High (80-100)"
        elif confidence >= 60:
            return "Medium-High (60-79)"
        elif confidence >= 40:
            return "Medium (40-59)"
        else:
            return "Low (0-39)"
    
    def _analyze_correlations(self, trades: List) -> Dict:
        """Analyze correlations between different factors"""
        try:
            # Create DataFrame for correlation analysis
            data = []
            for trade in trades:
                data.append({
                    'strategy': trade.strategy,
                    'symbol': trade.underlying,
                    'signal_type': trade.signal_type,
                    'confidence': getattr(trade, 'confidence', 50),
                    'pnl': trade.pnl or 0.0,
                    'volume': trade.entry_price * trade.quantity,
                    'hour': trade.timestamp.hour,
                    'day_of_week': trade.timestamp.weekday()
                })
            
            df = pd.DataFrame(data)
            
            # Calculate correlations
            correlations = {
                'confidence_vs_pnl': df['confidence'].corr(df['pnl']),
                'volume_vs_pnl': df['volume'].corr(df['pnl']),
                'hour_vs_pnl': df['hour'].corr(df['pnl']),
                'day_vs_pnl': df['day_of_week'].corr(df['pnl'])
            }
            
            return correlations
            
        except Exception as e:
            logger.error(f"❌ Error analyzing correlations: {e}")
            return {}
    
    def _generate_performance_insights(self, trades: List) -> Dict:
        """Generate performance insights and recommendations"""
        try:
            insights = {
                'best_performing_strategy': None,
                'worst_performing_strategy': None,
                'best_performing_symbol': None,
                'worst_performing_symbol': None,
                'best_time_to_trade': None,
                'worst_time_to_trade': None,
                'recommendations': []
            }
            
            # Analyze strategies
            strategy_pnl = defaultdict(float)
            for trade in trades:
                strategy_pnl[trade.strategy] += trade.pnl or 0.0
            
            if strategy_pnl:
                insights['best_performing_strategy'] = max(strategy_pnl, key=strategy_pnl.get)
                insights['worst_performing_strategy'] = min(strategy_pnl, key=strategy_pnl.get)
            
            # Analyze symbols
            symbol_pnl = defaultdict(float)
            for trade in trades:
                symbol_pnl[trade.underlying] += trade.pnl or 0.0
            
            if symbol_pnl:
                insights['best_performing_symbol'] = max(symbol_pnl, key=symbol_pnl.get)
                insights['worst_performing_symbol'] = min(symbol_pnl, key=symbol_pnl.get)
            
            # Generate recommendations
            if insights['best_performing_strategy']:
                insights['recommendations'].append(
                    f"Focus on {insights['best_performing_strategy']} strategy - highest PnL"
                )
            
            if insights['worst_performing_strategy']:
                insights['recommendations'].append(
                    f"Review {insights['worst_performing_strategy']} strategy - lowest PnL"
                )
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error generating performance insights: {e}")
            return {}
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio for a series of returns"""
        if not returns or len(returns) < 2:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        return np.mean(returns_array) / np.std(returns_array) if np.std(returns_array) != 0 else 0.0
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown for a series of returns"""
        if not returns:
            return 0.0
        
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        
        for ret in returns:
            cumulative += ret
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _generate_attribution_report(self, attribution_results: Dict):
        """Generate comprehensive attribution report"""
        try:
            logger.info("=" * 80)
            logger.info("📊 PnL ATTRIBUTION ANALYSIS REPORT")
            logger.info("=" * 80)
            
            # Strategy attribution
            if 'by_strategy' in attribution_results:
                logger.info("\n🎯 STRATEGY ATTRIBUTION:")
                for strategy, attr in attribution_results['by_strategy'].items():
                    logger.info(f"   {strategy}: ₹{attr.total_pnl:+,.2f} ({attr.total_trades} trades, {attr.win_rate:.1f}% win rate)")
            
            # Symbol attribution
            if 'by_symbol' in attribution_results:
                logger.info("\n📈 SYMBOL ATTRIBUTION:")
                for symbol, attr in attribution_results['by_symbol'].items():
                    logger.info(f"   {symbol}: ₹{attr.total_pnl:+,.2f} ({attr.total_trades} trades, {attr.win_rate:.1f}% win rate)")
            
            # Signal type attribution
            if 'by_signal_type' in attribution_results:
                logger.info("\n🔄 SIGNAL TYPE ATTRIBUTION:")
                for signal_type, attr in attribution_results['by_signal_type'].items():
                    logger.info(f"   {signal_type}: ₹{attr.total_pnl:+,.2f} ({attr.total_trades} trades, {attr.win_rate:.1f}% win rate)")
            
            # Performance insights
            if 'performance_insights' in attribution_results:
                insights = attribution_results['performance_insights']
                logger.info("\n💡 PERFORMANCE INSIGHTS:")
                if insights.get('best_performing_strategy'):
                    logger.info(f"   Best Strategy: {insights['best_performing_strategy']}")
                if insights.get('worst_performing_strategy'):
                    logger.info(f"   Worst Strategy: {insights['worst_performing_strategy']}")
                if insights.get('recommendations'):
                    logger.info("   Recommendations:")
                    for rec in insights['recommendations']:
                        logger.info(f"     - {rec}")
            
            # Save detailed report
            filename = f"pnl_attribution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(attribution_results, f, indent=2, default=str)
            
            logger.info(f"\n📄 Detailed report saved to: {filename}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error generating attribution report: {e}")
    
    def get_attribution_summary(self) -> Dict:
        """Get summary of attribution analysis"""
        if not self.attribution_data:
            return {}
        
        summary = {}
        for dimension, attributions in self.attribution_data.items():
            if isinstance(attributions, dict):
                # Find best and worst performers
                best = max(attributions.values(), key=lambda x: x.total_pnl) if attributions else None
                worst = min(attributions.values(), key=lambda x: x.total_pnl) if attributions else None
                
                summary[dimension] = {
                    'best': asdict(best) if best else None,
                    'worst': asdict(worst) if worst else None,
                    'total_attributions': len(attributions)
                }
        
        return summary

def main():
    """Main function to run PnL attribution analysis"""
    try:
        from live_paper_trading import LivePaperTradingSystem
        
        # Initialize trading system
        logger.info("🚀 Initializing trading system for PnL attribution...")
        trading_system = LivePaperTradingSystem(initial_capital=100000)
        
        # Initialize attribution engine
        attribution_engine = PnLAttributionEngine(trading_system)
        
        # Analyze PnL attribution
        results = attribution_engine.analyze_pnl_attribution(trading_system.closed_trades)
        
        if results:
            logger.info("✅ PnL attribution analysis completed")
        else:
            logger.info("⚠️ No trades available for attribution analysis")
        
    except Exception as e:
        logger.error(f"❌ PnL attribution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
