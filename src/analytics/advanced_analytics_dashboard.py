#!/usr/bin/env python3
"""
Advanced Analytics Dashboard with ML Insights
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedAnalyticsDashboard:
    """Advanced analytics dashboard with ML insights and predictive analytics"""
    
    def __init__(self):
        self.database = None
        self.real_time_data = None
        self.system_monitor = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize dashboard components"""
        try:
            from src.models.enhanced_database import EnhancedTradingDatabase
            from src.core.enhanced_real_time_manager import EnhancedRealTimeDataManager
            from src.api.fyers import FyersClient
            from src.monitoring.system_monitor import SystemMonitor
            
            self.database = EnhancedTradingDatabase("data/enhanced_trading.db")
            data_provider = FyersClient()
            symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
            self.real_time_data = EnhancedRealTimeDataManager(data_provider, symbols)
            self.system_monitor = SystemMonitor()
            
            logger.info("✅ Advanced analytics dashboard initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize analytics dashboard: {e}")
            raise
    
    def display_advanced_dashboard(self):
        """Display advanced analytics dashboard"""
        print("\n" + "="*100)
        print("🧠 ADVANCED ANALYTICS DASHBOARD - ML INSIGHTS & PREDICTIVE ANALYTICS")
        print("="*100)
        
        # Performance analytics section
        self._display_performance_analytics()
        
        # Risk analytics section
        self._display_risk_analytics()
        
        # Strategy analytics section
        self._display_strategy_analytics()
        
        # Market sentiment analysis
        self._display_market_sentiment()
        
        # Predictive analytics section
        self._display_predictive_analytics()
        
        # Portfolio optimization section
        self._display_portfolio_optimization()
        
        # System performance analytics
        self._display_system_performance_analytics()
        
        print("="*100)
        print(f"�� Advanced Analytics Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)
    
    def _display_performance_analytics(self):
        """Display advanced performance analytics"""
        print("\n📊 PERFORMANCE ANALYTICS")
        print("-" * 50)
        
        try:
            # Get comprehensive performance metrics
            stats = self.database.get_market_statistics("indian")
            
            # Calculate advanced metrics
            total_signals = stats.get('total_signals', 0)
            executed_signals = stats.get('executed_signals', 0)
            total_pnl = stats.get('total_pnl', 0.0)
            
            if total_signals > 0:
                execution_rate = (executed_signals / total_signals) * 100
                avg_pnl_per_signal = total_pnl / executed_signals if executed_signals > 0 else 0
                
                print(f"  📈 Signal Execution Rate: {execution_rate:.1f}%")
                print(f"  💰 Average P&L per Signal: ₹{avg_pnl_per_signal:.2f}")
                print(f"  🎯 Signal Efficiency: {execution_rate * (avg_pnl_per_signal / 100):.2f}")
                
                # Performance trends
                if execution_rate > 70:
                    print(f"  🟢 Performance Status: EXCELLENT")
                elif execution_rate > 50:
                    print(f"  🟡 Performance Status: GOOD")
                else:
                    print(f"  🔴 Performance Status: NEEDS IMPROVEMENT")
            else:
                print(f"  ⏳ No performance data available")
                
        except Exception as e:
            print(f"  ❌ Error in performance analytics: {e}")
    
    def _display_risk_analytics(self):
        """Display risk analytics and metrics"""
        print("\n🛡️ RISK ANALYTICS")
        print("-" * 50)
        
        try:
            # Get recent signals for risk analysis
            entry_signals = self.database.get_entry_signals("indian", limit=50)
            
            if entry_signals:
                # Calculate risk metrics
                confidences = [signal['confidence'] for signal in entry_signals]
                volatilities = [signal.get('volatility', 0.02) for signal in entry_signals]
                
                avg_confidence = np.mean(confidences)
                avg_volatility = np.mean(volatilities)
                confidence_std = np.std(confidences)
                
                print(f"  📊 Average Confidence: {avg_confidence:.1f}%")
                print(f"  📈 Average Volatility: {avg_volatility*100:.1f}%")
                print(f"  📉 Confidence Stability: {100 - confidence_std:.1f}%")
                
                # Risk assessment
                if avg_confidence > 70 and avg_volatility < 0.03:
                    print(f"  🟢 Risk Level: LOW")
                elif avg_confidence > 50 and avg_volatility < 0.05:
                    print(f"  🟡 Risk Level: MEDIUM")
                else:
                    print(f"  🔴 Risk Level: HIGH")
                
                # Risk distribution
                high_risk_signals = sum(1 for c in confidences if c < 50)
                medium_risk_signals = sum(1 for c in confidences if 50 <= c < 70)
                low_risk_signals = sum(1 for c in confidences if c >= 70)
                
                print(f"  🔴 High Risk Signals: {high_risk_signals}")
                print(f"  🟡 Medium Risk Signals: {medium_risk_signals}")
                print(f"  🟢 Low Risk Signals: {low_risk_signals}")
            else:
                print(f"  ⏳ No risk data available")
                
        except Exception as e:
            print(f"  ❌ Error in risk analytics: {e}")
    
    def _display_strategy_analytics(self):
        """Display strategy performance analytics"""
        print("\n🎯 STRATEGY ANALYTICS")
        print("-" * 50)
        
        try:
            # Get signals by strategy
            entry_signals = self.database.get_entry_signals("indian", limit=100)
            
            if entry_signals:
                # Group by strategy
                strategy_stats = {}
                for signal in entry_signals:
                    strategy = signal['strategy']
                    if strategy not in strategy_stats:
                        strategy_stats[strategy] = {
                            'count': 0,
                            'total_confidence': 0,
                            'signals': []
                        }
                    
                    strategy_stats[strategy]['count'] += 1
                    strategy_stats[strategy]['total_confidence'] += signal['confidence']
                    strategy_stats[strategy]['signals'].append(signal)
                
                # Display strategy performance
                for strategy, stats in strategy_stats.items():
                    avg_confidence = stats['total_confidence'] / stats['count']
                    print(f"  📊 {strategy}:")
                    print(f"    Signals: {stats['count']}")
                    print(f"    Avg Confidence: {avg_confidence:.1f}%")
                    
                    # Strategy effectiveness
                    if avg_confidence > 75:
                        print(f"    Status: 🟢 HIGHLY EFFECTIVE")
                    elif avg_confidence > 60:
                        print(f"    Status: 🟡 MODERATELY EFFECTIVE")
                    else:
                        print(f"    Status: 🔴 NEEDS OPTIMIZATION")
            else:
                print(f"  ⏳ No strategy data available")
                
        except Exception as e:
            print(f"  ❌ Error in strategy analytics: {e}")
    
    def _display_market_sentiment(self):
        """Display market sentiment analysis"""
        print("\n🌊 MARKET SENTIMENT ANALYSIS")
        print("-" * 50)
        
        try:
            # Get recent signals for sentiment analysis
            entry_signals = self.database.get_entry_signals("indian", limit=20)
            
            if entry_signals:
                # Analyze signal types
                buy_signals = sum(1 for s in entry_signals if 'BUY' in s['signal_type'])
                sell_signals = sum(1 for s in entry_signals if 'SELL' in s['signal_type'])
                
                total_signals = len(entry_signals)
                buy_ratio = (buy_signals / total_signals) * 100
                sell_ratio = (sell_signals / total_signals) * 100
                
                print(f"  📈 Bullish Signals: {buy_signals} ({buy_ratio:.1f}%)")
                print(f"  📉 Bearish Signals: {sell_signals} ({sell_ratio:.1f}%)")
                
                # Market sentiment
                if buy_ratio > 60:
                    print(f"  🟢 Market Sentiment: BULLISH")
                elif sell_ratio > 60:
                    print(f"  🔴 Market Sentiment: BEARISH")
                else:
                    print(f"  🟡 Market Sentiment: NEUTRAL")
                
                # Confidence-weighted sentiment
                buy_confidence = np.mean([s['confidence'] for s in entry_signals if 'BUY' in s['signal_type']]) if buy_signals > 0 else 0
                sell_confidence = np.mean([s['confidence'] for s in entry_signals if 'SELL' in s['signal_type']]) if sell_signals > 0 else 0
                
                print(f"  📊 Avg Buy Confidence: {buy_confidence:.1f}%")
                print(f"  📊 Avg Sell Confidence: {sell_confidence:.1f}%")
            else:
                print(f"  ⏳ No sentiment data available")
                
        except Exception as e:
            print(f"  ❌ Error in market sentiment analysis: {e}")
    
    def _display_predictive_analytics(self):
        """Display predictive analytics and forecasting"""
        print("\n🔮 PREDICTIVE ANALYTICS")
        print("-" * 50)
        
        try:
            # Get historical performance data
            entry_signals = self.database.get_entry_signals("indian", limit=100)
            
            if len(entry_signals) >= 10:
                # Simple trend analysis
                recent_signals = entry_signals[:10]
                older_signals = entry_signals[10:20] if len(entry_signals) >= 20 else entry_signals[10:]
                
                recent_avg_confidence = np.mean([s['confidence'] for s in recent_signals])
                older_avg_confidence = np.mean([s['confidence'] for s in older_signals]) if older_signals else recent_avg_confidence
                
                confidence_trend = recent_avg_confidence - older_avg_confidence
                
                print(f"  �� Recent Avg Confidence: {recent_avg_confidence:.1f}%")
                print(f"  📊 Historical Avg Confidence: {older_avg_confidence:.1f}%")
                print(f"  📈 Confidence Trend: {confidence_trend:+.1f}%")
                
                # Predictions
                if confidence_trend > 5:
                    print(f"  🔮 Prediction: IMPROVING PERFORMANCE")
                elif confidence_trend < -5:
                    print(f"  🔮 Prediction: DECLINING PERFORMANCE")
                else:
                    print(f"  🔮 Prediction: STABLE PERFORMANCE")
                
                # Signal frequency analysis
                signal_frequency = len(entry_signals) / 7  # signals per day (assuming 7 days of data)
                print(f"  📊 Signal Frequency: {signal_frequency:.1f} signals/day")
                
                if signal_frequency > 10:
                    print(f"  🔮 Prediction: HIGH ACTIVITY PERIOD")
                elif signal_frequency < 5:
                    print(f"  🔮 Prediction: LOW ACTIVITY PERIOD")
                else:
                    print(f"  🔮 Prediction: NORMAL ACTIVITY PERIOD")
            else:
                print(f"  ⏳ Insufficient data for predictions")
                
        except Exception as e:
            print(f"  ❌ Error in predictive analytics: {e}")
    
    def _display_portfolio_optimization(self):
        """Display portfolio optimization insights"""
        print("\n💼 PORTFOLIO OPTIMIZATION")
        print("-" * 50)
        
        try:
            # Get signals by symbol
            entry_signals = self.database.get_entry_signals("indian", limit=50)
            
            if entry_signals:
                # Group by symbol
                symbol_stats = {}
                for signal in entry_signals:
                    symbol = signal['symbol']
                    if symbol not in symbol_stats:
                        symbol_stats[symbol] = {
                            'count': 0,
                            'total_confidence': 0,
                            'signals': []
                        }
                    
                    symbol_stats[symbol]['count'] += 1
                    symbol_stats[symbol]['total_confidence'] += signal['confidence']
                    symbol_stats[symbol]['signals'].append(signal)
                
                # Display symbol performance
                print(f"  📊 Symbol Performance:")
                for symbol, stats in symbol_stats.items():
                    avg_confidence = stats['total_confidence'] / stats['count']
                    print(f"    {symbol}: {stats['count']} signals, {avg_confidence:.1f}% avg confidence")
                
                # Portfolio recommendations
                best_symbol = max(symbol_stats.items(), key=lambda x: x[1]['total_confidence'] / x[1]['count'])
                worst_symbol = min(symbol_stats.items(), key=lambda x: x[1]['total_confidence'] / x[1]['count'])
                
                print(f"  🟢 Best Performer: {best_symbol[0]} ({best_symbol[1]['total_confidence']/best_symbol[1]['count']:.1f}%)")
                print(f"  🔴 Needs Review: {worst_symbol[0]} ({worst_symbol[1]['total_confidence']/worst_symbol[1]['count']:.1f}%)")
                
                # Diversification analysis
                symbol_count = len(symbol_stats)
                if symbol_count >= 3:
                    print(f"  📊 Portfolio Diversification: GOOD ({symbol_count} symbols)")
                else:
                    print(f"  📊 Portfolio Diversification: LIMITED ({symbol_count} symbols)")
            else:
                print(f"  ⏳ No portfolio data available")
                
        except Exception as e:
            print(f"  ❌ Error in portfolio optimization: {e}")
    
    def _display_system_performance_analytics(self):
        """Display system performance analytics"""
        print("\n🖥️ SYSTEM PERFORMANCE ANALYTICS")
        print("-" * 50)
        
        try:
            # Get system metrics
            metrics = self.system_monitor.get_system_metrics()
            
            # Performance analysis
            cpu_status = "🟢 OPTIMAL" if metrics.cpu_percent < 50 else "🟡 MODERATE" if metrics.cpu_percent < 80 else "🔴 HIGH"
            memory_status = "🟢 OPTIMAL" if metrics.memory_percent < 60 else "🟡 MODERATE" if metrics.memory_percent < 85 else "🔴 HIGH"
            disk_status = "🟢 OPTIMAL" if metrics.disk_percent < 70 else "🟡 MODERATE" if metrics.disk_percent < 90 else "🔴 HIGH"
            
            print(f"  💻 CPU Performance: {cpu_status} ({metrics.cpu_percent:.1f}%)")
            print(f"  🧠 Memory Performance: {memory_status} ({metrics.memory_percent:.1f}%)")
            print(f"  💾 Disk Performance: {disk_status} ({metrics.disk_percent:.1f}%)")
            
            # WebSocket performance
            ws_status = self.real_time_data.get_connection_status()
            if ws_status.get('connected', False):
                print(f"  📡 WebSocket Performance: 🟢 CONNECTED")
            else:
                print(f"  📡 WebSocket Performance: 🔴 DISCONNECTED")
            
            # Overall system health
            overall_health = "🟢 EXCELLENT"
            if metrics.cpu_percent > 80 or metrics.memory_percent > 85 or metrics.disk_percent > 90:
                overall_health = "🔴 CRITICAL"
            elif metrics.cpu_percent > 60 or metrics.memory_percent > 70 or metrics.disk_percent > 80:
                overall_health = "🟡 WARNING"
            
            print(f"  🏥 Overall System Health: {overall_health}")
            
        except Exception as e:
            print(f"  ❌ Error in system performance analytics: {e}")
    
    def start_advanced_monitoring(self, interval_seconds: int = 60):
        """Start advanced monitoring"""
        print(f"🧠 Starting advanced analytics monitoring (every {interval_seconds}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.display_advanced_dashboard()
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 Advanced monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Advanced monitoring error: {e}")

def main():
    """Main function"""
    dashboard = AdvancedAnalyticsDashboard()
    
    try:
        # Display initial advanced dashboard
        dashboard.display_advanced_dashboard()
        
        # Start advanced monitoring
        dashboard.start_advanced_monitoring(interval_seconds=60)
        
    except Exception as e:
        logger.error(f"❌ Advanced analytics dashboard error: {e}")

if __name__ == "__main__":
    main()
