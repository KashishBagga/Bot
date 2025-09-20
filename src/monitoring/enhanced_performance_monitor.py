#!/usr/bin/env python3
"""
Enhanced Performance Monitoring System
Real-time performance tracking and analytics
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: datetime
    metric_name: str
    value: float
    market: str
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class PerformanceAlert:
    """Performance alert data structure."""
    timestamp: datetime
    level: AlertLevel
    title: str
    message: str
    market: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None

class EnhancedPerformanceMonitor:
    """Enhanced real-time performance monitoring system."""
    
    def __init__(self, db_path: str = "data/performance_monitoring.db"):
        self.db_path = db_path
        self.is_running = False
        self.monitor_thread = None
        self.metrics_buffer = []
        self.alerts_buffer = []
        self.last_update = datetime.now()
        
        # Performance thresholds
        self.thresholds = {
            'max_drawdown': -0.15,  # -15%
            'daily_loss_limit': -0.10,  # -10%
            'win_rate_min': 0.40,  # 40%
            'profit_factor_min': 1.2,
            'sharpe_ratio_min': 0.5,
            'max_consecutive_losses': 5,
            'position_limit': 15,
            'risk_per_trade_max': 0.05  # 5%
        }
        
        # Real-time metrics
        self.current_metrics = {
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 1.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'consecutive_losses': 0,
            'open_positions': 0,
            'capital_utilization': 0.0,
            'average_trade_duration': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize performance monitoring database."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        value REAL NOT NULL,
                        market TEXT NOT NULL,
                        symbol TEXT,
                        strategy TEXT,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Performance alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT NOT NULL,
                        market TEXT NOT NULL,
                        metric_value REAL,
                        threshold REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Daily performance summary
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS daily_performance_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        market TEXT NOT NULL,
                        total_pnl REAL NOT NULL,
                        realized_pnl REAL NOT NULL,
                        unrealized_pnl REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        winning_trades INTEGER NOT NULL,
                        losing_trades INTEGER NOT NULL,
                        win_rate REAL NOT NULL,
                        profit_factor REAL NOT NULL,
                        max_drawdown REAL NOT NULL,
                        sharpe_ratio REAL NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, market)
                    )
                ''')
                
                conn.commit()
                logger.info("✅ Performance monitoring database initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize performance database: {e}")
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.is_running:
            logger.warning("Performance monitoring already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 Enhanced performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        # Save any remaining metrics
        self._flush_buffers()
        logger.info("🛑 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Update metrics every 30 seconds
                self._update_performance_metrics()
                
                # Check thresholds and generate alerts
                self._check_performance_thresholds()
                
                # Flush buffers every minute
                if len(self.metrics_buffer) > 10 or len(self.alerts_buffer) > 5:
                    self._flush_buffers()
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _update_performance_metrics(self):
        """Update real-time performance metrics."""
        try:
            # Get latest data from trading databases
            crypto_metrics = self._get_market_performance('crypto')
            indian_metrics = self._get_market_performance('indian')
            
            # Update current metrics
            self._calculate_combined_metrics(crypto_metrics, indian_metrics)
            
            # Store metrics
            timestamp = datetime.now()
            for metric_name, value in self.current_metrics.items():
                metric = PerformanceMetric(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    value=value,
                    market='combined'
                )
                self.metrics_buffer.append(metric)
            
            self.last_update = timestamp
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _get_market_performance(self, market: str) -> Dict[str, Any]:
        """Get performance data for a specific market."""
        try:
            db_path = f"data/{market}_trading.db" if market == 'crypto' else "data/trading.db"
            
            if not os.path.exists(db_path):
                return {}
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Get basic trade statistics
                cursor.execute("SELECT COUNT(*) FROM closed_trades")
                total_trades = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT SUM(pnl) FROM closed_trades")
                total_pnl = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT COUNT(*) FROM closed_trades WHERE pnl > 0")
                winning_trades = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM closed_trades WHERE pnl < 0")
                losing_trades = cursor.fetchone()[0] or 0
                
                cursor.execute("SELECT COUNT(*) FROM open_trades")
                open_positions = cursor.fetchone()[0] or 0
                
                # Calculate win rate
                win_rate = (winning_trades / max(1, total_trades)) * 100
                
                # Get daily P&L
                cursor.execute("""
                    SELECT SUM(pnl) FROM closed_trades 
                    WHERE date(exit_time) = date('now')
                """)
                daily_pnl = cursor.fetchone()[0] or 0.0
                
                return {
                    'total_trades': total_trades,
                    'total_pnl': total_pnl,
                    'daily_pnl': daily_pnl,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'open_positions': open_positions
                }
                
        except Exception as e:
            logger.error(f"Error getting {market} performance: {e}")
            return {}
    
    def _calculate_combined_metrics(self, crypto_metrics: Dict[str, Any], indian_metrics: Dict[str, Any]):
        """Calculate combined performance metrics."""
        
        # Combine basic metrics
        self.current_metrics['total_trades'] = crypto_metrics.get('total_trades', 0) + indian_metrics.get('total_trades', 0)
        self.current_metrics['total_pnl'] = crypto_metrics.get('total_pnl', 0) + indian_metrics.get('total_pnl', 0)
        self.current_metrics['daily_pnl'] = crypto_metrics.get('daily_pnl', 0) + indian_metrics.get('daily_pnl', 0)
        self.current_metrics['winning_trades'] = crypto_metrics.get('winning_trades', 0) + indian_metrics.get('winning_trades', 0)
        self.current_metrics['losing_trades'] = crypto_metrics.get('losing_trades', 0) + indian_metrics.get('losing_trades', 0)
        self.current_metrics['open_positions'] = crypto_metrics.get('open_positions', 0) + indian_metrics.get('open_positions', 0)
        
        # Calculate derived metrics
        total_trades = self.current_metrics['total_trades']
        if total_trades > 0:
            self.current_metrics['win_rate'] = (self.current_metrics['winning_trades'] / total_trades) * 100
        
        # Calculate profit factor
        total_wins = crypto_metrics.get('total_pnl', 0) if crypto_metrics.get('total_pnl', 0) > 0 else 0
        total_wins += indian_metrics.get('total_pnl', 0) if indian_metrics.get('total_pnl', 0) > 0 else 0
        
        total_losses = abs(crypto_metrics.get('total_pnl', 0)) if crypto_metrics.get('total_pnl', 0) < 0 else 0
        total_losses += abs(indian_metrics.get('total_pnl', 0)) if indian_metrics.get('total_pnl', 0) < 0 else 0
        
        if total_losses > 0:
            self.current_metrics['profit_factor'] = total_wins / total_losses
        else:
            self.current_metrics['profit_factor'] = total_wins if total_wins > 0 else 1.0
        
        # Update drawdown (simplified calculation)
        if self.current_metrics['total_pnl'] < 0:
            self.current_metrics['current_drawdown'] = abs(self.current_metrics['total_pnl'])
            if self.current_metrics['current_drawdown'] > self.current_metrics['max_drawdown']:
                self.current_metrics['max_drawdown'] = self.current_metrics['current_drawdown']
    
    def _check_performance_thresholds(self):
        """Check performance against thresholds and generate alerts."""
        
        # Check drawdown
        if self.current_metrics['current_drawdown'] > abs(self.thresholds['max_drawdown']) * 10000:  # Assuming 10k capital
            self._create_alert(
                AlertLevel.CRITICAL,
                "Maximum Drawdown Exceeded",
                f"Current drawdown: ${self.current_metrics['current_drawdown']:.2f}",
                'combined'
            )
        
        # Check daily loss limit
        if self.current_metrics['daily_pnl'] < self.thresholds['daily_loss_limit'] * 10000:
            self._create_alert(
                AlertLevel.WARNING,
                "Daily Loss Limit Approached",
                f"Daily P&L: ${self.current_metrics['daily_pnl']:.2f}",
                'combined'
            )
        
        # Check win rate
        if self.current_metrics['total_trades'] > 10 and self.current_metrics['win_rate'] < self.thresholds['win_rate_min'] * 100:
            self._create_alert(
                AlertLevel.WARNING,
                "Low Win Rate",
                f"Current win rate: {self.current_metrics['win_rate']:.1f}%",
                'combined'
            )
        
        # Check profit factor
        if self.current_metrics['profit_factor'] < self.thresholds['profit_factor_min']:
            self._create_alert(
                AlertLevel.WARNING,
                "Low Profit Factor",
                f"Current profit factor: {self.current_metrics['profit_factor']:.2f}",
                'combined'
            )
        
        # Check position limits
        if self.current_metrics['open_positions'] > self.thresholds['position_limit']:
            self._create_alert(
                AlertLevel.ERROR,
                "Position Limit Exceeded",
                f"Open positions: {self.current_metrics['open_positions']}",
                'combined'
            )
    
    def _create_alert(self, level: AlertLevel, title: str, message: str, market: str, metric_value: Optional[float] = None, threshold: Optional[float] = None):
        """Create a performance alert."""
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            level=level,
            title=title,
            message=message,
            market=market,
            metric_value=metric_value,
            threshold=threshold
        )
        
        self.alerts_buffer.append(alert)
        logger.warning(f"🚨 {level.value}: {title} - {message}")
    
    def _flush_buffers(self):
        """Save buffered metrics and alerts to database."""
        try:
            if not self.metrics_buffer and not self.alerts_buffer:
                return
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Save metrics
                for metric in self.metrics_buffer:
                    cursor.execute('''
                        INSERT INTO performance_metrics 
                        (timestamp, metric_name, value, market, symbol, strategy, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metric.timestamp.isoformat(),
                        metric.metric_name,
                        metric.value,
                        metric.market,
                        metric.symbol,
                        metric.strategy,
                        json.dumps(metric.metadata) if metric.metadata else None
                    ))
                
                # Save alerts
                for alert in self.alerts_buffer:
                    cursor.execute('''
                        INSERT INTO performance_alerts 
                        (timestamp, level, title, message, market, metric_value, threshold)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.timestamp.isoformat(),
                        alert.level.value,
                        alert.title,
                        alert.message,
                        alert.market,
                        alert.metric_value,
                        alert.threshold
                    ))
                
                conn.commit()
                
                # Clear buffers
                self.metrics_buffer.clear()
                self.alerts_buffer.clear()
                
        except Exception as e:
            logger.error(f"Error flushing performance buffers: {e}")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time performance metrics."""
        return {
            **self.current_metrics,
            'last_update': self.last_update.isoformat(),
            'monitoring_active': self.is_running
        }
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for specified period."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent alerts
                cursor.execute('''
                    SELECT level, title, message, timestamp 
                    FROM performance_alerts 
                    WHERE timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                    LIMIT 10
                '''.format(days))
                
                recent_alerts = [
                    {
                        'level': row[0],
                        'title': row[1],
                        'message': row[2],
                        'timestamp': row[3]
                    }
                    for row in cursor.fetchall()
                ]
                
                # Get performance trends
                cursor.execute('''
                    SELECT metric_name, AVG(value), MIN(value), MAX(value)
                    FROM performance_metrics 
                    WHERE timestamp >= datetime('now', '-{} days')
                    AND market = 'combined'
                    GROUP BY metric_name
                '''.format(days))
                
                performance_trends = {
                    row[0]: {
                        'average': row[1],
                        'minimum': row[2],
                        'maximum': row[3]
                    }
                    for row in cursor.fetchall()
                }
                
                return {
                    'current_metrics': self.current_metrics,
                    'recent_alerts': recent_alerts,
                    'performance_trends': performance_trends,
                    'thresholds': self.thresholds,
                    'summary_period_days': days
                }
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                'current_metrics': self.current_metrics,
                'recent_alerts': [],
                'performance_trends': {},
                'thresholds': self.thresholds,
                'summary_period_days': days
            }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update performance thresholds."""
        self.thresholds.update(new_thresholds)
        logger.info(f"📊 Updated performance thresholds: {new_thresholds}")
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        metrics = self.get_real_time_metrics()
        summary = self.get_performance_summary(7)
        
        report = f"""
🚀 ENHANCED TRADING SYSTEM PERFORMANCE REPORT
=============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 CURRENT PERFORMANCE METRICS:
------------------------------
Total P&L: ${metrics['total_pnl']:.2f}
Daily P&L: ${metrics['daily_pnl']:.2f}
Win Rate: {metrics['win_rate']:.1f}%
Profit Factor: {metrics['profit_factor']:.2f}
Total Trades: {metrics['total_trades']}
Open Positions: {metrics['open_positions']}
Max Drawdown: ${metrics['max_drawdown']:.2f}

🎯 PERFORMANCE THRESHOLDS:
-------------------------
Max Drawdown Limit: {self.thresholds['max_drawdown']*100:.1f}%
Daily Loss Limit: {self.thresholds['daily_loss_limit']*100:.1f}%
Min Win Rate: {self.thresholds['win_rate_min']*100:.1f}%
Min Profit Factor: {self.thresholds['profit_factor_min']:.2f}

🚨 RECENT ALERTS ({len(summary['recent_alerts'])}):
------------------
"""
        
        for alert in summary['recent_alerts'][:5]:
            report += f"• {alert['level']}: {alert['title']} ({alert['timestamp']})\n"
        
        report += f"""
�� SYSTEM STATUS:
----------------
Monitoring Active: {'✅ Yes' if self.is_running else '❌ No'}
Last Update: {metrics['last_update']}

🔧 RECOMMENDATIONS:
------------------
"""
        
        # Add recommendations based on current metrics
        if metrics['win_rate'] < 50:
            report += "• Consider reviewing strategy parameters - win rate below 50%\n"
        
        if metrics['profit_factor'] < 1.5:
            report += "• Profit factor could be improved - consider tighter risk management\n"
        
        if metrics['open_positions'] > 10:
            report += "• High number of open positions - monitor risk exposure\n"
        
        if metrics['daily_pnl'] < -500:
            report += "• Daily losses significant - consider reducing position sizes\n"
        
        return report

# Global instance
_performance_monitor = None

def get_performance_monitor() -> EnhancedPerformanceMonitor:
    """Get or create performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = EnhancedPerformanceMonitor()
    
    return _performance_monitor
