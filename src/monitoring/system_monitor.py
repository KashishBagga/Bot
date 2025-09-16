"""
System Health Monitoring Dashboard
=================================
Monitors system performance, WebSocket health, trading metrics,
and provides alerts for critical events
"""

import logging
import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict
    timestamp: datetime

@dataclass
class TradingMetrics:
    """Trading system metrics"""
    active_trades: int
    total_pnl: float
    win_rate: float
    signals_generated: int
    signals_executed: int
    websocket_connected: bool
    last_signal_time: Optional[datetime]

@dataclass
class Alert:
    """System alert"""
    level: AlertLevel
    message: str
    timestamp: datetime
    component: str
    resolved: bool = False

class SystemMonitor:
    """System health monitoring and alerting"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.metrics_history: List[SystemMetrics] = []
        self.trading_metrics: Optional[TradingMetrics] = None
        self.monitoring = False
        self.monitor_thread = None
        
        # Thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 85.0
        self.disk_threshold = 90.0
        self.websocket_timeout = 60  # seconds
        
        logger.info("📊 System Monitor initialized")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"❌ Error getting system metrics: {e}")
            return SystemMetrics(0, 0, 0, {}, datetime.now())
    
    def get_websocket_health(self) -> Dict:
        """Get WebSocket connection health."""
        try:
            from src.core.fyers_websocket_manager import get_websocket_manager
            
            symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
            ws_manager = get_websocket_manager(symbols)
            
            return {
                'connected': ws_manager.is_connected,
                'running': ws_manager.is_running,
                'live_data_count': len(ws_manager.get_all_live_data()),
                'last_data_time': ws_manager.last_data_time,
                'connection_attempts': ws_manager.connection_attempts
            }
        except Exception as e:
            return {
                'connected': False,
                'running': False,
                'live_data_count': 0,
                'last_data_time': None,
                'connection_attempts': 0,
                'error': str(e)
            }
    
    def get_trading_metrics(self) -> TradingMetrics:
        """Get current trading system metrics."""
        try:
            from src.models.consolidated_database import ConsolidatedTradingDatabase
            
            db = ConsolidatedTradingDatabase("data/trading.db")
            
            # Get active trades
            active_trades = db.get_open_trades_count("indian") + db.get_open_trades_count("crypto")
            
            # Get total P&L
            indian_stats = db.get_market_statistics("indian")
            crypto_stats = db.get_market_statistics("crypto")
            
            total_pnl = 0.0
            if indian_stats:
                total_pnl += indian_stats[3]  # P&L is 4th element
            if crypto_stats:
                total_pnl += crypto_stats[3]
            
            # Get win rate
            win_rate = 0.0
            if indian_stats and indian_stats[1] > 0:  # closed_trades > 0
                win_rate = indian_stats[4]  # win_rate is 5th element
            
            # Get WebSocket health
            ws_health = self.get_websocket_health()
            
            # Get recent signals (simplified)
            recent_signals = db.get_recent_signals("indian", limit=1)
            last_signal_time = None
            if recent_signals:
                last_signal_time = recent_signals[0][7]  # timestamp is 8th element
            
            return TradingMetrics(
                active_trades=active_trades,
                total_pnl=total_pnl,
                win_rate=win_rate,
                signals_generated=0,  # Would need to track this
                signals_executed=0,   # Would need to track this
                websocket_connected=ws_health['connected'],
                last_signal_time=last_signal_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting trading metrics: {e}")
            return TradingMetrics(0, 0.0, 0.0, 0, 0, False, None)
    
    def check_system_health(self, metrics: SystemMetrics) -> List[Alert]:
        """Check system health and generate alerts."""
        alerts = []
        
        # CPU usage check
        if metrics.cpu_percent > self.cpu_threshold:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                timestamp=datetime.now(),
                component="system"
            ))
        
        # Memory usage check
        if metrics.memory_percent > self.memory_threshold:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High memory usage: {metrics.memory_percent:.1f}%",
                timestamp=datetime.now(),
                component="system"
            ))
        
        # Disk usage check
        if metrics.disk_percent > self.disk_threshold:
            alerts.append(Alert(
                level=AlertLevel.ERROR,
                message=f"High disk usage: {metrics.disk_percent:.1f}%",
                timestamp=datetime.now(),
                component="system"
            ))
        
        return alerts
    
    def check_websocket_health(self, ws_health: Dict) -> List[Alert]:
        """Check WebSocket health and generate alerts."""
        alerts = []
        
        # Connection status
        if not ws_health['connected']:
            alerts.append(Alert(
                level=AlertLevel.ERROR,
                message="WebSocket disconnected",
                timestamp=datetime.now(),
                component="websocket"
            ))
        
        # Data flow check
        if ws_health['connected'] and ws_health['live_data_count'] == 0:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message="WebSocket connected but no live data",
                timestamp=datetime.now(),
                component="websocket"
            ))
        
        # Connection attempts
        if ws_health['connection_attempts'] > 3:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Multiple connection attempts: {ws_health['connection_attempts']}",
                timestamp=datetime.now(),
                component="websocket"
            ))
        
        return alerts
    
    def check_trading_health(self, trading_metrics: TradingMetrics) -> List[Alert]:
        """Check trading system health and generate alerts."""
        alerts = []
        
        # High number of active trades
        if trading_metrics.active_trades > 50:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"High number of active trades: {trading_metrics.active_trades}",
                timestamp=datetime.now(),
                component="trading"
            ))
        
        # Large losses
        if trading_metrics.total_pnl < -1000:
            alerts.append(Alert(
                level=AlertLevel.ERROR,
                message=f"Large losses: {trading_metrics.total_pnl:.2f}",
                timestamp=datetime.now(),
                component="trading"
            ))
        
        # Low win rate
        if trading_metrics.win_rate < 30 and trading_metrics.active_trades > 10:
            alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Low win rate: {trading_metrics.win_rate:.1f}%",
                timestamp=datetime.now(),
                component="trading"
            ))
        
        # No recent signals
        if trading_metrics.last_signal_time:
            time_since_signal = datetime.now() - trading_metrics.last_signal_time
            if time_since_signal > timedelta(minutes=30):
                alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    message=f"No signals for {time_since_signal}",
                    timestamp=datetime.now(),
                    component="trading"
                ))
        
        return alerts
    
    def add_alert(self, alert: Alert):
        """Add a new alert."""
        self.alerts.append(alert)
        logger.log(
            getattr(logging, alert.level.value.upper()),
            f"🚨 {alert.component.upper()}: {alert.message}"
        )
    
    def resolve_alert(self, component: str, message: str):
        """Mark an alert as resolved."""
        for alert in self.alerts:
            if (alert.component == component and 
                message in alert.message and 
                not alert.resolved):
                alert.resolved = True
                logger.info(f"✅ Resolved {component} alert: {message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def cleanup_old_alerts(self, hours: int = 24):
        """Remove alerts older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get system metrics
                system_metrics = self.get_system_metrics()
                self.metrics_history.append(system_metrics)
                
                # Keep only last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Check system health
                system_alerts = self.check_system_health(system_metrics)
                for alert in system_alerts:
                    self.add_alert(alert)
                
                # Check WebSocket health
                ws_health = self.get_websocket_health()
                ws_alerts = self.check_websocket_health(ws_health)
                for alert in ws_alerts:
                    self.add_alert(alert)
                
                # Check trading health
                trading_metrics = self.get_trading_metrics()
                self.trading_metrics = trading_metrics
                trading_alerts = self.check_trading_health(trading_metrics)
                for alert in trading_alerts:
                    self.add_alert(alert)
                
                # Cleanup old alerts
                self.cleanup_old_alerts()
                
                # Sleep for 30 seconds
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(30)
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring:
            logger.warning("⚠️ Monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🚀 System monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 System monitoring stopped")
    
    def get_health_summary(self) -> Dict:
        """Get comprehensive health summary."""
        system_metrics = self.get_system_metrics()
        ws_health = self.get_websocket_health()
        trading_metrics = self.get_trading_metrics()
        active_alerts = self.get_active_alerts()
        
        return {
            'system': {
                'cpu_percent': system_metrics.cpu_percent,
                'memory_percent': system_metrics.memory_percent,
                'disk_percent': system_metrics.disk_percent,
                'status': 'healthy' if system_metrics.cpu_percent < 80 and system_metrics.memory_percent < 85 else 'warning'
            },
            'websocket': {
                'connected': ws_health['connected'],
                'live_data_count': ws_health['live_data_count'],
                'status': 'healthy' if ws_health['connected'] and ws_health['live_data_count'] > 0 else 'warning'
            },
            'trading': {
                'active_trades': trading_metrics.active_trades,
                'total_pnl': trading_metrics.total_pnl,
                'win_rate': trading_metrics.win_rate,
                'status': 'healthy' if trading_metrics.total_pnl > -500 and trading_metrics.win_rate > 40 else 'warning'
            },
            'alerts': {
                'total': len(active_alerts),
                'critical': len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                'error': len([a for a in active_alerts if a.level == AlertLevel.ERROR]),
                'warning': len([a for a in active_alerts if a.level == AlertLevel.WARNING])
            }
        }

# Global monitor instance
_monitor = None

def get_system_monitor() -> SystemMonitor:
    """Get or create global system monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor()
    return _monitor
