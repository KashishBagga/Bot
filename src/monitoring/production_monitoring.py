#!/usr/bin/env python3
"""
Production Monitoring & Alerting
HIGH PRIORITY #1: Comprehensive monitoring and alerting system
"""

import sys
import os
import time
import asyncio
import logging
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from typing import Any
from dataclasses import dataclass
from enum import Enum
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class MetricType(Enum):
    SYSTEM = "SYSTEM"
    TRADING = "TRADING"
    API = "API"
    DATABASE = "DATABASE"
    RISK = "RISK"

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metric_type: MetricType
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

class ProductionMonitor:
    """Production monitoring and alerting system"""
    
    def __init__(self):
        self.metrics = {}
        self.alerts = []
        self.monitoring_active = False
        self.alert_channels = {
            'email': True,
            'telegram': True,
            'slack': True,
            'webhook': True
        }
        
        # Thresholds
        self.thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 85},
            'memory_percent': {'warning': 80, 'critical': 90},
            'disk_percent': {'warning': 85, 'critical': 95},
            'api_error_rate': {'warning': 0.05, 'critical': 0.10},
            'order_latency': {'warning': 2.0, 'critical': 5.0},
            'fill_rate': {'warning': 0.90, 'critical': 0.80},
            'unreconciled_trades': {'warning': 1, 'critical': 5},
            'daily_pnl': {'warning': -1000, 'critical': -5000},
            'exposure_percent': {'warning': 0.70, 'critical': 0.85}
        }
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("🔍 Starting production monitoring")
        
        while self.monitoring_active:
            try:
                # Collect all metrics
                await self._collect_system_metrics()
                await self._collect_trading_metrics()
                await self._collect_api_metrics()
                await self._collect_database_metrics()
                await self._collect_risk_metrics()
                
                # Check thresholds and generate alerts
                await self._check_thresholds()
                
                # Send alerts
                await self._process_alerts()
                
                # Wait before next collection
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
    async def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._store_metric('cpu_percent', cpu_percent, '%', MetricType.SYSTEM)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self._store_metric('memory_percent', memory.percent, '%', MetricType.SYSTEM)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._store_metric('disk_percent', disk_percent, '%', MetricType.SYSTEM)
            
            # Network I/O
            network = psutil.net_io_counters()
            self._store_metric('network_bytes_sent', network.bytes_sent, 'bytes', MetricType.SYSTEM)
            self._store_metric('network_bytes_recv', network.bytes_recv, 'bytes', MetricType.SYSTEM)
            
            logger.debug(f"📊 System metrics collected: CPU={cpu_percent:.1f}%, Memory={memory.percent:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ System metrics collection failed: {e}")
    
    async def _collect_trading_metrics(self):
        """Collect trading metrics"""
        try:
            # Mock trading metrics - in real implementation, get from trading system
            self._store_metric('orders_per_minute', 5.2, 'orders/min', MetricType.TRADING)
            self._store_metric('fill_rate', 0.95, '%', MetricType.TRADING)
            self._store_metric('order_latency', 1.5, 'seconds', MetricType.TRADING)
            self._store_metric('daily_pnl', 2500.0, '₹', MetricType.TRADING)
            self._store_metric('open_positions', 12, 'positions', MetricType.TRADING)
            self._store_metric('unreconciled_trades', 0, 'trades', MetricType.TRADING)
            
            logger.debug("📊 Trading metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Trading metrics collection failed: {e}")
    
    async def _collect_api_metrics(self):
        """Collect API metrics"""
        try:
            # Mock API metrics - in real implementation, get from API clients
            self._store_metric('api_error_rate', 0.02, '%', MetricType.API)
            self._store_metric('api_response_time', 0.8, 'seconds', MetricType.API)
            self._store_metric('api_requests_per_minute', 45, 'requests/min', MetricType.API)
            self._store_metric('websocket_connections', 3, 'connections', MetricType.API)
            self._store_metric('websocket_reconnects', 0, 'reconnects', MetricType.API)
            
            logger.debug("📊 API metrics collected")
            
        except Exception as e:
            logger.error(f"❌ API metrics collection failed: {e}")
    
    async def _collect_database_metrics(self):
        """Collect database metrics"""
        try:
            # Mock database metrics - in real implementation, get from database
            self._store_metric('db_connections', 5, 'connections', MetricType.DATABASE)
            self._store_metric('db_query_time', 0.05, 'seconds', MetricType.DATABASE)
            self._store_metric('db_size_mb', 125.5, 'MB', MetricType.DATABASE)
            self._store_metric('db_backup_status', 1, 'status', MetricType.DATABASE)  # 1 = success
            
            logger.debug("📊 Database metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Database metrics collection failed: {e}")
    
    async def _collect_risk_metrics(self):
        """Collect risk metrics"""
        try:
            # Mock risk metrics - in real implementation, get from risk engine
            self._store_metric('exposure_percent', 0.45, '%', MetricType.RISK)
            self._store_metric('max_drawdown', 0.02, '%', MetricType.RISK)
            self._store_metric('var_95', 1500.0, '₹', MetricType.RISK)
            self._store_metric('consecutive_losses', 2, 'losses', MetricType.RISK)
            self._store_metric('circuit_breaker_status', 0, 'status', MetricType.RISK)  # 0 = normal
            
            logger.debug("📊 Risk metrics collected")
            
        except Exception as e:
            logger.error(f"❌ Risk metrics collection failed: {e}")
    
    def _store_metric(self, name: str, value: float, unit: str, metric_type: MetricType):
        """Store metric"""
        metric = Metric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            metric_type=metric_type,
            threshold_warning=self.thresholds.get(name, {}).get('warning'),
            threshold_critical=self.thresholds.get(name, {}).get('critical')
        )
        
        self.metrics[name] = metric
    
    async def _check_thresholds(self):
        """Check metric thresholds and generate alerts"""
        try:
            for name, metric in self.metrics.items():
                if metric.threshold_warning is None and metric.threshold_critical is None:
                    continue
                
                # Check critical threshold
                if metric.threshold_critical is not None and metric.value >= metric.threshold_critical:
                    await self._create_alert(name, metric.value, metric.threshold_critical, AlertLevel.CRITICAL)
                
                # Check warning threshold
                elif metric.threshold_warning is not None and metric.value >= metric.threshold_warning:
                    await self._create_alert(name, metric.value, metric.threshold_warning, AlertLevel.WARNING)
                
        except Exception as e:
            logger.error(f"❌ Threshold checking failed: {e}")
    
    async def _create_alert(self, metric_name: str, current_value: float, threshold_value: float, level: AlertLevel):
        """Create alert"""
        try:
            # Check if alert already exists and is not resolved
            existing_alert = None
            for alert in self.alerts:
                if (alert.metric_name == metric_name and 
                    alert.level == level and 
                    not alert.resolved and
                    (datetime.now() - alert.timestamp).total_seconds() < 300):  # 5 minutes
                    existing_alert = alert
                    break
            
            if existing_alert:
                return  # Alert already exists
            
            alert_id = f"ALERT_{int(time.time())}"
            message = f"{metric_name} is {current_value:.2f} (threshold: {threshold_value:.2f})"
            
            alert = Alert(
                id=alert_id,
                level=level,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                message=message,
                timestamp=datetime.now()
            )
            
            self.alerts.append(alert)
            logger.warning(f"🚨 Alert created: {message}")
            
        except Exception as e:
            logger.error(f"❌ Alert creation failed: {e}")
    
    async def _process_alerts(self):
        """Process and send alerts"""
        try:
            for alert in self.alerts:
                if not alert.acknowledged and not alert.resolved:
                    await self._send_alert(alert)
                    alert.acknowledged = True
            
            # Clean up old resolved alerts
            self.alerts = [alert for alert in self.alerts 
                          if not alert.resolved or 
                          (datetime.now() - alert.timestamp).total_seconds() < 86400]  # Keep for 24 hours
            
        except Exception as e:
            logger.error(f"❌ Alert processing failed: {e}")
    
    async def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        try:
            if self.alert_channels['email']:
                await self._send_email_alert(alert)
            
            if self.alert_channels['telegram']:
                await self._send_telegram_alert(alert)
            
            if self.alert_channels['slack']:
                await self._send_slack_alert(alert)
            
            if self.alert_channels['webhook']:
                await self._send_webhook_alert(alert)
            
        except Exception as e:
            logger.error(f"❌ Alert sending failed: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        # Mock implementation
        logger.info(f"📧 Email alert sent: {alert.message}")
    
    async def _send_telegram_alert(self, alert: Alert):
        """Send Telegram alert"""
        # Mock implementation
        logger.info(f"📱 Telegram alert sent: {alert.message}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        # Mock implementation
        logger.info(f"💬 Slack alert sent: {alert.message}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        # Mock implementation
        logger.info(f"🔗 Webhook alert sent: {alert.message}")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        logger.info("🛑 Production monitoring stopped")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'total_metrics': len(self.metrics),
            'total_alerts': len(self.alerts),
            'active_alerts': len([a for a in self.alerts if not a.resolved]),
            'critical_alerts': len([a for a in self.alerts if a.level == AlertLevel.CRITICAL and not a.resolved]),
            'warning_alerts': len([a for a in self.alerts if a.level == AlertLevel.WARNING and not a.resolved]),
            'metrics': {
                name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'timestamp': metric.timestamp.isoformat(),
                    'type': metric.metric_type.value
                }
                for name, metric in self.metrics.items()
            },
            'alerts': [
                {
                    'id': alert.id,
                    'level': alert.level.value,
                    'metric_name': alert.metric_name,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'acknowledged': alert.acknowledged,
                    'resolved': alert.resolved
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ]
        }
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"✅ Alert acknowledged: {alert_id}")
                break
    
    def resolve_alert(self, alert_id: str):
        """Resolve alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"✅ Alert resolved: {alert_id}")
                break

def main():
    """Main function for testing"""
    async def test_monitoring():
        monitor = ProductionMonitor()
        
        try:
            # Start monitoring
            monitoring_task = asyncio.create_task(monitor.start_monitoring())
            
            # Let it run for a bit
            await asyncio.sleep(65)  # Let it collect metrics and check thresholds
            
            # Get status
            status = monitor.get_monitoring_status()
            print(f"📊 Monitoring status: {json.dumps(status, indent=2, default=str)}")
            
        finally:
            monitor.stop_monitoring()
            monitoring_task.cancel()
    
    # Run test
    asyncio.run(test_monitoring())

if __name__ == "__main__":
    main()
