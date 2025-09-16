#!/usr/bin/env python3
"""
Monitoring & Alerting System
Real-time alerts for trades, risk, and system health
"""

import sys
import os
import time
import logging
import asyncio
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    TRADE_PLACED = "TRADE_PLACED"
    TRADE_FILLED = "TRADE_FILLED"
    TRADE_CANCELLED = "TRADE_CANCELLED"
    TRADE_FAILED = "TRADE_FAILED"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    CIRCUIT_BREAKER_ACTIVATED = "CIRCUIT_BREAKER_ACTIVATED"
    WEBSOCKET_DISCONNECTED = "WEBSOCKET_DISCONNECTED"
    API_ERROR = "API_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    DAILY_PNL_THRESHOLD = "DAILY_PNL_THRESHOLD"
    POSITION_RECONCILIATION_FAILED = "POSITION_RECONCILIATION_FAILED"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: AlertType
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class AlertConfig:
    """Alert configuration"""
    enable_email: bool = True
    enable_telegram: bool = True
    enable_slack: bool = False
    enable_webhook: bool = False
    
    # Email configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    
    # Telegram configuration
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # Slack configuration
    slack_webhook_url: str = ""
    
    # Webhook configuration
    webhook_url: str = ""
    
    # Alert thresholds
    daily_pnl_threshold: float = 1000.0
    risk_threshold: float = 0.8
    error_threshold: int = 5

class AlertManager:
    """Advanced alerting system with multiple channels"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.alerts = []
        self.alert_history = []
        self.rate_limits = {}
        self.last_alert_times = {}
        
    async def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        try:
            # Check rate limiting
            if self._is_rate_limited(alert):
                logger.warning(f"⚠️ Alert rate limited: {alert.alert_type.value}")
                return
            
            # Add to alerts list
            self.alerts.append(alert)
            self.alert_history.append(alert)
            
            # Send through different channels
            tasks = []
            
            if self.config.enable_email:
                tasks.append(self._send_email_alert(alert))
            
            if self.config.enable_telegram:
                tasks.append(self._send_telegram_alert(alert))
            
            if self.config.enable_slack:
                tasks.append(self._send_slack_alert(alert))
            
            if self.config.enable_webhook:
                tasks.append(self._send_webhook_alert(alert))
            
            # Execute all alert tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update rate limiting
            self._update_rate_limit(alert)
            
            logger.info(f"📢 Alert sent: {alert.alert_type.value} - {alert.title}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send alert: {e}")
    
    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert is rate limited"""
        alert_key = f"{alert.alert_type.value}_{alert.level.value}"
        current_time = datetime.now()
        
        if alert_key in self.last_alert_times:
            last_time = self.last_alert_times[alert_key]
            time_diff = (current_time - last_time).total_seconds()
            
            # Rate limit based on alert level
            if alert.level == AlertLevel.CRITICAL:
                min_interval = 60  # 1 minute
            elif alert.level == AlertLevel.ERROR:
                min_interval = 300  # 5 minutes
            elif alert.level == AlertLevel.WARNING:
                min_interval = 900  # 15 minutes
            else:
                min_interval = 1800  # 30 minutes
            
            if time_diff < min_interval:
                return True
        
        return False
    
    def _update_rate_limit(self, alert: Alert):
        """Update rate limiting for alert"""
        alert_key = f"{alert.alert_type.value}_{alert.level.value}"
        self.last_alert_times[alert_key] = datetime.now()
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            if not self.config.email_username or not self.config.email_password:
                logger.warning("⚠️ Email credentials not configured")
                return
            
            # Create email message
            subject = f"[{alert.level.value}] {alert.title}"
            body = f"""
Alert Type: {alert.alert_type.value}
Level: {alert.level.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert.message}

Data: {json.dumps(alert.data, indent=2, default=str)}
"""
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            for recipient in self.config.email_recipients or []:
                message = f"Subject: {subject}\n\n{body}"
                server.sendmail(self.config.email_username, recipient, message)
            
            server.quit()
            logger.info(f"📧 Email alert sent to {len(self.config.email_recipients or [])} recipients")
            
        except Exception as e:
            logger.error(f"❌ Failed to send email alert: {e}")
    
    async def _send_telegram_alert(self, alert: Alert):
        """Send Telegram alert"""
        try:
            if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
                logger.warning("⚠️ Telegram credentials not configured")
                return
            
            # Create message
            emoji_map = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨"
            }
            
            emoji = emoji_map.get(alert.level, "📢")
            message = f"{emoji} *{alert.title}*\n\n{alert.message}\n\nTime: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.config.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
            logger.info("📱 Telegram alert sent")
            
        except Exception as e:
            logger.error(f"❌ Failed to send Telegram alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            if not self.config.slack_webhook_url:
                logger.warning("⚠️ Slack webhook not configured")
                return
            
            # Create Slack message
            color_map = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning",
                AlertLevel.ERROR: "danger",
                AlertLevel.CRITICAL: "danger"
            }
            
            color = color_map.get(alert.level, "good")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Alert Type",
                                "value": alert.alert_type.value,
                                "short": True
                            },
                            {
                                "title": "Level",
                                "value": alert.level.value,
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "Trading System Alert",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(self.config.slack_webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("💬 Slack alert sent")
            
        except Exception as e:
            logger.error(f"❌ Failed to send Slack alert: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        try:
            if not self.config.webhook_url:
                logger.warning("⚠️ Webhook URL not configured")
                return
            
            # Create webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "data": alert.data
            }
            
            # Send webhook
            response = requests.post(self.config.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("🔗 Webhook alert sent")
            
        except Exception as e:
            logger.error(f"❌ Failed to send webhook alert: {e}")
    
    def create_alert(self, alert_type: AlertType, level: AlertLevel, title: str, 
                    message: str, data: Dict[str, Any] = None) -> Alert:
        """Create new alert"""
        alert_id = f"{alert_type.value}_{int(time.time())}"
        
        return Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            level=level,
            title=title,
            message=message,
            timestamp=datetime.now(),
            data=data or {}
        )
    
    async def alert_trade_placed(self, symbol: str, side: str, quantity: float, price: float):
        """Alert for trade placement"""
        alert = self.create_alert(
            AlertType.TRADE_PLACED,
            AlertLevel.INFO,
            "Trade Placed",
            f"Trade placed: {symbol} {side} {quantity} @ ₹{price:,.2f}",
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price
            }
        )
        await self.send_alert(alert)
    
    async def alert_trade_filled(self, symbol: str, side: str, quantity: float, price: float, pnl: float = None):
        """Alert for trade fill"""
        level = AlertLevel.INFO
        if pnl is not None:
            level = AlertLevel.WARNING if pnl < 0 else AlertLevel.INFO
        
        alert = self.create_alert(
            AlertType.TRADE_FILLED,
            level,
            "Trade Filled",
            f"Trade filled: {symbol} {side} {quantity} @ ₹{price:,.2f}" + (f" (P&L: ₹{pnl:,.2f})" if pnl is not None else ""),
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "pnl": pnl
            }
        )
        await self.send_alert(alert)
    
    async def alert_risk_limit_exceeded(self, limit_type: str, current_value: float, limit_value: float):
        """Alert for risk limit exceeded"""
        alert = self.create_alert(
            AlertType.RISK_LIMIT_EXCEEDED,
            AlertLevel.WARNING,
            "Risk Limit Exceeded",
            f"Risk limit exceeded: {limit_type} = {current_value:.2%} (limit: {limit_value:.2%})",
            {
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value
            }
        )
        await self.send_alert(alert)
    
    async def alert_circuit_breaker_activated(self, reason: str):
        """Alert for circuit breaker activation"""
        alert = self.create_alert(
            AlertType.CIRCUIT_BREAKER_ACTIVATED,
            AlertLevel.CRITICAL,
            "Circuit Breaker Activated",
            f"Circuit breaker activated: {reason}",
            {"reason": reason}
        )
        await self.send_alert(alert)
    
    async def alert_websocket_disconnected(self, symbol: str, duration: int):
        """Alert for WebSocket disconnection"""
        alert = self.create_alert(
            AlertType.WEBSOCKET_DISCONNECTED,
            AlertLevel.ERROR,
            "WebSocket Disconnected",
            f"WebSocket disconnected for {symbol} (duration: {duration}s)",
            {
                "symbol": symbol,
                "duration": duration
            }
        )
        await self.send_alert(alert)
    
    async def alert_daily_pnl_threshold(self, pnl: float, threshold: float):
        """Alert for daily P&L threshold"""
        level = AlertLevel.WARNING if pnl < 0 else AlertLevel.INFO
        alert = self.create_alert(
            AlertType.DAILY_PNL_THRESHOLD,
            level,
            "Daily P&L Threshold",
            f"Daily P&L threshold reached: ₹{pnl:,.2f} (threshold: ₹{threshold:,.2f})",
            {
                "pnl": pnl,
                "threshold": threshold
            }
        )
        await self.send_alert(alert)
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
        
        summary = {
            "total_alerts": len(recent_alerts),
            "by_level": {},
            "by_type": {},
            "recent_alerts": recent_alerts[-10:]  # Last 10 alerts
        }
        
        # Count by level
        for alert in recent_alerts:
            level = alert.level.value
            summary["by_level"][level] = summary["by_level"].get(level, 0) + 1
        
        # Count by type
        for alert in recent_alerts:
            alert_type = alert.alert_type.value
            summary["by_type"][alert_type] = summary["by_type"].get(alert_type, 0) + 1
        
        return summary

class SystemMonitor:
    """System monitoring with health checks and alerts"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.health_checks = {}
        self.last_health_check = None
        
    async def check_system_health(self):
        """Perform comprehensive system health check"""
        try:
            health_status = {
                "timestamp": datetime.now(),
                "overall_status": "HEALTHY",
                "checks": {}
            }
            
            # Check WebSocket connections
            websocket_status = await self._check_websocket_health()
            health_status["checks"]["websocket"] = websocket_status
            
            # Check API connectivity
            api_status = await self._check_api_health()
            health_status["checks"]["api"] = api_status
            
            # Check database connectivity
            db_status = await self._check_database_health()
            health_status["checks"]["database"] = db_status
            
            # Check system resources
            resource_status = await self._check_resource_health()
            health_status["checks"]["resources"] = resource_status
            
            # Determine overall status
            failed_checks = [name for name, status in health_status["checks"].items() if not status["healthy"]]
            if failed_checks:
                health_status["overall_status"] = "UNHEALTHY"
                
                # Send alert for system health issues
                alert = self.alert_manager.create_alert(
                    AlertType.SYSTEM_ERROR,
                    AlertLevel.ERROR,
                    "System Health Check Failed",
                    f"System health check failed: {', '.join(failed_checks)}",
                    {"failed_checks": failed_checks, "health_status": health_status}
                )
                await self.alert_manager.send_alert(alert)
            
            self.last_health_check = health_status
            return health_status
            
        except Exception as e:
            logger.error(f"❌ System health check failed: {e}")
            return {"overall_status": "ERROR", "error": str(e)}
    
    async def _check_websocket_health(self) -> Dict[str, Any]:
        """Check WebSocket health"""
        try:
            # In real implementation, check actual WebSocket connections
            return {
                "healthy": True,
                "connected_sockets": 3,
                "total_sockets": 3,
                "last_update": datetime.now()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now()
            }
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health"""
        try:
            # In real implementation, check actual API endpoints
            return {
                "healthy": True,
                "response_time": 0.5,
                "last_check": datetime.now()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now()
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            # In real implementation, check actual database connection
            return {
                "healthy": True,
                "connection_pool": "active",
                "last_check": datetime.now()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now()
            }
    
    async def _check_resource_health(self) -> Dict[str, Any]:
        """Check system resource health"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            healthy = cpu_percent < 80 and memory_percent < 85 and disk_percent < 90
            
            return {
                "healthy": healthy,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "last_check": datetime.now()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now()
            }

def main():
    """Main function for testing"""
    async def test_alerting_system():
        # Create alert configuration
        config = AlertConfig(
            enable_email=False,  # Disable for testing
            enable_telegram=False,  # Disable for testing
            enable_slack=False,  # Disable for testing
            enable_webhook=False  # Disable for testing
        )
        
        # Create alert manager
        alert_manager = AlertManager(config)
        
        # Test different alert types
        await alert_manager.alert_trade_placed("NSE:NIFTY50-INDEX", "BUY", 100, 19500)
        await alert_manager.alert_trade_filled("NSE:NIFTY50-INDEX", "BUY", 100, 19500, 500)
        await alert_manager.alert_risk_limit_exceeded("portfolio_exposure", 0.85, 0.8)
        await alert_manager.alert_circuit_breaker_activated("Daily loss limit exceeded")
        
        # Test system monitoring
        system_monitor = SystemMonitor(alert_manager)
        health_status = await system_monitor.check_system_health()
        print(f"System health: {health_status}")
        
        # Get alert summary
        summary = alert_manager.get_alert_summary(1)
        print(f"Alert summary: {summary}")
    
    # Run test
    asyncio.run(test_alerting_system())

if __name__ == "__main__":
    main()
