#!/usr/bin/env python3
"""
Live Health Dashboard
FastAPI endpoint exposing key metrics (PnL, exposure, trade count, API errors)
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, asdict
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

@dataclass
class HealthMetrics:
    """Health metrics data structure"""
    timestamp: str
    system_status: str
    financial_metrics: Dict
    trading_metrics: Dict
    performance_metrics: Dict
    api_health: Dict
    alerts: List[Dict]
    system_resources: Dict

class LiveHealthDashboard:
    """Live health dashboard with FastAPI endpoints"""
    
    def __init__(self, trading_system, port: int = 8000):
        self.trading_system = trading_system
        self.port = port
        self.app = None
        self.metrics_cache = {}
        self.is_monitoring = False
        self.monitor_thread = None
        
        if FASTAPI_AVAILABLE:
            self._create_fastapi_app()
        else:
            logger.error("FastAPI not available. Cannot create dashboard.")
    
    def _create_fastapi_app(self):
        """Create FastAPI application with endpoints"""
        self.app = FastAPI(
            title="Trading System Health Dashboard",
            description="Real-time monitoring of trading system metrics",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Define endpoints
        @self.app.get("/")
        async def root():
            return {"message": "Trading System Health Dashboard", "status": "running"}
        
        @self.app.get("/health")
        async def health():
            """Get system health status"""
            try:
                metrics = self._get_current_metrics()
                return metrics
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def metrics():
            """Get detailed metrics"""
            try:
                return self._get_detailed_metrics()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/pnl")
        async def pnl():
            """Get PnL metrics"""
            try:
                return self._get_pnl_metrics()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/trades")
        async def trades():
            """Get trade metrics"""
            try:
                return self._get_trade_metrics()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/alerts")
        async def alerts():
            """Get active alerts"""
            try:
                return self._get_active_alerts()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            """Get HTML dashboard"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/performance")
        async def performance():
            """Get performance metrics"""
            try:
                return self._get_performance_metrics()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/exposure")
        async def exposure():
            """Get exposure metrics"""
            try:
                return self._get_exposure_metrics()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        logger.info("✅ FastAPI dashboard created with endpoints")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.app:
            logger.error("FastAPI app not created. Cannot start monitoring.")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("📊 Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update metrics cache
                self.metrics_cache = self._get_current_metrics()
                time.sleep(1)  # Update every second
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _get_current_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            current_time = self.trading_system.now_kolkata()
            equity = self.trading_system._equity({})
            exposure = self.trading_system._current_total_exposure({})
            
            # Financial metrics
            financial_metrics = {
                'initial_capital': self.trading_system.initial_capital,
                'current_cash': self.trading_system.cash,
                'current_equity': equity,
                'total_return_pct': ((equity - self.trading_system.initial_capital) / self.trading_system.initial_capital) * 100,
                'daily_pnl': self.trading_system.daily_pnl,
                'max_drawdown': self.trading_system.max_drawdown,
                'current_exposure': exposure,
                'exposure_limit': self.trading_system.exposure_limit
            }
            
            # Trading metrics
            trading_metrics = {
                'open_trades': len(self.trading_system.open_trades),
                'closed_trades': len(self.trading_system.closed_trades),
                'total_signals_generated': self.trading_system.total_signals_generated,
                'total_signals_rejected': self.trading_system.total_signals_rejected,
                'total_trades_executed': self.trading_system.total_trades_executed,
                'total_trades_closed': self.trading_system.total_trades_closed,
                'winning_trades': self.trading_system.winning_trades,
                'losing_trades': self.trading_system.losing_trades,
                'win_rate': (self.trading_system.winning_trades / max(self.trading_system.total_trades_closed, 1)) * 100
            }
            
            # Performance metrics
            performance_metrics = self.trading_system.performance_stats.copy()
            
            # API health
            api_health = {
                'api_calls_made': performance_metrics.get('api_calls_made', 0),
                'api_failures': performance_metrics.get('api_failures', 0),
                'api_retries': performance_metrics.get('api_retries', 0),
                'failure_rate': (performance_metrics.get('api_failures', 0) / max(performance_metrics.get('api_calls_made', 1), 1)) * 100,
                'cache_hit_rate': (performance_metrics.get('cache_hits', 0) / max(performance_metrics.get('cache_hits', 0) + performance_metrics.get('cache_misses', 0), 1)) * 100
            }
            
            # System status
            system_status = {
                'is_running': self.trading_system.is_running,
                'market_open': self.trading_system._is_market_open(current_time),
                'health_score': self._calculate_health_score(financial_metrics, trading_metrics, api_health),
                'uptime_seconds': (current_time - self.trading_system.session_start).total_seconds() if self.trading_system.session_start else 0
            }
            
            # Active alerts
            alerts = self._get_active_alerts_list(financial_metrics, trading_metrics, api_health)
            
            # System resources
            system_resources = self._get_system_resources()
            
            return {
                'timestamp': current_time.isoformat(),
                'system_status': system_status,
                'financial_metrics': financial_metrics,
                'trading_metrics': trading_metrics,
                'performance_metrics': performance_metrics,
                'api_health': api_health,
                'alerts': alerts,
                'system_resources': system_resources
            }
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}
    
    def _calculate_health_score(self, financial_metrics: Dict, trading_metrics: Dict, api_health: Dict) -> int:
        """Calculate overall system health score (0-100)"""
        try:
            score = 100
            
            # Deduct for high exposure
            if financial_metrics['current_exposure'] > 0.8:
                score -= 20
            elif financial_metrics['current_exposure'] > 0.6:
                score -= 10
            
            # Deduct for high drawdown
            if financial_metrics['max_drawdown'] > 0.05:
                score -= 15
            elif financial_metrics['max_drawdown'] > 0.03:
                score -= 10
            
            # Deduct for API failures
            if api_health['failure_rate'] > 0.1:
                score -= 20
            elif api_health['failure_rate'] > 0.05:
                score -= 10
            
            # Deduct for daily loss
            if financial_metrics['daily_pnl'] < -(financial_metrics['initial_capital'] * 0.02):
                score -= 25
            
            return max(0, score)
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 0
    
    def _get_active_alerts_list(self, financial_metrics: Dict, trading_metrics: Dict, api_health: Dict) -> List[Dict]:
        """Get list of active alerts"""
        alerts = []
        
        try:
            # High exposure alert
            if financial_metrics['current_exposure'] > 0.7:
                alerts.append({
                    'type': 'HIGH_EXPOSURE',
                    'severity': 'WARNING',
                    'message': f"Exposure {financial_metrics['current_exposure']:.1%} is high",
                    'timestamp': datetime.now().isoformat()
                })
            
            # High drawdown alert
            if financial_metrics['max_drawdown'] > 0.03:
                alerts.append({
                    'type': 'HIGH_DRAWDOWN',
                    'severity': 'WARNING',
                    'message': f"Max drawdown {financial_metrics['max_drawdown']:.1%}",
                    'timestamp': datetime.now().isoformat()
                })
            
            # API failure alert
            if api_health['failure_rate'] > 0.1:
                alerts.append({
                    'type': 'API_FAILURES',
                    'severity': 'CRITICAL',
                    'message': f"API failure rate {api_health['failure_rate']:.1%}",
                    'timestamp': datetime.now().isoformat()
                })
            
            # Daily loss alert
            if financial_metrics['daily_pnl'] < -(financial_metrics['initial_capital'] * 0.01):
                alerts.append({
                    'type': 'DAILY_LOSS',
                    'severity': 'CRITICAL',
                    'message': f"Daily loss ₹{abs(financial_metrics['daily_pnl']):,.2f}",
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
        
        return alerts
    
    def _get_system_resources(self) -> Dict:
        """Get system resource usage"""
        try:
            import psutil
            
            process = psutil.Process()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'threads': process.num_threads(),
                'uptime_seconds': (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {}
    
    def _get_detailed_metrics(self) -> Dict:
        """Get detailed metrics"""
        return self.metrics_cache or self._get_current_metrics()
    
    def _get_pnl_metrics(self) -> Dict:
        """Get PnL-specific metrics"""
        metrics = self._get_current_metrics()
        return {
            'financial_metrics': metrics.get('financial_metrics', {}),
            'trading_metrics': metrics.get('trading_metrics', {}),
            'timestamp': metrics.get('timestamp', '')
        }
    
    def _get_trade_metrics(self) -> Dict:
        """Get trade-specific metrics"""
        metrics = self._get_current_metrics()
        return {
            'trading_metrics': metrics.get('trading_metrics', {}),
            'performance_metrics': metrics.get('performance_metrics', {}),
            'timestamp': metrics.get('timestamp', '')
        }
    
    def _get_active_alerts(self) -> Dict:
        """Get active alerts"""
        metrics = self._get_current_metrics()
        return {
            'alerts': metrics.get('alerts', []),
            'timestamp': metrics.get('timestamp', '')
        }
    
    def _get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        metrics = self._get_current_metrics()
        return {
            'performance_metrics': metrics.get('performance_metrics', {}),
            'api_health': metrics.get('api_health', {}),
            'system_resources': metrics.get('system_resources', {}),
            'timestamp': metrics.get('timestamp', '')
        }
    
    def _get_exposure_metrics(self) -> Dict:
        """Get exposure metrics"""
        metrics = self._get_current_metrics()
        return {
            'financial_metrics': {
                'current_exposure': metrics.get('financial_metrics', {}).get('current_exposure', 0),
                'exposure_limit': metrics.get('financial_metrics', {}).get('exposure_limit', 0),
                'current_equity': metrics.get('financial_metrics', {}).get('current_equity', 0),
                'current_cash': metrics.get('financial_metrics', {}).get('current_cash', 0)
            },
            'timestamp': metrics.get('timestamp', '')
        }
    
    def _get_dashboard_html(self) -> str:
        """Get HTML dashboard"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading System Health Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .metric-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
                .metric-label { color: #666; font-size: 14px; }
                .status-good { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-critical { color: #e74c3c; }
                .alerts { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }
                .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }
                .alert-warning { background: #fff3cd; border-left: 4px solid #f39c12; }
                .alert-critical { background: #f8d7da; border-left: 4px solid #e74c3c; }
                .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 3px; cursor: pointer; }
                .refresh-btn:hover { background: #2980b9; }
            </style>
            <script>
                function refreshData() {
                    fetch('/health')
                        .then(response => response.json())
                        .then(data => {
                            updateDashboard(data);
                        });
                }
                
                function updateDashboard(data) {
                    // Update timestamp
                    document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();
                    
                    // Update financial metrics
                    document.getElementById('equity').textContent = '₹' + data.financial_metrics.current_equity.toLocaleString();
                    document.getElementById('return').textContent = data.financial_metrics.total_return_pct.toFixed(2) + '%';
                    document.getElementById('exposure').textContent = (data.financial_metrics.current_exposure * 100).toFixed(1) + '%';
                    document.getElementById('drawdown').textContent = data.financial_metrics.max_drawdown.toFixed(2) + '%';
                    
                    // Update trading metrics
                    document.getElementById('open-trades').textContent = data.trading_metrics.open_trades;
                    document.getElementById('closed-trades').textContent = data.trading_metrics.closed_trades;
                    document.getElementById('win-rate').textContent = data.trading_metrics.win_rate.toFixed(1) + '%';
                    
                    // Update system status
                    const healthStatus = document.getElementById('health-status');
                    const healthScore = data.system_status.health_score;
                    healthStatus.textContent = healthScore + '/100';
                    if (healthScore >= 80) {
                        healthStatus.className = 'status-good';
                    } else if (healthScore >= 60) {
                        healthStatus.className = 'status-warning';
                    } else {
                        healthStatus.className = 'status-critical';
                    }
                    
                    // Update alerts
                    const alertsContainer = document.getElementById('alerts');
                    alertsContainer.innerHTML = '';
                    if (data.alerts.length === 0) {
                        alertsContainer.innerHTML = '<p>No active alerts</p>';
                    } else {
                        data.alerts.forEach(alert => {
                            const alertDiv = document.createElement('div');
                            alertDiv.className = 'alert alert-' + alert.severity.toLowerCase();
                            alertDiv.innerHTML = '<strong>' + alert.type + ':</strong> ' + alert.message;
                            alertsContainer.appendChild(alertDiv);
                        });
                    }
                }
                
                // Auto-refresh every 5 seconds
                setInterval(refreshData, 5000);
                
                // Initial load
                window.onload = refreshData;
            </script>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 Trading System Health Dashboard</h1>
                    <p>Last Updated: <span id="timestamp">Loading...</span></p>
                    <button class="refresh-btn" onclick="refreshData()">Refresh</button>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">System Health</div>
                        <div class="metric-value" id="health-status">Loading...</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Current Equity</div>
                        <div class="metric-value" id="equity">Loading...</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value" id="return">Loading...</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Current Exposure</div>
                        <div class="metric-value" id="exposure">Loading...</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value" id="drawdown">Loading...</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Open Trades</div>
                        <div class="metric-value" id="open-trades">Loading...</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Closed Trades</div>
                        <div class="metric-value" id="closed-trades">Loading...</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value" id="win-rate">Loading...</div>
                    </div>
                </div>
                
                <div class="alerts">
                    <h3>🚨 Active Alerts</h3>
                    <div id="alerts">Loading...</div>
                </div>
            </div>
        </body>
        </html>
        """
    
    def run_dashboard(self):
        """Run the dashboard server"""
        if not self.app:
            logger.error("FastAPI app not created. Cannot run dashboard.")
            return
        
        try:
            logger.info(f"🌐 Starting health dashboard on port {self.port}")
            logger.info(f"📊 Dashboard available at: http://localhost:{self.port}")
            logger.info(f"📊 Health endpoint: http://localhost:{self.port}/health")
            logger.info(f"📊 Metrics endpoint: http://localhost:{self.port}/metrics")
            
            # Start monitoring
            self.start_monitoring()
            
            # Run server
            uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")
            
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")

def main():
    """Main function to run health dashboard"""
    try:
        if not FASTAPI_AVAILABLE:
            logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
            return
        
        from live_paper_trading import LivePaperTradingSystem
        
        # Initialize trading system
        logger.info("🚀 Initializing trading system for health dashboard...")
        trading_system = LivePaperTradingSystem(initial_capital=100000)
        
        # Initialize health dashboard
        dashboard = LiveHealthDashboard(trading_system, port=8000)
        
        # Run dashboard
        dashboard.run_dashboard()
        
    except Exception as e:
        logger.error(f"❌ Health dashboard failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
