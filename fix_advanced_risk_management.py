#!/usr/bin/env python3
"""
Fix Advanced Risk Management - Timezone and Thread Safety Issues
"""

import re
import threading
from datetime import datetime

def fix_advanced_risk_management():
    """Fix timezone and thread safety issues in advanced_risk_management.py"""
    
    # Read the file
    with open('src/advanced_systems/advanced_risk_management.py', 'r') as f:
        content = f.read()
    
    # Add timezone imports and thread safety
    content = content.replace(
        'import json\nsys.path.append(os.path.join(os.path.dirname(__file__), \'src\'))',
        '''import json
import threading
from zoneinfo import ZoneInfo
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))'''
    )
    
    # Add timezone manager import
    content = content.replace(
        'from typing import Dict, List, Optional, Any, Tuple',
        '''from typing import Dict, List, Optional, Any, Tuple
from src.core.timezone_utils import timezone_manager, now, now_kolkata'''
    )
    
    # Fix timezone issues - replace datetime.now() with timezone-aware calls
    content = re.sub(
        r'datetime\.now\(\)',
        'now()',
        content
    )
    
    # Add thread safety to the class
    content = content.replace(
        '    def __init__(self, risk_limits: RiskLimits = None):',
        '''    def __init__(self, risk_limits: RiskLimits = None):
        # Thread safety
        self._lock = threading.RLock()
        self.tz = ZoneInfo('Asia/Kolkata')'''
    )
    
    # Add thread safety to critical methods
    methods_to_protect = [
        'add_position', 'update_position_price', 'add_trade_result',
        'check_risk_limits', 'trigger_circuit_breaker', 'reset_daily_metrics'
    ]
    
    for method in methods_to_protect:
        # Add lock protection to method
        pattern = f'(    def {method}\\(self[^:]*:\\):)'
        replacement = f'\\1\n        with self._lock:'
        content = re.sub(pattern, replacement, content)
    
    # Fix specific timezone issues in risk checks
    content = content.replace(
        'self.last_risk_check = datetime.now()',
        'self.last_risk_check = now()'
    )
    
    # Add rate limiting for alerts
    content = content.replace(
        '        self.risk_alerts = []',
        '''        self.risk_alerts = []
        self.alert_timestamps = {}  # For rate limiting
        self.alert_cooldown = 300  # 5 minutes'''
    )
    
    # Add alert rate limiting method
    alert_rate_limiting = '''
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if alert should be sent (rate limiting)"""
        current_time = now()
        last_alert = self.alert_timestamps.get(alert_type)
        
        if last_alert is None:
            self.alert_timestamps[alert_type] = current_time
            return True
        
        time_diff = (current_time - last_alert).total_seconds()
        if time_diff >= self.alert_cooldown:
            self.alert_timestamps[alert_type] = current_time
            return True
        
        return False
    
    def _add_risk_alert(self, alert_type: str, message: str, level: RiskLevel):
        """Add risk alert with rate limiting"""
        if self._should_send_alert(alert_type):
            alert = {
                'timestamp': now(),
                'type': alert_type,
                'message': message,
                'level': level.value
            }
            self.risk_alerts.append(alert)
            logger.warning(f"🚨 Risk Alert: {message}")
    
    def _persist_risk_event(self, event_type: str, data: dict):
        """Persist critical risk events to logs"""
        event = {
            'timestamp': now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        logger.critical(f"RISK_EVENT: {json.dumps(event)}")
'''
    
    # Insert alert rate limiting methods before the main methods
    content = content.replace(
        '    def add_position(self, symbol: str, quantity: float, price: float, timestamp: datetime):',
        alert_rate_limiting + '\n    def add_position(self, symbol: str, quantity: float, price: float, timestamp: datetime):'
    )
    
    # Make risk limits configurable
    content = content.replace(
        '        self.risk_check_interval = 60  # 1 minute',
        '''        self.risk_check_interval = int(os.getenv('RISK_CHECK_INTERVAL', '60'))  # 1 minute
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', '0.05'))
        self.max_portfolio_exposure = float(os.getenv('MAX_PORTFOLIO_EXPOSURE', '0.8'))
        self.max_single_position = float(os.getenv('MAX_SINGLE_POSITION', '0.2'))'''
    )
    
    # Write the fixed file
    with open('src/advanced_systems/advanced_risk_management.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed advanced_risk_management.py")

if __name__ == "__main__":
    fix_advanced_risk_management()
