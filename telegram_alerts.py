#!/usr/bin/env python3
"""
Telegram Alerts - Critical for Live Trading Monitoring
Get notified instantly if trades fail or data breaks
"""

def add_telegram_alerts():
    """Add Telegram alerts to live_paper_trading.py"""
    
    
    with open('live_paper_trading.py', 'r') as f:
        content = f.read()
    
    # Add requests import
    import re
    content = re.sub(
        r'(import json)',
        r'\1\nimport requests',
        content
    )
    
    # Add the alert method
    alert_method = '''
    def _send_alert(self, message: str, level: str = "INFO"):
        """Send alert via Telegram - critical for live trading monitoring."""
        try:
            # Get Telegram configuration from environment
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            
            if not bot_token or not chat_id:
                logger.debug("⚠️ Telegram not configured - skipping alert")
                return
            
            # Format message with timestamp and level
            timestamp = self.now_kolkata().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"🤖 **Trading Bot Alert**\\n"
            formatted_message += f"⏰ Time: {timestamp}\\n"
            formatted_message += f"📊 Level: {level}\\n"
            formatted_message += f"💬 Message: {message}"
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": formatted_message,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug(f"✅ Alert sent: {message[:50]}...")
            else:
                logger.error(f"❌ Failed to send alert: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error sending alert: {e}")
    '''
    
    # Insert before _validate_production_requirements
    content = re.sub(
        r'(def _validate_production_requirements\(self\):)',
        alert_method + r'\n    \1',
        content
    )
    
    # Add alerts to critical points
    # Trade opened alert
    content = re.sub(
        r'(logger\.info\(f"✅ Opened trade {trade_id\[:8\]}\.\.\. \| {signal\[\'signal\'\]} {signal\[\'strategy\'\]}"\))',
        r'\1\n                \n                # Send trade alert\n                self._send_alert(f"✅ Trade Opened: {signal[\'strategy\']} {signal[\'signal\']} | {signal[\'symbol\']}")',
        content
    )
    
    # Trade closed alert
    content = re.sub(
        r'(logger\.info\(f"🔒 Closed paper trade: {trade_id} \| P&L: ₹{pnl:\+\.2f} \({returns_pct:\+\.2f}%\)"\))',
        r'\1\n                \n                # Send trade alert\n                self._send_alert(f"🔒 Trade Closed: {trade.strategy} | P&L: ₹{pnl:+.2f} ({returns_pct:+.2f}%)")',
        content
    )
    
    # Daily loss limit alert
    content = re.sub(
        r'(self\.daily_loss_limit_hit = True)',
        r'\1\n                \n                # Send critical alert\n                self._send_alert(f"🚨 CRITICAL: Daily loss limit breached! P&L: ₹{self.daily_pnl:.2f}", "CRITICAL")',
        content
    )
    
    with open('live_paper_trading.py', 'w') as f:
        f.write(content)
    

if __name__ == "__main__":
    add_telegram_alerts()
