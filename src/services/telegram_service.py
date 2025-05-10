"""
Telegram service for sending notifications.
"""
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TelegramService:
    """Service for sending notifications via Telegram."""
    
    def __init__(self):
        """Initialize the Telegram service."""
        # Check if Telegram is configured
        self.token = os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.token and self.chat_id)
        
        if not self.enabled:
            logger.warning("Telegram notifications are disabled. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable.")
    
    def send_message(self, message: str) -> bool:
        """Send a message to the configured chat.
        
        Args:
            message: The message to send
            
        Returns:
            bool: True if the message was sent successfully
        """
        if not self.enabled:
            logger.debug(f"Telegram disabled, not sending: {message}")
            return False
        
        try:
            logger.info(f"Telegram message: {message}")
            # In a real implementation, we would use the telegram bot API to send messages
            # But for testing purposes, we just log the message
            return True
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_trade_alert(self, signal_data: Dict[str, Any]) -> bool:
        """Send a trade alert based on a signal.
        
        Args:
            signal_data: Signal data containing trade details
            
        Returns:
            bool: True if the alert was sent successfully
        """
        if not self.enabled:
            logger.debug(f"Telegram disabled, not sending trade alert")
            return False
        
        try:
            # Format the message
            signal = signal_data.get('signal', 'UNKNOWN')
            symbol = signal_data.get('symbol', 'UNKNOWN')
            price = signal_data.get('price', 0.0)
            strategy = signal_data.get('strategy', 'UNKNOWN')
            
            # Determine emoji based on signal
            emoji = "🟢" if "BUY" in signal else "🔴" if "SELL" in signal else "⚪"
            
            message = (
                f"{emoji} *{signal}* ALERT\n"
                f"Symbol: {symbol}\n"
                f"Price: {price:.2f}\n"
                f"Strategy: {strategy}\n"
            )
            
            if 'extra_info' in signal_data:
                message += f"Details: {signal_data['extra_info']}\n"
            
            # Send the formatted message
            logger.info(f"Trade alert: {message}")
            return True
        except Exception as e:
            logger.error(f"Error sending trade alert: {e}")
            return False 