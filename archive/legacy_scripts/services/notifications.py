"""
Notification service module.
Handles sending notifications to various channels (Telegram, etc.)
"""
import requests
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from src.config.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_FREE_GROUP_ID,
    setup_logging
)

# Set up logger
logger = setup_logging("notifications")

class NotificationProvider(ABC):
    """Base class for notification providers."""
    
    @abstractmethod
    def send_message(self, message: str) -> bool:
        """Send a notification message.
        
        Args:
            message: Message text
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def send_trade_alert(self, trade_data: Dict[str, Any]) -> bool:
        """Send a trade alert notification.
        
        Args:
            trade_data: Trade data
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass


class TelegramNotifier(NotificationProvider):
    """Telegram notification provider."""
    
    def __init__(self, bot_token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
        """Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message: str, chat_id: Optional[str] = None) -> bool:
        """Send a message to Telegram.
        
        Args:
            message: Message text
            chat_id: Override default chat ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        target_chat = chat_id or self.chat_id
        if not target_chat or not self.bot_token:
            logger.error("Missing Telegram credentials")
            return False
        
        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": target_chat,
                "text": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data)
            result = response.json()
            
            if result.get("ok"):
                logger.info(f"Message sent to Telegram: {message[:30]}...")
                return True
            else:
                logger.error(f"Failed to send message to Telegram: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending message to Telegram: {e}")
            return False
    
    def send_trade_alert(self, trade_data: Dict[str, Any], chat_id: Optional[str] = None) -> bool:
        """Send a trade alert to Telegram.
        
        Args:
            trade_data: Trade data
            chat_id: Override default chat ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Format trade alert message
        signal = trade_data.get("signal", "UNKNOWN")
        symbol = trade_data.get("symbol", "UNKNOWN")
        price = trade_data.get("price", 0)
        confidence = trade_data.get("confidence", "Low")
        
        # Get reasons if available
        reasons = []
        if "rsi_reason" in trade_data and trade_data["rsi_reason"]:
            reasons.append(trade_data["rsi_reason"])
        if "macd_reason" in trade_data and trade_data["macd_reason"]:
            reasons.append(trade_data["macd_reason"])
        if "price_reason" in trade_data and trade_data["price_reason"]:
            reasons.append(trade_data["price_reason"])
        
        # Construct message
        message = f"🚨 <b>TRADE ALERT</b> 🚨\n\n"
        message += f"<b>Signal:</b> {signal}\n"
        message += f"<b>Symbol:</b> {symbol}\n"
        message += f"<b>Price:</b> {price}\n"
        message += f"<b>Confidence:</b> {confidence}\n"
        
        # Add stoploss and targets if available
        if "stop_loss" in trade_data:
            message += f"<b>Stop Loss:</b> {trade_data['stop_loss']}\n"
        
        if "target" in trade_data:
            message += f"<b>Target 1:</b> {trade_data['target']}\n"
        
        if "target2" in trade_data:
            message += f"<b>Target 2:</b> {trade_data['target2']}\n"
        
        if "target3" in trade_data:
            message += f"<b>Target 3:</b> {trade_data['target3']}\n"
        
        # Add reasons
        if reasons:
            message += "\n<b>Reasons:</b>\n"
            for reason in reasons:
                message += f"• {reason}\n"
        
        # Send message
        return self.send_message(message, chat_id)
    
    def send_to_free_group(self, message: str) -> bool:
        """Send a message to the free group.
        
        Args:
            message: Message text
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not TELEGRAM_FREE_GROUP_ID:
            logger.warning("Free group ID not set, skipping message")
            return False
            
        return self.send_message(message, TELEGRAM_FREE_GROUP_ID)


class NotificationService:
    """Service for sending notifications through various providers."""
    
    def __init__(self):
        """Initialize the notification service."""
        self.providers: List[NotificationProvider] = []
        
        # Add Telegram provider if credentials are available
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            self.providers.append(TelegramNotifier())
    
    def add_provider(self, provider: NotificationProvider) -> None:
        """Add a notification provider.
        
        Args:
            provider: NotificationProvider instance
        """
        self.providers.append(provider)
    
    def send_message(self, message: str) -> bool:
        """Send a message to all providers.
        
        Args:
            message: Message text
            
        Returns:
            bool: True if all providers succeeded, False otherwise
        """
        if not self.providers:
            logger.warning("No notification providers configured")
            return False
            
        results = []
        for provider in self.providers:
            result = provider.send_message(message)
            results.append(result)
            
        return all(results)
    
    def send_trade_alert(self, trade_data: Dict[str, Any]) -> bool:
        """Send a trade alert to all providers.
        
        Args:
            trade_data: Trade data
            
        Returns:
            bool: True if all providers succeeded, False otherwise
        """
        if not self.providers:
            logger.warning("No notification providers configured")
            return False
            
        results = []
        for provider in self.providers:
            result = provider.send_trade_alert(trade_data)
            results.append(result)
            
        return all(results)

# Create a notification service instance for direct imports
notification_service = NotificationService() 