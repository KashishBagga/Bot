import os
from dotenv import load_dotenv
load_dotenv()
#!/usr/bin/env python3
"""
Risk Management Configuration
============================
Configurable risk management settings for trading systems
"""

import os
from typing import Dict, Any

class RiskConfig:
    """Risk management configuration with optional enforcement"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_default_config()
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default risk management configuration"""
        return {
            # Risk Management Toggle
            "enable_risk_management": os.getenv("ENABLE_RISK_MANAGEMENT", "true").lower() == "true",
            
            # Position Limits
            "max_total_positions": int(os.getenv("MAX_TOTAL_POSITIONS", "15")),
            "max_positions_per_symbol": int(os.getenv("MAX_POSITIONS_PER_SYMBOL", "3")),
            
            # Loss Limits (as percentages)
            "daily_loss_limit": float(os.getenv("DAILY_LOSS_LIMIT", "0.30")),  # 30%
            "emergency_stop_loss": float(os.getenv("EMERGENCY_STOP_LOSS", "0.50")),  # 50%
            
            # Trade Timing
            "trade_cooldown_seconds": int(os.getenv("TRADE_COOLDOWN_SECONDS", "300")),  # 5 minutes
            "max_trade_duration_hours": int(os.getenv("MAX_TRADE_DURATION_HOURS", "2")),  # 2 hours
            
            # Position Sizing
            "position_size_percent": float(os.getenv("POSITION_SIZE_PERCENT", "0.02")),  # 2% per trade
            "max_position_size_percent": float(os.getenv("MAX_POSITION_SIZE_PERCENT", "0.05")),  # 5% max
            
            # Stop Loss and Take Profit
            "default_stop_loss_percent": float(os.getenv("DEFAULT_STOP_LOSS_PERCENT", "0.02")),  # 2%
            "default_take_profit_percent": float(os.getenv("DEFAULT_TAKE_PROFIT_PERCENT", "0.04")),  # 4%
            
            # Signal Filtering
            "min_confidence_threshold": float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "25.0")),
            "max_signals_per_cycle": int(os.getenv("MAX_SIGNALS_PER_CYCLE", "3")),
        }
    
    def _load_from_file(self, config_file: str):
        """Load configuration from file (future enhancement)"""
        # TODO: Implement file-based configuration
        pass
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def is_risk_management_enabled(self) -> bool:
        """Check if risk management is enabled"""
        return self.config["enable_risk_management"]
    
    def get_daily_loss_limit(self) -> float:
        """Get daily loss limit (0.0 to 1.0)"""
        return self.config["daily_loss_limit"]
    
    def get_emergency_stop_loss(self) -> float:
        """Get emergency stop loss (0.0 to 1.0)"""
        return self.config["emergency_stop_loss"]
    
    def get_max_trade_duration_seconds(self) -> int:
        """Get maximum trade duration in seconds"""
        return self.config["max_trade_duration_hours"] * 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return self.config.copy()

# Global risk configuration instance
risk_config = RiskConfig()

# Environment variable examples:
# ENABLE_RISK_MANAGEMENT=false  # Disable all risk management
# DAILY_LOSS_LIMIT=0.50         # Set 50% daily loss limit
# MAX_TOTAL_POSITIONS=50        # Allow up to 50 positions
# MIN_CONFIDENCE_THRESHOLD=10.0 # Lower confidence threshold
