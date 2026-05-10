#!/usr/bin/env python3
"""
Simple Risk Configuration Module
"""

class RiskConfig:
    def __init__(self):
        self.config = {
            "max_positions_per_symbol": 2,
            "max_total_positions": 6,
            "emergency_stop_loss": 0.10,  # 10% of capital — tighter for small accounts
            "risk_management_enabled": True
        }
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def get_emergency_stop_loss(self):
        return self.config.get("emergency_stop_loss", 0.20)
    
    def is_risk_management_enabled(self):
        return self.config.get("risk_management_enabled", True)

# Create global instance
risk_config = RiskConfig()
