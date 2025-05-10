"""
StrategyEmaCrossover strategy.
Trading strategy implementation.
"""
import pandas as pd
from typing import Dict, Any
from src.strategies.ema_crossover import EmaCrossover

class StrategyEmaCrossover(EmaCrossover):
    """Trading strategy implementation for EMA Crossover.
    
    This is a wrapper class that inherits from the consolidated EmaCrossover implementation.
    Maintains the same interface but delegates all functionality to the parent class.
    """
    
    def __init__(self, params: Dict[str, Any] = None):
        """Initialize the strategy with the original strategy name.
        
        Args:
            params: Strategy parameters including crossover_strength and momentum
        """
        params = params or {}
        # Initialize the parent class but override the strategy name
        super().__init__(params)
        self.name = "strategy_ema_crossover"  # Override the name for database logging
    
    # All other methods are inherited from EmaCrossover
