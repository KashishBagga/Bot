"""
Backward compatibility wrapper for the EMA Crossover strategy.
This file redirects to the consolidated implementation in src/strategies/ema_crossover.py
"""
# Import directly from the module to avoid circular imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.ema_crossover import run_strategy

def ema_crossover(candle, index_name, future_data=None, crossover_strength=None, momentum=None):
    """
    Backward compatibility wrapper function for the EMA Crossover strategy.
    
    Args:
        candle: The candle data
        index_name: Name of the index being analyzed
        future_data: Optional future data for performance tracking
        crossover_strength: Optional crossover strength override
        momentum: Optional momentum override
        
    Returns:
        Dict with signal information
    """
    return run_strategy(candle, index_name, future_data, crossover_strength, momentum)
