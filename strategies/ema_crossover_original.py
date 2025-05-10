"""
Backward compatibility wrapper for the EMA Crossover strategy.
This file redirects to the consolidated implementation in src/strategies/ema_crossover.py
"""
import sys
import os
import pandas as pd
from datetime import datetime
from db import log_strategy_sql
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the class
from src.strategies.ema_crossover import EmaCrossover

def ema_crossover_original(candle, index_name, future_data=None, crossover_strength=None, momentum=None):
    """
    Backward compatibility wrapper function for the EMA Crossover strategy.
    Redirects to the consolidated EmaCrossover implementation.
    
    Args:
        candle: The candle data
        index_name: Name of the index being analyzed
        future_data: Optional future data for performance tracking
        crossover_strength: Optional crossover strength override
        momentum: Optional momentum override
        
    Returns:
        Dict with signal information
    """
    # Create the strategy
    strategy = EmaCrossover({
        'crossover_strength': crossover_strength,
        'momentum': momentum
    })
    
    # Create a single-row DataFrame from the candle
    if not isinstance(candle, pd.DataFrame):
        data = pd.DataFrame([candle])
    else:
        data = candle
        
    # Analyze the data
    result = strategy.analyze(data)
    
    # Add index name for database logging
    if result['signal'] != "NO TRADE":
        signal_data = result.copy()
        signal_data["signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        signal_data["index_name"] = index_name
        log_strategy_sql('ema_crossover', signal_data)  # Log to ema_crossover table
    
    return result 