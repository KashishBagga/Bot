"""
Trading strategies package.
Contains all available trading strategy implementations.
"""

# Import strategies for easier access
from importlib import import_module
from pathlib import Path
import os
import logging

# Skip list for redundant strategies
SKIP_STRATEGIES = ['ema_crossover_original', 'strategy_ema_crossover']

# Dynamically get all strategy modules (excluding __init__.py)
_strategy_files = [
    f.stem for f in Path(__file__).parent.glob("*.py")
    if f.name != "__init__.py" and not f.name.startswith("_") and f.stem not in SKIP_STRATEGIES
]

# Initialize logger
logger = logging.getLogger(__name__)

# Dictionary to store available strategies
available_strategies = {}

# Import each strategy lazily
for strategy_name in _strategy_files:
    try:
        # Import the module
        module = import_module(f"src.strategies.{strategy_name}")
        
        # Find the strategy class (assume it's the same as the module name with first letter capitalized)
        class_name = "".join(word.capitalize() for word in strategy_name.split("_"))
        
        # Check if the class exists in the module
        if hasattr(module, class_name):
            # Add to available strategies
            available_strategies[strategy_name] = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # Log the error
        logger.error(f"Error importing strategy {strategy_name}: {e}")

logger.info(f"Available strategies: {list(available_strategies.keys())}")

def get_available_strategies():
    """Get a list of available strategy names."""
    return list(available_strategies.keys())

def get_strategy_class(strategy_name):
    """Get a strategy class by name."""
    # Remove 'strategy_' prefix if it exists (for backward compatibility)
    if strategy_name.startswith('strategy_'):
        clean_name = strategy_name[9:]
        if clean_name in available_strategies:
            return available_strategies[clean_name]
    
    return available_strategies.get(strategy_name) 