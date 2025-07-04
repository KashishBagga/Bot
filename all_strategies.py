#!/usr/bin/env python3
"""
All Strategies Module - Interface for accessing all trading strategies
"""

import sys
import os
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
import importlib.util
import inspect

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def get_available_strategies() -> List[str]:
    """Get list of available strategy names."""
    strategies_dir = Path("src/strategies")
    strategies = []
    
    for file in strategies_dir.glob("*.py"):
        if file.name != "__init__.py":
            strategy_name = file.stem
            strategies.append(strategy_name)
    
    return strategies

def get_strategy_instance(strategy_name: str):
    """Get an instance of the specified strategy."""
    try:
        # Import the strategy module
        module_path = f"src.strategies.{strategy_name}"
        strategy_module = importlib.import_module(module_path)
        
        # Get the strategy class (assumes class name is CamelCase version of file name)
        class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
        strategy_class = getattr(strategy_module, class_name)
        
        # Return an instance
        return strategy_class()
    except Exception as e:
        print(f"Error loading strategy {strategy_name}: {e}")
        return None

def run_strategy(strategy_name: str, dataframes: Dict[str, pd.DataFrame], 
                multi_timeframe_dataframes: Dict[str, Dict[str, pd.DataFrame]], 
                save_to_db: bool = False) -> Dict[str, Any]:
    """
    Run a strategy on multiple symbols for backtesting.
    
    Args:
        strategy_name: Name of the strategy to run
        dataframes: Dictionary of {symbol: DataFrame} for primary timeframe
        multi_timeframe_dataframes: Dictionary of {symbol: {timeframe: DataFrame}}
        save_to_db: Whether to save results to database
        
    Returns:
        Dictionary of results for each symbol
    """
    try:
        # Import the strategy module
        module_path = f"src.strategies.{strategy_name}"
        strategy_module = importlib.import_module(module_path)
        
        results = {}
        
        # Run strategy on each symbol
        for symbol, df in dataframes.items():
            try:
                # Get multi-timeframe data for this symbol
                multi_tf_data = multi_timeframe_dataframes.get(symbol, {})
                
                # Check different strategy patterns
                if hasattr(strategy_module, 'run_strategy'):
                    # Pattern 1: run_strategy function (like ema_crossover)
                    signals = []
                    for i in range(len(df)):
                        if i < 20:  # Skip first 20 candles for indicators to stabilize
                            continue
                        candle = df.iloc[i]
                        result = strategy_module.run_strategy(candle, symbol)
                        if result and result.get('signal') not in ['NO TRADE', None]:
                            signals.append(result)
                    
                    # Process results
                    if signals:
                        signal_counts = {}
                        for signal in signals:
                            signal_type = signal.get('signal', 'NO TRADE')
                            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
                        
                        results[symbol] = {
                            'signals': signal_counts,
                            'total_signals': len(signals),
                            'raw_signals': signals if not save_to_db else None
                        }
                    else:
                        results[symbol] = {
                            'signals': {'NO TRADE': 1},
                            'total_signals': 0,
                            'raw_signals': []
                        }
                        
                elif hasattr(strategy_module, f'strategy_{strategy_name}'):
                    # Pattern 2: strategy_* function (like supertrend_ema)
                    strategy_func = getattr(strategy_module, f'strategy_{strategy_name}')
                    signals = []
                    for i in range(len(df)):
                        if i < 20:  # Skip first 20 candles for indicators to stabilize
                            continue
                        candle = df.iloc[i]
                        result = strategy_func(candle, symbol)
                        if result and result.get('signal') not in ['NO TRADE', None]:
                            signals.append(result)
                    
                    # Process results
                    if signals:
                        signal_counts = {}
                        for signal in signals:
                            signal_type = signal.get('signal', 'NO TRADE')
                            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
                        
                        results[symbol] = {
                            'signals': signal_counts,
                            'total_signals': len(signals),
                            'raw_signals': signals if not save_to_db else None
                        }
                    else:
                        results[symbol] = {
                            'signals': {'NO TRADE': 1},
                            'total_signals': 0,
                            'raw_signals': []
                        }
                        
                else:
                    # Pattern 3: Class-based approach (like insidebar_rsi, supertrend_macd_rsi_ema)
                    class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
                    strategy_class = getattr(strategy_module, class_name)
                    strategy_instance = strategy_class(timeframe_data=multi_tf_data)
                    
                    # Check if the analyze method expects individual candles or full dataframe
                    analyze_signature = inspect.signature(strategy_instance.analyze)
                    params = list(analyze_signature.parameters.keys())
                    
                    # Handle different strategy signatures
                    if False and strategy_name in ["ema_crossover", "supertrend_macd_rsi_ema"]:
                        pass  # Let the generic parameter-based branches handle these strategies
                    elif len(params) >= 3 and 'candle' in params and 'index' in params and 'df' in params:
                        # Method expects (candle, index, df, future_data) - iterate through candles
                        signals = []
                        for i in range(len(df)):
                            if i < 20:  # Skip first 20 candles for indicators to stabilize
                                continue
                            candle = df.iloc[i]
                            # Pass the candle, its index in the dataframe, and the full dataframe
                            result = strategy_instance.analyze(candle, i, df, multi_tf_data)
                            if result and result.get('signal') not in ['NO TRADE', None]:
                                signals.append(result)
                        
                        # Process results
                        if signals:
                            signal_counts = {}
                            for signal in signals:
                                signal_type = signal.get('signal', 'NO TRADE')
                                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
                            
                            results[symbol] = {
                                'signals': signal_counts,
                                'total_signals': len(signals),
                                'raw_signals': signals if not save_to_db else None
                            }
                        else:
                            results[symbol] = {
                                'signals': {'NO TRADE': 1},
                                'total_signals': 0,
                                'raw_signals': []
                            }
                    elif len(params) >= 3 and 'data' in params and 'index_name' in params:
                        # Method expects (data, index_name, future_data) - pass full dataframe
                        result = strategy_instance.analyze(df, symbol, multi_tf_data)
                        if result and result.get('signal') not in ['NO TRADE', None]:
                            signal_type = result.get('signal', 'NO TRADE')
                            results[symbol] = {
                                'signals': {signal_type: 1},
                                'total_signals': 1,
                                'raw_signals': [result] if not save_to_db else None
                            }
                        else:
                            results[symbol] = {
                                'signals': {'NO TRADE': 1},
                                'total_signals': 0,
                                'raw_signals': []
                            }
                    else:
                        # Method expects (data, symbol) - analyze full dataframe
                        result = strategy_instance.analyze(df, symbol)
                        
                        if result and result.get('signal') not in ['NO TRADE', None]:
                            signal_type = result.get('signal', 'NO TRADE')
                            results[symbol] = {
                                'signals': {signal_type: 1},
                                'total_signals': 1,
                                'raw_signals': [result] if not save_to_db else None
                            }
                        else:
                            results[symbol] = {
                                'signals': {'NO TRADE': 1},
                                'total_signals': 0,
                                'raw_signals': []
                            }
                    
            except Exception as e:
                print(f"Error running {strategy_name} on {symbol}: {e}")
                results[symbol] = {
                    'error': str(e),
                    'signals': {'ERROR': 1},
                    'total_signals': 0
                }
        
        return results
        
    except Exception as e:
        print(f"Error loading strategy {strategy_name}: {e}")
        return {'error': str(e)}

def run_strategy_backtest(strategy_name: str, symbol: str, dataframe: pd.DataFrame, 
                         multi_timeframe_data: Dict[str, pd.DataFrame] = None) -> List[Dict]:
    """
    Run a single strategy on a single symbol (for individual testing).
    
    Args:
        strategy_name: Name of the strategy to run
        symbol: Symbol to analyze
        dataframe: OHLC data for the symbol
        multi_timeframe_data: Optional multi-timeframe data
        
    Returns:
        List of signal dictionaries
    """
    try:
        # Import the strategy module
        module_path = f"src.strategies.{strategy_name}"
        strategy_module = importlib.import_module(module_path)
        
        # For strategies that expose both a class and a legacy run_strategy wrapper
        # (e.g., ema_crossover, supertrend_macd_rsi_ema), prefer the class-based
        # approach so we can pass the full DataFrame context for indicator look-back.
        if strategy_name in ["ema_crossover", "supertrend_macd_rsi_ema"]:
            class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
            strategy_class = getattr(strategy_module, class_name)
            strategy_instance = strategy_class(timeframe_data=multi_timeframe_data or {})

            # Iterate through candles, skip first 20 for indicator warm-up
            signals = []
            for i in range(len(dataframe)):
                if i < 20:
                    continue
                candle = dataframe.iloc[i]
                result = strategy_instance.analyze(candle, i, dataframe, multi_timeframe_data)
                if result and result.get('signal') not in ['NO TRADE', None]:
                    signals.append(result)
            return signals
            
        elif hasattr(strategy_module, 'run_strategy'):
            # Pattern 1: run_strategy function (like ema_crossover)
            signals = []
            for i in range(len(dataframe)):
                if i < 20:  # Skip first 20 candles for indicators to stabilize
                    continue
                candle = dataframe.iloc[i]
                result = strategy_module.run_strategy(candle, symbol)
                if result and result.get('signal') not in ['NO TRADE', None]:
                    signals.append(result)
            return signals
            
        elif hasattr(strategy_module, f'strategy_{strategy_name}'):
            # Pattern 2: strategy_* function (like supertrend_ema)
            strategy_func = getattr(strategy_module, f'strategy_{strategy_name}')
            signals = []
            for i in range(len(dataframe)):
                if i < 20:  # Skip first 20 candles for indicators to stabilize
                    continue
                candle = dataframe.iloc[i]
                result = strategy_func(candle, symbol)
                if result and result.get('signal') not in ['NO TRADE', None]:
                    signals.append(result)
            return signals
            
        else:
            # Pattern 3: Class-based approach (like insidebar_rsi, supertrend_macd_rsi_ema)
            class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
            strategy_class = getattr(strategy_module, class_name)
            strategy_instance = strategy_class(timeframe_data=multi_timeframe_data or {})
            
            # Check if the analyze method expects individual candles or full dataframe
            analyze_signature = inspect.signature(strategy_instance.analyze)
            params = list(analyze_signature.parameters.keys())
            
            if False and strategy_name == "ema_crossover":
                pass  # Let generic branches below handle
            elif False and strategy_name == "supertrend_macd_rsi_ema":
                pass
            elif len(params) >= 3 and 'data' in params and 'index_name' in params:
                # Method expects (data, index_name, future_data) - pass full dataframe
                result = strategy_instance.analyze(dataframe, symbol, multi_timeframe_data)
                if result and result.get('signal') not in ['NO TRADE', None]:
                    return [result]
                return []
            else:
                # Method expects (data, symbol) - analyze full dataframe
                result = strategy_instance.analyze(dataframe, symbol)
                return [result] if result and result.get('signal') not in ['NO TRADE', None] else []
        
    except Exception as e:
        print(f"Error running strategy {strategy_name}: {e}")
        return []

def get_strategy_description(strategy_name: str) -> str:
    """Get description of a strategy."""
    descriptions = {
        'ema_crossover': 'EMA Crossover strategy using 9 and 21 period EMAs',
        'insidebar_rsi': 'Inside Bar with RSI confirmation strategy',
        'supertrend_ema': 'Supertrend with EMA filter strategy',
        'supertrend_macd_rsi_ema': 'Advanced Supertrend with MACD, RSI, and EMA filters'
    }
    return descriptions.get(strategy_name, 'No description available')

def validate_strategy_name(strategy_name: str) -> bool:
    """Validate if a strategy name exists."""
    return strategy_name in get_available_strategies()

if __name__ == "__main__":
    # Test the module
    print("Available Trading Strategies:")
    strategies = get_available_strategies()
    for strategy in strategies:
        print(f"  - {strategy}: {get_strategy_description(strategy)}")
    
    print(f"\nTotal strategies: {len(strategies)}") 