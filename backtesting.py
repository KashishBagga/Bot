#!/usr/bin/env python3
"""
Main entry point for backtesting all strategies at once.
This is a simple wrapper around all_strategies.py.
"""
import sys
import os
import subprocess

# Ensure all DB tables and rows are up-to-date before running backtesting
# subprocess.run([sys.executable, 'auto_ensure_performance_metrics.py'], check=True)

if __name__ == "__main__":
    # Simply call the all_strategies.py script with the system arguments
    from all_strategies import run_all_strategies
    
    # Default parameters
    days_back = 60
    resolution = "15"
    save_to_db = True
    symbols = None
    
    # Process command-line arguments if provided
    if len(sys.argv) > 1:
        # Check for --days
        if "--days" in sys.argv:
            try:
                days_index = sys.argv.index("--days")
                if days_index + 1 < len(sys.argv):
                    days_back = int(sys.argv[days_index + 1])
            except (ValueError, IndexError):
                pass
                
        # Check for --resolution
        if "--resolution" in sys.argv:
            try:
                res_index = sys.argv.index("--resolution")
                if res_index + 1 < len(sys.argv):
                    resolution = sys.argv[res_index + 1]
            except (ValueError, IndexError):
                pass
                
        # Check for --no-save
        if "--no-save" in sys.argv:
            save_to_db = False
            
        # Check for --symbols
        if "--symbols" in sys.argv:
            try:
                sym_index = sys.argv.index("--symbols")
                if sym_index + 1 < len(sys.argv):
                    symbols_str = sys.argv[sym_index + 1]
                    # Process symbols
                    symbol_list = symbols_str.split(',')
                    symbols = {}
                    for symbol in symbol_list:
                        symbol = symbol.strip().upper()
                        if symbol == 'NIFTY50':
                            symbols["NSE:NIFTY50-INDEX"] = "NIFTY50"
                        elif symbol == 'BANKNIFTY':
                            symbols["NSE:NIFTYBANK-INDEX"] = "BANKNIFTY"
                        else:
                            # Assume it's a stock
                            symbols[f"NSE:{symbol}-EQ"] = symbol
            except (ValueError, IndexError):
                pass
    
    # Print the configuration
    print(f"Running backtesting with the following parameters:")
    print(f"  Days back: {days_back}")
    print(f"  Resolution: {resolution}")
    print(f"  Save to database: {save_to_db}")
    print(f"  Symbols: {symbols if symbols else 'NIFTY50,BANKNIFTY (default)'}")
    print("")
    
    # Run all strategies
    success = run_all_strategies(
        days_back=days_back,
        resolution=resolution,
        save_to_db=save_to_db,
        symbols=symbols
    )
    
    sys.exit(0 if success else 1)
