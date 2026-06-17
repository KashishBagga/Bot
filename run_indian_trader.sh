#!/bin/bash
# Run Indian Trader with proper Python path
export PYTHONPATH=$(pwd)
python3 src/trading/indian_trader.py "$@"
