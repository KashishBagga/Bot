#!/bin/bash
# Run Indian Trader with proper Python path and unbuffered output
export PYTHONPATH=$(pwd)
export PYTHONUNBUFFERED=1
python3 -u src/trading/indian_trader.py "$@"
