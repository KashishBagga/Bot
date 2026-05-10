#!/bin/bash
# Run Indian Trader with proper Python path
export PYTHONPATH=/Users/kashishbagga/Desktop/Bot
cd /Users/kashishbagga/Desktop/Bot
python3 src/trading/indian_trader.py "$@"
