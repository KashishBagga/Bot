#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
python3 "$PROJECT_ROOT/src/trading/trading_dashboard.py" "$@"
