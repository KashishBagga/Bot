#!/bin/bash
# Enhanced Trading Dashboard Runner
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$SCRIPT_DIR
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
python3 "$PROJECT_ROOT/src/trading/enhanced_dashboard.py" "$@"
