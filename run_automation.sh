#!/bin/bash
# Trading Automation System Runner
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT=$SCRIPT_DIR
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
python3 "$PROJECT_ROOT/src/automation/trading_automation.py" "$@"
