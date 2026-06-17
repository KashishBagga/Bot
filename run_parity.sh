#!/bin/bash
# Run Parity Audit with proper Python path
export PYTHONPATH=$(pwd)
python3 src/analytics/parity_engine.py "$@"
