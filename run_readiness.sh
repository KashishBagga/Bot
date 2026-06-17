#!/bin/bash
# Run Monday Readiness Report with proper Python path
export PYTHONPATH=$(pwd)
python3 src/analytics/monday_readiness_report.py "$@"
