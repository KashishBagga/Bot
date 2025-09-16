"""
Testing Systems Package

This package contains comprehensive testing systems for the trading platform:

CHAOS TESTING:
- chaos_testing.py: Stress and chaos testing engine

The chaos testing system simulates various failure scenarios including:
- Broker timeouts and failures
- Partial fills and order issues
- Database downtime and corruption
- API failures and network issues
- System resource pressure
- WebSocket disconnections
- Race conditions

All tests include recovery validation and data integrity checks.
"""

from .chaos_testing import ChaosTestingEngine, ChaosTestType, ChaosTestResult

__all__ = [
    'ChaosTestingEngine',
    'ChaosTestType',
    'ChaosTestResult'
]
