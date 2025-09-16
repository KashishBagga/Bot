"""
Production Systems Package

This package contains all the production-ready systems for the trading platform:

MUST IMPLEMENTATIONS:
- end_to_end_validation.py: End-to-end backtest → forward-test validation
- execution_reliability.py: Guaranteed order execution and reconciliation
- database_resilience.py: Atomic transactions and automated backup/restore
- robust_risk_engine.py: Portfolio-level risk controls and circuit breakers
- slippage_model.py: Realistic slippage and partial fill simulation
- pre_live_checklist.py: Formal go-live checklist with kill switch

HIGH PRIORITY SYSTEMS:
- broker_abstraction.py: Multi-broker failover and abstraction
- capital_efficiency.py: Dynamic capital allocation optimization

All systems are designed for production deployment with comprehensive
error handling, monitoring, and failover capabilities.
"""

from .end_to_end_validation import EndToEndValidator, ValidationConfig, ValidationResult
from .execution_reliability import ExecutionReliabilityManager, ReconciliationConfig
from .database_resilience import DatabaseResilienceManager, BackupConfig
from .robust_risk_engine import RobustRiskEngine, PortfolioConstraints
from .slippage_model import SlippageModel, SlippageConfig, PartialFillConfig
from .pre_live_checklist import PreLiveChecklist, ChecklistStatus, TradingStatus
from .broker_abstraction import BrokerFailoverManager, IBrokerAdapter
from .capital_efficiency import CapitalEfficiencyOptimizer

__all__ = [
    'EndToEndValidator',
    'ValidationConfig', 
    'ValidationResult',
    'ExecutionReliabilityManager',
    'ReconciliationConfig',
    'DatabaseResilienceManager',
    'BackupConfig',
    'RobustRiskEngine',
    'PortfolioConstraints',
    'SlippageModel',
    'SlippageConfig',
    'PartialFillConfig',
    'PreLiveChecklist',
    'ChecklistStatus',
    'TradingStatus',
    'BrokerFailoverManager',
    'IBrokerAdapter',
    'CapitalEfficiencyOptimizer'
]
