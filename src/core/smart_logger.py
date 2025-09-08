"""
Smart Logger for Trading System
Provides structured, actionable logging with different levels and analysis
"""

import logging
import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

class LogLevel(Enum):
    CRITICAL = "CRITICAL"
    IMPORTANT = "IMPORTANT" 
    INFO = "INFO"
    DEBUG = "DEBUG"
    VERBOSE = "VERBOSE"

@dataclass
class LogEntry:
    timestamp: str
    level: str
    module: str
    action: str
    details: Dict[str, Any]
    performance_data: Optional[Dict[str, Any]] = None

class SmartLogger:
    def __init__(self, name: str = "trading_system"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s [%(name)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Performance tracking
        self.performance_data = {
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_rejected': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'start_time': datetime.now(timezone.utc)
        }
        
        # Signal tracking
        self.signal_history = []
        self.trade_history = []
        
    def log_critical(self, module: str, action: str, details: Dict[str, Any]):
        """Log critical events: system errors, trade executions, capital changes"""
        self._log(LogLevel.CRITICAL, module, action, details)
        
    def log_important(self, module: str, action: str, details: Dict[str, Any]):
        """Log important events: signal generation, risk checks, position updates"""
        self._log(LogLevel.IMPORTANT, module, action, details)
        
    def log_info(self, module: str, action: str, details: Dict[str, Any]):
        """Log info events: system status, market conditions, performance"""
        self._log(LogLevel.INFO, module, action, details)
        
    def log_debug(self, module: str, action: str, details: Dict[str, Any]):
        """Log debug events: detailed calculations, strategy internals"""
        self._log(LogLevel.DEBUG, module, action, details)
        
    def log_verbose(self, module: str, action: str, details: Dict[str, Any]):
        """Log verbose events: raw data, API responses, technical details"""
        self._log(LogLevel.VERBOSE, module, action, details)
        
    def _log(self, level: LogLevel, module: str, action: str, details: Dict[str, Any]):
        """Internal logging method"""
        timestamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
        
        # Create structured message
        message = f"[{action}] {self._format_details(details)}"
        
        # Log based on level
        if level == LogLevel.CRITICAL:
            self.logger.critical(message)
        elif level == LogLevel.IMPORTANT:
            self.logger.warning(message)
        elif level == LogLevel.INFO:
            self.logger.info(message)
        elif level == LogLevel.DEBUG:
            self.logger.debug(message)
        else:  # VERBOSE
            self.logger.debug(f"[VERBOSE] {message}")
            
        # Track performance data
        self._update_performance_data(level, action, details)
        
    def _format_details(self, details: Dict[str, Any]) -> str:
        """Format details into readable string"""
        if not details:
            return ""
            
        formatted = []
        for key, value in details.items():
            if isinstance(value, (int, float)):
                if key in ['price', 'amount', 'pnl', 'equity']:
                    formatted.append(f"{key}: ${value:,.2f}")
                elif key in ['confidence', 'win_rate']:
                    formatted.append(f"{key}: {value:.1f}%")
                else:
                    formatted.append(f"{key}: {value}")
            else:
                formatted.append(f"{key}: {value}")
                
        return " | ".join(formatted)
        
    def _update_performance_data(self, level: LogLevel, action: str, details: Dict[str, Any]):
        """Update performance tracking data"""
        if action == "SIGNAL_GENERATED":
            self.performance_data['signals_generated'] += 1
            self.signal_history.append({
                'timestamp': datetime.now(timezone.utc),
                'symbol': details.get('symbol'),
                'strategy': details.get('strategy'),
                'confidence': details.get('confidence')
            })
            
        elif action == "TRADE_EXECUTED":
            self.performance_data['trades_executed'] += 1
            self.trade_history.append({
                'timestamp': datetime.now(timezone.utc),
                'symbol': details.get('symbol'),
                'action': details.get('action'),
                'amount': details.get('amount'),
                'price': details.get('price')
            })
            
        elif action == "TRADE_REJECTED":
            self.performance_data['trades_rejected'] += 1
            
        elif action == "PNL_UPDATE":
            self.performance_data['total_pnl'] = details.get('total_pnl', 0)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        runtime = datetime.now(timezone.utc) - self.performance_data['start_time']
        
        return {
            'runtime_minutes': runtime.total_seconds() / 60,
            'signals_generated': self.performance_data['signals_generated'],
            'trades_executed': self.performance_data['trades_executed'],
            'trades_rejected': self.performance_data['trades_rejected'],
            'total_pnl': self.performance_data['total_pnl'],
            'signal_rate_per_minute': self.performance_data['signals_generated'] / max(1, runtime.total_seconds() / 60),
            'trade_execution_rate': self.performance_data['trades_executed'] / max(1, self.performance_data['signals_generated']) * 100
        }
        
    def log_performance_summary(self):
        """Log current performance summary"""
        summary = self.get_performance_summary()
        self.log_info("PERFORMANCE", "SUMMARY", summary)
        
    def log_signal_analysis(self):
        """Log signal analysis"""
        if not self.signal_history:
            return
            
        # Analyze recent signals
        recent_signals = self.signal_history[-50:]  # Last 50 signals
        
        strategy_counts = {}
        symbol_counts = {}
        confidence_sum = 0
        
        for signal in recent_signals:
            strategy = signal.get('strategy', 'unknown')
            symbol = signal.get('symbol', 'unknown')
            confidence = signal.get('confidence', 0)
            
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            confidence_sum += confidence
            
        avg_confidence = confidence_sum / len(recent_signals) if recent_signals else 0
        
        analysis = {
            'total_signals': len(recent_signals),
            'avg_confidence': avg_confidence,
            'top_strategy': max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else 'none',
            'top_symbol': max(symbol_counts.items(), key=lambda x: x[1])[0] if symbol_counts else 'none',
            'strategy_distribution': strategy_counts,
            'symbol_distribution': symbol_counts
        }
        
        self.log_info("SIGNAL_ANALYSIS", "RECENT_SIGNALS", analysis)

# Global logger instance
smart_logger = SmartLogger()

# Convenience functions
def log_critical(module: str, action: str, details: Dict[str, Any]):
    smart_logger.log_critical(module, action, details)

def log_important(module: str, action: str, details: Dict[str, Any]):
    smart_logger.log_important(module, action, details)

def log_info(module: str, action: str, details: Dict[str, Any]):
    smart_logger.log_info(module, action, details)

def log_debug(module: str, action: str, details: Dict[str, Any]):
    smart_logger.log_debug(module, action, details)

def log_verbose(module: str, action: str, details: Dict[str, Any]):
    smart_logger.log_verbose(module, action, details)

def get_performance_summary() -> Dict[str, Any]:
    return smart_logger.get_performance_summary()

def log_performance_summary():
    smart_logger.log_performance_summary()

def log_signal_analysis():
    smart_logger.log_signal_analysis()
