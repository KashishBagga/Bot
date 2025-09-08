"""
Enhanced error handling system for the trading platform.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Callable
from functools import wraps
from enum import Enum
import time

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories."""
    API_ERROR = "api_error"
    DATA_ERROR = "data_error"
    TRADING_ERROR = "trading_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"

class TradingError(Exception):
    """Base exception for trading-related errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM_ERROR, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, details: Optional[Dict] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.timestamp = time.time()

class APIError(TradingError):
    """API-related errors."""
    
    def __init__(self, message: str, api_name: str = "unknown", 
                 status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message, ErrorCategory.API_ERROR, ErrorSeverity.HIGH, details)
        self.api_name = api_name
        self.status_code = status_code

class DataError(TradingError):
    """Data-related errors."""
    
    def __init__(self, message: str, data_source: str = "unknown", details: Optional[Dict] = None):
        super().__init__(message, ErrorCategory.DATA_ERROR, ErrorSeverity.MEDIUM, details)
        self.data_source = data_source

class TradingSystemError(TradingError):
    """Trading system errors."""
    
    def __init__(self, message: str, trade_id: Optional[str] = None, details: Optional[Dict] = None):
        super().__init__(message, ErrorCategory.TRADING_ERROR, ErrorSeverity.HIGH, details)
        self.trade_id = trade_id

class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history = 1000
    
    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle an error and return recovery information."""
        context = context or {}
        
        # Log the error
        self._log_error(error, context)
        
        # Track error counts
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to history
        self.error_history.append({
            'timestamp': time.time(),
            'error_type': error_type,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        })
        
        # Keep history size manageable
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        # Determine recovery action
        recovery_action = self._determine_recovery_action(error, context)
        
        return {
            'error_handled': True,
            'recovery_action': recovery_action,
            'should_retry': recovery_action.get('retry', False),
            'retry_delay': recovery_action.get('delay', 0),
            'should_stop': recovery_action.get('stop', False)
        }
    
    def _log_error(self, error: Exception, context: Dict):
        """Log error with appropriate level."""
        if isinstance(error, TradingError):
            if error.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"CRITICAL ERROR: {error}")
            elif error.severity == ErrorSeverity.HIGH:
                logger.error(f"HIGH SEVERITY: {error}")
            elif error.severity == ErrorSeverity.MEDIUM:
                logger.warning(f"MEDIUM SEVERITY: {error}")
            else:
                logger.info(f"LOW SEVERITY: {error}")
        else:
            logger.error(f"UNEXPECTED ERROR: {error}")
        
        if context:
            logger.debug(f"Error context: {context}")
    
    def _determine_recovery_action(self, error: Exception, context: Dict) -> Dict[str, Any]:
        """Determine the appropriate recovery action."""
        if isinstance(error, APIError):
            if error.status_code == 429:  # Rate limit
                return {'retry': True, 'delay': 60, 'stop': False}
            elif error.status_code in [500, 502, 503, 504]:  # Server errors
                return {'retry': True, 'delay': 30, 'stop': False}
            elif error.status_code == 401:  # Authentication error
                return {'retry': False, 'delay': 0, 'stop': True}
            else:
                return {'retry': True, 'delay': 10, 'stop': False}
        
        elif isinstance(error, DataError):
            return {'retry': True, 'delay': 5, 'stop': False}
        
        elif isinstance(error, TradingSystemError):
            return {'retry': False, 'delay': 0, 'stop': True}
        
        else:
            return {'retry': True, 'delay': 5, 'stop': False}
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'error_counts': self.error_counts,
            'total_errors': len(self.error_history),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }

# Global error handler instance
error_handler = ErrorHandler()

def handle_errors(category: ErrorCategory = ErrorCategory.SYSTEM_ERROR, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:100],  # Truncate for logging
                    'kwargs': str(kwargs)[:100]
                }
                
                result = error_handler.handle_error(e, context)
                
                if result['should_stop']:
                    raise
                elif result['should_retry']:
                    time.sleep(result['retry_delay'])
                    return func(*args, **kwargs)
                else:
                    return None
        
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, **kwargs) -> Optional[Any]:
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        context = {
            'function': func.__name__,
            'args': str(args)[:100],
            'kwargs': str(kwargs)[:100]
        }
        error_handler.handle_error(e, context)
        return None
