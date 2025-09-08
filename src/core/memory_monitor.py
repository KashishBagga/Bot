"""
Memory monitoring and management system.
"""

import psutil
import threading
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MemoryAlertLevel(Enum):
    """Memory alert levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MemoryStats:
    """Memory statistics."""
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    process_memory: float
    process_memory_percent: float
    timestamp: float

class MemoryMonitor:
    """Memory monitoring and management system."""
    
    def __init__(self, 
                 check_interval: float = 30.0,
                 low_threshold: float = 70.0,
                 medium_threshold: float = 80.0,
                 high_threshold: float = 90.0,
                 critical_threshold: float = 95.0):
        
        self.check_interval = check_interval
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.critical_threshold = critical_threshold
        
        self.process = psutil.Process()
        self.is_monitoring = False
        self.monitor_thread = None
        self.alert_handlers = {}
        self.memory_history = []
        self.max_history = 100
        
        logger.info("Memory monitor initialized")
    
    def start_monitoring(self):
        """Start memory monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("✅ Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Memory monitoring stopped")
    
    def add_alert_handler(self, level: MemoryAlertLevel, handler: Callable):
        """Add an alert handler for specific memory levels."""
        if level not in self.alert_handlers:
            self.alert_handlers[level] = []
        self.alert_handlers[level].append(handler)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process_memory = self.process.memory_info()
        process_memory_mb = process_memory.rss / 1024 / 1024
        
        # Process memory percentage
        process_memory_percent = self.process.memory_percent()
        
        return MemoryStats(
            total_memory=system_memory.total / 1024 / 1024 / 1024,  # GB
            available_memory=system_memory.available / 1024 / 1024 / 1024,  # GB
            used_memory=system_memory.used / 1024 / 1024 / 1024,  # GB
            memory_percent=system_memory.percent,
            process_memory=process_memory_mb,  # MB
            process_memory_percent=process_memory_percent,
            timestamp=time.time()
        )
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                stats = self.get_memory_stats()
                
                # Add to history
                self.memory_history.append(stats)
                if len(self.memory_history) > self.max_history:
                    self.memory_history = self.memory_history[-self.max_history:]
                
                # Check for alerts
                self._check_alerts(stats)
                
                # Log memory usage
                logger.debug(f"Memory: {stats.memory_percent:.1f}% system, "
                           f"{stats.process_memory_percent:.1f}% process")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _check_alerts(self, stats: MemoryStats):
        """Check for memory alerts."""
        memory_percent = stats.memory_percent
        
        if memory_percent >= self.critical_threshold:
            self._trigger_alert(MemoryAlertLevel.CRITICAL, stats)
        elif memory_percent >= self.high_threshold:
            self._trigger_alert(MemoryAlertLevel.HIGH, stats)
        elif memory_percent >= self.medium_threshold:
            self._trigger_alert(MemoryAlertLevel.MEDIUM, stats)
        elif memory_percent >= self.low_threshold:
            self._trigger_alert(MemoryAlertLevel.LOW, stats)
    
    def _trigger_alert(self, level: MemoryAlertLevel, stats: MemoryStats):
        """Trigger memory alert."""
        if level in self.alert_handlers:
            for handler in self.alert_handlers[level]:
                try:
                    handler(stats, level)
                except Exception as e:
                    logger.error(f"Error in memory alert handler: {e}")
        
        # Log alert
        logger.warning(f"MEMORY ALERT {level.value.upper()}: "
                      f"System memory {stats.memory_percent:.1f}%, "
                      f"Process memory {stats.process_memory_percent:.1f}%")
    
    def cleanup_memory(self):
        """Attempt to cleanup memory."""
        try:
            import gc
            gc.collect()
            
            # Log cleanup
            stats_after = self.get_memory_stats()
            logger.info(f"Memory cleanup completed. Process memory: "
                       f"{stats_after.process_memory:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def get_memory_trend(self, minutes: int = 10) -> Dict[str, Any]:
        """Get memory usage trend over time."""
        cutoff_time = time.time() - (minutes * 60)
        recent_stats = [s for s in self.memory_history if s.timestamp >= cutoff_time]
        
        if not recent_stats:
            return {'trend': 'no_data', 'change': 0}
        
        # Calculate trend
        first_memory = recent_stats[0].process_memory
        last_memory = recent_stats[-1].process_memory
        change = last_memory - first_memory
        change_percent = (change / first_memory) * 100 if first_memory > 0 else 0
        
        if change_percent > 5:
            trend = 'increasing'
        elif change_percent < -5:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_mb': change,
            'change_percent': change_percent,
            'data_points': len(recent_stats)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory monitoring summary."""
        current_stats = self.get_memory_stats()
        trend = self.get_memory_trend()
        
        return {
            'current': {
                'system_memory_percent': current_stats.memory_percent,
                'process_memory_mb': current_stats.process_memory,
                'process_memory_percent': current_stats.process_memory_percent
            },
            'trend': trend,
            'monitoring': self.is_monitoring,
            'history_size': len(self.memory_history),
            'thresholds': {
                'low': self.low_threshold,
                'medium': self.medium_threshold,
                'high': self.high_threshold,
                'critical': self.critical_threshold
            }
        }

# Global memory monitor instance
memory_monitor = MemoryMonitor()

def start_memory_monitoring():
    """Start global memory monitoring."""
    memory_monitor.start_monitoring()

def stop_memory_monitoring():
    """Stop global memory monitoring."""
    memory_monitor.stop_monitoring()

def get_memory_summary() -> Dict[str, Any]:
    """Get memory monitoring summary."""
    return memory_monitor.get_summary()
