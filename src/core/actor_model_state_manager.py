#!/usr/bin/env python3
"""
Actor Model State Manager
Thread-safe state management using actor model pattern
"""

import logging
import threading
import queue
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict
import sqlite3

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class StateEvent:
    """State change event"""
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str
    priority: int = 0

@dataclass
class AlertEvent:
    """Alert event with severity and cooldown"""
    alert_type: str
    severity: AlertSeverity
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    source: str

class ActorModelStateManager:
    """Actor model state manager for thread-safe operations"""
    
    def __init__(self, db_path: str = "trading_state.db"):
        self.db_path = db_path
        self.event_queue = queue.PriorityQueue()
        self.state = {}
        self.alert_history = defaultdict(list)
        self.alert_cooldowns = {}
        
        # Severity-based cooldown durations (seconds)
        self.cooldown_durations = {
            AlertSeverity.LOW: 300,      # 5 minutes
            AlertSeverity.MEDIUM: 180,   # 3 minutes
            AlertSeverity.HIGH: 60,      # 1 minute
            AlertSeverity.CRITICAL: 10   # 10 seconds
        }
        
        # Thread management
        self.state_thread = None
        self.db_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize state database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS state_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        data TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        source TEXT NOT NULL,
                        priority INTEGER DEFAULT 0
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        data TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        source TEXT NOT NULL
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS state_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        state_data TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
    
    def start(self):
        """Start the actor model state manager"""
        if self.running:
            return
        
        self.running = True
        
        # Start state management thread
        self.state_thread = threading.Thread(target=self._state_manager_loop, daemon=True)
        self.state_thread.start()
        
        # Start database worker thread
        self.db_thread = threading.Thread(target=self._db_worker_loop, daemon=True)
        self.db_thread.start()
        
        logger.info("✅ Actor model state manager started")
    
    def stop(self):
        """Stop the actor model state manager"""
        self.running = False
        
        # Wait for threads to finish
        if self.state_thread and self.state_thread.is_alive():
            self.state_thread.join(timeout=5)
        
        if self.db_thread and self.db_thread.is_alive():
            self.db_thread.join(timeout=5)
        
        logger.info("✅ Actor model state manager stopped")
    
    def post_event(self, event_type: str, data: Dict[str, Any], 
                   source: str = "unknown", priority: int = 0):
        """Post event to state manager (thread-safe)"""
        try:
            event = StateEvent(
                event_type=event_type,
                data=data,
                timestamp=datetime.now(),
                source=source,
                priority=priority
            )
            
            # Use negative priority for max-heap behavior
            self.event_queue.put((-priority, event))
            
        except Exception as e:
            logger.error(f"❌ Failed to post event: {e}")
    
    def post_alert(self, alert_type: str, severity: AlertSeverity, 
                   message: str, data: Dict[str, Any], source: str = "unknown"):
        """Post alert with severity-based cooldown (thread-safe)"""
        try:
            # Check cooldown
            if not self._should_send_alert(alert_type, severity):
                return
            
            alert = AlertEvent(
                alert_type=alert_type,
                severity=severity,
                message=message,
                data=data,
                timestamp=datetime.now(),
                source=source
            )
            
            # Post as high-priority event
            self.post_event("alert", asdict(alert), source, priority=100)
            
        except Exception as e:
            logger.error(f"❌ Failed to post alert: {e}")
    
    def _should_send_alert(self, alert_type: str, severity: AlertSeverity) -> bool:
        """Check if alert should be sent based on cooldown"""
        try:
            current_time = datetime.now()
            cooldown_duration = self.cooldown_durations.get(severity, 300)
            
            # Check cooldown
            last_alert_time = self.alert_cooldowns.get(alert_type)
            if last_alert_time:
                time_diff = (current_time - last_alert_time).total_seconds()
                if time_diff < cooldown_duration:
                    return False
            
            # Update cooldown
            self.alert_cooldowns[alert_type] = current_time
            return True
            
        except Exception as e:
            logger.error(f"❌ Alert cooldown check failed: {e}")
            return True  # Default to sending alert
    
    def _state_manager_loop(self):
        """Main state management loop (runs in dedicated thread)"""
        logger.info("🔄 State manager loop started")
        
        while self.running:
            try:
                # Get event with timeout
                try:
                    priority, event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process event
                self._process_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ State manager loop error: {e}")
                time.sleep(1)
        
        logger.info("🔄 State manager loop stopped")
    
    def _process_event(self, event: StateEvent):
        """Process state event"""
        try:
            with self.lock:
                # Update state based on event type
                if event.event_type == "position_update":
                    self._handle_position_update(event.data)
                elif event.event_type == "risk_check":
                    self._handle_risk_check(event.data)
                elif event.event_type == "alert":
                    self._handle_alert(event.data)
                elif event.event_type == "state_snapshot":
                    self._handle_state_snapshot(event.data)
                else:
                    logger.warning(f"⚠️ Unknown event type: {event.event_type}")
                
                # Store event in database (async)
                self._store_event_async(event)
                
        except Exception as e:
            logger.error(f"❌ Event processing failed: {e}")
    
    def _handle_position_update(self, data: Dict[str, Any]):
        """Handle position update event"""
        try:
            symbol = data.get('symbol')
            if symbol:
                self.state[f'position_{symbol}'] = data
                logger.debug(f"📊 Position updated: {symbol}")
        except Exception as e:
            logger.error(f"❌ Position update handling failed: {e}")
    
    def _handle_risk_check(self, data: Dict[str, Any]):
        """Handle risk check event"""
        try:
            self.state['risk_metrics'] = data
            logger.debug("⚠️ Risk metrics updated")
        except Exception as e:
            logger.error(f"❌ Risk check handling failed: {e}")
    
    def _handle_alert(self, data: Dict[str, Any]):
        """Handle alert event"""
        try:
            alert_type = data.get('alert_type')
            severity = data.get('severity')
            message = data.get('message')
            
            # Store in alert history
            self.alert_history[alert_type].append(data)
            
            # Log alert
            logger.warning(f"🚨 {severity} Alert: {message}")
            
        except Exception as e:
            logger.error(f"❌ Alert handling failed: {e}")
    
    def _handle_state_snapshot(self, data: Dict[str, Any]):
        """Handle state snapshot event"""
        try:
            self.state.update(data)
            logger.debug("📸 State snapshot updated")
        except Exception as e:
            logger.error(f"❌ State snapshot handling failed: {e}")
    
    def _store_event_async(self, event: StateEvent):
        """Store event in database asynchronously"""
        try:
            # This would be handled by the DB worker thread
            # For now, we'll store it directly
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO state_events (event_type, data, timestamp, source, priority)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    event.event_type,
                    json.dumps(event.data),
                    event.timestamp.isoformat(),
                    event.source,
                    event.priority
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to store event: {e}")
    
    def _db_worker_loop(self):
        """Database worker loop for async DB operations"""
        logger.info("🔄 Database worker loop started")
        
        while self.running:
            try:
                # Perform periodic database maintenance
                self._perform_db_maintenance()
                time.sleep(30)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Database worker loop error: {e}")
                time.sleep(5)
        
        logger.info("🔄 Database worker loop stopped")
    
    def _perform_db_maintenance(self):
        """Perform periodic database maintenance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clean old events (keep last 7 days)
                cutoff_date = (datetime.now() - timedelta(days=7)).isoformat()
                conn.execute('''
                    DELETE FROM state_events WHERE timestamp < ?
                ''', (cutoff_date,))
                
                # Clean old alerts (keep last 30 days)
                cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
                conn.execute('''
                    DELETE FROM alerts WHERE timestamp < ?
                ''', (cutoff_date,))
                
                # Create state snapshot
                self._create_state_snapshot(conn)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Database maintenance failed: {e}")
    
    def _create_state_snapshot(self, conn: sqlite3.Connection):
        """Create state snapshot"""
        try:
            with self.lock:
                state_data = json.dumps(self.state)
            
            conn.execute('''
                INSERT INTO state_snapshots (state_data, timestamp)
                VALUES (?, ?)
            ''', (state_data, datetime.now().isoformat()))
            
        except Exception as e:
            logger.error(f"❌ State snapshot creation failed: {e}")
    
    def get_state(self, key: str = None) -> Any:
        """Get state value (thread-safe)"""
        try:
            with self.lock:
                if key:
                    return self.state.get(key)
                return self.state.copy()
        except Exception as e:
            logger.error(f"❌ Failed to get state: {e}")
            return None
    
    def get_alert_history(self, alert_type: str = None) -> List[Dict[str, Any]]:
        """Get alert history (thread-safe)"""
        try:
            with self.lock:
                if alert_type:
                    return self.alert_history.get(alert_type, [])
                return dict(self.alert_history)
        except Exception as e:
            logger.error(f"❌ Failed to get alert history: {e}")
            return []

# Global state manager instance
state_manager = ActorModelStateManager()

# Convenience functions
def post_event(event_type: str, data: Dict[str, Any], source: str = "unknown", priority: int = 0):
    """Post event to state manager"""
    state_manager.post_event(event_type, data, source, priority)

def post_alert(alert_type: str, severity: AlertSeverity, message: str, 
               data: Dict[str, Any], source: str = "unknown"):
    """Post alert to state manager"""
    state_manager.post_alert(alert_type, severity, message, data, source)

def get_state(key: str = None) -> Any:
    """Get state value"""
    return state_manager.get_state(key)

def get_alert_history(alert_type: str = None) -> List[Dict[str, Any]]:
    """Get alert history"""
    return state_manager.get_alert_history(alert_type)
