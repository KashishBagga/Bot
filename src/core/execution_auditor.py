import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Optional, Any
from src.models.postgres_database import PostgresDatabase

logger = logging.getLogger("ExecutionAuditor")
IST = ZoneInfo("Asia/Kolkata")

class ExecutionAuditor:
    """Audits and logs trade execution state transitions to PostgreSQL for trace audits."""
    
    def __init__(self, db: PostgresDatabase):
        self.db = db
        
    def log_event(self, event_type: str, trade_id: Optional[str] = None, 
                  candidate_id: Optional[str] = None, payload: Optional[Dict[str, Any]] = None):
        """Saves a detailed trace execution event milestone."""
        # tz-aware IST — must match the market-loop timestamps on trade_events so
        # the dashboard can order a single trade's events and compute real latency
        # deltas. A naive datetime.now() here was off by the UTC↔IST offset.
        timestamp = datetime.now(IST)
        event_id = f"exec_{int(timestamp.timestamp() * 1000)}_{event_type.lower()}"
        
        event = {
            'event_id': event_id,
            'trade_id': trade_id,
            'candidate_id': candidate_id,
            'timestamp': timestamp,
            'event_type': event_type,
            'payload': payload or {}
        }
        
        logger.info(f"📊 EXECUTION AUDIT [{event_type}]: trade_id={trade_id}, candidate_id={candidate_id}, payload={payload}")
        self.db.save_execution_event(event)
