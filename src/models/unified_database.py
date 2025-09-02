import sqlite3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import json

logger = logging.getLogger(__name__)

class UnifiedDatabase:
    """Unified database for all trading data."""
    
    def __init__(self, db_path: str = 'unified_trading.db'):
        """Initialize unified database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Live trading signals
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS live_trading_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        reasoning TEXT,
                        price REAL NOT NULL,
                        stop_loss REAL,
                        target1 REAL,
                        target2 REAL,
                        target3 REAL,
                        position_multiplier REAL DEFAULT 1.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Rejected signals
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS rejected_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        rejection_reason TEXT NOT NULL,
                        rejection_type TEXT NOT NULL,
                        additional_data TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            
                conn.commit()
                logger.info("✅ Unified database schema created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating unified database: {e}")
            raise
    
    def save_live_trading_signal(self, signal_data: Dict):
        """Save live trading signal to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO live_trading_signals 
                    (timestamp, symbol, strategy, signal_type, confidence, reasoning, 
                     price, stop_loss, target1, target2, target3, position_multiplier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_data['timestamp'],
                    signal_data['symbol'],
                    signal_data['strategy'],
                    signal_data['signal_type'],
                    signal_data['confidence'],
                    signal_data.get('reasoning', ''),
                    signal_data['price'],
                    signal_data.get('stop_loss'),
                    signal_data.get('target1'),
                    signal_data.get('target2'),
                    signal_data.get('target3'),
                    signal_data.get('position_multiplier', 1.0)
                ))
                
                conn.commit()
                logger.info(f"✅ Live trading signal saved: {signal_data['symbol']} {signal_data['signal_type']}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving live trading signal: {e}")
            return False
    
    def save_rejected_signal(self, signal_data: Dict):
        """Save rejected signal to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO rejected_signals 
                    (timestamp, symbol, strategy, signal_type, confidence, 
                     rejection_reason, rejection_type, additional_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_data['timestamp'],
                    signal_data['symbol'],
                    signal_data['strategy'],
                    signal_data['signal_type'],
                    signal_data['confidence'],
                    signal_data['rejection_reason'],
                    signal_data['rejection_type'],
                    json.dumps(signal_data.get('additional_data', {}))
                ))
                
                conn.commit()
                logger.info(f"✅ Rejected signal saved: {signal_data['symbol']} - {signal_data['rejection_reason']}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error saving rejected signal: {e}")
            return False
    
    def get_live_trading_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get live trading signals from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute("""
                        SELECT * FROM live_trading_signals 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (symbol, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM live_trading_signals 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error getting live trading signals: {e}")
            return []
    
    def get_rejected_signals(self, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get rejected signals from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute("""
                        SELECT * FROM rejected_signals 
                        WHERE symbol = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (symbol, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM rejected_signals 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"❌ Error getting rejected signals: {e}")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total signals
                cursor.execute("SELECT COUNT(*) FROM live_trading_signals")
                total_signals = cursor.fetchone()[0]
                
                # Total rejected
                cursor.execute("SELECT COUNT(*) FROM rejected_signals")
                total_rejected = cursor.fetchone()[0]
                
                # Signals by type
                cursor.execute("""
                    SELECT signal_type, COUNT(*) 
                    FROM live_trading_signals 
                    GROUP BY signal_type
                """)
                signals_by_type = dict(cursor.fetchall())
                
                # Rejections by type
                cursor.execute("""
                    SELECT rejection_type, COUNT(*) 
                    FROM rejected_signals 
                    GROUP BY rejection_type
                """)
                rejections_by_type = dict(cursor.fetchall())
                
                return {
                    'total_signals': total_signals,
                    'total_rejected': total_rejected,
                    'signals_by_type': signals_by_type,
                    'rejections_by_type': rejections_by_type,
                    'success_rate': (total_signals / (total_signals + total_rejected)) * 100 if (total_signals + total_rejected) > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {}
