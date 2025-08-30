"""
Unified Trading Database
A comprehensive database system that combines live trading, backtesting, and analytics
in a single database with proper relationships and naming conventions.
"""

import sqlite3
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class UnifiedTradingDatabase:
    """Unified database for all trading and backtesting data."""
    
    def __init__(self, db_path: str = "unified_trading.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing tables
                cursor.execute("DROP TABLE IF EXISTS trading_signals")
                cursor.execute("DROP TABLE IF EXISTS rejected_signals")
                cursor.execute("DROP TABLE IF EXISTS open_option_positions")
                cursor.execute("DROP TABLE IF EXISTS closed_option_positions")
                cursor.execute("DROP TABLE IF EXISTS equity_curve")
                cursor.execute("DROP TABLE IF EXISTS performance_metrics")
                
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
                
                # Capital rejection logs
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS capital_rejection_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        required_capital REAL NOT NULL,
                        available_capital REAL NOT NULL,
                        capital_shortfall REAL NOT NULL,
                        option_premium REAL NOT NULL,
                        lot_size INTEGER NOT NULL,
                        total_cost_per_lot REAL NOT NULL,
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
                logger.debug(f"✅ Saved live trading signal for {signal_data['strategy']}")
        except Exception as e:
            logger.error(f"❌ Error saving live trading signal: {e}")

    def save_rejected_signal(self, rejection_data: Dict):
        """Save rejected signal to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO rejected_signals 
                    (timestamp, symbol, strategy, signal_type, confidence, rejection_reason, 
                     rejection_type, additional_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rejection_data['timestamp'],
                    rejection_data['symbol'],
                    rejection_data['strategy'],
                    rejection_data['signal_type'],
                    rejection_data['confidence'],
                    rejection_data['rejection_reason'],
                    rejection_data['rejection_type'],
                    json.dumps(rejection_data.get('additional_data', {}))
                ))
                conn.commit()
                logger.debug(f"✅ Saved rejected signal for {rejection_data['strategy']}")
        except Exception as e:
            logger.error(f"❌ Error saving rejected signal: {e}")

    def save_capital_rejection_log(self, rejection_data: Dict):
        """Save capital rejection log to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO capital_rejection_logs 
                    (timestamp, symbol, strategy, signal_type, confidence, 
                     required_capital, available_capital, capital_shortfall,
                     option_premium, lot_size, total_cost_per_lot)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rejection_data['timestamp'],
                    rejection_data['symbol'],
                    rejection_data['strategy'],
                    rejection_data['signal_type'],
                    rejection_data['confidence'],
                    rejection_data['required_capital'],
                    rejection_data['available_capital'],
                    rejection_data['capital_shortfall'],
                    rejection_data['option_premium'],
                    rejection_data['lot_size'],
                    rejection_data['total_cost_per_lot']
                ))
                conn.commit()
                logger.debug(f"✅ Saved capital rejection log for {rejection_data['strategy']}")
        except Exception as e:
            logger.error(f"❌ Error saving capital rejection log: {e}")

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get rejected signals count
                cursor.execute("""
                    SELECT COUNT(*) FROM rejected_signals
                """)
                rejected_signals = cursor.fetchone()[0]
                
                # Get capital rejections count
                cursor.execute("""
                    SELECT COUNT(*) FROM capital_rejection_logs
                """)
                capital_rejections = cursor.fetchone()[0]
                
                summary = {
                    'current_status': {
                        'rejected_signals': rejected_signals,
                        'capital_rejections': capital_rejections
                    }
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"❌ Error getting performance summary: {e}")
            return {}

    def get_rejection_summary(self) -> Dict:
        """Get summary of all rejections."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get rejection counts by type
                cursor.execute("""
                    SELECT rejection_type, COUNT(*) as count
                    FROM rejected_signals
                    GROUP BY rejection_type
                """)
                rejection_counts = dict(cursor.fetchall())
                
                # Get capital rejection summary
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_rejections,
                        AVG(capital_shortfall) as avg_shortfall,
                        SUM(capital_shortfall) as total_shortfall
                    FROM capital_rejection_logs
                """)
                capital_summary = cursor.fetchone()
                
                return {
                    'rejection_counts': rejection_counts,
                    'capital_rejections': {
                        'total': capital_summary[0] if capital_summary else 0,
                        'avg_shortfall': capital_summary[1] if capital_summary else 0,
                        'total_shortfall': capital_summary[2] if capital_summary else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting rejection summary: {e}")
            return {}

# Legacy compatibility
class UnifiedDatabase(UnifiedTradingDatabase):
    """Legacy class for backward compatibility."""
    pass
