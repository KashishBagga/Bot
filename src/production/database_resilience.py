#!/usr/bin/env python3
"""
Atomic Transactions and DB Resilience
MUST #3: Atomic transactions and automated backup/restore
"""

import sys
import os
import time
import sqlite3
import hashlib
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from typing import Any
from dataclasses import dataclass
from contextlib import contextmanager
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BackupConfig:
    """Backup configuration"""
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    backup_directory: str = "backups"
    checksum_algorithm: str = "sha256"
    restore_test_interval_days: int = 7
    max_backup_size_mb: int = 1000

@dataclass
class TransactionResult:
    """Transaction result"""
    success: bool
    transaction_id: str
    affected_rows: int
    error_message: Optional[str] = None
    execution_time: float = 0.0

class DatabaseResilienceManager:
    """Database resilience and backup manager"""
    
    def __init__(self, db_path: str, config: BackupConfig):
        self.db_path = db_path
        self.config = config
        self.backup_directory = os.path.join(os.path.dirname(db_path), config.backup_directory)
        self.last_backup = None
        self.last_restore_test = None
        
        # Create backup directory
        os.makedirs(self.backup_directory, exist_ok=True)
        
        # Initialize database with atomic transaction support
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with atomic transaction support"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys=ON")
                
                # Set synchronous mode for safety
                conn.execute("PRAGMA synchronous=FULL")
                
                # Create transaction log table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS transaction_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        transaction_id TEXT UNIQUE NOT NULL,
                        operation_type TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        affected_rows INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        success BOOLEAN NOT NULL,
                        error_message TEXT,
                        execution_time REAL
                    )
                ''')
                
                # Create backup metadata table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS backup_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        backup_file TEXT UNIQUE NOT NULL,
                        backup_timestamp TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        checksum TEXT NOT NULL,
                        restore_tested BOOLEAN DEFAULT FALSE,
                        restore_test_timestamp TEXT
                    )
                ''')
                
                conn.commit()
                logger.info("✅ Database initialized with atomic transaction support")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    @contextmanager
    def atomic_transaction(self, transaction_id: str = None):
        """Context manager for atomic transactions"""
        if transaction_id is None:
            transaction_id = f"TXN_{int(time.time())}"
        
        start_time = time.time()
        conn = None
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("BEGIN IMMEDIATE")
            
            yield conn, transaction_id
            
            conn.commit()
            execution_time = time.time() - start_time
            
            # Log successful transaction
            self._log_transaction(transaction_id, "SUCCESS", 0, None, execution_time)
            
        except Exception as e:
            if conn:
                conn.rollback()
            
            execution_time = time.time() - start_time
            error_message = str(e)
            
            # Log failed transaction
            self._log_transaction(transaction_id, "FAILED", 0, error_message, execution_time)
            
            logger.error(f"❌ Transaction failed: {transaction_id} - {error_message}")
            raise
            
        finally:
            if conn:
                conn.close()
    
    def execute_atomic_trade_transaction(self, trade_data: Dict[str, Any]) -> TransactionResult:
        """Execute atomic trade transaction (open/close trade)"""
        transaction_id = f"TRADE_{int(time.time())}"
        start_time = time.time()
        
        try:
            with self.atomic_transaction(transaction_id) as (conn, txn_id):
                cursor = conn.cursor()
                affected_rows = 0
                
                # Step 1: Update account balance
                if 'balance_change' in trade_data:
                    cursor.execute('''
                        UPDATE account_balance 
                        SET balance = balance + ?, 
                            last_updated = ?
                        WHERE account_id = ?
                    ''', (trade_data['balance_change'], datetime.now().isoformat(), trade_data['account_id']))
                    affected_rows += cursor.rowcount
                
                # Step 2: Insert/update trade record
                if trade_data['operation'] == 'OPEN':
                    cursor.execute('''
                        INSERT INTO open_trades (
                            trade_id, symbol, signal, entry_price, quantity, 
                            timestamp, strategy, confidence, stop_loss, take_profit
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade_data['trade_id'], trade_data['symbol'], trade_data['signal'],
                        trade_data['entry_price'], trade_data['quantity'], 
                        trade_data['timestamp'], trade_data['strategy'],
                        trade_data['confidence'], trade_data.get('stop_loss'),
                        trade_data.get('take_profit')
                    ))
                    affected_rows += cursor.rowcount
                
                elif trade_data['operation'] == 'CLOSE':
                    # Close trade
                    cursor.execute('''
                        UPDATE open_trades 
                        SET exit_price = ?, exit_timestamp = ?, pnl = ?
                        WHERE trade_id = ?
                    ''', (
                        trade_data['exit_price'], trade_data['exit_timestamp'],
                        trade_data['pnl'], trade_data['trade_id']
                    ))
                    affected_rows += cursor.rowcount
                    
                    # Move to closed trades
                    cursor.execute('''
                        INSERT INTO closed_trades 
                        SELECT *, ? as exit_price, ? as exit_timestamp, ? as pnl
                        FROM open_trades 
                        WHERE trade_id = ?
                    ''', (
                        trade_data['exit_price'], trade_data['exit_timestamp'],
                        trade_data['pnl'], trade_data['trade_id']
                    ))
                    affected_rows += cursor.rowcount
                    
                    # Remove from open trades
                    cursor.execute('DELETE FROM open_trades WHERE trade_id = ?', (trade_data['trade_id'],))
                    affected_rows += cursor.rowcount
                
                execution_time = time.time() - start_time
                
                return TransactionResult(
                    success=True,
                    transaction_id=transaction_id,
                    affected_rows=affected_rows,
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            return TransactionResult(
                success=False,
                transaction_id=transaction_id,
                affected_rows=0,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _log_transaction(self, transaction_id: str, operation_type: str, 
                        affected_rows: int, error_message: Optional[str], execution_time: float):
        """Log transaction to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO transaction_log (
                        transaction_id, operation_type, table_name, affected_rows,
                        timestamp, success, error_message, execution_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transaction_id, operation_type, 'multiple',
                    affected_rows, datetime.now().isoformat(),
                    error_message is None, error_message, execution_time
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to log transaction: {e}")
    
    def create_backup(self) -> bool:
        """Create database backup with checksum"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"trading_db_backup_{timestamp}.db"
            backup_path = os.path.join(self.backup_directory, backup_filename)
            
            logger.info(f"📦 Creating backup: {backup_filename}")
            
            # Create backup
            shutil.copy2(self.db_path, backup_path)
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_path)
            
            # Get file size
            file_size = os.path.getsize(backup_path)
            
            # Store backup metadata
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO backup_metadata (
                        backup_file, backup_timestamp, file_size, checksum
                    ) VALUES (?, ?, ?, ?)
                ''', (backup_filename, datetime.now().isoformat(), file_size, checksum))
                conn.commit()
            
            self.last_backup = datetime.now()
            logger.info(f"✅ Backup created successfully: {backup_filename} ({file_size} bytes, checksum: {checksum[:16]}...)")
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup creation failed: {e}")
            return False
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT backup_file FROM backup_metadata 
                    WHERE backup_timestamp < ?
                ''', (cutoff_date.isoformat(),))
                
                old_backups = cursor.fetchall()
                
                for (backup_file,) in old_backups:
                    backup_path = os.path.join(self.backup_directory, backup_file)
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                        logger.info(f"🗑️ Removed old backup: {backup_file}")
                
                # Remove metadata
                cursor.execute('''
                    DELETE FROM backup_metadata 
                    WHERE backup_timestamp < ?
                ''', (cutoff_date.isoformat(),))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Backup cleanup failed: {e}")
    
    def restore_backup(self, backup_filename: str) -> bool:
        """Restore database from backup"""
        try:
            backup_path = os.path.join(self.backup_directory, backup_filename)
            
            if not os.path.exists(backup_path):
                logger.error(f"❌ Backup file not found: {backup_filename}")
                return False
            
            logger.info(f"🔄 Restoring backup: {backup_filename}")
            
            # Create temporary backup of current database
            temp_backup = f"{self.db_path}.temp_backup"
            shutil.copy2(self.db_path, temp_backup)
            
            try:
                # Restore from backup
                shutil.copy2(backup_path, self.db_path)
                
                # Verify checksum
                expected_checksum = self._get_backup_checksum(backup_filename)
                actual_checksum = self._calculate_checksum(self.db_path)
                
                if expected_checksum != actual_checksum:
                    raise Exception("Checksum verification failed")
                
                # Test database integrity
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA integrity_check")
                    result = cursor.fetchone()
                    if result[0] != "ok":
                        raise Exception(f"Database integrity check failed: {result[0]}")
                
                logger.info(f"✅ Backup restored successfully: {backup_filename}")
                
                # Update restore test status
                self._update_restore_test_status(backup_filename, True)
                
                return True
                
            except Exception as e:
                # Restore original database
                shutil.copy2(temp_backup, self.db_path)
                logger.error(f"❌ Backup restore failed, original database restored: {e}")
                return False
                
            finally:
                # Clean up temporary backup
                if os.path.exists(temp_backup):
                    os.remove(temp_backup)
                    
        except Exception as e:
            logger.error(f"❌ Backup restore failed: {e}")
            return False
    
    def _get_backup_checksum(self, backup_filename: str) -> str:
        """Get checksum for backup file"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT checksum FROM backup_metadata 
                WHERE backup_file = ?
            ''', (backup_filename,))
            result = cursor.fetchone()
            return result[0] if result else ""
    
    def _update_restore_test_status(self, backup_filename: str, success: bool):
        """Update restore test status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE backup_metadata 
                    SET restore_tested = ?, restore_test_timestamp = ?
                    WHERE backup_file = ?
                ''', (success, datetime.now().isoformat(), backup_filename))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to update restore test status: {e}")
    
    def run_restore_test(self) -> bool:
        """Run restore test on latest backup"""
        try:
            # Get latest backup
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT backup_file FROM backup_metadata 
                    ORDER BY backup_timestamp DESC LIMIT 1
                ''')
                result = cursor.fetchone()
                
                if not result:
                    logger.warning("⚠️ No backups available for restore test")
                    return False
                
                backup_filename = result[0]
            
            # Run restore test
            success = self.restore_backup(backup_filename)
            
            if success:
                self.last_restore_test = datetime.now()
                logger.info("✅ Restore test completed successfully")
            else:
                logger.error("❌ Restore test failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Restore test failed: {e}")
            return False
    
    def start_automated_backup(self):
        """Start automated backup process"""
        try:
            while True:
                # Check if backup is needed
                if (self.last_backup is None or 
                    datetime.now() - self.last_backup > timedelta(hours=self.config.backup_interval_hours)):
                    
                    self.create_backup()
                
                # Check if restore test is needed
                if (self.last_restore_test is None or 
                    datetime.now() - self.last_restore_test > timedelta(days=self.config.restore_test_interval_days)):
                    
                    self.run_restore_test()
                
                # Wait before next check
                time.sleep(3600)  # Check every hour
                
        except KeyboardInterrupt:
            logger.info("🛑 Automated backup stopped")
        except Exception as e:
            logger.error(f"❌ Automated backup error: {e}")
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get backup status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get backup count
                cursor.execute('SELECT COUNT(*) FROM backup_metadata')
                backup_count = cursor.fetchone()[0]
                
                # Get latest backup
                cursor.execute('''
                    SELECT backup_file, backup_timestamp, file_size, checksum, restore_tested
                    FROM backup_metadata 
                    ORDER BY backup_timestamp DESC LIMIT 1
                ''')
                latest_backup = cursor.fetchone()
                
                # Get total backup size
                cursor.execute('SELECT SUM(file_size) FROM backup_metadata')
                total_size = cursor.fetchone()[0] or 0
                
                return {
                    'backup_count': backup_count,
                    'latest_backup': {
                        'filename': latest_backup[0] if latest_backup else None,
                        'timestamp': latest_backup[1] if latest_backup else None,
                        'size': latest_backup[2] if latest_backup else 0,
                        'checksum': latest_backup[3] if latest_backup else None,
                        'restore_tested': latest_backup[4] if latest_backup else False
                    },
                    'total_size_mb': total_size / (1024 * 1024),
                    'last_backup': self.last_backup.isoformat() if self.last_backup else None,
                    'last_restore_test': self.last_restore_test.isoformat() if self.last_restore_test else None
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get backup status: {e}")
            return {}
    
    def check_acceptance_criteria(self) -> bool:
        """Check if acceptance criteria are met"""
        try:
            # Check if backup and restore can be completed in < 5 minutes
            start_time = time.time()
            
            # Create test backup
            backup_success = self.create_backup()
            if not backup_success:
                return False
            
            # Get latest backup and test restore
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT backup_file FROM backup_metadata 
                    ORDER BY backup_timestamp DESC LIMIT 1
                ''')
                result = cursor.fetchone()
                
                if result:
                    restore_success = self.restore_backup(result[0])
                    if not restore_success:
                        return False
            
            total_time = time.time() - start_time
            
            # Acceptance criteria: backup + restore < 5 minutes
            return total_time < 300  # 5 minutes
            
        except Exception as e:
            logger.error(f"❌ Acceptance criteria check failed: {e}")
            return False

def main():
    """Main function for testing"""
    config = BackupConfig(
        backup_interval_hours=1,  # Test with 1 hour interval
        backup_retention_days=7,
        restore_test_interval_days=1
    )
    
    db_manager = DatabaseResilienceManager("data/trading.db", config)
    
    # Test atomic transaction
    trade_data = {
        'operation': 'OPEN',
        'trade_id': f"TEST_{int(time.time())}",
        'symbol': 'NSE:NIFTY50-INDEX',
        'signal': 'BUY',
        'entry_price': 19500,
        'quantity': 100,
        'timestamp': datetime.now().isoformat(),
        'strategy': 'test_strategy',
        'confidence': 75,
        'account_id': 'test_account',
        'balance_change': -1950000  # Cost of trade
    }
    
    result = db_manager.execute_atomic_trade_transaction(trade_data)
    print(f"✅ Atomic transaction result: {result}")
    
    # Test backup
    backup_success = db_manager.create_backup()
    print(f"✅ Backup created: {backup_success}")
    
    # Test restore
    restore_success = db_manager.run_restore_test()
    print(f"✅ Restore test: {restore_success}")
    
    # Check acceptance criteria
    criteria_met = db_manager.check_acceptance_criteria()
    print(f"✅ Acceptance criteria met: {criteria_met}")
    
    # Get status
    status = db_manager.get_backup_status()
    print(f"📊 Backup status: {status}")

if __name__ == "__main__":
    main()
