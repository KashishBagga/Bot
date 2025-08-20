#!/usr/bin/env python3
"""
Database optimization utility

This script analyzes the trading_signals.db database and:
1. Updates table schemas to ensure all required columns exist
2. Adds indexes for faster queries
3. Runs ANALYZE and VACUUM to optimize performance
4. Reports on database statistics

Usage:
    python optimize_db.py
"""
import os
import sqlite3
import sys
import time


def get_table_list(cursor):
    """Get a list of all tables in the database"""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor.fetchall()]


def get_table_columns(cursor, table_name):
    """Get a list of columns for a specified table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [col[1] for col in cursor.fetchall()]


def get_table_row_count(cursor, table_name):
    """Get the number of rows in a table"""
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]


def ensure_essential_columns(cursor, table_name):
    """Ensure essential columns exist in the table"""
    essential_columns = {
        "id": "INTEGER PRIMARY KEY",
        "signal_time": "TEXT",
        "index_name": "TEXT",
        "signal": "TEXT",
        "price": "REAL",
    }
    
    columns = get_table_columns(cursor, table_name)
    
    for col_name, col_type in essential_columns.items():
        if col_name not in columns and col_name != "id":  # Can't add a primary key
            try:
                print(f"Adding missing essential column '{col_name}' to {table_name}")
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
            except sqlite3.Error as e:
                print(f"Error adding column: {e}")


def add_strategy_specific_columns(cursor, table_name):
    """Add strategy-specific columns based on table name"""
    columns = get_table_columns(cursor, table_name)
    
    # Columns for strategies with "ema" in the name
    if "ema" in table_name.lower() and "crossover" in table_name.lower():
        for col_name, col_type in [
            ("ema_fast", "REAL"),
            ("ema_slow", "REAL"),
            ("crossover_strength", "REAL"),
            ("rsi", "REAL")
        ]:
            if col_name not in columns:
                try:
                    print(f"Adding strategy-specific column '{col_name}' to {table_name}")
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                except sqlite3.Error as e:
                    print(f"Error adding column: {e}")
    
    # Columns for strategies with "supertrend" in the name
    elif "supertrend" in table_name.lower():
        for col_name, col_type in [
            ("supertrend", "REAL"),
            ("supertrend_direction", "INTEGER")
        ]:
            if col_name not in columns:
                try:
                    print(f"Adding strategy-specific column '{col_name}' to {table_name}")
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                except sqlite3.Error as e:
                    print(f"Error adding column: {e}")
    
    # Columns for strategies with "rsi" in the name
    elif "rsi" in table_name.lower() and not "macd" in table_name.lower():
        for col_name, col_type in [
            ("rsi", "REAL"),
            ("rsi_upper", "REAL"),
            ("rsi_lower", "REAL")
        ]:
            if col_name not in columns:
                try:
                    print(f"Adding strategy-specific column '{col_name}' to {table_name}")
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
                except sqlite3.Error as e:
                    print(f"Error adding column: {e}")
    
    # Common trade-related columns all strategies should have
    common_columns = {
        "stop_loss": "INTEGER",
        "target": "INTEGER",
        "confidence": "TEXT",
        "outcome": "TEXT",
        "pnl": "REAL"
    }
    
    for col_name, col_type in common_columns.items():
        if col_name not in columns:
            try:
                print(f"Adding common trade column '{col_name}' to {table_name}")
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}")
            except sqlite3.Error as e:
                print(f"Error adding column: {e}")


def create_indexes(cursor, table_name):
    """Create indexes for faster queries"""
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_signal_time ON {table_name}(signal_time)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_index_name ON {table_name}(index_name)")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_signal ON {table_name}(signal)")
    print(f"Created indexes for {table_name}")


def optimize_database():
    """Main function to optimize the database"""
    db_path = "trading_signals.db"
    
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found")
        return False
    
    print(f"Optimizing database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get list of tables
    tables = get_table_list(cursor)
    
    if not tables:
        print("No tables found in the database")
        conn.close()
        return False
    
    print(f"Found {len(tables)} tables")
    
    # Process each table
    for table_name in tables:
        # Skip SQLite internal tables
        if table_name.startswith('sqlite_'):
            continue
        
        row_count = get_table_row_count(cursor, table_name)
        print(f"\nProcessing table: {table_name} ({row_count} rows)")
        
        # Ensure essential columns exist
        ensure_essential_columns(cursor, table_name)
        
        # Add strategy-specific columns
        add_strategy_specific_columns(cursor, table_name)
        
        # Create indexes for faster queries
        create_indexes(cursor, table_name)
    
    # Run ANALYZE to gather statistics
    print("\nRunning ANALYZE...")
    cursor.execute("ANALYZE")
    
    # Run VACUUM to compact the database
    print("Running VACUUM...")
    cursor.execute("VACUUM")
    
    # Final statistics
    print("\nDatabase Optimization Complete")
    print("\nFinal statistics:")
    for table_name in tables:
        if table_name.startswith('sqlite_'):
            continue
        row_count = get_table_row_count(cursor, table_name)
        print(f"  - {table_name}: {row_count} rows")
    
    conn.close()
    return True


if __name__ == "__main__":
    start_time = time.time()
    success = optimize_database()
    end_time = time.time()
    
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    sys.exit(0 if success else 1) 