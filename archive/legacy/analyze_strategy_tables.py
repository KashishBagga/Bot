import sqlite3
import pandas as pd

def get_strategy_tables():
    """Get a list of strategy tables from the database"""
    conn = sqlite3.connect('trading_signals.db')
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [table[0] for table in cursor.fetchall()]
    
    # Filter to only include strategy tables (those with signal and outcome columns)
    strategy_tables = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        if 'signal' in columns and 'outcome' in columns:
            strategy_tables.append(table)
    
    conn.close()
    return strategy_tables

conn = sqlite3.connect('trading_signals.db')

# Get strategy tables dynamically
for table in get_strategy_tables():
    print(f'\n===== {table.upper()} =====')
    try:
        df = pd.read_sql_query(
            f"SELECT outcome, COUNT(*) as count, ROUND(AVG(pnl),2) as avg_pnl, ROUND(SUM(pnl),2) as total_pnl "
            f"FROM {table} WHERE signal != 'NO TRADE' GROUP BY outcome",
            conn
        )
        print(df.to_string(index=False))
    except Exception as e:
        print(f'Error: {e}')

conn.close() 