"""
Generic schema module for strategy data.
Used as a fallback when specific strategy schemas aren't available.
"""
import sqlite3

# Generic fields for strategy tables
generic_fields = [
    "id",
    "signal_time",
    "index_name",
    "signal",
    "price",
    "rsi",
    "macd",
    "macd_signal",
    "ema_20",
    "atr",
    "outcome",
    "confidence",
    "trade_type",
    "pnl",
    "exit_time"
]

def setup_generic_table():
    """Set up a generic strategy table."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            index_name TEXT,
            signal TEXT,
            price REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            ema_20 REAL,
            atr REAL,
            outcome TEXT,
            confidence TEXT,
            trade_type TEXT,
            pnl REAL,
            exit_time TEXT
        )
    """)
    conn.commit()
    conn.close() 