import sqlite3
"""
Schema module for insidebar_bollinger strategy.
Defines the database table structure for the strategy.
"""

# Fields for the insidebar_bollinger strategy table
insidebar_bollinger_fields = [
    "id",
    "signal_time",
    "index_name",
    "signal",
    "price",
    "bollinger_width",
    "price_to_band_ratio",
    "inside_bar_size",
    "confidence",
    "price_reason",
    "trade_type",
    "rsi_reason",
    "macd_reason",
    "outcome",
    "pnl",
    "targets_hit",
    "stoploss_count",
    "failure_reason"
]

def setup_insidebar_bollinger_table():
    """Set up the insidebar_bollinger strategy table."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insidebar_bollinger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            index_name TEXT,
            signal TEXT,
            price REAL,
            bollinger_width REAL,
            price_to_band_ratio REAL,
            inside_bar_size REAL,
            confidence TEXT,
            price_reason TEXT,
            trade_type TEXT,
            rsi_reason TEXT,
            macd_reason TEXT,
            outcome TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT
        )
    """)
    conn.commit()
    conn.close()

