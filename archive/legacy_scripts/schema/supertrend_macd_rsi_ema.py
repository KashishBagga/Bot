import sqlite3
"""
Schema module for Supertrend MACD RSI EMA strategy.
Defines the database table structure for the strategy.
"""

# Fields for the Supertrend MACD RSI EMA strategy table
supertrend_macd_rsi_ema_fields = [
    "id",
    "signal_time",
    "index_name",
    "signal",
    "strike_price",
    "price",
    "rsi",
    "macd",
    "macd_signal",
    "ema_20",
    "atr",
    "supertrend",
    "supertrend_direction",
    "confidence",
    "trade_type",
    "stop_loss",
    "target",
    "target2",
    "target3",
    "rsi_reason",
    "macd_reason",
    "price_reason",
    "option_chain_confirmation",
    "option_symbol",
    "option_expiry",
    "option_strike",
    "option_type",
    "option_entry_price",
    "outcome",
    "pnl",
    "targets_hit",
    "stoploss_count",
    "failure_reason",
    "exit_time"
]

def setup_supertrend_macd_rsi_ema_table():
    """Set up the Supertrend MACD RSI EMA strategy table."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()

    # First check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='supertrend_macd_rsi_ema'")
    if cursor.fetchone():
        # Table exists, but we need to check if it has all required columns
        cursor.execute("PRAGMA table_info(supertrend_macd_rsi_ema)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Check if supertrend column exists
        if "supertrend" not in columns:
            try:
                print("Adding missing 'supertrend' column to supertrend_macd_rsi_ema table...")
                cursor.execute("ALTER TABLE supertrend_macd_rsi_ema ADD COLUMN supertrend REAL")
                conn.commit()
            except Exception as e:
                print(f"Error adding column: {e}")
        
        # Check if supertrend_direction column exists
        if "supertrend_direction" not in columns:
            try:
                print("Adding missing 'supertrend_direction' column to supertrend_macd_rsi_ema table...")
                cursor.execute("ALTER TABLE supertrend_macd_rsi_ema ADD COLUMN supertrend_direction INTEGER")
                conn.commit()
            except Exception as e:
                print(f"Error adding column: {e}")
    else:
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS supertrend_macd_rsi_ema (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_time TEXT,
                index_name TEXT,
                signal TEXT,
                strike_price INTEGER,
                price REAL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                ema_20 REAL,
                atr REAL,
                supertrend REAL,
                supertrend_direction INTEGER,
                confidence TEXT,
                trade_type TEXT,
                stop_loss INTEGER,
                target INTEGER,
                target2 INTEGER,
                target3 INTEGER,
                rsi_reason TEXT,
                macd_reason TEXT,
                price_reason TEXT,
                option_chain_confirmation TEXT,
                option_symbol TEXT,
                option_expiry TEXT,
                option_strike INTEGER,
                option_type TEXT,
                option_entry_price REAL,
                outcome TEXT,
                pnl REAL,
                targets_hit INTEGER,
                stoploss_count INTEGER,
                failure_reason TEXT,
                exit_time TEXT
            )
        """)
        conn.commit()
    
    conn.close()
 