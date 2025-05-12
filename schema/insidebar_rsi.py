import sqlite3

# Fields specific to insidebar_rsi strategy
insidebar_rsi_fields = [
    'signal_time', 'index_name', 'signal', 'price', 
    'strike_price', 'stop_loss', 'target', 'target2', 'target3',
    'rsi',  # Primary indicator for this strategy
    'rsi_level',  # Store the RSI level category (Extreme Oversold, Oversold, Neutral, Overbought, Extreme Overbought)
    'confidence', 'trade_type',
    'outcome', 'pnl', 'targets_hit', 'stoploss_count', 'failure_reason',
    'exit_time'
]

def setup_insidebar_rsi_table():
    """
    Creates the table structure specifically for the insidebar_rsi strategy
    """
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insidebar_rsi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            index_name TEXT,
            signal TEXT,
            strike_price INTEGER,
            stop_loss INTEGER,
            target INTEGER,
            target2 INTEGER,
            target3 INTEGER,
            price REAL,
            rsi REAL,
            rsi_level TEXT,
            confidence TEXT,
            trade_type TEXT,
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