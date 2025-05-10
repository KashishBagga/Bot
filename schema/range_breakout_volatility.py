import sqlite3

# Fields specific to range_breakout_volatility strategy
range_breakout_volatility_fields = [
    'signal_time', 'index_name', 'signal', 'price', 
    'strike_price', 'stop_loss', 'target', 'target2', 'target3',
    'atr',  # ATR is crucial for volatility measurement in this strategy
    'volatility_rank',  # Percentile rank of current ATR compared to recent history
    'range_width',  # Width of the price range that was broken
    'breakout_size',  # How far price broke out of the range (%)
    'confidence', 'trade_type',
    'price_reason',  # Used to store ATR-based reasoning
    'outcome', 'pnl', 'targets_hit', 'stoploss_count', 'failure_reason'
]

def setup_range_breakout_volatility_table():
    """
    Creates the table structure specifically for the range_breakout_volatility strategy
    """
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS range_breakout_volatility (
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
            atr REAL,
            volatility_rank REAL,
            range_width REAL,
            breakout_size REAL,
            confidence TEXT,
            trade_type TEXT,
            price_reason TEXT,
            outcome TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT
        )
    """)
    conn.commit()
    conn.close() 