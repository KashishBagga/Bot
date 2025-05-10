import sqlite3

# Fields specific to insidebar_bollinger strategy
insidebar_bollinger_fields = [
    'signal_time', 'index_name', 'signal', 'price', 
    'strike_price', 'stop_loss', 'target', 'target2', 'target3',
    'bollinger_width',  # Width of the Bollinger bands (volatility indicator)
    'price_to_band_ratio',  # How close price is to band edges (0-100%, 100% = at the band)
    'inside_bar_size',  # Size of inside bar relative to previous bar
    'confidence', 'trade_type',
    'price_reason',  # This stores the Bollinger Band related reasoning
    'outcome', 'pnl', 'targets_hit', 'stoploss_count', 'failure_reason'
]

def setup_insidebar_bollinger_table():
    """
    Creates the table structure specifically for the insidebar_bollinger strategy
    """
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS insidebar_bollinger (
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
            bollinger_width REAL,
            price_to_band_ratio REAL,
            inside_bar_size REAL,
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