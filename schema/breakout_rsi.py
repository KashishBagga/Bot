import sqlite3

# Fields specific to breakout_rsi strategy
breakout_rsi_fields = [
    'signal_time', 'index_name', 'signal', 'price', 
    'strike_price', 'stop_loss', 'target', 'target2', 'target3',
    'rsi',  # Primary indicator for this strategy
    'breakout_strength',  # How strong the breakout is (% beyond previous high/low)
    'rsi_alignment',  # Whether RSI aligns with breakout direction (confirming/diverging)
    'confidence', 'trade_type',
    'outcome', 'pnl', 'targets_hit', 'stoploss_count', 'failure_reason'
]

def setup_breakout_rsi_table():
    """
    Creates the table structure specifically for the breakout_rsi strategy
    """
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS breakout_rsi (
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
            breakout_strength REAL,
            rsi_alignment TEXT,
            confidence TEXT,
            trade_type TEXT,
            outcome TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT
        )
    """)
    conn.commit()
    conn.close() 