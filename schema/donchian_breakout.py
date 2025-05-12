import sqlite3

# Fields specific to donchian_breakout strategy - uses high/low channel breakouts
donchian_breakout_fields = [
    'signal_time', 'index_name', 'signal', 'price', 
    'strike_price', 'stop_loss', 'target', 'target2', 'target3',
    'channel_width',  # Width of the Donchian channel (high - low)
    'breakout_size',  # How far price broke out beyond the channel (%)
    'volume_ratio',  # Volume compared to average (breakout confirmation)
    'confidence', 'trade_type',
    'outcome', 'pnl', 'targets_hit', 'stoploss_count', 'failure_reason',
    'exit_time'
]

def setup_donchian_breakout_table():
    """
    Creates the table structure specifically for the donchian_breakout strategy
    """
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS donchian_breakout (
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
            channel_width REAL,
            breakout_size REAL,
            volume_ratio REAL,
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