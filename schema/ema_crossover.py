import sqlite3

# Fields specific to ema_crossover strategy
ema_crossover_fields = [
    'signal_time', 'index_name', 'signal', 'price', 
    'strike_price', 'stop_loss', 'target', 'target2', 'target3',
    'ema_20',  # Primary indicator for this strategy
    'ema_9',  # Need to include the other EMA for crossover calculation
    'crossover_strength',  # Distance between EMAs at crossover (larger = stronger signal)
    'momentum',  # Direction and strength of momentum
    'confidence', 'trade_type',
    'outcome', 'pnl', 'targets_hit', 'stoploss_count', 'failure_reason',
    'exit_time'
]

def setup_ema_crossover_table():
    """
    Creates the table structure specifically for the ema_crossover strategy
    """
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ema_crossover (
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
            ema_20 REAL,
            ema_9 REAL,
            crossover_strength REAL,
            momentum TEXT,
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