import sqlite3

# Fields specific to supertrend_macd_rsi_ema strategy
supertrend_macd_rsi_ema_fields = [
    'signal_time', 'index_name', 'signal', 'price', 
    'strike_price', 'stop_loss', 'target', 'target2', 'target3',
    'rsi', 'macd', 'macd_signal', 'ema_20', 'atr',
    'rsi_reason', 'macd_reason', 'price_reason',  # These are unique to this strategy
    'confidence', 'trade_type', 'option_chain_confirmation',
    'outcome', 'pnl', 'targets_hit', 'stoploss_count', 'failure_reason',
    'exit_time'
]

def setup_supertrend_macd_rsi_ema_table():
    """
    Creates the table structure specifically for the supertrend_macd_rsi_ema strategy
    """
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS supertrend_macd_rsi_ema (
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
            macd REAL,
            macd_signal REAL,
            ema_20 REAL,
            atr REAL,
            rsi_reason TEXT,
            macd_reason TEXT,
            price_reason TEXT,
            outcome TEXT,
            confidence TEXT,
            trade_type TEXT,
            option_chain_confirmation TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT,
            exit_time TEXT
        )
    """)
    conn.commit()
    conn.close() 