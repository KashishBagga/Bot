import sqlite3
import pandas as pd

tables = [
    'breakout_rsi', 'insidebar_bollinger', 'insidebar_rsi', 'supertrend_ema',
    'donchian_breakout', 'range_breakout_volatility', 'supertrend_macd_rsi_ema',
    'ema_crossover', 'ema_crossover_original'
]

conn = sqlite3.connect('trading_signals.db')

for table in tables:
    print(f'\n===== {table.upper()} =====')
    try:
        df = pd.read_sql_query(
            f"SELECT outcome, COUNT(*) as count, ROUND(AVG(pnl),2) as avg_pnl, ROUND(SUM(pnl),2) as total_pnl "
            f"FROM {table} WHERE signal != 'NO TRADE' GROUP BY outcome",
            conn
        )
        print(df.to_string(index=False))
    except Exception as e:
        print(f'Error: {e}')

conn.close() 