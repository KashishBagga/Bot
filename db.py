# db.py
import sqlite3
from datetime import datetime, timedelta
import pytz

def setup_sqlite():
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
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
            outcome TEXT,
            rsi_reason TEXT,
            macd_reason TEXT,
            price_reason TEXT,
            confidence TEXT,
            trade_type TEXT,
            option_chain_confirmation TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT,
            UNIQUE(index_name, signal_time)
        )
    """)
    conn.commit()
    conn.close()

def log_trade_sql(index_name, signal_data):
    setup_sqlite()
    atr = signal_data.get('atr', 0)
    stop_loss = int(round(atr))
    target = int(round(1.5 * atr))
    target2 = int(round(2.0 * atr))
    target3 = int(round(2.5 * atr))

    signal_time = signal_data.get('signal_time')

    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO signals (
                signal_time, index_name, signal, strike_price, stop_loss, target, target2, target3,
                price, rsi, macd, macd_signal, ema_20, atr, outcome,
                rsi_reason, macd_reason, price_reason, confidence, trade_type,
                option_chain_confirmation, pnl, targets_hit, stoploss_count, failure_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_time,
            index_name,
            signal_data.get('signal'),
            int(round(signal_data.get('price', 0) / 50) * 50),
            stop_loss,
            target,
            target2,
            target3,
            float(signal_data.get('price', 0)),
            float(signal_data.get('rsi', 0)),
            float(signal_data.get('macd', 0)),
            float(signal_data.get('macd_signal', 0)),
            float(signal_data.get('ema_20', 0)),
            float(signal_data.get('atr', 0)),
            "Pending",
            signal_data.get('rsi_reason', ''),
            signal_data.get('macd_reason', ''),
            signal_data.get('price_reason', ''),
            signal_data.get('confidence', 'Low'),
            signal_data.get('trade_type', 'Intraday'),
            signal_data.get('option_chain_confirmation', 'No'),
            0.0,
            0,
            0,
            ""
        ))
        conn.commit()
        print(f"✅ Trade logged in SQLite: {signal_data.get('signal')} at {signal_data.get('price')}")
    except Exception as e:
        print(f"❌ Failed to insert signal: {e}")
    finally:
        conn.close()

def setup_backtesting_table():
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS backtesting (
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
            outcome TEXT,
            rsi_reason TEXT,
            macd_reason TEXT,
            price_reason TEXT,
            confidence TEXT,
            trade_type TEXT,
            option_chain_confirmation TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_backtesting_sql(index_name, signal_data):
    setup_backtesting_table()
    atr = signal_data.get('atr', 0)
    stop_loss = int(round(atr))
    target = int(round(1.5 * atr))
    target2 = int(round(2.0 * atr))
    target3 = int(round(2.5 * atr))

    signal_time = signal_data.get('signal_time')

    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO backtesting (
            signal_time, index_name, signal, strike_price, stop_loss, target, target2, target3,
            price, rsi, macd, macd_signal, ema_20, atr, outcome,
            rsi_reason, macd_reason, price_reason, confidence, trade_type,
            option_chain_confirmation, pnl, targets_hit, stoploss_count, failure_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal_time,
        index_name,
        signal_data.get('signal'),
        int(round(signal_data.get('price', 0) / 50) * 50),
        stop_loss,
        target,
        target2,
        target3,
        float(signal_data.get('price', 0)),
        float(signal_data.get('rsi', 0)),
        float(signal_data.get('macd', 0)),
        float(signal_data.get('macd_signal', 0)),
        float(signal_data.get('ema_20', 0)),
        float(signal_data.get('atr', 0)),
        signal_data.get('outcome', 'Pending'),
        signal_data.get('rsi_reason', ''),
        signal_data.get('macd_reason', ''),
        signal_data.get('price_reason', ''),
        signal_data.get('confidence', 'Low'),
        signal_data.get('trade_type', 'Intraday'),
        signal_data.get('option_chain_confirmation', 'No'),
        signal_data.get('pnl', 0.0),
        signal_data.get('targets_hit', 0),
        signal_data.get('stoploss_count', 0),
        signal_data.get('failure_reason', '')
    ))
    conn.commit()
    conn.close()
    print(f"✅ Backtesting trade logged in SQLite: {signal_data.get('signal')} at {signal_data.get('price')}")
