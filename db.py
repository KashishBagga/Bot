# db.py
import sqlite3
from datetime import datetime, timedelta
import pytz
import importlib
import os

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

def get_table_columns(cursor, table_name):
    """Get a list of column names for a given table"""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        return [col[1] for col in cursor.fetchall()]
    except:
        return []

def log_strategy_sql(strategy_name, signal_data):
    """
    Log strategy signals to the corresponding table in the database.
    Validates schema and removes fields that don't exist in the table.
    """
    # Skip logging if this is a NO TRADE signal or None
    signal = signal_data.get('signal')
    if not signal or signal == 'NO TRADE' or signal == 'None':
        return
        
    # Handle strategy name backward compatibility
    original_strategy_name = strategy_name
    
    # Remove 'strategy_' prefix if it exists, for schema module lookup
    if strategy_name.startswith('strategy_'):
        strategy_name = strategy_name[9:]  # Remove 'strategy_' prefix for newer files
    
    # Dynamically import the appropriate schema module
    try:
        schema_module = importlib.import_module(f"src.models.schema.{strategy_name}")
        setup_func = getattr(schema_module, f"setup_{strategy_name}_table")
        fields_list = getattr(schema_module, f"{strategy_name}_fields")
    except (ImportError, AttributeError) as e:
        # Try with 'strategy_' prefix as fallback for backward compatibility
        try:
            schema_module = importlib.import_module(f"src.models.schema.strategy_{strategy_name}")
            setup_func = getattr(schema_module, f"setup_strategy_{strategy_name}_table")
            fields_list = getattr(schema_module, f"strategy_{strategy_name}_fields")
            
            # If we found it with the prefix, update the strategy name for table name
            strategy_name = f"strategy_{strategy_name}"
        except (ImportError, AttributeError):
            print(f"Error loading schema for {original_strategy_name}: {e}")
            print(f"Creating generic schema for {original_strategy_name}")
            # Fall back to a generic setup if specific schema not found
            from src.models.schema.generic import setup_generic_table, generic_fields
            setup_func = setup_generic_table
            fields_list = generic_fields
            
            # Use the original strategy name for the table
            strategy_name = original_strategy_name
    
    # Call the setup function to ensure the table exists
    setup_func()
    
    # Calculate essential trading values if they're missing
    
    # 1. ATR-based calculations for stop_loss and targets
    atr = signal_data.get('atr', 0)
    price = signal_data.get('price', 0)
    
    # Add these to signal_data if not already present with realistic values
    if 'strike_price' not in signal_data or not signal_data['strike_price']:
        signal_data['strike_price'] = int(round(price / 50) * 50) if price > 0 else 0
    
    # Set stop-loss based on ATR or percentage of price if ATR is unavailable
    if 'stop_loss' not in signal_data or not signal_data['stop_loss']:
        if atr > 0:
            signal_data['stop_loss'] = int(round(atr))
        else:
            # Use 0.5% of price as fallback stop-loss
            signal_data['stop_loss'] = int(round(price * 0.005))
    
    # Set targets based on stop-loss with realistic risk:reward ratios
    if 'target' not in signal_data or not signal_data['target']:
        signal_data['target'] = int(round(1.5 * signal_data['stop_loss']))
    
    if 'target2' not in signal_data or not signal_data['target2']:
        signal_data['target2'] = int(round(2.0 * signal_data['stop_loss']))
    
    if 'target3' not in signal_data or not signal_data['target3']:
        signal_data['target3'] = int(round(2.5 * signal_data['stop_loss']))
    
    # 2. Performance metrics
    # If these are missing, initialize with default values
    if 'pnl' not in signal_data or not signal_data['pnl']:
        signal = signal_data.get('signal', 'NO TRADE')
        if signal != "NO TRADE":
            # Estimate PnL based on historical performance (example calculation)
            if 'outcome' in signal_data and signal_data['outcome'] == 'Success':
                signal_data['pnl'] = signal_data['target'] * 1.5  # Average between target1 and target2
            elif 'outcome' in signal_data and signal_data['outcome'] == 'Failure':
                signal_data['pnl'] = -signal_data['stop_loss']
            else:
                signal_data['pnl'] = 0.0
    
    if 'targets_hit' not in signal_data or not signal_data['targets_hit']:
        if 'outcome' in signal_data and signal_data['outcome'] == 'Success':
            signal_data['targets_hit'] = 1  # Assume at least one target hit for successful trades
        else:
            signal_data['targets_hit'] = 0
    
    if 'stoploss_count' not in signal_data or not signal_data['stoploss_count']:
        if 'outcome' in signal_data and signal_data['outcome'] == 'Failure':
            signal_data['stoploss_count'] = 1
        else:
            signal_data['stoploss_count'] = 0
    
    # 3. Failure reason
    if ('failure_reason' not in signal_data or not signal_data['failure_reason']) and 'outcome' in signal_data and signal_data['outcome'] == 'Failure':
        # Generate a meaningful failure reason based on available data
        if strategy_name == 'supertrend_macd_rsi_ema':
            if 'rsi' in signal_data and (signal_data['rsi'] > 70 or signal_data['rsi'] < 30):
                signal_data['failure_reason'] = "RSI extremes may have caused price reversal"
            elif 'macd' in signal_data and 'macd_signal' in signal_data:
                signal_data['failure_reason'] = "MACD divergence with price action"
            else:
                signal_data['failure_reason'] = "Price moved against expected trend"
        elif 'rsi' in strategy_name.lower():
            signal_data['failure_reason'] = "RSI failed to indicate proper momentum"
        elif 'ema' in strategy_name.lower():
            signal_data['failure_reason'] = "Price reversed at EMA rejection"
        elif 'bollinger' in strategy_name.lower():
            signal_data['failure_reason'] = "Price failed to respect Bollinger Band boundaries"
        else:
            signal_data['failure_reason'] = "Target not reached before stop-loss hit"
    
    # Connect to the database
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    # Get the list of actual columns in the table
    actual_columns = get_table_columns(cursor, strategy_name)
    
    if not actual_columns:
        print(f"Error: Table {strategy_name} not found in database")
        # Create a basic table if it doesn't exist
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {strategy_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_time TEXT,
                index_name TEXT,
                signal TEXT,
                price REAL,
                confidence TEXT
            )
        """)
        conn.commit()
        actual_columns = ["id", "signal_time", "index_name", "signal", "price", "confidence"]
    
    # Make sure we have the required fields
    if "time" in signal_data and "signal_time" not in signal_data:
        signal_data["signal_time"] = signal_data["time"]
    
    if "signal_time" not in signal_data:
        signal_data["signal_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if "index_name" not in signal_data:
        signal_data["index_name"] = "UNKNOWN"
    
    # Filter signal_data to only include fields that exist in the table
    filtered_data = {}
    for field in signal_data:
        if field in actual_columns:
            filtered_data[field] = signal_data[field]
    
    # Add missing required fields with default values
    for col in ["signal", "price", "confidence"]:
        if col not in filtered_data and col in actual_columns:
            if col == "signal":
                filtered_data[col] = "NO TRADE"
            elif col == "price":
                filtered_data[col] = 0.0
            elif col == "confidence":
                filtered_data[col] = "Low"
    
    # Prepare columns and values for SQL
    columns = list(filtered_data.keys())
    placeholders = ', '.join(['?'] * len(columns))
    columns_str = ', '.join(columns)
    
    try:
        # Execute the SQL query
        cursor.execute(f"""
            INSERT INTO {strategy_name} ({columns_str})
            VALUES ({placeholders})
        """, tuple(filtered_data[col] for col in columns))
        conn.commit()
        # print(f"✅ Strategy {strategy_name} logged in SQLite: {signal_data.get('signal')} at {signal_data.get('price')}")
    except Exception as e:
        print(f"❌ Error inserting data into {strategy_name}: {e}")
    finally:
        conn.close()
