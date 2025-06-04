# db.py
import sqlite3
from datetime import datetime, timedelta
import pytz
import importlib
import os

# Define the timezone for consistent timestamp format
TIMEZONE = pytz.timezone('Asia/Kolkata')  # Indian Standard Time

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
            exit_time TEXT,
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
                option_chain_confirmation, pnl, targets_hit, stoploss_count, failure_reason, exit_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            "",
            None
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
            failure_reason TEXT,
            exit_time TEXT
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
            option_chain_confirmation, pnl, targets_hit, stoploss_count, failure_reason, exit_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        signal_data.get('failure_reason', ''),
        signal_data.get('exit_time', None)
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

def ensure_strategy_tables_exist(strategy_names):
    """
    Create tables for all strategies upfront, even if they don't produce any signals.
    This ensures consistent database structure across runs.
    """
    print("Creating tables for all strategies...")
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    for strategy_name in strategy_names:
        # Handle strategy name backward compatibility
        original_strategy_name = strategy_name
        
        # Remove 'strategy_' prefix if it exists, for schema module lookup
        if strategy_name.startswith('strategy_'):
            strategy_name = strategy_name[9:]  # Remove 'strategy_' prefix for newer files
        
        try:
            # First try to import the specific schema module
            try:
                schema_module = importlib.import_module(f"src.models.schema.{strategy_name}")
                setup_func = getattr(schema_module, f"setup_{strategy_name}_table")
            except (ImportError, AttributeError):
                try:
                    schema_module = importlib.import_module(f"src.models.schema.strategy_{strategy_name}")
                    setup_func = getattr(schema_module, f"setup_strategy_{strategy_name}_table")
                    # Use strategy_ prefix for table name
                    strategy_name = f"strategy_{strategy_name}"
                except (ImportError, AttributeError):
                    # Fall back to a generic setup
                    from src.models.schema.generic import setup_generic_table
                    setup_func = setup_generic_table
                    # Use the original strategy name for the table
                    strategy_name = original_strategy_name
            
            # Call the setup function to ensure the table exists
            setup_func()
            print(f"  ✅ Table created/verified: {strategy_name}")
            
        except Exception as e:
            # If all else fails, create a basic table
            print(f"  ⚠️ Creating basic table for {original_strategy_name}: {e}")
            try:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {original_strategy_name} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_time TEXT,
                        index_name TEXT,
                        signal TEXT,
                        price REAL,
                        confidence TEXT,
                        outcome TEXT,
                        pnl REAL,
                        stop_loss REAL,
                        target REAL,
                        target2 REAL,
                        target3 REAL,
                        targets_hit INTEGER,
                        stoploss_count INTEGER,
                        failure_reason TEXT,
                        exit_time TEXT
                    )
                """)
                conn.commit()
                print(f"  ✅ Basic table created: {original_strategy_name}")
            except Exception as create_e:
                print(f"  ❌ Failed to create table for {original_strategy_name}: {create_e}")
    
    conn.close()
    print("Table creation completed.")

def log_strategy_sql(strategy_name, signal_data):
    """
    Log strategy signals to the corresponding table in the database.
    Validates schema and removes fields that don't exist in the table.
    """
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
    
    # Ensure signal_time is present or add it
    if 'signal_time' not in signal_data:
        # Only set the current time if signal_time doesn't exist
        signal_data['signal_time'] = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
    else:
        # Convert signal_time to IST if it's a valid datetime string in the expected format
        signal_time = signal_data['signal_time']
        if isinstance(signal_time, str):
            try:
                dt_obj = datetime.strptime(signal_time, "%Y-%m-%d %H:%M:%S")
                # Check if it might not be in IST already (assuming IST trading hours 9:15 AM - 3:30 PM)
                if dt_obj.hour < 9 or (dt_obj.hour == 9 and dt_obj.minute < 15):
                    # Add 5 hours and 30 minutes to convert to IST
                    ist_dt = dt_obj + timedelta(hours=5, minutes=30)
                    signal_data['signal_time'] = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                # If not a valid datetime string, skip conversion and leave as-is
                pass
    # Handle exit_time conversion to IST if present and valid
    if 'exit_time' in signal_data and signal_data['exit_time']:
        exit_time = signal_data['exit_time']
        if isinstance(exit_time, str):
            try:
                dt_obj = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
                if dt_obj.hour < 9 or (dt_obj.hour == 9 and dt_obj.minute < 15):
                    ist_dt = dt_obj + timedelta(hours=5, minutes=30)
                    signal_data['exit_time'] = ist_dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
    
    # Skip logging if this is a NO TRADE signal or None
    signal = signal_data.get('signal')
    if not signal or signal == 'NO TRADE' or signal == 'None' or signal == None:
        return
    
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
    if 'outcome' not in signal_data or signal_data['outcome'] is None or signal_data['outcome'] == '':
        signal_data['outcome'] = 'Pending'
    
    if 'pnl' not in signal_data or not signal_data['pnl']:
        signal_data['pnl'] = 0.0
    
    if 'targets_hit' not in signal_data or not signal_data['targets_hit']:
        signal_data['targets_hit'] = 0
    
    if 'stoploss_count' not in signal_data or not signal_data['stoploss_count']:
        signal_data['stoploss_count'] = 0
    
    if 'failure_reason' not in signal_data or not signal_data['failure_reason']:
        signal_data['failure_reason'] = ""

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
    
    # Ensure index_name is set correctly - don't default to UNKNOWN
    if "index_name" not in signal_data or not signal_data["index_name"] or signal_data["index_name"] == "UNKNOWN":
        # Try to get index name from proper sources:
        # 1. Check if we're backtesting one of the main indices
        if "price" in signal_data:
            price = signal_data["price"]
            # Rough price-based guessing for main indices
            if 15000 <= price <= 25000:
                signal_data["index_name"] = "NIFTY50"
            elif 35000 <= price <= 60000:
                signal_data["index_name"] = "BANKNIFTY"
            else:
                # Default to NIFTY50 if we can't determine
                signal_data["index_name"] = "NIFTY50"
        else:
            # Default to NIFTY50 if we have no other information
            signal_data["index_name"] = "NIFTY50"
    
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
    except Exception as e:
        print(f"❌ Error inserting data into {strategy_name}: {e}")
    finally:
        conn.close()
