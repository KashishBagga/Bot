#!/usr/bin/env python3
"""
Trading System Command Center

This script provides a centralized interface for all trading system commands.
It allows you to:
1. Run all strategies or individual strategies
2. Manage the database (optimize, query, backup)
3. Perform system maintenance
4. View trading statistics and reports

Usage:
    python trading_commands.py --action [action_name] [parameters]

Examples:
    python trading_commands.py --action run_all --days 5 --resolution 15 --symbols RELIANCE,TCS
    python trading_commands.py --action optimize_db
    python trading_commands.py --action stats
    python trading_commands.py --action query --strategy supertrend_ema --limit 10
"""

import argparse
import os
import sys
import subprocess
import sqlite3
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Define color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message):
    """Print a formatted header message"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(80)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.END}\n")

def print_success(message):
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_error(message):
    """Print an error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.END}")

def print_warning(message):
    """Print a warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.END}")

def print_info(message):
    """Print an info message"""
    print(f"{Colors.CYAN}ℹ {message}{Colors.END}")

def run_command(command, silent=False):
    """Run a shell command and return the result"""
    try:
        if not silent:
            print_info(f"Running: {command}")
        
        result = subprocess.run(command, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
        
        if not silent and result.stdout:
            print(result.stdout)
            
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if not silent:
            print_error(f"Command failed with error: {e}")
            if e.stderr:
                print(e.stderr)
        return False, e.stderr

def get_available_strategies():
    """Get a list of all available strategies"""
    strategies = []
    
    # Check src/strategies directory
    if os.path.exists("src/strategies"):
        for file in os.listdir("src/strategies"):
            if file.endswith(".py") and not file.startswith("__"):
                strategy_name = file[:-3]  # Remove .py extension
                strategies.append(strategy_name)
    
    # Check strategies directory
    if os.path.exists("strategies"):
        for file in os.listdir("strategies"):
            if file.endswith(".py") and not file.startswith("__"):
                strategy_name = file[:-3]  # Remove .py extension
                if strategy_name not in strategies:
                    strategies.append(strategy_name)
    
    return strategies

# DATABASE OPERATIONS
def get_strategy_tables():
    """Get a list of all strategy tables in the database"""
    try:
        conn = sqlite3.connect("trading_signals.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Filter out system tables and other non-strategy tables
        system_tables = ['sqlite_stat1', 'sqlite_sequence']
        tables = [table for table in tables if table not in system_tables]
        
        conn.close()
        return tables
    except Exception as e:
        print_error(f"Error getting strategy tables: {e}")
        return []

def optimize_database():
    """Optimize the SQLite database"""
    print_header("Database Optimization")
    
    if not os.path.exists("optimize_db.py"):
        print_error("optimize_db.py script not found!")
        return False
    
    success, output = run_command("python3 optimize_db.py")
    if success:
        print_success("Database optimization completed successfully!")
        return True
    else:
        print_error("Database optimization failed!")
        return False

def backup_database():
    """Create a backup of the database"""
    print_header("Database Backup")
    
    if not os.path.exists("trading_signals.db"):
        print_error("Database file not found!")
        return False
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"backups/trading_signals_{timestamp}.db"
    
    # Create backups directory if it doesn't exist
    if not os.path.exists("backups"):
        os.makedirs("backups")
    
    success, _ = run_command(f"cp trading_signals.db {backup_path}")
    if success:
        print_success(f"Database backed up to {backup_path}")
        return True
    else:
        print_error("Database backup failed!")
        return False

def query_database(strategy, limit=10, order_by="signal_time DESC"):
    """Query the database for a specific strategy"""
    print_header(f"Database Query: {strategy}")
    
    try:
        conn = sqlite3.connect("trading_signals.db")
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{strategy}'")
        if not cursor.fetchone():
            print_error(f"Table '{strategy}' does not exist in the database!")
            conn.close()
            return False
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({strategy})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Build query
        query = f"SELECT * FROM {strategy} ORDER BY {order_by} LIMIT {limit}"
        
        # Execute query
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print_warning(f"No data found in table '{strategy}'")
            return False
        
        # Display results
        print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        print_info(f"Showing {len(df)} of {get_table_row_count(strategy)} records")
        return True
    
    except Exception as e:
        print_error(f"Error querying database: {e}")
        return False

def get_table_row_count(table_name):
    """Get the number of rows in a table"""
    try:
        conn = sqlite3.connect("trading_signals.db")
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except:
        return 0

def show_database_stats():
    """Show statistics for the database"""
    print_header("Database Statistics")
    
    try:
        conn = sqlite3.connect("trading_signals.db")
        cursor = conn.cursor()
        
        # Get all tables but filter out system tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Filter out system tables
        system_tables = ['sqlite_stat1', 'sqlite_sequence']
        tables = [table for table in tables if table not in system_tables]
        
        if not tables:
            print_warning("No tables found in the database!")
            conn.close()
            return False
        
        # Collect statistics
        stats = []
        for table in tables:
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            
            # Get signal distribution if applicable
            signal_distribution = {}
            try:
                cursor.execute(f"SELECT signal, COUNT(*) FROM {table} GROUP BY signal")
                for signal, count in cursor.fetchall():
                    signal_distribution[signal if signal else 'None'] = count
            except:
                signal_distribution = {"N/A": row_count}
            
            # Get timestamp range if applicable
            date_range = "N/A"
            try:
                cursor.execute(f"SELECT MIN(signal_time), MAX(signal_time) FROM {table}")
                min_date, max_date = cursor.fetchone()
                if min_date and max_date:
                    date_range = f"{min_date} to {max_date}"
            except:
                pass
            
            stats.append({
                "Table": table,
                "Records": row_count,
                "Signals": signal_distribution,
                "Date Range": date_range
            })
        
        conn.close()
        
        # Display statistics
        print(f"Total tables: {len(tables)}")
        print("\nTable Statistics:")
        for stat in stats:
            print(f"\n{Colors.BOLD}{stat['Table']}{Colors.END}")
            print(f"  Records: {stat['Records']}")
            print(f"  Date Range: {stat['Date Range']}")
            print("  Signal Distribution:")
            for signal, count in stat['Signals'].items():
                percentage = (count / stat['Records']) * 100 if stat['Records'] > 0 else 0
                print(f"    {signal}: {count} ({percentage:.1f}%)")
        
        return True
    
    except Exception as e:
        print_error(f"Error getting database statistics: {e}")
        return False

# STRATEGY OPERATIONS
def run_all_strategies(days=5, resolution=15, symbols=None, save_to_db=True):
    """Run all trading strategies"""
    print_header("Running All Strategies")
    
    command = f"python3 backtesting.py --days {days} --resolution {resolution}"
    
    if symbols:
        command += f" --symbols {symbols}"
    
    if not save_to_db:
        command += " --no-save"
    
    print_info(f"Running strategies with:")
    print(f"  Days: {days}")
    print(f"  Resolution: {resolution}")
    print(f"  Symbols: {symbols if symbols else 'Default (NIFTY50,BANKNIFTY)'}")
    print(f"  Save to DB: {save_to_db}")
    
    success, _ = run_command(command)
    if success:
        print_success("All strategies executed successfully!")
        return True
    else:
        print_error("Strategy execution failed!")
        return False

def run_single_strategy(strategy, days=5, resolution=15, symbols=None, save_to_db=True):
    """Run a single trading strategy"""
    print_header(f"Running Strategy: {strategy}")
    
    # Check if strategy exists
    strategies = get_available_strategies()
    if strategy not in strategies:
        print_error(f"Strategy '{strategy}' not found!")
        print_info(f"Available strategies: {', '.join(strategies)}")
        return False
    
    # Create a temporary script to run just this strategy
    with open("run_single_strategy.py", "w") as f:
        f.write("""#!/usr/bin/env python3
import sys
import os
from all_strategies import run_all_strategies, get_available_strategies

# Override get_available_strategies to return only the selected strategy
def get_single_strategy():
    return ["%s"]

# Patch the function
import all_strategies
all_strategies.get_available_strategies = get_single_strategy

# Run the strategy
if __name__ == "__main__":
    days = %d
    resolution = "%s"
    save_to_db = %s
    
    # Process symbols
    symbols_dict = None
    if "%s":
        try:
            symbols_list = "%s".split(',')
            symbols_dict = {symbol: symbol for symbol in symbols_list}
        except Exception as e:
            print(f"Error processing symbols: {e}")
    
    success = run_all_strategies(
        days_back=days,
        resolution=resolution,
        save_to_db=save_to_db,
        symbols=symbols_dict
    )
    
    sys.exit(0 if success else 1)
""" % (strategy, days, resolution, str(save_to_db).lower(), 
       symbols if symbols else "", symbols if symbols else ""))
    
    # Make the script executable
    os.chmod("run_single_strategy.py", 0o755)
    
    print_info(f"Running strategy with:")
    print(f"  Strategy: {strategy}")
    print(f"  Days: {days}")
    print(f"  Resolution: {resolution}")
    print(f"  Symbols: {symbols if symbols else 'Default (NIFTY50,BANKNIFTY)'}")
    print(f"  Save to DB: {save_to_db}")
    
    success, output = run_command("python3 run_single_strategy.py")
    
    # Clean up
    if os.path.exists("run_single_strategy.py"):
        os.remove("run_single_strategy.py")
    
    if success:
        print_success(f"Strategy {strategy} executed successfully!")
        return True
    else:
        print_error(f"Strategy {strategy} execution failed!")
        return False

# MAINTENANCE OPERATIONS
def check_system():
    """Check the trading system for potential issues"""
    print_header("System Check")
    
    issues_found = False
    
    # Check Python version
    print_info("Checking Python version...")
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    major, minor, _ = map(int, python_version.split("."))
    if major < 3 or (major == 3 and minor < 6):
        issues_found = True
        print_warning("Python version is below 3.6, which may cause compatibility issues")
    else:
        print_success("Python version is 3.6 or higher")
    
    # Check required files
    print_info("Checking required files...")
    required_files = [
        "backtesting.py", 
        "all_strategies.py", 
        "db.py", 
        "optimize_db.py"
    ]
    for file in required_files:
        if os.path.exists(file):
            print_success(f"Found {file}")
        else:
            issues_found = True
            print_error(f"Missing {file}")
    
    # Check database
    print_info("Checking database...")
    if os.path.exists("trading_signals.db"):
        print_success("Database file exists")
        
        # Check database tables
        try:
            conn = sqlite3.connect("trading_signals.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            print_success(f"Database contains {len(tables)} tables")
        except Exception as e:
            issues_found = True
            print_error(f"Error accessing database: {e}")
    else:
        issues_found = True
        print_error("Database file not found")
    
    # Check strategies
    print_info("Checking strategies...")
    strategies = get_available_strategies()
    if strategies:
        print_success(f"Found {len(strategies)} strategies: {', '.join(strategies)}")
    else:
        issues_found = True
        print_error("No strategies found")
    
    # Check required dependencies
    print_info("Checking dependencies...")
    dependencies = [
        "pandas", "numpy", "matplotlib", "ta", "fyers_apiv3", 
        "backoff", "dotenv", "sqlite3", "tabulate"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print_success(f"Dependency {dep} is installed")
        except ImportError:
            issues_found = True
            print_error(f"Dependency {dep} is missing")
    
    # Summary
    print("\nSystem Check Summary:")
    if issues_found:
        print_warning("Issues were found with the system. See above for details.")
    else:
        print_success("All checks passed. System appears to be in good condition.")
    
    return not issues_found

def generate_signal_report(strategy=None, days=30):
    """Generate a report of trading signals"""
    print_header("Signal Report")
    
    try:
        conn = sqlite3.connect("trading_signals.db")
        
        if strategy:
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{strategy}'")
            if not cursor.fetchone():
                print_error(f"Table '{strategy}' does not exist in the database!")
                conn.close()
                return False
            
            tables = [strategy]
        else:
            # Get all strategy tables
            tables = get_strategy_tables()
        
        if not tables:
            print_warning("No strategy tables found in the database!")
            conn.close()
            return False
        
        # Calculate date cutoff
        cutoff_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Collect signal data
        all_data = []
        for table in tables:
            try:
                # Check if table has required columns first
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'signal' not in columns or 'signal_time' not in columns:
                    print_warning(f"Table {table} doesn't have required signal columns - skipping")
                    continue
                
                # Get signal distribution
                query = f"""
                SELECT 
                    '{table}' as strategy, 
                    signal, 
                    COUNT(*) as count,
                    MIN(signal_time) as first_signal,
                    MAX(signal_time) as last_signal
                FROM {table}
                WHERE signal_time >= '{cutoff_date}'
                GROUP BY signal
                """
                
                df = pd.read_sql_query(query, conn)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print_warning(f"Error getting data for {table}: {e}")
        
        conn.close()
        
        if not all_data:
            print_warning(f"No signal data found in the last {days} days!")
            return False
        
        # Combine all data
        combined_data = pd.concat(all_data)
        
        # Display results
        print(f"Signal distribution for the last {days} days:")
        print(tabulate(combined_data, headers='keys', tablefmt='psql', showindex=False))
        
        # Generate signal distribution chart
        try:
            pivot_data = combined_data.pivot_table(
                index='strategy', columns='signal', values='count', fill_value=0
            )
            
            plt.figure(figsize=(12, 8))
            ax = pivot_data.plot(kind='bar', stacked=True)
            plt.title(f'Signal Distribution by Strategy (Last {days} Days)')
            plt.xlabel('Strategy')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            chart_path = f"signal_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path)
            print_success(f"Signal distribution chart saved to {chart_path}")
            
        except Exception as e:
            print_warning(f"Error generating chart: {e}")
        
        return True
    
    except Exception as e:
        print_error(f"Error generating signal report: {e}")
        return False

def main():
    """Main function to parse arguments and execute commands"""
    parser = argparse.ArgumentParser(description="Trading System Command Center")
    
    parser.add_argument("--action", required=True, help="Action to perform")
    parser.add_argument("--days", type=int, default=5, help="Number of days to backtest")
    parser.add_argument("--resolution", type=str, default="15", help="Candle resolution in minutes")
    parser.add_argument("--symbols", type=str, help="Symbols to test (comma-separated)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to database")
    parser.add_argument("--strategy", type=str, help="Strategy to run or query")
    parser.add_argument("--limit", type=int, default=10, help="Limit for database queries")
    
    args = parser.parse_args()
    
    if args.action == "run_all":
        run_all_strategies(args.days, args.resolution, args.symbols, not args.no_save)
    
    elif args.action == "run_strategy":
        if not args.strategy:
            print_error("Please specify a strategy with --strategy")
            return
        run_single_strategy(args.strategy, args.days, args.resolution, args.symbols, not args.no_save)
    
    elif args.action == "optimize_db":
        optimize_database()
    
    elif args.action == "backup_db":
        backup_database()
    
    elif args.action == "query":
        if not args.strategy:
            print_error("Please specify a strategy with --strategy")
            return
        query_database(args.strategy, args.limit)
    
    elif args.action == "stats":
        show_database_stats()
    
    elif args.action == "check":
        check_system()
    
    elif args.action == "report":
        generate_signal_report(args.strategy, args.days)
    
    elif args.action == "list_strategies":
        strategies = get_available_strategies()
        print_header("Available Strategies")
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy}")
    
    elif args.action == "list_tables":
        tables = get_strategy_tables()
        print_header("Database Tables")
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table}")
    
    else:
        print_error(f"Unknown action: {args.action}")
        print_info("Available actions: run_all, run_strategy, optimize_db, backup_db, query, stats, check, report, list_strategies, list_tables")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"An error occurred: {e}")
        sys.exit(1) 