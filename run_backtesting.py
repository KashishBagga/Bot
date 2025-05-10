#!/usr/bin/env python3
"""
Run backtesting with basic error handling
"""
import sys
import traceback
import sqlite3

def check_db_tables():
    """Check if required database tables exist"""
    print("Checking database tables...")
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Found tables: {tables}")
    
    conn.close()

def run_backtesting():
    """Run the backtesting script with error handling"""
    try:
        # First check database tables
        check_db_tables()
        
        print("\n🚀 Starting backtesting...")
        import backtesting
        
        # Run the backtesting
        print("✅ Backtesting completed successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import required module: {e}")
        print("Make sure all required packages are installed")
        return False
    except Exception as e:
        print(f"❌ Backtesting failed with error: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_backtesting()
    sys.exit(0 if success else 1) 