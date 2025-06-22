#!/usr/bin/env python3
"""
Database Cleanup Script
This script provides multiple options to drop database content
"""

import sqlite3
import os
import sys

def get_all_tables(db_path):
    """Get list of all tables in the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception as e:
        print(f"Error getting tables: {e}")
        return []

def drop_all_tables(db_path):
    """Drop all tables in the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
        
        if tables:
            print("\nDropping all tables...")
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"  ✅ Dropped table: {table}")
            
            conn.commit()
            print(f"\n🗑️  Successfully dropped {len(tables)} tables!")
        else:
            print("No tables found in database.")
        
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error dropping tables: {e}")
        return False

def delete_database_file(db_path):
    """Delete the entire database file"""
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"🗑️  Successfully deleted database file: {db_path}")
            return True
        else:
            print(f"Database file not found: {db_path}")
            return False
    except Exception as e:
        print(f"❌ Error deleting database file: {e}")
        return False

def clear_all_data(db_path):
    """Clear all data from all tables but keep table structure"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
        
        if tables:
            print("\nClearing all data from tables...")
            for table in tables:
                cursor.execute(f"DELETE FROM {table}")
                print(f"  🧹 Cleared data from: {table}")
            
            conn.commit()
            print(f"\n🧹 Successfully cleared data from {len(tables)} tables!")
        else:
            print("No tables found in database.")
        
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        return False

def main():
    db_path = "trading_signals.db"
    
    print("🗃️  Database Cleanup Script")
    print("=" * 50)
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    # Show current database info
    tables = get_all_tables(db_path)
    print(f"Database: {db_path}")
    print(f"Tables found: {len(tables)}")
    if tables:
        for table in tables:
            print(f"  - {table}")
    
    print("\nChoose an option:")
    print("1. Drop all tables (remove table structure)")
    print("2. Clear all data (keep table structure)")
    print("3. Delete entire database file")
    print("4. Show table information only")
    print("5. Exit")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            confirm = input("⚠️  This will DROP ALL TABLES. Are you sure? (yes/no): ").strip().lower()
            if confirm == "yes":
                drop_all_tables(db_path)
            else:
                print("Operation cancelled.")
                
        elif choice == "2":
            confirm = input("⚠️  This will CLEAR ALL DATA. Are you sure? (yes/no): ").strip().lower()
            if confirm == "yes":
                clear_all_data(db_path)
            else:
                print("Operation cancelled.")
                
        elif choice == "3":
            confirm = input("⚠️  This will DELETE THE ENTIRE DATABASE FILE. Are you sure? (yes/no): ").strip().lower()
            if confirm == "yes":
                delete_database_file(db_path)
            else:
                print("Operation cancelled.")
                
        elif choice == "4":
            print("\nDatabase information shown above.")
            
        elif choice == "5":
            print("Exiting...")
            
        else:
            print("Invalid choice. Please run the script again.")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 