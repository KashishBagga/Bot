#!/usr/bin/env python3
"""
Display a summary of all trading strategies.
This script queries the trading_signals.db database directly
and displays summary information about all strategies.
"""
import sqlite3
import os
import pandas as pd
from tabulate import tabulate
import glob
import re

def get_strategy_tables():
    """Get a list of strategy tables from the database"""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [table[0] for table in cursor.fetchall()]
    
    # Filter to only include strategy tables (those with signal and outcome columns)
    strategy_tables = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        if 'signal' in columns and 'outcome' in columns:
            strategy_tables.append(table)
    
    conn.close()
    return strategy_tables

def get_strategy_performance():
    """Get performance metrics for each strategy"""
    conn = sqlite3.connect("trading_signals.db")
    strategy_tables = get_strategy_tables()
    
    # Results list
    results = []
    
    # Process each strategy table
    for table in strategy_tables:
        try:
            query = f"""
            SELECT 
                '{table}' as strategy,
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'Loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN outcome = 'Pending' THEN 1 ELSE 0 END) as pending,
                ROUND(SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) * 100.0 / 
                      NULLIF(SUM(CASE WHEN outcome IN ('Win', 'Loss') THEN 1 ELSE 0 END), 0), 2) as win_rate,
                ROUND(SUM(pnl), 2) as total_pnl,
                ROUND(AVG(CASE WHEN outcome = 'Win' THEN pnl ELSE NULL END), 2) as avg_win,
                ROUND(AVG(CASE WHEN outcome = 'Loss' THEN pnl ELSE NULL END), 2) as avg_loss,
                ROUND(SUM(CASE WHEN outcome = 'Win' THEN pnl ELSE 0 END) / 
                      ABS(NULLIF(SUM(CASE WHEN outcome = 'Loss' THEN pnl ELSE 0 END), 0)), 2) as profit_factor
            FROM {table}
            WHERE signal != 'NO TRADE'
            """
            df = pd.read_sql_query(query, conn)
            if not df.empty and df.iloc[0]['total_trades'] > 0:
                results.append(df)
        except Exception as e:
            print(f"Error processing table {table}: {e}")
    
    # Combine all results into a single DataFrame
    if results:
        result_df = pd.concat(results, ignore_index=True)
        # Sort by total_pnl
        result_df = result_df.sort_values(by='total_pnl', ascending=False)
        return result_df
    else:
        return pd.DataFrame(columns=['strategy', 'total_trades', 'wins', 'losses', 'pending', 'win_rate', 'total_pnl', 'avg_win', 'avg_loss', 'profit_factor'])

def get_signal_distribution():
    """Get the signal distribution by strategy and index"""
    conn = sqlite3.connect("trading_signals.db")
    strategy_tables = get_strategy_tables()
    
    # Results list
    results = []
    
    # Process each strategy table
    for table in strategy_tables:
        try:
            query = f"""
            SELECT 
                '{table}' as strategy,
                index_name,
                signal,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / 
                      (SELECT COUNT(*) FROM {table} WHERE signal != 'NO TRADE'), 2) as percentage,
                ROUND(SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) * 100.0 / 
                      NULLIF(COUNT(CASE WHEN outcome IN ('Win', 'Loss') THEN 1 ELSE NULL END), 0), 2) as win_rate,
                ROUND(SUM(pnl), 2) as total_pnl
            FROM {table}
            WHERE signal != 'NO TRADE'
            GROUP BY index_name, signal
            """
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                results.append(df)
        except Exception as e:
            print(f"Error processing table {table} for signal distribution: {e}")
    
    # Combine all results into a single DataFrame
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['strategy', 'index_name', 'signal', 'count', 'percentage', 'win_rate', 'total_pnl'])

def get_strategy_criteria():
    """Get the criteria used by each strategy based on actual data"""
    conn = sqlite3.connect("trading_signals.db")
    strategy_tables = get_strategy_tables()
    
    # Results list
    results = []
    
    # Process each strategy table
    for table in strategy_tables:
        try:
            # Check if the table has the required columns
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            
            select_columns = []
            if 'rsi_reason' in columns:
                select_columns.append("rsi_reason")
            else:
                select_columns.append("'' as rsi_reason")
                
            if 'macd_reason' in columns:
                select_columns.append("macd_reason")
            else:
                select_columns.append("'' as macd_reason")
                
            if 'price_reason' in columns:
                select_columns.append("price_reason")
            else:
                select_columns.append("'' as price_reason")
            
            # Construct and execute the query
            query = f"""
            SELECT DISTINCT
                '{table}' as strategy,
                signal,
                {', '.join(select_columns)}
            FROM {table}
            WHERE signal != 'NO TRADE'
            GROUP BY signal
            """
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                results.append(df)
        except Exception as e:
            print(f"Error processing table {table} for criteria: {e}")
    
    # Combine all results into a single DataFrame
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['strategy', 'signal', 'rsi_reason', 'macd_reason', 'price_reason'])

def get_recent_signals(limit=10):
    """Get the most recent signals"""
    conn = sqlite3.connect("trading_signals.db")
    strategy_tables = get_strategy_tables()
    
    # Results list
    results = []
    
    # Process each strategy table
    for table in strategy_tables:
        try:
            query = f"""
            SELECT 
                '{table}' as strategy,
                signal_time,
                index_name,
                signal,
                price,
                confidence,
                outcome,
                pnl
            FROM {table}
            WHERE signal != 'NO TRADE'
            ORDER BY signal_time DESC
            LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
            if not df.empty:
                results.append(df)
        except Exception as e:
            print(f"Error processing table {table} for recent signals: {e}")
    
    # Combine all results into a single DataFrame, keep only the most recent overall
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        # Convert signal_time to datetime if it's not already
        if combined_df.signal_time.dtype == 'object':
            combined_df['signal_time'] = pd.to_datetime(combined_df.signal_time, errors='coerce')
        # Sort and limit
        sorted_df = combined_df.sort_values(by='signal_time', ascending=False)
        return sorted_df.head(limit)
    else:
        return pd.DataFrame(columns=['strategy', 'signal_time', 'index_name', 'signal', 'price', 'confidence', 'outcome', 'pnl'])

def get_strategy_descriptions():
    """Extract strategy descriptions from the strategy files"""
    strategy_files = glob.glob("src/strategies/*.py")
    descriptions = []
    
    for file_path in strategy_files:
        try:
            # Extract strategy name from file path
            strategy_name = os.path.basename(file_path).replace('.py', '')
            
            # Skip the __init__.py file
            if strategy_name == "__init__":
                continue
                
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract docstring
            docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
            docstring = docstring_match.group(1).strip() if docstring_match else "No description available"
            
            # Get a short description from the docstring
            short_desc = docstring.split("\n")[0] if docstring else "No description available"
            
            # Add strategy to results
            descriptions.append({
                'strategy': strategy_name,
                'description': short_desc,
                'indicators': extract_indicators(content),
            })
        except Exception as e:
            print(f"Error extracting description from {file_path}: {e}")
    
    return pd.DataFrame(descriptions)

def extract_indicators(content):
    """Extract indicators used in the strategy"""
    indicators = []
    
    # Check for common indicators
    if 'rsi' in content.lower():
        indicators.append('RSI')
    if 'macd' in content.lower():
        indicators.append('MACD')
    if 'ema' in content.lower() or '20-day' in content.lower():
        indicators.append('EMA')
    if 'bollinger' in content.lower():
        indicators.append('Bollinger Bands')
    if 'supertrend' in content.lower():
        indicators.append('Supertrend')
    if 'donchian' in content.lower():
        indicators.append('Donchian Channel')
    if 'atr' in content.lower():
        indicators.append('ATR')
    if 'inside' in content.lower() or 'insidebar' in content.lower():
        indicators.append('Inside Bar')
    
    return ', '.join(indicators) if indicators else 'Not specified'

def main():
    """Main function to run the program"""
    # Clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("===== Trading Strategy Analysis =====\n")
    
    # Display strategy performance metrics
    print("\n📊 STRATEGY PERFORMANCE SUMMARY")
    print("-------------------------------")
    performance_df = get_strategy_performance()
    if not performance_df.empty:
        print(tabulate(performance_df, headers='keys', tablefmt='psql', showindex=False))
    else:
        print("No strategy performance data found.")
    
    # Display strategy descriptions
    print("\n🔍 STRATEGY DESCRIPTIONS")
    print("-------------------")
    descriptions_df = get_strategy_descriptions()
    if not descriptions_df.empty:
        print(tabulate(descriptions_df, headers='keys', tablefmt='psql', showindex=False))
    else:
        print("No strategy descriptions found.")
    
    # Display signal distribution
    print("\n📈 SIGNAL DISTRIBUTION")
    print("--------------------")
    distribution_df = get_signal_distribution()
    if not distribution_df.empty:
        print(tabulate(distribution_df, headers='keys', tablefmt='psql', showindex=False))
    else:
        print("No signal distribution data found.")
    
    # Display strategy criteria
    print("\n📋 STRATEGY SIGNAL CRITERIA")
    print("-------------------------")
    criteria_df = get_strategy_criteria()
    if not criteria_df.empty:
        print(tabulate(criteria_df, headers='keys', tablefmt='psql', showindex=False))
    else:
        print("No strategy criteria found.")
    
    # Display recent signals
    print("\n🔄 MOST RECENT SIGNALS")
    print("--------------------")
    recent_df = get_recent_signals(10)
    if not recent_df.empty:
        print(tabulate(recent_df, headers='keys', tablefmt='psql', showindex=False))
    else:
        print("No recent signals found.")
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main() 