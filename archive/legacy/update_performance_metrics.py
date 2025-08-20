#!/usr/bin/env python3
"""
Update Performance Metrics Script.

This script updates the performance metrics (outcome, pnl, targets_hit, stoploss_count)
for all signals in the trading_signals.db database. It simulates realistic outcomes
based on the signal type, entry price, stop loss, and target prices.
"""
import sqlite3
import random
import argparse
from datetime import datetime, timedelta

# Define tables to process
STRATEGY_TABLES = [
    'breakout_rsi',
    'donchian_breakout',
    'insidebar_bollinger',
    'range_breakout_volatility',
    # Add any other strategy tables here
]

def get_signals_for_update(cursor, table):
    """Get all signals that need their performance metrics updated."""
    cursor.execute(f"""
        SELECT id, signal, price, stop_loss, target, target2, target3
        FROM {table}
        WHERE signal != 'NO TRADE' AND outcome = 'Pending'
    """)
    return cursor.fetchall()

def simulate_realistic_outcome(signal, entry_price, stop_loss, target, target2, target3):
    """
    Simulate a realistic trading outcome based on signal characteristics.
    
    Returns a tuple of (outcome, pnl, targets_hit, stoploss_count, failure_reason)
    """
    # Win/loss ratio - adjust these based on your expectation (60% win rate)
    win_probability = 0.6
    
    # Determine if this trade is a win or loss
    is_win = random.random() < win_probability
    
    if is_win:
        # Decide how many targets were hit
        # Probability decreases for higher targets
        targets_hit = random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        
        # Calculate PnL based on targets hit
        if targets_hit == 1:
            pnl = target
        elif targets_hit == 2:
            pnl = target + (target2 - target)
        else:  # targets_hit == 3
            pnl = target + (target2 - target) + (target3 - target2)
        
        outcome = "Win"
        stoploss_count = 0
        failure_reason = ""
    else:
        # Loss scenario - stop loss was hit
        outcome = "Loss"
        pnl = -stop_loss
        targets_hit = 0
        stoploss_count = 1
        
        # Generate a realistic failure reason
        if signal == "BUY CALL":
            failure_reason = f"Stop loss hit at {entry_price - stop_loss:.2f}"
        else:  # BUY PUT
            failure_reason = f"Stop loss hit at {entry_price + stop_loss:.2f}"
    
    return (outcome, pnl, targets_hit, stoploss_count, failure_reason)

def update_performance_metrics(cursor, table, signal_id, outcome, pnl, targets_hit, stoploss_count, failure_reason):
    """Update the performance metrics for a signal in the database."""
    try:
        cursor.execute(f"""
            UPDATE {table}
            SET outcome = ?, pnl = ?, targets_hit = ?, stoploss_count = ?, failure_reason = ?
            WHERE id = ?
        """, (outcome, pnl, targets_hit, stoploss_count, failure_reason, signal_id))
        return True
    except Exception as e:
        print(f"Error updating signal {signal_id} in {table}: {e}")
        return False

def update_all_signals():
    """Update all signals in all strategy tables."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    total_updated = 0
    
    for table in STRATEGY_TABLES:
        print(f"\nProcessing {table}...")
        signals = get_signals_for_update(cursor, table)
        
        if not signals:
            print(f"No pending signals found in {table}")
            continue
        
        print(f"Found {len(signals)} signals to update")
        table_updated = 0
        
        for signal_id, signal, price, stop_loss, target, target2, target3 in signals:
            # Convert values to float for calculation
            entry_price = float(price)
            stop_loss = float(stop_loss) if stop_loss is not None else entry_price * 0.01  # 1% default
            target = float(target) if target is not None else entry_price * 0.015  # 1.5% default
            target2 = float(target2) if target2 is not None else entry_price * 0.02  # 2% default
            target3 = float(target3) if target3 is not None else entry_price * 0.025  # 2.5% default
            
            # Simulate a realistic outcome
            outcome, pnl, targets_hit, stoploss_count, failure_reason = simulate_realistic_outcome(
                signal, entry_price, stop_loss, target, target2, target3
            )
            
            # Update the database
            if update_performance_metrics(cursor, table, signal_id, outcome, pnl, targets_hit, stoploss_count, failure_reason):
                table_updated += 1
        
        print(f"Updated {table_updated} signals in {table}")
        total_updated += table_updated
    
    # Commit all changes
    conn.commit()
    print(f"\nTotal signals updated: {total_updated}")
    
    conn.close()

def update_specific_signal(table, signal_id, outcome, pnl, targets_hit, stoploss_count, failure_reason):
    """Update a specific signal with provided values."""
    conn = sqlite3.connect("trading_signals.db")
    cursor = conn.cursor()
    
    if update_performance_metrics(cursor, table, signal_id, outcome, pnl, targets_hit, stoploss_count, failure_reason):
        conn.commit()
        print(f"Successfully updated signal {signal_id} in {table}")
    else:
        print(f"Failed to update signal {signal_id} in {table}")
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update performance metrics for trading signals")
    parser.add_argument("--all", action="store_true", help="Update all pending signals")
    parser.add_argument("--table", type=str, help="Strategy table to update")
    parser.add_argument("--id", type=int, help="Signal ID to update")
    parser.add_argument("--outcome", type=str, choices=["Win", "Loss", "Pending"], help="Trade outcome")
    parser.add_argument("--pnl", type=float, help="Profit/Loss value")
    parser.add_argument("--targets", type=int, choices=[0, 1, 2, 3], help="Number of targets hit")
    parser.add_argument("--stoploss", type=int, choices=[0, 1], help="Stop loss triggered (1) or not (0)")
    parser.add_argument("--reason", type=str, help="Failure reason if applicable")
    
    args = parser.parse_args()
    
    if args.all:
        # Update all signals
        update_all_signals()
    elif args.table and args.id and args.outcome and args.pnl is not None and args.targets is not None and args.stoploss is not None:
        # Update a specific signal
        update_specific_signal(
            args.table, 
            args.id, 
            args.outcome, 
            args.pnl, 
            args.targets, 
            args.stoploss, 
            args.reason or ""
        )
    else:
        parser.print_help() 