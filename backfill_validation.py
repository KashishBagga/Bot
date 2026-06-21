#!/usr/bin/env python3
"""
Backfill Validation — one-time migration script to validate and quarantine existing DB rows.
"""

import sys
import os

# Path injection
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.postgres_database import PostgresDatabase
from src.core.data_quality import validate_trade_data

def run_backfill():
    db = PostgresDatabase()
    
    # Initialize DB (which runs our new migrations to ensure columns exist)
    print("Initializing database tables and running migrations...")
    db._init_db()
    
    conn = db._get_connection()
    
    # 1. Backfill trade_performance
    print("\nProcessing trade_performance table...")
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM trade_performance")
        colnames = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        print(f"Found {len(rows)} rows to validate.")
        valid_count = 0
        quarantine_count = 0
        
        for row_tuple in rows:
            row_dict = dict(zip(colnames, row_tuple))
            trade_id = row_dict["trade_id"]
            entry_time = row_dict["entry_time"]
            
            is_valid, errors = validate_trade_data(row_dict)
            err_str = "; ".join(errors) if errors else None
            
            cursor.execute(
                """
                UPDATE trade_performance 
                SET valid = %s, validation_errors = %s 
                WHERE trade_id = %s AND entry_time = %s
                """,
                (is_valid, err_str, trade_id, entry_time)
            )
            
            if is_valid:
                valid_count += 1
            else:
                quarantine_count += 1
                print(f"Quarantined trade_performance {trade_id} ({row_dict['symbol']}): {err_str}")
                
        conn.commit()
        print(f"Completed trade_performance. Validated: {valid_count}, Quarantined: {quarantine_count}")
        
    # 2. Backfill counterfactual_results
    print("\nProcessing counterfactual_results table...")
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM counterfactual_results")
        colnames = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        print(f"Found {len(rows)} rows to validate.")
        valid_count = 0
        quarantine_count = 0
        
        for row_tuple in rows:
            row_dict = dict(zip(colnames, row_tuple))
            candidate_id = row_dict["candidate_id"]
            timestamp = row_dict["timestamp"]
            
            is_valid, errors = validate_trade_data(row_dict)
            err_str = "; ".join(errors) if errors else None
            
            cursor.execute(
                """
                UPDATE counterfactual_results 
                SET valid = %s, validation_errors = %s 
                WHERE candidate_id = %s AND timestamp = %s
                """,
                (is_valid, err_str, candidate_id, timestamp)
            )
            
            if is_valid:
                valid_count += 1
            else:
                quarantine_count += 1
                # Only print first few/interesting ones to avoid huge stdout log
                if quarantine_count <= 20:
                    print(f"Quarantined counterfactual_result {candidate_id}: {err_str}")
                    
        conn.commit()
        print(f"Completed counterfactual_results. Validated: {valid_count}, Quarantined: {quarantine_count}")
        
    print("\nBackfill complete!")

if __name__ == "__main__":
    run_backfill()
