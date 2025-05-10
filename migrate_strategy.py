#!/usr/bin/env python
"""
Strategy migration script.
Helps migrate existing strategies to the new structure.
"""
import os
import sys
import argparse
import shutil
import re
from pathlib import Path

def create_strategy_template(strategy_name):
    """Create a new strategy file from template."""
    strategy_class_name = "".join(word.capitalize() for word in strategy_name.split("_"))
    
    template = f"""\"\"\"
{strategy_class_name} strategy.
Trading strategy implementation.
\"\"\"
import pandas as pd
from typing import Dict, Any
from src.core.strategy import Strategy
from src.core.indicators import indicators

class {strategy_class_name}(Strategy):
    \"\"\"Trading strategy implementation.\"\"\"
    
    def __init__(self, params: Dict[str, Any] = None):
        \"\"\"Initialize the strategy.
        
        Args:
            params: Strategy parameters
        \"\"\"
        super().__init__("{strategy_name}", params)
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Add strategy-specific indicators to the data.
        
        Args:
            data: Market data with common indicators
            
        Returns:
            pd.DataFrame: Data with added strategy-specific indicators
        \"\"\"
        # Add your custom indicators here
        return data
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Analyze data and generate trading signals.
        
        Args:
            data: Market data with indicators
            
        Returns:
            Dict[str, Any]: Signal data
        \"\"\"
        # Calculate indicators if they haven't been calculated yet
        if 'some_indicator' not in data.columns:
            data = self.calculate_indicators(data)
        
        # Get the latest candle
        candle = data.iloc[-1]
        
        # Set default values
        signal = "None"
        confidence = "Low"
        trade_type = "Intraday"
        rsi_reason = macd_reason = price_reason = ""
        
        # Implement your strategy logic here
        # ...
        
        # Example:
        # if some_condition:
        #     signal = "BUY CALL"
        #     confidence = "High"
        
        # Return the signal data
        return {{
            "signal": signal,
            "price": candle['close'],
            "rsi": candle['rsi'],
            "macd": candle['macd'],
            "macd_signal": candle['macd_signal'],
            "ema_20": candle['ema'],
            "atr": candle['atr'],
            "confidence": confidence,
            "rsi_reason": rsi_reason,
            "macd_reason": macd_reason,
            "price_reason": price_reason,
            "trade_type": trade_type
        }}
"""

    target_path = os.path.join("src", "strategies", f"{strategy_name}.py")
    
    # Check if the file already exists
    if os.path.exists(target_path):
        print(f"⚠ Strategy file already exists at {target_path}")
        return False
    
    # Write the template to the file
    with open(target_path, "w") as f:
        f.write(template)
    
    print(f"✓ Created strategy template at {target_path}")
    return True

def create_schema_template(strategy_name):
    """Create a new schema file for the strategy."""
    template = f"""\"\"\"
Schema module for {strategy_name} strategy.
Defines the database table structure for the strategy.
\"\"\"

# Fields for the {strategy_name} strategy table
{strategy_name}_fields = [
    "id",
    "signal_time",
    "symbol",
    "signal",
    "price",
    "rsi",
    "macd",
    "macd_signal",
    "ema_20",
    "atr",
    "confidence",
    "trade_type",
    "stop_loss",
    "target",
    "target2",
    "target3",
    "rsi_reason",
    "macd_reason",
    "price_reason",
    "outcome",
    "pnl",
    "targets_hit",
    "stoploss_count",
    "failure_reason"
]

def setup_{strategy_name}_table(db):
    \"\"\"Set up the {strategy_name} strategy table.\"\"\"
    query = \"\"\"
        CREATE TABLE IF NOT EXISTS {strategy_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_time TEXT,
            symbol TEXT,
            signal TEXT,
            price REAL,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            ema_20 REAL,
            atr REAL,
            confidence TEXT,
            trade_type TEXT,
            stop_loss INTEGER,
            target INTEGER,
            target2 INTEGER,
            target3 INTEGER,
            rsi_reason TEXT,
            macd_reason TEXT,
            price_reason TEXT,
            outcome TEXT,
            pnl REAL,
            targets_hit INTEGER,
            stoploss_count INTEGER,
            failure_reason TEXT
        )
    \"\"\"
    db.execute_query(query)
"""

    target_path = os.path.join("src", "models", "schema", f"{strategy_name}.py")
    
    # Check if the file already exists
    if os.path.exists(target_path):
        print(f"⚠ Schema file already exists at {target_path}")
        return False
    
    # Write the template to the file
    with open(target_path, "w") as f:
        f.write(template)
    
    print(f"✓ Created schema template at {target_path}")
    return True

def migrate_strategy(source_path, strategy_name):
    """Migrate an existing strategy to the new structure."""
    # Check if the source file exists
    if not os.path.exists(source_path):
        print(f"⚠ Source file not found: {source_path}")
        return False
    
    # Create the target directory if it doesn't exist
    os.makedirs(os.path.join("src", "strategies"), exist_ok=True)
    os.makedirs(os.path.join("src", "models", "schema"), exist_ok=True)
    
    # Create the strategy and schema templates
    strategy_created = create_strategy_template(strategy_name)
    schema_created = create_schema_template(strategy_name)
    
    if not strategy_created or not schema_created:
        print("⚠ Migration failed. Please check the errors above.")
        return False
    
    # Copy the original file for reference
    backup_path = os.path.join("src", "strategies", f"{strategy_name}_original.py")
    shutil.copy(source_path, backup_path)
    print(f"✓ Copied original strategy to {backup_path} for reference")
    
    print("\n✓ Migration completed successfully!")
    print("\nNext steps:")
    print(f"1. Edit src/strategies/{strategy_name}.py to implement your strategy logic")
    print(f"2. Edit src/models/schema/{strategy_name}.py to customize the database schema if needed")
    print(f"3. Run the test script: python test_strategy.py --strategy {strategy_name}")
    
    return True

def list_legacy_strategies():
    """List the strategies in the legacy strategies directory."""
    if not os.path.exists("strategies"):
        print("⚠ Legacy strategies directory not found")
        return []
    
    strategies = []
    for file in os.listdir("strategies"):
        if file.endswith(".py") and not file.startswith("__"):
            strategy_name = os.path.splitext(file)[0]
            strategies.append((strategy_name, os.path.join("strategies", file)))
    
    return strategies

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Migrate a strategy to the new structure.")
    parser.add_argument("--strategy", help="Name of the strategy to migrate")
    parser.add_argument("--source", help="Path to the source strategy file")
    parser.add_argument("--list", action="store_true", help="List legacy strategies")
    parser.add_argument("--create", help="Create a new strategy with the specified name")
    
    args = parser.parse_args()
    
    if args.list:
        strategies = list_legacy_strategies()
        if strategies:
            print("Legacy strategies:")
            for i, (name, path) in enumerate(strategies, 1):
                print(f"{i}. {name} ({path})")
        else:
            print("No legacy strategies found")
        return
    
    if args.create:
        create_strategy_template(args.create)
        create_schema_template(args.create)
        return
    
    if not args.strategy:
        parser.print_help()
        return
    
    if args.source:
        migrate_strategy(args.source, args.strategy)
    else:
        # Try to find the strategy in the legacy directory
        strategies = list_legacy_strategies()
        for name, path in strategies:
            if name == args.strategy:
                migrate_strategy(path, args.strategy)
                return
        
        print(f"⚠ Source file not specified and strategy '{args.strategy}' not found in legacy directory")
        print("Run with --list to see available legacy strategies")

if __name__ == "__main__":
    main() 