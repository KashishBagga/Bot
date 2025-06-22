#!/usr/bin/env python3
"""
Backtesting Configuration
Centralized configuration to ensure all backtesting uses parquet data only
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

class BacktestConfig:
    """Configuration class for parquet-only backtesting"""
    
    # Data source configuration
    USE_PARQUET_ONLY = True  # Force all backtesting to use parquet data
    ALLOW_API_CALLS = False  # Disable API calls during backtesting
    
    # Data directory
    PARQUET_DATA_DIR = Path("data/parquet")
    
    # Default backtesting parameters
    DEFAULT_DAYS_BACK = 30
    DEFAULT_TIMEFRAME = "15min"
    DEFAULT_SAVE_TO_DB = True
    DEFAULT_PARALLEL_EXECUTION = True
    
    # Available timeframes (in order of preference)
    AVAILABLE_TIMEFRAMES = [
        "1min", "3min", "5min", "15min", "30min", 
        "1hour", "4hour", "1day"
    ]
    
    # Symbol mappings
    SYMBOL_MAPPINGS = {
        "NIFTY50": ["NSE:NIFTY50-INDEX", "NIFTY50"],
        "BANKNIFTY": ["NSE:NIFTYBANK-INDEX", "BANKNIFTY", "NIFTYBANK"],
        "NIFTYFIN": ["NSE:NIFTYFIN-INDEX", "NIFTYFIN"],
        "NIFTYIT": ["NSE:NIFTYIT-INDEX", "NIFTYIT"],
        "NIFTYPHARMA": ["NSE:NIFTYPHARMA-INDEX", "NIFTYPHARMA"],
        "NIFTYMETAL": ["NSE:NIFTYMETAL-INDEX", "NIFTYMETAL"],
        "NIFTYAUTO": ["NSE:NIFTYAUTO-INDEX", "NIFTYAUTO"],
        "NIFTYREALTY": ["NSE:NIFTYREALTY-INDEX", "NIFTYREALTY"],
        "NIFTYFMCG": ["NSE:NIFTYFMCG-INDEX", "NIFTYFMCG"],
        "NIFTYENERGY": ["NSE:NIFTYENERGY-INDEX", "NIFTYENERGY"]
    }
    
    # Backtesting scripts configuration
    BACKTESTING_SCRIPTS = {
        "parquet_only": {
            "script": "all_strategies_parquet.py",
            "description": "Parquet-only backtesting (recommended)",
            "uses_api": False,
            "data_source": "parquet"
        },
        "parquet_fast": {
            "script": "backtesting_parquet.py",
            "description": "Ultra-fast parquet backtesting",
            "uses_api": False,
            "data_source": "parquet"
        },
        "legacy_api": {
            "script": "all_strategies.py",
            "description": "Legacy API-based backtesting (deprecated)",
            "uses_api": True,
            "data_source": "api",
            "deprecated": True
        },
        "quick_test": {
            "script": "quick_backtest.py",
            "description": "Quick backtesting (may use API)",
            "uses_api": True,
            "data_source": "mixed"
        }
    }
    
    @classmethod
    def get_recommended_script(cls) -> str:
        """Get the recommended backtesting script"""
        return "all_strategies_parquet.py"
    
    @classmethod
    def get_parquet_only_scripts(cls) -> List[str]:
        """Get list of parquet-only scripts"""
        return [
            config["script"] for config in cls.BACKTESTING_SCRIPTS.values()
            if not config.get("uses_api", False)
        ]
    
    @classmethod
    def validate_data_availability(cls) -> Dict[str, bool]:
        """Check if parquet data is available"""
        results = {}
        
        if not cls.PARQUET_DATA_DIR.exists():
            return {"error": "Parquet data directory not found"}
        
        for symbol_name, symbol_variants in cls.SYMBOL_MAPPINGS.items():
            symbol_available = False
            
            for variant in symbol_variants:
                symbol_dir = cls.PARQUET_DATA_DIR / variant.replace(":", "_")
                if symbol_dir.exists():
                    # Check if any timeframe files exist
                    parquet_files = list(symbol_dir.glob("*.parquet"))
                    if parquet_files:
                        symbol_available = True
                        break
            
            results[symbol_name] = symbol_available
        
        return results
    
    @classmethod
    def get_available_symbols(cls) -> List[str]:
        """Get list of available symbols from parquet data"""
        available_symbols = []
        
        if not cls.PARQUET_DATA_DIR.exists():
            return available_symbols
        
        for symbol_dir in cls.PARQUET_DATA_DIR.iterdir():
            if symbol_dir.is_dir():
                # Check if any parquet files exist
                parquet_files = list(symbol_dir.glob("*.parquet"))
                if parquet_files:
                    # Convert directory name back to symbol
                    symbol_name = symbol_dir.name.replace("_", ":")
                    if not symbol_name.startswith("NSE:"):
                        symbol_name = symbol_dir.name
                    available_symbols.append(symbol_name)
        
        return sorted(available_symbols)
    
    @classmethod
    def get_available_timeframes(cls, symbol: str) -> List[str]:
        """Get available timeframes for a symbol"""
        timeframes = []
        
        # Find symbol directory
        symbol_dir = cls.PARQUET_DATA_DIR / symbol.replace(":", "_")
        if not symbol_dir.exists():
            return timeframes
        
        # Get all parquet files
        for parquet_file in symbol_dir.glob("*.parquet"):
            timeframe = parquet_file.stem
            if timeframe in cls.AVAILABLE_TIMEFRAMES:
                timeframes.append(timeframe)
        
        # Sort by preference
        timeframes.sort(key=lambda x: cls.AVAILABLE_TIMEFRAMES.index(x))
        return timeframes
    
    @classmethod
    def check_setup_status(cls) -> Dict[str, any]:
        """Check the overall setup status"""
        status = {
            "parquet_data_available": cls.PARQUET_DATA_DIR.exists(),
            "total_symbols": 0,
            "total_timeframes": 0,
            "total_files": 0,
            "total_size_mb": 0.0,
            "symbols": {},
            "recommendations": []
        }
        
        if not status["parquet_data_available"]:
            status["recommendations"].append(
                "Run 'python3 setup_20_year_parquet_data.py' to setup historical data"
            )
            return status
        
        # Analyze available data
        available_symbols = cls.get_available_symbols()
        status["total_symbols"] = len(available_symbols)
        
        total_size = 0
        total_files = 0
        
        for symbol in available_symbols:
            symbol_dir = cls.PARQUET_DATA_DIR / symbol.replace(":", "_")
            timeframes = cls.get_available_timeframes(symbol)
            
            symbol_size = 0
            symbol_files = 0
            
            for parquet_file in symbol_dir.glob("*.parquet"):
                file_size = parquet_file.stat().st_size
                symbol_size += file_size
                total_size += file_size
                symbol_files += 1
                total_files += 1
            
            status["symbols"][symbol] = {
                "timeframes": timeframes,
                "files": symbol_files,
                "size_mb": symbol_size / (1024 * 1024)
            }
            
            status["total_timeframes"] += len(timeframes)
        
        status["total_files"] = total_files
        status["total_size_mb"] = total_size / (1024 * 1024)
        
        # Generate recommendations
        if status["total_symbols"] == 0:
            status["recommendations"].append(
                "No parquet data found. Run 'python3 setup_20_year_parquet_data.py'"
            )
        elif status["total_symbols"] < 2:
            status["recommendations"].append(
                "Limited symbols available. Consider fetching more data"
            )
        
        if status["total_size_mb"] < 100:
            status["recommendations"].append(
                "Data size is small. Consider fetching longer historical periods"
            )
        
        return status

def enforce_parquet_only():
    """Decorator to enforce parquet-only backtesting"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not BacktestConfig.USE_PARQUET_ONLY:
                print("⚠️ Warning: Parquet-only mode is disabled")
            
            # Check if parquet data is available
            status = BacktestConfig.check_setup_status()
            if not status["parquet_data_available"]:
                print("❌ Parquet data not available!")
                for recommendation in status["recommendations"]:
                    print(f"💡 {recommendation}")
                return False
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def print_backtest_config():
    """Print current backtesting configuration"""
    print("🔧 BACKTESTING CONFIGURATION")
    print("=" * 50)
    print(f"📊 Parquet-only mode: {'✅ ENABLED' if BacktestConfig.USE_PARQUET_ONLY else '❌ DISABLED'}")
    print(f"🚫 API calls blocked: {'✅ YES' if not BacktestConfig.ALLOW_API_CALLS else '❌ NO'}")
    print(f"📁 Data directory: {BacktestConfig.PARQUET_DATA_DIR}")
    print(f"⏰ Default timeframe: {BacktestConfig.DEFAULT_TIMEFRAME}")
    print(f"📅 Default days back: {BacktestConfig.DEFAULT_DAYS_BACK}")
    print()
    
    # Show available scripts
    print("📜 AVAILABLE BACKTESTING SCRIPTS:")
    for name, config in BacktestConfig.BACKTESTING_SCRIPTS.items():
        status = "✅ RECOMMENDED" if not config.get("uses_api", False) else "⚠️ USES API"
        if config.get("deprecated", False):
            status = "❌ DEPRECATED"
        
        print(f"  {config['script']}: {config['description']} ({status})")
    print()
    
    # Show data status
    status = BacktestConfig.check_setup_status()
    print("📊 DATA STATUS:")
    print(f"  Symbols: {status['total_symbols']}")
    print(f"  Files: {status['total_files']}")
    print(f"  Size: {status['total_size_mb']:.1f} MB")
    
    if status["recommendations"]:
        print("\n💡 RECOMMENDATIONS:")
        for rec in status["recommendations"]:
            print(f"  • {rec}")
    
    print("=" * 50)

def main():
    """Main function to display configuration"""
    print_backtest_config()
    
    # Show available symbols
    symbols = BacktestConfig.get_available_symbols()
    if symbols:
        print(f"\n📈 AVAILABLE SYMBOLS ({len(symbols)}):")
        for symbol in symbols:
            timeframes = BacktestConfig.get_available_timeframes(symbol)
            print(f"  {symbol}: {len(timeframes)} timeframes ({', '.join(timeframes)})")
    
    print(f"\n🚀 RECOMMENDED COMMAND:")
    print(f"python3 {BacktestConfig.get_recommended_script()} --days 30 --timeframe 15min")

if __name__ == "__main__":
    main() 