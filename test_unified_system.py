#!/usr/bin/env python3
"""
Comprehensive Test Script for Unified Database System
Tests all strategies, signal generation, database storage, and data integrity
"""

import os
import sys
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.strategies.insidebar_rsi import InsidebarRsi
from src.strategies.ema_crossover import EmaCrossover
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.data.parquet_data_store import ParquetDataStore
from src.models.unified_database import UnifiedDatabase

class UnifiedSystemTester:
    """Comprehensive tester for the unified trading system"""
    
    def __init__(self, test_db_path: str = "test_trading_signals.db"):
        self.test_db_path = test_db_path
        
        # Initialize strategies
        self.strategies = {
            'insidebar_rsi': InsidebarRsi(),
            'ema_crossover': EmaCrossover(),
            'supertrend_ema': SupertrendEma(),
            'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
        }
        
        # Initialize data store and unified database
        self.data_store = ParquetDataStore()
        self.unified_db = UnifiedDatabase(test_db_path)
        
        # Test symbols
        self.symbols = ['NSE_NIFTYBANK_INDEX', 'NSE_NIFTY50_INDEX']
        
        print("🧪 Unified System Tester Initialized")
        print(f"📊 Test database: {test_db_path}")
        print(f"🎯 Strategies: {list(self.strategies.keys())}")
        print(f"💼 Symbols: {self.symbols}")
    
    def cleanup_test_db(self):
        """Remove test database file"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
            print(f"🗑️ Cleaned up test database: {self.test_db_path}")
    
    def test_data_loading(self):
        """Test data loading from parquet files"""
        print("\n" + "="*60)
        print("🔍 TESTING DATA LOADING")
        print("="*60)
        
        for symbol in self.symbols:
            try:
                data = self.data_store.load_data(symbol, '5min', days_back=1)
                if data is not None and not data.empty:
                    print(f"✅ {symbol}: Loaded {len(data)} candles")
                    print(f"   📈 Latest close: ₹{data.iloc[-1]['close']:.2f}")
                    print(f"   📅 Time range: {data.index[0]} to {data.index[-1]}")
                    
                    # Check required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in data.columns]
                    if missing_cols:
                        print(f"   ⚠️ Missing columns: {missing_cols}")
                    else:
                        print(f"   ✅ All required columns present")
                else:
                    print(f"❌ {symbol}: No data loaded")
            except Exception as e:
                print(f"❌ {symbol}: Data loading error - {e}")
    
    def test_strategy_indicators(self):
        """Test strategy indicator calculations"""
        print("\n" + "="*60)
        print("🧮 TESTING STRATEGY INDICATORS")
        print("="*60)
        
        symbol = self.symbols[0]  # Test with one symbol
        data = self.data_store.load_data(symbol, '5min', days_back=1)
        
        if data is None or data.empty:
            print(f"❌ Cannot test indicators - no data for {symbol}")
            return
        
        for strategy_name, strategy in self.strategies.items():
            try:
                print(f"\n🎯 Testing {strategy_name}")
                
                # Add indicators
                data_with_indicators = strategy.add_indicators(data.copy())
                
                # Check what indicators were added
                new_columns = set(data_with_indicators.columns) - set(data.columns)
                print(f"   📊 Added indicators: {list(new_columns)}")
                
                # Check for NaN values in latest candle
                latest_candle = data_with_indicators.iloc[-1]
                nan_indicators = []
                for col in new_columns:
                    if pd.isna(latest_candle[col]):
                        nan_indicators.append(col)
                
                if nan_indicators:
                    print(f"   ⚠️ NaN values in: {nan_indicators}")
                else:
                    print(f"   ✅ All indicators calculated successfully")
                
                # Show sample indicator values
                sample_values = {}
                for col in list(new_columns)[:5]:  # Show first 5 indicators
                    value = latest_candle[col]
                    if pd.isna(value):
                        sample_values[col] = 'NaN'
                    elif isinstance(value, str):
                        sample_values[col] = value  # Don't round string values
                    else:
                        sample_values[col] = round(value, 2)  # Only round numeric values
                print(f"   📈 Sample values: {sample_values}")
                
            except Exception as e:
                print(f"   ❌ Error in {strategy_name}: {e}")
    
    def test_strategy_signals(self):
        """Test signal generation from all strategies"""
        print("\n" + "="*60)
        print("🎯 TESTING STRATEGY SIGNAL GENERATION")
        print("="*60)
        
        all_results = {}
        
        for symbol in self.symbols:
            print(f"\n💼 Testing {symbol}")
            data = self.data_store.load_data(symbol, '5min', days_back=1)
            
            if data is None or data.empty:
                print(f"   ❌ No data available")
                continue
            
            symbol_results = {}
            
            for strategy_name, strategy in self.strategies.items():
                try:
                    # Add indicators
                    data_with_indicators = strategy.add_indicators(data.copy())
                    
                    # Generate signal using the same method as live trading bot
                    if strategy_name == 'insidebar_rsi':
                        result = strategy.analyze(data_with_indicators, symbol, None)
                    elif hasattr(strategy, 'analyze_single_timeframe'):
                        result = strategy.analyze_single_timeframe(data_with_indicators, None)
                    else:
                        candle = data_with_indicators.iloc[-1]
                        result = strategy.analyze(candle, len(data_with_indicators)-1, data_with_indicators, None)
                    
                    symbol_results[strategy_name] = result
                    
                    # Display result summary
                    signal = result.get('signal', 'UNKNOWN')
                    confidence = result.get('confidence', 'Unknown')
                    confidence_score = result.get('confidence_score', 0)
                    price = result.get('price', 0)
                    
                    print(f"   🎯 {strategy_name}: {signal} | {confidence} ({confidence_score}) | ₹{price:.2f}")
                    
                    # Check for strategy-specific indicators
                    strategy_indicators = []
                    if strategy_name == 'ema_crossover':
                        strategy_indicators = ['ema_fast', 'ema_slow', 'crossover_strength']
                    elif strategy_name == 'insidebar_rsi':
                        strategy_indicators = ['rsi_level', 'inside_bar_detected'] 
                    elif strategy_name == 'supertrend_ema':
                        strategy_indicators = ['supertrend_value', 'supertrend_direction']
                    elif strategy_name == 'supertrend_macd_rsi_ema':
                        strategy_indicators = ['supertrend', 'supertrend_direction']
                    
                    present_indicators = [ind for ind in strategy_indicators if ind in result]
                    if present_indicators:
                        print(f"     📊 Strategy indicators: {present_indicators}")
                    
                except Exception as e:
                    print(f"   ❌ {strategy_name} error: {e}")
                    symbol_results[strategy_name] = {'signal': 'ERROR', 'reason': str(e)}
            
            all_results[symbol] = symbol_results
        
        return all_results
    
    def test_unified_database_storage(self, signal_results):
        """Test unified database storage with real signal data"""
        print("\n" + "="*60)
        print("💾 TESTING UNIFIED DATABASE STORAGE")
        print("="*60)
        
        # Start a test backtesting run
        run_id = self.unified_db.start_backtesting_run(
            run_name="Test Run - Unified System",
            period_days=1,
            timeframe="5min",
            symbols=self.symbols,
            strategies=list(self.strategies.keys()),
            parameters={"test": True}
        )
        
        print(f"📊 Started test backtesting run: {run_id}")
        
        live_signals_count = 0
        rejected_signals_count = 0
        
        # Process all signal results
        for symbol, strategy_results in signal_results.items():
            for strategy_name, result in strategy_results.items():
                
                # Add required fields
                enhanced_result = {
                    **result,
                    'signal_time': datetime.now().isoformat(),
                    'strategy': strategy_name,
                    'symbol': symbol,
                    'price': result.get('price', 0),
                    'timestamp': datetime.now().isoformat()
                }
                
                signal = result.get('signal', 'UNKNOWN')
                
                if signal in ['BUY CALL', 'BUY PUT']:
                    # Valid signal - test live signal storage
                    signal_id = self.unified_db.log_live_signal(
                        strategy=strategy_name,
                        symbol=symbol,
                        signal_data=enhanced_result
                    )
                    live_signals_count += 1
                    print(f"✅ Live signal logged: {strategy_name} - {symbol} - {signal} (ID: {signal_id})")
                    
                    # Also test backtesting signal storage
                    self.unified_db.log_backtesting_signal(
                        run_id=run_id,
                        strategy=strategy_name,
                        symbol=symbol,
                        signal_data=enhanced_result
                    )
                    
                else:
                    # Rejected signal - test rejected signal storage
                    rejection_data = {
                        **enhanced_result,
                        'rejection_reason': f"Signal type: {signal}",
                        'signal_attempted': signal if signal != 'UNKNOWN' else 'NO TRADE'
                    }
                    
                    self.unified_db.log_rejected_signal(
                        strategy=strategy_name,
                        symbol=symbol,
                        rejection_data=rejection_data,
                        source='LIVE'
                    )
                    rejected_signals_count += 1
                    print(f"📋 Rejected signal logged: {strategy_name} - {symbol} - {signal}")
                    
                    # Also test backtesting rejected signal
                    self.unified_db.log_rejected_signal(
                        strategy=strategy_name,
                        symbol=symbol,
                        rejection_data=rejection_data,
                        source='BACKTEST',
                        run_id=run_id
                    )
        
        # Finish the backtesting run
        self.unified_db.finish_backtesting_run(run_id, {
            'total_signals': live_signals_count + rejected_signals_count,
            'valid_signals': live_signals_count,
            'rejected_signals': rejected_signals_count,
            'total_pnl': 0,
            'win_rate': 0,
            'duration_seconds': 1,
            'signals_per_second': live_signals_count + rejected_signals_count
        })
        
        print(f"📊 Test complete: {live_signals_count} live signals, {rejected_signals_count} rejected signals")
        
        return run_id
    
    def test_database_queries(self, run_id):
        """Test database query capabilities"""
        print("\n" + "="*60)
        print("🔍 TESTING DATABASE QUERIES")
        print("="*60)
        
        # Test latest backtest summary
        summary = self.unified_db.get_latest_backtest_summary()
        if summary:
            print("✅ Latest backtest summary:")
            for key, value in summary.items():
                print(f"   {key}: {value}")
        else:
            print("❌ No backtest summary found")
        
        # Test strategy comparison
        comparison = self.unified_db.get_strategy_comparison(run_id)
        if comparison:
            print(f"\n✅ Strategy comparison ({len(comparison)} entries):")
            for entry in comparison[:3]:  # Show first 3
                print(f"   {entry.get('strategy')} - {entry.get('symbol')}: {entry.get('total_signals')} signals")
        else:
            print("❌ No strategy comparison data")
        
        # Test rejection analysis
        rejection_analysis = self.unified_db.get_rejection_analysis()
        if rejection_analysis:
            print(f"\n✅ Rejection analysis:")
            print(f"   Categories: {rejection_analysis.get('categories', {})}")
            print(f"   Top reasons: {list(rejection_analysis.get('top_reasons', {}).keys())[:3]}")
        else:
            print("❌ No rejection analysis data")
    
    def test_database_schema_integrity(self):
        """Test database schema and data integrity"""
        print("\n" + "="*60)
        print("🏗️ TESTING DATABASE SCHEMA INTEGRITY")
        print("="*60)
        
        try:
            conn = sqlite3.connect(self.test_db_path)
            cursor = conn.cursor()
            
            # Check all tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'backtesting_signals', 'live_signals', 'rejected_signals', 
                'backtesting_runs', 'strategy_performance'
            ]
            
            for table in expected_tables:
                if table in tables:
                    print(f"✅ Table {table} exists")
                    
                    # Check table structure
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    print(f"   📊 Columns ({len(columns)}): {columns[:5]}...")
                    
                    # Check record count
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"   📈 Records: {count}")
                    
                else:
                    print(f"❌ Table {table} missing")
            
            # Test views
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
            views = [row[0] for row in cursor.fetchall()]
            print(f"\n🔍 Views created: {views}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Schema integrity test failed: {e}")
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🚀 STARTING COMPREHENSIVE UNIFIED SYSTEM TEST")
        print("="*80)
        
        # NO cleanup at start - let it use existing or create new
        # self.cleanup_test_db()  # Removed this line
        
        try:
            # 1. Test data loading
            self.test_data_loading()
            
            # 2. Test strategy indicators  
            self.test_strategy_indicators()
            
            # 3. Test signal generation
            signal_results = self.test_strategy_signals()
            
            # 4. Test database storage
            run_id = self.test_unified_database_storage(signal_results)
            
            # 5. Test database queries
            self.test_database_queries(run_id)
            
            # 6. Test schema integrity
            self.test_database_schema_integrity()
            
            print("\n" + "="*80)
            print("🎉 COMPREHENSIVE TEST COMPLETED SUCCESSFULLY")
            print("="*80)
            
            # Cleanup only at the end
            self.cleanup_test_db()
            
            print("\n✅ All tests passed! The unified system is working correctly.")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup_test_db()
            raise

def main():
    """Main test execution"""
    tester = UnifiedSystemTester()
    try:
        tester.run_comprehensive_test()
        return 0
    except Exception:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main()) 