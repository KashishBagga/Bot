#!/usr/bin/env python3
"""
Codebase Cleanup Script
Identifies and removes unnecessary files to streamline the system.
"""

import os
import shutil
import logging

logger = logging.getLogger('cleanup_codebase')

# Files to KEEP (essential functionality)
ESSENTIAL_FILES = {
    # Core System Files
    'enhanced_unified_accumulator.py',
    'options_analytics_dashboard.py',
    'migrate_database.py',
    'live_paper_trading.py',
    'simple_backtest.py',
    'enhanced_backtest_system.py',
    'daily_backtest_scheduler.py',
    'automated_fyers_auth.py',
    'performance_tracker.py',
    'trading_analytics_dashboard.py',
    
    # Database Files
    'unified_trading.db',
    
    # Documentation
    'SYSTEM_FEATURES_DOCUMENT.md',
    'ENHANCED_UNIFIED_SYSTEM_SUMMARY.md',
    'README.md',
    'requirements.txt',
    '.gitignore',
    
    # Configuration
    'auto_start_trading.sh',
    '.python-version',
    
    # Source Code
    'src/',
    
    # Data and Logs
    'data/',
    'logs/',
    'historical_data_20yr/',
    
    # Tests
    'tests/',
    
    # Indicators
    'indicators/',
}

# Files to REMOVE (redundant or obsolete)
FILES_TO_REMOVE = [
    # Old accumulators (replaced by enhanced_unified_accumulator.py)
    'options_data_accumulator.py',
    'enhanced_options_accumulator.py',
    'market_status_checker.py',
    
    # Old databases (consolidated into unified_trading.db)
    'enhanced_options.db',
    'backtest_results.db',
    'trading_signals.db',
    'test_unified_trading.db',
    
    # Old documentation (replaced by new docs)
    'SMART_OPTIONS_ACCUMULATOR_GUIDE.md',
    'FINAL_SYSTEM_SUMMARY.md',
    'COMPREHENSIVE_ANALYSIS_AND_FIXES.md',
    'PAPER_TRADING_IMPROVEMENTS.md',
    'LIVE_PAPER_TRADING_GUIDE.md',
    'LIVE_TRADING_ROADMAP.md',
    'OPTIONS_TRADING_GUIDE.md',
    'PAPER_TRADING_README.md',
    
    # Old trading bots (consolidated into live_paper_trading.py)
    'options_trading_bot.py',
    'paper_trading_bot.py',
    'live_options_trading_system.py',
    'live_trading_bot.py',
    
    # Old test files (keep only essential ones)
    'test_broker_connection.py',
    'test_fyers.py',
    'test_live_data_only.py',
    'test_live_paper_trading.py',
    'test_paper_trading_unit.py',
    'test_paper_trading.py',
    'test_options_system.py',
    'test_options_mapper.py',
    'test_options_trading.py',
    'test_option_contract_fixes.py',
    'test_unified_system.py',
    'quick_broker_test.py',
    
    # Old monitoring files
    'paper_trading_monitor.py',
    'view_rejection_summary.py',
    'view_capital_rejections.py',
    'check_backtest_db.py',
    'check_database.py',
    
    # Old backtest files
    'run_today_backtest_simple.py',
    'live_strategy_feedback.py',
    
    # Old logs (keep only recent ones)
    'live_paper_trading.log',
    'fyersApi.log',
    'enhanced_backtest.log',
    'broker_test.log',
    'historical_options_backtest.log',
    'today_simple_backtest.log',
    'today_backtest.log',
    'options_trading.log',
    'paper_trading.log',
    
    # Old HTML reports
    'backtest_report_20250830_005929.html',
    'backtest_report_20250830_010328.html',
    'backtest_report_20250830_010422.html',
    'backtest_report_20250830_010523.html',
    'backtest_report_20250830_010617.html',
    'backtest_report_20250830_010705.html',
    
    # Old utility files
    'refresh_fyers_token.py',
    'database_analyzer.py',
]

# Directories to REMOVE
DIRECTORIES_TO_REMOVE = [
    'archive/',
    '.pytest_cache/',
]

def cleanup_codebase():
    """Clean up the codebase by removing unnecessary files."""
    logger.info("🧹 Starting codebase cleanup...")
    
    removed_files = []
    removed_dirs = []
    errors = []
    
    # Remove files
    for file_path in FILES_TO_REMOVE:
        if os.path.exists(file_path):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    removed_files.append(file_path)
                    logger.info(f"🗑️ Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    removed_dirs.append(file_path)
                    logger.info(f"🗑️ Removed directory: {file_path}")
            except Exception as e:
                errors.append(f"Error removing {file_path}: {e}")
                logger.error(f"❌ Error removing {file_path}: {e}")
    
    # Remove directories
    for dir_path in DIRECTORIES_TO_REMOVE:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                removed_dirs.append(dir_path)
                logger.info(f"🗑️ Removed directory: {dir_path}")
            except Exception as e:
                errors.append(f"Error removing {dir_path}: {e}")
                logger.error(f"❌ Error removing {dir_path}: {e}")
    
    # Summary
    logger.info(f"\n📊 CLEANUP SUMMARY")
    logger.info(f"Files removed: {len(removed_files)}")
    logger.info(f"Directories removed: {len(removed_dirs)}")
    logger.info(f"Errors: {len(errors)}")
    
    if removed_files:
        logger.info(f"\n🗑️ Removed Files:")
        for file in removed_files:
            logger.info(f"  - {file}")
    
    if removed_dirs:
        logger.info(f"\n🗑️ Removed Directories:")
        for dir in removed_dirs:
            logger.info(f"  - {dir}")
    
    if errors:
        logger.info(f"\n❌ Errors:")
        for error in errors:
            logger.info(f"  - {error}")
    
    logger.info(f"\n✅ Codebase cleanup completed!")
    return len(removed_files) + len(removed_dirs)

def list_remaining_files():
    """List all remaining files after cleanup."""
    logger.info(f"\n📋 REMAINING FILES:")
    
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in dirs:
            dirs.remove('.git')
        
        for file in files:
            file_path = os.path.join(root, file)
            # Remove leading ./
            if file_path.startswith('./'):
                file_path = file_path[2:]
            logger.info(f"  - {file_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Ask for confirmation
    print("🧹 This will remove unnecessary files to streamline the codebase.")
    print("The following files will be REMOVED:")
    for file in FILES_TO_REMOVE:
        print(f"  - {file}")
    for dir in DIRECTORIES_TO_REMOVE:
        print(f"  - {dir}")
    
    response = input("\nDo you want to proceed? (y/N): ")
    if response.lower() == 'y':
        removed_count = cleanup_codebase()
        list_remaining_files()
        print(f"\n🎉 Cleanup completed! Removed {removed_count} items.")
    else:
        print("❌ Cleanup cancelled.") 