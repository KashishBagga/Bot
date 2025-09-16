#!/usr/bin/env python3
"""
Fix Import Issues - Create missing classes and fix imports
"""

def fix_import_issues():
    """Fix import issues in the modified files"""
    
    # Fix ai_trade_review.py - add missing AsyncTradeReviewProcessor
    with open('src/advanced_systems/ai_trade_review.py', 'r') as f:
        content = f.read()
    
    # Add the missing class
    async_processor_class = '''
class AsyncTradeReviewProcessor:
    """Async processor for heavy AI trade review tasks"""
    
    def __init__(self):
        self.tz = ZoneInfo('Asia/Kolkata')
    
    async def generate_report_async(self, trade_data):
        """Generate trade review report asynchronously"""
        try:
            # Mock async processing
            return {"status": "completed", "timestamp": now().isoformat()}
        except Exception as e:
            logger.error(f"❌ Async report generation failed: {e}")
            return {}
    
    def close(self):
        """Close processor"""
        pass

'''
    
    # Insert before the main class
    content = content.replace(
        'class AITradeReview:',
        async_processor_class + '\nclass AITradeReview:'
    )
    
    with open('src/advanced_systems/ai_trade_review.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed ai_trade_review.py imports")
    
    # Fix indian_market.py - add missing IndianHolidayManager
    with open('src/markets/indian/indian_market.py', 'r') as f:
        content = f.read()
    
    # Add the missing class
    holiday_manager_class = '''
class IndianHolidayManager:
    """Manage Indian market holidays"""
    
    def __init__(self):
        self.tz = ZoneInfo('Asia/Kolkata')
        self.holidays_cache = set()
    
    def is_holiday(self, date=None):
        """Check if given date is a market holiday"""
        if date is None:
            date = now()
        
        # Simple holiday check - in real implementation, use proper holiday API
        date_str = date.strftime('%Y-%m-%d')
        
        # Mock holidays for testing
        mock_holidays = {
            '2024-01-26', '2024-03-08', '2024-03-29', '2024-04-11',
            '2024-04-17', '2024-05-01', '2024-06-17', '2024-08-15',
            '2024-08-26', '2024-10-02', '2024-10-12', '2024-11-01',
            '2024-11-15', '2024-12-25'
        }
        
        return date_str in mock_holidays

'''
    
    # Insert before the main class
    content = content.replace(
        'class IndianMarket(MarketInterface):',
        holiday_manager_class + '\nclass IndianMarket(MarketInterface):'
    )
    
    with open('src/markets/indian/indian_market.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed indian_market.py imports")
    
    # Fix crypto_market.py - add missing CryptoMarketPerformanceTracker
    with open('src/markets/crypto/crypto_market.py', 'r') as f:
        content = f.read()
    
    # Add the missing class
    performance_tracker_class = '''
class CryptoMarketPerformanceTracker:
    """Track fill rates and latency per symbol"""
    
    def __init__(self):
        self.fill_rates = {}
        self.latencies = {}
        self.api_errors = {}
    
    def record_fill_rate(self, symbol, fill_rate):
        """Record fill rate for symbol"""
        if symbol not in self.fill_rates:
            self.fill_rates[symbol] = []
        self.fill_rates[symbol].append(fill_rate)
    
    def record_latency(self, symbol, latency):
        """Record API latency for symbol"""
        if symbol not in self.latencies:
            self.latencies[symbol] = []
        self.latencies[symbol].append(latency)
    
    def record_api_error(self, symbol, error_type):
        """Record API error for symbol"""
        key = f"{symbol}_{error_type}"
        self.api_errors[key] = self.api_errors.get(key, 0) + 1
    
    def get_performance_stats(self, symbol):
        """Get performance statistics for symbol"""
        fill_rates = self.fill_rates.get(symbol, [])
        latencies = self.latencies.get(symbol, [])
        
        return {
            'avg_fill_rate': sum(fill_rates) / len(fill_rates) if fill_rates else 0,
            'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
            'total_errors': sum(self.api_errors.values()),
            'sample_size': len(fill_rates)
        }

'''
    
    # Insert before the main class
    content = content.replace(
        'class CryptoMarket(MarketInterface):',
        performance_tracker_class + '\nclass CryptoMarket(MarketInterface):'
    )
    
    with open('src/markets/crypto/crypto_market.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed crypto_market.py imports")

if __name__ == "__main__":
    fix_import_issues()
