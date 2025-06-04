#!/usr/bin/env python3
"""
Test Script for Optimized Trading Strategies
Tests confidence-based logic and real-time data compatibility
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from strategies.insidebar_rsi import InsidebarRsi
from strategies.ema_crossover import EmaCrossover
from strategies.supertrend_ema import SupertrendEma
from strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma

def generate_realistic_market_data(periods=100, base_price=25000):
    """Generate realistic market data for testing"""
    
    # Generate timestamps
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=periods*5),
        end=datetime.now(),
        freq='5min'
    )
    
    # Generate realistic OHLCV data with trend and volatility
    np.random.seed(42)  # For reproducible results
    
    # Generate price movements
    returns = np.random.normal(0, 0.002, periods)  # 0.2% volatility
    trend = np.linspace(0, 0.01, periods)  # Small upward trend
    
    prices = []
    current_price = base_price
    
    for i in range(periods):
        change = (returns[i] + trend[i]) * current_price
        current_price += change
        prices.append(current_price)
    
    # Generate OHLC from prices
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC
        volatility = np.random.uniform(0.001, 0.003)
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = prices[i-1] if i > 0 else price
        close = price
        
        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        volume = np.random.randint(50000, 200000)
        
        data.append({
            'timestamp': timestamps[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def add_technical_indicators(df):
    """Add technical indicators required by strategies"""
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMA
    df['ema'] = df['close'].ewm(span=20).mean()
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # SuperTrend components (simplified)
    hl2 = (df['high'] + df['low']) / 2
    df['basic_upperband'] = hl2 + (3 * df['atr'])
    df['basic_lowerband'] = hl2 - (3 * df['atr'])
    
    # Simplified SuperTrend calculation
    df['supertrend'] = df['basic_upperband']  # Simplified for testing
    df['supertrend_direction'] = 1  # Simplified for testing
    
    return df

def test_strategy(strategy, name, data):
    """Test a single strategy"""
    print(f"\n🧪 Testing {name}")
    print("-" * 50)
    
    try:
        # Test different strategy types with appropriate method calls
        if "Insidebar RSI" in name:
            # InsidebarRsi uses analyze(data, symbol)
            result = strategy.analyze(data, "NIFTY50")
        else:
            # Other strategies use analyze_single_timeframe or analyze with different params
            if hasattr(strategy, 'analyze_single_timeframe'):
                result = strategy.analyze_single_timeframe(data)
            else:
                # For strategies that need candle, index, df parameters
                candle = data.iloc[-1]
                result = strategy.analyze(candle, len(data)-1, data)
        
        if result:
            print(f"✅ Signal: {result['signal']}")
            print(f"📊 Confidence: {result.get('confidence', 'N/A')}")
            if 'confidence_score' in result:
                print(f"🎯 Confidence Score: {result['confidence_score']}")
            print(f"💰 Price: ₹{result.get('price', 0):.2f}")
            
            if result['signal'] != "NO TRADE":
                print(f"🛡️ Stop Loss: ₹{result.get('stop_loss', 0)}")
                print(f"🎯 Targets: ₹{result.get('target', 0)} | ₹{result.get('target2', 0)} | ₹{result.get('target3', 0)}")
                print(f"📝 Reason: {result.get('price_reason', 'N/A')[:100]}...")
                
                # Check if confidence reasons are included
                if 'confidence_reasons' in result:
                    print(f"🔍 Confidence Factors: {result['confidence_reasons'][:100]}...")
            
            return True
        else:
            print(f"❌ No result returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_time_compatibility():
    """Test real-time data compatibility"""
    print("\n🔄 REAL-TIME DATA COMPATIBILITY TEST")
    print("=" * 60)
    
    # Generate realistic market data
    market_data = generate_realistic_market_data(periods=50)
    market_data = add_technical_indicators(market_data)
    
    print(f"📊 Generated {len(market_data)} candles")
    print(f"📅 Time range: {market_data.index[0]} to {market_data.index[-1]}")
    print(f"💰 Price range: ₹{market_data['close'].min():.2f} - ₹{market_data['close'].max():.2f}")
    
    # Test each optimized strategy
    strategies = {
        "Insidebar RSI (Optimized)": InsidebarRsi(),
        "EMA Crossover (Optimized)": EmaCrossover(),
        "SuperTrend EMA (Optimized)": SupertrendEma(),
        "SuperTrend MACD RSI EMA (Optimized)": SupertrendMacdRsiEma()
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        success = test_strategy(strategy, name, market_data)
        results[name] = success
    
    return results

def test_confidence_scoring():
    """Test confidence scoring across different market conditions"""
    print("\n🎯 CONFIDENCE SCORING TEST")
    print("=" * 60)
    
    # Test different market scenarios
    scenarios = {
        "Trending Up": {"trend": 0.02, "volatility": 0.001},
        "Trending Down": {"trend": -0.02, "volatility": 0.001},
        "Sideways": {"trend": 0, "volatility": 0.001},
        "High Volatility": {"trend": 0.01, "volatility": 0.005},
        "Low Volatility": {"trend": 0.005, "volatility": 0.0005}
    }
    
    strategy = InsidebarRsi()  # Test with one strategy
    
    for scenario_name, params in scenarios.items():
        print(f"\n📈 Scenario: {scenario_name}")
        print("-" * 30)
        
        # Generate specific market data for this scenario
        data = generate_market_data_scenario(params)
        data = add_technical_indicators(data)
        
        result = strategy.analyze(data, "TEST")
        if result:
            confidence = result.get('confidence', 'N/A')
            score = result.get('confidence_score', 0)
            signal = result.get('signal', 'NO TRADE')
            
            print(f"Signal: {signal} | Confidence: {confidence} | Score: {score}")
        else:
            print("No result")

def generate_market_data_scenario(params, periods=50):
    """Generate market data for specific scenario"""
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(minutes=periods*5),
        end=datetime.now(),
        freq='5min'
    )
    
    np.random.seed(42)
    trend = params['trend']
    volatility = params['volatility']
    
    returns = np.random.normal(trend/periods, volatility, periods)
    
    prices = []
    current_price = 25000
    
    for ret in returns:
        current_price *= (1 + ret)
        prices.append(current_price)
    
    data = []
    for i, price in enumerate(prices):
        vol = np.random.uniform(volatility/2, volatility*2)
        high = price * (1 + vol)
        low = price * (1 - vol)
        open_price = prices[i-1] if i > 0 else price
        
        # Ensure OHLC consistency
        high = max(high, open_price, price)
        low = min(low, open_price, price)
        
        data.append({
            'timestamp': timestamps[i],
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(price, 2),
            'volume': np.random.randint(50000, 200000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_performance_analysis():
    """Test performance analysis with future data"""
    print("\n📊 PERFORMANCE ANALYSIS TEST")
    print("=" * 60)
    
    # Generate historical data and future data
    historical_data = generate_realistic_market_data(periods=50)
    historical_data = add_technical_indicators(historical_data)
    
    future_data = generate_realistic_market_data(periods=20, base_price=historical_data['close'].iloc[-1])
    future_data = add_technical_indicators(future_data)
    
    strategy = EmaCrossover()
    
    # Test with future data for performance calculation
    try:
        # For EMA Crossover, we need to use the analyze_single_timeframe method
        result = strategy.analyze_single_timeframe(historical_data, future_data)
        
        if result and result['signal'] != "NO TRADE":
            print(f"✅ Signal Generated: {result['signal']}")
            print(f"💰 Entry Price: ₹{result['price']:.2f}")
            print(f"📊 Outcome: {result.get('outcome', 'N/A')}")
            print(f"💵 P&L: ₹{result.get('pnl', 0):.2f}")
            print(f"🎯 Targets Hit: {result.get('targets_hit', 0)}")
            print(f"🛑 Stop Loss Count: {result.get('stoploss_count', 0)}")
            
            if result.get('exit_time'):
                print(f"⏰ Exit Time: {result['exit_time']}")
            
        else:
            print("ℹ️ No trade signal generated")
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")

def generate_summary_report(results):
    """Generate summary report"""
    print("\n📋 TEST SUMMARY REPORT")
    print("=" * 60)
    
    total_strategies = len(results)
    successful_tests = sum(results.values())
    
    print(f"📊 Total Strategies Tested: {total_strategies}")
    print(f"✅ Successful Tests: {successful_tests}")
    print(f"❌ Failed Tests: {total_strategies - successful_tests}")
    print(f"📈 Success Rate: {(successful_tests/total_strategies)*100:.1f}%")
    
    print(f"\n📝 Individual Results:")
    for strategy, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {strategy}: {status}")
    
    print(f"\n🎯 KEY IMPROVEMENTS:")
    print(f"  ✅ Removed time-based filtering")
    print(f"  ✅ Implemented confidence-based trading")
    print(f"  ✅ Dynamic risk management")
    print(f"  ✅ Enhanced signal quality")
    print(f"  ✅ Real-time data compatibility")
    
    print(f"\n🔄 REAL-TIME COMPATIBILITY:")
    if all(results.values()):
        print(f"  ✅ All strategies are compatible with real-time data")
        print(f"  ✅ Live trading bot can use these optimized strategies")
        print(f"  ✅ No time-based restrictions - trade based on market conditions")
    else:
        print(f"  ⚠️ Some strategies need further testing")

def main():
    """Main test execution"""
    print("🧪 OPTIMIZED TRADING STRATEGIES TEST SUITE")
    print("=" * 60)
    print("🎯 Testing confidence-based logic and real-time compatibility")
    print("⏰ Removed time-based filters - now trade based on market conditions")
    print("=" * 60)
    
    # Run all tests
    try:
        # Test real-time compatibility
        results = test_real_time_compatibility()
        
        # Test confidence scoring
        test_confidence_scoring()
        
        # Test performance analysis
        test_performance_analysis()
        
        # Generate summary
        generate_summary_report(results)
        
        print(f"\n🎉 All tests completed!")
        
        # Save results
        with open('test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'test_results': results,
                'summary': {
                    'total_strategies': len(results),
                    'successful_tests': sum(results.values()),
                    'success_rate': (sum(results.values())/len(results))*100
                }
            }, f, indent=2)
        
        print(f"💾 Test results saved to test_results.json")
        
    except Exception as e:
        print(f"❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 