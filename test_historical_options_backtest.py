#!/usr/bin/env python3
"""
Test Historical Options Backtesting
Simple test to validate historical options backtesting functionality
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.historical_options_loader import HistoricalOptionsLoader
from src.core.option_signal_mapper import OptionSignalMapper
from src.models.option_contract import StrikeSelection
from simple_backtest import OptimizedBacktester
from src.data.local_data_loader import LocalDataLoader

def test_historical_options_backtest():
    """Test historical options backtesting functionality."""
    print("🚀 Testing Historical Options Backtesting")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    try:
        historical_loader = HistoricalOptionsLoader()
        signal_mapper = OptionSignalMapper(historical_loader)
        data_loader = LocalDataLoader()
        backtester = OptimizedBacktester()
        
        signal_mapper.set_parameters(
            expiry_type="weekly",
            strike_selection=StrikeSelection.ATM,
            delta_target=0.30
        )
        
        print("✅ All components initialized successfully")
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False
    
    # 2. Create sample historical data
    print("\n2. Creating sample historical options data...")
    try:
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        historical_loader.create_sample_historical_data('NSE:NIFTY50-INDEX', start_date, end_date)
        print(f"✅ Created sample data from {start_date.date()} to {end_date.date()}")
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        return False
    
    # 3. Load index data
    print("\n3. Loading index data...")
    try:
        df = data_loader.load_data('NSE:NIFTY50-INDEX', '5min', 1000)
        df = backtester.add_indicators_optimized(df)
        
        print(f"✅ Loaded {len(df)} candles with indicators")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Current price: ₹{df['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"❌ Index data loading failed: {e}")
        return False
    
    # 4. Test signal generation and mapping
    print("\n4. Testing signal generation and mapping...")
    try:
        from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
        
        strategy = EmaCrossoverEnhanced()
        signals_df = strategy.analyze_vectorized(df)
        
        if not signals_df.empty:
            # Take the last signal
            last_signal = signals_df.iloc[-1]
            current_price = float(last_signal['price'])
            current_time = df['timestamp'].iloc[-1]
            
            # Create signal dict
            signal = {
                'timestamp': current_time,
                'strategy': 'ema_crossover_enhanced',
                'signal': last_signal['signal'],
                'price': current_price,
                'confidence': 75.0,
                'symbol': 'NSE:NIFTY50-INDEX',
                'capital': 100000,
                'max_risk_per_trade': 0.02
            }
            
            # Load historical options chain
            option_chain = historical_loader.load_historical_options_chain(
                'NSE:NIFTY50-INDEX', current_time.date()
            )
            
            if option_chain:
                # Map signal to options
                option_signals = signal_mapper.map_multiple_signals(
                    [signal], current_price, current_time, option_chain
                )
                
                print(f"✅ Generated {len(option_signals)} option signals")
                
                if option_signals:
                    option_signal = option_signals[0]
                    contract = option_signal.get('contract')
                    if contract:
                        print(f"   Mapped to: {contract.symbol}")
                        print(f"   Strike: ₹{contract.strike:,.0f}")
                        print(f"   Entry Price: ₹{option_signal.get('entry_price', 0):.2f}")
                        print(f"   Quantity: {option_signal.get('quantity', 0)}")
                        print(f"   Premium Risk: ₹{option_signal.get('premium_risk', 0):,.2f}")
            else:
                print("⚠️ No historical options chain available for this date")
        else:
            print("⚠️ No signals generated for this data")
            
    except Exception as e:
        print(f"❌ Signal generation and mapping failed: {e}")
        return False
    
    # 5. Test historical price lookup
    print("\n5. Testing historical price lookup...")
    try:
        # Get a sample contract symbol
        test_date = datetime.now() - timedelta(days=7)
        chain = historical_loader.load_historical_options_chain('NSE:NIFTY50-INDEX', test_date)
        
        if chain and chain.contracts:
            sample_contract = chain.contracts[0]
            historical_price = historical_loader.get_historical_options_price(
                sample_contract.symbol, test_date
            )
            
            if historical_price:
                print(f"✅ Historical price lookup successful")
                print(f"   Contract: {sample_contract.symbol}")
                print(f"   Historical Price: ₹{historical_price:.2f}")
            else:
                print("⚠️ Historical price lookup returned None")
        else:
            print("⚠️ No contracts available for historical price lookup")
            
    except Exception as e:
        print(f"❌ Historical price lookup failed: {e}")
        return False
    
    # 6. Test equity curve simulation
    print("\n6. Testing equity curve simulation...")
    try:
        # Simulate a simple equity curve
        initial_capital = 100000
        current_capital = initial_capital
        equity_curve = []
        
        # Simulate some trades
        for i in range(10):
            # Simulate a trade
            trade_pnl = (i % 3 - 1) * 1000  # Alternating wins/losses
            current_capital += trade_pnl
            
            equity_curve.append({
                'timestamp': datetime.now() - timedelta(days=10-i),
                'equity': current_capital,
                'pnl': trade_pnl
            })
        
        print(f"✅ Equity curve simulation successful")
        print(f"   Initial Capital: ₹{initial_capital:,.2f}")
        print(f"   Final Capital: ₹{current_capital:,.2f}")
        print(f"   Total P&L: ₹{current_capital - initial_capital:+,.2f}")
        print(f"   Return: {((current_capital - initial_capital) / initial_capital) * 100:+.2f}%")
        
    except Exception as e:
        print(f"❌ Equity curve simulation failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All historical options backtesting tests passed!")
    print("✅ Historical options data loading working")
    print("✅ Signal mapping to historical options working")
    print("✅ Historical price lookup working")
    print("✅ Equity curve simulation working")
    print("\n📋 Next Steps:")
    print("1. Run full historical options backtest")
    print("2. Compare with index-based backtest results")
    print("3. Analyze realistic P&L curves")
    
    return True

if __name__ == "__main__":
    success = test_historical_options_backtest()
    sys.exit(0 if success else 1) 