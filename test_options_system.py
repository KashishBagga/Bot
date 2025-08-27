#!/usr/bin/env python3
"""
Test Options Trading System
Comprehensive validation of options trading components
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from options_trading_bot import OptionsTradingBot
from src.models.option_contract import OptionContract, OptionType, StrikeSelection
from src.data.option_chain_loader import OptionChainLoader
from src.core.option_signal_mapper import OptionSignalMapper

def test_options_trading_system():
    """Test the complete options trading system."""
    print("🚀 Testing Options Trading System")
    print("=" * 50)
    
    # 1. Test Options Trading Bot Initialization
    print("\n1. Testing Options Trading Bot Initialization...")
    try:
        bot = OptionsTradingBot(
            initial_capital=100000,
            max_risk_per_trade=0.02,
            confidence_cutoff=40.0
        )
        print("✅ Options Trading Bot initialized successfully")
        print(f"   Capital: ₹{bot.initial_capital:,.2f}")
        print(f"   Risk per trade: {bot.max_risk_per_trade*100:.1f}%")
        print(f"   Confidence cutoff: {bot.confidence_cutoff}")
        print(f"   Strategies loaded: {len(bot.strategies)}")
    except Exception as e:
        print(f"❌ Bot initialization failed: {e}")
        return False
    
    # 2. Test Data Loading
    print("\n2. Testing Data Loading...")
    try:
        df = bot.get_latest_data('NSE:NIFTY50-INDEX', '5min', 100)
        print(f"✅ Data loaded successfully: {len(df)} candles")
        if len(df) > 0:
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Current price: ₹{df['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # 3. Test Signal Generation
    print("\n3. Testing Signal Generation...")
    try:
        signals = bot.generate_option_signals(df, 'NSE:NIFTY50-INDEX')
        print(f"✅ Signal generation successful: {len(signals)} signals")
        
        if signals:
            print("   Sample signals:")
            for i, signal in enumerate(signals[:3]):
                contract = signal.get('contract')
                if contract:
                    print(f"     Signal {i+1}: {signal.get('signal_type', 'UNKNOWN')}")
                    print(f"       Contract: {contract.symbol}")
                    print(f"       Strike: ₹{contract.strike:,.0f}")
                    print(f"       Entry Price: ₹{signal.get('entry_price', 0):.2f}")
                    print(f"       Quantity: {signal.get('quantity', 0)}")
                    print(f"       Confidence: {signal.get('confidence', 0):.1f}%")
        else:
            print("   ⚠️ No signals generated (this may be normal for current market conditions)")
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False
    
    # 4. Test Option Chain Loader
    print("\n4. Testing Option Chain Loader...")
    try:
        loader = OptionChainLoader()
        current_price = float(df['close'].iloc[-1])
        current_time = df['timestamp'].iloc[-1]
        
        # Test simulation
        chain = loader.simulate_option_chain('NSE:NIFTY50-INDEX', current_price, current_time)
        print(f"✅ Option chain simulation successful: {len(chain.contracts)} contracts")
        
        if chain.contracts:
            sample_contract = chain.contracts[0]
            print(f"   Sample contract: {sample_contract.symbol}")
            print(f"   Strike: ₹{sample_contract.strike:,.0f}")
            print(f"   Bid: ₹{sample_contract.bid:.2f}, Ask: ₹{sample_contract.ask:.2f}")
            print(f"   Delta: {sample_contract.delta:.3f}")
    except Exception as e:
        print(f"❌ Option chain loader failed: {e}")
        return False
    
    # 5. Test Signal Mapper
    print("\n5. Testing Signal Mapper...")
    try:
        mapper = OptionSignalMapper(loader)
        mapper.set_parameters(
            expiry_type="weekly",
            strike_selection=StrikeSelection.ATM,
            delta_target=0.30
        )
        print("✅ Signal mapper initialized successfully")
        
        # Test mapping
        test_signal = {
            'timestamp': current_time,
            'strategy': 'ema_crossover_enhanced',
            'signal': 'BUY CALL',
            'price': current_price,
            'confidence': 75.0,
            'symbol': 'NSE:NIFTY50-INDEX',
            'capital': 100000,
            'max_risk_per_trade': 0.02
        }
        
        option_signals = mapper.map_multiple_signals([test_signal], current_price, current_time, chain)
        print(f"✅ Signal mapping successful: {len(option_signals)} option signals")
        
        if option_signals:
            mapped_signal = option_signals[0]
            print(f"   Mapped to: {mapped_signal.get('contract', {}).symbol}")
            print(f"   Entry price: ₹{mapped_signal.get('entry_price', 0):.2f}")
            print(f"   Quantity: {mapped_signal.get('quantity', 0)}")
    except Exception as e:
        print(f"❌ Signal mapper failed: {e}")
        return False
    
    # 6. Test Position Sizing
    print("\n6. Testing Position Sizing...")
    try:
        if option_signals:
            signal = option_signals[0]
            contract = signal.get('contract')
            entry_price = signal.get('entry_price', 0)
            quantity = signal.get('quantity', 0)
            
            if contract and entry_price > 0 and quantity > 0:
                premium_per_lot = entry_price * contract.lot_size
                total_premium = premium_per_lot * (quantity / contract.lot_size)
                
                print(f"✅ Position sizing successful:")
                print(f"   Premium per lot: ₹{premium_per_lot:,.2f}")
                print(f"   Total premium: ₹{total_premium:,.2f}")
                print(f"   Risk percentage: {(total_premium / 100000) * 100:.2f}%")
            else:
                print("⚠️ Position sizing test skipped (no valid signal)")
        else:
            print("⚠️ Position sizing test skipped (no signals generated)")
    except Exception as e:
        print(f"❌ Position sizing test failed: {e}")
        return False
    
    # 7. Test Risk Management
    print("\n7. Testing Risk Management...")
    try:
        # Test exposure calculation
        exposure = bot._current_total_exposure()
        print(f"✅ Risk management initialized:")
        print(f"   Current exposure: {exposure:.2%}")
        print(f"   Exposure limit: {bot.exposure_limit:.2%}")
        print(f"   Daily loss limit: {bot.max_daily_loss_pct:.2%}")
        
        # Test position opening logic
        if option_signals:
            should_open = bot.should_open_option_position(option_signals[0])
            print(f"   Should open position: {should_open}")
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("✅ Options trading system is ready for paper trading")
    print("\n📋 Next Steps:")
    print("1. Run paper trading: python3 options_trading_bot.py")
    print("2. Monitor performance with analytics dashboard")
    print("3. Fine-tune parameters based on results")
    
    return True

if __name__ == "__main__":
    success = test_options_trading_system()
    sys.exit(0 if success else 1) 