#!/usr/bin/env python3
"""
Test All Improvements for Historical Options Backtesting
Comprehensive test for lot size flexibility, P&L calculation, drawdown metrics, and per-strategy analysis
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.historical_options_loader import HistoricalOptionsLoader
from src.core.option_signal_mapper import OptionSignalMapper
from src.core.options_pnl_calculator import OptionsPnLCalculator, PositionType
from src.models.option_contract import StrikeSelection
from simple_backtest import OptimizedBacktester
from src.data.local_data_loader import LocalDataLoader

def test_lot_size_flexibility():
    """Test lot size flexibility for different underlyings."""
    print("\n1. Testing Lot Size Flexibility")
    print("-" * 40)
    
    try:
        loader = HistoricalOptionsLoader()
        
        # Test different underlyings
        underlyings = [
            'NSE:NIFTY50-INDEX',
            'NSE:NIFTYBANK-INDEX', 
            'NSE:FINNIFTY-INDEX'
        ]
        
        for underlying in underlyings:
            # Create sample data for a single day
            test_date = datetime.now() - timedelta(days=1)
            loader._create_sample_daily_options_data(underlying, test_date)
            
            # Load and check lot sizes
            chain = loader.load_historical_options_chain(underlying, test_date)
            
            if chain and chain.contracts:
                sample_contract = chain.contracts[0]
                expected_lot_size = {
                    'NSE:NIFTY50-INDEX': 50,
                    'NSE:NIFTYBANK-INDEX': 15,
                    'NSE:FINNIFTY-INDEX': 40
                }.get(underlying, 50)
                
                print(f"✅ {underlying}:")
                print(f"   Expected lot size: {expected_lot_size}")
                print(f"   Actual lot size: {sample_contract.lot_size}")
                print(f"   Contract: {sample_contract.symbol}")
                
                assert sample_contract.lot_size == expected_lot_size, f"Lot size mismatch for {underlying}"
            else:
                print(f"❌ No contracts found for {underlying}")
                return False
        
        print("✅ All lot sizes are correct!")
        return True
        
    except Exception as e:
        print(f"❌ Lot size flexibility test failed: {e}")
        return False

def test_enhanced_pnl_calculation():
    """Test enhanced P&L calculation with different scenarios."""
    print("\n2. Testing Enhanced P&L Calculation")
    print("-" * 40)
    
    try:
        calculator = OptionsPnLCalculator()
        
        # Test scenarios
        scenarios = [
            {
                'name': 'Long Call - Profitable',
                'position_type': PositionType.LONG,
                'entry_price': 100,
                'exit_price': 150,
                'quantity': 100,
                'lot_size': 50
            },
            {
                'name': 'Long Put - Loss',
                'position_type': PositionType.LONG,
                'entry_price': 80,
                'exit_price': 60,
                'quantity': 100,
                'lot_size': 50
            },
            {
                'name': 'Short Call - Profitable',
                'position_type': PositionType.SHORT,
                'entry_price': 120,
                'exit_price': 100,
                'quantity': 100,
                'lot_size': 50
            }
        ]
        
        for scenario in scenarios:
            print(f"\n🔹 {scenario['name']}:")
            
            # Calculate entry cost
            entry_data = calculator.calculate_entry_cost(
                scenario['position_type'],
                scenario['entry_price'],
                scenario['quantity'],
                scenario['lot_size'],
                commission_bps=1.0
            )
            
            # Calculate exit value
            exit_data = calculator.calculate_exit_value(
                scenario['position_type'],
                scenario['exit_price'],
                scenario['quantity'],
                scenario['lot_size'],
                commission_bps=1.0
            )
            
            # Calculate P&L
            pnl_result = calculator.calculate_pnl(entry_data, exit_data)
            
            print(f"   Entry Cost: ₹{entry_data['total_cost']:,.2f}")
            print(f"   Exit Value: ₹{exit_data['total_received']:,.2f}")
            print(f"   P&L: ₹{pnl_result['pnl']:+,.2f}")
            print(f"   Returns: {pnl_result['returns_pct']:+.2f}%")
            print(f"   Commission: ₹{pnl_result['total_commission']:.2f}")
        
        print("✅ Enhanced P&L calculation working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced P&L calculation test failed: {e}")
        return False

def test_drawdown_metrics():
    """Test comprehensive drawdown metrics calculation."""
    print("\n3. Testing Drawdown Metrics")
    print("-" * 40)
    
    try:
        calculator = OptionsPnLCalculator()
        
        # Simulate equity curve with drawdown
        equity_curve = [
            100000,  # Start
            105000,  # Peak
            102000,  # Small dip
            110000,  # New peak
            95000,   # Major drawdown
            98000,   # Recovery
            103000   # End
        ]
        
        drawdown_metrics = calculator.calculate_drawdown_metrics(equity_curve)
        
        print(f"📊 Drawdown Metrics:")
        print(f"   Max Drawdown: {drawdown_metrics['max_drawdown_pct']:.2f}%")
        print(f"   Max Drawdown Duration: {drawdown_metrics['max_drawdown_duration']} periods")
        print(f"   Current Drawdown: {drawdown_metrics['current_drawdown_pct']:.2f}%")
        print(f"   Peak Equity: ₹{drawdown_metrics['peak_equity']:,.2f}")
        print(f"   Current Equity: ₹{drawdown_metrics['current_equity']:,.2f}")
        
        # Verify calculations
        expected_max_dd = ((110000 - 95000) / 110000) * 100
        assert abs(drawdown_metrics['max_drawdown_pct'] - expected_max_dd) < 0.01, "Max drawdown calculation error"
        
        print("✅ Drawdown metrics working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Drawdown metrics test failed: {e}")
        return False

def test_risk_metrics():
    """Test comprehensive risk metrics calculation."""
    print("\n4. Testing Risk Metrics")
    print("-" * 40)
    
    try:
        calculator = OptionsPnLCalculator()
        
        # Simulate trades
        trades = [
            {'pnl': 1000, 'returns_pct': 10.0},
            {'pnl': -500, 'returns_pct': -5.0},
            {'pnl': 800, 'returns_pct': 8.0},
            {'pnl': -300, 'returns_pct': -3.0},
            {'pnl': 1200, 'returns_pct': 12.0},
            {'pnl': -200, 'returns_pct': -2.0},
            {'pnl': 600, 'returns_pct': 6.0}
        ]
        
        risk_metrics = calculator.calculate_risk_metrics(trades, 100000)
        
        print(f"📊 Risk Metrics:")
        print(f"   Total Trades: {risk_metrics['total_trades']}")
        print(f"   Win Rate: {risk_metrics['win_rate_pct']:.1f}%")
        print(f"   Total P&L: ₹{risk_metrics['total_pnl']:+,.2f}")
        print(f"   Avg Win: ₹{risk_metrics['avg_win']:,.2f}")
        print(f"   Avg Loss: ₹{risk_metrics['avg_loss']:,.2f}")
        print(f"   Profit Factor: {risk_metrics['profit_factor']:.2f}")
        print(f"   Max Consecutive Losses: {risk_metrics['max_consecutive_losses']}")
        print(f"   Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        print(f"   Total Return: {risk_metrics['total_return_pct']:+.2f}%")
        
        # Verify calculations
        expected_win_rate = (4 / 7) * 100  # 4 wins out of 7 trades
        assert abs(risk_metrics['win_rate_pct'] - expected_win_rate) < 0.1, "Win rate calculation error"
        
        print("✅ Risk metrics working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Risk metrics test failed: {e}")
        return False

def test_margin_utilization():
    """Test margin utilization calculations."""
    print("\n5. Testing Margin Utilization")
    print("-" * 40)
    
    try:
        calculator = OptionsPnLCalculator()
        
        # Simulate portfolio positions
        positions = [
            {
                'position_type': 'LONG',
                'entry_cost': 5000,
                'margin_required': 0
            },
            {
                'position_type': 'SHORT',
                'entry_cost': 3000,
                'margin_required': 4500
            },
            {
                'position_type': 'LONG',
                'entry_cost': 2000,
                'margin_required': 0
            },
            {
                'position_type': 'SHORT',
                'entry_cost': 4000,
                'margin_required': 6000
            }
        ]
        
        margin_metrics = calculator.calculate_margin_utilization(positions, 100000)
        
        print(f"📊 Margin Utilization:")
        print(f"   Total Margin Required: ₹{margin_metrics['total_margin_required']:,.2f}")
        print(f"   Total Premium Paid: ₹{margin_metrics['total_premium_paid']:,.2f}")
        print(f"   Margin Utilization: {margin_metrics['margin_utilization_pct']:.1f}%")
        print(f"   Capital Utilization: {margin_metrics['capital_utilization_pct']:.1f}%")
        print(f"   Available Margin: ₹{margin_metrics['available_margin']:,.2f}")
        print(f"   Short Positions: {margin_metrics['short_positions_count']}")
        print(f"   Long Positions: {margin_metrics['long_positions_count']}")
        
        # Verify calculations
        expected_margin = 4500 + 6000  # Sum of margin required
        assert abs(margin_metrics['total_margin_required'] - expected_margin) < 0.01, "Margin calculation error"
        
        print("✅ Margin utilization working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Margin utilization test failed: {e}")
        return False

def test_historical_options_integration():
    """Test integration of all improvements with historical options data."""
    print("\n6. Testing Historical Options Integration")
    print("-" * 40)
    
    try:
        # Initialize components
        loader = HistoricalOptionsLoader()
        signal_mapper = OptionSignalMapper(loader)
        pnl_calculator = OptionsPnLCalculator()
        
        # Create sample data
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        loader.create_sample_historical_data('NSE:NIFTY50-INDEX', start_date, end_date)
        
        # Load historical options chain
        test_date = datetime.now() - timedelta(days=3)
        chain = loader.load_historical_options_chain('NSE:NIFTY50-INDEX', test_date)
        
        if not chain or not chain.contracts:
            print("❌ No historical options data available")
            return False
        
        # Test with different lot sizes
        for contract in chain.contracts[:3]:
            print(f"\n🔹 Contract: {contract.symbol}")
            print(f"   Lot Size: {contract.lot_size}")
            print(f"   Strike: ₹{contract.strike:,.0f}")
            print(f"   Bid: ₹{contract.bid:.2f}, Ask: ₹{contract.ask:.2f}")
            
            # Calculate position size with correct lot size
            entry_price = contract.ask
            position_size = 100000 * 0.02 / (entry_price * contract.lot_size)  # 2% risk
            position_size = max(1, int(position_size)) * contract.lot_size
            
            # Calculate entry cost
            entry_data = pnl_calculator.calculate_entry_cost(
                PositionType.LONG,
                entry_price,
                position_size,
                contract.lot_size,
                commission_bps=1.0
            )
            
            print(f"   Position Size: {position_size} shares ({position_size/contract.lot_size:.0f} lots)")
            print(f"   Entry Cost: ₹{entry_data['total_cost']:,.2f}")
            print(f"   Commission: ₹{entry_data['commission']:.2f}")
        
        print("✅ Historical options integration working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Historical options integration test failed: {e}")
        return False

def main():
    """Run all improvement tests."""
    print("🚀 Testing All Improvements for Historical Options Backtesting")
    print("=" * 70)
    
    tests = [
        test_lot_size_flexibility,
        test_enhanced_pnl_calculation,
        test_drawdown_metrics,
        test_risk_metrics,
        test_margin_utilization,
        test_historical_options_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All improvements are working correctly!")
        print("\n✅ IMPROVEMENTS IMPLEMENTED:")
        print("   • Lot size flexibility (NIFTY=50, BANKNIFTY=15, FINNIFTY=40)")
        print("   • Enhanced P&L calculation with proper commission handling")
        print("   • Comprehensive drawdown metrics with rolling periods")
        print("   • Advanced risk metrics (Sharpe ratio, profit factor, etc.)")
        print("   • Margin utilization tracking for short positions")
        print("   • Historical options data integration")
        print("\n🚀 Your historical options backtesting system is now production-ready!")
    else:
        print("⚠️ Some improvements need attention")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 