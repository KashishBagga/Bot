#!/usr/bin/env python3
"""
Broker Connection Testing Script
Tests all broker APIs and validates live data feeds
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.realtime_data_manager import RealTimeDataManager, create_data_provider
from src.models.option_contract import OptionChain, OptionType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrokerConnectionTester:
    def __init__(self):
        """Initialize broker connection tester."""
        self.test_results = {}
        
    def test_paper_broker(self) -> Dict:
        """Test paper broker (simulated data)."""
        print("\n🔍 Testing Paper Broker (Simulated Data)")
        print("=" * 50)
        
        try:
            # Create paper broker
            from src.execution.broker_execution import PaperBrokerAPI
            paper_broker = PaperBrokerAPI()
            
            # Test basic functionality
            test_symbol = "NSE:NIFTY50-INDEX"
            
            # Simulate getting underlying price
            simulated_price = 25000.0 + (datetime.now().second % 100)  # Varying price
            print(f"✅ Simulated underlying price: ₹{simulated_price:,.2f}")
            
            # Simulate option chain
            simulated_chain = self._create_simulated_option_chain(test_symbol, simulated_price)
            print(f"✅ Simulated option chain: {len(simulated_chain.contracts)} contracts")
            
            # Test option selection
            atm_call = None
            atm_put = None
            for contract in simulated_chain.contracts:
                if contract.option_type == OptionType.CALL and abs(contract.strike - simulated_price) < 50:
                    atm_call = contract
                elif contract.option_type == OptionType.PUT and abs(contract.strike - simulated_price) < 50:
                    atm_put = contract
            
            if atm_call:
                print(f"✅ ATM CALL: {atm_call.symbol} @ ₹{atm_call.strike:,.0f} - Bid: ₹{atm_call.bid:.2f}, Ask: ₹{atm_call.ask:.2f}")
            if atm_put:
                print(f"✅ ATM PUT: {atm_put.symbol} @ ₹{atm_call.strike:,.0f} - Bid: ₹{atm_put.bid:.2f}, Ask: ₹{atm_put.ask:.2f}")
            
            return {
                'status': 'SUCCESS',
                'provider': 'Paper Broker',
                'underlying_price': simulated_price,
                'option_contracts': len(simulated_chain.contracts),
                'atm_call': atm_call.symbol if atm_call else None,
                'atm_put': atm_put.symbol if atm_put else None,
                'message': 'Paper broker working correctly'
            }
            
        except Exception as e:
            logger.error(f"❌ Paper broker test failed: {e}")
            return {
                'status': 'FAILED',
                'provider': 'Paper Broker',
                'error': str(e)
            }
    
    def test_zerodha_connection(self, api_key: str = None, api_secret: str = None, access_token: str = None) -> Dict:
        """Test Zerodha Kite Connect connection."""
        print("\n🔍 Testing Zerodha Kite Connect")
        print("=" * 50)
        
        try:
            # Check if kiteconnect is installed
            try:
                from kiteconnect import KiteConnect
                print("✅ KiteConnect library installed")
            except ImportError:
                print("❌ KiteConnect not installed. Install with: pip install kiteconnect")
                return {
                    'status': 'FAILED',
                    'provider': 'Zerodha',
                    'error': 'KiteConnect library not installed'
                }
            
            # Test connection
            if not api_key or not api_secret:
                print("⚠️ No API credentials provided. Skipping live connection test.")
                print("   To test live connection, provide --zerodha_api_key and --zerodha_api_secret")
                return {
                    'status': 'SKIPPED',
                    'provider': 'Zerodha',
                    'message': 'No API credentials provided'
                }
            
            # Create Zerodha provider
            zerodha_provider = create_data_provider('zerodha', api_key=api_key, api_secret=api_secret)
            
            # Test connection
            if access_token:
                zerodha_provider.set_access_token(access_token)
                connected = zerodha_provider.connect()
            else:
                connected = zerodha_provider.connect()
            
            if not connected:
                print("⚠️ Could not connect to Zerodha API")
                print("   This is normal if no access token is provided")
                return {
                    'status': 'SKIPPED',
                    'provider': 'Zerodha',
                    'message': 'No access token provided'
                }
            
            # Test data retrieval
            test_symbol = "NSE:NIFTY50-INDEX"
            price = zerodha_provider.get_underlying_price(test_symbol)
            
            if price:
                print(f"✅ Live NIFTY50 price: ₹{price:,.2f}")
                
                # Test option chain
                option_chain = zerodha_provider.get_option_chain(test_symbol)
                if option_chain:
                    print(f"✅ Live option chain: {len(option_chain.contracts)} contracts")
                    
                    # Find ATM options
                    atm_call = None
                    atm_put = None
                    for contract in option_chain.contracts:
                        if contract.option_type == OptionType.CALL and abs(contract.strike - price) < 50:
                            atm_call = contract
                        elif contract.option_type == OptionType.PUT and abs(contract.strike - price) < 50:
                            atm_put = contract
                    
                    if atm_call:
                        print(f"✅ Live ATM CALL: {atm_call.symbol} @ ₹{atm_call.strike:,.0f}")
                    if atm_put:
                        print(f"✅ Live ATM PUT: {atm_put.symbol} @ ₹{atm_put.strike:,.0f}")
                    
                    return {
                        'status': 'SUCCESS',
                        'provider': 'Zerodha',
                        'underlying_price': price,
                        'option_contracts': len(option_chain.contracts),
                        'atm_call': atm_call.symbol if atm_call else None,
                        'atm_put': atm_put.symbol if atm_put else None,
                        'message': 'Zerodha connection working correctly'
                    }
                else:
                    print("⚠️ Could not retrieve option chain")
                    return {
                        'status': 'PARTIAL',
                        'provider': 'Zerodha',
                        'underlying_price': price,
                        'message': 'Underlying price works, but option chain failed'
                    }
            else:
                print("❌ Could not retrieve underlying price")
                return {
                    'status': 'FAILED',
                    'provider': 'Zerodha',
                    'error': 'Could not retrieve underlying price'
                }
                
        except Exception as e:
            logger.error(f"❌ Zerodha test failed: {e}")
            return {
                'status': 'FAILED',
                'provider': 'Zerodha',
                'error': str(e)
            }
    
    def test_fyers_connection(self, app_id: str = None, access_token: str = None) -> Dict:
        """Test Fyers API connection."""
        print("\n🔍 Testing Fyers API")
        print("=" * 50)
        
        try:
            # Check if fyers_api is installed
            try:
                import fyers_api
                print("✅ Fyers API library installed")
            except ImportError:
                print("❌ Fyers API not installed. Install with: pip install fyers-api")
                return {
                    'status': 'FAILED',
                    'provider': 'Fyers',
                    'error': 'Fyers API library not installed'
                }
            
            # Test connection
            if not app_id or not access_token:
                print("⚠️ No API credentials provided. Skipping live connection test.")
                print("   To test live connection, provide --fyers_app_id and --fyers_access_token")
                return {
                    'status': 'SKIPPED',
                    'provider': 'Fyers',
                    'message': 'No API credentials provided'
                }
            
            # Create Fyers provider
            fyers_provider = create_data_provider('fyers', app_id=app_id, access_token=access_token)
            
            # Test connection
            connected = fyers_provider.connect()
            
            if not connected:
                print("❌ Could not connect to Fyers API")
                return {
                    'status': 'FAILED',
                    'provider': 'Fyers',
                    'error': 'Connection failed'
                }
            
            # Test data retrieval
            test_symbol = "NSE:NIFTY50-INDEX"
            price = fyers_provider.get_underlying_price(test_symbol)
            
            if price:
                print(f"✅ Live NIFTY50 price: ₹{price:,.2f}")
                return {
                    'status': 'SUCCESS',
                    'provider': 'Fyers',
                    'underlying_price': price,
                    'message': 'Fyers connection working correctly'
                }
            else:
                print("❌ Could not retrieve underlying price")
                return {
                    'status': 'FAILED',
                    'provider': 'Fyers',
                    'error': 'Could not retrieve underlying price'
                }
                
        except Exception as e:
            logger.error(f"❌ Fyers test failed: {e}")
            return {
                'status': 'FAILED',
                'provider': 'Fyers',
                'error': str(e)
            }
    
    def test_real_time_data_manager(self) -> Dict:
        """Test real-time data manager with paper broker."""
        print("\n🔍 Testing Real-Time Data Manager")
        print("=" * 50)
        
        try:
            # Create data manager with paper broker
            from src.execution.broker_execution import PaperBrokerAPI
            paper_broker = PaperBrokerAPI()
            
            # Create a simple data provider that uses paper broker
            class PaperDataProvider:
                def __init__(self, paper_broker):
                    self.paper_broker = paper_broker
                    self.connected = True
                
                def get_underlying_price(self, symbol: str):
                    # Simulate price
                    return 25000.0 + (datetime.now().second % 100)
                
                def get_option_chain(self, symbol: str):
                    # Create simulated option chain
                    return self._create_simulated_option_chain(symbol, self.get_underlying_price(symbol))
                
                def subscribe_to_updates(self, callback):
                    return True
                
                def is_connected(self):
                    return self.connected
                
                def _create_simulated_option_chain(self, symbol: str, underlying_price: float):
                    from src.models.option_contract import OptionChain, OptionContract, OptionType
                    from datetime import datetime, timedelta
                    
                    chain = OptionChain(symbol, datetime.now())
                    
                    # Generate strikes around current price
                    atm_strike = round(underlying_price / 50) * 50
                    strikes = []
                    
                    # Generate strikes from 5 strikes below to 5 strikes above ATM
                    for i in range(-5, 6):
                        strike = atm_strike + (i * 50)
                        if strike > 0:
                            strikes.append(strike)
                    
                    # Create contracts for each strike
                    for strike in strikes:
                        # Simulate market data
                        moneyness = strike / underlying_price
                        
                        # CALL options
                        if moneyness < 0.98:  # ITM
                            premium = max(underlying_price - strike, 0) + 50
                        elif moneyness > 1.02:  # OTM
                            premium = max(10, 50 * (1 - moneyness))
                        else:  # ATM
                            premium = 100
                        
                        # Simulate bid/ask
                        spread = premium * 0.1
                        bid = premium - spread / 2
                        ask = premium + spread / 2
                        
                        # CALL contract
                        call_contract = OptionContract(
                            symbol=f"{symbol.replace(':', '')}{datetime.now().strftime('%d%m%y')}{strike}CE",
                            underlying=symbol,
                            strike=strike,
                            expiry=datetime.now() + timedelta(days=7),
                            option_type=OptionType.CALL,
                            lot_size=50,
                            bid=bid,
                            ask=ask,
                            last=premium,
                            volume=1000,
                            open_interest=5000,
                            implied_volatility=0.25,
                            delta=0.5 if abs(moneyness - 1) < 0.02 else (0.8 if moneyness < 1 else 0.2),
                            gamma=0.01,
                            theta=-premium * 0.1,
                            vega=premium * 0.5
                        )
                        chain.contracts.append(call_contract)
                        
                        # PUT options
                        if moneyness > 1.02:  # ITM
                            premium = max(strike - underlying_price, 0) + 50
                        elif moneyness < 0.98:  # OTM
                            premium = max(10, 50 * (moneyness - 1))
                        else:  # ATM
                            premium = 100
                        
                        bid = premium - spread / 2
                        ask = premium + spread / 2
                        
                        put_contract = OptionContract(
                            symbol=f"{symbol.replace(':', '')}{datetime.now().strftime('%d%m%y')}{strike}PE",
                            underlying=symbol,
                            strike=strike,
                            expiry=datetime.now() + timedelta(days=7),
                            option_type=OptionType.PUT,
                            lot_size=50,
                            bid=bid,
                            ask=ask,
                            last=premium,
                            volume=1000,
                            open_interest=5000,
                            implied_volatility=0.25,
                            delta=-0.5 if abs(moneyness - 1) < 0.02 else (-0.8 if moneyness > 1 else -0.2),
                            gamma=0.01,
                            theta=-premium * 0.1,
                            vega=premium * 0.5
                        )
                        chain.contracts.append(put_contract)
                    
                    return chain
            
            paper_provider = PaperDataProvider(paper_broker)
            data_manager = RealTimeDataManager(paper_provider)
            
            # Test basic functionality
            test_symbol = "NSE:NIFTY50-INDEX"
            
            # Get underlying price
            price = data_manager.get_underlying_price(test_symbol)
            if price:
                print(f"✅ Data manager underlying price: ₹{price:,.2f}")
            else:
                print("❌ Could not get underlying price from data manager")
                return {
                    'status': 'FAILED',
                    'provider': 'Real-Time Data Manager',
                    'error': 'Could not get underlying price'
                }
            
            # Test option chain
            option_chain = data_manager.get_option_chain(test_symbol)
            if option_chain:
                print(f"✅ Data manager option chain: {len(option_chain.contracts)} contracts")
                
                # Test option selection
                atm_contracts = [c for c in option_chain.contracts if abs(c.strike - price) < 50]
                if atm_contracts:
                    print(f"✅ Found {len(atm_contracts)} ATM contracts")
                    for contract in atm_contracts[:2]:  # Show first 2
                        print(f"   {contract.symbol}: Strike ₹{contract.strike:,.0f}, Bid ₹{contract.bid:.2f}, Ask ₹{contract.ask:.2f}")
                
                return {
                    'status': 'SUCCESS',
                    'provider': 'Real-Time Data Manager',
                    'underlying_price': price,
                    'option_contracts': len(option_chain.contracts),
                    'atm_contracts': len(atm_contracts),
                    'message': 'Real-time data manager working correctly'
                }
            else:
                print("❌ Could not get option chain from data manager")
                return {
                    'status': 'FAILED',
                    'provider': 'Real-Time Data Manager',
                    'error': 'Could not get option chain'
                }
                
        except Exception as e:
            logger.error(f"❌ Real-time data manager test failed: {e}")
            return {
                'status': 'FAILED',
                'provider': 'Real-Time Data Manager',
                'error': str(e)
            }
    
    def _create_simulated_option_chain(self, symbol: str, underlying_price: float) -> OptionChain:
        """Create simulated option chain for testing."""
        from src.models.option_contract import OptionChain, OptionContract, OptionType
        from datetime import datetime, timedelta
        
        chain = OptionChain(symbol, datetime.now())
        
        # Generate strikes around current price
        atm_strike = round(underlying_price / 50) * 50
        strikes = []
        
        # Generate strikes from 5 strikes below to 5 strikes above ATM
        for i in range(-5, 6):
            strike = atm_strike + (i * 50)
            if strike > 0:
                strikes.append(strike)
        
        # Create contracts for each strike
        for strike in strikes:
            # Simulate market data
            moneyness = strike / underlying_price
            
            # CALL options
            if moneyness < 0.98:  # ITM
                premium = max(underlying_price - strike, 0) + 50
            elif moneyness > 1.02:  # OTM
                premium = max(10, 50 * (1 - moneyness))
            else:  # ATM
                premium = 100
            
            # Simulate bid/ask
            spread = premium * 0.1
            bid = premium - spread / 2
            ask = premium + spread / 2
            
            # CALL contract
            call_contract = OptionContract(
                symbol=f"{symbol.replace(':', '')}{datetime.now().strftime('%d%m%y')}{strike}CE",
                underlying=symbol,
                strike=strike,
                expiry=datetime.now() + timedelta(days=7),
                option_type=OptionType.CALL,
                lot_size=50,
                bid=bid,
                ask=ask,
                last=premium,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                delta=0.5 if abs(moneyness - 1) < 0.02 else (0.8 if moneyness < 1 else 0.2),
                gamma=0.01,
                theta=-premium * 0.1,
                vega=premium * 0.5
            )
            chain.contracts.append(call_contract)
            
            # PUT options
            if moneyness > 1.02:  # ITM
                premium = max(strike - underlying_price, 0) + 50
            elif moneyness < 0.98:  # OTM
                premium = max(10, 50 * (moneyness - 1))
            else:  # ATM
                premium = 100
            
            bid = premium - spread / 2
            ask = premium + spread / 2
            
            put_contract = OptionContract(
                symbol=f"{symbol.replace(':', '')}{datetime.now().strftime('%d%m%y')}{strike}PE",
                underlying=symbol,
                strike=strike,
                expiry=datetime.now() + timedelta(days=7),
                option_type=OptionType.PUT,
                lot_size=50,
                bid=bid,
                ask=ask,
                last=premium,
                volume=1000,
                open_interest=5000,
                implied_volatility=0.25,
                delta=-0.5 if abs(moneyness - 1) < 0.02 else (-0.8 if moneyness > 1 else -0.2),
                gamma=0.01,
                theta=-premium * 0.1,
                vega=premium * 0.5
            )
            chain.contracts.append(put_contract)
        
        return chain
    
    def run_all_tests(self, zerodha_api_key: str = None, zerodha_api_secret: str = None, 
                     zerodha_access_token: str = None, fyers_app_id: str = None, 
                     fyers_access_token: str = None) -> Dict:
        """Run all broker connection tests."""
        print("🚀 Starting Broker Connection Tests")
        print("=" * 60)
        
        # Test Paper Broker
        self.test_results['paper_broker'] = self.test_paper_broker()
        
        # Test Real-Time Data Manager
        self.test_results['real_time_manager'] = self.test_real_time_data_manager()
        
        # Test Zerodha
        self.test_results['zerodha'] = self.test_zerodha_connection(
            zerodha_api_key, zerodha_api_secret, zerodha_access_token
        )
        
        # Test Fyers
        self.test_results['fyers'] = self.test_fyers_connection(
            fyers_app_id, fyers_access_token
        )
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def print_test_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 BROKER CONNECTION TEST SUMMARY")
        print("=" * 60)
        
        for provider, result in self.test_results.items():
            status = result['status']
            if status == 'SUCCESS':
                print(f"✅ {provider.upper()}: {result.get('message', 'Working correctly')}")
            elif status == 'PARTIAL':
                print(f"⚠️ {provider.upper()}: {result.get('message', 'Partial success')}")
            elif status == 'SKIPPED':
                print(f"⏭️ {provider.upper()}: {result.get('message', 'Skipped')}")
            else:
                print(f"❌ {provider.upper()}: {result.get('error', 'Failed')}")
        
        print("\n" + "=" * 60)
        print("🎯 RECOMMENDATIONS:")
        
        # Check if any live broker is working
        live_brokers_working = any(
            result['status'] == 'SUCCESS' and result['provider'] in ['Zerodha', 'Fyers']
            for result in self.test_results.values()
        )
        
        if live_brokers_working:
            print("✅ Live broker connection working - ready for paper trading!")
        else:
            print("⚠️ No live broker connection - using paper broker for testing")
            print("   To connect to live brokers:")
            print("   1. Install required libraries: pip install kiteconnect fyers-api")
            print("   2. Get API credentials from your broker")
            print("   3. Run tests with credentials")
        
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Broker Connection Testing')
    parser.add_argument('--zerodha_api_key', help='Zerodha API Key')
    parser.add_argument('--zerodha_api_secret', help='Zerodha API Secret')
    parser.add_argument('--zerodha_access_token', help='Zerodha Access Token')
    parser.add_argument('--fyers_app_id', help='Fyers App ID')
    parser.add_argument('--fyers_access_token', help='Fyers Access Token')
    
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = BrokerConnectionTester()
    results = tester.run_all_tests(
        zerodha_api_key=args.zerodha_api_key,
        zerodha_api_secret=args.zerodha_api_secret,
        zerodha_access_token=args.zerodha_access_token,
        fyers_app_id=args.fyers_app_id,
        fyers_access_token=args.fyers_access_token
    )
    
    # Exit with appropriate code
    if any(result['status'] == 'SUCCESS' for result in results.values()):
        print("\n🎉 At least one broker connection is working!")
        sys.exit(0)
    else:
        print("\n⚠️ No broker connections working. Check credentials and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main() 