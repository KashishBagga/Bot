#!/usr/bin/env python3
"""
Options Trading Strategies with Advanced Analytics
"""

import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptionContract:
    """Option contract data structure"""
    symbol: str
    strike_price: float
    expiry_date: str
    option_type: str  # 'CE' for Call, 'PE' for Put
    premium: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float

@dataclass
class OptionsStrategy:
    """Options trading strategy data structure"""
    strategy_name: str
    strategy_type: str  # 'BULLISH', 'BEARISH', 'NEUTRAL', 'VOLATILITY'
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    contracts: List[OptionContract]
    entry_price: float
    target_price: float
    stop_loss_price: float
    confidence: float

class OptionsTradingStrategies:
    """Advanced options trading strategies with analytics"""
    
    def __init__(self):
        self.database = None
        self.real_time_data = None
        self._initialize_components()
        
        # Options data cache
        self.options_data = {}
        self.strategies = []
        
    def _initialize_components(self):
        """Initialize components"""
        try:
            from src.models.enhanced_database import EnhancedTradingDatabase
            from src.core.enhanced_real_time_manager import EnhancedRealTimeDataManager
            from src.api.fyers import FyersClient
            
            self.database = EnhancedTradingDatabase("data/enhanced_trading.db")
            data_provider = FyersClient()
            symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
            self.real_time_data = EnhancedRealTimeDataManager(data_provider, symbols)
            
            logger.info("✅ Options trading strategies initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize options strategies: {e}")
            raise
    
    def get_options_chain(self, underlying_symbol: str, expiry_date: str = None) -> Dict[str, List[OptionContract]]:
        """Get options chain for underlying symbol"""
        try:
            # Mock options chain data (in real implementation, this would fetch from API)
            current_price = self.real_time_data.get_current_price(underlying_symbol) or 19500.0
            
            if not expiry_date:
                # Get next Thursday (typical NSE expiry)
                today = datetime.now()
                days_ahead = (3 - today.weekday()) % 7  # Thursday is 3
                if days_ahead == 0:
                    days_ahead = 7
                expiry_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
            options_chain = {}
            
            # Generate call options
            call_options = []
            for i in range(-10, 11):  # 10 strikes above and below current price
                strike = current_price + (i * 50)  # 50 point intervals
                call_options.append(OptionContract(
                    symbol=f"{underlying_symbol.replace('NSE:', '')}{expiry_date.replace('-', '')}CE{int(strike)}",
                    strike_price=strike,
                    expiry_date=expiry_date,
                    option_type="CE",
                    premium=max(10, (current_price - strike) * 0.1 + np.random.uniform(5, 50)),
                    volume=np.random.randint(100, 10000),
                    open_interest=np.random.randint(1000, 50000),
                    implied_volatility=np.random.uniform(0.15, 0.35),
                    delta=max(0, min(1, 0.5 + (current_price - strike) / (current_price * 0.1))),
                    gamma=np.random.uniform(0.001, 0.01),
                    theta=-np.random.uniform(0.1, 2.0),
                    vega=np.random.uniform(0.1, 0.5)
                ))
            
            # Generate put options
            put_options = []
            for i in range(-10, 11):
                strike = current_price + (i * 50)
                put_options.append(OptionContract(
                    symbol=f"{underlying_symbol.replace('NSE:', '')}{expiry_date.replace('-', '')}PE{int(strike)}",
                    strike_price=strike,
                    expiry_date=expiry_date,
                    option_type="PE",
                    premium=max(10, (strike - current_price) * 0.1 + np.random.uniform(5, 50)),
                    volume=np.random.randint(100, 10000),
                    open_interest=np.random.randint(1000, 50000),
                    implied_volatility=np.random.uniform(0.15, 0.35),
                    delta=max(-1, min(0, -0.5 + (strike - current_price) / (current_price * 0.1))),
                    gamma=np.random.uniform(0.001, 0.01),
                    theta=-np.random.uniform(0.1, 2.0),
                    vega=np.random.uniform(0.1, 0.5)
                ))
            
            options_chain["CE"] = call_options
            options_chain["PE"] = put_options
            
            self.options_data[underlying_symbol] = options_chain
            logger.info(f"✅ Options chain generated for {underlying_symbol}")
            
            return options_chain
            
        except Exception as e:
            logger.error(f"❌ Error getting options chain: {e}")
            return {}
    
    def long_call_strategy(self, underlying_symbol: str, strike_price: float, 
                          current_price: float, expiry_date: str = None) -> OptionsStrategy:
        """Long Call strategy - Bullish outlook"""
        try:
            options_chain = self.get_options_chain(underlying_symbol, expiry_date)
            
            # Find closest call option
            call_options = options_chain.get("CE", [])
            closest_call = min(call_options, key=lambda x: abs(x.strike_price - strike_price))
            
            # Calculate strategy parameters
            premium_paid = closest_call.premium
            max_profit = float('inf')  # Unlimited upside
            max_loss = premium_paid
            breakeven = strike_price + premium_paid
            
            # Calculate confidence based on delta and IV
            confidence = min(95, max(30, (abs(closest_call.delta) * 100) + 
                                   (1 - closest_call.implied_volatility) * 50))
            
            strategy = OptionsStrategy(
                strategy_name="Long Call",
                strategy_type="BULLISH",
                risk_level="MEDIUM",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                contracts=[closest_call],
                entry_price=premium_paid,
                target_price=breakeven * 1.1,  # 10% above breakeven
                stop_loss_price=premium_paid * 0.5,  # 50% of premium
                confidence=confidence
            )
            
            logger.info(f"✅ Long Call strategy created for {underlying_symbol}")
            return strategy
            
        except Exception as e:
            logger.error(f"❌ Error creating Long Call strategy: {e}")
            return None
    
    def long_put_strategy(self, underlying_symbol: str, strike_price: float,
                         current_price: float, expiry_date: str = None) -> OptionsStrategy:
        """Long Put strategy - Bearish outlook"""
        try:
            options_chain = self.get_options_chain(underlying_symbol, expiry_date)
            
            # Find closest put option
            put_options = options_chain.get("PE", [])
            closest_put = min(put_options, key=lambda x: abs(x.strike_price - strike_price))
            
            # Calculate strategy parameters
            premium_paid = closest_put.premium
            max_profit = strike_price - premium_paid  # Limited by underlying going to zero
            max_loss = premium_paid
            breakeven = strike_price - premium_paid
            
            # Calculate confidence
            confidence = min(95, max(30, (abs(closest_put.delta) * 100) + 
                                   (1 - closest_put.implied_volatility) * 50))
            
            strategy = OptionsStrategy(
                strategy_name="Long Put",
                strategy_type="BEARISH",
                risk_level="MEDIUM",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                contracts=[closest_put],
                entry_price=premium_paid,
                target_price=breakeven * 0.9,  # 10% below breakeven
                stop_loss_price=premium_paid * 0.5,
                confidence=confidence
            )
            
            logger.info(f"✅ Long Put strategy created for {underlying_symbol}")
            return strategy
            
        except Exception as e:
            logger.error(f"❌ Error creating Long Put strategy: {e}")
            return None
    
    def covered_call_strategy(self, underlying_symbol: str, strike_price: float,
                             current_price: float, shares_owned: int = 100,
                             expiry_date: str = None) -> OptionsStrategy:
        """Covered Call strategy - Neutral to slightly bullish"""
        try:
            options_chain = self.get_options_chain(underlying_symbol, expiry_date)
            
            # Find closest call option
            call_options = options_chain.get("CE", [])
            closest_call = min(call_options, key=lambda x: abs(x.strike_price - strike_price))
            
            # Calculate strategy parameters
            premium_received = closest_call.premium
            max_profit = (strike_price - current_price) * shares_owned + premium_received
            max_loss = (current_price - 0) * shares_owned - premium_received  # If underlying goes to zero
            breakeven = current_price - premium_received
            
            # Calculate confidence
            confidence = min(90, max(40, 60 + (premium_received / current_price) * 100))
            
            strategy = OptionsStrategy(
                strategy_name="Covered Call",
                strategy_type="NEUTRAL",
                risk_level="LOW",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven],
                contracts=[closest_call],
                entry_price=current_price - premium_received,
                target_price=strike_price,
                stop_loss_price=breakeven * 0.95,
                confidence=confidence
            )
            
            logger.info(f"✅ Covered Call strategy created for {underlying_symbol}")
            return strategy
            
        except Exception as e:
            logger.error(f"❌ Error creating Covered Call strategy: {e}")
            return None
    
    def iron_condor_strategy(self, underlying_symbol: str, current_price: float,
                            expiry_date: str = None) -> OptionsStrategy:
        """Iron Condor strategy - Neutral outlook with limited risk"""
        try:
            options_chain = self.get_options_chain(underlying_symbol, expiry_date)
            
            # Define strikes (typically 1-2 standard deviations from current price)
            put_strike_1 = current_price * 0.95  # 5% below current
            put_strike_2 = current_price * 0.97  # 3% below current
            call_strike_1 = current_price * 1.03  # 3% above current
            call_strike_2 = current_price * 1.05  # 5% above current
            
            # Find closest options
            put_options = options_chain.get("PE", [])
            call_options = options_chain.get("CE", [])
            
            put_1 = min(put_options, key=lambda x: abs(x.strike_price - put_strike_1))
            put_2 = min(put_options, key=lambda x: abs(x.strike_price - put_strike_2))
            call_1 = min(call_options, key=lambda x: abs(x.strike_price - call_strike_1))
            call_2 = min(call_options, key=lambda x: abs(x.strike_price - call_strike_2))
            
            # Calculate strategy parameters
            net_credit = put_1.premium + call_2.premium - put_2.premium - call_1.premium
            max_profit = net_credit
            max_loss = (put_strike_2 - put_strike_1) - net_credit
            breakeven_low = put_strike_2 - net_credit
            breakeven_high = call_strike_1 + net_credit
            
            # Calculate confidence
            confidence = min(85, max(35, 50 + (net_credit / current_price) * 200))
            
            strategy = OptionsStrategy(
                strategy_name="Iron Condor",
                strategy_type="NEUTRAL",
                risk_level="MEDIUM",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_low, breakeven_high],
                contracts=[put_1, put_2, call_1, call_2],
                entry_price=net_credit,
                target_price=current_price,
                stop_loss_price=max_loss * 0.8,
                confidence=confidence
            )
            
            logger.info(f"✅ Iron Condor strategy created for {underlying_symbol}")
            return strategy
            
        except Exception as e:
            logger.error(f"❌ Error creating Iron Condor strategy: {e}")
            return None
    
    def straddle_strategy(self, underlying_symbol: str, strike_price: float,
                         current_price: float, expiry_date: str = None) -> OptionsStrategy:
        """Straddle strategy - Volatility play"""
        try:
            options_chain = self.get_options_chain(underlying_symbol, expiry_date)
            
            # Find ATM call and put
            call_options = options_chain.get("CE", [])
            put_options = options_chain.get("PE", [])
            
            atm_call = min(call_options, key=lambda x: abs(x.strike_price - strike_price))
            atm_put = min(put_options, key=lambda x: abs(x.strike_price - strike_price))
            
            # Calculate strategy parameters
            total_premium = atm_call.premium + atm_put.premium
            max_profit = float('inf')  # Unlimited in both directions
            max_loss = total_premium
            breakeven_low = strike_price - total_premium
            breakeven_high = strike_price + total_premium
            
            # Calculate confidence based on implied volatility
            confidence = min(90, max(40, (atm_call.implied_volatility + atm_put.implied_volatility) * 100))
            
            strategy = OptionsStrategy(
                strategy_name="Long Straddle",
                strategy_type="VOLATILITY",
                risk_level="HIGH",
                max_profit=max_profit,
                max_loss=max_loss,
                breakeven_points=[breakeven_low, breakeven_high],
                contracts=[atm_call, atm_put],
                entry_price=total_premium,
                target_price=strike_price,
                stop_loss_price=total_premium * 0.6,
                confidence=confidence
            )
            
            logger.info(f"✅ Straddle strategy created for {underlying_symbol}")
            return strategy
            
        except Exception as e:
            logger.error(f"❌ Error creating Straddle strategy: {e}")
            return None
    
    def analyze_strategy_performance(self, strategy: OptionsStrategy) -> Dict[str, Any]:
        """Analyze strategy performance and risk metrics"""
        try:
            analysis = {
                "strategy_name": strategy.strategy_name,
                "risk_reward_ratio": strategy.max_profit / strategy.max_loss if strategy.max_loss > 0 else float('inf'),
                "confidence": strategy.confidence,
                "risk_level": strategy.risk_level,
                "max_profit": strategy.max_profit,
                "max_loss": strategy.max_loss,
                "breakeven_points": strategy.breakeven_points,
                "contracts_count": len(strategy.contracts),
                "total_premium": sum(contract.premium for contract in strategy.contracts),
                "average_iv": np.mean([contract.implied_volatility for contract in strategy.contracts]),
                "total_delta": sum(contract.delta for contract in strategy.contracts),
                "total_gamma": sum(contract.gamma for contract in strategy.contracts),
                "total_theta": sum(contract.theta for contract in strategy.contracts),
                "total_vega": sum(contract.vega for contract in strategy.contracts)
            }
            
            # Risk assessment
            if analysis["risk_reward_ratio"] > 3:
                analysis["risk_assessment"] = "EXCELLENT"
            elif analysis["risk_reward_ratio"] > 2:
                analysis["risk_assessment"] = "GOOD"
            elif analysis["risk_reward_ratio"] > 1:
                analysis["risk_assessment"] = "FAIR"
            else:
                analysis["risk_assessment"] = "POOR"
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing strategy performance: {e}")
            return {}
    
    def get_recommended_strategies(self, underlying_symbol: str, market_outlook: str = "NEUTRAL") -> List[OptionsStrategy]:
        """Get recommended strategies based on market outlook"""
        try:
            current_price = self.real_time_data.get_current_price(underlying_symbol) or 19500.0
            recommended_strategies = []
            
            if market_outlook == "BULLISH":
                # Long Call
                strategy = self.long_call_strategy(underlying_symbol, current_price * 1.02, current_price)
                if strategy:
                    recommended_strategies.append(strategy)
                
                # Covered Call
                strategy = self.covered_call_strategy(underlying_symbol, current_price * 1.05, current_price)
                if strategy:
                    recommended_strategies.append(strategy)
                    
            elif market_outlook == "BEARISH":
                # Long Put
                strategy = self.long_put_strategy(underlying_symbol, current_price * 0.98, current_price)
                if strategy:
                    recommended_strategies.append(strategy)
                    
            elif market_outlook == "NEUTRAL":
                # Iron Condor
                strategy = self.iron_condor_strategy(underlying_symbol, current_price)
                if strategy:
                    recommended_strategies.append(strategy)
                
                # Covered Call
                strategy = self.covered_call_strategy(underlying_symbol, current_price * 1.03, current_price)
                if strategy:
                    recommended_strategies.append(strategy)
                    
            elif market_outlook == "VOLATILE":
                # Straddle
                strategy = self.straddle_strategy(underlying_symbol, current_price, current_price)
                if strategy:
                    recommended_strategies.append(strategy)
            
            # Sort by confidence
            recommended_strategies.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"✅ Generated {len(recommended_strategies)} recommended strategies for {market_outlook} outlook")
            return recommended_strategies
            
        except Exception as e:
            logger.error(f"❌ Error getting recommended strategies: {e}")
            return []
    
    def display_strategies_dashboard(self):
        """Display options strategies dashboard"""
        print("\n" + "="*100)
        print("📊 OPTIONS TRADING STRATEGIES DASHBOARD")
        print("="*100)
        
        symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "NSE:FINNIFTY-INDEX"]
        market_outlooks = ["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE"]
        
        for symbol in symbols:
            print(f"\n🎯 {symbol}")
            print("-" * 60)
            
            current_price = self.real_time_data.get_current_price(symbol) or 19500.0
            print(f"  Current Price: ₹{current_price:,.2f}")
            
            for outlook in market_outlooks:
                print(f"\n  📈 {outlook} Strategies:")
                strategies = self.get_recommended_strategies(symbol, outlook)
                
                for i, strategy in enumerate(strategies[:2], 1):  # Show top 2 strategies
                    analysis = self.analyze_strategy_performance(strategy)
                    
                    print(f"    {i}. {strategy.strategy_name}")
                    print(f"       Confidence: {strategy.confidence:.1f}%")
                    print(f"       Risk Level: {strategy.risk_level}")
                    print(f"       Max Profit: ₹{strategy.max_profit:,.2f}")
                    print(f"       Max Loss: ₹{strategy.max_loss:,.2f}")
                    print(f"       Risk/Reward: {analysis['risk_reward_ratio']:.2f}")
                    print(f"       Assessment: {analysis['risk_assessment']}")
        
        print("\n" + "="*100)
        print(f"📊 Options Strategies Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)

def main():
    """Main function"""
    options_strategies = OptionsTradingStrategies()
    
    try:
        # Display strategies dashboard
        options_strategies.display_strategies_dashboard()
        
    except Exception as e:
        logger.error(f"❌ Options strategies error: {e}")

if __name__ == "__main__":
    main()
