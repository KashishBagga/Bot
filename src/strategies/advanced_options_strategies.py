#!/usr/bin/env python3
"""
Advanced Options Trading Strategies
Comprehensive options strategies for different market conditions
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
from enum import Enum
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    VOLATILE = "VOLATILE"
    SIDEWAYS = "SIDEWAYS"
    BREAKOUT = "BREAKOUT"

class OptionsStrategy(Enum):
    LONG_CALL = "LONG_CALL"
    LONG_PUT = "LONG_PUT"
    COVERED_CALL = "COVERED_CALL"
    PROTECTIVE_PUT = "PROTECTIVE_PUT"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    IRON_CONDOR = "IRON_CONDOR"
    BUTTERFLY_SPREAD = "BUTTERFLY_SPREAD"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    DIAGONAL_SPREAD = "DIAGONAL_SPREAD"

@dataclass
class OptionsChain:
    """Options chain data structure"""
    symbol: str
    expiry_date: datetime
    strike_prices: List[float]
    call_options: Dict[float, Dict[str, Any]]
    put_options: Dict[float, Dict[str, Any]]
    underlying_price: float
    timestamp: datetime

@dataclass
class OptionsPosition:
    """Options position data structure"""
    strategy: OptionsStrategy
    symbol: str
    positions: List[Dict[str, Any]]
    entry_price: float
    current_price: float
    quantity: int
    expiry_date: datetime
    greeks: Dict[str, float]
    pnl: float
    timestamp: datetime

@dataclass
class StrategyRecommendation:
    """Strategy recommendation data structure"""
    strategy: OptionsStrategy
    market_regime: MarketRegime
    confidence: float
    expected_return: float
    max_loss: float
    probability_of_profit: float
    time_to_expiry: int
    reasoning: str
    risk_level: str

class AdvancedOptionsStrategies:
    """Advanced options trading strategies system"""
    
    def __init__(self):
        self.strategies = {}
        self.market_regimes = {}
        self.options_chains = {}
        
    def generate_options_chain(self, symbol: str, underlying_price: float, 
                             expiry_date: datetime) -> OptionsChain:
        """Generate synthetic options chain"""
        try:
            # Generate strike prices around current price
            strike_range = underlying_price * 0.2  # 20% range
            strike_prices = np.arange(
                underlying_price - strike_range,
                underlying_price + strike_range,
                underlying_price * 0.01  # 1% intervals
            )
            
            call_options = {}
            put_options = {}
            
            for strike in strike_prices:
                # Calculate option prices using Black-Scholes approximation
                call_price = self._calculate_option_price(underlying_price, strike, 
                                                        expiry_date, 'call')
                put_price = self._calculate_option_price(underlying_price, strike, 
                                                       expiry_date, 'put')
                
                # Calculate Greeks
                call_greeks = self._calculate_greeks(underlying_price, strike, 
                                                   expiry_date, 'call')
                put_greeks = self._calculate_greeks(underlying_price, strike, 
                                                  expiry_date, 'put')
                
                call_options[strike] = {
                    'price': call_price,
                    'bid': call_price * 0.98,
                    'ask': call_price * 1.02,
                    'volume': np.random.randint(100, 1000),
                    'open_interest': np.random.randint(1000, 10000),
                    'implied_volatility': np.random.uniform(0.15, 0.35),
                    'greeks': call_greeks
                }
                
                put_options[strike] = {
                    'price': put_price,
                    'bid': put_price * 0.98,
                    'ask': put_price * 1.02,
                    'volume': np.random.randint(100, 1000),
                    'open_interest': np.random.randint(1000, 10000),
                    'implied_volatility': np.random.uniform(0.15, 0.35),
                    'greeks': put_greeks
                }
            
            options_chain = OptionsChain(
                symbol=symbol,
                expiry_date=expiry_date,
                strike_prices=strike_prices.tolist(),
                call_options=call_options,
                put_options=put_options,
                underlying_price=underlying_price,
                timestamp=datetime.now()
            )
            
            self.options_chains[symbol] = options_chain
            
            logger.info(f"✅ Generated options chain for {symbol} with {len(strike_prices)} strikes")
            
            return options_chain
            
        except Exception as e:
            logger.error(f"❌ Options chain generation failed: {e}")
            return None
    
    def _calculate_option_price(self, underlying: float, strike: float, 
                              expiry: datetime, option_type: str) -> float:
        """Calculate option price using simplified Black-Scholes"""
        try:
            time_to_expiry = (expiry - datetime.now()).days / 365.0
            if time_to_expiry <= 0:
                return 0.0
            
            # Simplified Black-Scholes parameters
            risk_free_rate = 0.05
            volatility = 0.25
            
            # Calculate d1 and d2
            d1 = (np.log(underlying / strike) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                 (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            # Calculate option price
            if option_type == 'call':
                price = (underlying * self._normal_cdf(d1) - 
                        strike * np.exp(-risk_free_rate * time_to_expiry) * 
                        self._normal_cdf(d2))
            else:  # put
                price = (strike * np.exp(-risk_free_rate * time_to_expiry) * 
                        self._normal_cdf(-d2) - underlying * self._normal_cdf(-d1))
            
            return max(0.0, price)
            
        except Exception as e:
            logger.error(f"❌ Option price calculation failed: {e}")
            return 0.0
    
    def _calculate_greeks(self, underlying: float, strike: float, 
                         expiry: datetime, option_type: str) -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            time_to_expiry = (expiry - datetime.now()).days / 365.0
            if time_to_expiry <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            risk_free_rate = 0.05
            volatility = 0.25
            
            # Calculate d1 and d2
            d1 = (np.log(underlying / strike) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                 (volatility * np.sqrt(time_to_expiry))
            d2 = d1 - volatility * np.sqrt(time_to_expiry)
            
            # Calculate Greeks
            delta = self._normal_cdf(d1) if option_type == 'call' else self._normal_cdf(d1) - 1
            gamma = self._normal_pdf(d1) / (underlying * volatility * np.sqrt(time_to_expiry))
            theta = (-(underlying * self._normal_pdf(d1) * volatility) / (2 * np.sqrt(time_to_expiry)) - 
                    risk_free_rate * strike * np.exp(-risk_free_rate * time_to_expiry) * 
                    self._normal_cdf(d2)) / 365
            vega = underlying * self._normal_pdf(d1) * np.sqrt(time_to_expiry) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
            
        except Exception as e:
            logger.error(f"❌ Greeks calculation failed: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def _normal_cdf(self, x: float) -> float:
        """Normal cumulative distribution function"""
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))
    
    def _normal_pdf(self, x: float) -> float:
        """Normal probability density function"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def detect_market_regime(self, price_data: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        try:
            if len(price_data) < 20:
                return MarketRegime.SIDEWAYS
            
            # Calculate technical indicators
            price_data['sma_20'] = price_data['close'].rolling(20).mean()
            price_data['volatility'] = price_data['close'].pct_change().rolling(20).std()
            price_data['momentum'] = price_data['close'].pct_change(20)
            
            current_price = price_data['close'].iloc[-1]
            sma_20 = price_data['sma_20'].iloc[-1]
            volatility = price_data['volatility'].iloc[-1]
            momentum = price_data['momentum'].iloc[-1]
            
            # Regime detection logic
            if volatility > 0.03:  # High volatility
                return MarketRegime.VOLATILE
            elif abs(momentum) > 0.05:  # Strong trend
                return MarketRegime.TRENDING
            elif abs(current_price - sma_20) / sma_20 < 0.02:  # Sideways
                return MarketRegime.SIDEWAYS
            else:
                return MarketRegime.BREAKOUT
                
        except Exception as e:
            logger.error(f"❌ Market regime detection failed: {e}")
            return MarketRegime.SIDEWAYS
    
    def recommend_strategy(self, symbol: str, market_regime: MarketRegime, 
                          risk_tolerance: str = 'MODERATE') -> List[StrategyRecommendation]:
        """Recommend options strategies based on market regime"""
        try:
            recommendations = []
            
            if market_regime == MarketRegime.TRENDING:
                # Trending market strategies
                recommendations.extend([
                    StrategyRecommendation(
                        strategy=OptionsStrategy.LONG_CALL,
                        market_regime=market_regime,
                        confidence=0.8,
                        expected_return=0.15,
                        max_loss=1.0,
                        probability_of_profit=0.6,
                        time_to_expiry=30,
                        reasoning="Strong upward trend detected, long calls for directional play",
                        risk_level="HIGH"
                    ),
                    StrategyRecommendation(
                        strategy=OptionsStrategy.COVERED_CALL,
                        market_regime=market_regime,
                        confidence=0.7,
                        expected_return=0.08,
                        max_loss=0.3,
                        probability_of_profit=0.7,
                        time_to_expiry=30,
                        reasoning="Generate income while maintaining upside potential",
                        risk_level="MODERATE"
                    )
                ])
            
            elif market_regime == MarketRegime.VOLATILE:
                # Volatile market strategies
                recommendations.extend([
                    StrategyRecommendation(
                        strategy=OptionsStrategy.STRADDLE,
                        market_regime=market_regime,
                        confidence=0.75,
                        expected_return=0.20,
                        max_loss=1.0,
                        probability_of_profit=0.5,
                        time_to_expiry=30,
                        reasoning="High volatility expected, straddle for non-directional play",
                        risk_level="HIGH"
                    ),
                    StrategyRecommendation(
                        strategy=OptionsStrategy.IRON_CONDOR,
                        market_regime=market_regime,
                        confidence=0.6,
                        expected_return=0.12,
                        max_loss=0.4,
                        probability_of_profit=0.6,
                        time_to_expiry=30,
                        reasoning="Range-bound strategy for volatile but sideways market",
                        risk_level="MODERATE"
                    )
                ])
            
            elif market_regime == MarketRegime.SIDEWAYS:
                # Sideways market strategies
                recommendations.extend([
                    StrategyRecommendation(
                        strategy=OptionsStrategy.IRON_CONDOR,
                        market_regime=market_regime,
                        confidence=0.8,
                        expected_return=0.10,
                        max_loss=0.3,
                        probability_of_profit=0.7,
                        time_to_expiry=30,
                        reasoning="Sideways market ideal for range-bound strategies",
                        risk_level="LOW"
                    ),
                    StrategyRecommendation(
                        strategy=OptionsStrategy.CALENDAR_SPREAD,
                        market_regime=market_regime,
                        confidence=0.7,
                        expected_return=0.08,
                        max_loss=0.2,
                        probability_of_profit=0.65,
                        time_to_expiry=30,
                        reasoning="Time decay strategy for sideways market",
                        risk_level="LOW"
                    )
                ])
            
            elif market_regime == MarketRegime.BREAKOUT:
                # Breakout market strategies
                recommendations.extend([
                    StrategyRecommendation(
                        strategy=OptionsStrategy.STRANGLE,
                        market_regime=market_regime,
                        confidence=0.7,
                        expected_return=0.18,
                        max_loss=0.8,
                        probability_of_profit=0.55,
                        time_to_expiry=30,
                        reasoning="Breakout expected, strangle for directional play with lower cost",
                        risk_level="MODERATE"
                    ),
                    StrategyRecommendation(
                        strategy=OptionsStrategy.BUTTERFLY_SPREAD,
                        market_regime=market_regime,
                        confidence=0.6,
                        expected_return=0.12,
                        max_loss=0.3,
                        probability_of_profit=0.6,
                        time_to_expiry=30,
                        reasoning="Limited risk strategy for potential breakout",
                        risk_level="LOW"
                    )
                ])
            
            # Filter by risk tolerance
            if risk_tolerance == 'LOW':
                recommendations = [r for r in recommendations if r.risk_level in ['LOW', 'MODERATE']]
            elif risk_tolerance == 'HIGH':
                recommendations = [r for r in recommendations if r.risk_level in ['MODERATE', 'HIGH']]
            
            # Sort by confidence
            recommendations.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.info(f"✅ Generated {len(recommendations)} strategy recommendations for {market_regime.value} market")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Strategy recommendation failed: {e}")
            return []
    
    def calculate_strategy_pnl(self, position: OptionsPosition, 
                             current_underlying_price: float) -> Dict[str, Any]:
        """Calculate strategy P&L"""
        try:
            total_pnl = 0.0
            total_delta = 0.0
            total_gamma = 0.0
            total_theta = 0.0
            total_vega = 0.0
            
            for pos in position.positions:
                option_type = pos['type']
                strike = pos['strike']
                quantity = pos['quantity']
                entry_price = pos['entry_price']
                
                # Calculate current option price
                current_price = self._calculate_option_price(
                    current_underlying_price, strike, position.expiry_date, option_type
                )
                
                # Calculate P&L
                pnl = (current_price - entry_price) * quantity
                total_pnl += pnl
                
                # Calculate Greeks
                greeks = self._calculate_greeks(
                    current_underlying_price, strike, position.expiry_date, option_type
                )
                
                total_delta += greeks['delta'] * quantity
                total_gamma += greeks['gamma'] * quantity
                total_theta += greeks['theta'] * quantity
                total_vega += greeks['vega'] * quantity
            
            return {
                'total_pnl': total_pnl,
                'total_delta': total_delta,
                'total_gamma': total_gamma,
                'total_theta': total_theta,
                'total_vega': total_vega,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ P&L calculation failed: {e}")
            return {}
    
    def run_options_analysis(self, symbol: str, underlying_price: float) -> Dict[str, Any]:
        """Run comprehensive options analysis"""
        try:
            logger.info(f"🚀 Starting options analysis for {symbol}")
            
            # Generate options chain
            expiry_date = datetime.now() + timedelta(days=30)
            options_chain = self.generate_options_chain(symbol, underlying_price, expiry_date)
            
            if not options_chain:
                return {'error': 'Failed to generate options chain'}
            
            # Detect market regime (using synthetic data)
            price_data = pd.DataFrame({
                'close': np.random.uniform(underlying_price * 0.95, underlying_price * 1.05, 50)
            })
            market_regime = self.detect_market_regime(price_data)
            
            # Get strategy recommendations
            recommendations = self.recommend_strategy(symbol, market_regime)
            
            # Analyze best strategy
            best_strategy = recommendations[0] if recommendations else None
            
            results = {
                'symbol': symbol,
                'underlying_price': underlying_price,
                'market_regime': market_regime.value,
                'options_chain': {
                    'strikes_count': len(options_chain.strike_prices),
                    'expiry_date': expiry_date.isoformat(),
                    'timestamp': options_chain.timestamp.isoformat()
                },
                'recommendations': [
                    {
                        'strategy': rec.strategy.value,
                        'confidence': rec.confidence,
                        'expected_return': rec.expected_return,
                        'max_loss': rec.max_loss,
                        'probability_of_profit': rec.probability_of_profit,
                        'risk_level': rec.risk_level,
                        'reasoning': rec.reasoning
                    } for rec in recommendations
                ],
                'best_strategy': {
                    'strategy': best_strategy.strategy.value if best_strategy else None,
                    'confidence': best_strategy.confidence if best_strategy else 0,
                    'expected_return': best_strategy.expected_return if best_strategy else 0,
                    'reasoning': best_strategy.reasoning if best_strategy else None
                } if best_strategy else None,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Options analysis completed for {symbol}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Options analysis failed: {e}")
            return {'error': str(e)}

def main():
    """Run advanced options strategies analysis"""
    options_system = AdvancedOptionsStrategies()
    
    # Test with sample data
    symbol = "NSE:NIFTY50-INDEX"
    underlying_price = 19500.0
    
    results = options_system.run_options_analysis(symbol, underlying_price)
    
    print("\n" + "="*80)
    print("�� ADVANCED OPTIONS STRATEGIES ANALYSIS")
    print("="*80)
    
    if 'error' in results:
        print(f"❌ Analysis failed: {results['error']}")
        return False
    
    print(f"\n🎯 ANALYSIS RESULTS:")
    print(f"   Symbol: {results['symbol']}")
    print(f"   Underlying Price: {results['underlying_price']}")
    print(f"   Market Regime: {results['market_regime']}")
    print(f"   Options Chain: {results['options_chain']['strikes_count']} strikes")
    
    print(f"\n📋 STRATEGY RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"\n   {i}. {rec['strategy']}")
        print(f"      Confidence: {rec['confidence']:.1%}")
        print(f"      Expected Return: {rec['expected_return']:.1%}")
        print(f"      Max Loss: {rec['max_loss']:.1%}")
        print(f"      Probability of Profit: {rec['probability_of_profit']:.1%}")
        print(f"      Risk Level: {rec['risk_level']}")
        print(f"      Reasoning: {rec['reasoning']}")
    
    if results['best_strategy']:
        print(f"\n🏆 BEST STRATEGY:")
        best = results['best_strategy']
        print(f"   Strategy: {best['strategy']}")
        print(f"   Confidence: {best['confidence']:.1%}")
        print(f"   Expected Return: {best['expected_return']:.1%}")
        print(f"   Reasoning: {best['reasoning']}")
    
    print("\n" + "="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
