#!/usr/bin/env python3
"""
Enhanced Options Pricing with Market IV
Realistic options pricing using market-implied volatility and analytic Greeks
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math
from scipy.stats import norm
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

class OptionType(Enum):
    CALL = "CALL"
    PUT = "PUT"

@dataclass
class OptionGreeks:
    """Option Greeks with stability checks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    stability_score: float  # 0-1, higher is more stable

@dataclass
class OptionPrice:
    """Option price with market data"""
    theoretical_price: float
    market_price: float
    implied_volatility: float
    bid_price: float
    ask_price: float
    mid_price: float
    spread: float
    greeks: OptionGreeks

class EnhancedOptionsPricer:
    """Enhanced options pricing with market IV and stability checks"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.dividend_yield = 0.0   # No dividends for index options
        
    def calculate_black_scholes_price(self, underlying_price: float, strike_price: float,
                                    time_to_expiry: float, volatility: float,
                                    option_type: OptionType, risk_free_rate: float = None) -> float:
        """Calculate Black-Scholes option price"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            if time_to_expiry <= 0:
                return 0.0
            
            # Calculate d1 and d2
            d1 = (math.log(underlying_price / strike_price) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                 (volatility * math.sqrt(time_to_expiry))
            d2 = d1 - volatility * math.sqrt(time_to_expiry)
            
            # Calculate option price
            if option_type == OptionType.CALL:
                price = (underlying_price * norm.cdf(d1) - 
                        strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
            else:  # PUT
                price = (strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                        underlying_price * norm.cdf(-d1))
            
            return max(0.0, price)
            
        except Exception as e:
            logger.error(f"❌ Black-Scholes pricing failed: {e}")
            return 0.0
    
    def calculate_implied_volatility(self, market_price: float, underlying_price: float,
                                   strike_price: float, time_to_expiry: float,
                                   option_type: OptionType, risk_free_rate: float = None) -> float:
        """Calculate implied volatility from market price"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            if time_to_expiry <= 0 or market_price <= 0:
                return 0.0
            
            def objective(vol):
                theoretical_price = self.calculate_black_scholes_price(
                    underlying_price, strike_price, time_to_expiry, vol, option_type, risk_free_rate
                )
                return theoretical_price - market_price
            
            try:
                # Use Brent's method to find implied volatility
                iv = brentq(objective, 0.01, 5.0, xtol=1e-6)
                return iv
            except ValueError:
                # Fallback to simple approximation
                return self._approximate_implied_volatility(
                    market_price, underlying_price, strike_price, time_to_expiry, option_type
                )
                
        except Exception as e:
            logger.error(f"❌ Implied volatility calculation failed: {e}")
            return 0.0
    
    def _approximate_implied_volatility(self, market_price: float, underlying_price: float,
                                      strike_price: float, time_to_expiry: float,
                                      option_type: OptionType) -> float:
        """Approximate implied volatility using simple formula"""
        try:
            # Simple approximation for ATM options
            if abs(underlying_price - strike_price) / underlying_price < 0.05:  # ATM
                iv = market_price / (underlying_price * math.sqrt(time_to_expiry)) * math.sqrt(2 * math.pi)
                return min(max(iv, 0.01), 2.0)  # Clamp between 1% and 200%
            else:
                # For OTM/ITM options, use a more complex approximation
                moneyness = underlying_price / strike_price
                iv = market_price / (underlying_price * math.sqrt(time_to_expiry)) * math.sqrt(2 * math.pi) * moneyness
                return min(max(iv, 0.01), 2.0)
                
        except Exception as e:
            logger.error(f"❌ Approximate IV calculation failed: {e}")
            return 0.25  # Default 25% volatility
    
    def calculate_analytic_greeks(self, underlying_price: float, strike_price: float,
                                time_to_expiry: float, volatility: float,
                                option_type: OptionType, risk_free_rate: float = None) -> OptionGreeks:
        """Calculate analytic Greeks with stability checks"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            if time_to_expiry <= 0:
                return OptionGreeks(0, 0, 0, 0, 0, 0.0)
            
            # Calculate d1 and d2
            d1 = (math.log(underlying_price / strike_price) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / \
                 (volatility * math.sqrt(time_to_expiry))
            d2 = d1 - volatility * math.sqrt(time_to_expiry)
            
            # Calculate Greeks
            delta = norm.cdf(d1) if option_type == OptionType.CALL else norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (underlying_price * volatility * math.sqrt(time_to_expiry))
            theta = (-(underlying_price * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry)) - 
                    risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
            vega = underlying_price * norm.pdf(d1) * math.sqrt(time_to_expiry) / 100
            rho = strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
            
            # Stability check
            stability_score = self._calculate_greeks_stability(
                underlying_price, strike_price, time_to_expiry, volatility, option_type
            )
            
            return OptionGreeks(delta, gamma, theta, vega, rho, stability_score)
            
        except Exception as e:
            logger.error(f"❌ Greeks calculation failed: {e}")
            return OptionGreeks(0, 0, 0, 0, 0, 0.0)
    
    def _calculate_greeks_stability(self, underlying_price: float, strike_price: float,
                                  time_to_expiry: float, volatility: float,
                                  option_type: OptionType) -> float:
        """Calculate stability score for Greeks"""
        try:
            # Check for extreme values that might indicate instability
            stability_factors = []
            
            # Time to expiry check
            if time_to_expiry > 0.1:  # More than 36 days
                stability_factors.append(1.0)
            elif time_to_expiry > 0.01:  # More than 3.6 days
                stability_factors.append(0.8)
            else:
                stability_factors.append(0.5)
            
            # Volatility check
            if 0.1 <= volatility <= 1.0:  # 10% to 100%
                stability_factors.append(1.0)
            elif 0.05 <= volatility <= 2.0:  # 5% to 200%
                stability_factors.append(0.8)
            else:
                stability_factors.append(0.3)
            
            # Moneyness check
            moneyness = underlying_price / strike_price
            if 0.8 <= moneyness <= 1.2:  # Near ATM
                stability_factors.append(1.0)
            elif 0.5 <= moneyness <= 2.0:  # Reasonable range
                stability_factors.append(0.8)
            else:
                stability_factors.append(0.5)
            
            return np.mean(stability_factors)
            
        except Exception as e:
            logger.error(f"❌ Stability calculation failed: {e}")
            return 0.5
    
    def calculate_finite_difference_greeks(self, underlying_price: float, strike_price: float,
                                         time_to_expiry: float, volatility: float,
                                         option_type: OptionType, risk_free_rate: float = None) -> OptionGreeks:
        """Calculate Greeks using finite differences with adaptive step size"""
        try:
            if risk_free_rate is None:
                risk_free_rate = self.risk_free_rate
            
            # Adaptive step size based on underlying price
            price_step = underlying_price * 0.01  # 1% of underlying price
            vol_step = volatility * 0.01  # 1% of volatility
            time_step = 1/365  # 1 day
            
            # Calculate base price
            base_price = self.calculate_black_scholes_price(
                underlying_price, strike_price, time_to_expiry, volatility, option_type, risk_free_rate
            )
            
            # Delta calculation
            price_up = self.calculate_black_scholes_price(
                underlying_price + price_step, strike_price, time_to_expiry, volatility, option_type, risk_free_rate
            )
            price_down = self.calculate_black_scholes_price(
                underlying_price - price_step, strike_price, time_to_expiry, volatility, option_type, risk_free_rate
            )
            delta = (price_up - price_down) / (2 * price_step)
            
            # Gamma calculation
            gamma = (price_up - 2 * base_price + price_down) / (price_step ** 2)
            
            # Theta calculation
            time_down = self.calculate_black_scholes_price(
                underlying_price, strike_price, time_to_expiry - time_step, volatility, option_type, risk_free_rate
            )
            theta = (time_down - base_price) / time_step
            
            # Vega calculation
            vol_up = self.calculate_black_scholes_price(
                underlying_price, strike_price, time_to_expiry, volatility + vol_step, option_type, risk_free_rate
            )
            vega = (vol_up - base_price) / vol_step
            
            # Rho calculation
            rate_up = self.calculate_black_scholes_price(
                underlying_price, strike_price, time_to_expiry, volatility, option_type, risk_free_rate + 0.01
            )
            rho = (rate_up - base_price) / 0.01
            
            # Stability check
            stability_score = self._calculate_greeks_stability(
                underlying_price, strike_price, time_to_expiry, volatility, option_type
            )
            
            return OptionGreeks(delta, gamma, theta, vega, rho, stability_score)
            
        except Exception as e:
            logger.error(f"❌ Finite difference Greeks calculation failed: {e}")
            return OptionGreeks(0, 0, 0, 0, 0, 0.0)
    
    def price_option_with_market_data(self, underlying_price: float, strike_price: float,
                                    time_to_expiry: float, market_price: float,
                                    bid_price: float, ask_price: float,
                                    option_type: OptionType) -> OptionPrice:
        """Price option using market data and implied volatility"""
        try:
            # Calculate implied volatility from market price
            implied_vol = self.calculate_implied_volatility(
                market_price, underlying_price, strike_price, time_to_expiry, option_type
            )
            
            # Calculate theoretical price using implied volatility
            theoretical_price = self.calculate_black_scholes_price(
                underlying_price, strike_price, time_to_expiry, implied_vol, option_type
            )
            
            # Calculate Greeks using analytic formulas
            greeks = self.calculate_analytic_greeks(
                underlying_price, strike_price, time_to_expiry, implied_vol, option_type
            )
            
            # Calculate mid price and spread
            mid_price = (bid_price + ask_price) / 2
            spread = ask_price - bid_price
            
            return OptionPrice(
                theoretical_price=theoretical_price,
                market_price=market_price,
                implied_volatility=implied_vol,
                bid_price=bid_price,
                ask_price=ask_price,
                mid_price=mid_price,
                spread=spread,
                greeks=greeks
            )
            
        except Exception as e:
            logger.error(f"❌ Option pricing with market data failed: {e}")
            return OptionPrice(0, 0, 0, 0, 0, 0, 0, OptionGreeks(0, 0, 0, 0, 0, 0.0))
    
    def generate_options_chain_with_iv(self, symbol: str, underlying_price: float,
                                     expiry_date: datetime) -> Dict[str, Any]:
        """Generate options chain with market-implied volatility"""
        try:
            time_to_expiry = (expiry_date - datetime.now()).days / 365.0
            
            if time_to_expiry <= 0:
                return {'error': 'Expiry date in the past'}
            
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
                # Generate realistic market data
                market_data = self._generate_realistic_market_data(
                    underlying_price, strike, time_to_expiry
                )
                
                # Price call option
                call_price = self.price_option_with_market_data(
                    underlying_price, strike, time_to_expiry,
                    market_data['call_market_price'], market_data['call_bid'],
                    market_data['call_ask'], OptionType.CALL
                )
                
                # Price put option
                put_price = self.price_option_with_market_data(
                    underlying_price, strike, time_to_expiry,
                    market_data['put_market_price'], market_data['put_bid'],
                    market_data['put_ask'], OptionType.PUT
                )
                
                call_options[strike] = {
                    'price': call_price.theoretical_price,
                    'market_price': call_price.market_price,
                    'bid': call_price.bid_price,
                    'ask': call_price.ask_price,
                    'mid': call_price.mid_price,
                    'spread': call_price.spread,
                    'implied_volatility': call_price.implied_volatility,
                    'greeks': {
                        'delta': call_price.greeks.delta,
                        'gamma': call_price.greeks.gamma,
                        'theta': call_price.greeks.theta,
                        'vega': call_price.greeks.vega,
                        'rho': call_price.greeks.rho,
                        'stability_score': call_price.greeks.stability_score
                    },
                    'volume': market_data['volume'],
                    'open_interest': market_data['open_interest']
                }
                
                put_options[strike] = {
                    'price': put_price.theoretical_price,
                    'market_price': put_price.market_price,
                    'bid': put_price.bid_price,
                    'ask': put_price.ask_price,
                    'mid': put_price.mid_price,
                    'spread': put_price.spread,
                    'implied_volatility': put_price.implied_volatility,
                    'greeks': {
                        'delta': put_price.greeks.delta,
                        'gamma': put_price.greeks.gamma,
                        'theta': put_price.greeks.theta,
                        'vega': put_price.greeks.vega,
                        'rho': put_price.greeks.rho,
                        'stability_score': put_price.greeks.stability_score
                    },
                    'volume': market_data['volume'],
                    'open_interest': market_data['open_interest']
                }
            
            return {
                'symbol': symbol,
                'underlying_price': underlying_price,
                'expiry_date': expiry_date.isoformat(),
                'time_to_expiry': time_to_expiry,
                'strike_prices': strike_prices.tolist(),
                'call_options': call_options,
                'put_options': put_options,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Options chain generation failed: {e}")
            return {'error': str(e)}
    
    def _generate_realistic_market_data(self, underlying_price: float, strike_price: float,
                                      time_to_expiry: float) -> Dict[str, float]:
        """Generate realistic market data for options"""
        try:
            # Base volatility
            base_vol = 0.25  # 25% base volatility
            
            # Volatility smile (higher IV for OTM options)
            moneyness = underlying_price / strike_price
            if moneyness < 0.95 or moneyness > 1.05:  # OTM
                vol_adjustment = 1.2
            else:  # ATM
                vol_adjustment = 1.0
            
            implied_vol = base_vol * vol_adjustment
            
            # Calculate theoretical prices
            call_price = self.calculate_black_scholes_price(
                underlying_price, strike_price, time_to_expiry, implied_vol, OptionType.CALL
            )
            put_price = self.calculate_black_scholes_price(
                underlying_price, strike_price, time_to_expiry, implied_vol, OptionType.PUT
            )
            
            # Add realistic spreads (1-5% of price)
            call_spread = call_price * np.random.uniform(0.01, 0.05)
            put_spread = put_price * np.random.uniform(0.01, 0.05)
            
            return {
                'call_market_price': call_price,
                'call_bid': call_price - call_spread / 2,
                'call_ask': call_price + call_spread / 2,
                'put_market_price': put_price,
                'put_bid': put_price - put_spread / 2,
                'put_ask': put_price + put_spread / 2,
                'volume': np.random.randint(100, 1000),
                'open_interest': np.random.randint(1000, 10000)
            }
            
        except Exception as e:
            logger.error(f"❌ Market data generation failed: {e}")
            return {}

# Global options pricer instance
options_pricer = EnhancedOptionsPricer()

# Convenience functions
def calculate_black_scholes_price(underlying_price: float, strike_price: float,
                                time_to_expiry: float, volatility: float,
                                option_type: OptionType, risk_free_rate: float = None) -> float:
    """Calculate Black-Scholes option price"""
    return options_pricer.calculate_black_scholes_price(
        underlying_price, strike_price, time_to_expiry, volatility, option_type, risk_free_rate
    )

def calculate_implied_volatility(market_price: float, underlying_price: float,
                               strike_price: float, time_to_expiry: float,
                               option_type: OptionType, risk_free_rate: float = None) -> float:
    """Calculate implied volatility from market price"""
    return options_pricer.calculate_implied_volatility(
        market_price, underlying_price, strike_price, time_to_expiry, option_type, risk_free_rate
    )

def calculate_analytic_greeks(underlying_price: float, strike_price: float,
                            time_to_expiry: float, volatility: float,
                            option_type: OptionType, risk_free_rate: float = None) -> OptionGreeks:
    """Calculate analytic Greeks"""
    return options_pricer.calculate_analytic_greeks(
        underlying_price, strike_price, time_to_expiry, volatility, option_type, risk_free_rate
    )

def generate_options_chain_with_iv(symbol: str, underlying_price: float,
                                 expiry_date: datetime) -> Dict[str, Any]:
    """Generate options chain with market-implied volatility"""
    return options_pricer.generate_options_chain_with_iv(symbol, underlying_price, expiry_date)
