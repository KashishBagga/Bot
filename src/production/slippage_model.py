#!/usr/bin/env python3
"""
Slippage & Partial Fills Simulation
MUST #5: Realistic slippage and partial fill simulation for backtests
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
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SlippageConfig:
    """Slippage configuration"""
    base_slippage: float = 0.0005  # 0.05% base slippage
    volatility_multiplier: float = 2.0  # Multiply by volatility
    volume_impact: float = 0.1  # Volume impact factor
    time_impact: float = 0.05  # Time of day impact
    market_impact: float = 0.2  # Market condition impact

@dataclass
class PartialFillConfig:
    """Partial fill configuration"""
    base_fill_rate: float = 0.95  # 95% base fill rate
    volatility_impact: float = -0.1  # High volatility reduces fill rate
    volume_impact: float = -0.05  # Large orders reduce fill rate
    time_impact: float = 0.1  # Market open/close impact
    market_impact: float = -0.15  # Bad market conditions reduce fill rate

@dataclass
class FillResult:
    """Fill result"""
    filled_quantity: float
    average_price: float
    total_slippage: float
    fill_rate: float
    partial_fills: List[Dict[str, Any]]
    execution_time: float

class SlippageModel:
    """Realistic slippage and partial fill model"""
    
    def __init__(self, slippage_config: SlippageConfig, partial_fill_config: PartialFillConfig):
        self.slippage_config = slippage_config
        self.partial_fill_config = partial_fill_config
        self.slippage_history = []
        self.fill_history = []
        
    def simulate_order_execution(self, symbol: str, side: str, quantity: float, 
                               limit_price: float, market_data: Dict[str, Any]) -> FillResult:
        """Simulate realistic order execution with slippage and partial fills"""
        try:
            start_time = time.time()
            
            # Calculate slippage
            slippage = self._calculate_slippage(symbol, quantity, market_data)
            
            # Calculate fill rate
            fill_rate = self._calculate_fill_rate(symbol, quantity, market_data)
            
            # Simulate partial fills
            partial_fills = self._simulate_partial_fills(symbol, side, quantity, limit_price, slippage, fill_rate)
            
            # Calculate final results
            total_filled = sum(fill['quantity'] for fill in partial_fills)
            total_slippage = sum(fill['slippage'] for fill in partial_fills)
            average_price = sum(fill['price'] * fill['quantity'] for fill in partial_fills) / total_filled if total_filled > 0 else limit_price
            
            execution_time = time.time() - start_time
            
            result = FillResult(
                filled_quantity=total_filled,
                average_price=average_price,
                total_slippage=total_slippage,
                fill_rate=total_filled / quantity if quantity > 0 else 0,
                partial_fills=partial_fills,
                execution_time=execution_time
            )
            
            # Store for analysis
            self.slippage_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'slippage': slippage,
                'quantity': quantity,
                'volatility': market_data.get('volatility', 0.02)
            })
            
            self.fill_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'fill_rate': fill_rate,
                'quantity': quantity,
                'market_condition': market_data.get('market_condition', 'NORMAL')
            })
            
            logger.info(f"📊 Order execution simulated: {symbol} {side} {quantity} -> {total_filled} filled @ {average_price:.2f} (slippage: {total_slippage:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Order execution simulation failed: {e}")
            raise
    
    def _calculate_slippage(self, symbol: str, quantity: float, market_data: Dict[str, Any]) -> float:
        """Calculate realistic slippage based on market conditions"""
        try:
            # Base slippage
            slippage = self.slippage_config.base_slippage
            
            # Volatility impact
            volatility = market_data.get('volatility', 0.02)
            slippage += volatility * self.slippage_config.volatility_multiplier
            
            # Volume impact (larger orders = more slippage)
            volume_impact = min(quantity / 10000, 1.0) * self.slippage_config.volume_impact
            slippage += volume_impact
            
            # Time of day impact
            current_hour = datetime.now().hour
            if current_hour in [9, 10, 15, 16]:  # Market open/close
                slippage += self.slippage_config.time_impact
            elif current_hour in [12, 13]:  # Lunch time
                slippage += self.slippage_config.time_impact * 0.5
            
            # Market condition impact
            market_condition = market_data.get('market_condition', 'NORMAL')
            if market_condition == 'VOLATILE':
                slippage += self.slippage_config.market_impact
            elif market_condition == 'STRESSED':
                slippage += self.slippage_config.market_impact * 2
            
            # Symbol-specific adjustments
            if 'NIFTY' in symbol:
                slippage *= 0.8  # Index futures have lower slippage
            elif 'BANK' in symbol:
                slippage *= 1.2  # Bank stocks have higher slippage
            
            # Add some randomness
            slippage *= np.random.uniform(0.8, 1.2)
            
            return max(0.0, slippage)
            
        except Exception as e:
            logger.error(f"❌ Slippage calculation failed: {e}")
            return self.slippage_config.base_slippage
    
    def _calculate_fill_rate(self, symbol: str, quantity: float, market_data: Dict[str, Any]) -> float:
        """Calculate realistic fill rate based on market conditions"""
        try:
            # Base fill rate
            fill_rate = self.partial_fill_config.base_fill_rate
            
            # Volatility impact
            volatility = market_data.get('volatility', 0.02)
            fill_rate += volatility * self.partial_fill_config.volatility_impact
            
            # Volume impact (larger orders = lower fill rate)
            volume_impact = min(quantity / 10000, 1.0) * self.partial_fill_config.volume_impact
            fill_rate += volume_impact
            
            # Time of day impact
            current_hour = datetime.now().hour
            if current_hour in [9, 10, 15, 16]:  # Market open/close
                fill_rate += self.partial_fill_config.time_impact
            elif current_hour in [12, 13]:  # Lunch time
                fill_rate += self.partial_fill_config.time_impact * 0.5
            
            # Market condition impact
            market_condition = market_data.get('market_condition', 'NORMAL')
            if market_condition == 'VOLATILE':
                fill_rate += self.partial_fill_config.market_impact
            elif market_condition == 'STRESSED':
                fill_rate += self.partial_fill_config.market_impact * 2
            
            # Symbol-specific adjustments
            if 'NIFTY' in symbol:
                fill_rate += 0.05  # Index futures have higher fill rate
            elif 'BANK' in symbol:
                fill_rate -= 0.05  # Bank stocks have lower fill rate
            
            # Add some randomness
            fill_rate *= np.random.uniform(0.9, 1.1)
            
            return max(0.0, min(1.0, fill_rate))
            
        except Exception as e:
            logger.error(f"❌ Fill rate calculation failed: {e}")
            return self.partial_fill_config.base_fill_rate
    
    def _simulate_partial_fills(self, symbol: str, side: str, quantity: float, 
                              limit_price: float, slippage: float, fill_rate: float) -> List[Dict[str, Any]]:
        """Simulate partial fills"""
        try:
            partial_fills = []
            remaining_quantity = quantity * fill_rate
            
            # Simulate multiple partial fills
            while remaining_quantity > 0:
                # Calculate fill size (random between 10% and 50% of remaining)
                fill_size = remaining_quantity * np.random.uniform(0.1, 0.5)
                fill_size = min(fill_size, remaining_quantity)
                
                # Calculate fill price with slippage
                if side == 'BUY':
                    fill_price = limit_price * (1 + slippage * np.random.uniform(0.5, 1.5))
                else:
                    fill_price = limit_price * (1 - slippage * np.random.uniform(0.5, 1.5))
                
                # Calculate individual slippage
                individual_slippage = abs(fill_price - limit_price) / limit_price
                
                partial_fills.append({
                    'quantity': fill_size,
                    'price': fill_price,
                    'slippage': individual_slippage,
                    'timestamp': datetime.now(),
                    'fill_number': len(partial_fills) + 1
                })
                
                remaining_quantity -= fill_size
                
                # Add small delay between fills
                time.sleep(0.001)
            
            return partial_fills
            
        except Exception as e:
            logger.error(f"❌ Partial fill simulation failed: {e}")
            return []
    
    def calibrate_from_live_data(self, live_data: List[Dict[str, Any]]):
        """Calibrate slippage model from live trading data"""
        try:
            if not live_data:
                logger.warning("⚠️ No live data available for calibration")
                return
            
            # Analyze slippage patterns
            slippages = [d['slippage'] for d in live_data if 'slippage' in d]
            if slippages:
                median_slippage = np.median(slippages)
                std_slippage = np.std(slippages)
                
                # Update base slippage
                self.slippage_config.base_slippage = median_slippage
                
                logger.info(f"📊 Slippage model calibrated: median={median_slippage:.4f}, std={std_slippage:.4f}")
            
            # Analyze fill rate patterns
            fill_rates = [d['fill_rate'] for d in live_data if 'fill_rate' in d]
            if fill_rates:
                median_fill_rate = np.median(fill_rates)
                
                # Update base fill rate
                self.partial_fill_config.base_fill_rate = median_fill_rate
                
                logger.info(f"📊 Fill rate model calibrated: median={median_fill_rate:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Model calibration failed: {e}")
    
    def get_slippage_statistics(self) -> Dict[str, Any]:
        """Get slippage statistics"""
        try:
            if not self.slippage_history:
                return {}
            
            slippages = [h['slippage'] for h in self.slippage_history]
            
            return {
                'total_orders': len(slippages),
                'median_slippage': np.median(slippages),
                'mean_slippage': np.mean(slippages),
                'std_slippage': np.std(slippages),
                'min_slippage': np.min(slippages),
                'max_slippage': np.max(slippages),
                'slippage_by_symbol': self._get_slippage_by_symbol(),
                'slippage_by_volatility': self._get_slippage_by_volatility()
            }
            
        except Exception as e:
            logger.error(f"❌ Slippage statistics failed: {e}")
            return {}
    
    def _get_slippage_by_symbol(self) -> Dict[str, float]:
        """Get slippage statistics by symbol"""
        symbol_slippages = {}
        
        for record in self.slippage_history:
            symbol = record['symbol']
            if symbol not in symbol_slippages:
                symbol_slippages[symbol] = []
            symbol_slippages[symbol].append(record['slippage'])
        
        return {symbol: np.median(slippages) for symbol, slippages in symbol_slippages.items()}
    
    def _get_slippage_by_volatility(self) -> Dict[str, float]:
        """Get slippage statistics by volatility level"""
        volatility_slippages = {'LOW': [], 'MEDIUM': [], 'HIGH': []}
        
        for record in self.slippage_history:
            volatility = record['volatility']
            if volatility < 0.015:
                volatility_slippages['LOW'].append(record['slippage'])
            elif volatility < 0.025:
                volatility_slippages['MEDIUM'].append(record['slippage'])
            else:
                volatility_slippages['HIGH'].append(record['slippage'])
        
        return {level: np.median(slippages) if slippages else 0.0 
                for level, slippages in volatility_slippages.items()}
    
    def get_fill_rate_statistics(self) -> Dict[str, Any]:
        """Get fill rate statistics"""
        try:
            if not self.fill_history:
                return {}
            
            fill_rates = [h['fill_rate'] for h in self.fill_history]
            
            return {
                'total_orders': len(fill_rates),
                'median_fill_rate': np.median(fill_rates),
                'mean_fill_rate': np.mean(fill_rates),
                'std_fill_rate': np.std(fill_rates),
                'min_fill_rate': np.min(fill_rates),
                'max_fill_rate': np.max(fill_rates),
                'fill_rate_by_symbol': self._get_fill_rate_by_symbol(),
                'fill_rate_by_market_condition': self._get_fill_rate_by_market_condition()
            }
            
        except Exception as e:
            logger.error(f"❌ Fill rate statistics failed: {e}")
            return {}
    
    def _get_fill_rate_by_symbol(self) -> Dict[str, float]:
        """Get fill rate statistics by symbol"""
        symbol_fill_rates = {}
        
        for record in self.fill_history:
            symbol = record['symbol']
            if symbol not in symbol_fill_rates:
                symbol_fill_rates[symbol] = []
            symbol_fill_rates[symbol].append(record['fill_rate'])
        
        return {symbol: np.median(fill_rates) for symbol, fill_rates in symbol_fill_rates.items()}
    
    def _get_fill_rate_by_market_condition(self) -> Dict[str, float]:
        """Get fill rate statistics by market condition"""
        condition_fill_rates = {}
        
        for record in self.fill_history:
            condition = record['market_condition']
            if condition not in condition_fill_rates:
                condition_fill_rates[condition] = []
            condition_fill_rates[condition].append(record['fill_rate'])
        
        return {condition: np.median(fill_rates) for condition, fill_rates in condition_fill_rates.items()}

def main():
    """Main function for testing"""
    slippage_config = SlippageConfig(
        base_slippage=0.0005,
        volatility_multiplier=2.0,
        volume_impact=0.1,
        time_impact=0.05,
        market_impact=0.2
    )
    
    partial_fill_config = PartialFillConfig(
        base_fill_rate=0.95,
        volatility_impact=-0.1,
        volume_impact=-0.05,
        time_impact=0.1,
        market_impact=-0.15
    )
    
    slippage_model = SlippageModel(slippage_config, partial_fill_config)
    
    # Test order execution simulation
    market_data = {
        'volatility': 0.02,
        'market_condition': 'NORMAL',
        'volume': 1000
    }
    
    result = slippage_model.simulate_order_execution(
        symbol="NSE:NIFTY50-INDEX",
        side="BUY",
        quantity=100,
        limit_price=19500,
        market_data=market_data
    )
    
    print(f"✅ Order execution result:")
    print(f"  Filled quantity: {result.filled_quantity}")
    print(f"  Average price: {result.average_price:.2f}")
    print(f"  Total slippage: {result.total_slippage:.4f}")
    print(f"  Fill rate: {result.fill_rate:.2%}")
    print(f"  Partial fills: {len(result.partial_fills)}")
    
    # Get statistics
    slippage_stats = slippage_model.get_slippage_statistics()
    fill_rate_stats = slippage_model.get_fill_rate_statistics()
    
    print(f"\n📊 Slippage statistics: {slippage_stats}")
    print(f"�� Fill rate statistics: {fill_rate_stats}")

if __name__ == "__main__":
    main()
