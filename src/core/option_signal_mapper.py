#!/usr/bin/env python3
"""
Option Signal Mapper
Convert index signals to option contracts for execution
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

from src.models.option_contract import OptionContract, OptionChain, OptionType, StrikeSelection
from src.data.option_chain_loader import OptionChainLoader

logger = logging.getLogger(__name__)

class OptionSignalMapper:
    """Map index signals to option contracts for execution."""
    
    def __init__(self, option_loader: OptionChainLoader):
        self.option_loader = option_loader
        
        # Default mapping parameters
        self.default_expiry_type = "weekly"  # weekly or monthly
        self.default_strike_selection = StrikeSelection.ATM
        self.default_delta_target = 0.30  # For delta-based selection
        self.min_liquidity_oi = 100  # Minimum open interest
        self.min_liquidity_volume = 10  # Minimum volume
        
        logger.info("🎯 Option Signal Mapper initialized")
    
    def map_signal_to_option(self, signal: Dict, underlying_price: float, 
                           timestamp: datetime, option_chain: OptionChain = None) -> Optional[Dict]:
        """
        Map a spot signal to an option contract.
        
        Args:
            signal: Original signal dictionary
            underlying_price: Current underlying price
            timestamp: Current timestamp
            option_chain: Option chain (if None, will be loaded/simulated)
            
        Returns:
            Option order specification or None
        """
        try:
            # Extract signal details
            signal_type = signal.get('signal', '')
            underlying = signal.get('symbol', '')
            confidence = signal.get('confidence', 0)
            
            # Validate signal
            if signal_type not in ['BUY CALL', 'BUY PUT']:
                logger.warning(f"⚠️ Unsupported signal type: {signal_type}")
                return None
            
            # Load option chain if not provided
            if option_chain is None:
                option_chain = self.option_loader.load_option_chain(underlying, timestamp)
                
                # If no historical data, simulate
                if option_chain is None or not option_chain.contracts:
                    logger.info(f"📊 Simulating option chain for {underlying}")
                    option_chain = self.option_loader.simulate_option_chain(
                        underlying, underlying_price, timestamp
                    )
            
            if not option_chain or not option_chain.contracts:
                logger.warning(f"⚠️ No option contracts available for {underlying}")
                return None
            
            # Get nearest expiry
            expiry = self._get_nearest_expiry(option_chain, timestamp)
            if expiry is None:
                logger.warning(f"⚠️ No valid expiry found for {underlying}")
                return None
            
            # Select option type
            option_type = OptionType.CALL if signal_type == 'BUY CALL' else OptionType.PUT
            
            # Select contract based on strike selection method
            contract = self._select_contract(
                option_chain, option_type, expiry, underlying_price
            )
            
            if contract is None:
                logger.warning(f"⚠️ No suitable contract found for {signal_type}")
                return None
            
            # Determine realistic entry price (prefer ask → mid → last)
            ask = getattr(contract, 'ask', None) or 0.0
            bid = getattr(contract, 'bid', None) or 0.0
            last = getattr(contract, 'last', None) or getattr(contract, 'ltp', None) or 0.0
            
            entry_px = 0.0
            if ask and ask > 0:
                entry_px = float(ask)
            elif bid and bid > 0:
                entry_px = float((bid + ask) / 2.0 if ask and ask > 0 else bid)
            elif last and last > 0:
                entry_px = float(last)
            else:
                logger.warning(f"⚠️ No valid price (ask/bid/last) for contract {contract.symbol}, skipping")
                return None
            
            # Calculate position size based on premium risk
            position_size = self._calculate_option_position_size(
                contract, signal, underlying_price, entry_px
            )
            
            if position_size <= 0:
                logger.warning(f"⚠️ Invalid position size calculated: {position_size}")
                return None
            
            # Create option order specification
            order_spec = {
                'contract': contract,
                'quantity': position_size,  # Number of lots
                'entry_price': entry_px,  # Realistic entry price
                'signal_type': signal_type,
                'underlying_price': underlying_price,
                'confidence': confidence,
                'strategy': signal.get('strategy', ''),
                'reasoning': signal.get('reasoning', ''),
                'timestamp': timestamp,
                'expiry': expiry,
                'strike': contract.strike,
                'option_type': contract.option_type.value,
                'lot_size': contract.lot_size,
                'premium_risk': entry_px * position_size * contract.lot_size,
                'delta': contract.delta,
                'gamma': contract.gamma,
                'theta': contract.theta,
                'vega': contract.vega
            }
            
            logger.info(f"✅ Mapped {signal_type} to {contract.symbol} "
                       f"(Strike: {contract.strike}, Expiry: {expiry.strftime('%Y-%m-%d')}, "
                       f"Lots: {position_size}, Premium: ₹{contract.ask:.2f})")
            
            return order_spec
            
        except Exception as e:
            logger.error(f"❌ Error mapping signal to option: {e}")
            return None
    
    def _get_nearest_expiry(self, option_chain: OptionChain, timestamp: datetime) -> Optional[datetime]:
        """Get nearest expiry date (deterministic and fallback-friendly)."""
        try:
            # Get unique expiry dates (sorted)
            expiries = sorted({c.expiry for c in option_chain.contracts if c.expiry is not None})
            if not expiries:
                return None
            
            # Filter by expiry type
            if self.default_expiry_type == "weekly":
                expiries_filtered = [e for e in expiries if e.weekday() == 3]  # Thursday
                if not expiries_filtered:
                    expiries_filtered = expiries  # Fallback to all expiries
            elif self.default_expiry_type == "monthly":
                monthly = []
                for e in expiries:
                    if e.date() == self._get_last_thursday_of_month(e.year, e.month):
                        monthly.append(e)
                expiries_filtered = monthly if monthly else expiries  # Fallback to all expiries
            else:
                expiries_filtered = expiries
            
            # Nearest expiry strictly after timestamp
            for exp in expiries_filtered:
                if exp > timestamp:
                    return exp
            
            # Fallback: pick the last future expiry even if not matching filters
            future = [e for e in expiries if e > timestamp]
            return future[0] if future else None
            
        except Exception as e:
            logger.error(f"❌ Error getting nearest expiry: {e}")
            return None
    
    def _get_last_thursday_of_month(self, year: int, month: int) -> datetime.date:
        """Get last Thursday of the month."""
        # Get last day of month
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
        
        # Find last Thursday
        while last_day.weekday() != 3:  # Thursday
            last_day -= timedelta(days=1)
        
        return last_day.date()
    
    def _select_contract(self, option_chain: OptionChain, option_type: OptionType, 
                        expiry: datetime, underlying_price: float) -> Optional[OptionContract]:
        """Select the best contract based on strike selection method."""
        try:
            # Get contracts for the specified type and expiry
            contracts = [c for c in option_chain.contracts 
                        if c.option_type == option_type and c.expiry == expiry]
            
            if not contracts:
                return None
            
            # Enhanced liquidity filtering with relative thresholds
            by_strike = sorted(contracts, key=lambda c: abs(c.strike - underlying_price))
            candidates = by_strike[:20]  # Nearest 20 strikes
            
            # Choose those with OI/volume above percentile
            oi_vals = [c.open_interest for c in candidates if c.open_interest is not None and c.open_interest > 0]
            
            if oi_vals:
                # Use 40th percentile as threshold
                threshold = np.percentile(oi_vals, 40)
                liquid_contracts = [c for c in candidates if (c.open_interest or 0) >= threshold]
                
                if not liquid_contracts:
                    logger.warning(f"⚠️ No contracts above OI threshold {threshold}, using all candidates")
                    liquid_contracts = candidates
            else:
                # Fallback to basic filtering
                liquid_contracts = [c for c in contracts 
                                  if c.open_interest >= self.min_liquidity_oi 
                                  and c.volume >= self.min_liquidity_volume]
                
                if not liquid_contracts:
                    logger.warning(f"⚠️ No liquid contracts found, using all contracts")
                    liquid_contracts = contracts
            
            # Select based on strike selection method
            if self.default_strike_selection == StrikeSelection.ATM:
                return self._select_atm_contract(liquid_contracts, underlying_price)
            elif self.default_strike_selection == StrikeSelection.DELTA:
                return self._select_delta_contract(liquid_contracts, self.default_delta_target)
            elif self.default_strike_selection == StrikeSelection.OTM:
                return self._select_otm_contract(liquid_contracts, underlying_price)
            elif self.default_strike_selection == StrikeSelection.ITM:
                return self._select_itm_contract(liquid_contracts, underlying_price)
            else:
                return self._select_atm_contract(liquid_contracts, underlying_price)
                
        except Exception as e:
            logger.error(f"❌ Error selecting contract: {e}")
            return None
    
    def _select_atm_contract(self, contracts: List[OptionContract], 
                           underlying_price: float) -> Optional[OptionContract]:
        """Select at-the-money contract."""
        if not contracts:
            return None
        
        # Find contract closest to underlying price
        best_contract = None
        min_distance = float('inf')
        
        for contract in contracts:
            distance = abs(contract.strike - underlying_price)
            if distance < min_distance:
                min_distance = distance
                best_contract = contract
        
        return best_contract
    
    def _select_delta_contract(self, contracts: List[OptionContract], 
                             target_delta: float) -> Optional[OptionContract]:
        """Select contract closest to target delta (handle missing deltas)."""
        if not contracts:
            return None
        
        # Find contract with delta closest to target
        best_contract = None
        min_delta_diff = float('inf')
        
        for contract in contracts:
            delta = getattr(contract, 'delta', None)
            if delta is None:
                continue  # Skip contracts with missing delta
            
            delta_diff = abs(abs(delta) - target_delta)
            if delta_diff < min_delta_diff:
                min_delta_diff = delta_diff
                best_contract = contract
        
        return best_contract
    
    def _select_otm_contract(self, contracts: List[OptionContract], 
                           underlying_price: float) -> Optional[OptionContract]:
        """Select out-of-the-money contract."""
        if not contracts:
            return None
        
        # For calls: strike > underlying_price
        # For puts: strike < underlying_price
        otm_contracts = []
        
        for contract in contracts:
            if contract.option_type == OptionType.CALL:
                if contract.strike > underlying_price:
                    otm_contracts.append(contract)
            else:  # PUT
                if contract.strike < underlying_price:
                    otm_contracts.append(contract)
        
        if not otm_contracts:
            return self._select_atm_contract(contracts, underlying_price)
        
        # Select closest OTM contract
        return self._select_atm_contract(otm_contracts, underlying_price)
    
    def _select_itm_contract(self, contracts: List[OptionContract], 
                           underlying_price: float) -> Optional[OptionContract]:
        """Select in-the-money contract."""
        if not contracts:
            return None
        
        # For calls: strike < underlying_price
        # For puts: strike > underlying_price
        itm_contracts = []
        
        for contract in contracts:
            if contract.option_type == OptionType.CALL:
                if contract.strike < underlying_price:
                    itm_contracts.append(contract)
            else:  # PUT
                if contract.strike > underlying_price:
                    itm_contracts.append(contract)
        
        if not itm_contracts:
            return self._select_atm_contract(contracts, underlying_price)
        
        # Select closest ITM contract
        return self._select_atm_contract(itm_contracts, underlying_price)
    
    def _calculate_option_position_size(self, contract: OptionContract, 
                                      signal: Dict, underlying_price: float, entry_price: float) -> int:
        """Calculate position size based on premium risk with caps."""
        try:
            # Get risk parameters from signal or use defaults
            max_risk_per_trade = signal.get('max_risk_per_trade', 0.02)  # 2% default
            capital = signal.get('capital', 100000)  # Default capital
            confidence = signal.get('confidence', 50)
            
            # Calculate maximum risk amount
            max_risk_amount = capital * max_risk_per_trade
            
            # Adjust risk based on confidence
            confidence_multiplier = min(max(confidence / 50.0, 0.5), 1.5)
            adjusted_risk = max_risk_amount * confidence_multiplier
            
            # For buying options, risk = premium paid
            premium_per_lot = entry_price * contract.lot_size
            
            if premium_per_lot <= 0:
                return 0
            
            # Calculate maximum lots based on risk
            max_lots = int(adjusted_risk / premium_per_lot)
            
            # Cap by available capital
            available_capital = capital
            max_affordable_lots = int(available_capital // premium_per_lot)
            max_lots = min(max_lots, max_affordable_lots)
            
            # Apply per-contract caps
            per_contract_max_lots = int(signal.get('max_lots_per_contract', 100))
            max_lots = min(max_lots, per_contract_max_lots)
            
            # Ensure minimum 1 lot if we can afford it
            if max_lots < 1:
                if adjusted_risk >= premium_per_lot:
                    return 1
                else:
                    return 0
            
            return max_lots
            
        except Exception as e:
            logger.error(f"❌ Error calculating option position size: {e}")
            return 0
    
    def map_multiple_signals(self, signals: List[Dict], underlying_price: float,
                           timestamp: datetime, option_chain: OptionChain = None) -> List[Dict]:
        """Map multiple signals to option contracts (batch-friendly)."""
        option_signals = []
        
        # Load chain once for all signals
        if option_chain is None and signals:
            underlying = signals[0].get('symbol')
            if underlying:
                option_chain = self.option_loader.load_option_chain(underlying, timestamp)
                if option_chain is None or not option_chain.contracts:
                    option_chain = self.option_loader.simulate_option_chain(underlying, underlying_price, timestamp)
        
        for signal in signals:
            option_signal = self.map_signal_to_option(signal, underlying_price, timestamp, option_chain)
            if option_signal:
                option_signals.append(option_signal)
        
        # Dedupe by contract symbol - keep highest confidence
        deduped = {}
        for sig in option_signals:
            key = sig['contract'].symbol
            if key not in deduped or sig.get('confidence', 0) > deduped[key].get('confidence', 0):
                deduped[key] = sig
        
        return list(deduped.values())
    
    def set_parameters(self, expiry_type: str = None, strike_selection: StrikeSelection = None,
                      delta_target: float = None, min_oi: int = None, min_volume: int = None):
        """Update mapping parameters."""
        if expiry_type:
            self.default_expiry_type = expiry_type
        if strike_selection:
            self.default_strike_selection = strike_selection
        if delta_target:
            self.default_delta_target = delta_target
        if min_oi:
            self.min_liquidity_oi = min_oi
        if min_volume:
            self.min_liquidity_volume = min_volume
        
        logger.info(f"⚙️ Updated mapping parameters: "
                   f"expiry={self.default_expiry_type}, "
                   f"strike={self.default_strike_selection.value}, "
                   f"delta={self.default_delta_target}") 