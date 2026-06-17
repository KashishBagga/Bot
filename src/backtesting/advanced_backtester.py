#!/usr/bin/env python3
"""
Transparent Parameter Backtester (Phase 1)
==========================================
Audits every trade with detailed reasoning, expectations, and outcomes.
Uses R-Multiple normalization for pure expectancy research.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Any, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.core.enhanced_strategy_engine import EnhancedStrategyEngine
from src.core.technical_indicators import calculate_all_indicators
from src.adapters.market_factory import MarketFactory
from src.adapters.market_interface import MarketType

# Setup specialized logging for the backtester
logger = logging.getLogger("Backtester")
logger.setLevel(logging.INFO)
logger.handlers = []

# Console Stream Handler
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(sh)

# File Handler per run
os.makedirs("backtest_runs", exist_ok=True)
run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
fh = logging.FileHandler(f"backtest_runs/backtest_run_{run_time}.log")
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

class TransparentBacktester:
    def __init__(self, symbols: List[str], days: int = 30):
        self.symbols = symbols
        self.days = days
        self.market = MarketFactory.create_market(MarketType.INDIAN_STOCKS)
        self.data_provider = self.market.get_data_provider()
        self.historical_data = {}
        
    def fetch_data(self):
        """Fetch real multi-timeframe data for backtesting."""
        logger.info(f"📥 Fetching {self.days} days of MTF data for {len(self.symbols)} symbols...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)
        
        for symbol in self.symbols:
            # Fetch 1h and 5m data
            d1 = self.data_provider.get_historical_data(symbol, start_date - timedelta(days=100), end_date, "1d")
            h1 = self.data_provider.get_historical_data(symbol, start_date, end_date, "1h")
            m5 = self.data_provider.get_historical_data(symbol, start_date, end_date, "5m")
            
            if h1 is not None and m5 is not None:
                self.historical_data[symbol] = {
                    '1d': calculate_all_indicators(d1) if d1 is not None else None,
                    '1h': calculate_all_indicators(h1),
                    '5m': calculate_all_indicators(m5)
                }
                logger.info(f"✅ Loaded data for {symbol}")

    def simulate_trades(self, params: Dict[str, Any], verbose: bool = False) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Simulates trades and returns metrics + a detailed trade log.
        """
        engine = EnhancedStrategyEngine(
            self.symbols, 
            min_zone_score=params.get('min_zone_score', 60.0),
            rvol_threshold=params.get('rvol_threshold', 2.0)
        )
        trade_log = []
        
        for symbol, data_bundle in self.historical_data.items():
            m5_df = data_bundle['5m']
            h1_df = data_bundle['1h']
            d1_df = data_bundle['1d']
            
            last_exit_time = None
            
            # Slide window over 5m data
            for i in range(50, len(m5_df)):
                current_time = m5_df.index[i]
                
                # --- 1. Mandatory Cooldown Check (1 Hour / 12 Candles) ---
                if last_exit_time and (current_time - last_exit_time).total_seconds() < 3600:
                    continue
                
                # Filter HTF data to only what was known at this 5m timestamp
                h1_window = h1_df[h1_df.index < current_time]
                d1_window = d1_df[d1_df.index < current_time] if d1_df is not None else None
                m5_window = m5_df.iloc[:i+1]
                
                current_price = m5_window['close'].iloc[-1]
                
                # Bundle for engine
                mtf_bundle = {
                    symbol: {
                        '1d': d1_window,
                        '1h': h1_window,
                        '5m': m5_window
                    }
                }
                
                signals = engine.generate_signals_for_all_symbols(mtf_bundle, {symbol: current_price})
                
                for sig in signals:
                    # Confidence is now deterministic in the engine
                    # Only execute accepted candidate signals
                    if sig.get('accepted', True):
                        # --- Trade Execution (Structural) ---
                        entry_price = sig['price']
                        sl_price = sig['stop_loss']
                        tp_price = sig['take_profit']
                        tp1_price = sig['tp1']
                        
                        # --- Outcome Simulation (Step-Forward) ---
                        outcome_r = 0.0
                        exit_reason = "EXPIRED"
                        exit_price = 0.0
                        exit_time = None
                        current_sl = sl_price
                        hit_tp1 = False
                        
                        future_data = m5_df.iloc[i+1 : i+101]
                        for f_idx, f_candle in future_data.iterrows():
                            f_h, f_l = f_candle['high'], f_candle['low']
                            
                            # Check for Partial Profit (TP1) -> Move to BE
                            if not hit_tp1:
                                if (sig['signal'] == 'BUY CALL' and f_h >= tp1_price) or \
                                   (sig['signal'] == 'BUY PUT' and f_l <= tp1_price):
                                    hit_tp1 = True
                                    current_sl = entry_price # Move to Break-even
                            
                            # Check for Exit
                            if sig['signal'] == 'BUY CALL':
                                if f_l <= current_sl:
                                    outcome_r = -1.0 if not hit_tp1 else 0.75 # BE + Partial
                                    exit_reason = "STOP_LOSS" if not hit_tp1 else "BREAK_EVEN_PLUS"; exit_price = current_sl; exit_time = f_idx; break
                                elif f_h >= tp_price:
                                    outcome_r = sig['rr_ratio']; exit_reason = "STRUCTURAL_TP"; exit_price = tp_price; exit_time = f_idx; break
                            else: # BUY PUT
                                if f_h >= current_sl:
                                    outcome_r = -1.0 if not hit_tp1 else 0.75
                                    exit_reason = "STOP_LOSS" if not hit_tp1 else "BREAK_EVEN_PLUS"; exit_price = current_sl; exit_time = f_idx; break
                                elif f_l <= tp_price:
                                    outcome_r = sig['rr_ratio']; exit_reason = "STRUCTURAL_TP"; exit_price = tp_price; exit_time = f_idx; break
                        
                        if exit_reason != "EXPIRED":
                            # Standardized Transaction Cost (Slippage + Brokerage) = 0.05R
                            outcome_r -= 0.05
                            last_exit_time = exit_time
                            
                            trade_detail = {
                                'time': current_time.strftime("%Y-%m-%d %H:%M"),
                                'symbol': symbol,
                                'signal': sig['signal'],
                                'reason': sig['strategy'],
                                'entry': round(entry_price, 2),
                                'sl_target': round(sl_price, 2),
                                'tp_target': round(tp_price, 2),
                                'exit': round(exit_price, 2),
                                'reason_exit': exit_reason,
                                'pnl_r': round(outcome_r, 2)
                            }
                            trade_log.append(trade_detail)
                            
                            if verbose:
                                logger.info(f"[{trade_detail['time']}] {symbol} {trade_detail['signal']} | {trade_detail['reason']}")
                                logger.info(f"   Entry: {trade_detail['entry']} | SL: {trade_detail['sl_target']} | TP: {trade_detail['tp_target']}")
                                logger.info(f"   Outcome: {trade_detail['reason_exit']} | PnL: {trade_detail['pnl_r']}R")
                                logger.info("-" * 40)
                                
        # Calculate Aggregated Metrics
        if not trade_log:
            return {'expectancy': 0, 'win_rate': 0, 'total_r': 0, 'trades': 0}, []
            
        wins = [t for t in trade_log if t['pnl_r'] > 0]
        losses = [t for t in trade_log if t['pnl_r'] <= 0]
        total_r = sum(t['pnl_r'] for t in trade_log)
        
        metrics = {
            'expectancy': total_r / len(trade_log),
            'win_rate': len(wins) / len(trade_log),
            'total_r': total_r,
            'trades': len(trade_log)
        }
        
        return metrics, trade_log

    def run_full_audit(self):
        self.fetch_data()
        
        # Test a high-conviction parameter set
        test_params = {
            'confidence_cutoff': 70.0,
            'min_zone_score': 50.0,
            'rvol_threshold': 1.0,
        }
        
        logger.info("\n" + "="*60)
        logger.info("🕵️ STRATEGY AUDIT: DETAILED TRADE LOG")
        logger.info("="*60)
        
        metrics, trades = self.simulate_trades(test_params, verbose=True)
        
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL PERFORMANCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Trades: {metrics['trades']}")
        logger.info(f"Win Rate:     {metrics['win_rate']*100:.1f}%")
        logger.info(f"Total Return: {metrics['total_r']:.2f}R")
        logger.info(f"Expectancy:   {metrics['expectancy']:.2f}R per trade")
        logger.info("="*60)

if __name__ == "__main__":
    import sys
    days = 30
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            pass
    symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"]
    tester = TransparentBacktester(symbols, days=days)
    tester.run_full_audit()
