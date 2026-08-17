#!/usr/bin/env python3
"""
Transparent Parameter Backtester (Phase 2 — real pipeline)
============================================================
Replays historical candles through the SAME MarketSnapshot -> IndicatorPipeline
-> ExperimentRegistry pipeline the live trader (src/trading/indian_trader.py)
uses, via the shared experiment set in src/core/experiment_factory.py. Every
registered single-leg experiment gets backtested, not just the frozen
EnhancedStrategyEngine core — see CLAUDE.md's Strategy Research Framework
section for why that distinction matters.

Known scope limits (disclosed, not silently dropped):
- Multi-leg combo signals (Vertical Spread, Straddle/Strangle, Iron Condor,
  Butterfly, Iron Butterfly, Credit Spread, Expiry-Aware Theta — 8 of the
  ~26 registered experiments) are SKIPPED here. They need a combined-premium
  multi-leg PnL engine (see indian_trader.py's _handle_combo_signal /
  _enter_combo_position), which this backtester does not simulate. Skipped
  count is logged, never silently absorbed into "0 signals".
- Options-dependent single-leg strategies (OIWallReaction, PCRExtremeReversal,
  OI_Scalping) still get evaluated, but IndicatorPipeline is constructed with
  BacktestDBStub (see backtest_db_stub.py) so they correctly see "no options
  data" rather than accidentally reading today's live option chain against a
  replayed historical candle.
- Trade outcomes are a step-forward walk against future index-point bars
  (stop/target crossing), not the live position engine's real fill/premium
  simulation (_update_position/_premium_pnl_r) — this stays an index-point
  proxy, deliberately, to avoid inheriting execution-realism assumptions this
  backtester can't independently verify without historical option data.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.adapters.market_factory import MarketFactory
from src.adapters.market_interface import MarketType
from src.core.indicator_pipeline import IndicatorPipeline
from src.core.experiment_factory import build_registry
from src.backtesting.backtest_db_stub import BacktestDBStub

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


def _aggregate(trade_log: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trade_log:
        return {'expectancy': 0.0, 'win_rate': 0.0, 'total_r': 0.0, 'trades': 0}
    wins = [t for t in trade_log if t['pnl_r'] > 0]
    total_r = sum(t['pnl_r'] for t in trade_log)
    return {
        'expectancy': total_r / len(trade_log),
        'win_rate': len(wins) / len(trade_log),
        'total_r': total_r,
        'trades': len(trade_log),
    }


class TransparentBacktester:
    def __init__(self, symbols: List[str], days: int = 30):
        self.symbols = symbols
        self.days = days
        self.run_id = f"bt_{run_time}"
        self.market = MarketFactory.create_market(MarketType.INDIAN_STOCKS)
        self.data_provider = self.market.get_data_provider()
        self.historical_data = {}

    def fetch_data(self):
        """Fetch real multi-timeframe data for backtesting. Reads/writes through
        the local candle cache (FyersDataProvider -> postgres_database.candles),
        so repeat runs over the same window replay from Postgres instead of
        re-hitting the Fyers API."""
        logger.info(f"📥 Fetching {self.days} days of MTF data for {len(self.symbols)} symbols...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)

        for symbol in self.symbols:
            d1 = self.data_provider.get_historical_data(symbol, start_date - timedelta(days=100), end_date, "D")
            h1 = self.data_provider.get_historical_data(symbol, start_date, end_date, "60")
            m5 = self.data_provider.get_historical_data(symbol, start_date, end_date, "5")

            if h1 is not None and m5 is not None:
                self.historical_data[symbol] = {'1d': d1, '1h': h1, '5m': m5}
                logger.info(f"✅ Loaded data for {symbol} ({len(m5)} 5m bars)")
            else:
                logger.warning(f"⚠️ Insufficient data for {symbol} — skipping")

    def simulate_trades(self, verbose: bool = False) -> Dict[str, Any]:
        """Replays every symbol's 5m bars through IndicatorPipeline + the full
        registered Experiment set, in true chronological order ACROSS symbols
        (not one symbol fully then the next) — RelativeValueStrategy needs
        both NIFTY and BANKNIFTY snapshots interleaved in real time order to
        correlate correctly, same as it sees them live.
        """
        pipeline = IndicatorPipeline(
            pivot_window=3,
            zone_cluster_pct=0.002,
            min_zone_score=50.0,
            db=BacktestDBStub(),
        )
        registry = build_registry()

        m5_by_symbol = {s: b['5m'] for s, b in self.historical_data.items()}
        h1_by_symbol = {s: b['1h'] for s, b in self.historical_data.items()}
        d1_by_symbol = {s: b['1d'] for s, b in self.historical_data.items()}

        start_i = max(50, pipeline.required_history)
        pointer = {s: start_i for s in m5_by_symbol}

        # Union of every symbol's own bar timestamps, walked in order, so each
        # symbol only advances on ITS OWN bar — this is what interleaves
        # symbols in true chronological order without requiring their 5m
        # indices to be identically aligned.
        all_times = sorted(set().union(*[df.index[start_i:] for df in m5_by_symbol.values() if len(df) > start_i]))

        trade_log: List[Dict[str, Any]] = []
        blocked_until: Dict[Tuple[str, str], Any] = {}
        combo_skipped = 0
        eval_errors = 0

        for current_time in all_times:
            for symbol, m5_df in m5_by_symbol.items():
                i = pointer[symbol]
                if i >= len(m5_df) or m5_df.index[i] != current_time:
                    continue
                pointer[symbol] = i + 1

                h1_df = h1_by_symbol[symbol]
                d1_df = d1_by_symbol[symbol]
                # Leakage-safe windowing: HTF data only up to (not including)
                # this candle; m5 includes this candle as the last CLOSED bar.
                h1_window = h1_df[h1_df.index < current_time]
                d1_window = d1_df[d1_df.index < current_time] if d1_df is not None else None
                m5_window = m5_df.iloc[:i + 1]

                current_price = float(m5_window['close'].iloc[-1])

                snapshot = pipeline.compute(symbol, current_price, d1_window, h1_window, m5_window, current_time)
                if snapshot is None:
                    continue

                results = registry.run(snapshot)
                for result in results:
                    if result.errors:
                        eval_errors += len(result.errors)
                    experiment_name = result.experiment_name
                    for sig in result.accepted_signals:
                        if 'combo_legs' in sig:
                            combo_skipped += 1
                            continue

                        key = (symbol, experiment_name)
                        if key in blocked_until and current_time < blocked_until[key]:
                            continue

                        signal_type = sig.get('signal')
                        entry_price = sig.get('price')
                        sl_price = sig.get('stop_loss')
                        tp_price = sig.get('take_profit')
                        if signal_type not in ('BUY CALL', 'BUY PUT') or not entry_price or not sl_price or not tp_price:
                            continue
                        tp1_price = sig.get('tp1', tp_price)
                        risk = abs(entry_price - sl_price)
                        rr_ratio = sig.get('rr_ratio') or (abs(tp_price - entry_price) / risk if risk > 0 else 1.0)

                        outcome_r = 0.0
                        exit_reason = "EXPIRED"
                        exit_price = 0.0
                        exit_time = None
                        current_sl = sl_price
                        hit_tp1 = False

                        future_data = m5_df.iloc[i + 1: i + 101]
                        for f_idx, f_candle in future_data.iterrows():
                            f_h, f_l = f_candle['high'], f_candle['low']

                            if not hit_tp1:
                                if (signal_type == 'BUY CALL' and f_h >= tp1_price) or \
                                   (signal_type == 'BUY PUT' and f_l <= tp1_price):
                                    hit_tp1 = True
                                    current_sl = entry_price  # Move to break-even

                            if signal_type == 'BUY CALL':
                                if f_l <= current_sl:
                                    outcome_r = -1.0 if not hit_tp1 else 0.75
                                    exit_reason = "STOP_LOSS" if not hit_tp1 else "BREAK_EVEN_PLUS"
                                    exit_price = current_sl; exit_time = f_idx; break
                                elif f_h >= tp_price:
                                    outcome_r = rr_ratio; exit_reason = "STRUCTURAL_TP"
                                    exit_price = tp_price; exit_time = f_idx; break
                            else:  # BUY PUT
                                if f_h >= current_sl:
                                    outcome_r = -1.0 if not hit_tp1 else 0.75
                                    exit_reason = "STOP_LOSS" if not hit_tp1 else "BREAK_EVEN_PLUS"
                                    exit_price = current_sl; exit_time = f_idx; break
                                elif f_l <= tp_price:
                                    outcome_r = rr_ratio; exit_reason = "STRUCTURAL_TP"
                                    exit_price = tp_price; exit_time = f_idx; break

                        if exit_reason != "EXPIRED":
                            outcome_r -= 0.05  # Standardized transaction cost (slippage + brokerage)
                            blocked_until[key] = exit_time

                            trade_detail = {
                                'time': current_time.strftime("%Y-%m-%d %H:%M"),
                                'entry_time': current_time,
                                'exit_time': exit_time,
                                'symbol': symbol,
                                'experiment': experiment_name,
                                'signal': signal_type,
                                'reason': sig.get('strategy', experiment_name),
                                'entry': round(entry_price, 2),
                                'sl_target': round(sl_price, 2),
                                'tp_target': round(tp_price, 2),
                                'exit': round(exit_price, 2),
                                'reason_exit': exit_reason,
                                'pnl_r': round(outcome_r, 2),
                            }
                            trade_log.append(trade_detail)

                            if verbose:
                                logger.info(f"[{trade_detail['time']}] {symbol} [{experiment_name}] {signal_type}")
                                logger.info(f"   Entry: {trade_detail['entry']} | SL: {trade_detail['sl_target']} | TP: {trade_detail['tp_target']}")
                                logger.info(f"   Outcome: {trade_detail['reason_exit']} | PnL: {trade_detail['pnl_r']}R")
                                logger.info("-" * 40)

        if combo_skipped:
            logger.info(f"ℹ️ Skipped {combo_skipped} multi-leg combo signal(s) — not simulated by this backtester (see module docstring)")
        if eval_errors:
            logger.warning(f"⚠️ {eval_errors} experiment evaluation error(s) during replay (see StrategyResult.errors)")

        per_experiment: Dict[str, Dict[str, float]] = {}
        for name in {t['experiment'] for t in trade_log}:
            per_experiment[name] = _aggregate([t for t in trade_log if t['experiment'] == name])

        return {
            'overall': _aggregate(trade_log),
            'per_experiment': per_experiment,
            'trades': trade_log,
            'combo_signals_skipped': combo_skipped,
            'eval_errors': eval_errors,
        }

    def run_full_audit(self):
        self.fetch_data()
        if not self.historical_data:
            logger.error("❌ No historical data loaded for any symbol — aborting")
            return None

        logger.info("\n" + "=" * 60)
        logger.info("🕵️ STRATEGY AUDIT: DETAILED TRADE LOG (full experiment set)")
        logger.info("=" * 60)

        result = self.simulate_trades(verbose=True)
        overall = result['overall']

        logger.info("\n" + "=" * 60)
        logger.info("📊 FINAL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Trades: {overall['trades']}")
        logger.info(f"Win Rate:     {overall['win_rate']*100:.1f}%")
        logger.info(f"Total Return: {overall['total_r']:.2f}R")
        logger.info(f"Expectancy:   {overall['expectancy']:.2f}R per trade")
        logger.info("=" * 60)

        logger.info("\n📈 PER-EXPERIMENT BREAKDOWN")
        logger.info("=" * 60)
        for name, m in sorted(result['per_experiment'].items(), key=lambda kv: kv[1]['total_r'], reverse=True):
            logger.info(f"{name:45s} trades={m['trades']:4d}  win%={m['win_rate']*100:5.1f}  "
                        f"total_r={m['total_r']:8.2f}  exp={m['expectancy']:6.2f}R")
        logger.info("=" * 60)

        self._persist_results(result)
        return result

    def _persist_results(self, result: Dict[str, Any]):
        """Persist this run's config/summary + per-trade rows so backtests are
        comparable across runs instead of living only in a throwaway log file."""
        try:
            from src.models.postgres_database import PostgresDatabase
            db = PostgresDatabase()
            db.save_backtest_run({
                'run_id': self.run_id,
                'created_at': datetime.now(),
                'days': self.days,
                'symbols': self.symbols,
                'overall_trades': result['overall']['trades'],
                'overall_win_rate': result['overall']['win_rate'],
                'overall_total_r': result['overall']['total_r'],
                'overall_expectancy': result['overall']['expectancy'],
                'combo_signals_skipped': result['combo_signals_skipped'],
                'eval_errors': result['eval_errors'],
                'per_experiment': result['per_experiment'],
            })
            db.save_backtest_trades(self.run_id, result['trades'])
            logger.info(f"💾 Persisted run {self.run_id} ({len(result['trades'])} trades) to backtest_runs/backtest_trades")
        except Exception as e:
            logger.warning(f"⚠️ Could not persist backtest results to DB: {e}")


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
