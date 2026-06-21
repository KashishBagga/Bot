#!/usr/bin/env python3
"""
Institutional Structural Trader (Live Paper Mode)
=================================================
Version: 4.0 (Strategy Research Framework)
- Multi-experiment framework: ExperimentRegistry + IndicatorPipeline
- Single market snapshot per symbol per candle
- Per-experiment independent positions: (symbol, experiment_name)
- Portfolio analytics per experiment (passive observer)
- EnhancedStrategyEngine preserved unchanged inside StructuralStrategy
"""

import os
import sys
import time
import logging
import schedule
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple

# Path Injection
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.adapters.data.fyers_data_provider import FyersDataProvider
from src.models.postgres_database import PostgresDatabase

# Strategy Research Framework
from src.core.indicator_pipeline import IndicatorPipeline
from src.core.experiment import Experiment
from src.core.experiment_registry import ExperimentRegistry
from src.core.portfolio import PortfolioManager
from src.strategies.structural_strategy import StructuralStrategy
from src.strategies.ema_pullback import EmaPullbackStrategy
from src.strategies.vwap_reversion import VwapReversionStrategy
from src.strategies.prev_day_extremes import PrevDayExtremesStrategy
from src.strategies.orb import OrbStrategy
from src.strategies.atr_squeeze import AtrSqueezeStrategy

# Setup Logging
os.makedirs("logs", exist_ok=True)

# Clear any root handlers to prevent double logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

# Console Handler
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(sh)

class DailyRotatingFileHandler(logging.FileHandler):
    def __init__(self, filename_format, mode='a', encoding=None, delay=False):
        self.filename_format = filename_format
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        filename = self.filename_format.format(self.current_date)
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        today = datetime.now().strftime('%Y-%m-%d')
        if today != self.current_date:
            self.current_date = today
            self.close()
            dir_name = os.path.dirname(self.filename_format)
            file_name = os.path.basename(self.filename_format).format(self.current_date)
            self.baseFilename = os.path.abspath(os.path.join(dir_name, file_name))
            self.stream = self._open()
        super().emit(record)

# Daily Rotating File Handler using current date in name
rfh = DailyRotatingFileHandler(
    filename_format="logs/paper_trading_{}.log",
    encoding="utf-8"
)
rfh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
root_logger.addHandler(rfh)

# Specific LiveTrader Logger (propagates to root logger)
logger = logging.getLogger("LiveTrader")
logger.setLevel(logging.INFO)
logger.handlers = []
logger.propagate = True

class StructuralPaperTrader:
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_provider = FyersDataProvider()
        self.db = PostgresDatabase()
        self.tz = ZoneInfo("Asia/Kolkata")

        # ── Strategy Research Framework ──────────────────────────────────
        self.pipeline = IndicatorPipeline(
            pivot_window=3,
            zone_cluster_pct=0.002,
            min_zone_score=50.0,
        )

        self.registry = ExperimentRegistry()
        
        # 1. Structural
        _structural_exp = Experiment(
            name="Structural_v3.2_RVOL1.0",
            strategy=StructuralStrategy(rvol_threshold=1.0, min_zone_score=50.0),
            params={"rvol_threshold": 1.0, "min_zone_score": 50.0},
            description="Production structural strategy — RVOL threshold 1.0x"
        )
        self.registry.register(_structural_exp)
        self.db.save_experiment(_structural_exp.to_db_dict())

        _structural_08_exp = Experiment(
            name="Structural_v3.2_RVOL0.8",
            strategy=StructuralStrategy(rvol_threshold=0.8, min_zone_score=50.0),
            params={"rvol_threshold": 0.8, "min_zone_score": 50.0},
            description="Parallel experiment — RVOL threshold 0.8x"
        )
        self.registry.register(_structural_08_exp)
        self.db.save_experiment(_structural_08_exp.to_db_dict())

        # 2. EMA Pullback
        _ema_pullback_exp = Experiment(
            name="EMA_Pullback_20_50_RVOL1.0",
            strategy=EmaPullbackStrategy(rvol_threshold=1.0, min_efficiency=0.6),
            params={"rvol_threshold": 1.0, "min_efficiency": 0.6},
            description="EMA Pullback strategy — RVOL threshold 1.0x"
        )
        self.registry.register(_ema_pullback_exp)
        self.db.save_experiment(_ema_pullback_exp.to_db_dict())

        # 3. VWAP Reversion
        _vwap_reversion_exp = Experiment(
            name="VWAP_Reversion_1.5ATR_RVOL1.0",
            strategy=VwapReversionStrategy(rvol_threshold=1.0, vwap_stretch_multiplier=1.5),
            params={"rvol_threshold": 1.0, "vwap_stretch_multiplier": 1.5},
            description="VWAP Reversion strategy — RVOL threshold 1.0x"
        )
        self.registry.register(_vwap_reversion_exp)
        self.db.save_experiment(_vwap_reversion_exp.to_db_dict())

        # 4. Previous Day High/Low
        _prev_day_exp = Experiment(
            name="PrevDay_Extremes_RVOL1.2",
            strategy=PrevDayExtremesStrategy(breakout_rvol_threshold=1.2, reversal_rvol_threshold=1.0),
            params={"breakout_rvol_threshold": 1.2, "reversal_rvol_threshold": 1.0, "proximity_multiplier": 0.3},
            description="Previous Day High/Low sweeps and breakouts"
        )
        self.registry.register(_prev_day_exp)
        self.db.save_experiment(_prev_day_exp.to_db_dict())

        # 5. ORB (15m and 30m)
        _orb_15_exp = Experiment(
            name="ORB_15m_RVOL1.2",
            strategy=OrbStrategy(rvol_threshold=1.2, opening_range_minutes=15),
            params={"rvol_threshold": 1.2, "opening_range_minutes": 15},
            description="Opening Range Breakout 15m range"
        )
        self.registry.register(_orb_15_exp)
        self.db.save_experiment(_orb_15_exp.to_db_dict())

        _orb_30_exp = Experiment(
            name="ORB_30m_RVOL1.2",
            strategy=OrbStrategy(rvol_threshold=1.2, opening_range_minutes=30),
            params={"rvol_threshold": 1.2, "opening_range_minutes": 30},
            description="Opening Range Breakout 30m range"
        )
        self.registry.register(_orb_30_exp)
        self.db.save_experiment(_orb_30_exp.to_db_dict())

        # 6. ATR Squeeze Breakout
        _atr_squeeze_exp = Experiment(
            name="ATR_Squeeze_RVOL1.0",
            strategy=AtrSqueezeStrategy(rvol_threshold=1.0, atr_percentile_threshold=0.20),
            params={"rvol_threshold": 1.0, "atr_percentile_threshold": 0.20},
            description="ATR Squeeze Breakout volatility compression"
        )
        self.registry.register(_atr_squeeze_exp)
        self.db.save_experiment(_atr_squeeze_exp.to_db_dict())

        self.portfolios = PortfolioManager()
        self.portfolios.register("Structural_v3.2_RVOL1.0")
        self.portfolios.register("Structural_v3.2_RVOL0.8")
        self.portfolios.register("EMA_Pullback_20_50_RVOL1.0")
        self.portfolios.register("VWAP_Reversion_1.5ATR_RVOL1.0")
        self.portfolios.register("PrevDay_Extremes_RVOL1.2")
        self.portfolios.register("ORB_15m_RVOL1.2")
        self.portfolios.register("ORB_30m_RVOL1.2")
        self.portfolios.register("ATR_Squeeze_RVOL1.0")

        # active_trades keyed by (symbol, experiment_name) — independent per experiment
        self.active_trades: Dict[Tuple[str, str], Dict] = {}
        # active_counterfactuals keyed by candidate_id (multiple per symbol, unchanged)
        self.active_counterfactuals: Dict[str, Dict] = {}
        # active_cf_theses: deduplication index — one CF per (symbol, exp, setup_type, direction)
        # Prevents one structural opportunity from spawning one CF per candle
        self.active_cf_theses: Dict[Tuple[str, str, str, str], str] = {}  # key → candidate_id
        # EOD report: generated once per session at 15:35, reset on new day
        self._report_generated_today: bool = False
        self._report_date: str = ""
        
        # Load open real positions from DB on startup
        open_reals = self.db.get_open_positions()
        for op in open_reals:
            symbol = op['symbol']
            experiment_name = op.get('experiment_name', 'Structural_v3.2_RVOL1.0')
            key = (symbol, experiment_name)
            self.active_trades[key] = {
                'trade_id': op['trade_id'],
                'candidate_id': op['candidate_id'],
                'symbol': symbol,
                'experiment_name': experiment_name,
                'strategy_id': op.get('strategy_id', 'structural'),
                'version': op.get('version', 'v3.2'),
                'signal': op['signal_type'],
                'entry_price': op['entry_price'],
                'entry_time': op['entry_time'],
                'stop_loss': op['stop_loss'],
                'take_profit': op['take_profit'],
                'initial_stop_loss': op['initial_stop_loss'],
                'initial_take_profit': op['initial_take_profit'],
                'stop_loss_distance': op['stop_loss_distance'],
                'highest_price': op['highest_price'],
                'lowest_price': op['lowest_price'],
                'strategy': op['strategy'],
                'features': op.get('features', {}),
                'bars_held': op.get('bars_held', 0),
                'max_closed_profit_r': op.get('max_closed_profit_r', 0.0) or 0.0,
                'setup_type': op['setup_type'],
                'strategy_version': op.get('signal_logic_version', 'v3.2'),
                'market_regime': op.get('market_regime', 'UNKNOWN'),
                'is_counterfactual': False,
                'confidence': op.get('confidence'),
                'diagnostics': op.get('diagnostics')
            }
            # recovered_after_minutes: distinguish quick restart from multi-hour outage
            recovered_after_minutes = round(
                (datetime.now(self.tz) - op['entry_time']).total_seconds() / 60.0, 1
            )
            evt = {
                'event_id': f"evt_{int(datetime.now().timestamp())}_{symbol}_recovered",
                'trade_id': op['trade_id'],
                'timestamp': datetime.now(self.tz),
                'event_type': 'POSITION_RECOVERED',
                'payload': {
                    'candidate_id': op['candidate_id'],
                    'trade_id': op['trade_id'],
                    'stop_loss': op['stop_loss'],
                    'take_profit': op['take_profit'],
                    'highest_price': op['highest_price'],
                    'lowest_price': op['lowest_price'],
                    'max_closed_profit_r': op.get('max_closed_profit_r', 0.0) or 0.0,
                    'experiment_name': experiment_name,
                    'recovered_after_minutes': recovered_after_minutes
                }
            }
            self.db.save_trade_event(evt)
        if open_reals:
            logger.info(f"🔄 Recovered {len(open_reals)} active real positions: {list(self.active_trades.keys())}")

        # Load open counterfactual positions from DB on startup
        open_cfs = self.db.get_open_counterfactuals()
        for op in open_cfs:
            cand_id = op['candidate_id']
            symbol = op['symbol']
            self.active_counterfactuals[cand_id] = {
                'candidate_id': cand_id,
                'symbol': symbol,
                'experiment_name': op.get('experiment_name', 'Structural_v3.2_RVOL1.0'),
                'signal': op['signal_type'],
                'entry_price': op['entry_price'],
                'entry_time': op['timestamp'],
                'stop_loss': op['stop_loss'],
                'take_profit': op['take_profit'],
                'initial_stop_loss': op['initial_stop_loss'],
                'initial_take_profit': op['initial_take_profit'],
                'stop_loss_distance': op['stop_loss_distance'],
                'highest_price': op['highest_price'],
                'lowest_price': op['lowest_price'],
                'strategy': op['setup_type'],
                'features': {},
                'bars_held': op.get('bars_held', 0),
                'max_closed_profit_r': 0.0,
                'setup_type': op['setup_type'],
                'strategy_version': op.get('strategy_version', 'v3.2'),
                'market_regime': 'UNKNOWN',
                'is_counterfactual': True,
                'rejection_reasons': op.get('rejection_reasons', []),
                'confidence': op.get('confidence'),
                'diagnostics': op.get('diagnostics')
            }
            # Rebuild thesis deduplication index from recovered positions
            exp_name = op.get('experiment_name', 'Structural_v3.2_RVOL1.0')
            thesis_key = (symbol, exp_name, op['setup_type'], op['signal_type'])
            self.active_cf_theses[thesis_key] = cand_id
            # Log POSITION_RECOVERED event to database
            # recovered_after_minutes: distinguish quick restart from multi-hour outage
            recovered_after_minutes = round(
                (datetime.now(self.tz) - op['timestamp']).total_seconds() / 60.0, 1
            )
            evt = {
                'event_id': f"evt_{int(datetime.now().timestamp())}_{cand_id}_recovered_cf",
                'candidate_id': cand_id,
                'symbol': symbol,
                'timestamp': datetime.now(self.tz),
                'event_type': 'POSITION_RECOVERED',
                'payload': {
                    'candidate_id': cand_id,
                    'stop_loss': op['stop_loss'],
                    'take_profit': op['take_profit'],
                    'highest_price': op['highest_price'],
                    'lowest_price': op['lowest_price'],
                    'recovered_after_minutes': recovered_after_minutes
                }
            }
            self.db.save_counterfactual_event(evt)
        if open_cfs:
            logger.info(f"🔄 Recovered {len(open_cfs)} active counterfactual positions: {list(self.active_counterfactuals.keys())}")
        
        logger.info("🏛️ Structural Paper Trader Initialized | Active Position Tracking Enabled")

    def market_loop(self):
        """Main loop to be run every 5 minutes during market hours."""
        now = datetime.now(self.tz)

        # Only run between 09:00 and 15:59 IST
        if not (9 <= now.hour < 16):
            return

        logger.info(f"--- {now.strftime('%H:%M:%S')} Market Pulse ---")

        try:
            # 1. Fetch Multi-Timeframe Data (once per symbol)
            end_date = datetime.now(self.tz)
            start_date_d1 = end_date - timedelta(days=40)
            start_date_h1 = end_date - timedelta(days=10)
            start_date_m5 = end_date - timedelta(days=5)

            current_prices = {}
            fetched = {}  # symbol -> (d1, h1, m5)

            for symbol in self.symbols:
                d1 = self.data_provider.get_historical_data(symbol, start_date_d1, end_date, "1D")
                h1 = self.data_provider.get_historical_data(symbol, start_date_h1, end_date, "60")
                m5 = self.data_provider.get_historical_data(symbol, start_date_m5, end_date, "5")
                if d1 is not None and h1 is not None and m5 is not None:
                    fetched[symbol] = (d1, h1, m5)
                    current_prices[symbol] = float(m5['close'].iloc[-1])
                else:
                    logger.warning(f"⚠️ Could not fetch complete MTF data for {symbol}")

            # 2. Update Active Trades & Counterfactuals
            self._update_active_trades(current_prices, now)

            # 3. Close-of-session guard — no new entries after 15:25
            # The force-exit at 15:25 closes positions, but without this guard new
            # CFs are immediately re-entered and force-exited on every subsequent
            # candle until the scheduler winds down — creating a spurious loop.
            SESSION_CUTOFF_HOUR, SESSION_CUTOFF_MIN = 15, 25
            SESSION_REPORT_MIN = 35
            if now.hour > SESSION_CUTOFF_HOUR or (
                now.hour == SESSION_CUTOFF_HOUR and now.minute >= SESSION_CUTOFF_MIN
            ):
                logger.info("🔒 Session cutoff reached (15:25) — no new entries.")
                # Write one summary row per experiment into experiment_daily_metrics
                today_str = now.strftime("%Y-%m-%d")
                for exp in self.registry.active_experiments:
                    self.db.save_experiment_daily_metrics(
                        date_str=today_str,
                        experiment_name=exp.name,
                        config_hash=exp.config_hash,
                    )
                # Generate EOD report at 15:35 — once per session
                if (
                    now.minute >= SESSION_REPORT_MIN
                    and (
                        not self._report_generated_today
                        or self._report_date != today_str
                    )
                ):
                    try:
                        from src.reports.eod_report_generator import EODReportGenerator
                        gen = EODReportGenerator(self.db, self.data_provider)
                        md_path, json_path = gen.generate(today_str)
                        logger.info(f"📝 EOD report: {md_path}")
                        self._report_generated_today = True
                        self._report_date = today_str
                    except Exception as e:
                        logger.error(f"❌ EOD report generation failed: {e}", exc_info=True)
                return

            # 4. Compute snapshot + run experiments per symbol
            total_signals = 0
            for symbol, (d1, h1, m5) in fetched.items():
                price = current_prices[symbol]

                snapshot = self.pipeline.compute(symbol, price, d1, h1, m5, now)
                if snapshot is None:
                    logger.warning(f"⚠️ Pipeline returned None for {symbol}")
                    continue

                results = self.registry.run(snapshot)

                for result in results:
                    if result.errors:
                        logger.warning(f"⚡ [{result.experiment_name}] errors: {result.errors}")
                    if result.warnings:
                        logger.info(f"⚡ [{result.experiment_name}] warnings: {result.warnings}")

                    for sig in result.signals:
                        experiment_name = sig.get('experiment_name', result.experiment_name)
                        trade_key = (symbol, experiment_name)
                        total_signals += 1

                        # Suffix candidate_id with experiment_name to isolate counterfactual positions
                        if 'candidate_id' in sig and sig['candidate_id']:
                            if not sig['candidate_id'].endswith(f"_{experiment_name}"):
                                sig['candidate_id'] = f"{sig['candidate_id']}_{experiment_name}"

                        if sig['accepted']:
                            # One real trade per (symbol, experiment_name)
                            if trade_key in self.active_trades:
                                logger.debug(f"↩️  [{experiment_name}] Already have open position on {symbol}, skipping.")
                                continue
                            logger.info(f"🚀 SIGNAL: {symbol} {sig['signal']} | [{experiment_name}]")
                            logger.info(f"   Entry: {sig['price']} | SL: {sig['stop_loss']} | TP: {sig['take_profit']} (RR: {sig['rr_ratio']})")
                            self._enter_position(sig, now, trade_key, is_counterfactual=False)
                            self.portfolios.on_entry(experiment_name, now)
                        else:
                            MAX_ACTIVE_COUNTERFACTUALS = 500
                            if len(self.active_counterfactuals) >= MAX_ACTIVE_COUNTERFACTUALS:
                                logger.warning(f"⚠️ CF safety limit reached, skipping {symbol}")
                                continue

                            # Deduplication: strategy-defined thesis key, prepend experiment_name
                            exp_obj = self.registry.get(result.experiment_name)
                            if exp_obj:
                                thesis_base = exp_obj.strategy.thesis_key(sig)
                            else:
                                setup_type = sig.get('strategy', 'UNKNOWN')
                                direction = sig.get('signal', '')
                                thesis_base = (symbol, setup_type, direction)

                            thesis_key = (result.experiment_name,) + thesis_base

                            if thesis_key in self.active_cf_theses:
                                existing_cand = self.active_cf_theses[thesis_key]
                                logger.debug(
                                    f"↩️  [{experiment_name}] Thesis already tracked: "
                                    f"{thesis_base} → {existing_cand[-30:]}"
                                )
                                continue

                            logger.info(
                                f"👻 [{experiment_name}] CF {sig.get('strategy','')} {symbol} {sig.get('signal','')} "
                                f"| Rejected: {sig['rejection_reasons']}"
                            )
                            self._enter_position(sig, now, trade_key, is_counterfactual=True)

            if total_signals == 0:
                logger.info("🧘 Status: Sidelined (No Institutional Alignment)")

        except Exception as e:
            logger.error(f"❌ Error in market loop: {e}", exc_info=True)

    def _update_active_trades(self, current_prices: Dict[str, float], timestamp):
        """Evaluate open positions against latest market prices."""
        # Update real trades — keyed by (symbol, experiment_name)
        for key in list(self.active_trades.keys()):
            symbol, experiment_name = key
            if symbol not in current_prices:
                continue
            pos = self.active_trades[key]
            is_closed = self._update_position(pos, current_prices[symbol], timestamp)
            if is_closed:
                pnl_r = pos.get('_last_pnl_r', 0.0)
                self.portfolios.on_exit(experiment_name, pnl_r, timestamp)
                self.active_trades.pop(key)

        # Update counterfactual trades — keyed by candidate_id (unchanged)
        for cand_id in list(self.active_counterfactuals.keys()):
            pos = self.active_counterfactuals[cand_id]
            symbol = pos['symbol']
            if symbol not in current_prices:
                continue
            is_closed = self._update_position(pos, current_prices[symbol], timestamp)
            if is_closed:
                # Clean up thesis deduplication index so next candle can start a fresh CF
                exp_name = pos.get('experiment_name', 'Structural_v3.2_RVOL1.0')
                # Rebuild thesis_base via the strategy if available, else use fallback
                exp_obj = self.registry.get(exp_name)
                if exp_obj:
                    # Reconstruct a minimal sig-like dict for thesis_key()
                    _sig_proxy = {'symbol': symbol, 'strategy': pos.get('setup_type', ''), 'signal': pos.get('signal', '')}
                    thesis_base = exp_obj.strategy.thesis_key(_sig_proxy)
                else:
                    thesis_base = (symbol, pos.get('setup_type', ''), pos.get('signal', ''))
                thesis_key = (exp_name,) + thesis_base
                self.active_cf_theses.pop(thesis_key, None)
                self.active_counterfactuals.pop(cand_id)

    def _update_position(self, pos: Dict, current_price: float, timestamp) -> bool:
        """Evaluate a position against the latest market tick. Returns True if position exited."""
        symbol = pos['symbol']
        is_cf = pos.get('is_counterfactual', False)
        stop_loss_distance = pos['stop_loss_distance']
        
        # Increment bars held
        pos['bars_held'] = pos.get('bars_held', 0) + 1
        
        # Update extremes
        old_highest = pos['highest_price']
        old_lowest = pos['lowest_price']
        pos['highest_price'] = max(old_highest, current_price)
        pos['lowest_price'] = min(old_lowest, current_price)
        
        # Calculate current R PnL
        if pos['signal'] == 'BUY CALL':
            current_pnl_r = (current_price - pos['entry_price']) / stop_loss_distance if stop_loss_distance > 0 else 0.0
        else: # BUY PUT
            current_pnl_r = (pos['entry_price'] - current_price) / stop_loss_distance if stop_loss_distance > 0 else 0.0
            
        pos['max_closed_profit_r'] = max(pos.get('max_closed_profit_r', 0.0), current_pnl_r)

        is_closed = False
        exit_reason = None
        exit_price = current_price

        if pos['signal'] == 'BUY CALL':
            # Check SL breach
            if current_price <= pos['stop_loss']:
                is_closed = True
                exit_price = pos['stop_loss']
                exit_reason = 'STOP_LOSS'
            # Check TP expansion — capture old values before modifying pos
            elif current_price >= pos['take_profit']:
                old_sl, old_tp = pos['stop_loss'], pos['take_profit']
                pos['take_profit'] = old_tp + stop_loss_distance
                new_sl = current_price - stop_loss_distance
                pos['stop_loss'] = max(old_sl, new_sl)
                self._log_position_update(pos, current_price, timestamp, 'TP_EXPANSION',
                                          old_sl=old_sl, old_tp=old_tp)
            # Check trailing SL
            elif current_price > old_highest:
                old_sl = pos['stop_loss']
                new_sl = current_price - stop_loss_distance
                if new_sl > old_sl:
                    pos['stop_loss'] = new_sl
                    self._log_position_update(pos, current_price, timestamp, 'TRAILING_SL',
                                              old_sl=old_sl, old_tp=None)
                    
        elif pos['signal'] == 'BUY PUT':
            # Check SL breach
            if current_price >= pos['stop_loss']:
                is_closed = True
                exit_price = pos['stop_loss']
                exit_reason = 'STOP_LOSS'
            # Check TP expansion — capture old values before modifying pos
            elif current_price <= pos['take_profit']:
                old_sl, old_tp = pos['stop_loss'], pos['take_profit']
                pos['take_profit'] = old_tp - stop_loss_distance
                new_sl = current_price + stop_loss_distance
                pos['stop_loss'] = min(old_sl, new_sl)
                self._log_position_update(pos, current_price, timestamp, 'TP_EXPANSION',
                                          old_sl=old_sl, old_tp=old_tp)
            # Check trailing SL
            elif current_price < old_lowest:
                old_sl = pos['stop_loss']
                new_sl = current_price + stop_loss_distance
                if new_sl < old_sl:
                    pos['stop_loss'] = new_sl
                    self._log_position_update(pos, current_price, timestamp, 'TRAILING_SL',
                                              old_sl=old_sl, old_tp=None)

        # Session force exit check (15:25 PM IST)
        if not is_closed and timestamp.hour == 15 and timestamp.minute >= 25:
            is_closed = True
            exit_price = current_price
            exit_reason = 'SESSION_END'

        if is_closed:
            # Calculate final PnL R-units
            if pos['signal'] == 'BUY CALL':
                pnl_r = (exit_price - pos['entry_price']) / stop_loss_distance if stop_loss_distance > 0 else 0.0
            else: # BUY PUT
                pnl_r = (pos['entry_price'] - exit_price) / stop_loss_distance if stop_loss_distance > 0 else 0.0
            
            pnl_r -= 0.05  # Transaction cost buffer
            self._exit_position(pos, exit_price, exit_reason, timestamp, pnl_r)
            return True

        return False

    def _enter_position(self, sig: Dict, timestamp, trade_key: Tuple, is_counterfactual: bool):
        symbol = sig['symbol']
        experiment_name = sig.get('experiment_name', 'Structural_v3.2_RVOL1.0')
        strategy_id = sig.get('strategy_id', 'structural')
        version = sig.get('version', 'v3.2')
        entry_price = sig['price']
        sl_price = sig['stop_loss']
        tp_price = sig['take_profit']
        candidate_id = sig.get('candidate_id')
        trade_id = f"trade_{symbol.replace(':', '_').replace('-', '_')}_{int(timestamp.timestamp())}"

        pos = {
            'trade_id': trade_id if not is_counterfactual else None,
            'candidate_id': candidate_id,
            'symbol': symbol,
            'experiment_name': experiment_name,
            'strategy_id': strategy_id,
            'version': version,
            'signal': sig['signal'],
            'entry_price': entry_price,
            'entry_time': timestamp,
            'stop_loss': sl_price,
            'take_profit': tp_price,
            'initial_stop_loss': sl_price,
            'initial_take_profit': tp_price,
            'stop_loss_distance': abs(entry_price - sl_price) if sl_price else 0.0,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'strategy': sig['strategy'],
            'features': sig.get('features', {}),
            'bars_held': 0,
            'max_closed_profit_r': 0.0,
            'setup_type': sig.get('strategy'),
            'strategy_version': version,
            'market_regime': sig.get('features', {}).get('market_regime', 'UNKNOWN'),
            'is_counterfactual': is_counterfactual,
            'rejection_reasons': sig.get('rejection_reasons', []),
            '_last_pnl_r': 0.0,  # Set by _exit_position for portfolio tracking
            'confidence': sig.get('confidence'),
            'diagnostics': sig.get('diagnostics'),
        }
        
        if is_counterfactual:
            self.active_counterfactuals[candidate_id] = pos

            # Register in thesis deduplication index — use strategy.thesis_key to match
            # the same key generated in market_loop (consistency is critical)
            exp_obj = self.registry.get(experiment_name)
            if exp_obj:
                thesis_base = exp_obj.strategy.thesis_key(sig)
            else:
                thesis_base = (symbol, sig.get('strategy', ''), sig.get('signal', ''))
            thesis_key = (experiment_name,) + thesis_base
            self.active_cf_theses[thesis_key] = candidate_id
            
            # Save ENTRY event
            event = {
                'event_id': f"evt_{int(timestamp.timestamp())}_{candidate_id}_entry_cf",
                'candidate_id': candidate_id,
                'symbol': symbol,
                'timestamp': timestamp,
                'event_type': 'ENTRY',
                'payload': {
                    'entry_price': entry_price,
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'rejection_reasons': sig.get('rejection_reasons', [])
                }
            }
            self.db.save_counterfactual_event(event)

            # Save to counterfactual_results
            result = {
                'candidate_id': candidate_id,
                'timestamp': timestamp,
                'symbol': symbol,
                'signal_type': sig['signal'],
                'setup_type': sig['strategy'],
                'rejection_reasons': sig.get('rejection_reasons', []),
                'primary_rejection_reason': sig.get('rejection_reasons', ['NONE'])[0] if sig.get('rejection_reasons') else 'NONE',
                'entry_price': entry_price,
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'initial_stop_loss': sl_price,
                'initial_take_profit': tp_price,
                'highest_price': entry_price,
                'lowest_price': entry_price,
                'stop_loss_distance': abs(entry_price - sl_price) if sl_price else 0.0,
                'exit_time': None,
                'exit_price': None,
                'mfe_r': 0.0,
                'mae_r': 0.0,
                'final_pnl_r': 0.0,
                'duration_minutes': 0.0,
                'bars_held': 0,
                'exit_reason': 'OPEN',
                'strategy_version': version,
                'capture_rate': 0.0,
                'experiment_name': experiment_name,
                'strategy_id': strategy_id,
                'version': version,
                'confidence': sig.get('confidence'),
                'diagnostics': sig.get('diagnostics'),
            }
            self.db.save_counterfactual_result(result)
        else:
            self.active_trades[trade_key] = pos
            
            # Log to CSV
            self._log_to_journal(
                timestamp=timestamp.isoformat(),
                symbol=symbol,
                action='ENTRY',
                signal_type=sig['signal'],
                price=entry_price,
                stop_loss=sl_price,
                take_profit=tp_price,
                strategy=sig['strategy'],
                pnl_r=0.0,
                mfe_r=0.0,
                mae_r=0.0,
                max_closed_profit_r=0.0,
                duration_minutes=0.0,
                bars_held=0,
                reason='INITIAL'
            )
            
            # Save ENTRY event
            event = {
                'event_id': f"evt_{int(timestamp.timestamp())}_{symbol}_entry",
                'trade_id': trade_id,
                'timestamp': timestamp,
                'event_type': 'ENTRY',
                'payload': {
                    'entry_price': entry_price,
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'candidate_id': candidate_id
                }
            }
            self.db.save_trade_event(event)

            # Save to trade_performance
            perf = {
                'trade_id': trade_id,
                'candidate_id': candidate_id,
                'entry_time': timestamp,
                'exit_time': None,
                'strategy': sig['strategy'],
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': None,
                'mfe': 0.0,
                'mae': 0.0,
                'pnl': 0.0,
                'exit_reason': 'OPEN',
                'features': sig.get('features', {}),
                'setup_type': sig['strategy'],
                'mfe_r': 0.0,
                'mae_r': 0.0,
                'max_closed_profit_r': 0.0,
                'final_pnl_r': 0.0,
                'duration_minutes': 0.0,
                'bars_held': 0,
                'market_regime': pos['market_regime'],
                'signal_logic_version': version,
                'position_logic_version': 'v3.1',
                'risk_logic_version': 'v1.1',
                'stop_loss': sl_price,
                'take_profit': tp_price,
                'initial_stop_loss': sl_price,
                'initial_take_profit': tp_price,
                'highest_price': entry_price,
                'lowest_price': entry_price,
                'stop_loss_distance': abs(entry_price - sl_price) if sl_price else 0.0,
                'signal_type': sig['signal'],
                'capture_rate': 0.0,
                'experiment_name': experiment_name,
                'strategy_id': strategy_id,
                'version': version,
                'confidence': sig.get('confidence'),
                'diagnostics': sig.get('diagnostics'),
            }
            self.db.save_trade_performance(perf)
            
        cand_short = (candidate_id or '')[-20:]  # last 20 chars for readability
        if is_counterfactual:
            logger.info(
                f"🟢 ENTRY [CF|{experiment_name}|{cand_short}]: "
                f"{symbol} {sig['signal']} @ {entry_price:.2f} | "
                f"SL: {sl_price:.2f} TP: {tp_price:.2f}"
            )
        else:
            logger.info(
                f"🟢 ENTRY [{experiment_name}|{trade_id[-16:]}]: "
                f"{symbol} {sig['signal']} @ {entry_price:.2f} | "
                f"SL: {sl_price:.2f} TP: {tp_price:.2f}"
            )

    def _log_position_update(self, pos: Dict, current_price: float, timestamp, reason: str,
                              old_sl: float = None, old_tp: float = None):
        symbol = pos['symbol']
        entry_price = pos['entry_price']
        stop_loss_distance = pos['stop_loss_distance']
        highest = pos['highest_price']
        lowest = pos['lowest_price']
        is_cf = pos.get('is_counterfactual', False)
        
        # Calculate excursions
        if pos['signal'] == 'BUY CALL':
            mfe_r = (highest - entry_price) / stop_loss_distance if stop_loss_distance > 0 else 0.0
            mae_r = (entry_price - lowest) / stop_loss_distance if stop_loss_distance > 0 else 0.0
        else: # BUY PUT
            mfe_r = (entry_price - lowest) / stop_loss_distance if stop_loss_distance > 0 else 0.0
            mae_r = (highest - entry_price) / stop_loss_distance if stop_loss_distance > 0 else 0.0
            
        max_closed_profit_r = pos.get('max_closed_profit_r', 0.0)
        duration_minutes = (timestamp - pos['entry_time']).total_seconds() / 60.0
        bars_held = pos.get('bars_held', 0)

        if not is_cf:
            # Log to CSV
            self._log_to_journal(
                timestamp=timestamp.isoformat(),
                symbol=symbol,
                action='UPDATE',
                signal_type=pos['signal'],
                price=current_price,
                stop_loss=pos['stop_loss'],
                take_profit=pos['take_profit'],
                strategy=pos['strategy'],
                pnl_r=0.0,
                mfe_r=mfe_r,
                mae_r=mae_r,
                max_closed_profit_r=max_closed_profit_r,
                duration_minutes=duration_minutes,
                bars_held=bars_held,
                reason=reason
            )
            
            # Log to trade_events table
            event = {
                'event_id': f"evt_{int(timestamp.timestamp())}_{symbol}_{reason.lower()}",
                'trade_id': pos['trade_id'],
                'timestamp': timestamp,
                'event_type': 'SL_TRAIL' if reason == 'TRAILING_SL' else 'TP_EXPANSION',
                'payload': {
                    'current_price': current_price,
                    'old_sl': old_sl,
                    'old_tp': old_tp,
                    'new_sl': pos['stop_loss'],
                    'new_tp': pos['take_profit'],
                    'reason': reason,
                    'mfe_r': mfe_r,
                    'mae_r': mae_r,
                    'max_closed_profit_r': max_closed_profit_r
                }
            }
            self.db.save_trade_event(event)

            # Update trade_performance table
            perf = {
                'trade_id': pos['trade_id'],
                'candidate_id': pos['candidate_id'],
                'entry_time': pos['entry_time'],
                'exit_time': None,
                'strategy': pos['strategy'],
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': None,
                'mfe': mfe_r,
                'mae': mae_r,
                'pnl': 0.0,
                'exit_reason': f'OPEN_{reason}',
                'features': pos.get('features', {}),
                'setup_type': pos['setup_type'],
                'mfe_r': mfe_r,
                'mae_r': mae_r,
                'max_closed_profit_r': max_closed_profit_r,
                'final_pnl_r': 0.0,
                'duration_minutes': duration_minutes,
                'bars_held': bars_held,
                'market_regime': pos['market_regime'],
                'signal_logic_version': pos['strategy_version'],
                'position_logic_version': 'v3.1',
                'risk_logic_version': 'v1.1',
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit'],
                'initial_stop_loss': pos['initial_stop_loss'],
                'initial_take_profit': pos['initial_take_profit'],
                'highest_price': highest,
                'lowest_price': lowest,
                'stop_loss_distance': stop_loss_distance,
                'signal_type': pos['signal'],
                'capture_rate': 0.0,
                'confidence': pos.get('confidence'),
                'diagnostics': pos.get('diagnostics')
            }
            self.db.save_trade_performance(perf)
        else:
            # Save counterfactual event
            event = {
                'event_id': f"evt_{int(timestamp.timestamp())}_{pos['candidate_id']}_{reason.lower()}_cf",
                'candidate_id': pos['candidate_id'],
                'symbol': symbol,
                'timestamp': timestamp,
                'event_type': 'SL_TRAIL' if reason == 'TRAILING_SL' else 'TP_EXPANSION',
                'payload': {
                    'current_price': current_price,
                    'old_sl': old_sl,
                    'old_tp': old_tp,
                    'new_sl': pos['stop_loss'],
                    'new_tp': pos['take_profit'],
                    'reason': reason,
                    'mfe_r': mfe_r,
                    'mae_r': mae_r,
                    'max_closed_profit_r': max_closed_profit_r
                }
            }
            self.db.save_counterfactual_event(event)

            # Update counterfactual result in DB
            result = {
                'candidate_id': pos['candidate_id'],
                'timestamp': pos['entry_time'],
                'symbol': symbol,
                'signal_type': pos['signal'],
                'setup_type': pos['setup_type'],
                'rejection_reasons': pos.get('rejection_reasons', []),
                'primary_rejection_reason': pos.get('rejection_reasons', ['NONE'])[0] if pos.get('rejection_reasons') else 'NONE',
                'entry_price': entry_price,
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit'],
                'initial_stop_loss': pos['initial_stop_loss'],
                'initial_take_profit': pos['initial_take_profit'],
                'highest_price': highest,
                'lowest_price': lowest,
                'stop_loss_distance': stop_loss_distance,
                'exit_time': None,
                'exit_price': None,
                'mfe_r': mfe_r,
                'mae_r': mae_r,
                'final_pnl_r': 0.0,
                'duration_minutes': duration_minutes,
                'bars_held': bars_held,
                'exit_reason': f'OPEN_{reason}',
                'strategy_version': pos['strategy_version'],
                'capture_rate': 0.0,
                'confidence': pos.get('confidence'),
                'diagnostics': pos.get('diagnostics')
            }
            self.db.save_counterfactual_result(result)
            
        exp_tag = pos.get('experiment_name', '')
        cand_short = (pos.get('candidate_id') or pos.get('trade_id') or '')[-20:]
        logger.info(
            f"🟡 UPDATE ({reason}) {'[CF' if is_cf else '['}{exp_tag}|{cand_short}]: "
            f"{symbol} @ {current_price:.2f} | SL→{pos['stop_loss']:.2f} TP→{pos['take_profit']:.2f}"
        )

    def _exit_position(self, pos: Dict, exit_price: float, reason: str, timestamp, pnl_r: float):
        symbol = pos['symbol']
        entry_price = pos['entry_price']
        stop_loss_distance = pos['stop_loss_distance']
        highest = max(pos['highest_price'], exit_price)
        lowest = min(pos['lowest_price'], exit_price)
        is_cf = pos.get('is_counterfactual', False)
        
        # Excursions
        if pos['signal'] == 'BUY CALL':
            mfe_r = (highest - entry_price) / stop_loss_distance if stop_loss_distance > 0 else 0.0
            mae_r = (entry_price - lowest) / stop_loss_distance if stop_loss_distance > 0 else 0.0
        else: # BUY PUT
            mfe_r = (entry_price - lowest) / stop_loss_distance if stop_loss_distance > 0 else 0.0
            mae_r = (highest - entry_price) / stop_loss_distance if stop_loss_distance > 0 else 0.0

        max_closed_profit_r = max(pos.get('max_closed_profit_r', 0.0), pnl_r + 0.05)
        duration_minutes = (timestamp - pos['entry_time']).total_seconds() / 60.0
        bars_held = pos.get('bars_held', 0)

        # Map exit reason codes
        mapped_reason = reason
        if reason == 'STOP_LOSS':
            if pos['stop_loss'] == pos['initial_stop_loss']:
                mapped_reason = 'INITIAL_SL'
            else:
                mapped_reason = 'TRAILING_SL'
        elif reason == 'TARGET_ZONE':
            mapped_reason = 'TARGET_ZONE'
        elif reason == 'SESSION_END':
            mapped_reason = 'SESSION_END'

        # Guard capture rate — store NULL (None) when there was nothing to capture
        # (MFE <= 0 means price never moved in our favour)
        # capture=0.0 falsely implies "terrible efficiency"; NULL means "N/A"
        capture_rate = None
        if mfe_r > 0.0:
            capture_rate = round(pnl_r / mfe_r, 4)

        # Holding efficiency: R earned per bar held
        # A 4R in 5 bars vs 4R in 60 bars are very different strategies
        holding_efficiency = round(pnl_r / max(bars_held, 1), 4)

        # Store pnl_r on pos so _update_active_trades can pass it to PortfolioManager
        pos['_last_pnl_r'] = pnl_r

        if not is_cf:
            # Log to CSV
            self._log_to_journal(
                timestamp=timestamp.isoformat(),
                symbol=symbol,
                action='EXIT',
                signal_type=pos['signal'],
                price=exit_price,
                stop_loss=pos['stop_loss'],
                take_profit=pos['take_profit'],
                strategy=pos['strategy'],
                pnl_r=pnl_r,
                mfe_r=mfe_r,
                mae_r=mae_r,
                max_closed_profit_r=max_closed_profit_r,
                duration_minutes=duration_minutes,
                bars_held=bars_held,
                reason=mapped_reason
            )
            
            # Log to trade_events table
            event = {
                'event_id': f"evt_{int(timestamp.timestamp())}_{symbol}_exit",
                'trade_id': pos['trade_id'],
                'timestamp': timestamp,
                'event_type': 'EXIT',
                'payload': {
                    'exit_price': exit_price,
                    'exit_reason': mapped_reason,
                    'final_pnl_r': pnl_r,
                    'duration_minutes': duration_minutes,
                    'bars_held': bars_held
                }
            }
            self.db.save_trade_event(event)

            # Save to trade_performance table
            perf = {
                'trade_id': pos['trade_id'],
                'candidate_id': pos['candidate_id'],
                'entry_time': pos['entry_time'],
                'exit_time': timestamp,
                'strategy': pos['strategy'],
                'symbol': symbol,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'mfe': mfe_r,
                'mae': mae_r,
                'pnl': pnl_r,
                'exit_reason': mapped_reason,
                'features': pos.get('features', {}),
                'setup_type': pos['setup_type'],
                'mfe_r': mfe_r,
                'mae_r': mae_r,
                'max_closed_profit_r': max_closed_profit_r,
                'final_pnl_r': pnl_r,
                'duration_minutes': duration_minutes,
                'bars_held': bars_held,
                'market_regime': pos['market_regime'],
                'signal_logic_version': pos['strategy_version'],
                'position_logic_version': 'v3.1',
                'risk_logic_version': 'v1.1',
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit'],
                'initial_stop_loss': pos['initial_stop_loss'],
                'initial_take_profit': pos['initial_take_profit'],
                'highest_price': highest,
                'lowest_price': lowest,
                'stop_loss_distance': stop_loss_distance,
                'signal_type': pos['signal'],
                'capture_rate': capture_rate,
                'holding_efficiency': holding_efficiency,
                'confidence': pos.get('confidence'),
                'diagnostics': pos.get('diagnostics')
            }
            self.db.save_trade_performance(perf)
        else:
            # Save counterfactual event
            event = {
                'event_id': f"evt_{int(timestamp.timestamp())}_{pos['candidate_id']}_exit_cf",
                'candidate_id': pos['candidate_id'],
                'symbol': symbol,
                'timestamp': timestamp,
                'event_type': 'EXIT',
                'payload': {
                    'exit_price': exit_price,
                    'exit_reason': mapped_reason,
                    'final_pnl_r': pnl_r,
                    'duration_minutes': duration_minutes,
                    'bars_held': bars_held
                }
            }
            self.db.save_counterfactual_event(event)

            # Save to counterfactual_results table
            result = {
                'candidate_id': pos['candidate_id'],
                'timestamp': pos['entry_time'],
                'symbol': symbol,
                'signal_type': pos['signal'],
                'setup_type': pos['setup_type'],
                'rejection_reasons': pos.get('rejection_reasons', []),
                'primary_rejection_reason': pos.get('rejection_reasons', ['NONE'])[0] if pos.get('rejection_reasons') else 'NONE',
                'entry_price': entry_price,
                'stop_loss': pos['stop_loss'],
                'take_profit': pos['take_profit'],
                'initial_stop_loss': pos['initial_stop_loss'],
                'initial_take_profit': pos['initial_take_profit'],
                'highest_price': highest,
                'lowest_price': lowest,
                'stop_loss_distance': stop_loss_distance,
                'exit_time': timestamp,
                'exit_price': exit_price,
                'mfe_r': mfe_r,
                'mae_r': mae_r,
                'final_pnl_r': pnl_r,
                'duration_minutes': duration_minutes,
                'bars_held': bars_held,
                'exit_reason': mapped_reason,
                'strategy_version': pos['strategy_version'],
                'capture_rate': capture_rate,
                'holding_efficiency': holding_efficiency,
                'experiment_name': pos.get('experiment_name', ''),
                'strategy_id': pos.get('strategy_id', ''),
                'version': pos.get('version', ''),
                'confidence': pos.get('confidence'),
                'diagnostics': pos.get('diagnostics'),
            }
            self.db.save_counterfactual_result(result)
            
        exp_tag = pos.get('experiment_name', '')
        cand_short = (pos.get('candidate_id') or pos.get('trade_id') or '')[-20:]
        capture_str = f"{capture_rate:.0%}" if capture_rate is not None else "N/A"
        logger.info(
            f"🔴 EXIT ({mapped_reason}) {'[CF' if is_cf else '['}{exp_tag}|{cand_short}]: "
            f"{symbol} @ {exit_price:.2f} | PnL {pnl_r:+.2f}R "
            f"| MFE {mfe_r:.2f}R | Capture {capture_str} "
            f"| {bars_held}bars HoldEff {holding_efficiency:+.3f}R/bar"
        )

    def _log_to_journal(self, timestamp, symbol, action, signal_type, price, stop_loss, take_profit, strategy, pnl_r, mfe_r, mae_r, max_closed_profit_r, duration_minutes, bars_held, reason):
        file_path = "trade_journal.csv"
        file_exists = os.path.exists(file_path)
        
        with open(file_path, "a") as f:
            if not file_exists:
                f.write("timestamp,symbol,action,signal_type,price,stop_loss,take_profit,strategy,pnl_r,mfe_r,mae_r,max_closed_profit_r,duration_minutes,bars_held,reason\n")
            f.write(f"{timestamp},{symbol},{action},{signal_type},{price:.2f},{stop_loss:.2f},{take_profit:.2f},{strategy},{pnl_r:.2f},{mfe_r:.2f},{mae_r:.2f},{max_closed_profit_r:.2f},{duration_minutes:.2f},{bars_held},{reason}\n")

def main():
    trader = StructuralPaperTrader(["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"])
    
    # Schedule to run every 5 minutes
    schedule.every(5).minutes.do(trader.market_loop)
    
    logger.info("⏱️ Scheduler started. Waiting for next 5-minute candle...")
    
    # Run once immediately for testing
    trader.market_loop()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
