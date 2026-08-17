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
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple, Optional

# Path Injection
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from src.adapters.data.fyers_data_provider import FyersDataProvider
from src.models.postgres_database import PostgresDatabase
from src.core.regime_router import is_regime_eligible

# Strategy Research Framework
from src.core.indicator_pipeline import IndicatorPipeline
from src.core.experiment_factory import build_registry
from src.core.portfolio import PortfolioManager
from src.core.expiry_blackout import ExpiryBlackoutManager
from src.warehouse.premarket_collector import PreMarketCollector

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

        # ── Real option-chain OI warehouse (background) ──────────────────
        # OptionWarehouse.run() is an asyncio loop that fetches real OI via the
        # Fyers depth endpoint and writes option_snapshots — the only source of
        # real (non-placeholder) OI in the system. It previously had no caller
        # anywhere in the live path, so option_snapshots was empty during
        # trading hours and PCR/max-pain in the dashboard had no real data.
        # Runs in its own daemon thread/event loop so a warehouse-side failure
        # can never block or crash the main 5-minute candle loop below.
        import threading
        import asyncio
        from src.warehouse.option_warehouse import OptionWarehouse
        self._option_warehouse = OptionWarehouse(list(symbols))

        def _run_option_warehouse():
            try:
                asyncio.run(self._option_warehouse.run())
            except Exception as e:
                logger.error(f"❌ Option warehouse thread died: {e}")

        threading.Thread(target=_run_option_warehouse, daemon=True, name="OptionWarehouse").start()
        logger.info("📡 Option warehouse started in background thread (real OI capture)")

        # ── Pre-market collector (9:00 poll → 9:15 freeze → 9:20 opening data) ─
        self._premarket_collector = PreMarketCollector(list(symbols))
        threading.Thread(
            target=self._premarket_collector.run,
            daemon=True,
            name="PreMarketCollector",
        ).start()
        logger.info("🌅 Pre-market collector started in background thread")

        from src.core.options_execution_engine import OptionExecutionEngine
        self.option_engine = OptionExecutionEngine(self.db, self.data_provider, strike_policy="ATM")

        from src.core.multi_leg_execution_engine import MultiLegExecutionEngine
        self.multi_leg_engine = MultiLegExecutionEngine(self.db, self.data_provider)

        from src.core.execution_auditor import ExecutionAuditor
        self.execution_auditor = ExecutionAuditor(self.db)

        from src.core.position_sizer import PositionSizer
        self.RISK_CAPITAL = 100000.0
        self.sizer = PositionSizer(capital=self.RISK_CAPITAL)

        # ── Live risk governor (REAL trades only) ────────────────────────
        # There was previously NO aggregate risk control in the live path: up to
        # (experiments × symbols) real positions could open with no daily-loss
        # halt and no exposure ceiling. These gates apply across all experiments.
        self.DAILY_LOSS_LIMIT_R    = -6.0   # halt new real entries once realized R for the day <= this
        self.MAX_CONCURRENT_REAL   = 6      # max simultaneous real positions (all experiments/symbols)
        self.MAX_DEPLOYED_FRACTION = 0.40   # max fraction of capital deployed across open real trades
        self.MAX_ATTEMPTS_PER_LEVEL = 2     # woodchopper: max real entries per (symbol, direction, price-bucket) per day
        self._risk_day             = None   # date-string the daily counters belong to
        self.daily_realized_r      = 0.0    # sum of realized pnl_r on real trades today
        self.trading_halted_today  = False  # set once the daily loss limit trips
        # Woodchopper protection: (symbol, direction, bucket) -> attempt count for today
        self._daily_level_attempts: Dict[tuple, int] = {}

        # ── Circuit breaker (sustained data-feed failure) ────────────────
        # Separate from trading_halted_today (daily-loss halt) on purpose: a
        # data outage isn't a daily-risk event and shouldn't reset at day-roll,
        # only when quotes actually start flowing again — and a real daily-loss
        # halt shouldn't get silently cleared just because one quote succeeded.
        self.QUOTE_FAILURE_HALT_THRESHOLD = 5   # consecutive all-quotes-failed ticks (~2.5min @ 30s cadence)
        self._quote_failure_streak = 0
        self.data_feed_halted = False
        if self.db.has_open_system_alert("DATA_FEED_DOWN"):
            # A previous process died mid-outage — stay halted until a live
            # quote actually proves the feed is back (checked every tick below).
            self.data_feed_halted = True
            logger.warning("⚠️ Resuming with an unresolved DATA_FEED_DOWN alert from a prior run — new real entries stay halted until quotes recover.")

        # Expiry & event blackout manager (Bug 18 fix)
        self.expiry_blackout = ExpiryBlackoutManager()

        # ── Live order placement (first cut, defaults to paper) ─────────
        # LIVE_MODE off by default (src/config/settings.py). Real trades only
        # (never counterfactual/shadow) place real orders once True. A run of
        # LIVE_ORDER_FAILURE_HALT_THRESHOLD consecutive order failures flips
        # live_mode back off automatically — see _on_live_order_failure().
        from src.config import settings as _settings
        from src.execution.fyers_order_executor import FyersOrderExecutor
        self.live_mode = _settings.LIVE_MODE
        self.max_live_lots = _settings.MAX_LIVE_LOTS
        self.live_order_executor = FyersOrderExecutor(self.data_provider.client)
        self.LIVE_ORDER_FAILURE_HALT_THRESHOLD = 2
        self.live_order_failure_streak = 0
        if self.live_mode:
            logger.warning(
                f"🔴 LIVE_MODE is ON — real orders will be placed for real trades "
                f"(max {self.max_live_lots} lot(s)/trade). Counterfactual trades stay simulated."
            )

        # ── Strategy Research Framework ──────────────────────────────────
        self.pipeline = IndicatorPipeline(
            pivot_window=3,
            zone_cluster_pct=0.002,
            min_zone_score=50.0,
        )

        # Experiment set lives in experiment_factory.py — the single source of
        # truth both this live loop and the backtester build from, so neither
        # can drift out of sync with the other.
        self.registry = build_registry()
        for _exp in self.registry.experiments:
            self.db.save_experiment(_exp.to_db_dict())

        # Last successfully computed MarketSnapshot per symbol — one candle
        # stale relative to _update_active_trades (which runs before this
        # candle's snapshot exists). Lets _update_position() read live
        # ATR/regime/structure for exit-management without forcing snapshot
        # computation to run on every tick (including session-cutoff/blackout
        # early-returns that currently skip it).
        self._last_snapshot: Dict[str, object] = {}

        self.portfolios = PortfolioManager()
        self.portfolios.register("Structural_v3.2_RVOL1.0")
        self.portfolios.register("Structural_v3.2_RVOL0.8")
        self.portfolios.register("Structural_v3.3_ExitMgmt")
        self.portfolios.register("EMA_Pullback_20_50_RVOL0.5")
        self.portfolios.register("VWAP_Reversion_1.5ATR_RVOL1.0")
        self.portfolios.register("PrevDay_Extremes_RVOL1.2")
        self.portfolios.register("ATR_Squeeze_RVOL1.5")
        self.portfolios.register("Geometry_v1.0_Score35")
        self.portfolios.register("Geometry_v1.0_Score50")
        self.portfolios.register("OrderFlow_v1.0")
        self.portfolios.register("VWAP_Reclaim_v1.0")
        self.portfolios.register("CPR_v1.0")
        self.portfolios.register("ORB_60m_IB_RVOL1.2")
        self.portfolios.register("VerticalSpread_v1.0")
        self.portfolios.register("Straddle_v1.0_VolCompression")
        self.portfolios.register("Strangle_v1.0_VolCompression")
        self.portfolios.register("OIWallReaction_v1.0")
        self.portfolios.register("PCRExtremeReversal_v1.0")
        self.portfolios.register("GapRegime_v2.0")
        self.portfolios.register("CreditSpread_v1.0_PCRFade")
        self.portfolios.register("IronCondor_v1.0")
        self.portfolios.register("Butterfly_v1.0")
        self.portfolios.register("IronButterfly_v1.0")
        self.portfolios.register("OI_Scalping_v1.0")
        self.portfolios.register("Consolidation_Breakout_v1.0")
        self.portfolios.register("Consolidation_Breakout_Tight_v1.0")
        self.portfolios.register("RSI2_MeanReversion_v1.0")
        self.portfolios.register("ExpiryAwareTheta_v1.0")
        self.portfolios.register("RelativeValue_NIFTY_BANKNIFTY_v1.0")
        self.portfolios.register("MomentumBurst_5m_v1.0_RVOL2.0")
        self.portfolios.register("HtfPullback_v1.0_Tol0.6pct")

        # active_trades keyed by (symbol, experiment_name) — independent per experiment
        self.active_trades: Dict[Tuple[str, str], Dict] = {}
        # active_counterfactuals keyed by candidate_id (multiple per symbol, unchanged)
        self.active_counterfactuals: Dict[str, Dict] = {}
        # active_cf_theses: deduplication index — one CF per (symbol, exp, setup_type, direction)
        # Prevents one structural opportunity from spawning one CF per candle
        self.active_cf_theses: Dict[Tuple[str, str, str, str], str] = {}  # key → candidate_id

        # Multi-leg options combos (vertical spreads, straddle/strangle) — kept
        # entirely separate from active_trades/active_counterfactuals since a
        # combo's PnL is combined-premium-based, not a single-leg index R-multiple.
        self.active_combo_trades: Dict[Tuple[str, str], Dict] = {}
        self.active_cf_combos: Dict[str, Dict] = {}
        self.active_cf_combo_theses: Dict[Tuple, str] = {}

        # EOD report: generated once per session at 15:35, reset on new day
        self._report_generated_today: bool = False
        self._report_date: str = ""

        # Position state lock & candle tracking
        self.position_lock = threading.RLock()
        self.last_processed_m5_time = None
        
        # Load open real positions from DB on startup
        from src.core.options_mapper import OptionsMapper
        open_reals = self.db.get_open_positions()
        now = datetime.now(self.tz)
        for op in open_reals:
            symbol = op['symbol']
            experiment_name = op.get('experiment_name', 'Structural_v3.2_RVOL1.0')
            key = (symbol, experiment_name)
            
            # Prevent prior-day state leakage (Self-Healing)
            entry_time = op['entry_time']
            if entry_time.date() < now.date():
                # Force-close the real position in the database at 15:25 on its entry day
                exit_time = entry_time.replace(hour=15, minute=25, second=0, microsecond=0)
                op_exit = dict(op)
                op_exit['exit_time'] = exit_time
                op_exit['exit_price'] = op['entry_price']
                op_exit['pnl'] = -0.05  # Transaction cost buffer only
                op_exit['final_pnl_r'] = -0.05
                op_exit['exit_reason'] = 'SESSION_END'
                op_exit['valid'] = False
                op_exit['validation_errors'] = "Orphaned recovery: closed on next startup."
                self.db.save_trade_performance(op_exit)
                logger.info(f"🧹 Self-Healed orphaned prior-day Real position: {op['trade_id']} entered on {entry_time.date()}")
                continue

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
                'tp1': op.get('tp1'),
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
                'status': 'OPEN',
                'confidence': op.get('confidence'),
                'diagnostics': op.get('diagnostics'),
                # Notional deployed — must be recovered or _deployed_capital()
                # silently undercounts this position after every restart.
                'position_size_inr': op.get('position_size_inr', 0.0) or 0.0,
                'lots': op.get('lots', 1.0) or 1.0,
                # Restore real-fill premium tracking so a mid-session restart
                # doesn't silently fall back to the index-point P&L proxy.
                'option_symbol': op.get('option_symbol'),
                'entry_premium': op.get('entry_premium'),
                'exit_premium': None,
                'lot_size': (
                    OptionsMapper.get_lot_size(op['option_symbol']) if op.get('option_symbol') else None
                ),
                'pnl_calculation_method': None,
            }
            # recovered_after_minutes: distinguish quick restart from multi-hour outage
            recovered_after_minutes = round(
                (datetime.now(self.tz) - op['entry_time']).total_seconds() / 60.0, 1
            )
            evt = {
                'event_id': f"evt_{int(datetime.now().timestamp())}_{symbol}_{experiment_name}_recovered",
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

        # Reconstruct today's realized R / halt state so a same-day restart
        # can't silently undo an already-tripped daily-loss kill switch.
        self._recover_daily_risk_state(now)

        # Load open counterfactual positions from DB on startup
        open_cfs = self.db.get_open_counterfactuals()
        for op in open_cfs:
            cand_id = op['candidate_id']
            symbol = op['symbol']
            
            # Prevent prior-day state leakage (Self-Healing)
            entry_time = op['timestamp']
            if entry_time.date() < now.date():
                # Force-close the counterfactual position in the database at 15:25 on its entry day
                exit_time = entry_time.replace(hour=15, minute=25, second=0, microsecond=0)
                op_exit = dict(op)
                op_exit['exit_time'] = exit_time
                op_exit['exit_price'] = op['entry_price']
                op_exit['final_pnl_r'] = -0.05  # Transaction cost buffer only
                op_exit['exit_reason'] = 'SESSION_END'
                op_exit['valid'] = False
                op_exit['validation_errors'] = "Orphaned recovery: closed on next startup."
                
                # Convert serializable types
                if 'rejection_reasons' in op_exit and isinstance(op_exit['rejection_reasons'], str):
                    import json
                    try:
                        op_exit['rejection_reasons'] = json.loads(op_exit['rejection_reasons'])
                    except Exception:
                        op_exit['rejection_reasons'] = []
                        
                self.db.save_counterfactual_result(op_exit)
                logger.info(f"🧹 Self-Healed orphaned prior-day CF position: {cand_id} entered on {entry_time.date()}")
                continue

            self.active_counterfactuals[cand_id] = {
                'candidate_id': cand_id,
                'symbol': symbol,
                'experiment_name': op.get('experiment_name', 'Structural_v3.2_RVOL1.0'),
                'signal': op['signal_type'],
                'entry_price': op['entry_price'],
                'entry_time': op['timestamp'],
                'stop_loss': op['stop_loss'],
                'take_profit': op['take_profit'],
                'tp1': op.get('tp1'),
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
                'status': 'OPEN',
                'rejection_reasons': op.get('rejection_reasons', []),
                'confidence': op.get('confidence'),
                'diagnostics': op.get('diagnostics')
            }
            # Rebuild thesis deduplication index from recovered positions - key order matches market_loop check
            exp_name = op.get('experiment_name', 'Structural_v3.2_RVOL1.0')
            thesis_key = (exp_name, symbol, op['setup_type'], op['signal_type'])
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

        # Recover open combo positions (real + CF) — same self-healing rule as
        # single-leg: a combo still open from a prior calendar day is force-closed
        # flat rather than trusted, since we can't reconstruct what each leg's
        # premium actually did while the process was down.
        open_combos = self.db.get_open_combo_positions()
        for op in open_combos:
            key = (op['symbol'], op.get('experiment_name', ''))
            entry_time = op['entry_time']
            if entry_time.date() < now.date():
                op_exit = dict(op)
                op_exit['exit_time'] = entry_time.replace(hour=15, minute=25, second=0, microsecond=0)
                op_exit['underlying_exit_price'] = op.get('underlying_entry_price')
                op_exit['final_pnl_r'] = -0.05
                op_exit['exit_reason'] = 'SESSION_END'
                op_exit['valid'] = False
                op_exit['validation_errors'] = "Orphaned recovery: closed on next startup."
                self.db.save_combo_trade(op_exit)
                logger.info(f"🧹 Self-Healed orphaned prior-day combo: {op['combo_id']} entered on {entry_time.date()}")
                continue
            combo_dict = dict(op)
            combo_dict['status'] = 'OPEN'
            self.active_combo_trades[key] = combo_dict
        if open_combos:
            logger.info(f"🔄 Recovered {len(open_combos)} active real combo positions: {list(self.active_combo_trades.keys())}")

        open_cf_combos = self.db.get_open_counterfactual_combos()
        for op in open_cf_combos:
            entry_time = op['entry_time']
            if entry_time.date() < now.date():
                op_exit = dict(op)
                op_exit['exit_time'] = entry_time.replace(hour=15, minute=25, second=0, microsecond=0)
                op_exit['underlying_exit_price'] = op.get('underlying_entry_price')
                op_exit['final_pnl_r'] = -0.05
                op_exit['exit_reason'] = 'SESSION_END'
                op_exit['valid'] = False
                op_exit['validation_errors'] = "Orphaned recovery: closed on next startup."
                self.db.save_counterfactual_combo_result(op_exit)
                logger.info(f"🧹 Self-Healed orphaned prior-day CF combo: {op['combo_id']} entered on {entry_time.date()}")
                continue
            combo_dict = dict(op)
            combo_dict['status'] = 'OPEN'
            self.active_cf_combos[op['combo_id']] = combo_dict
            exp_name = op.get('experiment_name', '')
            thesis_base = (op['symbol'], op.get('setup_type', ''), op.get('combo_type', ''))
            self.active_cf_combo_theses[(exp_name,) + thesis_base] = op['combo_id']
        if open_cf_combos:
            logger.info(f"🔄 Recovered {len(open_cf_combos)} active CF combo positions: {list(self.active_cf_combos.keys())}")

        logger.info("🏛️ Structural Paper Trader Initialized | Active Position Tracking Enabled")

    def market_loop(self):
        """Main loop to be run every 5 minutes during market hours."""
        now = datetime.now(self.tz)

        # Only run between 09:00 and 15:59 IST
        if not (9 <= now.hour < 16):
            return

        self.position_lock.acquire()
        try:
            # Reset daily risk counters at the first pulse of a new day
            self._roll_risk_day(now)

            logger.info(f"--- {now.strftime('%H:%M:%S')} Market Pulse ---")

            # Fetch live prices in batch for accurate position evaluation
            live_prices = self.data_provider.get_current_prices_batch(self.symbols)
            self._record_quote_batch_result(live_prices, now)

            # 1. Fetch Multi-Timeframe Data (once per symbol)
            end_date = datetime.now(self.tz)
            start_date_d1 = end_date - timedelta(days=40)
            start_date_h1 = end_date - timedelta(days=10)
            start_date_m5 = end_date - timedelta(days=5)

            current_prices = {}
            current_bars = {}  # symbol -> last CLOSED m5 OHLC (for intrabar SL/TP)
            fetched = {}  # symbol -> (d1, h1, m5)

            for symbol in self.symbols:
                d1 = self.data_provider.get_historical_data(symbol, start_date_d1, end_date, "1D")
                h1 = self.data_provider.get_historical_data(symbol, start_date_h1, end_date, "60")
                m5 = self.data_provider.get_historical_data(symbol, start_date_m5, end_date, "5")
                if d1 is not None and h1 is not None and m5 is not None and len(m5) >= 2:
                    fetched[symbol] = (d1, h1, m5)
                    # iloc[-1] is the still-forming candle; mark positions and check
                    # stops on the last fully CLOSED candle (iloc[-2]) so live and
                    # backtest agree on the decision/mark bar.
                    closed = m5.iloc[-2]
                    
                    # Use live LTP if available and fresh, otherwise fall back to closed candle close
                    live_ltp = live_prices.get(symbol)
                    if live_ltp is not None:
                        current_prices[symbol] = live_ltp
                    else:
                        current_prices[symbol] = float(closed['close'])
                        logger.warning(f"⚠️ Live LTP quote for {symbol} not available in market_loop; falling back to closed candle close: {current_prices[symbol]}")

                    current_bars[symbol] = {
                        'open': float(closed['open']),
                        'high': float(closed['high']),
                        'low': float(closed['low']),
                        'close': float(closed['close']),
                    }
                else:
                    logger.warning(f"⚠️ Could not fetch complete MTF data for {symbol}")

            # 2. Update Active Trades & Counterfactuals
            self._update_active_trades(current_prices, now, current_bars)

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

            # 3b. Expiry / event blackout gate (Bug 18 fix). Re-enabled 2026-08-08
            # with include_lunch_hour=False: the lunch-hour check blocked ~2 hours
            # of EVERY session (not just expiry/event days), which is why the
            # entire gate got disabled on 2026-07-30 instead of just that piece.
            # Weekly/monthly expiry + RBI/Budget windows are genuine tail-risk
            # protection and stay on; lunch-hour liquidity filtering (if wanted)
            # belongs in a separate, dedicated filter, not bundled here.
            is_blackout, blackout_reason = self.expiry_blackout.is_blackout(include_lunch_hour=False)
            if is_blackout:
                logger.info(f"🚫 Expiry/Event blackout active — no new entries. Reason: {blackout_reason}")
                # Positions were already updated in step 2 above; do NOT update
                # again here. The old double-call inflated bars_held / duration /
                # holding_efficiency on every blackout candle (lunch 11:30–13:30
                # daily + all Thursdays), corrupting those metrics for most trades.
                return

            # 4. Compute snapshot + run experiments per symbol
            total_signals = 0
            for symbol, (d1, h1, m5) in fetched.items():
                # Signals must decide on the last CLOSED 5m candle, never the
                # still-forming one. The loop runs ~5s into a new candle, and the
                # Fyers feed returns that in-progress bar as the last row (RVOL≈0,
                # partial OHLC). Dropping it makes live decide on the same bar the
                # backtester does (which only ever sees closed bars), eliminating
                # the live/backtest divergence. current_prices[symbol] (the live
                # LTP) is still used for open-position SL/TP marking above.
                m5_closed = m5.iloc[:-1] if len(m5) > 1 else m5
                if len(m5_closed) < 1:
                    logger.warning(f"⚠️ No closed 5m candle for {symbol}, skipping signals")
                    continue
                price = float(m5_closed['close'].iloc[-1])

                snapshot = self.pipeline.compute(symbol, price, d1, h1, m5_closed, now)
                if snapshot is None:
                    logger.warning(f"⚠️ Pipeline returned None for {symbol}")
                    continue

                # Cache for _update_position()'s exit-management checks next tick —
                # by the time _update_active_trades() runs (step 2, above this loop),
                # this candle's snapshot doesn't exist yet.
                self._last_snapshot[symbol] = snapshot

                # Publish current market state for the dashboard — bias, regime,
                # RVOL/ATR/efficiency, S/R zones, in-progress chart patterns.
                # Previously this only ever existed in-memory for this one candle.
                try:
                    self.db.upsert_market_state(self._build_market_state_record(snapshot))
                    # Persist high-quality zones into the durable sr_zones table.
                    # Unlike market_state.zones (overwritten every candle), sr_zones
                    # accumulates touch counts and survives across sessions.
                    self._persist_sr_zones(snapshot)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to build/save market_state for {symbol}: {e}")


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

                        # Stamp regime context onto every signal from every
                        # strategy (not just the ones that set it themselves —
                        # 16 of 17 strategies never did, which is why regime-based
                        # position sizing was silently always falling through to
                        # the UNKNOWN default). Single point of truth: snapshot.
                        sig_features = sig.setdefault('features', {})
                        sig_features['regime_primary'] = snapshot.regime_detail.primary
                        sig_features['regime_vol_state'] = snapshot.regime_detail.vol_state
                        sig_features['market_regime'] = snapshot.regime_detail.label

                        # Suffix candidate_id with experiment_name to isolate counterfactual positions
                        if 'candidate_id' in sig and sig['candidate_id']:
                            if not sig['candidate_id'].endswith(f"_{experiment_name}"):
                                sig['candidate_id'] = f"{sig['candidate_id']}_{experiment_name}"

                        # Multi-leg combo signals (vertical spreads, straddle/strangle)
                        # take a completely separate path — combined-premium PnL,
                        # not a single directional index R-multiple.
                        if 'combo_legs' in sig:
                            self._handle_combo_signal(sig, now, symbol, experiment_name, trade_key, result.experiment_name, snapshot.regime_detail)
                            continue

                        if sig['accepted']:
                            # Regime router: is this experiment eligible for REAL
                            # capital in the current regime? If not, the signal
                            # isn't discarded — it's routed into the CF path
                            # below, tagged REGIME_MISMATCH, so its outcome stays
                            # measurable (same "same engine, different storage"
                            # philosophy as every other rejection).
                            if not is_regime_eligible(result.experiment_name, snapshot.regime_detail):
                                sig['rejection_reasons'] = sig.get('rejection_reasons', []) + ['REGIME_MISMATCH']
                                sig['regime_at_decision'] = snapshot.regime_detail.label
                                logger.info(
                                    f"🧭 [{experiment_name}] Regime router: {symbol} blocked from REAL "
                                    f"capital (regime={snapshot.regime_detail.label}) — routed to CF"
                                )
                                self._enter_counterfactual(sig, now, symbol, experiment_name, trade_key, result, snapshot=snapshot)
                                continue

                            # One real trade per (symbol, experiment_name)
                            if trade_key in self.active_trades:
                                logger.debug(f"↩️  [{experiment_name}] Already have open position on {symbol}, skipping.")
                                continue
                            # Aggregate risk gate (daily-loss halt / concurrency / exposure / woodchopper)
                            can_enter, gate_reason = self._can_enter_real(now, sig)
                            if not can_enter:
                                logger.warning(
                                    f"⛔ [{experiment_name}] Real entry on {symbol} blocked by risk governor: {gate_reason}"
                                )
                                self.db.save_risk_governor_block({
                                    'block_id': f"blk_{symbol.replace(':', '_').replace('-', '_')}_{experiment_name}_{int(now.timestamp())}",
                                    'timestamp': now,
                                    'symbol': symbol,
                                    'experiment_name': experiment_name,
                                    'setup_type': sig.get('strategy'),
                                    'signal_type': sig.get('signal'),
                                    'candidate_id': sig.get('candidate_id'),
                                    'gate_reason': gate_reason,
                                    'entry_price': sig.get('price'),
                                    'stop_loss': sig.get('stop_loss'),
                                    'take_profit': sig.get('take_profit'),
                                    'rr_ratio': sig.get('rr_ratio'),
                                })
                                continue
                            logger.info(f"🚀 SIGNAL: {symbol} {sig['signal']} | [{experiment_name}]")
                            logger.info(f"   Entry: {sig['price']} | SL: {sig['stop_loss']} | TP: {sig['take_profit']} (RR: {sig['rr_ratio']})")
                            self._enter_position(sig, now, trade_key, is_counterfactual=False, snapshot=snapshot)
                            self._record_level_attempt(sig)   # woodchopper counter
                            self.portfolios.on_entry(experiment_name, now)

                        else:
                            self._enter_counterfactual(sig, now, symbol, experiment_name, trade_key, result, snapshot=snapshot)

            if total_signals == 0:
                logger.info("🧘 Status: Sidelined (No Institutional Alignment)")

        except Exception as e:
            logger.error(f"❌ Error in market loop: {e}", exc_info=True)
        finally:
            self.position_lock.release()

    def position_tracking_loop(self):
        """Lightweight loop to check SL/TP and exit positions in real time (every 30s)."""
        now = datetime.now(self.tz)
        
        # Only run between 09:00 and 15:59 IST
        if not (9 <= now.hour < 16):
            return
            
        with self.position_lock:
            # Check if we have any active trades to update (avoid API overhead if book is empty)
            has_active = (
                len(self.active_trades) > 0 or 
                len(self.active_counterfactuals) > 0 or 
                len(self.active_combo_trades) > 0 or 
                len(self.active_cf_combos) > 0
            )
            if not has_active:
                return
                
            logger.info(f"⏱️ [Real-time Exit Loop] Checking open positions @ {now.strftime('%H:%M:%S')}...")
            
            try:
                # Fetch live LTP in batch
                live_prices = self.data_provider.get_current_prices_batch(self.symbols)
                self._record_quote_batch_result(live_prices, now)

                # Filter out None/failed quotes
                valid_prices = {}
                for symbol in self.symbols:
                    price = live_prices.get(symbol)
                    if price is not None and price > 0:
                        valid_prices[symbol] = price
                    else:
                        logger.warning(f"⚠️ Live LTP quote for {symbol} not available or invalid: {price}")
                
                # If all quotes failed, do NOT update or trigger exits (data sanity guard)
                if not valid_prices:
                    logger.warning("❌ All symbol quotes failed or invalid — skipping position tracking loop")
                    return
                
                # Update open trades using these live prices without passing candle bar objects
                # and with increment_bar_count=False to avoid modifying bars_held
                self._update_active_trades(valid_prices, now, current_bars=None, increment_bar_count=False)
                
            except Exception as e:
                logger.error(f"❌ Error in position_tracking_loop: {e}", exc_info=True)

    def _build_market_state_record(self, snapshot) -> Dict:
        """Flatten one MarketSnapshot into the plain dict upsert_market_state() persists."""
        zones = sorted(snapshot.h1_zones or [], key=lambda z: z.score, reverse=True)[:8]
        zones_json = [
            {
                'level': round(float(z.level), 2),
                'type': z.zone_type,
                'score': round(float(z.score), 1),
                'rejection_count': z.rejection_count,
                'freshness': round(float(z.freshness), 1),
            }
            for z in zones
        ]

        patterns_json = []
        patterns_ctx = getattr(snapshot.market, "patterns", None)
        if patterns_ctx:
            live_states = {"FORMING", "ACTIVE", "READY", "BREAKOUT", "CONFIRMED"}
            for p in patterns_ctx.patterns:
                if p.state.value in live_states:
                    patterns_json.append({
                        'type': p.type.value,
                        'state': p.state.value,
                        'direction': p.direction.value,
                        'confidence': round(p.confidence, 3),
                        'completion_pct': round(p.completion_pct, 3),
                        'breakout_level': round(float(p.breakout_level), 2),
                        'invalidation': round(float(p.current_invalidation), 2),
                        'targets': [round(float(t), 2) for t in p.targets],
                    })

        geo = getattr(snapshot.market, "geometry", None)
        narrative = getattr(geo, "narrative", None) if geo else None

        return {
            'symbol': snapshot.symbol,
            'updated_at': snapshot.timestamp,
            'current_price': snapshot.current_price,
            'daily_bias': snapshot.daily_bias,
            'market_regime': snapshot.market_regime,
            'rvol': getattr(snapshot.volume_report, 'rvol_tod', None),
            'atr': snapshot.features.get_float("atr"),
            'move_efficiency': snapshot.features.get_float("move_efficiency"),
            'wickiness': snapshot.features.get_float("wickiness"),
            'narrative_bias': narrative.bias.value if narrative else None,
            'narrative_confidence': round(narrative.bias_confidence, 3) if narrative else None,
            'zones': zones_json,
            'patterns': patterns_json,
        }

    def _deployed_capital(self) -> float:
        """Sum of notional currently deployed across OPEN real trades (CFs excluded).

        Includes combo/multi-leg positions (max_loss × lots is their capital-at-risk) —
        previously only single-leg active_trades was counted, so combo capital was
        invisible to MAX_PORTFOLIO_EXPOSURE.
        """
        single_leg = sum(float(p.get('position_size_inr', 0.0)) for p in self.active_trades.values())
        combo = sum(float(p.get('max_loss', 0.0)) * float(p.get('lots', 1)) for p in self.active_combo_trades.values())
        return single_leg + combo

    def _short_vol_group_deployed_capital(self) -> float:
        """Combined capital-at-risk across all currently open short-vol/theta-selling
        combo positions (see position_sizer.SHORT_VOL_GROUP_EXPERIMENTS) — these all
        want the same thing (low realized movement) and don't diversify each other."""
        from src.core.position_sizer import SHORT_VOL_GROUP_EXPERIMENTS
        return sum(
            float(p.get('max_loss', 0.0)) * float(p.get('lots', 1))
            for p in self.active_combo_trades.values()
            if p.get('experiment_name') in SHORT_VOL_GROUP_EXPERIMENTS
        )

    def _persist_sr_zones(self, snapshot) -> None:
        """Upsert high-quality zones from a MarketSnapshot into the persistent sr_zones table.

        Only persists zones with score >= 2.0 to avoid populating the table with
        marginal / noise-level zones.  Zone IDs are deterministic hashes, so the
        same zone across multiple candles increments touch_count rather than
        inserting duplicate rows.
        """
        import hashlib
        from datetime import timezone as _tz

        now = snapshot.timestamp or datetime.now(_tz.utc)
        atr = float(snapshot.features.get_float("atr") or 100.0)

        # Persist zones from all three timeframes. Timeframe is baked into the
        # zone_id bucket (and the timeframe column) so an m5 zone and a d1 zone
        # at a similar price stay distinct rows with their own touch_count —
        # they're different-strength signals, not the same zone re-detected.
        for zones, tf in (
            (snapshot.h1_zones, "h1"),
            (getattr(snapshot, "m5_zones", None), "m5"),
            (getattr(snapshot, "d1_zones", None), "d1"),
        ):
            for z in (zones or []):
                try:
                    score = float(getattr(z, "score", 0.0))
                    if score < 2.0:
                        continue

                    level = float(z.level)
                    zone_type = z.zone_type          # 'SUPPLY' or 'DEMAND'
                    # half-ATR bucket so nearby zones share the same ID
                    bucket = round(level / (atr * 0.5)) * int(atr * 0.5)

                    raw_id = f"{snapshot.symbol}|{zone_type}|{tf}|{bucket}"
                    zone_id = "z_" + hashlib.sha1(raw_id.encode()).hexdigest()[:12]

                    # price band: ±0.25 ATR around the level
                    half_band = atr * 0.25
                    self.db.upsert_sr_zone({
                        "zone_id":    zone_id,
                        "symbol":     snapshot.symbol,
                        "zone_type":  zone_type,
                        "price_low":  round(level - half_band, 2),
                        "price_high": round(level + half_band, 2),
                        "strength":   score,
                        "now":        now,
                        "timeframe":  tf,
                    })
                except Exception as e:
                    logger.debug(f"⚠️ _persist_sr_zones: skipping {tf} zone — {e}")

        # OI walls (from real option-chain data, see options_intelligence_engine) —
        # persisted using the OI_RESISTANCE/OI_SUPPORT zone_type this table already
        # supports, so strike-based S/R shows up alongside price-action zones.
        # Skipped entirely when options data is missing/stale rather than persisting
        # a guess.
        options = getattr(snapshot.market, "options", None) if snapshot.market else None
        if options is not None and not options.is_stale:
            try:
                # NSE only ever lists NIFTY strikes as multiples of 50 and
                # BANKNIFTY strikes as multiples of 100, so relevance_factor's
                # "else" tier below is a defensive fallback, not something a
                # real strike hits — for BANKNIFTY every strike is %100==0, so
                # relevance_factor is always 1.0 there; the 100/50 split only
                # differentiates NIFTY strikes.
                half_band = 5.0 if "BANK" not in snapshot.symbol else 15.0
                oi_wall_zones = []
                for walls, zone_type in (
                    (getattr(options, "call_oi_walls", []), "OI_RESISTANCE"),
                    (getattr(options, "put_oi_walls", []), "OI_SUPPORT"),
                ):
                    for wall in (walls or []):
                        if wall is None or wall.strike is None:
                            continue

                        strike = float(wall.strike)

                        if strike % 100 == 0:
                            relevance_factor = 1.0
                        elif strike % 50 == 0:
                            relevance_factor = 0.5
                        else:
                            relevance_factor = 0.1

                        raw_id = f"{snapshot.symbol}|{zone_type}|{strike}"
                        zone_id = "z_" + hashlib.sha1(raw_id.encode()).hexdigest()[:12]

                        # Heuristic normalization: 1 lakh OI ~= strength 100, scaled by relevance factor
                        base_strength = min(wall.oi / 100_000.0 * 100.0, 100.0)
                        strength = base_strength * relevance_factor

                        oi_wall_zones.append({
                            "zone_id":    zone_id,
                            "symbol":     snapshot.symbol,
                            "zone_type":  zone_type,
                            "price_low":  strike - half_band,
                            "price_high": strike + half_band,
                            "strength":   round(strength, 1),
                            "now":        now,
                            "timeframe":  "options",
                        })
                # One connection for up to ~20 walls, not one per wall.
                self.db.upsert_sr_zones_batch(oi_wall_zones)
            except Exception as e:
                logger.debug(f"⚠️ _persist_sr_zones: skipping OI wall — {e}")


    def _recover_daily_risk_state(self, now):
        """Reconstruct daily_realized_r / trading_halted_today from the DB on startup.

        Without this, a same-day crash-restart always sees self._risk_day is None
        and _roll_risk_day() unconditionally zeroes the counters — silently
        re-enabling new real entries even if the daily loss limit had already
        tripped earlier that day.
        """
        today = now.strftime('%Y-%m-%d')
        self._risk_day = today
        self.daily_realized_r = self.db.get_realized_r_today(today)
        self.trading_halted_today = self.daily_realized_r <= self.DAILY_LOSS_LIMIT_R
        if self.trading_halted_today:
            logger.critical(
                f"🛑 Restart recovered an ALREADY-TRIPPED daily loss limit: "
                f"realized {self.daily_realized_r:.2f}R <= {self.DAILY_LOSS_LIMIT_R}R. "
                f"New real entries remain halted for {today}."
            )
        elif self.daily_realized_r != 0.0:
            logger.info(f"🗓️ Recovered daily realized R: {self.daily_realized_r:+.2f}R for {today}.")

    def _roll_risk_day(self, now):
        """Reset the daily risk counters at the first pulse of a new trading day."""
        today = now.strftime('%Y-%m-%d')
        if self._risk_day != today:
            self._risk_day = today
            self.daily_realized_r = 0.0
            self.trading_halted_today = False
            self._daily_level_attempts = {}  # reset woodchopper counters for new day
            logger.info(f"🗓️ Risk day rolled to {today} — daily counters reset.")

    def _record_quote_batch_result(self, prices: Dict[str, Optional[float]], now) -> None:
        """Circuit breaker: track consecutive all-quotes-failed ticks and halt
        new real entries once sustained failure looks like a genuine outage
        rather than one transient blip. Auto-clears the moment quotes succeed
        again — existing open positions keep being marked whenever a quote
        does come back, only NEW entries are gated (via _can_enter_real)."""
        success = any(v is not None and v > 0 for v in prices.values())
        if success:
            if self.data_feed_halted:
                logger.info("✅ Live quotes recovered — clearing DATA_FEED_DOWN halt.")
                self.db.resolve_system_alert("DATA_FEED_DOWN", now)
            self._quote_failure_streak = 0
            self.data_feed_halted = False
            return

        self._quote_failure_streak += 1
        if self._quote_failure_streak >= self.QUOTE_FAILURE_HALT_THRESHOLD and not self.data_feed_halted:
            self.data_feed_halted = True
            msg = (
                f"All symbol quotes failed for {self._quote_failure_streak} consecutive ticks "
                f"(~{self._quote_failure_streak * 30}s) — halting new real entries until the feed recovers."
            )
            logger.critical(f"🛑 CIRCUIT BREAKER TRIPPED: {msg}")
            self.db.save_system_alert(now, "DATA_FEED_DOWN", msg)

    def _can_enter_real(self, now, sig: Dict = None) -> Tuple[bool, str]:
        """Aggregate risk gate for NEW real entries. Returns (allowed, reason)."""
        if self.data_feed_halted:
            return False, "DATA_FEED_DOWN"
        if self.trading_halted_today:
            return False, "DAILY_LOSS_HALT"
        if self.daily_realized_r <= self.DAILY_LOSS_LIMIT_R:
            self.trading_halted_today = True
            logger.critical(
                f"🛑 DAILY LOSS LIMIT hit: realized {self.daily_realized_r:.2f}R "
                f"<= {self.DAILY_LOSS_LIMIT_R}R. Halting ALL new real entries for the day."
            )
            return False, "DAILY_LOSS_HALT"
        if len(self.active_trades) >= self.MAX_CONCURRENT_REAL:
            return False, "MAX_CONCURRENT"
        if self._deployed_capital() >= self.MAX_DEPLOYED_FRACTION * self.RISK_CAPITAL:
            return False, "MAX_DEPLOYED"

        # Woodchopper protection: block a 3rd real attempt at the same price level
        # + direction today (Aug-03 finding: BANKNIFTY BUY PUT hammered 22 times
        # at the same level over 5 days, −57.4R total on one broken thesis).
        if sig:
            symbol    = sig.get('symbol', '')
            direction = sig.get('signal', '')
            price     = float(sig.get('price') or 0.0)
            # Use a wide ATR-bucket so "same level" = ±0.5×ATR
            atr_raw   = sig.get('diagnostics', {}).get('atr') or sig.get('features', {}).get('atr') or 100.0
            atr       = float(atr_raw)
            bucket    = round(price / max(atr, 1.0)) * int(max(atr, 1.0))
            level_key = (symbol, direction, bucket)
            attempts  = self._daily_level_attempts.get(level_key, 0)
            if attempts >= self.MAX_ATTEMPTS_PER_LEVEL:
                return False, f"LEVEL_REPEAT_CAP({attempts}x@{bucket})"

        return True, "OK"

    def _place_live_entry_order(self, pos: Dict) -> None:
        """Places a REAL market BUY for a real (non-CF) entry. Mutates `pos`
        in place with the outcome; never raises into the caller — a failure
        here must not stop the paper-simulated position from being tracked."""
        symbol = pos.get('option_symbol') or pos['symbol']
        qty = int((pos.get('lots') or 1) * (pos.get('lot_size') or 1))
        try:
            result = self.live_order_executor.place(symbol, qty, side="BUY")
        except Exception as e:
            logger.error(f"❌ Live entry order raised for {symbol} qty={qty}: {e}")
            self._on_live_order_failure(f"ENTRY {symbol} qty={qty}: {e}")
            return

        pos['is_live'] = result.success
        pos['live_order_id'] = result.order_id
        pos['live_fill_price'] = result.fill_price
        if result.success:
            self.live_order_failure_streak = 0
            logger.info(f"✅ LIVE ENTRY: {symbol} qty={qty} order_id={result.order_id} fill={result.fill_price}")
        else:
            self._on_live_order_failure(f"ENTRY {symbol} qty={qty}: {result.message}")

    def _place_live_exit_order(self, pos: Dict, exit_price: float, reason: str) -> None:
        """Places a REAL market SELL to square off the live entry opened above.
        Mutates `pos` in place; never raises into the caller."""
        symbol = pos.get('option_symbol') or pos['symbol']
        qty = int((pos.get('lots') or 1) * (pos.get('lot_size') or 1))
        try:
            result = self.live_order_executor.place(symbol, qty, side="SELL")
        except Exception as e:
            logger.error(f"❌ Live exit order raised for {symbol} qty={qty}: {e}")
            self._on_live_order_failure(f"EXIT {symbol} qty={qty} reason={reason}: {e}")
            return

        pos['live_exit_order_id'] = result.order_id
        pos['live_exit_fill_price'] = result.fill_price
        if result.success:
            self.live_order_failure_streak = 0
            logger.info(f"✅ LIVE EXIT: {symbol} qty={qty} reason={reason} order_id={result.order_id} fill={result.fill_price}")
        else:
            self._on_live_order_failure(f"EXIT {symbol} qty={qty} reason={reason}: {result.message}")

    def _on_live_order_failure(self, message: str) -> None:
        """Circuit breaker for real order placement — stricter than the 5-tick
        data-feed threshold, since an order rejection is a much higher-signal
        failure than one missed quote. Flips live_mode off (falling back to
        paper) rather than crashing the loop, and fires the same system-alert
        mechanism DATA_FEED_DOWN already uses."""
        self.live_order_failure_streak += 1
        logger.error(
            f"❌ Live order failure ({self.live_order_failure_streak}/"
            f"{self.LIVE_ORDER_FAILURE_HALT_THRESHOLD}): {message}"
        )
        if self.live_order_failure_streak >= self.LIVE_ORDER_FAILURE_HALT_THRESHOLD and self.live_mode:
            self.live_mode = False
            reason = f"{self.live_order_failure_streak} consecutive live order failures: {message}"
            logger.critical(f"🛑 LIVE_MODE disabled — reverting to paper. {reason}")
            self.db.save_system_alert(datetime.now(self.tz), "LIVE_ORDER_HALTED", reason)
            self._send_circuit_breaker_alert(reason)

    def _send_circuit_breaker_alert(self, reason: str) -> None:
        """Sync bridge into the (currently unused elsewhere) async AlertManager —
        the trading loop is sync, so this wraps the one async call it needs in
        asyncio.run(). Never raises; a failed alert must not affect trading."""
        try:
            import asyncio
            from src.execution.monitoring_alerting_system import AlertManager, AlertConfig
            from src.config import settings as _settings
            if not hasattr(self, '_alert_manager'):
                self._alert_manager = AlertManager(AlertConfig(
                    enable_email=False,
                    enable_slack=False,
                    enable_webhook=False,
                    enable_telegram=bool(_settings.TELEGRAM_BOT_TOKEN and _settings.TELEGRAM_CHAT_ID),
                    telegram_bot_token=_settings.TELEGRAM_BOT_TOKEN or "",
                    telegram_chat_id=_settings.TELEGRAM_CHAT_ID or "",
                ))
            asyncio.run(self._alert_manager.alert_circuit_breaker_activated(reason))
        except Exception as e:
            logger.error(f"❌ Failed to send live-order circuit breaker alert: {e}")

    def _record_level_attempt(self, sig: Dict) -> None:
        """Increment the woodchopper attempt counter for a real trade that was entered."""
        symbol    = sig.get('symbol', '')
        direction = sig.get('signal', '')
        price     = float(sig.get('price') or 0.0)
        atr_raw   = sig.get('diagnostics', {}).get('atr') or sig.get('features', {}).get('atr') or 100.0
        atr       = float(atr_raw)
        bucket    = round(price / max(atr, 1.0)) * int(max(atr, 1.0))
        level_key = (symbol, direction, bucket)
        self._daily_level_attempts[level_key] = self._daily_level_attempts.get(level_key, 0) + 1
        logger.debug(
            f"🧩 Level attempt #{self._daily_level_attempts[level_key]} "
            f"for {symbol} {direction} @ bucket {bucket}"
        )

    def _update_active_trades(self, current_prices: Dict[str, float], timestamp, current_bars: Dict = None, increment_bar_count: bool = True):
        """Evaluate open positions against latest market prices.

        Each position is updated inside its own try/except so that one malformed
        position (e.g. a bad field after a schema change) cannot abort SL/TP
        management for the rest of the book — previously an exception here
        unwound the whole loop and left every remaining position unmanaged.
        """
        with self.position_lock:
            current_bars = current_bars or {}

            # One batch quote call for every open position's resolved option
            # leg, instead of one call per position per tick (see
            # _batch_resolve_option_quotes / _premium_pnl_r).
            option_symbols = {
                pos['option_symbol'] for pos in self.active_trades.values() if pos.get('option_symbol')
            } | {
                pos['option_symbol'] for pos in self.active_counterfactuals.values() if pos.get('option_symbol')
            }
            option_quotes = self._batch_resolve_option_quotes(list(option_symbols))

            # Update real trades — keyed by (symbol, experiment_name)
            for key in list(self.active_trades.keys()):
                symbol, experiment_name = key
                if symbol not in current_prices:
                    continue
                try:
                    pos = self.active_trades[key]
                    is_closed = self._update_position(
                        pos, current_prices[symbol], timestamp, bar=current_bars.get(symbol),
                        increment_bar_count=increment_bar_count,
                        option_quote=option_quotes.get(pos.get('option_symbol')),
                        live_snapshot=self._last_snapshot.get(symbol),
                    )
                    if is_closed:
                        pnl_r = pos.get('_last_pnl_r', 0.0)
                        # Accrue realized R for the daily-loss kill switch
                        self.daily_realized_r += pnl_r
                        self.portfolios.on_exit(experiment_name, pnl_r, timestamp)
                        self.active_trades.pop(key)
                except Exception as e:
                    logger.critical(
                        f"🚨 Position update FAILED for real trade {key}: {e}. "
                        f"Position left open and quarantined for manual review.",
                        exc_info=True,
                    )

            # Update counterfactual trades — keyed by candidate_id (unchanged)
            for cand_id in list(self.active_counterfactuals.keys()):
                try:
                    pos = self.active_counterfactuals[cand_id]
                    symbol = pos['symbol']
                    if symbol not in current_prices:
                        continue
                    is_closed = self._update_position(
                        pos, current_prices[symbol], timestamp, bar=current_bars.get(symbol),
                        increment_bar_count=increment_bar_count,
                        option_quote=option_quotes.get(pos.get('option_symbol')),
                        live_snapshot=self._last_snapshot.get(symbol),
                    )
                except Exception as e:
                    logger.error(f"⚠️ Position update failed for counterfactual {cand_id}: {e}", exc_info=True)
                    continue
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

            # Update real combo positions — keyed by (symbol, experiment_name), same
            # dedup contract as active_trades, entirely separate lifecycle logic.
            for key in list(self.active_combo_trades.keys()):
                symbol, experiment_name = key
                if symbol not in current_prices:
                    continue
                try:
                    pos = self.active_combo_trades[key]
                    is_closed = self._update_combo_position(pos, current_prices[symbol], timestamp, is_cf=False)
                    if is_closed:
                        pnl_r = pos.get('_last_pnl_r', 0.0)
                        self.daily_realized_r += pnl_r
                        self.portfolios.on_exit(experiment_name, pnl_r, timestamp)
                        self._notify_strategy_exit(experiment_name, symbol, pnl_r, timestamp)
                        self.active_combo_trades.pop(key)
                except Exception as e:
                    logger.critical(
                        f"🚨 Combo position update FAILED for {key}: {e}. "
                        f"Position left open and quarantined for manual review.",
                        exc_info=True,
                    )

            # Update counterfactual combo positions — keyed by combo_id.
            for combo_id in list(self.active_cf_combos.keys()):
                try:
                    pos = self.active_cf_combos[combo_id]
                    symbol = pos['symbol']
                    if symbol not in current_prices:
                        continue
                    is_closed = self._update_combo_position(pos, current_prices[symbol], timestamp, is_cf=True)
                except Exception as e:
                    logger.error(f"⚠️ Combo position update failed for CF {combo_id}: {e}", exc_info=True)
                    continue
                if is_closed:
                    exp_name = pos.get('experiment_name', '')
                    thesis_base = (pos['symbol'], pos.get('setup_type', ''), pos.get('combo_type', ''))
                    self.active_cf_combo_theses.pop((exp_name,) + thesis_base, None)
                    self._notify_strategy_exit(exp_name, pos['symbol'], pos.get('_last_pnl_r', 0.0), timestamp)
                    self.active_cf_combos.pop(combo_id)

    def _batch_resolve_option_quotes(self, option_symbols: List[str]) -> Dict[str, Tuple[float, float, float]]:
        """One get_quotes() call for every open position's resolved option leg,
        instead of one call per position per 30s tick. Returns
        {option_symbol: (premium, bid, ask)} for whatever the batch call
        actually returned a live LTP for — missing symbols are simply absent,
        never fabricated."""
        if not option_symbols:
            return {}
        try:
            quotes = self.data_provider.client.get_quotes(option_symbols)
        except Exception as e:
            logger.warning(f"⚠️ Batch option quote fetch failed: {e}")
            return {}
        result = {}
        if quotes and isinstance(quotes, list):
            symbol_set = set(option_symbols)
            for q in quotes:
                name = q.get('n')
                if name in symbol_set:
                    v = q.get('v', {})
                    ltp = v.get('lp')
                    if ltp is not None:
                        result[name] = (float(ltp), float(v.get('bid', 0.0) or 0.0), float(v.get('ask', 0.0) or 0.0))
        return result

    def _resolve_current_premium(self, pos: Dict, option_quote: Optional[Tuple[float, float, float]]) -> Optional[float]:
        """Realistic current fill for closing this position's option leg — a
        long CALL/PUT always closes by SELLING, so it fills at bid, never at
        raw LTP (options_execution_engine.realistic_fill_price). Uses the
        pre-batched quote if the caller supplied one, else falls back to a
        single live-quote lookup. Returns None (never fabricates a price) if
        there's no option leg or no quote could be resolved."""
        option_symbol = pos.get('option_symbol')
        if not option_symbol:
            return None
        if option_quote is not None:
            premium, bid, ask = option_quote
        else:
            try:
                quotes = self.data_provider.client.get_quotes([option_symbol])
            except Exception as e:
                logger.warning(f"⚠️ Could not resolve current premium for {option_symbol}: {e}")
                return None
            premium = bid = ask = None
            if quotes and isinstance(quotes, list):
                for q in quotes:
                    if q.get('n') == option_symbol:
                        v = q.get('v', {})
                        premium, bid, ask = v.get('lp'), v.get('bid', 0.0), v.get('ask', 0.0)
                        break
            if premium is None:
                return None
        from src.core.options_execution_engine import realistic_fill_price
        return realistic_fill_price(float(premium), float(bid or 0.0), float(ask or 0.0), 'SELL')

    def _premium_pnl_r(self, pos: Dict, option_quote: Optional[Tuple[float, float, float]]) -> Optional[float]:
        """Real option-premium P&L expressed in R.

        R is defined as the premium actually paid for the position
        (entry_premium * lot_size * lots) — the standard convention for a long
        option, whose maximum loss IS the premium paid. This bounds R at -1.0
        on the downside and keeps it comparable across trades of different
        sizes, same as the index-point proxy's R is comparable across trades
        of different stop distances.

        Previous formula (fixed 2026-08-18): risk was computed as
        `position_size_inr * stop_loss_distance / entry_price` — the correct
        risk-per-R for a LINEAR instrument (futures/cash), where a small
        index-point move against you costs only that fraction of notional.
        Applied to a leveraged option premium P&L, that denominator is tiny
        (fraction of a percent of notional) while the numerator is the full
        leveraged premium swing, producing R-multiples off by roughly
        entry_price/stop_loss_distance — this is what produced trades logged
        at -50R to +650R in the same session. Never fabricates a price —
        returns None (caller falls back to the index-point proxy) when this
        position has no resolved option leg or the current premium can't be
        resolved."""
        if not pos.get('option_symbol') or not pos.get('entry_premium') or not pos.get('lot_size'):
            return None
        current_premium = self._resolve_current_premium(pos, option_quote)
        if current_premium is None:
            return None
        risk_amount_inr = pos['entry_premium'] * pos['lot_size'] * pos.get('lots', 1.0)
        if risk_amount_inr <= 0:
            return None
        premium_pnl_inr = (current_premium - pos['entry_premium']) * pos['lot_size'] * pos.get('lots', 1.0)
        pos['_last_current_premium'] = current_premium
        return premium_pnl_inr / risk_amount_inr

    def _update_position(self, pos: Dict, current_price: float, timestamp, bar: Dict = None,
                          increment_bar_count: bool = True, option_quote: Tuple[float, float, float] = None,
                          live_snapshot=None) -> bool:
        """Evaluate a position against the latest market tick. Returns True if position exited.

        ``bar`` is the last CLOSED candle's OHLC ({'open','high','low','close'}).
        When present, the stop-loss is evaluated intrabar (against the candle's
        low/high, not just its close) and gap-through the stop is filled at the
        candle open — modelling the worst-case fill instead of assuming a perfect
        fill exactly at the stop price.

        ``option_quote`` is an optional pre-batched (premium, bid, ask) for this
        position's resolved option leg (see _batch_resolve_option_quotes) — when
        the position has one, pnl_r is priced off the real premium fill instead
        of the index-point proxy (see _premium_pnl_r).

        ``live_snapshot`` is the most recently computed MarketSnapshot for this
        symbol (see self._last_snapshot in market_loop()) — always exactly one
        candle stale relative to this tick's mark price, since a fresh snapshot
        for *this* candle doesn't exist yet when positions are marked. None on
        the first candle of a session/after a restart; every check below must
        fall back to legacy behavior in that case, never crash. Only consulted
        when the owning experiment opted into exit_management via `pos['exit_mgmt']`
        — every flag there defaults to off, preserving today's exact behavior
        for the other ~25 experiments that haven't opted in.
        """
        if pos.get('status', 'OPEN') != 'OPEN':
            return False

        symbol = pos['symbol']
        is_cf = pos.get('is_counterfactual', False)
        stop_loss_distance = pos['stop_loss_distance']
        # pos.get(...) — positions restored from the DB on restart (built in
        # __init__ from get_open_positions()) predate this field and won't have
        # it; every flag defaults to off/unlimited, i.e. today's exact behavior.
        exit_mgmt = pos.get('exit_mgmt') or {}

        # Intrabar extremes for stop evaluation (fall back to the mark price).
        bar_high = float(bar['high']) if bar else current_price
        bar_low = float(bar['low']) if bar else current_price
        bar_open = float(bar['open']) if bar else current_price

        # Increment bars held
        if increment_bar_count:
            pos['bars_held'] = pos.get('bars_held', 0) + 1
        
        # Update extremes
        old_highest = pos['highest_price']
        old_lowest = pos['lowest_price']
        pos['highest_price'] = max(old_highest, current_price)
        pos['lowest_price'] = min(old_lowest, current_price)
        
        # Calculate current R PnL — index-point proxy (assumes the option
        # tracked the index point-for-point: delta=1, zero theta/IV decay).
        if pos['signal'] == 'BUY CALL':
            index_pnl_r = (current_price - pos['entry_price']) / stop_loss_distance if stop_loss_distance > 0 else 0.0
        else: # BUY PUT
            index_pnl_r = (pos['entry_price'] - current_price) / stop_loss_distance if stop_loss_distance > 0 else 0.0

        # Prefer the real option-premium fill whenever this position has a
        # resolved option leg — None means no leg / premium unresolvable, in
        # which case we fall back to the index-point proxy above rather than
        # fabricate a price.
        premium_pnl_r = self._premium_pnl_r(pos, option_quote)
        current_pnl_r = premium_pnl_r if premium_pnl_r is not None else index_pnl_r

        pos['max_closed_profit_r'] = max(pos.get('max_closed_profit_r', 0.0), current_pnl_r)

        is_closed = False
        exit_reason = None
        exit_price = current_price

        if pos['signal'] == 'BUY CALL':
            # Check SL breach intrabar (candle low), gap-aware fill at open
            if bar_low <= pos['stop_loss']:
                is_closed = True
                # If the candle OPENED below the stop (gap-down), the real fill is
                # at the open, which is worse than the stop.
                exit_price = min(pos['stop_loss'], bar_open)
                exit_reason = 'STOP_LOSS'
            # Structural invalidation — checked BEFORE TP_EXPANSION/trailing so it
            # can close a trade before price ever reaches the stop; checked AFTER
            # STOP_LOSS since the hard price stop is a non-negotiable risk floor
            # that must never be pre-empted by a qualitative structure judgment.
            elif (exit_mgmt.get('structure_invalidation')
                  and pos.get('structure_invalidation_level') is not None
                  and bar_low <= pos['structure_invalidation_level']):
                is_closed = True
                exit_price = min(pos['structure_invalidation_level'], bar_open)
                exit_reason = 'STRUCTURE_INVALIDATED'
            # Check TP expansion — capture old values before modifying pos
            elif current_price >= pos['take_profit']:
                cap = exit_mgmt.get('tp_expansion_cap')
                regime_ok = (live_snapshot is None) or (
                    live_snapshot.regime_detail.primary in ('STRONG_TREND_UP', 'WEAK_TREND_UP')
                )
                if cap is not None and (pos.get('tp_expansion_count', 0) >= cap or not regime_ok):
                    is_closed = True
                    exit_price = current_price
                    exit_reason = 'TP_EXPANSION_CAPPED'
                else:
                    old_sl, old_tp = pos['stop_loss'], pos['take_profit']
                    pos['take_profit'] = old_tp + stop_loss_distance
                    new_sl = current_price - stop_loss_distance
                    pos['stop_loss'] = max(old_sl, new_sl)
                    pos['tp_expansion_count'] = pos.get('tp_expansion_count', 0) + 1
                    pos['last_tp_expansion_regime'] = live_snapshot.regime_detail.primary if live_snapshot else None
                    self._log_position_update(pos, current_price, timestamp, 'TP_EXPANSION',
                                              old_sl=old_sl, old_tp=old_tp)
            # Check trailing SL
            elif current_price > old_highest:
                old_sl = pos['stop_loss']
                # FIX: Tighten trail step to 0.75× once 1.5R is in the bag
                trail_mult = 0.75 if current_pnl_r >= 1.5 else 1.0
                if exit_mgmt.get('atr_adaptive_trailing') and live_snapshot is not None:
                    current_atr = live_snapshot.features.get_float('atr', pos.get('atr_at_entry') or stop_loss_distance)
                    # k1=1.5: Chandelier-style multiple of CURRENT ATR (replaces the
                    # frozen entry-time R distance). k2=0.5: floor so the trail can
                    # never tighten below half the original entry-risk distance
                    # purely because one candle's ATR momentarily collapsed.
                    trail_distance = max(current_atr * 1.5, stop_loss_distance * 0.5) * trail_mult
                else:
                    trail_distance = stop_loss_distance * trail_mult
                new_sl = current_price - trail_distance
                if new_sl > old_sl:
                    pos['stop_loss'] = new_sl
                    self._log_position_update(pos, current_price, timestamp, 'TRAILING_SL',
                                              old_sl=old_sl, old_tp=None)

        elif pos['signal'] == 'BUY PUT':
            # Check SL breach intrabar (candle high), gap-aware fill at open
            if bar_high >= pos['stop_loss']:
                is_closed = True
                # If the candle OPENED above the stop (gap-up), the real fill is
                # at the open, which is worse than the stop.
                exit_price = max(pos['stop_loss'], bar_open)
                exit_reason = 'STOP_LOSS'
            # Structural invalidation — see BUY CALL branch for priority rationale.
            elif (exit_mgmt.get('structure_invalidation')
                  and pos.get('structure_invalidation_level') is not None
                  and bar_high >= pos['structure_invalidation_level']):
                is_closed = True
                exit_price = max(pos['structure_invalidation_level'], bar_open)
                exit_reason = 'STRUCTURE_INVALIDATED'
            # Check TP expansion — capture old values before modifying pos
            elif current_price <= pos['take_profit']:
                cap = exit_mgmt.get('tp_expansion_cap')
                regime_ok = (live_snapshot is None) or (
                    live_snapshot.regime_detail.primary in ('STRONG_TREND_DOWN', 'WEAK_TREND_DOWN')
                )
                if cap is not None and (pos.get('tp_expansion_count', 0) >= cap or not regime_ok):
                    is_closed = True
                    exit_price = current_price
                    exit_reason = 'TP_EXPANSION_CAPPED'
                else:
                    old_sl, old_tp = pos['stop_loss'], pos['take_profit']
                    pos['take_profit'] = old_tp - stop_loss_distance
                    new_sl = current_price + stop_loss_distance
                    pos['stop_loss'] = min(old_sl, new_sl)
                    pos['tp_expansion_count'] = pos.get('tp_expansion_count', 0) + 1
                    pos['last_tp_expansion_regime'] = live_snapshot.regime_detail.primary if live_snapshot else None
                    self._log_position_update(pos, current_price, timestamp, 'TP_EXPANSION',
                                              old_sl=old_sl, old_tp=old_tp)
            # Check trailing SL
            elif current_price < old_lowest:
                old_sl = pos['stop_loss']
                # FIX: Tighten trail step to 0.75× once 1.5R is in the bag
                trail_mult = 0.75 if current_pnl_r >= 1.5 else 1.0
                if exit_mgmt.get('atr_adaptive_trailing') and live_snapshot is not None:
                    current_atr = live_snapshot.features.get_float('atr', pos.get('atr_at_entry') or stop_loss_distance)
                    trail_distance = max(current_atr * 1.5, stop_loss_distance * 0.5) * trail_mult
                else:
                    trail_distance = stop_loss_distance * trail_mult
                new_sl = current_price + trail_distance
                if new_sl < old_sl:
                    pos['stop_loss'] = new_sl
                    self._log_position_update(pos, current_price, timestamp, 'TRAILING_SL',
                                              old_sl=old_sl, old_tp=None)

        # Time stop — closes stagnant trades that have neither hit stop/target
        # nor moved meaningfully in either direction within the bar budget.
        if (not is_closed and exit_mgmt.get('time_stop_bars') is not None
                and pos.get('bars_held', 0) >= exit_mgmt['time_stop_bars']
                and abs(current_pnl_r) < exit_mgmt.get('time_stop_min_r', 0.3)):
            is_closed = True
            exit_price = current_price
            exit_reason = 'TIME_STOP'

        # Session force exit check (15:25 PM IST)
        if not is_closed and timestamp.hour == 15 and timestamp.minute >= 25:
            is_closed = True
            exit_price = current_price
            exit_reason = 'SESSION_END'

        if is_closed:
            pos['status'] = 'EXIT_PENDING'
            if premium_pnl_r is not None:
                # Real fill: reuse this tick's resolved premium as the exit fill.
                # (There's no historical option-bar feed to pin the exact
                # index-stop-cross moment — the latest available quote is the
                # best available approximation, same convention combo legs use.)
                pnl_r = premium_pnl_r
                pos['exit_premium'] = pos.pop('_last_current_premium', None)
                pos['pnl_calculation_method'] = 'premium'
                # No separate transaction-cost buffer here — realistic_fill_price
                # (sell-at-bid) already prices in the real bid/ask spread.
            else:
                # Index-point proxy — no resolved option leg (or premium
                # unresolvable this tick). Calculate final PnL R-units.
                if pos['signal'] == 'BUY CALL':
                    pnl_r = (exit_price - pos['entry_price']) / stop_loss_distance if stop_loss_distance > 0 else 0.0
                else: # BUY PUT
                    pnl_r = (pos['entry_price'] - exit_price) / stop_loss_distance if stop_loss_distance > 0 else 0.0
                pnl_r -= 0.05  # Transaction cost buffer
                pos['pnl_calculation_method'] = 'index_proxy'
            # Stashed for _exit_position to persist — atr_at_entry was already
            # captured at signal time; this is the live reading at the moment
            # of exit, one candle stale like every other live_snapshot use here.
            pos['_atr_at_exit'] = live_snapshot.features.get_float('atr') if live_snapshot is not None else None
            self._exit_position(pos, exit_price, exit_reason, timestamp, pnl_r)
            pos['status'] = 'CLOSED'
            return True

        # Live dashboard heartbeat — refresh current price/PnL for still-open
        # REAL positions every candle, not just when a trail/expansion happens,
        # so the DB row doesn't go stale between those events.
        if not is_cf:
            mfe_r, mae_r = self._excursions(pos)
            self.db.update_live_heartbeat(
                trade_id=pos['trade_id'],
                current_price=current_price,
                unrealized_pnl_r=current_pnl_r,
                mfe_r=mfe_r,
                mae_r=mae_r,
                bars_held=pos['bars_held'],
                stop_loss=pos['stop_loss'],
                take_profit=pos['take_profit'],
                timestamp=timestamp,
            )

        return False

    @staticmethod
    def _excursions(pos: Dict) -> Tuple[float, float]:
        """Current (mfe_r, mae_r) for an open position from its recorded extremes."""
        dist = pos['stop_loss_distance']
        if dist <= 0:
            return 0.0, 0.0
        entry = pos['entry_price']
        highest, lowest = pos['highest_price'], pos['lowest_price']
        if pos['signal'] == 'BUY CALL':
            return (highest - entry) / dist, (entry - lowest) / dist
        return (entry - lowest) / dist, (highest - entry) / dist

    def _enter_counterfactual(self, sig: Dict, now, symbol: str, experiment_name: str,
                               trade_key: Tuple, result, snapshot=None) -> bool:
        """Shared CF-entry path: dedup by thesis key, then open a shadow
        position via the exact same _update_position() engine real trades use.
        Used both for strategy-rejected signals and for signals the regime
        router blocked from real capital (see the main signal loop) — either
        way, the candidate isn't discarded, just researched instead of funded.
        Returns True if a CF position was opened (or attempted), False if
        skipped (safety limit / duplicate thesis).
        """
        MAX_ACTIVE_COUNTERFACTUALS = 500
        if len(self.active_counterfactuals) >= MAX_ACTIVE_COUNTERFACTUALS:
            logger.warning(f"⚠️ CF safety limit reached, skipping {symbol}")
            return False

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
            return False

        logger.info(
            f"👻 [{experiment_name}] CF {sig.get('strategy','')} {symbol} {sig.get('signal','')} "
            f"| Rejected: {sig['rejection_reasons']}"
        )
        self._enter_position(sig, now, trade_key, is_counterfactual=True, snapshot=snapshot)
        return True

    def _resolve_exit_mgmt_params(self, experiment_name: str) -> Dict:
        """Resolve an experiment's exit_management config, defaulting every
        flag to today's exact behavior (structure_invalidation/atr_adaptive_
        trailing off, tp_expansion_cap/time_stop_bars disabled). Frozen onto
        `pos` at entry so a position's exit behavior never changes mid-trade
        even if the experiment's registered params are edited later.
        """
        exp = self.registry.get(experiment_name)
        cfg = (exp.params.get('exit_management') if exp and isinstance(exp.params, dict) else None) or {}
        return {
            'structure_invalidation': bool(cfg.get('structure_invalidation', False)),
            'atr_adaptive_trailing': bool(cfg.get('atr_adaptive_trailing', False)),
            'tp_expansion_cap': cfg.get('tp_expansion_cap'),
            'time_stop_bars': cfg.get('time_stop_bars'),
            'time_stop_min_r': cfg.get('time_stop_min_r', 0.3),
        }

    def _enter_position(self, sig: Dict, timestamp, trade_key: Tuple, is_counterfactual: bool, snapshot=None):
        symbol = sig['symbol']
        experiment_name = sig.get('experiment_name', 'Structural_v3.2_RVOL1.0')
        strategy_id = sig.get('strategy_id', 'structural')
        version = sig.get('version', 'v3.2')
        entry_price = sig['price']
        sl_price = sig['stop_loss']
        tp_price = sig['take_profit']
        # Every strategy already computes a 1.5R partial target (tp1) alongside
        # the full take_profit, but it was never persisted or surfaced anywhere
        # — the live dashboard's "multiple targets" view reads it from here.
        tp1_price = sig.get('tp1')
        candidate_id = sig.get('candidate_id')
        # experiment_name is part of the id: multiple experiments can open a
        # real trade on the same symbol in the same candle (they share `timestamp`
        # exactly), and without this suffix they'd collide on trade_id, causing
        # ON CONFLICT (trade_id, entry_time) to silently merge two independent
        # trades into one trade_performance row.
        trade_id = f"trade_{symbol.replace(':', '_').replace('-', '_')}_{experiment_name}_{int(timestamp.timestamp())}"

        # ── Audit Lifecycle: Signal Generated ────────────────────────────
        t_id = None if is_counterfactual else trade_id
        self.execution_auditor.log_event("SIGNAL_GENERATED", trade_id=t_id, candidate_id=candidate_id, payload=sig)

        # ── Option Contract Strike Selection Redesign ────────────────────
        option_contract = None
        if "INDEX" in symbol:
            try:
                option_contract = self.option_engine.resolve(sig, entry_price)
                logger.info(
                    f"⚡ Option strike selection resolved index signal to contract {option_contract.symbol} "
                    f"@ premium {option_contract.premium} (type: {option_contract.option_type}, strike: {option_contract.strike})"
                )
                
                # ── Audit Lifecycle: Option Resolved ────────────────────────
                self.execution_auditor.log_event(
                    "STRIKE_SELECTED", 
                    trade_id=t_id, 
                    candidate_id=candidate_id, 
                    payload={
                        "symbol": option_contract.symbol,
                        "strike": option_contract.strike,
                        "expiry": option_contract.expiry,
                        "type": option_contract.option_type
                    }
                )
                self.execution_auditor.log_event(
                    "PREMIUM_RETRIEVED", 
                    trade_id=t_id, 
                    candidate_id=candidate_id, 
                    payload={
                        "premium": option_contract.premium,
                        "bid": option_contract.bid,
                        "ask": option_contract.ask
                    }
                )
            except Exception as e:
                logger.error(f"❌ Failed to resolve option contract: {e}", exc_info=True)

            # Never place a REAL order on an unresolved/fabricated contract. If the
            # premium could not be resolved from the warehouse or a live quote, skip
            # the entry entirely. (Counterfactuals are research-only and may proceed.)
            if option_contract is None and not is_counterfactual:
                logger.critical(
                    f"🛑 No valid option contract resolved for {symbol}; SKIPPING real entry "
                    f"(refusing to trade on a fabricated/unknown premium)."
                )
                self.execution_auditor.log_event(
                    "ENTRY_ABORTED", trade_id=t_id, candidate_id=candidate_id,
                    payload={"reason": "OPTION_UNRESOLVED"}
                )
                return

        # ── Audit Lifecycle: Order Placement/Fill ────────────────────────
        self.execution_auditor.log_event(
            "ORDER_SUBMITTED" if not is_counterfactual else "CF_SUBMITTED",
            trade_id=t_id,
            candidate_id=candidate_id,
            payload={"price": entry_price, "sl": sl_price, "tp": tp_price}
        )
        self.execution_auditor.log_event(
            "ORDER_FILLED" if not is_counterfactual else "CF_FILLED",
            trade_id=t_id,
            candidate_id=candidate_id,
            payload={"price": entry_price, "sl": sl_price, "tp": tp_price}
        )

        # Calculate Position Size (Bug 21)
        position_size_inr = 1000.0
        lots = 1.0
        regime_primary = sig.get('features', {}).get('regime_primary', 'UNKNOWN')
        regime_vol_state = sig.get('features', {}).get('regime_vol_state', 'NORMAL')
        confidence = sig.get('confidence', 70.0) or 70.0

        if sl_price and entry_price and sl_price != entry_price:
            position_size_inr = self.sizer.get_position_size(
                entry_price=entry_price,
                stop_loss_price=sl_price,
                strategy=sig['strategy'],
                confidence=confidence,
                regime_primary=regime_primary,
                regime_vol_state=regime_vol_state,
                # Pass currently-deployed notional so the 40% portfolio-exposure
                # cap actually binds. Real trades only; CFs don't consume capital.
                deployed_capital=(0.0 if is_counterfactual else self._deployed_capital()),
            )
            
        lot_size = None
        entry_premium = None
        if option_contract:
            from src.core.options_mapper import OptionsMapper
            from src.core.options_execution_engine import realistic_fill_price
            lot_size = OptionsMapper.get_lot_size(option_contract.symbol)
            premium = option_contract.premium or 100.0
            if premium > 0 and lot_size > 0:
                lots = max(1, int(position_size_inr / (premium * lot_size)))
            # Hard live-order lot cap (LIVE_MODE only, real trades only) — a
            # ceiling independent of PositionSizer's own exposure caps, so a
            # sizing bug can never place more than MAX_LIVE_LOTS on a real order.
            if self.live_mode and not is_counterfactual:
                lots = min(lots, self.max_live_lots)
            # Real fill for P&L purposes: a BUY CALL/PUT closes long, so it opens
            # by paying the ask (never the raw LTP) — same convention combo legs
            # already use (options_execution_engine.realistic_fill_price).
            entry_premium = realistic_fill_price(
                option_contract.premium, option_contract.bid, option_contract.ask, 'BUY'
            )

        diagnostics = sig.get('diagnostics') or {}
        if not isinstance(diagnostics, dict):
            diagnostics = {"raw": diagnostics}
        diagnostics['position_size_inr'] = position_size_inr
        diagnostics['lots'] = lots
        if option_contract:
            diagnostics['option_symbol'] = option_contract.symbol
            diagnostics['option_premium'] = option_contract.premium

        # Structural-invalidation snapshot (read-only reuse of StructureEngine's
        # already-computed swing output — no edits to structure_engine.py or the
        # frozen enhanced_strategy_engine.py). `snapshot` is the same MarketSnapshot
        # that produced this signal, passed through from market_loop().
        structure_trend_at_entry = None
        struct_level = None
        struct_side = None
        if snapshot is not None and getattr(snapshot, 'market', None) and getattr(snapshot.market, 'structure', None):
            st = snapshot.market.structure
            structure_trend_at_entry = getattr(st, 'trend', None)
            if sig['signal'] == 'BUY CALL' and getattr(st, 'last_swing_low', None) is not None:
                struct_level = st.last_swing_low.price
                struct_side = 'SWING_LOW'
            elif sig['signal'] == 'BUY PUT' and getattr(st, 'last_swing_high', None) is not None:
                struct_level = st.last_swing_high.price
                struct_side = 'SWING_HIGH'

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
            'tp1': tp1_price,
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
            'status': 'OPEN',
            'rejection_reasons': sig.get('rejection_reasons', []),
            'position_size_inr': position_size_inr,  # notional; drives deployed-capital exposure gate
            'lots': lots,
            '_last_pnl_r': 0.0,  # Set by _exit_position for portfolio tracking
            'confidence': sig.get('confidence'),
            'diagnostics': diagnostics,
            'option_symbol': option_contract.symbol if option_contract else None,
            'option_premium': option_contract.premium if option_contract else None,
            # Real-fill premium tracking (single-leg P&L fix) — when present,
            # _update_position/_exit_position price pnl_r off the actual option
            # premium instead of assuming the option tracked the index
            # point-for-point. None for raw-index CFs with no resolved contract.
            'entry_premium': entry_premium,
            'exit_premium': None,
            'lot_size': lot_size,
            'pnl_calculation_method': None,
            # First-cut live order placement — always present, only ever
            # populated for real (non-CF) trades when LIVE_MODE is on.
            'is_live': False,
            'live_order_id': None,
            'live_fill_price': None,
            'live_exit_order_id': None,
            'live_exit_fill_price': None,
            # ── Context-aware exit management (opt-in per experiment, see
            # Experiment.params['exit_management'] / _resolve_exit_mgmt_params) ──
            'exit_mgmt': self._resolve_exit_mgmt_params(experiment_name),
            'atr_at_entry': sig.get('features', {}).get('atr'),
            'structure_trend_at_entry': structure_trend_at_entry,
            'structure_invalidation_level': struct_level,
            'structure_invalidation_side': struct_side,
            'tp_expansion_count': 0,
            'last_tp_expansion_regime': None,
        }

        # ── Live entry order (REAL trades only, LIVE_MODE gated) ─────────
        # Fires before storage below so trade_performance's initial insert
        # already carries the real order id/fill, not a later backfill.
        if not is_counterfactual and self.live_mode:
            self._place_live_entry_order(pos)

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
                    'rejection_reasons': sig.get('rejection_reasons', []),
                    'option_symbol': option_contract.symbol if option_contract else None,
                    'option_premium': option_contract.premium if option_contract else None,
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
                'tp1': tp1_price,
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
                'event_id': f"evt_{int(timestamp.timestamp())}_{symbol}_{experiment_name}_entry",
                'trade_id': trade_id,
                'timestamp': timestamp,
                'event_type': 'ENTRY',
                'payload': {
                    'entry_price': entry_price,
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'candidate_id': candidate_id,
                    'option_symbol': option_contract.symbol if option_contract else None,
                    'option_premium': option_contract.premium if option_contract else None,
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
                'tp1': tp1_price,
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
                'position_size_inr': position_size_inr,
                'lots': lots,
                'current_price': entry_price,
                'unrealized_pnl_r': 0.0,
                'last_heartbeat_at': timestamp,
                'is_live': pos.get('is_live', False),
                'live_order_id': pos.get('live_order_id'),
                'live_fill_price': pos.get('live_fill_price'),
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
        
        # ── Audit Lifecycle: Order Modification ─────────────────────────
        t_id = pos.get('trade_id')
        cand_id = pos.get('candidate_id')
        event_type = "SL_MODIFIED" if reason == "TRAILING_SL" else "TP_EXPANDED" if reason == "TP_EXPANSION" else f"ORDER_{reason}"
        self.execution_auditor.log_event(
            event_type if not is_cf else f"CF_{event_type}",
            trade_id=t_id,
            candidate_id=cand_id,
            payload={
                "price": current_price,
                "old_sl": old_sl,
                "new_sl": pos['stop_loss'],
                "old_tp": old_tp,
                "new_tp": pos['take_profit'],
                "reason": reason
            }
        )
        
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
                'event_id': f"evt_{int(timestamp.timestamp())}_{symbol}_{pos.get('experiment_name','')}_{reason.lower()}",
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
                'tp1': pos.get('tp1'),
                'initial_stop_loss': pos['initial_stop_loss'],
                'initial_take_profit': pos['initial_take_profit'],
                'highest_price': highest,
                'lowest_price': lowest,
                'stop_loss_distance': stop_loss_distance,
                'signal_type': pos['signal'],
                'capture_rate': 0.0,
                'confidence': pos.get('confidence'),
                'diagnostics': pos.get('diagnostics'),
                'position_size_inr': pos.get('position_size_inr', 0.0),
                'lots': pos.get('lots', 1.0),
                'entry_premium': pos.get('entry_premium'),
                'exit_premium': None,
                'option_symbol': pos.get('option_symbol'),
                'pnl_calculation_method': None,
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

        # Guard capture rate — store NULL (None) when there was nothing meaningful to
        # capture. MFE <= 0 means price never moved in our favour; capture=0.0 would
        # falsely imply "terrible efficiency" there, so we store NULL ("N/A") instead.
        # We also floor MFE at MIN_MFE_FOR_CAPTURE_R: dividing by a near-zero MFE
        # (e.g. +0.02R) blows pnl_r/mfe_r up to extreme ratios (-5000%+) on ordinary
        # reversal trades, which then dominates AVG(capture_rate) in daily rollups —
        # not a real efficiency signal, just a division artifact.
        MIN_MFE_FOR_CAPTURE_R = 0.15
        capture_rate = None
        if mfe_r >= MIN_MFE_FOR_CAPTURE_R:
            capture_rate = round(pnl_r / mfe_r, 4)

        # Holding efficiency: R earned per bar held
        # A 4R in 5 bars vs 4R in 60 bars are very different strategies
        holding_efficiency = round(pnl_r / max(bars_held, 1), 4)

        # Store pnl_r on pos so _update_active_trades can pass it to PortfolioManager
        pos['_last_pnl_r'] = pnl_r

        # Update PositionSizer Kelly fraction stats
        if not is_cf:
            self.sizer.record_trade_result(pos['strategy'], pnl_r)

        # ── Live exit order (REAL trades only, LIVE_MODE gated) ──────────
        # Squares off whatever live position was opened at entry. The exit
        # itself is still decided by the simulated engine above (SL/TP/session
        # end) — only the fill is real.
        if not is_cf and self.live_mode and pos.get('live_order_id'):
            self._place_live_exit_order(pos, exit_price, mapped_reason)

        # ── Audit Lifecycle: Order Exited ──────────────────────────────
        t_id = pos.get('trade_id')
        cand_id = pos.get('candidate_id')
        self.execution_auditor.log_event(
            "ORDER_EXITED" if not is_cf else "CF_EXITED",
            trade_id=t_id,
            candidate_id=cand_id,
            payload={
                "exit_price": exit_price,
                "exit_reason": mapped_reason,
                "pnl_r": pnl_r,
                "duration_minutes": duration_minutes,
                "bars_held": bars_held
            }
        )

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
                'event_id': f"evt_{int(timestamp.timestamp())}_{symbol}_{pos.get('experiment_name','')}_exit",
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
                'tp1': pos.get('tp1'),
                'initial_stop_loss': pos['initial_stop_loss'],
                'initial_take_profit': pos['initial_take_profit'],
                'highest_price': highest,
                'lowest_price': lowest,
                'stop_loss_distance': stop_loss_distance,
                'signal_type': pos['signal'],
                'capture_rate': capture_rate,
                'holding_efficiency': holding_efficiency,
                'confidence': pos.get('confidence'),
                'diagnostics': pos.get('diagnostics'),
                'position_size_inr': pos.get('position_size_inr', 0.0),
                'lots': pos.get('lots', 1.0),
                'entry_premium': pos.get('entry_premium'),
                'exit_premium': pos.get('exit_premium'),
                'option_symbol': pos.get('option_symbol'),
                'pnl_calculation_method': pos.get('pnl_calculation_method'),
                'live_exit_order_id': pos.get('live_exit_order_id'),
                'live_exit_fill_price': pos.get('live_exit_fill_price'),
                'atr_at_entry': pos.get('atr_at_entry'),
                'atr_at_exit': pos.get('_atr_at_exit'),
                'tp_expansion_count': pos.get('tp_expansion_count', 0),
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
                'entry_premium': pos.get('entry_premium'),
                'exit_premium': pos.get('exit_premium'),
                'option_symbol': pos.get('option_symbol'),
                'pnl_calculation_method': pos.get('pnl_calculation_method'),
                'atr_at_entry': pos.get('atr_at_entry'),
                'atr_at_exit': pos.get('_atr_at_exit'),
                'tp_expansion_count': pos.get('tp_expansion_count', 0),
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

    # ── Multi-leg options combos (vertical spreads, straddle/strangle) ──────

    def _handle_combo_signal(self, sig: Dict, timestamp, symbol: str, experiment_name: str, trade_key: Tuple, result_experiment_name: str, regime_detail=None):
        """Combo counterpart of the accepted/CF branching in market_loop, kept as
        its own method so market_loop doesn't have to interleave two dedup/risk
        models inline."""
        if sig['accepted']:
            # Regime router: same check the single-leg path applies (market_loop,
            # above) — combo signals previously skipped this entirely, so
            # regime_router.py's affinity entries (e.g. short-vol/theta strategies
            # restricted to RANGE/COMPRESSION) were silently never enforced for
            # any combo strategy. Ineligible signals aren't discarded — routed to
            # CF, same "same engine, different storage" philosophy as every
            # other filter.
            if regime_detail is not None and not is_regime_eligible(experiment_name, regime_detail):
                sig['rejection_reasons'] = sig.get('rejection_reasons', []) + ['REGIME_MISMATCH']
                sig['regime_at_decision'] = regime_detail.label
                logger.info(
                    f"🧭 [{experiment_name}] Regime router: {symbol} combo blocked from REAL "
                    f"capital (regime={regime_detail.label}) — routed to CF"
                )
                self._enter_combo_position(sig, timestamp, trade_key, is_counterfactual=True)
                return

            if trade_key in self.active_combo_trades:
                logger.debug(f"↩️  [{experiment_name}] Already have an open combo position on {symbol}, skipping.")
                return
            can_enter, gate_reason = self._can_enter_real(timestamp)
            if not can_enter:
                logger.warning(f"⛔ [{experiment_name}] Combo entry on {symbol} blocked by risk governor: {gate_reason}")
                self.db.save_risk_governor_block({
                    'block_id': f"blk_{symbol.replace(':', '_').replace('-', '_')}_{experiment_name}_{int(timestamp.timestamp())}",
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'experiment_name': experiment_name,
                    'setup_type': sig.get('strategy'),
                    'signal_type': sig.get('signal'),
                    'candidate_id': sig.get('candidate_id'),
                    'gate_reason': gate_reason,
                    'entry_price': sig.get('price'),
                    'stop_loss': None,
                    'take_profit': None,
                    'rr_ratio': sig.get('rr_ratio'),
                })
                return
            logger.info(f"🚀 COMBO SIGNAL: {symbol} {sig['signal']} | [{experiment_name}]")
            self._enter_combo_position(sig, timestamp, trade_key, is_counterfactual=False)
            self.portfolios.on_entry(experiment_name, timestamp)
        else:
            MAX_ACTIVE_CF_COMBOS = 200
            if len(self.active_cf_combos) >= MAX_ACTIVE_CF_COMBOS:
                logger.warning(f"⚠️ CF combo safety limit reached, skipping {symbol}")
                return

            exp_obj = self.registry.get(result_experiment_name)
            if exp_obj:
                thesis_base = exp_obj.strategy.thesis_key(sig)
            else:
                thesis_base = (symbol, sig.get('strategy', 'UNKNOWN'), sig.get('signal', ''))
            thesis_key = (result_experiment_name,) + thesis_base

            if thesis_key in self.active_cf_combo_theses:
                logger.debug(f"↩️  [{experiment_name}] Combo thesis already tracked: {thesis_base}")
                return

            logger.info(
                f"👻 [{experiment_name}] CF COMBO {sig.get('strategy','')} {symbol} {sig.get('signal','')} "
                f"| Rejected: {sig['rejection_reasons']}"
            )
            self._enter_combo_position(sig, timestamp, trade_key, is_counterfactual=True)

    def _enter_combo_position(self, sig: Dict, timestamp, trade_key: Tuple, is_counterfactual: bool):
        symbol = sig['symbol']
        experiment_name = sig.get('experiment_name', '')
        combo_type = sig['signal']
        underlying_price = sig['price']

        try:
            resolved = self.multi_leg_engine.resolve(symbol, underlying_price, combo_type, sig['combo_legs'])
        except Exception as e:
            # Unlike single-leg CFs (which can proceed without a resolved option
            # contract, since single-leg PnL is index-R-based), a combo's PnL is
            # ALWAYS premium-based — there is no meaningful combo research data
            # without real leg premiums. Both real and CF entries skip here.
            logger.error(f"❌ Failed to resolve combo legs for {symbol} {combo_type}: {e}")
            return

        lots = 1
        if not is_counterfactual:
            from src.core.position_sizer import SHORT_VOL_GROUP_EXPERIMENTS
            is_short_vol = experiment_name in SHORT_VOL_GROUP_EXPERIMENTS
            lots = self.sizer.get_combo_lots(
                max_loss_per_lot=resolved.max_loss,
                strategy=sig.get('strategy', experiment_name),
                confidence=sig.get('confidence', 70.0) or 70.0,
                regime_primary=sig.get('features', {}).get('regime_primary', 'UNKNOWN'),
                regime_vol_state=sig.get('features', {}).get('regime_vol_state', 'NORMAL'),
                deployed_capital=self._deployed_capital(),
                group_deployed_capital=self._short_vol_group_deployed_capital() if is_short_vol else 0.0,
                is_short_vol_group=is_short_vol,
            )
            if lots <= 0:
                logger.info(f"⛔ [{experiment_name}] Combo entry on {symbol} skipped — capital/group exposure exhausted")
                return

        from src.core.options_execution_engine import realistic_fill_price
        combo_id = sig.get('candidate_id') or f"combo_{symbol.replace(':', '_').replace('-', '_')}_{experiment_name}_{int(timestamp.timestamp())}"
        legs_payload = [
            {
                'option_symbol': leg.contract.symbol, 'strike': leg.contract.strike,
                'option_type': leg.option_type, 'side': leg.side, 'expiry': leg.contract.expiry,
                # Matches the realistic-fill price baked into resolved.net_premium_paid
                # (MultiLegExecutionEngine.resolve), not the raw LTP — keeps the
                # per-leg audit trail consistent with the actual P&L math.
                'entry_premium': realistic_fill_price(
                    leg.contract.premium, leg.contract.bid, leg.contract.ask, leg.side
                ),
                'exit_premium': None,
            }
            for leg in resolved.legs
        ]

        pos = {
            'combo_id': combo_id,
            'symbol': symbol,
            'experiment_name': experiment_name,
            'strategy_id': sig.get('strategy_id', ''),
            'version': sig.get('version', ''),
            'combo_type': combo_type,
            'setup_type': sig.get('strategy'),
            'entry_time': timestamp,
            'underlying_entry_price': underlying_price,
            'legs': legs_payload,
            'lots': lots,
            'net_premium_paid': resolved.net_premium_paid,
            'max_loss': resolved.max_loss,
            'max_profit': resolved.max_profit,
            'target_r': sig.get('target_r', 1.5),
            'stop_r': sig.get('stop_r', -0.5),
            'current_pnl_r': 0.0,
            'confidence': sig.get('confidence'),
            'diagnostics': sig.get('diagnostics'),
            'is_counterfactual': is_counterfactual,
            'status': 'OPEN',
            'rejection_reasons': sig.get('rejection_reasons', []),
            '_last_pnl_r': 0.0,
        }

        if is_counterfactual:
            self.active_cf_combos[combo_id] = pos
            thesis_base = (symbol, sig.get('strategy', ''), combo_type)
            self.active_cf_combo_theses[(experiment_name,) + thesis_base] = combo_id
            self.db.save_counterfactual_combo_result(dict(pos))
        else:
            self.active_combo_trades[trade_key] = pos
            self.db.save_combo_trade(dict(pos))

        event = {
            'event_id': f"evt_{int(timestamp.timestamp())}_{combo_id}_entry",
            'combo_id': combo_id,
            'timestamp': timestamp,
            'event_type': 'ENTRY',
            'payload': {
                'combo_type': combo_type, 'legs': legs_payload,
                'net_premium_paid': resolved.net_premium_paid, 'max_loss': resolved.max_loss,
            },
        }
        (self.db.save_counterfactual_combo_event if is_counterfactual else self.db.save_combo_event)(event)

        tag = "CF " if is_counterfactual else ""
        logger.info(
            f"🟢 {tag}COMBO ENTRY [{experiment_name}|{combo_id[-20:]}]: {symbol} {combo_type} "
            f"| Net premium: {resolved.net_premium_paid:.2f} | Max loss: {resolved.max_loss:.2f}"
        )

    def _notify_strategy_exit(self, experiment_name: str, symbol: str, pnl_r: float, timestamp) -> None:
        """Tell a strategy instance its position just closed, e.g. so it can
        start a loss-cooldown. No-op for strategies that don't opt in (most
        don't need this — added for ButterflyStrategy's re-entry-after-loss guard)."""
        experiment = self.registry.get(experiment_name)
        if experiment is not None and hasattr(experiment.strategy, 'notify_exit'):
            experiment.strategy.notify_exit(symbol, pnl_r, timestamp)

    def _update_combo_position(self, pos: Dict, underlying_price: float, timestamp, is_cf: bool) -> bool:
        """Re-fetch each leg's live premium, compute combined PnL, and check
        target/stop/session-end. Returns True if the combo exited."""
        if pos.get('status', 'OPEN') != 'OPEN':
            return False

        from src.core.options_execution_engine import PremiumResolver, realistic_fill_price
        premium_resolver = PremiumResolver(self.db, self.data_provider)

        symbol = pos['symbol']
        legs = pos['legs']
        current_net_value = 0.0
        for leg in legs:
            # Closing reverses the fill side: a long (BUY) leg must be SOLD to
            # close (fills at bid), a short (SELL) leg must be BOUGHT back
            # (fills at ask) — the opposite of the entry-side convention.
            closing_side = 'SELL' if leg['side'] == 'BUY' else 'BUY'
            try:
                premium, bid, ask, _ = premium_resolver.resolve_premium(
                    symbol, leg['strike'], leg['option_type'], leg['expiry'], leg['option_symbol'],
                )
                premium = realistic_fill_price(premium, bid, ask, closing_side)
            except Exception as e:
                logger.warning(f"⚠️ Could not refresh leg {leg['option_symbol']}: {e} — using last known premium")
                premium = leg.get('exit_premium') or leg['entry_premium']
            leg['exit_premium'] = premium
            current_net_value += premium if leg['side'] == 'BUY' else -premium

        pnl = current_net_value - pos['net_premium_paid']
        pnl_r = round(pnl / pos['max_loss'], 3) if pos['max_loss'] > 0 else 0.0
        pos['current_pnl_r'] = pnl_r

        is_closed = False
        exit_reason = None

        # For bounded-profit combos (credit spreads, iron condor), the strategy's
        # fixed target_r can exceed what this specific trade's own strike/premium
        # selection can ever pay out (max_profit/max_loss < target_r) — e.g. a
        # credit spread with max_loss=73.6, max_profit=26.4 caps out at 0.36R,
        # so a target_r=0.5 could never fire and the trade could only ever end
        # in STOP_R or SESSION_END. Cap the effective target at what's actually
        # reachable. Unbounded-profit combos (straddle/strangle, max_profit=None)
        # keep the strategy's stated target_r unchanged.
        effective_target_r = pos['target_r']
        max_profit = pos.get('max_profit')
        if max_profit is not None and pos['max_loss'] > 0:
            effective_target_r = min(effective_target_r, max_profit / pos['max_loss'])

        if pnl_r >= effective_target_r:
            is_closed = True
            exit_reason = 'TARGET_R'
        elif pnl_r <= pos['stop_r']:
            is_closed = True
            exit_reason = 'STOP_R'
        elif timestamp.hour == 15 and timestamp.minute >= 25:
            is_closed = True
            exit_reason = 'SESSION_END'

        if is_closed:
            pos['status'] = 'EXIT_PENDING'
            self._exit_combo_position(pos, underlying_price, exit_reason, timestamp, pnl_r)
            pos['status'] = 'CLOSED'
            return True

        # Heartbeat — persist current leg premiums / pnl_r every candle, same
        # cadence as the single-leg live-heartbeat.
        (self.db.save_counterfactual_combo_result if is_cf else self.db.save_combo_trade)(dict(pos))
        return False

    def _exit_combo_position(self, pos: Dict, underlying_exit_price: float, reason: str, timestamp, pnl_r: float):
        symbol = pos['symbol']
        is_cf = pos.get('is_counterfactual', False)
        duration_minutes = (timestamp - pos['entry_time']).total_seconds() / 60.0
        pos['_last_pnl_r'] = pnl_r

        result = dict(pos)
        result['exit_time'] = timestamp
        result['underlying_exit_price'] = underlying_exit_price
        result['final_pnl_r'] = pnl_r
        result['exit_reason'] = reason
        result['duration_minutes'] = round(duration_minutes, 2)

        event = {
            'event_id': f"evt_{int(timestamp.timestamp())}_{pos['combo_id']}_exit",
            'combo_id': pos['combo_id'],
            'timestamp': timestamp,
            'event_type': 'EXIT',
            'payload': {
                'exit_reason': reason, 'final_pnl_r': pnl_r,
                'duration_minutes': duration_minutes, 'legs': pos['legs'],
            },
        }

        if is_cf:
            self.db.save_counterfactual_combo_event(event)
            self.db.save_counterfactual_combo_result(result)
        else:
            self.db.save_combo_event(event)
            self.db.save_combo_trade(result)
            # Feed Kelly bookkeeping from real combo outcomes — previously never
            # wired up, so get_combo_lots()'s Kelly fraction for every combo
            # strategy would have stayed at the default RISK_FRACTION forever.
            self.sizer.record_trade_result(pos.get('setup_type', pos.get('experiment_name', '')), pnl_r)

        tag = "CF " if is_cf else ""
        logger.info(
            f"🔴 {tag}COMBO EXIT ({reason}) [{pos.get('experiment_name','')}|{pos['combo_id'][-20:]}]: "
            f"{symbol} {pos['combo_type']} | PnL {pnl_r:+.2f}R | {duration_minutes:.0f}min"
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

    tz = ZoneInfo("Asia/Kolkata")
    
    # 1. Align scheduler to the next 30-second grid offset by 5 seconds (i.e. :05 or :35 of the minute)
    now = datetime.now(tz)
    target_seconds = 5 if now.second < 5 or now.second >= 35 else 35
    if target_seconds == 5 and now.second >= 35:
        sleep_s = (60 - now.second) + 5
    else:
        sleep_s = target_seconds - now.second
        
    logger.info(f"⏱️ Aligning scheduler to 30-second grid offset by 5s. Sleeping {sleep_s}s until {(now + timedelta(seconds=sleep_s)).strftime('%H:%M:%S')} IST...")
    time.sleep(sleep_s)

    logger.info("⏱️ Scheduler aligned and running. Evaluating positions every 30s, signals on new completed M5 candles.")

    # 2. Run one immediate initial evaluation
    # We round down to the nearest 5-minute block to initialize last_processed_m5_time
    now = datetime.now(tz)
    current_m5_block = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 5)
    trader.last_processed_m5_time = current_m5_block
    
    logger.info(f"🔔 [Initial Run] Running full market loop at {now.strftime('%H:%M:%S')}...")
    trader.market_loop()

    # 3. Independent Scheduler Loop
    while True:
        loop_start = datetime.now(tz)
        
        # Check if we are close to a new 5-minute boundary (minute is multiple of 5 and second is around :05)
        # We calculate current 5-minute block
        current_m5_block = loop_start.replace(second=0, microsecond=0) - timedelta(minutes=loop_start.minute % 5)
        
        is_new_m5_candle = False
        # If the current block is different from the last processed one, and we are at least at the :05 mark
        # of that block, it means the candle has closed and Fyers has had time to populate it.
        if trader.last_processed_m5_time != current_m5_block:
            # Check if we are past the 5-second buffer (to allow for latency in data feed)
            if loop_start.second >= 5:
                is_new_m5_candle = True

        if is_new_m5_candle:
            logger.info(f"🔔 [New Candle Closed] Running entry pipeline for candle {current_m5_block.strftime('%H:%M')} at {loop_start.strftime('%H:%M:%S')}...")
            trader.market_loop()
            trader.last_processed_m5_time = current_m5_block
        else:
            # Run lightweight real-time position exit check
            trader.position_tracking_loop()

        # Sleep to align to the next :05 / :35 boundary
        now = datetime.now(tz)
        if 5 <= now.second < 35:
            sleep_s = 35 - now.second
        elif now.second < 5:
            sleep_s = 5 - now.second
        else:
            sleep_s = (60 - now.second) + 5
            
        # Safety margin: ensure we sleep at least 5s to avoid tight loops/multiple executions
        if sleep_s < 5:
            sleep_s += 30
            
        time.sleep(sleep_s)

if __name__ == "__main__":
    main()
