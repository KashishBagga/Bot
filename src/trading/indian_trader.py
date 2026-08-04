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
from src.core.expiry_blackout import ExpiryBlackoutManager
from src.strategies.structural_strategy import StructuralStrategy
from src.strategies.ema_pullback import EmaPullbackStrategy
from src.strategies.vwap_reversion import VwapReversionStrategy
from src.strategies.prev_day_extremes import PrevDayExtremesStrategy
from src.strategies.orb import OrbStrategy
from src.strategies.atr_squeeze import AtrSqueezeStrategy
from src.strategies.geometry_strategy import GeometryStrategy
from src.strategies.order_flow_strategy import OrderFlowStrategy
from src.strategies.chart_pattern_strategy import ChartPatternStrategy
from src.strategies.vwap_reclaim import VwapReclaimStrategy
from src.strategies.cpr_strategy import CprStrategy
from src.strategies.gap_strategy import GapStrategy
from src.strategies.vertical_spread_strategy import VerticalSpreadStrategy
from src.strategies.straddle_strangle_strategy import StraddleStrangleStrategy

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

        # Expiry & event blackout manager (Bug 18 fix)
        self.expiry_blackout = ExpiryBlackoutManager()

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
        # Aug-03 review: LOW_RVOL blocked +52R of with-trend signals. Pullback
        # entries structurally have quiet volume during the retrace phase — the
        # expansion follows the pullback, not before it. Using a breakout-tuned
        # threshold (1.0×) here was killing the best trend-following strategy.
        # Lowered to 0.5x (just checks for non-zero activity) + efficiency 0.45.
        _ema_pullback_exp = Experiment(
            name="EMA_Pullback_20_50_RVOL1.0",
            strategy=EmaPullbackStrategy(rvol_threshold=0.5, min_efficiency=0.45),
            params={"rvol_threshold": 0.5, "min_efficiency": 0.45},
            description="EMA Pullback — RVOL 0.5x (pullbacks are quiet by nature), efficiency 0.45"
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
        # Aug-03 review: 6 live trades taken (all counter-trend BUY PUT on a
        # strong up day), net −1.86R expectancy, 5 of 6 stopped on bar-1.
        # Squeezes NEED confirmed volume expansion at the breakout — raised RVOL
        # to 1.5x (was 1.0x).  Also added a move-efficiency floor (0.50) to block
        # counter-trend squeezes in one-directional trend days.
        _atr_squeeze_exp = Experiment(
            name="ATR_Squeeze_RVOL1.0",
            strategy=AtrSqueezeStrategy(rvol_threshold=1.5, atr_percentile_threshold=0.20),
            params={"rvol_threshold": 1.5, "atr_percentile_threshold": 0.20},
            description="ATR Squeeze — RVOL 1.5x (requires actual expansion volume at breakout)"
        )
        self.registry.register(_atr_squeeze_exp)
        self.db.save_experiment(_atr_squeeze_exp.to_db_dict())

        # 7. Geometry Strategy — purely from MKE Stage 5 GeometryContext
        _geometry_v1_exp = Experiment(
            name="Geometry_v1.0_Score35",
            strategy=GeometryStrategy(
                min_confluence_score=50.0,   # BUG FIX: was 35 — too low, accepts junk zones
                zone_tolerance_pct=0.002,
                min_body_fraction=0.40,
                min_bias_confidence=0.45,
                atr_sl_buffer_mult=0.15,
                tp_atr_cap=3.0,
                min_rr=1.5,
                trendline_break_enabled=True,
            ),
            params={
                "min_confluence_score": 50.0,
                "zone_tolerance_pct": 0.002,
                "min_body_fraction": 0.40,
                "min_bias_confidence": 0.45,
                "min_rr": 1.5,
                "trendline_break_enabled": True,
            },
            description="Geometry Strategy v1.0 — confluence bounce + trendline retest. Score threshold=50 (fixed from 35)."
        )
        self.registry.register(_geometry_v1_exp)
        self.db.save_experiment(_geometry_v1_exp.to_db_dict())

        _geometry_v1_tight_exp = Experiment(
            name="Geometry_v1.0_Score50",
            strategy=GeometryStrategy(
                min_confluence_score=50.0,
                zone_tolerance_pct=0.002,
                min_body_fraction=0.40,
                min_bias_confidence=0.50,
                atr_sl_buffer_mult=0.15,
                tp_atr_cap=3.0,
                min_rr=1.8,
                trendline_break_enabled=True,
            ),
            params={
                "min_confluence_score": 50.0,
                "zone_tolerance_pct": 0.002,
                "min_body_fraction": 0.40,
                "min_bias_confidence": 0.50,
                "min_rr": 1.8,
                "trendline_break_enabled": True,
            },
            description="Geometry Strategy v1.0 tighter — Score threshold=50 (3+ sources), RR>=1.8."
        )
        self.registry.register(_geometry_v1_tight_exp)
        self.db.save_experiment(_geometry_v1_tight_exp.to_db_dict())

        # 8. Order Flow Strategy (Milestone 2C)
        _order_flow_exp = Experiment(
            name="OrderFlow_v1.0",
            strategy=OrderFlowStrategy(
                min_sweep_confidence=0.60,  # BUG FIX: was 0.45 — too close to coin flip
                min_imb_confidence=0.55,    # BUG FIX: was 0.45
                min_body_fraction=0.40,
                atr_sl_buffer_mult=0.15,
                tp_atr_cap=3.0,
                min_rr=1.5
            ),
            params={
                "min_sweep_confidence": 0.60,
                "min_imb_confidence": 0.55,
                "min_body_fraction": 0.40,
                "atr_sl_buffer_mult": 0.15,
                "tp_atr_cap": 3.0,
                "min_rr": 1.5
            },
            description="Order Flow Strategy v1.0 — stop sweeps and imbalance pullbacks (M2C)"
        )
        self.registry.register(_order_flow_exp)
        self.db.save_experiment(_order_flow_exp.to_db_dict())

        # 9. Chart Pattern Strategy — trades the previously-unused pattern engine
        # (Double Top/Bottom, H&S, triangles, flags), confirmed by a named
        # candlestick pattern and zone_engine S/R confluence. Two variants to
        # compare a moderate vs. looser acceptance threshold, same as the other
        # multi-variant experiments above.
        _chart_pattern_exp = Experiment(
            name="ChartPattern_v1.0_Conf55",
            strategy=ChartPatternStrategy(
                min_pattern_confidence=0.55,
                min_candle_strength=0.40,
                zone_tolerance_pct=0.003,
                tp_atr_cap=5.0,
                min_rr=1.5,
            ),
            params={
                "min_pattern_confidence": 0.55,
                "min_candle_strength": 0.40,
                "zone_tolerance_pct": 0.003,
                "tp_atr_cap": 5.0,
                "min_rr": 1.5,
            },
            description="Chart Pattern Strategy — pattern engine + candlestick confirmation + zone confluence"
        )
        self.registry.register(_chart_pattern_exp)
        self.db.save_experiment(_chart_pattern_exp.to_db_dict())

        _chart_pattern_loose_exp = Experiment(
            name="ChartPattern_v1.0_Conf40",
            strategy=ChartPatternStrategy(
                min_pattern_confidence=0.40,
                min_candle_strength=0.30,
                zone_tolerance_pct=0.005,
                tp_atr_cap=5.0,
                min_rr=1.3,
            ),
            params={
                "min_pattern_confidence": 0.40,
                "min_candle_strength": 0.30,
                "zone_tolerance_pct": 0.005,
                "tp_atr_cap": 5.0,
                "min_rr": 1.3,
            },
            description="Chart Pattern Strategy — looser acceptance variant for research comparison"
        )
        self.registry.register(_chart_pattern_loose_exp)
        self.db.save_experiment(_chart_pattern_loose_exp.to_db_dict())

        # 10. VWAP Reclaim — trend-continuation on a VWAP cross.
        # Same family as EMA_Pullback (trend-following), so also lowers efficiency
        # floor to 0.45 (was 0.55) — continuations have moderate efficiency dips.
        _vwap_reclaim_exp = Experiment(
            name="VWAP_Reclaim_v1.0",
            strategy=VwapReclaimStrategy(rvol_threshold=1.0, min_efficiency=0.45),
            params={"rvol_threshold": 1.0, "min_efficiency": 0.45},
            description="VWAP Reclaim — trend continuation on a VWAP cross (efficiency 0.45)"
        )
        self.registry.register(_vwap_reclaim_exp)
        self.db.save_experiment(_vwap_reclaim_exp.to_db_dict())

        # 11. CPR (Central Pivot Range) Breakout — India-specific pivot geometry,
        # not previously in this framework.
        _cpr_exp = Experiment(
            name="CPR_v1.0",
            strategy=CprStrategy(rvol_threshold=1.1, min_efficiency=0.55),
            params={"rvol_threshold": 1.1, "min_efficiency": 0.55},
            description="Central Pivot Range breakout — prior-day TC/BC value area"
        )
        self.registry.register(_cpr_exp)
        self.db.save_experiment(_cpr_exp.to_db_dict())

        # 12. Gap — Gap-and-Go / Gap-Fill resolved dynamically as ONE strategy
        # (see gap_strategy.py docstring for why these can't be two independent
        # always-on strategies without a regime discriminator).
        _gap_exp = Experiment(
            name="Gap_v1.0",
            strategy=GapStrategy(
                gap_threshold_pct=0.15, rvol_threshold=1.1,
                min_efficiency=0.55, decision_window_minutes=45,
            ),
            params={
                "gap_threshold_pct": 0.15, "rvol_threshold": 1.1,
                "min_efficiency": 0.55, "decision_window_minutes": 45,
            },
            description="Gap continuation (Gap-and-Go) or reversion (Gap-Fill), decided dynamically"
        )
        self.registry.register(_gap_exp)
        self.db.save_experiment(_gap_exp.to_db_dict())

        # 13. Initial Balance Breakout — same OrbStrategy, 60-min window
        # (only possible now that the 15/30-hardcoded cutoff bug is fixed).
        _ib_exp = Experiment(
            name="ORB_60m_IB_RVOL1.2",
            strategy=OrbStrategy(rvol_threshold=1.2, opening_range_minutes=60),
            params={"rvol_threshold": 1.2, "opening_range_minutes": 60},
            description="Initial Balance Breakout — 60-minute opening range"
        )
        self.registry.register(_ib_exp)
        self.db.save_experiment(_ib_exp.to_db_dict())

        # 14. Vertical Spread — directional thesis (EMA cross + bias), executed
        # as a debit spread instead of a naked single-leg option. First combo
        # (multi-leg) experiment in this framework — see _handle_combo_signal()/
        # _enter_combo_position() in this file for the separate execution path.
        _vertical_spread_exp = Experiment(
            name="VerticalSpread_v1.0",
            strategy=VerticalSpreadStrategy(
                rvol_threshold=1.0, min_efficiency=0.55,
                spread_width_strikes=2, target_r=1.0, stop_r=-0.6,
            ),
            params={
                "rvol_threshold": 1.0, "min_efficiency": 0.55,
                "spread_width_strikes": 2, "target_r": 1.0, "stop_r": -0.6,
            },
            description="Bull Call Spread / Bear Put Spread — directional thesis, debit-spread execution"
        )
        self.registry.register(_vertical_spread_exp)
        self.db.save_experiment(_vertical_spread_exp.to_db_dict())

        # 15. Straddle/Strangle — long volatility on realized-vol compression
        # (a proxy for implied-vol rank, which this system doesn't have data for
        # — see straddle_strangle_strategy.py's module docstring).
        _straddle_exp = Experiment(
            name="Straddle_v1.0_VolCompression",
            strategy=StraddleStrangleStrategy(
                atr_percentile_threshold=0.20, wing_strikes=0,
                decision_cutoff_hour=14, target_r=1.2, stop_r=-0.5,
            ),
            params={
                "atr_percentile_threshold": 0.20, "wing_strikes": 0,
                "decision_cutoff_hour": 14, "target_r": 1.2, "stop_r": -0.5,
            },
            description="Long Straddle on ATR-percentile volatility compression"
        )
        self.registry.register(_straddle_exp)
        self.db.save_experiment(_straddle_exp.to_db_dict())

        # Strangle variant — wider wings, cheaper premium, wider breakevens.
        _strangle_exp = Experiment(
            name="Strangle_v1.0_VolCompression",
            strategy=StraddleStrangleStrategy(
                atr_percentile_threshold=0.20, wing_strikes=2,
                decision_cutoff_hour=14, target_r=1.2, stop_r=-0.5,
            ),
            params={
                "atr_percentile_threshold": 0.20, "wing_strikes": 2,
                "decision_cutoff_hour": 14, "target_r": 1.2, "stop_r": -0.5,
            },
            description="Long Strangle (2-strike wings) on ATR-percentile volatility compression"
        )
        self.registry.register(_strangle_exp)
        self.db.save_experiment(_strangle_exp.to_db_dict())

        self.portfolios = PortfolioManager()
        self.portfolios.register("Structural_v3.2_RVOL1.0")
        self.portfolios.register("Structural_v3.2_RVOL0.8")
        self.portfolios.register("EMA_Pullback_20_50_RVOL1.0")
        self.portfolios.register("VWAP_Reversion_1.5ATR_RVOL1.0")
        self.portfolios.register("PrevDay_Extremes_RVOL1.2")
        self.portfolios.register("ORB_15m_RVOL1.2")
        self.portfolios.register("ORB_30m_RVOL1.2")
        self.portfolios.register("ATR_Squeeze_RVOL1.0")
        self.portfolios.register("Geometry_v1.0_Score35")
        self.portfolios.register("Geometry_v1.0_Score50")
        self.portfolios.register("OrderFlow_v1.0")
        self.portfolios.register("ChartPattern_v1.0_Conf55")
        self.portfolios.register("ChartPattern_v1.0_Conf40")
        self.portfolios.register("VWAP_Reclaim_v1.0")
        self.portfolios.register("CPR_v1.0")
        self.portfolios.register("Gap_v1.0")
        self.portfolios.register("ORB_60m_IB_RVOL1.2")
        self.portfolios.register("VerticalSpread_v1.0")
        self.portfolios.register("Straddle_v1.0_VolCompression")
        self.portfolios.register("Strangle_v1.0_VolCompression")

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
        
        # Load open real positions from DB on startup
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
                'confidence': op.get('confidence'),
                'diagnostics': op.get('diagnostics'),
                # Notional deployed — must be recovered or _deployed_capital()
                # silently undercounts this position after every restart.
                'position_size_inr': op.get('position_size_inr', 0.0) or 0.0,
                'lots': op.get('lots', 1.0) or 1.0,
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
            self.active_combo_trades[key] = dict(op)
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
            self.active_cf_combos[op['combo_id']] = dict(op)
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

        # Reset daily risk counters at the first pulse of a new day
        self._roll_risk_day(now)

        logger.info(f"--- {now.strftime('%H:%M:%S')} Market Pulse ---")

        try:
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
                    current_prices[symbol] = float(closed['close'])
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

            # 3b. Expiry / event blackout gate (Bug 18 fix) — DISABLED as of 2026-07-30
            # (lunch-hour + weekly/monthly expiry + RBI/Budget windows were blocking
            # too much of the session; re-enable by uncommenting below)
            # is_blackout, blackout_reason = self.expiry_blackout.is_blackout()
            # if is_blackout:
            #     logger.info(f"🚫 Expiry/Event blackout active — no new entries. Reason: {blackout_reason}")
            #     # Positions were already updated in step 2 above; do NOT update
            #     # again here. The old double-call inflated bars_held / duration /
            #     # holding_efficiency on every blackout candle (lunch 11:30–13:30
            #     # daily + all Thursdays), corrupting those metrics for most trades.
            #     return

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

                        # Suffix candidate_id with experiment_name to isolate counterfactual positions
                        if 'candidate_id' in sig and sig['candidate_id']:
                            if not sig['candidate_id'].endswith(f"_{experiment_name}"):
                                sig['candidate_id'] = f"{sig['candidate_id']}_{experiment_name}"

                        # Multi-leg combo signals (vertical spreads, straddle/strangle)
                        # take a completely separate path — combined-premium PnL,
                        # not a single directional index R-multiple.
                        if 'combo_legs' in sig:
                            self._handle_combo_signal(sig, now, symbol, experiment_name, trade_key, result.experiment_name)
                            continue

                        if sig['accepted']:
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
                            self._enter_position(sig, now, trade_key, is_counterfactual=False)
                            self._record_level_attempt(sig)   # woodchopper counter
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
        """Sum of notional currently deployed across OPEN real trades (CFs excluded)."""
        return sum(float(p.get('position_size_inr', 0.0)) for p in self.active_trades.values())

    def _persist_sr_zones(self, snapshot) -> None:
        """Upsert high-quality zones from a MarketSnapshot into the persistent sr_zones table.

        Only persists zones with score >= 2.0 to avoid populating the table with
        marginal / noise-level zones.  Zone IDs are deterministic hashes, so the
        same zone across multiple candles increments touch_count rather than
        inserting duplicate rows.
        """
        import hashlib
        from datetime import timezone as _tz

        zones = snapshot.h1_zones or []
        now = snapshot.timestamp or datetime.now(_tz.utc)

        for z in zones:
            try:
                score = float(getattr(z, "score", 0.0))
                if score < 2.0:
                    continue

                level = float(z.level)
                zone_type = z.zone_type          # 'SUPPLY' or 'DEMAND'
                # half-ATR bucket so nearby zones share the same ID
                atr = float(snapshot.features.get_float("atr") or 100.0)
                bucket = round(level / (atr * 0.5)) * int(atr * 0.5)

                raw_id = f"{snapshot.symbol}|{zone_type}|{bucket}"
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
                })
            except Exception as e:
                logger.debug(f"⚠️ _persist_sr_zones: skipping zone — {e}")


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

    def _can_enter_real(self, now, sig: Dict = None) -> Tuple[bool, str]:
        """Aggregate risk gate for NEW real entries. Returns (allowed, reason)."""
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

    def _update_active_trades(self, current_prices: Dict[str, float], timestamp, current_bars: Dict = None):
        """Evaluate open positions against latest market prices.

        Each position is updated inside its own try/except so that one malformed
        position (e.g. a bad field after a schema change) cannot abort SL/TP
        management for the rest of the book — previously an exception here
        unwound the whole loop and left every remaining position unmanaged.
        """
        current_bars = current_bars or {}
        # Update real trades — keyed by (symbol, experiment_name)
        for key in list(self.active_trades.keys()):
            symbol, experiment_name = key
            if symbol not in current_prices:
                continue
            try:
                pos = self.active_trades[key]
                is_closed = self._update_position(
                    pos, current_prices[symbol], timestamp, bar=current_bars.get(symbol)
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
                    pos, current_prices[symbol], timestamp, bar=current_bars.get(symbol)
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
                self.active_cf_combos.pop(combo_id)

    def _update_position(self, pos: Dict, current_price: float, timestamp, bar: Dict = None) -> bool:
        """Evaluate a position against the latest market tick. Returns True if position exited.

        ``bar`` is the last CLOSED candle's OHLC ({'open','high','low','close'}).
        When present, the stop-loss is evaluated intrabar (against the candle's
        low/high, not just its close) and gap-through the stop is filled at the
        candle open — modelling the worst-case fill instead of assuming a perfect
        fill exactly at the stop price.
        """
        symbol = pos['symbol']
        is_cf = pos.get('is_counterfactual', False)
        stop_loss_distance = pos['stop_loss_distance']

        # Intrabar extremes for stop evaluation (fall back to the mark price).
        bar_high = float(bar['high']) if bar else current_price
        bar_low = float(bar['low']) if bar else current_price
        bar_open = float(bar['open']) if bar else current_price

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
            # Check SL breach intrabar (candle low), gap-aware fill at open
            if bar_low <= pos['stop_loss']:
                is_closed = True
                # If the candle OPENED below the stop (gap-down), the real fill is
                # at the open, which is worse than the stop.
                exit_price = min(pos['stop_loss'], bar_open)
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
                # FIX: Tighten trail step to 0.75× once 1.5R is in the bag
                trail_mult = 0.75 if current_pnl_r >= 1.5 else 1.0
                new_sl = current_price - (stop_loss_distance * trail_mult)
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
                # FIX: Tighten trail step to 0.75× once 1.5R is in the bag
                trail_mult = 0.75 if current_pnl_r >= 1.5 else 1.0
                new_sl = current_price + (stop_loss_distance * trail_mult)
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

    def _enter_position(self, sig: Dict, timestamp, trade_key: Tuple, is_counterfactual: bool):
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
        regime = sig.get('features', {}).get('market_regime', 'UNKNOWN')
        confidence = sig.get('confidence', 70.0) or 70.0
        
        if sl_price and entry_price and sl_price != entry_price:
            position_size_inr = self.sizer.get_position_size(
                entry_price=entry_price,
                stop_loss_price=sl_price,
                strategy=sig['strategy'],
                confidence=confidence,
                regime=regime,
                # Pass currently-deployed notional so the 40% portfolio-exposure
                # cap actually binds. Real trades only; CFs don't consume capital.
                deployed_capital=(0.0 if is_counterfactual else self._deployed_capital()),
            )
            
        if option_contract:
            from src.core.options_mapper import OptionsMapper
            lot_size = OptionsMapper.get_lot_size(option_contract.symbol)
            premium = option_contract.premium or 100.0
            if premium > 0 and lot_size > 0:
                lots = max(1, int(position_size_inr / (premium * lot_size)))

        diagnostics = sig.get('diagnostics') or {}
        if not isinstance(diagnostics, dict):
            diagnostics = {"raw": diagnostics}
        diagnostics['position_size_inr'] = position_size_inr
        diagnostics['lots'] = lots

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
            'rejection_reasons': sig.get('rejection_reasons', []),
            'position_size_inr': position_size_inr,  # notional; drives deployed-capital exposure gate
            'lots': lots,
            '_last_pnl_r': 0.0,  # Set by _exit_position for portfolio tracking
            'confidence': sig.get('confidence'),
            'diagnostics': diagnostics,
            'option_symbol': option_contract.symbol if option_contract else None,
            'option_premium': option_contract.premium if option_contract else None,
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

        # Update PositionSizer Kelly fraction stats
        if not is_cf:
            self.sizer.record_trade_result(pos['strategy'], pnl_r)

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

    # ── Multi-leg options combos (vertical spreads, straddle/strangle) ──────

    def _handle_combo_signal(self, sig: Dict, timestamp, symbol: str, experiment_name: str, trade_key: Tuple, result_experiment_name: str):
        """Combo counterpart of the accepted/CF branching in market_loop, kept as
        its own method so market_loop doesn't have to interleave two dedup/risk
        models inline."""
        if sig['accepted']:
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

        combo_id = sig.get('candidate_id') or f"combo_{symbol.replace(':', '_').replace('-', '_')}_{experiment_name}_{int(timestamp.timestamp())}"
        legs_payload = [
            {
                'option_symbol': leg.contract.symbol, 'strike': leg.contract.strike,
                'option_type': leg.option_type, 'side': leg.side, 'expiry': leg.contract.expiry,
                'entry_premium': leg.contract.premium, 'exit_premium': None,
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
            'net_premium_paid': resolved.net_premium_paid,
            'max_loss': resolved.max_loss,
            'max_profit': resolved.max_profit,
            'target_r': sig.get('target_r', 1.5),
            'stop_r': sig.get('stop_r', -0.5),
            'current_pnl_r': 0.0,
            'confidence': sig.get('confidence'),
            'diagnostics': sig.get('diagnostics'),
            'is_counterfactual': is_counterfactual,
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

    def _update_combo_position(self, pos: Dict, underlying_price: float, timestamp, is_cf: bool) -> bool:
        """Re-fetch each leg's live premium, compute combined PnL, and check
        target/stop/session-end. Returns True if the combo exited."""
        from src.core.options_execution_engine import PremiumResolver
        premium_resolver = PremiumResolver(self.db, self.data_provider)

        symbol = pos['symbol']
        legs = pos['legs']
        current_net_value = 0.0
        for leg in legs:
            try:
                premium, _, _, _ = premium_resolver.resolve_premium(
                    symbol, leg['strike'], leg['option_type'], leg['expiry'], leg['option_symbol'],
                )
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

        if pnl_r >= pos['target_r']:
            is_closed = True
            exit_reason = 'TARGET_R'
        elif pnl_r <= pos['stop_r']:
            is_closed = True
            exit_reason = 'STOP_R'
        elif timestamp.hour == 15 and timestamp.minute >= 25:
            is_closed = True
            exit_reason = 'SESSION_END'

        if is_closed:
            self._exit_combo_position(pos, underlying_price, exit_reason, timestamp, pnl_r)
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

    # BUG FIX (Bug 19): Align scheduler to actual 5-minute candle close boundaries.
    # NSE 5-min candles close at 09:20, 09:25, 09:30 ... 15:25 IST.
    # Running at an arbitrary offset (e.g. start at 10:03 → runs at 10:03,10:08...)
    # means we sometimes evaluate incomplete forming candles or stale closed candles.
    tz = ZoneInfo("Asia/Kolkata")
    now = datetime.now(tz)
    # Calculate seconds until the next 5-minute boundary (with a 5-second buffer for data latency)
    seconds_past_boundary = (now.minute % 5) * 60 + now.second
    seconds_to_next = (5 * 60 - seconds_past_boundary) + 5  # +5s data latency buffer
    if seconds_to_next < 10:
        seconds_to_next += 300  # Already very close — wait for the one after
    logger.info(f"⏱️ Aligning to 5-min candle boundary. Sleeping {seconds_to_next}s until {(now + timedelta(seconds=seconds_to_next)).strftime('%H:%M:%S')} IST...")
    time.sleep(seconds_to_next)

    logger.info("⏱️ Scheduler aligned and running. Next candle evaluation at 5-min intervals.")

    # Run once immediately (now aligned to a candle boundary)
    trader.market_loop()

    # Boundary-aligned loop. `schedule.every(5).minutes` anchors the next run to
    # when the PREVIOUS run finished, so each slow fetch/DB cycle adds drift and
    # by afternoon the loop runs minutes past the boundary (deep into the forming
    # candle, and possibly missing the 15:25 force-exit tick). Re-aligning to the
    # fixed 5-min grid every cycle eliminates that drift and guarantees a tick
    # lands in the 15:25 square-off window whenever the process is alive.
    while True:
        now = datetime.now(tz)
        seconds_past = (now.minute % 5) * 60 + now.second
        sleep_s = (5 * 60 - seconds_past) + 5  # +5s data-latency buffer
        if sleep_s < 10:
            sleep_s += 300  # too close to the boundary — wait for the next one
        time.sleep(sleep_s)
        trader.market_loop()

if __name__ == "__main__":
    main()
