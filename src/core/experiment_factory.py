#!/usr/bin/env python3
"""
Experiment Factory — the single source of truth for "which experiments exist".
================================================================================
Extracted from indian_trader.py so the live trader and the backtester
(src/backtesting/advanced_backtester.py) register the EXACT same set of
Experiments against the EXACT same params. Before this existed, the backtester
only ever exercised the frozen EnhancedStrategyEngine directly — none of the
other ~26 registered experiments here were ever backtested, so "does this
strategy actually work" had no offline answer for any of them.

This module has no DB side effects — building the registry is pure. Live-only
bookkeeping (persisting each Experiment's config to the `experiments` table)
stays the caller's responsibility; see indian_trader.py's post-build loop.

To add a strategy: subclass BaseStrategy, register one Experiment(...) here.
Nothing else needs to change — both live and backtest pick it up automatically.
"""

from src.core.experiment import Experiment
from src.core.experiment_registry import ExperimentRegistry
from src.strategies.structural_strategy import StructuralStrategy
from src.strategies.ema_pullback import EmaPullbackStrategy
from src.strategies.vwap_reversion import VwapReversionStrategy
from src.strategies.prev_day_extremes import PrevDayExtremesStrategy
from src.strategies.orb import OrbStrategy
from src.strategies.atr_squeeze import AtrSqueezeStrategy
from src.strategies.geometry_strategy import GeometryStrategy
from src.strategies.order_flow_strategy import OrderFlowStrategy
from src.strategies.vwap_reclaim import VwapReclaimStrategy
from src.strategies.cpr_strategy import CprStrategy
from src.strategies.vertical_spread_strategy import VerticalSpreadStrategy
from src.strategies.straddle_strangle_strategy import StraddleStrangleStrategy
from src.strategies.oi_wall_reaction_strategy import OIWallReactionStrategy
from src.strategies.pcr_extreme_reversal_strategy import PCRExtremeReversalStrategy
from src.strategies.credit_spread_strategy import CreditSpreadStrategy
from src.strategies.gap_strategy import GapStrategy
from src.strategies.iron_condor_strategy import IronCondorStrategy
from src.strategies.butterfly_strategy import ButterflyStrategy
from src.strategies.iron_butterfly_strategy import IronButterflyStrategy
from src.strategies.options_scalping_strategy import OptionsScalpingStrategy
from src.strategies.consolidation_breakout_strategy import ConsolidationBreakoutStrategy
from src.strategies.rsi2_mean_reversion_strategy import Rsi2MeanReversionStrategy
from src.strategies.expiry_aware_theta_strategy import ExpiryAwareThetaStrategy
from src.strategies.relative_value_strategy import RelativeValueStrategy
from src.strategies.momentum_burst_5m import MomentumBurst5mStrategy
from src.strategies.htf_pullback_reversal import HtfPullbackReversalStrategy


def build_registry() -> ExperimentRegistry:
    """Construct a fresh ExperimentRegistry with every currently-active
    experiment registered. Pure — no DB writes, no shared state with any
    previous call (each call returns an independent registry/strategy set)."""
    registry = ExperimentRegistry()

    # 1. Structural
    registry.register(Experiment(
        name="Structural_v3.2_RVOL1.0",
        strategy=StructuralStrategy(rvol_threshold=1.0, min_zone_score=50.0),
        params={"rvol_threshold": 1.0, "min_zone_score": 50.0},
        description="Production structural strategy — RVOL threshold 1.0x"
    ))

    registry.register(Experiment(
        name="Structural_v3.2_RVOL0.8",
        strategy=StructuralStrategy(rvol_threshold=0.8, min_zone_score=50.0),
        params={"rvol_threshold": 0.8, "min_zone_score": 50.0},
        description="Parallel experiment — RVOL threshold 0.8x"
    ))

    # Structural v3.3 — same entry logic as v3.2_RVOL1.0 (EnhancedStrategyEngine
    # stays frozen/untouched), but opts into the new context-aware exit
    # management in indian_trader.py's _update_position(): structural
    # invalidation exit, ATR-adaptive trailing, capped TP expansion, time stop.
    # Shadow-only clone deliberately kept separate from Structural_v3.2_RVOL1.0
    # (which stays on legacy exit behavior) so the two are directly A/B
    # comparable via filter_attribution.py before any real-capital exit
    # behavior changes — see the regime-decay finding motivating this change
    # (Structural_v3.2 BREAKOUT flipped from +0.8..+2.1R/day to -0.1..-0.5R/day
    # once the market turned choppy in late July 2026).
    registry.register(Experiment(
        name="Structural_v3.3_ExitMgmt",
        strategy=StructuralStrategy(rvol_threshold=1.0, min_zone_score=50.0),
        params={
            "rvol_threshold": 1.0,
            "min_zone_score": 50.0,
            "exit_management": {
                "structure_invalidation": True,
                "atr_adaptive_trailing": True,
                "tp_expansion_cap": 3,
                "time_stop_bars": 24,
                "time_stop_min_r": 0.3,
            },
        },
        description="Structural_v3.2_RVOL1.0 entry logic + context-aware exit management (shadow-only A/B clone)"
    ))

    # 2. EMA Pullback
    registry.register(Experiment(
        name="EMA_Pullback_20_50_RVOL0.5",
        strategy=EmaPullbackStrategy(rvol_threshold=0.5, min_efficiency=0.45),
        params={"rvol_threshold": 0.5, "min_efficiency": 0.45},
        description="EMA Pullback — RVOL 0.5x (pullbacks are quiet by nature), efficiency 0.45"
    ))

    # 3. VWAP Reversion
    registry.register(Experiment(
        name="VWAP_Reversion_1.5ATR_RVOL1.0",
        strategy=VwapReversionStrategy(rvol_threshold=1.0, vwap_stretch_multiplier=1.5),
        params={"rvol_threshold": 1.0, "vwap_stretch_multiplier": 1.5},
        description="VWAP Reversion strategy — RVOL threshold 1.0x"
    ))

    # 4. Previous Day High/Low
    registry.register(Experiment(
        name="PrevDay_Extremes_RVOL1.2",
        strategy=PrevDayExtremesStrategy(breakout_rvol_threshold=1.2, reversal_rvol_threshold=1.0),
        params={"breakout_rvol_threshold": 1.2, "reversal_rvol_threshold": 1.0, "proximity_multiplier": 0.3},
        description="Previous Day High/Low sweeps and breakouts"
    ))

    # 5. ORB (15m and 30m) — retired 2026-08-08: 0 real trades in 11 days
    # despite running as CF the whole time (25 & 31 CF trades), and net
    # negative CF pnl_r (-15.53 / -8.17). Not a thin sample, just losing.

    # 6. ATR Squeeze Breakout
    registry.register(Experiment(
        name="ATR_Squeeze_RVOL1.5",
        strategy=AtrSqueezeStrategy(rvol_threshold=1.5, atr_percentile_threshold=0.20),
        params={"rvol_threshold": 1.5, "atr_percentile_threshold": 0.20},
        description="ATR Squeeze — RVOL 1.5x (requires actual expansion volume at breakout)"
    ))

    # 7. Geometry Strategy — purely from MKE Stage 5 GeometryContext
    registry.register(Experiment(
        name="Geometry_v1.0_Score35",
        strategy=GeometryStrategy(
            min_confluence_score=35.0,
            zone_tolerance_pct=0.002,
            min_body_fraction=0.40,
            min_bias_confidence=0.45,
            atr_sl_buffer_mult=0.15,
            tp_atr_cap=3.0,
            min_rr=1.5,
            trendline_break_enabled=True,
        ),
        params={
            "min_confluence_score": 35.0,
            "zone_tolerance_pct": 0.002,
            "min_body_fraction": 0.40,
            "min_bias_confidence": 0.45,
            "min_rr": 1.5,
            "trendline_break_enabled": True,
        },
        description="Geometry Strategy v1.0 — confluence bounce + trendline retest. Score threshold=35 (loose arm; A/B against Score50)."
    ))

    registry.register(Experiment(
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
    ))

    # 8. Order Flow Strategy (Milestone 2C)
    registry.register(Experiment(
        name="OrderFlow_v1.0",
        strategy=OrderFlowStrategy(
            min_sweep_confidence=0.60,
            min_imb_confidence=0.55,
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
    ))

    # 9. Chart Pattern Strategy — retired 2026-08-08: both confidence
    # variants (Conf55, Conf40) ran 7 days with large CF samples (92 & 91
    # trades) and net negative pnl_r (-10.55 / -13.58). Consistent loser,
    # not a filter-threshold problem.

    # 10. VWAP Reclaim — trend-continuation on a VWAP cross.
    registry.register(Experiment(
        name="VWAP_Reclaim_v1.0",
        strategy=VwapReclaimStrategy(rvol_threshold=1.0, min_efficiency=0.45),
        params={"rvol_threshold": 1.0, "min_efficiency": 0.45},
        description="VWAP Reclaim — trend continuation on a VWAP cross (efficiency 0.45)"
    ))

    # 11. CPR (Central Pivot Range) Breakout
    registry.register(Experiment(
        name="CPR_v1.0",
        strategy=CprStrategy(rvol_threshold=1.1, min_efficiency=0.55),
        params={"rvol_threshold": 1.1, "min_efficiency": 0.55},
        description="Central Pivot Range breakout — prior-day TC/BC value area"
    ))

    # 12. Gap — v2.0
    registry.register(Experiment(
        name="GapRegime_v2.0",
        strategy=GapStrategy(gap_threshold_pct=0.4, rvol_threshold=1.1, min_efficiency=0.55),
        params={"gap_threshold_pct": 0.4, "rvol_threshold": 1.1, "min_efficiency": 0.55},
        description="Gap-and-Go / Gap-Fill, regime-gated to actual GAP days"
    ))

    # 13. Initial Balance Breakout
    registry.register(Experiment(
        name="ORB_60m_IB_RVOL1.2",
        strategy=OrbStrategy(rvol_threshold=1.2, opening_range_minutes=60),
        params={"rvol_threshold": 1.2, "opening_range_minutes": 60},
        description="Initial Balance Breakout — 60-minute opening range"
    ))

    # 14. Vertical Spread — first combo (multi-leg) experiment in this framework.
    registry.register(Experiment(
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
    ))

    # 15. Straddle/Strangle — long volatility on realized-vol compression.
    registry.register(Experiment(
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
    ))

    registry.register(Experiment(
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
    ))

    # 16. Credit Spread — PCR-extreme contrarian thesis, theta-positive.
    registry.register(Experiment(
        name="CreditSpread_v1.0_PCRFade",
        strategy=CreditSpreadStrategy(
            rvol_ceiling=1.3, max_efficiency=0.55,
            spread_width_strikes=2, target_r=0.5, stop_r=-1.0,
        ),
        params={
            "rvol_ceiling": 1.3, "max_efficiency": 0.55,
            "spread_width_strikes": 2, "target_r": 0.5, "stop_r": -1.0,
        },
        description="Bull Put Spread / Bear Call Spread — PCR-extreme thesis, credit-spread execution"
    ))

    # Channel bounce/breakout — retired 2026-08-08: net negative pnl_r
    # (-6.81) over its 3 days running, on a meaningful 33-trade CF sample.

    # OI-wall reaction — consumes MarketContext.options (real OI).
    registry.register(Experiment(
        name="OIWallReaction_v1.0",
        strategy=OIWallReactionStrategy(
            zone_tolerance_pct=0.0015, min_body_fraction=0.40,
            atr_sl_buffer_mult=0.15, breakout_rvol_threshold=1.3,
            tp_atr_cap=3.0, min_rr=1.5,
        ),
        params={
            "zone_tolerance_pct": 0.0015, "min_body_fraction": 0.40,
            "atr_sl_buffer_mult": 0.15, "breakout_rvol_threshold": 1.3,
            "tp_atr_cap": 3.0, "min_rr": 1.5,
        },
        description="Fade or breakout reaction to real option-chain OI walls (call/put strikes with outlier OI)"
    ))

    # PCR-extreme contrarian reversal
    registry.register(Experiment(
        name="PCRExtremeReversal_v1.0",
        strategy=PCRExtremeReversalStrategy(
            min_confluence_score=40.0, zone_tolerance_pct=0.0015,
            min_body_fraction=0.40, atr_sl_buffer_mult=0.15,
            tp_atr_cap=3.0, min_rr=1.5,
        ),
        params={
            "min_confluence_score": 40.0, "zone_tolerance_pct": 0.0015,
            "min_body_fraction": 0.40, "atr_sl_buffer_mult": 0.15,
            "tp_atr_cap": 3.0, "min_rr": 1.5,
        },
        description="Contrarian reversal on PCR extremes, gated by zone confluence + candle confirmation"
    ))

    # 17. Iron Condor — Sideways/Range market income play
    registry.register(Experiment(
        name="IronCondor_v1.0",
        strategy=IronCondorStrategy(
            rvol_ceiling=1.3, max_efficiency=0.55,
            spread_width_strikes=2, target_r=0.4, stop_r=-1.0,
        ),
        params={
            "rvol_ceiling": 1.3, "max_efficiency": 0.55,
            "spread_width_strikes": 2, "target_r": 0.4, "stop_r": -1.0,
        },
        description="Iron Condor (OTM Call spread + OTM Put spread) sideways credit play"
    ))

    # 18. Butterfly Spread — Sideways/Range market defined-risk debit play
    registry.register(Experiment(
        name="Butterfly_v1.0",
        strategy=ButterflyStrategy(
            rvol_ceiling=1.3, max_efficiency=0.55,
            wing_width_strikes=2, target_r=1.5, stop_r=-0.5,
        ),
        params={
            "rvol_ceiling": 1.3, "max_efficiency": 0.55,
            "wing_width_strikes": 2, "target_r": 1.5, "stop_r": -0.5,
        },
        description="Butterfly Spread (Long ITM Call + 2x Short ATM Call + Long OTM Call) sideways debit play"
    ))

    # 18b. Iron Butterfly — ATM-centered credit theta-harvest.
    registry.register(Experiment(
        name="IronButterfly_v1.0",
        strategy=IronButterflyStrategy(
            rvol_ceiling=1.2, max_efficiency=0.50,
            wing_width_strikes=4, target_r=0.35, stop_r=-1.0,
        ),
        params={
            "rvol_ceiling": 1.2, "max_efficiency": 0.50,
            "wing_width_strikes": 4, "target_r": 0.35, "stop_r": -1.0,
        },
        description="Iron Butterfly (Sell ATM Call + ATM Put, buy wings) ATM credit theta-harvest"
    ))

    # v3.1: OI Scalping (PAPER)
    registry.register(Experiment(
        name="OI_Scalping_v1.0",
        strategy=OptionsScalpingStrategy(
            stop_loss_pct=0.50, target_multiple=2.0,
            min_rvol=1.5, min_votes=3, lookback_minutes=10,
        ),
        params={"stop_loss_pct": 0.50, "target_multiple": 2.0, "min_rvol": 1.5},
        description=(
            "PAPER: OI×premium 4-quadrant positioning inference scalper. "
            "3/5 windows must agree on direction. BSM Greeks required. "
            "Exit: bid-based premium stop/target + 15-min time stop."
        ),
    ))

    # v3.1: Consolidation Breakout Standard (PAPER)
    registry.register(Experiment(
        name="Consolidation_Breakout_v1.0",
        strategy=ConsolidationBreakoutStrategy(
            rvol_threshold=1.5, breakout_score_min=60,
            atr_pct_threshold=30.0, min_touches=3,
        ),
        params={"rvol_threshold": 1.5, "breakout_score_min": 60},
        description=(
            "PAPER: 1H consolidation squeeze + M5 breakout + RVOL>=1.5. "
            "RSI Momentum Confirmation. Explicit 2R SL/TP. "
            "Boundary candles excluded from touch count."
        ),
    ))

    # v3.1: Consolidation Breakout Tight (PAPER)
    registry.register(Experiment(
        name="Consolidation_Breakout_Tight_v1.0",
        strategy=ConsolidationBreakoutStrategy(
            rvol_threshold=2.0, breakout_score_min=60,
            atr_pct_threshold=30.0, min_touches=3,
        ),
        params={"rvol_threshold": 2.0, "breakout_score_min": 60},
        description=(
            "PAPER: Same as Consolidation_Breakout_v1.0 but RVOL>=2.0. "
            "A/B against standard to test whether stronger participation "
            "improves profit factor. Data decides."
        ),
    ))

    # RSI-2 Mean Reversion
    registry.register(Experiment(
        name="RSI2_MeanReversion_v1.0",
        strategy=Rsi2MeanReversionStrategy(
            rsi_oversold=10.0, rsi_overbought=90.0, min_body_fraction=0.40,
            atr_sl_buffer_mult=0.15, tp_atr_cap=3.0, min_rr=1.5, rvol_ceiling=1.5,
        ),
        params={
            "rsi_oversold": 10.0, "rsi_overbought": 90.0, "min_body_fraction": 0.40,
            "atr_sl_buffer_mult": 0.15, "tp_atr_cap": 3.0, "min_rr": 1.5, "rvol_ceiling": 1.5,
        },
        description="RSI-2 extreme fade (<10 / >90), confirmed by a reversal candle, targets EMA20"
    ))

    # Expiry-Aware Theta
    registry.register(Experiment(
        name="ExpiryAwareTheta_v1.0",
        strategy=ExpiryAwareThetaStrategy(
            rvol_ceiling_far=1.4, rvol_ceiling_near=0.8, max_efficiency=0.55,
            wing_width_far=2, wing_width_near=5,
            target_r_far=0.5, target_r_near=0.25, stop_r=-1.0,
        ),
        params={
            "rvol_ceiling_far": 1.4, "rvol_ceiling_near": 0.8, "max_efficiency": 0.55,
            "wing_width_far": 2, "wing_width_near": 5,
            "target_r_far": 0.5, "target_r_near": 0.25, "stop_r": -1.0,
        },
        description="Iron Condor with wing width/RVOL ceiling/target scaled continuously by time-to-expiry"
    ))

    # NIFTY-BankNifty Relative Value
    registry.register(Experiment(
        name="RelativeValue_NIFTY_BANKNIFTY_v1.0",
        strategy=RelativeValueStrategy(
            nifty_symbol="NSE:NIFTY50-INDEX", banknifty_symbol="NSE:NIFTYBANK-INDEX",
            lookback_bars=60, z_entry=2.0, min_rr=1.2, tp_ratio_reversion_fraction=0.6,
        ),
        params={
            "lookback_bars": 60, "z_entry": 2.0, "min_rr": 1.2,
            "tp_ratio_reversion_fraction": 0.6,
        },
        description="NIFTY/BankNifty ratio divergence from its own rolling mean — fade the rich leg, buy the cheap leg"
    ))

    # Momentum Burst (5m-only, no MTF gating) — deliberately fills the gap left
    # by MTF-gated strategies (Structural_v3.2 decays to -0.1..-0.5R/day in
    # choppy regimes per shadow-trade history) by reacting purely to 5m
    # range-expansion + RVOL burst + follow-through, no Daily/1H permission-seeking.
    registry.register(Experiment(
        name="MomentumBurst_5m_v1.0_RVOL2.0",
        strategy=MomentumBurst5mStrategy(
            range_atr_mult=1.8, min_body_fraction=0.55,
            rvol_burst_threshold=2.0, follow_through_giveback_pct=0.35,
            target_rr=2.2, sl_atr_floor_mult=0.6,
        ),
        params={
            "range_atr_mult": 1.8, "min_body_fraction": 0.55,
            "rvol_burst_threshold": 2.0, "follow_through_giveback_pct": 0.35,
            "target_rr": 2.2,
        },
        description="Pure 5m momentum-burst — range-expansion + RVOL burst + follow-through, no HTF gating"
    ))

    # HTF Pullback Reversal — Daily bias + 1H EMA20 pullback + 5M rejection
    # trigger. Distinct from Structural_v3.2 (zone/fractal SWEEP-BREAKOUT-TRAP
    # logic): this is a moving-average pullback-in-trend hypothesis, not a
    # structural-break hypothesis.
    registry.register(Experiment(
        name="HtfPullback_v1.0_Tol0.6pct",
        strategy=HtfPullbackReversalStrategy(
            pullback_tolerance_pct=0.006, sl_buffer_atr_mult=0.3,
            target_rr_floor=1.8, rvol_threshold=0.9,
        ),
        params={
            "pullback_tolerance_pct": 0.006, "sl_buffer_atr_mult": 0.3,
            "target_rr_floor": 1.8, "rvol_threshold": 0.9,
        },
        description="Daily bias + 1H EMA20 pullback + 5M rejection-candle trigger — swing continuation, not structural break"
    ))

    return registry
