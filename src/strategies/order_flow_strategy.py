#!/usr/bin/env python3
"""
OrderFlowStrategy — Institutional signal generation driven by Imbalances and Sweeps (M2C).
========================================================================================
Hypothesis:
    Stop sweeps at high-value liquidity pools (PDH/PDL, EQH/EQL) and pullbacks into
    unmitigated imbalances (FVGs) represent high-probability order-flow reversal
    and continuation zones.
"""

import logging
from datetime import datetime, time
from typing import List, Dict, Any, Optional, Tuple

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot
from src.core.market_geometry import NarrativeBias, GeometryContext
from src.core.market_liquidity import (
    LiquidityContext, Imbalance, LiquidityPool, LiquiditySweep, SweepType, SweepState, ImbalanceType
)
from src.core.market_patterns import PatternDirection

logger = logging.getLogger(__name__)


def _is_bullish_reversal_body(candle, min_body_fraction: float = 0.40) -> bool:
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    candle_range = h - l
    if candle_range < 1e-9:
        return False
    body = abs(c - o)
    return c > o and (body / candle_range) >= min_body_fraction


def _is_bearish_reversal_body(candle, min_body_fraction: float = 0.40) -> bool:
    o, c, h, l = float(candle["open"]), float(candle["close"]), float(candle["high"]), float(candle["low"])
    candle_range = h - l
    if candle_range < 1e-9:
        return False
    body = abs(c - o)
    return c < o and (body / candle_range) >= min_body_fraction


def _candidate_id(symbol: str, setup_type: str, price: float, ts: datetime) -> str:
    safe = symbol.replace(":", "_").replace("-", "_")
    return f"cand_{safe}_{setup_type}_{price:.2f}_{ts.strftime('%Y%m%d_%H%M%S')}"


class OrderFlowStrategy(BaseStrategy):
    """
    Order Flow Strategy v1.0.
    """

    metadata = StrategyMetadata(
        id="order_flow",
        name="Institutional Order Flow Strategy",
        hypothesis_id="order_flow_imbalances_sweeps",
        hypothesis_family="OrderFlow",
        hypothesis_text=(
            "Price respects institutional stop hunts (sweeps) and unfilled imbalances (FVGs). "
            "Reversal confirmations at these zones yield high-probability, high-R setups."
        ),
        version="v1.0",
        maturity="RESEARCH",
        tags=["order-flow", "liquidity", "imbalance", "fvg", "sweep"],
    )

    def __init__(
        self,
        min_sweep_confidence: float = 0.50,
        min_imb_confidence: float = 0.50,
        min_body_fraction: float = 0.40,
        atr_sl_buffer_mult: float = 0.15,
        tp_atr_cap: float = 3.0,
        min_rr: float = 1.5,
    ):
        self.min_sweep_confidence = min_sweep_confidence
        self.min_imb_confidence = min_imb_confidence
        self.min_body_fraction = min_body_fraction
        self.atr_sl_buffer_mult = atr_sl_buffer_mult
        self.tp_atr_cap = tp_atr_cap
        self.min_rr = min_rr
        logger.info(
            f"💧 OrderFlowStrategy initialized [sweep>={min_sweep_confidence}, imbalance>={min_imb_confidence}]"
        )

    def evaluate(
        self,
        snapshot: MarketSnapshot,
        experiment_name: str,
    ) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            # 1. Extract context
            market = snapshot.market
            liquidity: Optional[LiquidityContext] = getattr(market, "liquidity", None)
            geo: Optional[GeometryContext] = getattr(market, "geometry", None)

            if liquidity is None:
                return self._empty_result(
                    experiment_name,
                    errors=["LIQUIDITY_MISSING: snapshot.market.liquidity is None"],
                )

            # 2. Extract ATR
            atr: float = snapshot.features.get_float("atr") or 0.0
            if atr <= 0:
                return self._empty_result(
                    experiment_name,
                    errors=["FEATURE_MISSING:atr"],
                )

            price = snapshot.current_price
            ts = snapshot.timestamp
            m5_df = snapshot.m5
            if m5_df is None or len(m5_df) < 5:
                return self._empty_result(
                    experiment_name,
                    errors=["INSUFFICIENT_DATA:m5"],
                )

            last_candle = m5_df.iloc[-1]

            # 3. Time filters (Removed)
            global_rejections = []

            # 4. Narrative gate
            narrative = getattr(geo, "narrative", None)
            bias = narrative.bias if narrative else NarrativeBias.NEUTRAL
            bias_confidence = narrative.bias_confidence if narrative else 0.5

            # 5. Evaluate Sweep Reversal Setup
            sweep_sig = self._evaluate_sweep_setup(
                liquidity, price, atr, last_candle, ts, snapshot, experiment_name,
                bias, bias_confidence, global_rejections
            )
            if sweep_sig:
                signals.append(sweep_sig)

            # 6. Evaluate Imbalance Pullback Setup
            imb_sig = self._evaluate_imbalance_setup(
                liquidity, price, atr, last_candle, ts, snapshot, experiment_name,
                bias, bias_confidence, global_rejections
            )
            if imb_sig:
                signals.append(imb_sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[OrderFlowStrategy] Error evaluating {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.id,
            version=self.version,
            signals=signals,
            diagnostics={
                "min_sweep_confidence": self.min_sweep_confidence,
                "min_imb_confidence": self.min_imb_confidence,
                "min_rr": self.min_rr
            },
            errors=errors,
            warnings=warnings
        )

    def _evaluate_sweep_setup(
        self,
        liq: LiquidityContext,
        price: float,
        atr: float,
        last_candle,
        ts: datetime,
        snapshot: MarketSnapshot,
        experiment_name: str,
        bias: NarrativeBias,
        bias_confidence: float,
        global_rejections: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate reversal setup following a confirmed liquidity stop sweep."""
        if not liq.liq_map or not liq.liq_map.active_sweep:
            return None

        sw = liq.liq_map.active_sweep
        if sw.confidence < self.min_sweep_confidence:
            return None

        side = None
        if sw.type == SweepType.BULLISH: # Swept low stops, expecting price to go UP
            if _is_bullish_reversal_body(last_candle, self.min_body_fraction):
                side = "BUY CALL"
        else: # Bearish sweep, swept high stops, expecting price to go DOWN
            if _is_bearish_reversal_body(last_candle, self.min_body_fraction):
                side = "BUY PUT"

        if side is None:
            return None

        # Settle SL/TP levels
        if side == "BUY CALL":
            sl = sw.level_swept - (atr * self.atr_sl_buffer_mult)
            risk_dist = price - sl
            tp = self._find_buy_target(liq, price, atr)
            tp_dist = tp - price
        else:
            sl = sw.level_swept + (atr * self.atr_sl_buffer_mult)
            risk_dist = sl - price
            tp = self._find_sell_target(liq, price, atr)
            tp_dist = price - tp

        # ── FIX: Enforce minimum SL floor of 0.5×ATR from entry ──────────────
        min_sl_dist = atr * 0.5
        if side == "BUY CALL" and risk_dist < min_sl_dist:
            sl = price - min_sl_dist
            risk_dist = min_sl_dist
        elif side == "BUY PUT" and risk_dist < min_sl_dist:
            sl = price + min_sl_dist
            risk_dist = min_sl_dist
        # ─────────────────────────────────────────────────────────────────────

        if risk_dist <= 0:
            return None

        rr = round(tp_dist / risk_dist, 2)

        rejection_reasons = list(global_rejections)
        if side == "BUY CALL" and bias == NarrativeBias.REVERSAL and bias_confidence >= 0.55:
            rejection_reasons.append("NARRATIVE_BIAS_OPPOSED")
        elif side == "BUY PUT" and bias != NarrativeBias.REVERSAL and bias_confidence >= 0.55:
            # Narrative Reversal is positive (Bullish bounce target)
            pass

        if rr < self.min_rr:
            rejection_reasons.append(f"LOW_RR:{rr}")

        accepted = len(rejection_reasons) == 0
        tp1 = price + (risk_dist * 1.5) if side == "BUY CALL" else price - (risk_dist * 1.5)

        return self._build_signal_dict(
            setup_type="LIQUIDITY_SWEEP",
            side=side,
            price=price,
            sl=sl,
            tp=tp,
            tp1=tp1,
            rr=rr,
            confidence=sw.confidence,
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            symbol=snapshot.symbol,
            ts=ts,
            experiment_name=experiment_name,
            extra_diagnostics={
                "swept_level": round(sw.level_swept, 2),
                "swept_object": sw.swept_object_type,
                "volume_multiplier": sw.volume_multiplier,
                "rejection_wick_size": sw.rejection_wick_size
            }
        )

    def _evaluate_imbalance_setup(
        self,
        liq: LiquidityContext,
        price: float,
        atr: float,
        last_candle,
        ts: datetime,
        snapshot: MarketSnapshot,
        experiment_name: str,
        bias: NarrativeBias,
        bias_confidence: float,
        global_rejections: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate pullback bounce inside active unfilled imbalances."""
        if not liq.liq_map:
            return None

        # Check buy setups
        nearest_bull = liq.liq_map.nearest_bullish_imbalance
        if nearest_bull and nearest_bull.confidence >= self.min_imb_confidence:
            # Check price pulled back into FVG
            if price <= nearest_bull.top and price >= nearest_bull.bottom:
                if _is_bullish_reversal_body(last_candle, self.min_body_fraction):
                    sl = nearest_bull.bottom - (atr * self.atr_sl_buffer_mult)
                    risk_dist = price - sl
                    tp = self._find_buy_target(liq, price, atr)
                    tp_dist = tp - price
                    
                    if risk_dist > 0:
                        rr = round(tp_dist / risk_dist, 2)
                        rejection_reasons = list(global_rejections)
                        if rr < self.min_rr:
                            rejection_reasons.append(f"LOW_RR:{rr}")
                        
                        accepted = len(rejection_reasons) == 0
                        tp1 = price + (risk_dist * 1.5)
                        
                        return self._build_signal_dict(
                            setup_type="IMBALANCE_PULLBACK",
                            side="BUY CALL",
                            price=price,
                            sl=sl,
                            tp=tp,
                            tp1=tp1,
                            rr=rr,
                            confidence=nearest_bull.confidence,
                            accepted=accepted,
                            rejection_reasons=rejection_reasons,
                            symbol=snapshot.symbol,
                            ts=ts,
                            experiment_name=experiment_name,
                            extra_diagnostics={
                                "imbalance_id": nearest_bull.id,
                                "imbalance_top": nearest_bull.top,
                                "imbalance_bottom": nearest_bull.bottom,
                                "fill_percentage": nearest_bull.fill_percentage
                            }
                        )

        # Check sell setups
        nearest_bear = liq.liq_map.nearest_bearish_imbalance
        if nearest_bear and nearest_bear.confidence >= self.min_imb_confidence:
            if price >= nearest_bear.bottom and price <= nearest_bear.top:
                if _is_bearish_reversal_body(last_candle, self.min_body_fraction):
                    sl = nearest_bear.top + (atr * self.atr_sl_buffer_mult)
                    risk_dist = sl - price
                    tp = self._find_sell_target(liq, price, atr)
                    tp_dist = price - tp
                    
                    if risk_dist > 0:
                        rr = round(tp_dist / risk_dist, 2)
                        rejection_reasons = list(global_rejections)
                        if rr < self.min_rr:
                            rejection_reasons.append(f"LOW_RR:{rr}")
                        
                        accepted = len(rejection_reasons) == 0
                        tp1 = price - (risk_dist * 1.5)
                        
                        return self._build_signal_dict(
                            setup_type="IMBALANCE_PULLBACK",
                            side="BUY PUT",
                            price=price,
                            sl=sl,
                            tp=tp,
                            tp1=tp1,
                            rr=rr,
                            confidence=nearest_bear.confidence,
                            accepted=accepted,
                            rejection_reasons=rejection_reasons,
                            symbol=snapshot.symbol,
                            ts=ts,
                            experiment_name=experiment_name,
                            extra_diagnostics={
                                "imbalance_id": nearest_bear.id,
                                "imbalance_top": nearest_bear.top,
                                "imbalance_bottom": nearest_bear.bottom,
                                "fill_percentage": nearest_bear.fill_percentage
                            }
                        )

        return None

    def _find_buy_target(self, liq: LiquidityContext, price: float, atr: float) -> float:
        cap = price + (atr * self.tp_atr_cap)
        if liq.liq_map and liq.liq_map.nearest_liquidity_above:
            return min(liq.liq_map.nearest_liquidity_above.center_price, cap)
        return cap

    def _find_sell_target(self, liq: LiquidityContext, price: float, atr: float) -> float:
        cap = price - (atr * self.tp_atr_cap)
        if liq.liq_map and liq.liq_map.nearest_liquidity_below:
            return max(liq.liq_map.nearest_liquidity_below.center_price, cap)
        return cap

    def _build_signal_dict(
        self,
        setup_type: str,
        side: str,
        price: float,
        sl: float,
        tp: float,
        tp1: float,
        rr: float,
        confidence: float,
        accepted: bool,
        rejection_reasons: List[str],
        symbol: str,
        ts: datetime,
        experiment_name: str,
        extra_diagnostics: Dict[str, Any]
    ) -> Dict[str, Any]:
        cid = _candidate_id(symbol, setup_type, price, ts)
        sig = {
            "symbol": symbol,
            "candidate_id": cid,
            "signal": side,
            "price": price,
            "stop_loss": sl,
            "take_profit": tp,
            "tp1": tp1,
            "rr_ratio": rr,
            "strategy": setup_type,
            "confidence": confidence,
            "accepted": accepted,
            "rejection_reasons": rejection_reasons,
            "timestamp": ts,
            "diagnostics": extra_diagnostics
        }
        return self._tag_signal(sig, experiment_name)
