#!/usr/bin/env python3
"""
Options Scalping Strategy — Greeks + OI Positioning Inference (v1.0)
=====================================================================
Hypothesis: Combining per-snapshot OI change % with premium direction
classifies market positioning (Long Buildup / Short Buildup / Short Covering /
Long Unwind) with enough reliability to scalp ATM options directionally.

Signal: 3 of the last 5 non-FLAT positioning windows must agree on a
direction. Spot movement and RVOL confirm entry. Exit is premium-based
(not underlying-move based) with a 15-minute time stop.

Registered as: OI_Scalping_v1.0 (PAPER)

Key design decisions:
- oi_change from the DB is a daily cumulative field (oi - pdoi).
  We use consecutive raw oi diffs instead.
- Thresholds are SEED values from one trading day (2026-08-11, NIFTY 24400CE).
  See THRESHOLD_SOURCE for recalibration history.
- IV-invalid observations → rejected (accepted=False), NOT silently downgraded.
- Single-leg directional trades only. No delta-neutral logic.
- Exit uses executable prices (bid for long positions), not LTP.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.core.base_strategy import BaseStrategy, StrategyResult, StrategyMetadata
from src.core.market_snapshot import MarketSnapshot, OptionSnapshotRow
from src.core.bsm_utils import (
    validate_option_price, solve_iv_brent, bsm_greeks, bsm_price
)
from src.core.indicator_pipeline import IndicatorPipeline

logger = logging.getLogger(__name__)


# ── Seed thresholds (NOT calibrated edge parameters) ─────────────────────────
# Source: 330 snapshots, NIFTY 24400CE, 2026-08-11 only.
# 1 day · 1 strike · 1 CE · 1 regime = seed values, not validated edge.
# Recalibrate via scripts/calibrate_oi_thresholds.py after 30 trading days.
# Promote via A/B experiment; do not auto-update.
THRESHOLD_SOURCE     = "seed_2026-08-11_NIFTY_24400CE"
OI_UP_THRESHOLD      = +0.010    # > +1.0% per snapshot (empirical p75)
OI_DOWN_THRESHOLD    = -0.007    # < -0.7% per snapshot (empirical p25)
PREM_UP_THRESHOLD    = +0.018    # > +1.8% per snapshot (empirical p75)
PREM_DOWN_THRESHOLD  = -0.020    # < -2.0% per snapshot (empirical p25)

# Quote validation
MAX_SPREAD_RATIO     = 0.05      # reject if (ask-bid)/mid > 5%

# Positioning vote
BULLISH_POSITIONING  = frozenset({"LONG_BUILDUP_INFERENCE", "SHORT_COVERING_INFERENCE"})
BEARISH_POSITIONING  = frozenset({"SHORT_BUILDUP_INFERENCE", "LONG_UNWIND_INFERENCE"})

# Risk / reward
LOT_SIZES = {"NIFTY": 75, "BANKNIFTY": 30}
RISK_FREE_RATE = 0.065           # INR risk-free rate (6.5%)

# Exit thresholds
MIN_DELTA_LONG       = 0.25      # exit long if delta falls below this (gone OTM)
TIME_STOP_MINUTES    = 15        # exit if no progress after N minutes
TIME_STOP_MIN_R      = 0.30      # minimum R gain required to avoid time stop
NEAR_EXPIRY_MINUTES  = 15        # don't trade within 15 min of expiry


# ── Pure functions (testable, no side effects) ────────────────────────────────

def validate_quote(row: OptionSnapshotRow) -> Optional[OptionSnapshotRow]:
    """None = reject (zero bid, crossed market, or spread > 5%)."""
    if row.bid <= 0 or row.ask <= 0:
        return None
    if row.ask < row.bid:
        return None
    if row.spread_ratio > MAX_SPREAD_RATIO:
        return None
    return row


def compute_oi_change_pct(rows: List[OptionSnapshotRow]) -> List[float]:
    """
    Per-snapshot OI percentage change: (oi_t - oi_{t-1}) / oi_{t-1}.

    Returns a list of length len(rows)-1.

    NOTE: dimensionless per-snapshot % change, NOT velocity.
    The thresholds above were calibrated against this definition.
    """
    deltas = []
    for i in range(1, len(rows)):
        prev = rows[i - 1].oi
        if prev <= 0:
            deltas.append(0.0)
            continue
        deltas.append((rows[i].oi - prev) / prev)
    return deltas


def classify_positioning_inference(doi_pct: float, dprem_pct: float) -> str:
    """
    4-quadrant OI × premium positioning heuristic.
    Returns inference label — NOT ground truth about actual trader intent.
    """
    oi_up   = doi_pct  >= OI_UP_THRESHOLD
    oi_down = doi_pct  <= OI_DOWN_THRESHOLD
    pr_up   = dprem_pct >= PREM_UP_THRESHOLD
    pr_down = dprem_pct <= PREM_DOWN_THRESHOLD

    if oi_up   and pr_up:   return "LONG_BUILDUP_INFERENCE"
    if oi_up   and pr_down: return "SHORT_BUILDUP_INFERENCE"
    if oi_down and pr_up:   return "SHORT_COVERING_INFERENCE"
    if oi_down and pr_down: return "LONG_UNWIND_INFERENCE"
    return "FLAT"


def resolve_directional_vote(
    inferences: List[str],
    min_votes: int = 3,
) -> Optional[str]:
    """
    Examines the latest 5 windows; FLAT windows are excluded from the vote count.
    Requires min_votes bullish or bearish inferences in the examined set.
    Returns 'BULLISH' | 'BEARISH' | None.
    """
    recent = inferences[-5:]
    active = [x for x in recent if x != "FLAT"]
    if len(active) < min_votes:
        return None
    bull = sum(1 for x in active if x in BULLISH_POSITIONING)
    bear = sum(1 for x in active if x in BEARISH_POSITIONING)
    if bull >= min_votes:
        return "BULLISH"
    if bear >= min_votes:
        return "BEARISH"
    return None


def _lot_size(symbol: str) -> int:
    if "BANK" in symbol.upper():
        return LOT_SIZES["BANKNIFTY"]
    return LOT_SIZES["NIFTY"]


def estimate_costs(
    entry_ask: float,
    exit_bid: float,
    lot_size: int,
    risk_per_lot: float,
) -> Dict[str, float]:
    """
    Statutory transaction costs for NSE index options (single-leg long):
    brokerage, STT, exchange charges, GST. Returns costs in rupees and as a
    fraction of risk_per_lot (in R units).

    Deliberately excludes bid/ask spread cost. A prior version tried to
    derive it from entry_ask/exit_bid alone via an implied "mid" — but those
    two numbers can't disentangle "cost of the spread" from "the market
    moved", so on any winning trade (exit_bid > entry_ask, the normal case
    for a long option hitting target) it came out strongly *negative*,
    silently crediting instead of charging. It was also redundant: the real
    round-trip spread cost is already priced into pnl_r via the ask-in/
    bid-out fill convention (_premium_pnl_r / realistic_fill_price in
    indian_trader.py) — charging it again here would double-count it even
    if the formula were fixed. Removed rather than patched.
    """
    brokerage = 20.0 * 2                                    # ₹20 flat per leg
    stt       = exit_bid * lot_size * 0.000625             # 0.0625% on sell
    exchange  = (entry_ask + exit_bid) * lot_size * 0.0006 # ~0.06% turnover
    gst       = (brokerage + exchange) * 0.18

    total = brokerage + stt + exchange + gst
    cost_in_r = total / risk_per_lot if risk_per_lot > 0 else 0.0

    return {
        "brokerage":    round(brokerage,   2),
        "stt":          round(stt,         2),
        "exchange":     round(exchange,    2),
        "gst":          round(gst,         2),
        "total_cost":   round(total,       2),
        "cost_in_R":    round(cost_in_r,   4),
        "cost_2x_R":    round(cost_in_r * 2, 4),
        "cost_3x_R":    round(cost_in_r * 3, 4),
    }


# ── Strategy ──────────────────────────────────────────────────────────────────

class OptionsScalpingStrategy(BaseStrategy):
    """
    OI × premium positioning inference scalper. PAPER maturity.
    Single-leg long call or long put based on 3/5 positioning vote + spot confirmation.
    """

    metadata = StrategyMetadata(
        id="oi_scalping",
        name="OI Scalping Strategy",
        hypothesis_id="oi_premium_positioning_inference",
        hypothesis_family="OptionsIntelligence",
        hypothesis_text=(
            "Combining per-snapshot OI change % with premium direction classifies "
            "market positioning with enough reliability to scalp ATM options. "
            "3/5 consecutive windows agreeing + spot confirmation + RVOL ≥ 1.5 "
            "produces positive expectancy after transaction costs."
        ),
        version="v1.0",
        archetype="OptionsFlow",
        # Was PREMIUM_UNWIRED: the signal previously had no stop_loss/take_profit/
        # rr_ratio, which crashed indian_trader.py's market_loop() the moment this
        # signal was ever accepted (see options_scalping_strategy.py's STEP 9 —
        # stop_premium/target_premium are now also mapped into index-price terms
        # via entry delta so the shared single-leg engine can manage the position).
        exit_profile="INDEX_TP_EXPANSION",
        maturity="PAPER",
        tags=["options", "oi", "premium", "scalp", "greeks", "delta_mapped_exit"],
        expected_holding=(5, 15),
    )

    def __init__(
        self,
        stop_loss_pct: float = 0.50,      # exit if premium drops to 50% of entry
        target_multiple: float = 2.0,      # exit at 2× entry premium
        min_rvol: float = 1.5,
        min_delta_long: float = MIN_DELTA_LONG,
        time_stop_minutes: int = TIME_STOP_MINUTES,
        min_votes: int = 3,
        lookback_minutes: int = 10,
    ):
        self.stop_loss_pct    = stop_loss_pct
        self.target_multiple  = target_multiple
        self.min_rvol         = min_rvol
        self.min_delta_long   = min_delta_long
        self.time_stop_minutes = time_stop_minutes
        self.min_votes        = min_votes
        self.lookback_minutes = lookback_minutes

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _tte(self, now: datetime, symbol: str) -> float:
        """Time to expiry in years. Returns 0.0 if expiry info unavailable."""
        try:
            from src.core.options_mapper import get_expiry_datetime
            expiry_dt = get_expiry_datetime(symbol)
            if expiry_dt is None:
                return 7.0 / 365.0   # safe fallback: assume 1 week
            tte_sec = (expiry_dt - now).total_seconds()
            return max(tte_sec / (365 * 24 * 3600), 1e-6)
        except Exception:
            return 7.0 / 365.0

    def _tte_minutes(self, now: datetime, symbol: str) -> float:
        return self._tte(now, symbol) * 365 * 24 * 60

    def _resolve_atm_rows(
        self, chain: List[OptionSnapshotRow], spot: float
    ) -> Tuple[List[OptionSnapshotRow], List[OptionSnapshotRow]]:
        """Return recent rows for the ATM strike CE and PE."""
        # Determine ATM strike
        strikes = sorted(set(r.strike for r in chain))
        if not strikes:
            return [], []
        atm_strike = min(strikes, key=lambda k: abs(k - spot))

        ce_rows = [r for r in chain if r.strike == atm_strike and r.option_type == "CE"]
        pe_rows = [r for r in chain if r.strike == atm_strike and r.option_type == "PE"]
        return ce_rows, pe_rows

    def _infer_positioning_sequence(
        self, rows: List[OptionSnapshotRow]
    ) -> List[str]:
        """Compute per-snapshot positioning inference labels."""
        if len(rows) < 2:
            return []

        oi_deltas   = compute_oi_change_pct(rows)
        prem_deltas = []
        for i in range(1, len(rows)):
            prev_mid = rows[i - 1].mid
            curr_mid = rows[i].mid
            if prev_mid <= 0:
                prem_deltas.append(0.0)
            else:
                prem_deltas.append((curr_mid - prev_mid) / prev_mid)

        return [
            classify_positioning_inference(doi, dp)
            for doi, dp in zip(oi_deltas, prem_deltas)
        ]

    # ── Main evaluate() ────────────────────────────────────────────────────────

    def evaluate(self, snapshot: MarketSnapshot, experiment_name: str) -> StrategyResult:
        errors: List[str] = []
        warnings: List[str] = []
        signals: List[Dict[str, Any]] = []

        try:
            # ── STEP 1: FETCH DATA ───────────────────────────────────────────
            chain = snapshot.atm_chain
            if chain is None or len(chain) < 5:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_OI_DATA"])

            rvol = snapshot.volume_report.rvol_tod if snapshot.volume_report else 0.0
            spot = snapshot.current_price
            now  = snapshot.timestamp
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)

            # ── STEP 2: EXPIRY GUARD ─────────────────────────────────────────
            # We infer the underlying symbol from the snapshot symbol
            tte_min = self._tte_minutes(now, snapshot.symbol)
            if tte_min <= NEAR_EXPIRY_MINUTES:
                return self._empty_result(experiment_name, errors=["NEAR_EXPIRY"])

            # ── STEP 3: SPLIT ATM CE / PE ────────────────────────────────────
            ce_rows, pe_rows = self._resolve_atm_rows(chain, spot)
            if len(ce_rows) < 3 or len(pe_rows) < 3:
                return self._empty_result(experiment_name, errors=["INSUFFICIENT_ATM_DATA"])

            # ── STEP 4: VALIDATE LATEST QUOTES ──────────────────────────────
            latest_ce = validate_quote(ce_rows[-1])
            latest_pe = validate_quote(pe_rows[-1])
            if latest_ce is None and latest_pe is None:
                return self._empty_result(experiment_name, warnings=["STALE_OR_CROSSED_QUOTES"])

            # ── STEP 5: POSITIONING INFERENCE ───────────────────────────────
            ce_inferences = self._infer_positioning_sequence(ce_rows)
            pe_inferences = self._infer_positioning_sequence(pe_rows)

            ce_vote = resolve_directional_vote(ce_inferences, self.min_votes)
            pe_vote = resolve_directional_vote(pe_inferences, self.min_votes)

            # ── STEP 6: DIRECTION RESOLUTION ─────────────────────────────────
            # CE BULLISH + spot rising → BUY CALL
            # PE BULLISH + spot falling → BUY PUT
            # (Long buildup in CE or short covering in CE → buy CE)
            # (Long buildup in PE or short covering in PE → buy PE)
            direction = None
            target_row = None
            option_type = None

            spot_delta_pct = 0.0
            if snapshot.m5 is not None and len(snapshot.m5) >= 2:
                prev_close = float(snapshot.m5.iloc[-2]["close"])
                if prev_close > 0:
                    spot_delta_pct = (spot - prev_close) / prev_close * 100

            if ce_vote == "BULLISH" and spot_delta_pct > 0 and latest_ce is not None:
                direction = "BUY CALL"
                target_row = latest_ce
                option_type = "CE"
            elif pe_vote == "BULLISH" and spot_delta_pct < 0 and latest_pe is not None:
                direction = "BUY PUT"
                target_row = latest_pe
                option_type = "PE"

            if direction is None:
                return self._empty_result(
                    experiment_name,
                    diagnostics={
                        "ce_vote": ce_vote,
                        "pe_vote": pe_vote,
                        "spot_delta_pct": round(spot_delta_pct, 4),
                    },
                )

            # ── STEP 7: RVOL CONFIRMATION ────────────────────────────────────
            rejection_reasons: List[str] = []
            if rvol < self.min_rvol:
                rejection_reasons.append(f"LOW_RVOL:{rvol:.2f}")

            # ── STEP 8: BSM GREEKS ────────────────────────────────────────────
            tte = self._tte(now, snapshot.symbol)
            mid = target_row.mid
            strike = target_row.strike

            price_ok, price_reason = validate_option_price(
                spot, strike, tte, RISK_FREE_RATE, mid, option_type
            )
            greeks_valid = False
            greeks = {}

            if not price_ok:
                warnings.append(f"PRICE_BOUNDS:{price_reason}")
                rejection_reasons.append("IV_UNSOLVABLE")
            else:
                iv = solve_iv_brent(spot, strike, tte, RISK_FREE_RATE, mid, option_type)
                if iv is None:
                    rejection_reasons.append("IV_UNSOLVABLE")
                    warnings.append("IV_UNSOLVABLE — trade rejected in v1")
                else:
                    greeks = bsm_greeks(spot, strike, tte, RISK_FREE_RATE, iv, option_type)
                    greeks_valid = True

            if not greeks_valid:
                # v1: reject IV-invalid trades outright
                return self._empty_result(
                    experiment_name,
                    warnings=warnings,
                    diagnostics={"option_type": option_type, "mid": mid},
                )

            # ── STEP 9: CONSTRUCT SIGNAL ─────────────────────────────────────
            entry_ask      = target_row.ask
            stop_premium   = round(entry_ask * (1.0 - self.stop_loss_pct), 2)
            target_premium = round(entry_ask * self.target_multiple, 2)
            lot            = _lot_size(snapshot.symbol)
            risk_per_unit  = entry_ask - stop_premium
            risk_per_lot   = risk_per_unit * lot

            # indian_trader.py's single-leg engine (_enter_position/_update_position)
            # decides SL/TP hits off INDEX price, not option premium — it has no
            # combo_legs-style alternate path for a premium-only signal. Without an
            # index-price stop_loss/take_profit/rr_ratio, an accepted signal here
            # crashes market_loop() at `sig['stop_loss']`. Translate the premium
            # stop/target into index-price terms via entry delta (linear
            # approximation — gamma/theta drift over the trade's life, but this is
            # a v1/PAPER strategy; an approximate index-mapped exit beats a crash).
            # The original premium fields are kept below for audit/diagnostics —
            # this mapping is purely to satisfy the shared exit engine's contract.
            MIN_DELTA_FOR_INDEX_MAPPING = 0.05
            abs_delta = abs(greeks.get("delta") or 0.0)
            if abs_delta < MIN_DELTA_FOR_INDEX_MAPPING:
                rejection_reasons.append(f"DELTA_TOO_LOW:{abs_delta:.3f}")
                index_stop_dist = 0.0
                index_target_dist = 0.0
            else:
                index_stop_dist = risk_per_unit / abs_delta
                index_target_dist = (target_premium - entry_ask) / abs_delta

            if direction == "BUY CALL":
                stop_loss = spot - index_stop_dist
                take_profit = spot + index_target_dist
                tp1 = spot + 1.5 * index_stop_dist
            else:
                stop_loss = spot + index_stop_dist
                take_profit = spot - index_target_dist
                tp1 = spot - 1.5 * index_stop_dist
            rr_ratio = round(index_target_dist / index_stop_dist, 3) if index_stop_dist > 0 else 0.0

            accepted = len(rejection_reasons) == 0
            candidate_id = (
                f"cand_{snapshot.symbol.replace(':', '_').replace('-', '_')}"
                f"_OI_SCALP_{direction.replace(' ', '_')}"
                f"_{now.strftime('%Y%m%d_%H%M%S')}"
            )

            sig: Dict[str, Any] = {
                "symbol":             snapshot.symbol,
                "signal":             direction,
                "strategy":           "OI_SCALP",
                "option_type":        option_type,
                "strike":             strike,
                "price":              spot,
                # ── Index-price terms for the shared single-leg exit engine ──
                "stop_loss":          round(stop_loss, 2),
                "take_profit":        round(take_profit, 2),
                "tp1":                round(tp1, 2),
                "rr_ratio":           rr_ratio,
                # ── Execution prices (executable, not LTP) ────────────────
                "entry_ask":          entry_ask,
                "stop_premium":       stop_premium,
                "target_premium":     target_premium,
                # ── Canonical R definition ────────────────────────────────
                "risk_per_unit":      round(risk_per_unit, 2),
                "risk_per_lot":       round(risk_per_lot, 2),
                # ── Greeks ───────────────────────────────────────────────
                "greeks_valid":       greeks_valid,
                "delta":              greeks.get("delta"),
                "gamma":              greeks.get("gamma"),
                "theta":              greeks.get("theta"),
                "iv":                 round(iv, 4) if greeks_valid else None,
                # P&L/cost tracking now happens downstream, at exit, in
                # indian_trader.py's _exit_position() — it calls estimate_costs()
                # below with the real exit fill and writes the breakdown into
                # pos['diagnostics']['costs'] / 'net_pnl_r' / 'gross_pnl_r_before_costs'
                # (persisted via the standard diagnostics JSON column on both
                # trade_performance and counterfactual_results). Nothing to
                # pre-populate here at signal time — the exit fill doesn't exist yet.
                # ── Signal metadata ───────────────────────────────────────
                "timestamp":          now.isoformat(),
                "accepted":           accepted,
                "rejection_reasons":  rejection_reasons,
                "candidate_id":       candidate_id,
                "lot_size":           lot,
                # ── Experiment provenance ─────────────────────────────────
                "strategy_version":   "OI_Scalping_v1.0",
                "threshold_source":   THRESHOLD_SOURCE,
                # ── Diagnostics ───────────────────────────────────────────
                "diagnostics": {
                    "ce_vote":          ce_vote,
                    "pe_vote":          pe_vote,
                    "ce_inferences":    ce_inferences[-5:],
                    "pe_inferences":    pe_inferences[-5:],
                    "spot_delta_pct":   round(spot_delta_pct, 4),
                    "rvol":             round(rvol, 3),
                    "mid":              round(mid, 2),
                    "spread_ratio":     round(target_row.spread_ratio, 4),
                    "tte_days":         round(tte * 365, 2),
                },
            }
            self._tag_signal(sig, experiment_name)
            signals.append(sig)

        except Exception as e:
            errors.append(f"ENGINE_ERROR:{type(e).__name__}:{e}")
            logger.error(f"[OptionsScalpingStrategy] {snapshot.symbol}: {e}", exc_info=True)

        return StrategyResult(
            experiment_name=experiment_name,
            strategy_id=self.metadata.id,
            version=self.metadata.version,
            signals=signals,
            diagnostics={},
            errors=errors,
            warnings=warnings,
        )
