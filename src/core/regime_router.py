#!/usr/bin/env python3
"""
Regime-Aware Capital Router
============================
Decides which of the registered experiments are eligible for REAL capital
in the current market regime. All experiments keep running as counterfactual
research trades every cycle regardless of regime — this router only affects
the real-vs-CF fork of an already-accepted signal (see indian_trader.py's
main signal loop). A regime-ineligible accepted signal is not discarded; it
is routed into the counterfactual path instead, so its outcome stays
measurable via filter_attribution.py, same as every other filter.
"""

from typing import Optional

# RegimeLabel.primary -> coarse routing category.
REGIME_CATEGORY_MAP = {
    "STRONG_TREND_UP":   "TREND_UP",
    "WEAK_TREND_UP":     "TREND_UP",
    "STRONG_TREND_DOWN": "TREND_DOWN",
    "WEAK_TREND_DOWN":   "TREND_DOWN",
    "RANGE":             "RANGE",
    "COMPRESSION":       "COMPRESSION",
    "GAP_UP":            "GAP",
    "GAP_DOWN":          "GAP",
    "UNKNOWN":           "UNKNOWN",
}

_TREND_CATEGORIES = {"TREND_UP", "TREND_DOWN"}

# Experiment name -> set of eligible categories, or "ANY" for structurally
# hybrid (bounce-or-break) experiments that shouldn't be regime-restricted.
EXPERIMENT_REGIME_AFFINITY = {
    # Trend-continuation
    "EMA_Pullback_20_50_RVOL0.5":      {"TREND_UP", "TREND_DOWN"},
    "VWAP_Reclaim_v1.0":               {"TREND_UP", "TREND_DOWN"},
    "VerticalSpread_v1.0":             {"TREND_UP", "TREND_DOWN"},

    # Breakout
    "ATR_Squeeze_RVOL1.5":             {"TREND_UP", "TREND_DOWN", "GAP", "COMPRESSION"},
    "CPR_v1.0":                       {"TREND_UP", "TREND_DOWN", "GAP"},
    "ORB_60m_IB_RVOL1.2":             {"TREND_UP", "TREND_DOWN", "GAP"},

    # Mean-reversion / fade
    "VWAP_Reversion_1.5ATR_RVOL1.0":  {"RANGE"},
    "PCRExtremeReversal_v1.0":        {"RANGE"},
    "CreditSpread_v1.0_PCRFade":      {"RANGE"},

    # Volatility / compression
    "Straddle_v1.0_VolCompression":   {"COMPRESSION"},
    "Strangle_v1.0_VolCompression":   {"COMPRESSION"},

    # Short-vol / theta harvest — all want low realized movement, so RANGE +
    # COMPRESSION only. Previously unmapped (fell through to fail-open "ANY"),
    # which meant these could sell premium on a day already classified TREND.
    "IronCondor_v1.0":                {"RANGE", "COMPRESSION"},
    "IronButterfly_v1.0":             {"RANGE", "COMPRESSION"},
    "ExpiryAwareTheta_v1.0":          {"RANGE", "COMPRESSION"},

    # Gap
    "GapRegime_v2.0":                 {"GAP"},

    # Momentum / fast-reversal — pure 5m, deliberately designed to catch fast
    # reversals that MTF strategies miss/arrive-late-to in choppy/range/
    # compression conditions and gap days where the open itself is the burst.
    # Excluded from TREND_UP/TREND_DOWN on purpose: in a confirmed trend,
    # Structural_v3.2 and the trend-continuation strategies already own that
    # regime — a momentum burst inside an already-strong trend is more likely
    # a late/exhausted move than a genuine new one. Provisional pending a real
    # backtest run (see experiment_factory.py); revisit once signal counts exist.
    "MomentumBurst_5m_v1.0_RVOL2.0":  {"RANGE", "COMPRESSION", "GAP"},

    # HTF Pullback — a trend-continuation-via-pullback hypothesis, so it
    # belongs with EMA_Pullback_20_50/VWAP_Reclaim/VerticalSpread, not "ANY"
    # like Structural_v3.2 (deliberately hybrid bounce-or-break, regime-agnostic
    # by design). Provisional pending a real backtest run.
    "HtfPullback_v1.0_Tol0.6pct":     {"TREND_UP", "TREND_DOWN"},

    # Hybrid / structural — unrestricted
    "Structural_v3.2_RVOL1.0":  "ANY",
    "Structural_v3.2_RVOL0.8":  "ANY",

    # Shadow-only A/B clone of Structural_v3.2_RVOL1.0 testing the new
    # context-aware exit management (see indian_trader.py's exit_mgmt).
    # Empty set == never eligible for real capital under a classified regime
    # (`category not in affinity` is always True for an empty set) — every
    # accepted signal routes to counterfactual only, until the exit-management
    # fix is validated via a replay-harness comparison against the baseline.
    # An UNKNOWN/unclassified regime now also fails CLOSED for any explicitly
    # scoped experiment (see is_regime_eligible) — this is a true "never real
    # capital" guarantee, not an interim one.
    "Structural_v3.3_ExitMgmt": set(),
    "PrevDay_Extremes_RVOL1.2": "ANY",
    "Geometry_v1.0_Score35":    "ANY",
    "Geometry_v1.0_Score50":    "ANY",
    "OrderFlow_v1.0":           "ANY",
    "OIWallReaction_v1.0":      "ANY",
}


def is_regime_eligible(experiment_name: str, regime_label: Optional[object]) -> bool:
    """True if `experiment_name` should be allowed to deploy REAL capital
    given the current `regime_label` (a RegimeLabel, or None/unavailable).

    Fails open (returns True) only when an experiment has no declared
    affinity, or explicitly declares "ANY" — a missing mapping should never
    silently block real trades for an experiment nobody has scoped.

    For experiments with an explicitly declared, non-"ANY" affinity set,
    an under-confident regime read (UNKNOWN, or no regime_label at all)
    fails CLOSED instead: a strategy that was deliberately scoped to e.g.
    RANGE-only should not get real capital just because the classifier
    couldn't confidently label the regime this cycle.
    """
    affinity = EXPERIMENT_REGIME_AFFINITY.get(experiment_name, "ANY")
    if affinity == "ANY":
        return True

    if regime_label is None:
        return False

    category = REGIME_CATEGORY_MAP.get(getattr(regime_label, "primary", "UNKNOWN"), "UNKNOWN")
    if category == "UNKNOWN":
        return False

    if category not in affinity:
        return False

    # For trend-direction categories specifically, also require h1 structure
    # doesn't outright contradict the m5-derived trend direction — an m5
    # STRONG_TREND_UP that h1 structure disagrees with isn't a confirmed
    # trend worth deploying trend-continuation capital into.
    if category in _TREND_CATEGORIES:
        h1_aligned = getattr(regime_label, "h1_trend_aligned", None)
        if h1_aligned is False:
            return False

    return True
