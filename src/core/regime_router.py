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

    # Hybrid / structural — unrestricted
    "Structural_v3.2_RVOL1.0":  "ANY",
    "Structural_v3.2_RVOL0.8":  "ANY",
    "PrevDay_Extremes_RVOL1.2": "ANY",
    "Geometry_v1.0_Score35":    "ANY",
    "Geometry_v1.0_Score50":    "ANY",
    "OrderFlow_v1.0":           "ANY",
    "OIWallReaction_v1.0":      "ANY",
}


def is_regime_eligible(experiment_name: str, regime_label: Optional[object]) -> bool:
    """True if `experiment_name` should be allowed to deploy REAL capital
    given the current `regime_label` (a RegimeLabel, or None/unavailable).

    Fails open (returns True) when the regime classifier is under-confident
    (UNKNOWN) or when an experiment has no declared affinity — a missing
    mapping should never silently block real trades.
    """
    affinity = EXPERIMENT_REGIME_AFFINITY.get(experiment_name, "ANY")
    if affinity == "ANY":
        return True

    if regime_label is None:
        return True

    category = REGIME_CATEGORY_MAP.get(getattr(regime_label, "primary", "UNKNOWN"), "UNKNOWN")
    if category == "UNKNOWN":
        return True

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
