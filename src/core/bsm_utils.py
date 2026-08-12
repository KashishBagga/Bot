#!/usr/bin/env python3
"""
BSM Utilities — Black-Scholes-Merton Greeks + Brent IV Solver
==============================================================
Pure-Python implementation. No scipy, py_vollib, or other optional deps.

Public API:
    validate_option_price(S, K, T, r, market_price, flag) -> (bool, reason)
    solve_iv_brent(S, K, T, r, market_price, flag)        -> Optional[float]
    bsm_greeks(S, K, T, r, sigma, flag)                   -> Dict[str, float]
    bsm_price(S, K, T, r, sigma, flag)                    -> float

Notes:
    - flag: "CE" (call) or "PE" (put)
    - T: time-to-expiry in years (e.g. 7/365)
    - r: risk-free rate (e.g. 0.065 for 6.5%)
    - All prices in same currency units (INR for NSE)
"""

import math
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_SQRT_2PI = math.sqrt(2 * math.pi)
_MIN_T    = 1e-6   # minimum time-to-expiry to avoid division by zero
_MIN_SIGMA = 0.001
_MAX_SIGMA = 10.0
_PRICE_TOL = 0.50  # ₹0.50 arbitrage-bound tolerance for stale/wide quotes


# ── Standard Normal helpers ────────────────────────────────────────────────────

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    """Abramowitz & Stegun approximation — max error 7.5e-8."""
    if x >= 0:
        k = 1.0 / (1.0 + 0.2316419 * x)
        poly = k * (0.319381530
                    + k * (-0.356563782
                           + k * (1.781477937
                                  + k * (-1.821255978
                                         + k * 1.330274429))))
        return 1.0 - _norm_pdf(x) * poly
    return 1.0 - _norm_cdf(-x)


# ── BSM Core ──────────────────────────────────────────────────────────────────

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
    sqrt_T = math.sqrt(max(T, _MIN_T))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bsm_price(S: float, K: float, T: float, r: float, sigma: float, flag: str) -> float:
    """Black-Scholes-Merton theoretical price."""
    T = max(T, _MIN_T)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    Kd = K * math.exp(-r * T)
    if flag == "CE":
        return S * _norm_cdf(d1) - Kd * _norm_cdf(d2)
    else:  # PE
        return Kd * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bsm_greeks(S: float, K: float, T: float, r: float, sigma: float, flag: str) -> Dict[str, float]:
    """
    Returns {delta, gamma, theta, vega} for one option.

    Theta is expressed in rupees-per-calendar-day (divided by 365).
    Vega is expressed per 1% move in IV.
    """
    T = max(T, _MIN_T)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T  = math.sqrt(T)
    pdf_d1  = _norm_pdf(d1)
    Kd      = K * math.exp(-r * T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega  = S * pdf_d1 * sqrt_T / 100.0   # per 1% IV move

    if flag == "CE":
        delta = _norm_cdf(d1)
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 - r * Kd * _norm_cdf(d2)) / 365.0
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 + r * Kd * _norm_cdf(-d2)) / 365.0

    return {
        "delta": round(delta, 6),
        "gamma": round(gamma, 8),
        "theta": round(theta, 4),
        "vega":  round(vega,  4),
    }


# ── Arbitrage Bounds Validation ───────────────────────────────────────────────

def validate_option_price(
    S: float, K: float, T: float, r: float,
    market_price: float, flag: str,
) -> Tuple[bool, Optional[str]]:
    """
    Check whether market_price satisfies BSM arbitrage bounds.
    Returns (True, None) if valid; (False, reason_str) if not.

    Bounds (with ₹0.50 tolerance for stale/wide quotes):
        CE: max(S - K·e^{-rT}, 0) ≤ price ≤ S
        PE: max(K·e^{-rT} - S, 0) ≤ price ≤ K·e^{-rT}
    """
    T = max(T, _MIN_T)
    Kd = K * math.exp(-r * T)

    if flag == "CE":
        intrinsic = max(S - Kd, 0.0)
        upper     = S
    else:
        intrinsic = max(Kd - S, 0.0)
        upper     = Kd

    if market_price < intrinsic - _PRICE_TOL:
        return False, f"BELOW_INTRINSIC(mkt={market_price:.2f},intrinsic={intrinsic:.2f})"
    if market_price > upper + _PRICE_TOL:
        return False, f"ABOVE_UPPER_BOUND(mkt={market_price:.2f},upper={upper:.2f})"
    return True, None


# ── Brent Bisection IV Solver ─────────────────────────────────────────────────

def solve_iv_brent(
    S: float, K: float, T: float, r: float,
    market_price: float, flag: str,
    iv_low: float = 0.001,
    iv_high: float = 10.0,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Implied volatility via Brent's method (bisection + secant hybrid).

    Advantages over Newton-Raphson:
      - Guaranteed convergence when root is bracketed (no Vega dependency)
      - Robust for deep ITM/OTM, short-dated options, low-vega situations
      - Falls back to bisection automatically when secant step is unreliable

    Returns None if:
      - No root in [iv_low, iv_high] (market_price outside model range)
      - T ≤ 0 (expired option)
      - Result outside sensible IV bounds [0.03, 5.0]
    """
    T = max(T, _MIN_T)
    if T < 1e-5:
        logger.debug("solve_iv_brent: T too small (%.6f) — option near expiry", T)
        return None

    f = lambda sigma: bsm_price(S, K, T, r, sigma, flag) - market_price
    fa, fb = f(iv_low), f(iv_high)

    if fa * fb > 0:
        # Root not bracketed — market price outside [BSM(iv_low), BSM(iv_high)]
        logger.debug(
            "solve_iv_brent: root not bracketed "
            "(fa=%.4f, fb=%.4f, mkt=%.2f, S=%.0f, K=%.0f)", fa, fb, market_price, S, K
        )
        return None

    # Brent's method core
    a, b = iv_low, iv_high
    fa, fb = f(a), f(b)

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    mflag = True
    s = 0.0
    d = 0.0

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant
            s = b - fb * (b - a) / (fb - fa)

        # Conditions under which bisection is used instead
        cond1 = not (3 * a + b) / 4 < s < b if b > a else not b < s < (3 * a + b) / 4
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d, c, fc = c, b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    iv = b
    if not (0.03 <= iv <= 5.0):
        logger.debug("solve_iv_brent: IV=%.4f outside sensible bounds [0.03, 5.0]", iv)
        return None
    return iv
