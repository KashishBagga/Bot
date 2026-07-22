#!/usr/bin/env python3
"""
Head & Shoulders Geometry Helpers
==================================
Isolates calculations for neckline fits, shoulder symmetry, head prominence,
and neckline breakouts to share between H&S and Inverse H&S detectors.
"""

from typing import Tuple


class NecklineBuilder:
    """Calculates linear neckline projection using TradingClock bar coordinates."""

    @staticmethod
    def build_neckline(
        bar_l: int, price_l: float, bar_r: int, price_r: float
    ) -> Tuple[float, float]:
        """
        Fits a line through two neckline pivots.
        Returns: (slope, intercept) where price = slope * bar_index + intercept
        """
        if bar_r == bar_l:
            return 0.0, price_l
        m = (price_r - price_l) / (bar_r - bar_l)
        c = price_l - m * bar_l
        return m, c

    @staticmethod
    def project(bar_index: int, m: float, c: float) -> float:
        """Projects neckline price at a specific bar index."""
        return m * bar_index + c


class ShoulderSymmetry:
    """Checks structural symmetry between left and right shoulders."""

    @staticmethod
    def check(
        ls_price: float, rs_price: float, head_price: float, max_ratio: float = 0.015
    ) -> Tuple[bool, float]:
        """
        Verify that LS and RS prices are within max_ratio of the Head price.
        Returns: (passed: bool, symmetry_score: float)
        """
        if head_price <= 0:
            return False, 0.0
        
        diff = abs(ls_price - rs_price)
        ratio = diff / head_price
        
        passed = ratio <= max_ratio
        # Score is 1.0 when perfectly symmetric (ratio=0), down to 0.0 at max_ratio
        score = max(0.0, min(1.0, 1.0 - (ratio / max_ratio)))
        return passed, round(score, 3)


class HeadProminence:
    """Checks prominence of the head relative to the shoulders."""

    @staticmethod
    def check(
        ls_price: float,
        rs_price: float,
        head_price: float,
        atr: float,
        is_inverse: bool = False,
        min_prominence_atr: float = 0.5
    ) -> Tuple[bool, float]:
        """
        Verify that the Head price exceeds the shoulders by at least min_prominence_atr * ATR.
        Returns: (passed: bool, prominence_score: float)
        """
        if atr <= 0:
            atr = 1.0

        if not is_inverse:
            # H&S (head is high)
            prominence = head_price - max(ls_price, rs_price)
        else:
            # Inverse H&S (head is low)
            prominence = min(ls_price, rs_price) - head_price

        required_prominence = min_prominence_atr * atr
        passed = prominence >= required_prominence
        
        # Score: 0.0 when prominence <= 0, reaches 1.0 when prominence >= required_prominence
        if required_prominence <= 0:
            score = 1.0 if passed else 0.0
        else:
            score = max(0.0, min(1.0, prominence / required_prominence))
            
        return passed, round(score, 3)


class NecklineBreak:
    """Evaluates neckline breakout levels."""

    @staticmethod
    def check(
        close_price: float,
        neckline_price: float,
        is_inverse: bool = False,
        atr: float = 1.0,
        buffer_atr: float = 0.0
    ) -> bool:
        """
        Checks if price has broken through the neckline.
        If buffer_atr > 0, requires close to be beyond the level by that buffer.
        """
        buffer = buffer_atr * atr
        if not is_inverse:
            # Bearish neckline break: price must close below neckline - buffer
            return close_price < (neckline_price - buffer)
        else:
            # Bullish neckline break: price must close above neckline + buffer
            return close_price > (neckline_price + buffer)
