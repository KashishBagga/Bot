#!/usr/bin/env python3
"""
FeatureStore — Typed wrapper for computed indicator values.
============================================================
Prevents silent KeyError / wrong-type access in strategy code.

Usage:
    get_float(key)     — optional feature, returns default if missing
    require_float(key) — required feature, raises KeyError("FEATURE_MISSING:key")

Strategy authors: catch KeyError from require_* in evaluate(), append to errors,
return empty signals. Never let it propagate.
"""

from typing import Any


class FeatureStore:
    """
    Typed, safe wrapper around a dict of computed indicator values.
    Populated once by IndicatorPipeline.compute() and consumed read-only
    by all strategies in the same candle cycle.
    """

    def __init__(self, data: dict):
        self._data = data

    # ── Optional access ────────────────────────────────────────────────────
    # Returns a default if the key is absent. Silent. Use for nice-to-have features.

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        val = self._data.get(key, default)
        return float(val) if val is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        val = self._data.get(key, default)
        return int(val) if val is not None else default

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self._data.get(key, default)
        return bool(val) if val is not None else default

    def get_str(self, key: str, default: str = "") -> str:
        val = self._data.get(key, default)
        return str(val) if val is not None else default

    # ── Required access ────────────────────────────────────────────────────
    # Raises KeyError("FEATURE_MISSING:key") if absent. Loud.
    # Use when the strategy fundamentally cannot evaluate without this feature.

    def require_float(self, key: str) -> float:
        if key not in self._data or self._data[key] is None:
            raise KeyError(f"FEATURE_MISSING:{key}")
        return float(self._data[key])

    def require_int(self, key: str) -> int:
        if key not in self._data or self._data[key] is None:
            raise KeyError(f"FEATURE_MISSING:{key}")
        return int(self._data[key])

    def require_bool(self, key: str) -> bool:
        if key not in self._data or self._data[key] is None:
            raise KeyError(f"FEATURE_MISSING:{key}")
        return bool(self._data[key])

    def require_str(self, key: str) -> str:
        if key not in self._data or self._data[key] is None:
            raise KeyError(f"FEATURE_MISSING:{key}")
        return str(self._data[key])

    # ── Utility ────────────────────────────────────────────────────────────

    def has(self, key: str) -> bool:
        """Returns True if the key exists and is not None."""
        return key in self._data and self._data[key] is not None

    def keys(self):
        return self._data.keys()

    def to_dict(self) -> dict:
        """Returns a plain dict copy for serialisation (e.g. DB JSON columns)."""
        return dict(self._data)

    def __repr__(self) -> str:
        return f"FeatureStore(keys={list(self._data.keys())})"
