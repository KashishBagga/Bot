#!/usr/bin/env python3
"""
BaseSection — Abstract base class for all report sections.
==========================================================

Each section:
  1. compute()       → run DB queries, return structured data dict
  2. render_md(data) → return Markdown string
  3. render_json(data) → return JSON-serializable dict (default: return data)

The orchestrator calls compute() once, then renders both formats.
One section failure never crashes the full report.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseSection(ABC):
    """Abstract base for all 12 report sections."""

    section_id: str = "base"
    section_title: str = "Section"

    def __init__(
        self,
        db,
        data_provider,
        date_str: str,
        rolling: Dict[str, Any],
    ):
        self.db = db
        self.data_provider = data_provider
        self.date_str = date_str      # "2026-06-21"
        self.rolling = rolling        # Pre-fetched 5d/20d rolling stats

    # ── Abstract interface ──────────────────────────────────────────────────

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """Run all DB/API queries. Return structured data dict. Never raise."""

    @abstractmethod
    def render_md(self, data: Dict[str, Any]) -> str:
        """Convert computed data to Markdown string."""

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert computed data to JSON-serializable dict. Default: return data."""
        return data

    # ── Safe wrappers (used by orchestrator) ───────────────────────────────

    def safe_compute(self) -> Dict[str, Any]:
        """Wraps compute() so one section never crashes the report."""
        try:
            return self.compute()
        except Exception as e:
            logger.error(f"❌ [{self.section_id}] compute() failed: {e}", exc_info=True)
            return {"_error": str(e), "_section": self.section_id}

    def safe_render_md(self, data: Dict[str, Any]) -> str:
        """Wraps render_md() with graceful error display."""
        if "_error" in data:
            return (
                f"\n---\n\n## {self.section_title}\n\n"
                f"⚠️ *Section unavailable: {data['_error']}*\n"
            )
        try:
            return self.render_md(data)
        except Exception as e:
            logger.error(f"❌ [{self.section_id}] render_md() failed: {e}", exc_info=True)
            return (
                f"\n---\n\n## {self.section_title}\n\n"
                f"⚠️ *Render error: {e}*\n"
            )

    # ── DB helper ───────────────────────────────────────────────────────────

    def _query(self, sql: str, params: tuple = ()) -> list:
        """Execute a SELECT query and return list of rows."""
        with self.db._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    # ── Formatting helpers ──────────────────────────────────────────────────

    @staticmethod
    def _stars(n: int, max_n: int = 5) -> str:
        filled = min(n, max_n)
        return "★" * filled + "☆" * (max_n - filled)

    @staticmethod
    def _pnl_str(r: float) -> str:
        """Format R value with sign and colour emoji."""
        if r is None:
            return "N/A"
        sign = "+" if r >= 0 else ""
        return f"{sign}{r:.2f}R"

    @staticmethod
    def _pct(v: float) -> str:
        if v is None:
            return "N/A"
        return f"{v*100:.0f}%"
