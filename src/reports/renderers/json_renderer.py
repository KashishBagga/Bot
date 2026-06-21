#!/usr/bin/env python3
"""
JsonRenderer — assembles section JSON dicts into the final analytics payload.
The JSON structure is designed for future Streamlit consumption:
  no duplicate analytics logic in the dashboard.
"""

import json
from typing import Dict, List


class JsonRenderer:
    """Knows nothing about SQL. Just assembles section dicts into report JSON."""

    def assemble(
        self,
        date_str: str,
        section_jsons: Dict[str, dict],
        generated_at: str,
    ) -> dict:
        """Merge all section data into a single analytics payload."""
        return {
            "date": date_str,
            "generated_at": generated_at,
            "sections": section_jsons,
            # Top-level shortcuts for Streamlit (populated from section data)
            "equity_curve": section_jsons.get("strategy_health", {}).get("equity_curve", []),
            "filter_stats": section_jsons.get("counterfactual_insights", {}).get("filters", []),
            "trade_distribution": section_jsons.get("market_behaviour", {}).get("hourly_performance", []),
            "experiment_rankings": section_jsons.get("experiment_ranking", {}).get("rankings", []),
            "research_queue": section_jsons.get("research_queue", {}).get("queue", []),
        }

    def to_string(self, payload: dict) -> str:
        return json.dumps(payload, indent=2, default=str)
