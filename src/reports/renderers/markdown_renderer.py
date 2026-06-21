#!/usr/bin/env python3
"""
MarkdownRenderer — assembles section markdown strings into a final report.
"""

from datetime import datetime
from typing import Dict, List


class MarkdownRenderer:
    """Knows nothing about SQL. Just assembles Markdown strings."""

    def assemble(
        self,
        date_str: str,
        section_mds: List[str],
        generated_at: str,
    ) -> str:
        """Join section strings with a header and footer."""
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            date_pretty = dt.strftime("%B %d, %Y")
        except ValueError:
            date_pretty = date_str

        header = f"# 📊 Trading Journal — {date_pretty}\n\n"
        footer = (
            f"\n\n---\n\n"
            f"*Generated: {generated_at} IST*  \n"
            f"*System: Structural Paper Trader v3.2*\n"
        )

        return header + "\n".join(section_mds) + footer
