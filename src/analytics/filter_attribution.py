#!/usr/bin/env python3
"""
Filter Attribution Report
==========================
Answers "is this filter net positive?" by joining each rejection reason to
the counterfactual (shadow) trade outcome for that same candidate, and
comparing against the outcome of accepted trades in the same experiment.

Previous version only counted how often each `rejected_reason` fired
(rejection frequency) — it never looked at whether the rejected trade would
have won or lost, so it could not answer the actual research question.
"""

import logging
from src.models.postgres_database import PostgresDatabase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FilterAttribution")

# Below this many closed trades, an expectancy reading is not trustworthy —
# flag it rather than let it drive a promotion/rejection decision.
MIN_SAMPLE_SIZE = 30


def _fmt_row(label, n, win_rate, avg_r, total_r):
    flag = " ⚠ LOW N" if n < MIN_SAMPLE_SIZE else ""
    win_str = f"{win_rate * 100:5.1f}%" if n else "   n/a"
    avg_str = f"{avg_r:+.3f}R" if n else "    n/a"
    tot_str = f"{total_r:+.1f}R" if n else "   n/a"
    print(f"{label:28} | {n:5} | {win_str:>7} | {avg_str:>8} | {tot_str:>9}{flag}")


class FilterAttribution:
    def __init__(self):
        self.db = PostgresDatabase()

    def run_attribution(self, experiment_name=None):
        """
        For each rejection reason, report the win rate / expectancy of the
        counterfactual (shadow) trades rejected for that reason, and compare
        against the expectancy of accepted trades in the same experiment.

        A filter is doing its job if reason-X expectancy is materially worse
        than the accepted-trade baseline. If it's comparable or better, the
        filter is discarding trades it shouldn't.
        """
        logger.info("Analyzing filter attribution (outcome-joined)...")
        exp_clause = "AND experiment_name = %s" if experiment_name else ""
        params = (experiment_name,) if experiment_name else ()

        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Baseline: accepted trades that have actually closed.
                    cursor.execute(f"""
                        SELECT COUNT(*),
                               AVG(CASE WHEN final_pnl_r > 0 THEN 1.0 ELSE 0.0 END),
                               AVG(final_pnl_r),
                               SUM(final_pnl_r)
                        FROM trade_performance
                        WHERE exit_time IS NOT NULL {exp_clause}
                    """, params)
                    acc_n, acc_win, acc_avg, acc_tot = cursor.fetchone()
                    acc_n = acc_n or 0

                    # Rejected trades, grouped by the reason they were rejected for,
                    # joined against their own shadow-trade outcome (same row).
                    cursor.execute(f"""
                        SELECT primary_rejection_reason,
                               COUNT(*),
                               AVG(CASE WHEN final_pnl_r > 0 THEN 1.0 ELSE 0.0 END),
                               AVG(final_pnl_r),
                               SUM(final_pnl_r)
                        FROM counterfactual_results
                        WHERE exit_time IS NOT NULL
                          AND primary_rejection_reason IS NOT NULL
                          {exp_clause}
                        GROUP BY primary_rejection_reason
                        ORDER BY COUNT(*) DESC
                    """, params)
                    rejection_rows = cursor.fetchall()

                    scope = experiment_name or "ALL EXPERIMENTS"
                    print("\nFilter Attribution Report — outcome-joined")
                    print(f"Scope: {scope}")
                    print("=" * 80)
                    print(f"{'Reason':28} | {'n':>5} | {'WinRt':>7} | {'AvgR':>8} | {'TotalR':>9}")
                    print("-" * 80)
                    _fmt_row("ACCEPTED (baseline)", acc_n, acc_win or 0.0, acc_avg or 0.0, acc_tot or 0.0)
                    print("-" * 80)

                    if not rejection_rows:
                        print("No closed counterfactual trades with a recorded rejection reason yet.")
                    for reason, n, win_rate, avg_r, total_r in rejection_rows:
                        verdict = ""
                        if n >= MIN_SAMPLE_SIZE and acc_n >= MIN_SAMPLE_SIZE:
                            if avg_r is not None and acc_avg is not None:
                                verdict = "  <- filter looks CORRECT (rejects worse-than-baseline trades)" \
                                    if avg_r < acc_avg else \
                                    "  <- filter looks COSTLY (rejected trades beat baseline)"
                        _fmt_row(reason, n, win_rate or 0.0, avg_r or 0.0, total_r or 0.0)
                        if verdict:
                            print(verdict)

                    print("=" * 80)
                    print(f"Reasons with n < {MIN_SAMPLE_SIZE} are flagged ⚠ LOW N — do not act on them yet.")

        except Exception as e:
            logger.error(f"Failed to run attribution: {e}")

    def compare_experiments(self, experiment_a, experiment_b, label_a=None, label_b=None):
        """
        Compare closed-trade expectancy between two experiments — e.g. a real
        experiment vs. its shadow-only variant (Structural_v3.2 vs
        Structural_v3.3_ExitMgmt) — to decide whether to promote a variant.

        Reports n / win rate / avg R / total R for each side plus a sample-size
        warning, so a promotion decision isn't made off too few trades.
        """
        label_a = label_a or experiment_a
        label_b = label_b or experiment_b
        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    print("\nExperiment Comparison")
                    print("=" * 80)
                    print(f"{'Experiment':28} | {'n':>5} | {'WinRt':>7} | {'AvgR':>8} | {'TotalR':>9}")
                    print("-" * 80)
                    for exp_name, label in ((experiment_a, label_a), (experiment_b, label_b)):
                        cursor.execute("""
                            SELECT COUNT(*),
                                   AVG(CASE WHEN final_pnl_r > 0 THEN 1.0 ELSE 0.0 END),
                                   AVG(final_pnl_r),
                                   SUM(final_pnl_r)
                            FROM trade_performance
                            WHERE exit_time IS NOT NULL AND experiment_name = %s
                        """, (exp_name,))
                        n, win, avg_r, tot = cursor.fetchone()
                        n = n or 0
                        _fmt_row(label, n, win or 0.0, avg_r or 0.0, tot or 0.0)

                        # Also surface shadow-only performance for the same name,
                        # in case the variant hasn't been promoted to real capital
                        # yet (e.g. Structural_v3.3_ExitMgmt).
                        cursor.execute("""
                            SELECT COUNT(*),
                                   AVG(CASE WHEN final_pnl_r > 0 THEN 1.0 ELSE 0.0 END),
                                   AVG(final_pnl_r),
                                   SUM(final_pnl_r)
                            FROM counterfactual_results
                            WHERE exit_time IS NOT NULL AND experiment_name = %s
                        """, (exp_name,))
                        cn, cwin, cavg, ctot = cursor.fetchone()
                        cn = cn or 0
                        if cn:
                            _fmt_row(f"{label} (shadow)", cn, cwin or 0.0, cavg or 0.0, ctot or 0.0)
                    print("=" * 80)
                    print(f"Do not promote/change routing off n < {MIN_SAMPLE_SIZE} on either side.")
        except Exception as e:
            logger.error(f"Failed to run experiment comparison: {e}")

    def oi_change_bias_attribution(self, experiment_name=None):
        """
        Joins the oi_change_bias diagnostic (currently logged but unused as a
        gate — see pcr_extreme_reversal_strategy.py) against realized outcome,
        for both accepted (trade_performance) and shadow (counterfactual_results)
        trades, so we can tell whether OI-buildup agreement predicts anything
        before wiring it in as a filter or size multiplier.
        """
        exp_clause = "AND experiment_name = %s" if experiment_name else ""
        params = (experiment_name,) if experiment_name else ()
        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    print("\nOI-Change-Bias Attribution (diagnostics -> outcome)")
                    print("=" * 80)
                    print(f"{'Source':10} | {'oi_change_bias':16} | {'n':>5} | {'WinRt':>7} | {'AvgR':>8} | {'TotalR':>9}")
                    print("-" * 80)
                    for table in ("trade_performance", "counterfactual_results"):
                        cursor.execute(f"""
                            SELECT diagnostics->>'oi_change_bias' AS bias,
                                   COUNT(*),
                                   AVG(CASE WHEN final_pnl_r > 0 THEN 1.0 ELSE 0.0 END),
                                   AVG(final_pnl_r),
                                   SUM(final_pnl_r)
                            FROM {table}
                            WHERE exit_time IS NOT NULL
                              AND diagnostics ? 'oi_change_bias'
                              {exp_clause}
                            GROUP BY bias
                            ORDER BY COUNT(*) DESC
                        """, params)
                        rows = cursor.fetchall()
                        if not rows:
                            print(f"{table:10} | (no rows with oi_change_bias in diagnostics yet)")
                        for bias, n, win, avg_r, tot in rows:
                            n = n or 0
                            flag = " ⚠ LOW N" if n < MIN_SAMPLE_SIZE else ""
                            win_str = f"{(win or 0.0) * 100:5.1f}%"
                            avg_str = f"{(avg_r or 0.0):+.3f}R"
                            tot_str = f"{(tot or 0.0):+.1f}R"
                            print(f"{table:10} | {str(bias):16} | {n:5} | {win_str:>7} | {avg_str:>8} | {tot_str:>9}{flag}")
                    print("=" * 80)
        except Exception as e:
            logger.error(f"Failed to run OI-change-bias attribution: {e}")


if __name__ == "__main__":
    attr = FilterAttribution()
    attr.run_attribution()
    attr.compare_experiments("Structural_v3.2_RVOL1.0", "Structural_v3.3_ExitMgmt",
                              label_a="v3.2 (real)", label_b="v3.3_ExitMgmt (shadow)")
    attr.oi_change_bias_attribution()
