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

    def run_attribution_per_strategy(self):
        """
        Run the outcome-joined filter attribution separately for every
        experiment that has closed trades, instead of pooling all 26
        experiments into one baseline. A filter (e.g. RVOL, daily bias) can
        be net-positive for one strategy and net-negative for another —
        pooling hides that.
        """
        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT DISTINCT experiment_name FROM (
                            SELECT experiment_name FROM trade_performance WHERE exit_time IS NOT NULL
                            UNION
                            SELECT experiment_name FROM counterfactual_results WHERE exit_time IS NOT NULL
                        ) x
                        WHERE experiment_name IS NOT NULL
                        ORDER BY experiment_name
                    """)
                    experiments = [r[0] for r in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list experiments for per-strategy attribution: {e}")
            return

        if not experiments:
            print("No experiments with closed trades yet.")
            return

        for exp_name in experiments:
            self.run_attribution(experiment_name=exp_name)

    def regime_session_attribution(self, experiment_name=None):
        """
        Breaks accepted-trade expectancy down by market_regime and by
        intraday session bucket, for real trades and their shadow
        counterparts. counterfactual_results has no market_regime column of
        its own, so it's joined back to signal_audit on candidate_id to
        recover the regime the candidate fired in.

        Session buckets follow the CLAUDE.md trading hours (09:15-15:30 IST).
        """
        exp_clause = "AND t.experiment_name = %s" if experiment_name else ""
        params = (experiment_name,) if experiment_name else ()

        session_case = """
            CASE
                WHEN (entry_time AT TIME ZONE 'Asia/Kolkata')::time BETWEEN '09:15' AND '10:00' THEN '09:15-10:00'
                WHEN (entry_time AT TIME ZONE 'Asia/Kolkata')::time BETWEEN '10:00' AND '12:00' THEN '10:00-12:00'
                WHEN (entry_time AT TIME ZONE 'Asia/Kolkata')::time BETWEEN '12:00' AND '14:00' THEN '12:00-14:00'
                WHEN (entry_time AT TIME ZONE 'Asia/Kolkata')::time BETWEEN '14:00' AND '15:00' THEN '14:00-15:00'
                ELSE '15:00-15:25'
            END
        """

        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    scope = experiment_name or "ALL EXPERIMENTS"
                    print("\nRegime / Session Attribution")
                    print(f"Scope: {scope}")

                    print("\n-- By market_regime --")
                    print("=" * 90)
                    print(f"{'Source':10} | {'Regime':16} | {'n':>5} | {'WinRt':>7} | {'AvgR':>8} | {'TotalR':>9}")
                    print("-" * 90)
                    cursor.execute(f"""
                        SELECT market_regime, COUNT(*),
                               AVG(CASE WHEN final_pnl_r > 0 THEN 1.0 ELSE 0.0 END),
                               AVG(final_pnl_r), SUM(final_pnl_r)
                        FROM trade_performance t
                        WHERE exit_time IS NOT NULL {exp_clause}
                        GROUP BY market_regime ORDER BY COUNT(*) DESC
                    """, params)
                    for regime, n, win, avg_r, tot in cursor.fetchall():
                        self._print_bucket("real", regime, n, win, avg_r, tot)

                    cursor.execute(f"""
                        SELECT sa.market_regime, COUNT(*),
                               AVG(CASE WHEN c.final_pnl_r > 0 THEN 1.0 ELSE 0.0 END),
                               AVG(c.final_pnl_r), SUM(c.final_pnl_r)
                        FROM counterfactual_results c
                        JOIN signal_audit sa ON sa.candidate_id = c.candidate_id
                        WHERE c.exit_time IS NOT NULL {exp_clause.replace('t.experiment_name', 'c.experiment_name')}
                        GROUP BY sa.market_regime ORDER BY COUNT(*) DESC
                    """, params)
                    for regime, n, win, avg_r, tot in cursor.fetchall():
                        self._print_bucket("shadow", regime, n, win, avg_r, tot)

                    print("\n-- By session --")
                    print("=" * 90)
                    print(f"{'Source':10} | {'Session':16} | {'n':>5} | {'WinRt':>7} | {'AvgR':>8} | {'TotalR':>9}")
                    print("-" * 90)
                    cursor.execute(f"""
                        SELECT {session_case}, COUNT(*),
                               AVG(CASE WHEN final_pnl_r > 0 THEN 1.0 ELSE 0.0 END),
                               AVG(final_pnl_r), SUM(final_pnl_r)
                        FROM trade_performance t
                        WHERE exit_time IS NOT NULL {exp_clause}
                        GROUP BY 1 ORDER BY 1
                    """, params)
                    for session, n, win, avg_r, tot in cursor.fetchall():
                        self._print_bucket("real", session, n, win, avg_r, tot)

                    cursor.execute(f"""
                        SELECT {session_case.replace('entry_time', 'c.timestamp')}, COUNT(*),
                               AVG(CASE WHEN c.final_pnl_r > 0 THEN 1.0 ELSE 0.0 END),
                               AVG(c.final_pnl_r), SUM(c.final_pnl_r)
                        FROM counterfactual_results c
                        WHERE c.exit_time IS NOT NULL {exp_clause.replace('t.experiment_name', 'c.experiment_name')}
                        GROUP BY 1 ORDER BY 1
                    """, params)
                    for session, n, win, avg_r, tot in cursor.fetchall():
                        self._print_bucket("shadow", session, n, win, avg_r, tot)

                    print("=" * 90)
                    print(f"Buckets with n < {MIN_SAMPLE_SIZE} are flagged ⚠ LOW N - do not act on them yet.")
        except Exception as e:
            logger.error(f"Failed to run regime/session attribution: {e}")

    @staticmethod
    def _print_bucket(source, bucket, n, win, avg_r, tot):
        n = n or 0
        flag = " ⚠ LOW N" if n < MIN_SAMPLE_SIZE else ""
        win_str = f"{(win or 0.0) * 100:5.1f}%"
        avg_str = f"{(avg_r or 0.0):+.3f}R"
        tot_str = f"{(tot or 0.0):+.1f}R"
        print(f"{source:10} | {str(bucket):16} | {n:5} | {win_str:>7} | {avg_str:>8} | {tot_str:>9}{flag}")

    def signal_overlap_matrix(self, min_cofires=10, limit=40):
        """
        Detects candidate-level overlap between strategy pairs: how often two
        experiments fire a candidate on the same symbol at the same candle,
        whether they agree on direction, and whether their closed-trade
        outcomes correlate. If several strategies are mostly co-firing in
        the same direction with correlated P&L, they're likely one alpha
        wearing different names, not independent edges.

        Built from trade_performance UNION counterfactual_results, since
        every strategy writes every candidate (accepted or rejected) into
        one of those two tables via the same _update_position() engine.
        signal_audit is populated only by the Structural experiments, not
        the other single-leg strategies, so it can't serve as the common
        candidate log. Combo (options-structure) strategies write to
        separate combo tables and are out of scope here — see #11, they
        need their own comparison track.
        """
        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        WITH candidates AS (
                            SELECT experiment_name, symbol, entry_time AS candle_time,
                                   signal_type AS direction, final_pnl_r
                            FROM trade_performance
                            WHERE experiment_name IS NOT NULL
                            UNION ALL
                            SELECT experiment_name, symbol, timestamp AS candle_time,
                                   signal_type AS direction, final_pnl_r
                            FROM counterfactual_results
                            WHERE experiment_name IS NOT NULL
                        ),
                        totals AS (
                            SELECT experiment_name, COUNT(*) AS n
                            FROM candidates GROUP BY experiment_name
                        )
                        SELECT a.experiment_name, b.experiment_name,
                               COUNT(*) AS co_fires,
                               AVG(CASE WHEN a.direction = b.direction THEN 1.0 ELSE 0.0 END) AS same_dir_rate,
                               CORR(a.final_pnl_r, b.final_pnl_r) AS pnl_corr,
                               COUNT(*) FILTER (WHERE a.final_pnl_r IS NOT NULL AND b.final_pnl_r IS NOT NULL) AS closed_pairs,
                               ta.n AS n_a, tb.n AS n_b
                        FROM candidates a
                        JOIN candidates b
                          ON a.symbol = b.symbol
                         AND a.candle_time = b.candle_time
                         AND a.experiment_name < b.experiment_name
                        JOIN totals ta ON ta.experiment_name = a.experiment_name
                        JOIN totals tb ON tb.experiment_name = b.experiment_name
                        GROUP BY a.experiment_name, b.experiment_name, ta.n, tb.n
                        HAVING COUNT(*) >= %s
                        ORDER BY co_fires DESC
                        LIMIT %s
                    """, (min_cofires, limit))
                    rows = cursor.fetchall()

                    print("\nSignal Overlap Matrix (candidate-level co-firing)")
                    print("=" * 112)
                    print(f"{'Strategy A':28} | {'Strategy B':28} | {'CoFire':>7} | {'Ovlp%':>6} | {'SameDir':>7} | {'PnLCorr':>8}")
                    print("-" * 112)
                    if not rows:
                        print(f"No strategy pairs co-fired >= {min_cofires} times yet.")
                    for exp_a, exp_b, co_fires, same_dir, corr, closed_pairs, n_a, n_b in rows:
                        overlap_pct = co_fires / max(1, min(n_a, n_b)) * 100
                        same_dir_str = f"{(same_dir or 0.0) * 100:5.1f}%"
                        corr_str = f"{corr:+.2f}" if corr is not None else "n/a"
                        flag = "" if closed_pairs >= MIN_SAMPLE_SIZE else " ⚠ LOW N(closed)"
                        print(f"{exp_a:28} | {exp_b:28} | {co_fires:7} | {overlap_pct:5.1f}% | {same_dir_str:>7} | {corr_str:>8}{flag}")
                    print("=" * 112)
                    print("Ovlp% = co-fires / min(n_A, n_B) candidates. PnLCorr uses only pairs where both sides closed.")
                    print(f"Pairs with < {MIN_SAMPLE_SIZE} closed co-fires are flagged - don't act on PnLCorr for them yet.")
        except Exception as e:
            logger.error(f"Failed to compute signal overlap matrix: {e}")

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
    import sys

    attr = FilterAttribution()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("all", "pooled"):
        attr.run_attribution()
    if mode in ("all", "per-strategy"):
        attr.run_attribution_per_strategy()
    if mode in ("all", "regime-session"):
        attr.regime_session_attribution()
    if mode in ("all", "overlap"):
        attr.signal_overlap_matrix()
    if mode in ("all", "compare"):
        attr.compare_experiments("Structural_v3.2_RVOL1.0", "Structural_v3.3_ExitMgmt",
                                  label_a="v3.2 (real)", label_b="v3.3_ExitMgmt (shadow)")
    if mode in ("all", "oi-bias"):
        attr.oi_change_bias_attribution()
