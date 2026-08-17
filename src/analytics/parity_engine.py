#!/usr/bin/env python3
"""
Live <-> Replay Parity Framework (P0)
===================================
Verifies the system behaves identically in Live vs Backtest.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from zoneinfo import ZoneInfo

from src.models.postgres_database import PostgresDatabase

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ParityEngine")

IST = ZoneInfo("Asia/Kolkata")
ENTRY_MATCH_TOLERANCE = timedelta(minutes=5)   # one 5m bar
PNL_MATCH_TOLERANCE_R = 0.25                    # same tolerance data_quality.py uses


class ParityEngine:
    """Audits live/counterfactual trades against a same-day backtest replay.

    Uses TransparentBacktester (src/backtesting/advanced_backtester.py) —
    the same MarketSnapshot/IndicatorPipeline/ExperimentRegistry pipeline
    live trading runs — instead of a separately-instantiated engine, so a
    parity mismatch actually means "backtest and live disagree", not
    "two different code paths disagree with each other".
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.db = PostgresDatabase()

    def run_fill_parity_test(self, target_date: Optional[date] = None, days: int = 1) -> Dict[str, Any]:
        """Compare a backtest replay against that day's live + counterfactual
        trades. `target_date` defaults to today — run this after market close
        so the day's live trades already exist in the DB. `days` controls how
        far back TransparentBacktester's own bar-fetch window reaches (must be
        >= 1 to cover target_date's session); it does not change which date's
        live trades get compared.

        HONESTY CONTRACT: every field below is a real, computed comparison —
        no field is fabricated or assumed to pass. Where a pairing is
        genuinely unavailable (no live trades that day, or the backtester's
        outcome-simulation window differs from live's exit mechanics),
        the corresponding *_status field says so instead of reporting a
        number.
        """
        target_date = target_date or datetime.now(IST).date()
        logger.info(f"🧪 Starting fill-parity audit for {target_date}...")

        from src.backtesting.advanced_backtester import TransparentBacktester
        backtester = TransparentBacktester(self.symbols, days=days)
        backtester.fetch_data()
        if not backtester.historical_data:
            return self._empty_scorecard(target_date, "NO_BACKTEST_DATA")
        replay = backtester.simulate_trades(verbose=False)
        replay_trades = [t for t in replay['trades'] if self._entry_date(t['entry_time']) == target_date]

        live_trades = self._fetch_live_trades(target_date)

        if not live_trades:
            scorecard = self._empty_scorecard(target_date, "NO_LIVE_DATA")
            scorecard['replay_trade_count'] = len(replay_trades)
            return scorecard
        if not replay_trades:
            scorecard = self._empty_scorecard(target_date, "NO_REPLAY_TRADES")
            scorecard['live_trade_count'] = len(live_trades)
            return scorecard

        # Determinism: same replay window run twice must produce identical trades.
        replay_2 = backtester.simulate_trades(verbose=False)
        determinism_pass = (replay['trades'] == replay_2['trades'])
        if not determinism_pass:
            logger.error("🚨 DETERMINISM FAILURE: replaying the same window twice gave different trades!")

        # Match live trades to replay trades by (symbol, experiment_name, entry
        # bar within one 5m candle) — the only key both sides share meaningfully.
        matched = []
        unmatched_live = []
        for live in live_trades:
            candidate = self._find_match(live, replay_trades)
            if candidate is not None:
                matched.append((live, candidate))
            else:
                unmatched_live.append(live)

        entry_match_pct = round(100.0 * len(matched) / len(live_trades), 2)

        exit_agree = 0
        pnl_agree = 0
        for live, rep in matched:
            if self._exit_bucket(live) == self._exit_bucket_replay(rep):
                exit_agree += 1
            live_index_r = self._live_index_point_r(live)
            if live_index_r is not None and abs(live_index_r - rep['pnl_r']) <= PNL_MATCH_TOLERANCE_R:
                pnl_agree += 1

        exit_match_pct = round(100.0 * exit_agree / len(matched), 2) if matched else None
        pnl_match_pct = round(100.0 * pnl_agree / len(matched), 2) if matched else None

        scorecard = {
            'target_date': str(target_date),
            'replay_determinism': "PASS" if determinism_pass else "FAIL",
            'live_trade_count': len(live_trades),
            'replay_trade_count': len(replay_trades),
            'entry_match_pct': entry_match_pct,
            'entry_match_status': "COMPUTED",
            'exit_match_pct': exit_match_pct,
            'exit_match_status': "COMPUTED" if matched else "NO_MATCHED_PAIRS",
            'pnl_match_pct': pnl_match_pct,
            'pnl_match_status': "COMPUTED" if matched else "NO_MATCHED_PAIRS",
            'fill_parity_status': "COMPUTED",
            'unmatched_live_count': len(unmatched_live),
            'parity_alert': (
                not determinism_pass
                or entry_match_pct < 80.0
                or (exit_match_pct is not None and exit_match_pct < 70.0)
                or (pnl_match_pct is not None and pnl_match_pct < 70.0)
            ),
        }

        logger.info(
            f"✅ Fill-parity audit complete for {target_date}. Determinism: {scorecard['replay_determinism']} | "
            f"entry_match={entry_match_pct}% | exit_match={exit_match_pct} | pnl_match={pnl_match_pct}"
        )
        return scorecard

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_scorecard(target_date: date, status: str) -> Dict[str, Any]:
        return {
            'target_date': str(target_date),
            'replay_determinism': None,
            'live_trade_count': 0,
            'replay_trade_count': 0,
            'entry_match_pct': None,
            'entry_match_status': status,
            'exit_match_pct': None,
            'exit_match_status': status,
            'pnl_match_pct': None,
            'pnl_match_status': status,
            'fill_parity_status': status,
            'unmatched_live_count': 0,
            'parity_alert': True,
        }

    @staticmethod
    def _entry_date(ts) -> date:
        if ts.tzinfo is None:
            return ts.date()
        return ts.astimezone(IST).date()

    def _fetch_live_trades(self, target_date: date) -> List[Dict[str, Any]]:
        """Real + counterfactual trades for target_date, symbol-filtered.
        Never fabricates rows — returns [] if the DB has none for that date."""
        rows: List[Dict[str, Any]] = []
        try:
            with self.db._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute('''
                        SELECT symbol, experiment_name, entry_time, exit_time, entry_price,
                               stop_loss, exit_price, exit_reason, final_pnl_r, signal_type, 'real' AS source
                        FROM trade_performance
                        WHERE DATE(entry_time AT TIME ZONE 'Asia/Kolkata') = %s
                          AND symbol = ANY(%s) AND valid = TRUE
                    ''', (target_date, self.symbols))
                    cols = ["symbol", "experiment_name", "entry_time", "exit_time", "entry_price",
                            "stop_loss", "exit_price", "exit_reason", "final_pnl_r", "signal_type", "source"]
                    rows.extend(dict(zip(cols, r)) for r in cur.fetchall())

                    cur.execute('''
                        SELECT symbol, experiment_name, timestamp, exit_time, entry_price,
                               stop_loss, exit_price, exit_reason, final_pnl_r, signal_type, 'cf' AS source
                        FROM counterfactual_results
                        WHERE DATE(timestamp AT TIME ZONE 'Asia/Kolkata') = %s
                          AND symbol = ANY(%s) AND valid = TRUE AND exit_time IS NOT NULL
                    ''', (target_date, self.symbols))
                    rows.extend(dict(zip(cols, r)) for r in cur.fetchall())
        except Exception as e:
            logger.error(f"Could not fetch live trades for {target_date}: {e}")
        return rows

    @staticmethod
    def _find_match(live: Dict[str, Any], replay_trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for rep in replay_trades:
            if rep['symbol'] != live['symbol'] or rep['experiment'] != live['experiment_name']:
                continue
            if abs(rep['entry_time'] - live['entry_time']) <= ENTRY_MATCH_TOLERANCE:
                return rep
        return None

    @staticmethod
    def _exit_bucket(live: Dict[str, Any]) -> str:
        """Bucket live's real exit_reason vocabulary (STOP_LOSS/TRAILING_SL/
        TP_EXPANSION/SESSION_END) into WIN/LOSS/BREAKEVEN so it's comparable to
        the backtester's different (but analogous) exit vocabulary — exact
        exit-mechanism parity isn't realistic since the backtester doesn't
        model live's trailing-stop state machine."""
        pnl = live.get('final_pnl_r')
        if pnl is None:
            return "UNKNOWN"
        if pnl > 0.05:
            return "WIN"
        if pnl < -0.05:
            return "LOSS"
        return "BREAKEVEN"

    @staticmethod
    def _exit_bucket_replay(rep: Dict[str, Any]) -> str:
        pnl = rep.get('pnl_r')
        if pnl is None:
            return "UNKNOWN"
        if pnl > 0.05:
            return "WIN"
        if pnl < -0.05:
            return "LOSS"
        return "BREAKEVEN"

    @staticmethod
    def _live_index_point_r(live: Dict[str, Any]) -> Optional[float]:
        """Recompute live's PnL on the SAME index-point-proxy basis the
        backtester uses, regardless of whether the live trade was actually
        priced via real option premium (see indian_trader.py:_premium_pnl_r).
        Comparing premium-priced R directly against the backtester's index-
        point R would fail every trade for reasons that have nothing to do
        with parity — the two are different (both valid) definitions of R.
        This puts both sides on the same footing before comparing."""
        entry = live.get('entry_price')
        exit_ = live.get('exit_price')
        sl = live.get('stop_loss')
        if not entry or not exit_ or not sl or entry == sl:
            return None
        stop_loss_distance = abs(entry - sl)
        sig = str(live.get('signal_type') or '').upper()
        is_long = "CALL" in sig or ("BUY" in sig and "PUT" not in sig)
        if is_long:
            return (exit_ - entry) / stop_loss_distance
        return (entry - exit_) / stop_loss_distance


if __name__ == "__main__":
    engine = ParityEngine(["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"])
    print(f"Parity Scorecard: {engine.run_fill_parity_test()}")
