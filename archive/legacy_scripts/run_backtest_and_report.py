#!/usr/bin/env python3
import argparse
import subprocess
import sys
import sqlite3
from datetime import datetime

DEFAULT_SYMBOLS = [
	'NSE:NIFTYBANK-INDEX',
	'NSE:NIFTY50-INDEX',
]


def run_backtests(strategies, symbols, timeframe, days):
	for strat in strategies:
		for sym in symbols:
			print(f"\n=== Running {strat} on {sym} ({timeframe}, {days}d) ===")
			subprocess.run([
				sys.executable, 'backtesting_parquet.py',
				'--strategy', strat,
				'--symbol', sym,
				'--timeframe', timeframe,
				'--days', str(days)
			], check=False)


def _compute_drawdown(rows):
	cum = 0.0
	peak = 0.0
	max_dd = 0.0
	for (_, _, _, _, pnl, _) in rows:
		cum += pnl or 0.0
		if cum > peak:
			peak = cum
		dd = peak - cum
		if dd > max_dd:
			max_dd = dd
	return max_dd


def report_pnl(strategies, days):
	conn = sqlite3.connect('trading_signals.db')
	c = conn.cursor()
	print("\nP&L summary (last {} days):".format(days))
	q = f"""
	SELECT strategy, symbol, COUNT(*), SUM(pnl), AVG(pnl),
	       SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) AS wins,
	       SUM(CASE WHEN pnl<0 THEN 1 ELSE 0 END) AS losses
	FROM trades_backtest
	WHERE timestamp >= datetime('now','-{days} days')
	  AND strategy IN ({','.join(['?']*len(strategies))})
	GROUP BY strategy, symbol
	ORDER BY strategy, symbol
	"""
	c.execute(q, strategies)
	summary_rows = c.fetchall()
	# Compute drawdowns per group
	for strategy, symbol, n, total, avg, wins, losses in summary_rows:
		# Fetch ordered trades for drawdown
		c.execute(
			f"""
			SELECT timestamp, strategy, symbol, signal, pnl, outcome
			FROM trades_backtest
			WHERE timestamp >= datetime('now','-{days} days')
			  AND strategy = ? AND symbol = ?
			ORDER BY timestamp ASC
			""",
			(strategy, symbol)
		)
		rows = c.fetchall()
		dd = _compute_drawdown(rows)
		print(f"{strategy:26s} {symbol:22s} trades={n:5d} total={total:12.2f} avg={avg:8.2f} W/L={wins}/{losses} dd={dd:10.2f}")

	# Monthly breakdown for the covered range
	print("\nMonthly P&L breakdown:")
	qm = f"""
	SELECT strftime('%Y-%m', timestamp) AS ym, strategy, symbol,
	       COUNT(*) AS trades,
	       SUM(pnl) AS total_pnl,
	       AVG(pnl) AS avg_pnl,
	       SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) AS wins,
	       SUM(CASE WHEN pnl<0 THEN 1 ELSE 0 END) AS losses
	FROM trades_backtest
	WHERE timestamp >= datetime('now','-{days} days')
	  AND strategy IN ({','.join(['?']*len(strategies))})
	GROUP BY ym, strategy, symbol
	ORDER BY ym, strategy, symbol
	"""
	c.execute(qm, strategies)
	rows = c.fetchall()
	for ym, strategy, symbol, n, total, avg, wins, losses in rows:
		print(f"{ym}  {strategy:26s} {symbol:22s} trades={n:4d} total={total:10.2f} avg={avg:7.2f} W/L={wins}/{losses}")
	conn.close()


def main():
	ap = argparse.ArgumentParser(description='Run backtests and summarize P&L from trades_backtest')
	ap.add_argument('--strategies', type=str, required=True, help='Comma-separated list of strategies')
	ap.add_argument('--symbols', type=str, default=','.join(DEFAULT_SYMBOLS), help='Comma-separated list of symbols')
	ap.add_argument('--timeframe', type=str, default='5min')
	ap.add_argument('--days', type=int, default=30)
	args = ap.parse_args()

	strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]
	symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

	run_backtests(strategies, symbols, args.timeframe, args.days)
	report_pnl(strategies, args.days)


if __name__ == '__main__':
	main() 