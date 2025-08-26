import sqlite3
from datetime import datetime
from typing import Dict, List, Any
import json

class BacktestingSummary:
    """Manages consolidated backtesting results and provides summary views."""
    
    def __init__(self, db_path="trading_signals.db"):
        self.db_path = db_path
        self.setup_tables()
    
    def setup_tables(self):
        """Set up the backtesting summary tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create backtesting_runs table to track each backtest execution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtesting_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT NOT NULL,
                period_days INTEGER,
                timeframe TEXT,
                symbols TEXT,  -- JSON array of symbols
                strategies TEXT,  -- JSON array of strategies
                total_signals INTEGER,
                total_pnl REAL,
                duration_seconds REAL,
                performance_rate REAL,  -- signals per second
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create backtesting_strategy_results table for detailed strategy results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtesting_strategy_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signals_count INTEGER,
                pnl REAL,
                win_rate REAL,
                total_trades INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES backtesting_runs (id)
            )
        """)
        
        # Create view for latest backtesting results
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS latest_backtesting_summary AS
            SELECT 
                br.run_timestamp,
                br.period_days,
                br.timeframe,
                br.symbols,
                br.strategies,
                br.total_signals,
                br.total_pnl,
                br.duration_seconds,
                br.performance_rate,
                COUNT(bsr.id) as strategy_count,
                AVG(bsr.win_rate) as avg_win_rate
            FROM backtesting_runs br
            LEFT JOIN backtesting_strategy_results bsr ON br.id = bsr.run_id
            WHERE br.id = (SELECT MAX(id) FROM backtesting_runs)
            GROUP BY br.id
        """)
        
        # Create view for strategy performance comparison
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS strategy_performance_comparison AS
            SELECT 
                bsr.strategy_name,
                bsr.symbol,
                bsr.signals_count,
                bsr.pnl,
                bsr.win_rate,
                bsr.total_trades,
                br.run_timestamp,
                br.period_days,
                br.timeframe
            FROM backtesting_strategy_results bsr
            JOIN backtesting_runs br ON bsr.run_id = br.id
            WHERE br.id = (SELECT MAX(id) FROM backtesting_runs)
            ORDER BY bsr.pnl DESC
        """)
        
        conn.commit()
        conn.close()
    
    def log_backtest_run(self, backtest_data: Dict[str, Any]) -> int:
        """Log a new backtest run and return the run ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert backtest run
        cursor.execute("""
            INSERT INTO backtesting_runs (
                run_timestamp, period_days, timeframe, symbols, strategies,
                total_signals, total_pnl, duration_seconds, performance_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            backtest_data['run_timestamp'],
            backtest_data['period_days'],
            backtest_data['timeframe'],
            json.dumps(backtest_data['symbols']),
            json.dumps(backtest_data['strategies']),
            backtest_data['total_signals'],
            backtest_data['total_pnl'],
            backtest_data['duration_seconds'],
            backtest_data['performance_rate']
        ))
        
        run_id = cursor.lastrowid
        
        # Insert strategy results
        for strategy_result in backtest_data['strategy_results']:
            cursor.execute("""
                INSERT INTO backtesting_strategy_results (
                    run_id, strategy_name, symbol, signals_count, pnl, win_rate, total_trades
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                strategy_result['strategy_name'],
                strategy_result['symbol'],
                strategy_result['signals_count'],
                strategy_result['pnl'],
                strategy_result['win_rate'],
                strategy_result['total_trades']
            ))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def get_latest_summary(self) -> Dict[str, Any]:
        """Get the latest backtesting summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM latest_backtesting_summary")
        result = cursor.fetchone()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            summary = dict(zip(columns, result))
            
            # Parse JSON fields
            summary['symbols'] = json.loads(summary['symbols'])
            summary['strategies'] = json.loads(summary['strategies'])
            
            conn.close()
            return summary
        
        conn.close()
        return {}
    
    def get_strategy_performance(self) -> List[Dict[str, Any]]:
        """Get strategy performance comparison from latest run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM strategy_performance_comparison")
        results = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        performance_data = [dict(zip(columns, row)) for row in results]
        
        conn.close()
        return performance_data
    
    def get_historical_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical backtest runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM backtesting_runs 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        runs = []
        
        for row in results:
            run_data = dict(zip(columns, row))
            run_data['symbols'] = json.loads(run_data['symbols'])
            run_data['strategies'] = json.loads(run_data['strategies'])
            runs.append(run_data)
        
        conn.close()
        return runs
    
    def print_latest_summary(self):
        """Print a formatted summary of the latest backtest."""
        summary = self.get_latest_summary()
        
        if not summary:
            print("❌ No backtesting data found")
            return
        
        print("\n" + "="*80)
        print("📊 LATEST BACKTESTING SUMMARY")
        print("="*80)
        print(f"🕒 Run Time: {summary['run_timestamp']}")
        print(f"📅 Period: {summary['period_days']} days")
        print(f"⏰ Timeframe: {summary['timeframe']}")
        print(f"📈 Symbols: {', '.join(summary['symbols'])}")
        print(f"🧠 Strategies: {', '.join(summary['strategies'])}")
        print(f"🎯 Total Signals: {summary['total_signals']}")
        print(f"💰 Total P&L: ₹{summary['total_pnl']:.2f}")
        print(f"📊 Average Win Rate: {summary['avg_win_rate']:.1f}%")
        print(f"⚡ Performance: {summary['performance_rate']:.1f} signals/second")
        print(f"⏱️ Duration: {summary['duration_seconds']:.2f} seconds")
        
        # Print strategy performance
        performance = self.get_strategy_performance()
        if performance:
            print("\n🎯 STRATEGY PERFORMANCE:")
            print("-" * 80)
            for strategy in performance:
                print(f"📈 {strategy['strategy_name'].upper()} ({strategy['symbol']}):")
                print(f"   Signals: {strategy['signals_count']}, P&L: ₹{strategy['pnl']:.2f}, Win Rate: {strategy['win_rate']:.1f}%")
        
        print("="*80)
    
    def clear_old_runs(self, keep_last_n: int = 50):
        """Clear old backtest runs, keeping only the last N runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get IDs of runs to keep
        cursor.execute("""
            SELECT id FROM backtesting_runs 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (keep_last_n,))
        
        keep_ids = [row[0] for row in cursor.fetchall()]
        
        if keep_ids:
            placeholders = ','.join(['?' for _ in keep_ids])
            
            # Delete old strategy results
            cursor.execute(f"""
                DELETE FROM backtesting_strategy_results 
                WHERE run_id NOT IN ({placeholders})
            """, keep_ids)
            
            # Delete old runs
            cursor.execute(f"""
                DELETE FROM backtesting_runs 
                WHERE id NOT IN ({placeholders})
            """, keep_ids)
            
            conn.commit()
            print(f"✅ Cleaned up old backtest runs, kept last {keep_last_n} runs")
        
        conn.close() 