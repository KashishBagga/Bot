"""
Unified Database Manager for Trading Bot System
Consolidates all database operations and provides optimized schema for complete signal tracking.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


class UnifiedDatabase:
    """Unified database manager that consolidates all trading signal storage needs."""
    
    def __init__(self, db_path: str = "trading_signals.db"):
        self.db_path = db_path
        self.setup_all_tables()
    
    def setup_all_tables(self):
        """Setup all required tables with optimized schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. BACKTESTING MASTER TABLE - For all backtesting signals
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtesting_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,  -- Links to backtesting_runs
                    signal_time TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,  -- BUY CALL, BUY PUT, NO TRADE, ERROR
                    
                    -- Core signal data
                    price REAL,
                    confidence TEXT,
                    confidence_score INTEGER,
                    strike_price INTEGER,
                    stop_loss REAL,
                    target REAL,
                    target2 REAL,
                    target3 REAL,
                    trade_type TEXT,
                    
                    -- Outcome data
                    outcome TEXT,  -- WIN, LOSS, BREAKEVEN, PENDING
                    pnl REAL,
                    targets_hit INTEGER,
                    stoploss_count INTEGER,
                    exit_time TEXT,
                    failure_reason TEXT,
                    
                    -- Universal indicators (common across strategies)
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    ema_20 REAL,
                    atr REAL,
                    
                    -- Strategy-specific indicators (JSON for flexibility)
                    indicators_data TEXT,  -- JSON with strategy-specific indicators
                    reasoning TEXT,  -- Strategy reasoning
                    
                    -- Metadata
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (run_id) REFERENCES backtesting_runs (id)
                )
            ''')
            
            # 2. LIVE TRADING MASTER TABLE - For all live trading signals
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_time TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,  -- BUY CALL, BUY PUT, NO TRADE, ERROR
                    
                    -- Core signal data
                    price REAL,
                    confidence TEXT,
                    confidence_score INTEGER,
                    strike_price INTEGER,
                    stop_loss REAL,
                    target REAL,
                    target2 REAL,
                    target3 REAL,
                    trade_type TEXT,
                    
                    -- Execution tracking
                    status TEXT DEFAULT 'GENERATED',  -- GENERATED, EXECUTED, REJECTED, CLOSED
                    execution_price REAL,
                    execution_time TEXT,
                    exit_price REAL,
                    exit_time TEXT,
                    exit_reason TEXT,
                    
                    -- Outcome data
                    outcome TEXT,  -- WIN, LOSS, BREAKEVEN, PENDING
                    pnl REAL,
                    pnl_percentage REAL,
                    targets_hit INTEGER,
                    stoploss_count INTEGER,
                    failure_reason TEXT,
                    
                    -- Performance metrics
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    holding_period_minutes INTEGER,
                    
                    -- Universal indicators (common across strategies)
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    ema_20 REAL,
                    atr REAL,
                    
                    -- Strategy-specific indicators (JSON for flexibility)
                    indicators_data TEXT,  -- JSON with strategy-specific indicators
                    reasoning TEXT,  -- Strategy reasoning
                    
                    -- Market context
                    market_condition TEXT,
                    volatility_at_entry REAL,
                    volume_at_entry REAL,
                    
                    -- Metadata
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 3. REJECTED SIGNALS TABLE - For comprehensive rejected signal analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rejected_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_time TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_attempted TEXT,  -- What signal was attempted
                    rejection_reason TEXT NOT NULL,
                    rejection_category TEXT,  -- NO_TRADE, LOW_CONFIDENCE, ERROR, RISK_LIMIT
                    
                    -- Market data at rejection time
                    price REAL,
                    confidence_score INTEGER,
                    
                    -- Universal indicators (for analysis of why rejected)
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    ema_20 REAL,
                    atr REAL,
                    
                    -- Strategy-specific indicators (JSON for flexibility)
                    indicators_data TEXT,  -- JSON with ALL indicator values
                    reasoning TEXT,  -- Detailed reasoning for rejection
                    
                    -- Potential trade data (what would have been)
                    potential_stop_loss REAL,
                    potential_target REAL,
                    potential_target2 REAL,
                    potential_target3 REAL,
                    
                    -- Source tracking
                    source TEXT DEFAULT 'LIVE',  -- LIVE or BACKTEST
                    run_id INTEGER,  -- For backtesting runs
                    
                    -- Metadata
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 4. BACKTESTING RUNS TABLE - For tracking backtest executions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtesting_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TEXT NOT NULL,
                    run_name TEXT,
                    period_days INTEGER,
                    timeframe TEXT,
                    symbols TEXT,  -- JSON array of symbols
                    strategies TEXT,  -- JSON array of strategies
                    parameters TEXT,  -- JSON of run parameters
                    
                    -- Performance summary
                    total_signals INTEGER DEFAULT 0,
                    valid_signals INTEGER DEFAULT 0,
                    rejected_signals INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    
                    -- Execution metrics
                    duration_seconds REAL,
                    signals_per_second REAL,
                    
                    -- Status
                    status TEXT DEFAULT 'RUNNING',  -- RUNNING, COMPLETED, FAILED
                    error_message TEXT,
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 5. STRATEGY PERFORMANCE SUMMARY - For quick performance queries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    period_start TEXT,
                    period_end TEXT,
                    source TEXT,  -- LIVE or BACKTEST
                    run_id INTEGER,
                    
                    -- Signal counts
                    total_analyses INTEGER,
                    valid_signals INTEGER,
                    rejected_signals INTEGER,
                    no_trade_count INTEGER,
                    error_count INTEGER,
                    
                    -- Performance metrics
                    total_pnl REAL,
                    win_rate REAL,
                    avg_confidence REAL,
                    max_drawdown REAL,
                    
                    -- Rejection analysis
                    low_confidence_rejections INTEGER,
                    no_trade_rejections INTEGER,
                    error_rejections INTEGER,
                    
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 6. DATABASE VIEWS for easy querying
            # Latest backtesting results view
            cursor.execute('''
                CREATE VIEW IF NOT EXISTS latest_backtest_summary AS
                SELECT 
                    br.id as run_id,
                    br.run_timestamp,
                    br.period_days,
                    br.timeframe,
                    br.total_signals,
                    br.total_pnl,
                    COUNT(bs.id) as detailed_signals,
                    SUM(CASE WHEN bs.signal != 'NO TRADE' THEN 1 ELSE 0 END) as valid_signals,
                    AVG(bs.confidence_score) as avg_confidence,
                    SUM(bs.pnl) as calculated_pnl
                FROM backtesting_runs br
                LEFT JOIN backtesting_signals bs ON br.id = bs.run_id
                WHERE br.id = (SELECT MAX(id) FROM backtesting_runs)
                GROUP BY br.id
            ''')
            
            # Strategy comparison view
            cursor.execute('''
                CREATE VIEW IF NOT EXISTS strategy_comparison AS
                SELECT 
                    strategy,
                    symbol,
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN signal NOT IN ('NO TRADE', 'ERROR') THEN 1 ELSE 0 END) as valid_signals,
                    AVG(confidence_score) as avg_confidence,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) * 100.0 / 
                        NULLIF(SUM(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0) as win_rate
                FROM backtesting_signals 
                WHERE run_id = (SELECT MAX(id) FROM backtesting_runs)
                GROUP BY strategy, symbol
                ORDER BY total_pnl DESC
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Unified database schema created successfully")
            
        except Exception as e:
            print(f"❌ Database setup error: {e}")
            raise
    
    def start_backtesting_run(self, run_name: str, period_days: int, timeframe: str, 
                             symbols: List[str], strategies: List[str], 
                             parameters: Dict = None) -> int:
        """Start a new backtesting run and return run_id."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO backtesting_runs (
                    run_timestamp, run_name, period_days, timeframe, symbols, strategies, parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                run_name,
                period_days,
                timeframe,
                json.dumps(symbols),
                json.dumps(strategies),
                json.dumps(parameters or {})
            ))
            
            run_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"✅ Started backtesting run {run_id}: {run_name}")
            return run_id
            
        except Exception as e:
            print(f"❌ Error starting backtesting run: {e}")
            return -1
    
    def log_backtesting_signal(self, run_id: int, strategy: str, symbol: str, 
                              signal_data: Dict[str, Any]):
        """Log a backtesting signal with complete indicator data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract strategy-specific indicators
            indicators_data = self._extract_strategy_indicators(strategy, signal_data)
            
            cursor.execute('''
                INSERT INTO backtesting_signals (
                    run_id, signal_time, strategy, symbol, signal,
                    price, confidence, confidence_score, strike_price, stop_loss, target, target2, target3, trade_type,
                    outcome, pnl, targets_hit, stoploss_count, exit_time, failure_reason,
                    rsi, macd, macd_signal, ema_20, atr,
                    indicators_data, reasoning
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                signal_data.get('signal_time', datetime.now().isoformat()),
                strategy,
                symbol,
                signal_data.get('signal', 'NO TRADE'),
                
                # Core data
                signal_data.get('price', 0),
                signal_data.get('confidence', 'Unknown'),
                signal_data.get('confidence_score', 0),
                signal_data.get('strike_price', 0),
                signal_data.get('stop_loss', 0),
                signal_data.get('target', 0),
                signal_data.get('target2', 0),
                signal_data.get('target3', 0),
                signal_data.get('trade_type', 'Intraday'),
                
                # Outcome
                signal_data.get('outcome', 'Pending'),
                signal_data.get('pnl', 0),
                signal_data.get('targets_hit', 0),
                signal_data.get('stoploss_count', 0),
                signal_data.get('exit_time'),
                signal_data.get('failure_reason', ''),
                
                # Universal indicators
                signal_data.get('rsi', 0),
                signal_data.get('macd', 0),
                signal_data.get('macd_signal', 0),
                signal_data.get('ema_20', 0),
                signal_data.get('atr', 0),
                
                # Strategy-specific data
                json.dumps(indicators_data),
                signal_data.get('reasoning', signal_data.get('rsi_reason', '') + ' | ' + signal_data.get('macd_reason', '') + ' | ' + signal_data.get('price_reason', ''))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error logging backtesting signal: {e}")
    
    def log_live_signal(self, strategy: str, symbol: str, signal_data: Dict[str, Any]) -> int:
        """Log a live trading signal and return signal_id."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract strategy-specific indicators
            indicators_data = self._extract_strategy_indicators(strategy, signal_data)
            
            cursor.execute('''
                INSERT INTO live_signals (
                    signal_time, strategy, symbol, signal,
                    price, confidence, confidence_score, strike_price, stop_loss, target, target2, target3, trade_type,
                    rsi, macd, macd_signal, ema_20, atr,
                    indicators_data, reasoning,
                    market_condition, volatility_at_entry, volume_at_entry
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data.get('signal_time', datetime.now().isoformat()),
                strategy,
                symbol,
                signal_data.get('signal', 'NO TRADE'),
                
                # Core data
                signal_data.get('price', 0),
                signal_data.get('confidence', 'Unknown'),
                signal_data.get('confidence_score', 0),
                signal_data.get('strike_price', 0),
                signal_data.get('stop_loss', 0),
                signal_data.get('target', 0),
                signal_data.get('target2', 0),
                signal_data.get('target3', 0),
                signal_data.get('trade_type', 'Intraday'),
                
                # Universal indicators
                signal_data.get('rsi', 0),
                signal_data.get('macd', 0),
                signal_data.get('macd_signal', 0),
                signal_data.get('ema_20', 0),
                signal_data.get('atr', 0),
                
                # Strategy-specific and context
                json.dumps(indicators_data),
                signal_data.get('reasoning', ''),
                signal_data.get('market_condition', 'Unknown'),
                signal_data.get('volatility_at_entry', 0),
                signal_data.get('volume_at_entry', 0)
            ))
            
            signal_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return signal_id
            
        except Exception as e:
            print(f"❌ Error logging live signal: {e}")
            return -1
    
    def log_rejected_signal(self, strategy: str, symbol: str, rejection_data: Dict[str, Any], 
                           source: str = 'LIVE', run_id: int = None):
        """Log a rejected signal with complete analysis data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract strategy-specific indicators
            indicators_data = self._extract_strategy_indicators(strategy, rejection_data)
            
            # Determine rejection category
            rejection_reason = rejection_data.get('rejection_reason', 'Unknown')
            if 'low confidence' in rejection_reason.lower():
                rejection_category = 'LOW_CONFIDENCE'
            elif 'no trade' in rejection_reason.lower():
                rejection_category = 'NO_TRADE'
            elif 'error' in rejection_reason.lower():
                rejection_category = 'ERROR'
            elif 'risk' in rejection_reason.lower():
                rejection_category = 'RISK_LIMIT'
            else:
                rejection_category = 'OTHER'
            
            cursor.execute('''
                INSERT INTO rejected_signals (
                    signal_time, strategy, symbol, signal_attempted, rejection_reason, rejection_category,
                    price, confidence_score,
                    rsi, macd, macd_signal, ema_20, atr,
                    indicators_data, reasoning,
                    potential_stop_loss, potential_target, potential_target2, potential_target3,
                    source, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rejection_data.get('timestamp', rejection_data.get('signal_time', datetime.now().isoformat())),
                strategy,
                symbol,
                rejection_data.get('signal', 'UNKNOWN'),
                rejection_reason,
                rejection_category,
                
                # Market data
                rejection_data.get('price', 0),
                rejection_data.get('confidence_score', 0),
                
                # Universal indicators
                rejection_data.get('rsi', 0),
                rejection_data.get('macd', 0),
                rejection_data.get('macd_signal', 0),
                rejection_data.get('ema_20', 0),
                rejection_data.get('atr', 0),
                
                # Strategy-specific data
                json.dumps(indicators_data),
                rejection_data.get('reasoning', rejection_data.get('reason', '')),
                
                # Potential trade data
                rejection_data.get('stop_loss', 0),
                rejection_data.get('target', 0),
                rejection_data.get('target2', 0),
                rejection_data.get('target3', 0),
                
                # Source tracking
                source,
                run_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error logging rejected signal: {e}")
    
    def update_live_signal_outcome(self, signal_id: int, outcome_data: Dict[str, Any]):
        """Update live signal with outcome and performance data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE live_signals SET
                    status = ?, exit_price = ?, exit_time = ?, exit_reason = ?,
                    outcome = ?, pnl = ?, pnl_percentage = ?, targets_hit = ?, stoploss_count = ?,
                    max_favorable_excursion = ?, max_adverse_excursion = ?, holding_period_minutes = ?,
                    failure_reason = ?, updated_at = ?
                WHERE id = ?
            ''', (
                outcome_data.get('status', 'CLOSED'),
                outcome_data.get('exit_price', 0),
                outcome_data.get('exit_time', datetime.now().isoformat()),
                outcome_data.get('exit_reason', ''),
                outcome_data.get('outcome', 'UNKNOWN'),
                outcome_data.get('pnl', 0),
                outcome_data.get('pnl_percentage', 0),
                outcome_data.get('targets_hit', 0),
                outcome_data.get('stoploss_count', 0),
                outcome_data.get('max_favorable_excursion', 0),
                outcome_data.get('max_adverse_excursion', 0),
                outcome_data.get('holding_period_minutes', 0),
                outcome_data.get('failure_reason', ''),
                datetime.now().isoformat(),
                signal_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error updating signal outcome: {e}")
    
    def finish_backtesting_run(self, run_id: int, summary_data: Dict[str, Any]):
        """Finish a backtesting run with summary statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE backtesting_runs SET
                    total_signals = ?, valid_signals = ?, rejected_signals = ?, total_pnl = ?, win_rate = ?,
                    duration_seconds = ?, signals_per_second = ?, status = ?
                WHERE id = ?
            ''', (
                summary_data.get('total_signals', 0),
                summary_data.get('valid_signals', 0),
                summary_data.get('rejected_signals', 0),
                summary_data.get('total_pnl', 0),
                summary_data.get('win_rate', 0),
                summary_data.get('duration_seconds', 0),
                summary_data.get('signals_per_second', 0),
                'COMPLETED',
                run_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error finishing backtesting run: {e}")
    
    def _extract_strategy_indicators(self, strategy: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract strategy-specific indicators from signal data."""
        indicators = {}
        
        if strategy == 'insidebar_rsi':
            indicators.update({
                'rsi_level': signal_data.get('rsi_level'),
                'inside_bar_detected': signal_data.get('inside_bar_detected'),
                'rsi_reason': signal_data.get('rsi_reason'),
                'price_reason': signal_data.get('price_reason')
            })
        
        elif strategy == 'ema_crossover':
            indicators.update({
                'ema_fast': signal_data.get('ema_fast'),
                'ema_slow': signal_data.get('ema_slow'),
                'ema_9': signal_data.get('ema_9'),
                'ema_21': signal_data.get('ema_21'),
                'crossover_strength': signal_data.get('crossover_strength'),
                'momentum': signal_data.get('momentum'),
                'rsi_reason': signal_data.get('rsi_reason'),
                'macd_reason': signal_data.get('macd_reason'),
                'price_reason': signal_data.get('price_reason')
            })
        
        elif strategy == 'supertrend_ema':
            indicators.update({
                'supertrend_value': signal_data.get('supertrend_value'),
                'supertrend_direction': signal_data.get('supertrend_direction'),
                'supertrend_upperband': signal_data.get('supertrend_upperband'),
                'supertrend_lowerband': signal_data.get('supertrend_lowerband'),
                'price_to_ema_ratio': signal_data.get('price_to_ema_ratio'),
                'price_reason': signal_data.get('price_reason')
            })
        
        elif strategy == 'supertrend_macd_rsi_ema':
            indicators.update({
                'supertrend': signal_data.get('supertrend'),
                'supertrend_direction': signal_data.get('supertrend_direction'),
                'option_chain_confirmation': signal_data.get('option_chain_confirmation'),
                'option_symbol': signal_data.get('option_symbol'),
                'option_expiry': signal_data.get('option_expiry'),
                'option_strike': signal_data.get('option_strike'),
                'option_type': signal_data.get('option_type'),
                'option_entry_price': signal_data.get('option_entry_price'),
                'rsi_reason': signal_data.get('rsi_reason'),
                'macd_reason': signal_data.get('macd_reason'),
                'price_reason': signal_data.get('price_reason')
            })
        
        # Remove None values
        return {k: v for k, v in indicators.items() if v is not None}
    
    def get_latest_backtest_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of the latest backtesting run."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM latest_backtest_summary')
            result = cursor.fetchone()
            
            if result:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, result))
            
            conn.close()
            return None
            
        except Exception as e:
            print(f"❌ Error getting latest backtest summary: {e}")
            return None
    
    def get_strategy_comparison(self, run_id: int = None) -> List[Dict[str, Any]]:
        """Get strategy performance comparison."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if run_id:
                cursor.execute('''
                    SELECT strategy, symbol, 
                           COUNT(*) as total_signals,
                           SUM(CASE WHEN signal NOT IN ('NO TRADE', 'ERROR') THEN 1 ELSE 0 END) as valid_signals,
                           AVG(confidence_score) as avg_confidence,
                           SUM(pnl) as total_pnl,
                           AVG(pnl) as avg_pnl,
                           SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) * 100.0 / 
                               NULLIF(SUM(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END), 0) as win_rate
                    FROM backtesting_signals 
                    WHERE run_id = ?
                    GROUP BY strategy, symbol
                    ORDER BY total_pnl DESC
                ''', (run_id,))
            else:
                cursor.execute('SELECT * FROM strategy_comparison')
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            conn.close()
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            print(f"❌ Error getting strategy comparison: {e}")
            return []
    
    def get_rejection_analysis(self, strategy: str = None, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive rejection analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            where_clause = "WHERE signal_time >= datetime('now', '-{} days')".format(days)
            if strategy:
                where_clause += f" AND strategy = '{strategy}'"
            
            # Get rejection counts by category
            cursor.execute(f'''
                SELECT rejection_category, COUNT(*) as count
                FROM rejected_signals
                {where_clause}
                GROUP BY rejection_category
                ORDER BY count DESC
            ''')
            
            categories = dict(cursor.fetchall())
            
            # Get rejection reasons
            cursor.execute(f'''
                SELECT rejection_reason, COUNT(*) as count
                FROM rejected_signals
                {where_clause}
                GROUP BY rejection_reason
                ORDER BY count DESC
                LIMIT 10
            ''')
            
            reasons = dict(cursor.fetchall())
            
            # Get strategy breakdown
            cursor.execute(f'''
                SELECT strategy, rejection_category, COUNT(*) as count
                FROM rejected_signals
                {where_clause}
                GROUP BY strategy, rejection_category
                ORDER BY strategy, count DESC
            ''')
            
            strategy_breakdown = {}
            for row in cursor.fetchall():
                strategy_name, category, count = row
                if strategy_name not in strategy_breakdown:
                    strategy_breakdown[strategy_name] = {}
                strategy_breakdown[strategy_name][category] = count
            
            conn.close()
            
            return {
                'categories': categories,
                'top_reasons': reasons,
                'strategy_breakdown': strategy_breakdown,
                'period_days': days
            }
            
        except Exception as e:
            print(f"❌ Error getting rejection analysis: {e}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old rejected signals and backtesting data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Keep only recent rejected signals for live trading
            cursor.execute('''
                DELETE FROM rejected_signals 
                WHERE source = 'LIVE' AND signal_time < datetime('now', '-{} days')
            '''.format(days_to_keep))
            
            live_deleted = cursor.rowcount
            
            # For backtesting, keep only the last 10 runs
            cursor.execute('''
                DELETE FROM backtesting_runs 
                WHERE id NOT IN (
                    SELECT id FROM backtesting_runs 
                    ORDER BY id DESC LIMIT 10
                )
            ''')
            
            backtest_runs_deleted = cursor.rowcount
            
            # Clean up orphaned backtesting signals
            cursor.execute('''
                DELETE FROM backtesting_signals 
                WHERE run_id NOT IN (SELECT id FROM backtesting_runs)
            ''')
            
            backtest_signals_deleted = cursor.rowcount
            
            # Clean up orphaned rejected signals from backtesting
            cursor.execute('''
                DELETE FROM rejected_signals 
                WHERE source = 'BACKTEST' AND run_id NOT IN (SELECT id FROM backtesting_runs)
            ''')
            
            rejected_backtest_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"✅ Cleanup completed:")
            print(f"   Live rejected signals deleted: {live_deleted}")
            print(f"   Backtest runs deleted: {backtest_runs_deleted}")
            print(f"   Backtest signals deleted: {backtest_signals_deleted}")
            print(f"   Rejected backtest signals deleted: {rejected_backtest_deleted}")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}") 