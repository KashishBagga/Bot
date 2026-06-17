#!/usr/bin/env python3
"""
PostgreSQL / TimescaleDB Adapter (Monday Readiness)
==================================================
Handles persistent storage for high-frequency trading data.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from dotenv import load_dotenv
import numpy as np

load_dotenv()

logger = logging.getLogger("PostgresDB")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

class PostgresDatabase:
    def __init__(self):
        self.conn_str = os.getenv("DATABASE_URL", "postgresql://trader:trading_pass@127.0.0.1:5433/trading_warehouse")
        self._init_db()

    def _get_connection(self):
        return psycopg2.connect(self.conn_str)

    def _init_db(self):
        """Initialize tables and hyper-tables for TimescaleDB"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 1. Option Snapshots (Hypertable)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS option_snapshots (
                            time TIMESTAMPTZ NOT NULL,
                            underlying TEXT NOT NULL,
                            strike REAL NOT NULL,
                            expiry TEXT NOT NULL,
                            option_type TEXT NOT NULL,
                            ltp REAL,
                            bid REAL,
                            ask REAL,
                            volume INTEGER,
                            oi INTEGER,
                            oi_change INTEGER
                        )
                    ''')

                    # 2. Signal Audit (Complete candidate log for research)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS signal_audit (
                            candidate_id TEXT NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            symbol TEXT NOT NULL,
                            accepted BOOLEAN NOT NULL,
                            setup_type TEXT,
                            rejection_reasons JSONB,
                            score_breakdown JSONB,
                            daily_bias TEXT,
                            hourly_bias TEXT,
                            market_regime TEXT,
                            signal_logic_version TEXT NOT NULL,
                            position_logic_version TEXT NOT NULL,
                            risk_logic_version TEXT NOT NULL,
                            entry_price REAL,
                            stop_loss REAL,
                            take_profit REAL,
                            rr_ratio REAL,
                            PRIMARY KEY (candidate_id, timestamp)
                        )
                    ''')

                    # 3. Trade Signals (Accepted signals only)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS signals (
                            signal_id TEXT NOT NULL,
                            candidate_id TEXT NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            strategy TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            regime TEXT,
                            strength REAL,
                            accepted BOOLEAN,
                            rejected_reason TEXT,
                            executed BOOLEAN DEFAULT FALSE,
                            context JSONB,
                            setup_type TEXT,
                            score_breakdown JSONB,
                            daily_bias TEXT,
                            hourly_bias TEXT,
                            market_regime TEXT,
                            signal_logic_version TEXT,
                            position_logic_version TEXT,
                            risk_logic_version TEXT,
                            PRIMARY KEY (signal_id, timestamp)
                        )
                    ''')

                    # 4. Trade Performance (Summarized lifecycle of execution)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS trade_performance (
                            trade_id TEXT NOT NULL,
                            candidate_id TEXT NOT NULL,
                            entry_time TIMESTAMPTZ NOT NULL,
                            exit_time TIMESTAMPTZ,
                            strategy TEXT,
                            symbol TEXT,
                            entry_price REAL,
                            exit_price REAL,
                            mfe REAL DEFAULT 0.0,
                            mae REAL DEFAULT 0.0,
                            pnl REAL,
                            exit_reason TEXT,
                            features JSONB,
                            setup_type TEXT,
                            mfe_r REAL,
                            mae_r REAL,
                            max_closed_profit_r REAL,
                            final_pnl_r REAL,
                            duration_minutes REAL,
                            bars_held INTEGER,
                            market_regime TEXT,
                            signal_logic_version TEXT,
                            position_logic_version TEXT,
                            risk_logic_version TEXT,
                            stop_loss REAL,
                            take_profit REAL,
                            initial_stop_loss REAL,
                            initial_take_profit REAL,
                            highest_price REAL,
                            lowest_price REAL,
                            stop_loss_distance REAL,
                            signal_type TEXT,
                            capture_rate REAL,
                            PRIMARY KEY (trade_id, entry_time)
                        )
                    ''')

                    # 5. Trade Events (Lifecycle state transitions)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS trade_events (
                            event_id TEXT NOT NULL,
                            trade_id TEXT NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            event_type TEXT NOT NULL,
                            payload JSONB,
                            PRIMARY KEY (event_id, timestamp)
                        )
                    ''')

                    # 6. Counterfactual Results (NEW)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS counterfactual_results (
                            candidate_id TEXT NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            symbol TEXT NOT NULL,
                            signal_type TEXT,
                            setup_type TEXT,
                            rejection_reasons JSONB,
                            primary_rejection_reason TEXT,
                            entry_price REAL,
                            stop_loss REAL,
                            take_profit REAL,
                            initial_stop_loss REAL,
                            initial_take_profit REAL,
                            highest_price REAL,
                            lowest_price REAL,
                            stop_loss_distance REAL,
                            exit_time TIMESTAMPTZ,
                            exit_price REAL,
                            mfe_r REAL DEFAULT 0.0,
                            mae_r REAL DEFAULT 0.0,
                            final_pnl_r REAL DEFAULT 0.0,
                            duration_minutes REAL DEFAULT 0.0,
                            bars_held INTEGER DEFAULT 0,
                            exit_reason TEXT,
                            strategy_version TEXT,
                            capture_rate REAL DEFAULT 0.0,
                            PRIMARY KEY (candidate_id, timestamp)
                        )
                    ''')

                    # 7. Counterfactual Trade Events (NEW)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS counterfactual_trade_events (
                            event_id TEXT NOT NULL,
                            candidate_id TEXT NOT NULL,
                            symbol TEXT NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            event_type TEXT NOT NULL,
                            payload JSONB,
                            PRIMARY KEY (event_id, timestamp)
                        )
                    ''')

                    # --- Migration Checks to Alter Existing Tables safely ---
                    # trade_performance additions
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS stop_loss REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS take_profit REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS initial_stop_loss REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS initial_take_profit REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS highest_price REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS lowest_price REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS stop_loss_distance REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS signal_type TEXT")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS capture_rate REAL")

                    # signal_audit additions
                    cursor.execute("ALTER TABLE signal_audit ADD COLUMN IF NOT EXISTS entry_price REAL")
                    cursor.execute("ALTER TABLE signal_audit ADD COLUMN IF NOT EXISTS stop_loss REAL")
                    cursor.execute("ALTER TABLE signal_audit ADD COLUMN IF NOT EXISTS take_profit REAL")
                    cursor.execute("ALTER TABLE signal_audit ADD COLUMN IF NOT EXISTS rr_ratio REAL")

                conn.commit()
                logger.info("✅ PostgreSQL tables and migrations checked/initialized")
                
                # Convert to hypertables and create indexes
                with conn.cursor() as cursor:
                    try:
                        cursor.execute("SELECT create_hypertable('option_snapshots', 'time', if_not_exists => TRUE)")
                    except Exception:
                        pass
                    try:
                        cursor.execute("SELECT create_hypertable('signal_audit', 'timestamp', if_not_exists => TRUE)")
                    except Exception:
                        pass
                    try:
                        cursor.execute("SELECT create_hypertable('signals', 'timestamp', if_not_exists => TRUE)")
                    except Exception:
                        pass
                    try:
                        cursor.execute("SELECT create_hypertable('trade_performance', 'entry_time', if_not_exists => TRUE)")
                    except Exception:
                        pass
                    try:
                        cursor.execute("SELECT create_hypertable('trade_events', 'timestamp', if_not_exists => TRUE)")
                    except Exception:
                        pass
                    try:
                        cursor.execute("SELECT create_hypertable('counterfactual_results', 'timestamp', if_not_exists => TRUE)")
                    except Exception:
                        pass
                    try:
                        cursor.execute("SELECT create_hypertable('counterfactual_trade_events', 'timestamp', if_not_exists => TRUE)")
                    except Exception:
                        pass

                    # Create Database Indexes
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_audit_cand ON signal_audit(candidate_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_audit_sym ON signal_audit(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_audit_setup ON signal_audit(setup_type)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_cand ON signals(candidate_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_sym ON signals(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_perf_cand ON trade_performance(candidate_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_perf_sym ON trade_performance(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_perf_setup ON trade_performance(setup_type)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_events_trade ON trade_events(trade_id)")
                    
                    # New Counterfactual indexes
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_res_cand ON counterfactual_results(candidate_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_res_sym ON counterfactual_results(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_res_prim_rej ON counterfactual_results(primary_rejection_reason)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_events_cand ON counterfactual_trade_events(candidate_id)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_events_sym ON counterfactual_trade_events(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_events_type ON counterfactual_trade_events(event_type)")

                    # Create Retention Policy on signal_audit (180 days)
                    try:
                        cursor.execute('''
                            DO $$
                            BEGIN
                                IF NOT EXISTS (
                                    SELECT 1 FROM timescaledb_information.jobs 
                                    WHERE proc_name = 'policy_retention' AND hypertable_name = 'signal_audit'
                                ) THEN
                                    PERFORM add_retention_policy('signal_audit', INTERVAL '180 days');
                                END IF;
                            END $$;
                        ''')
                    except Exception as e:
                        logger.warning(f"⚠️ TimescaleDB retention policy not registered: {e}")

                conn.commit()
                logger.info("✅ Option snapshots & audits converted to hypertables, indexes generated")

            # Recreate views safely
            conn2 = self._get_connection()
            conn2.autocommit = True
            with conn2.cursor() as cur:
                try:
                    cur.execute("DROP MATERIALIZED VIEW IF EXISTS research_trade_mart CASCADE")
                    cur.execute("DROP MATERIALIZED VIEW IF EXISTS research_signal_mart CASCADE")
                    
                    cur.execute('''
                        CREATE MATERIALIZED VIEW research_trade_mart AS
                        SELECT 
                            tp.trade_id, tp.symbol, tp.strategy, tp.pnl, tp.mfe, tp.mae,
                            tp.exit_reason,
                            tp.signal_logic_version as version,
                            tp.market_regime as regime,
                            tp.setup_type,
                            tp.mfe_r,
                            tp.mae_r
                        FROM trade_performance tp
                    ''')
                except Exception as e:
                    logger.warning(f"⚠️ Failed to recreate research_trade_mart view: {e}")
                try:
                    cur.execute('''
                        CREATE MATERIALIZED VIEW research_signal_mart AS
                        SELECT signal_id, timestamp, strategy, symbol, regime,
                            accepted, rejected_reason, setup_type, market_regime,
                            score_breakdown->>'rvol' as rvol
                        FROM signals
                    ''')
                except Exception as e:
                    logger.warning(f"⚠️ Failed to recreate research_signal_mart view: {e}")
            conn2.close()
            logger.info("✅ PostgreSQL / TimescaleDB fully initialized")
        except Exception as e:
            logger.error(f"❌ Postgres Init Failed: {e}")

    def save_option_snapshots(self, snapshots: List[Dict[str, Any]]):
        """Bulk insert option snapshots"""
        if not snapshots: return
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    columns = snapshots[0].keys()
                    query = f"INSERT INTO option_snapshots ({','.join(columns)}) VALUES %s"
                    values = [[s[col] for col in columns] for s in snapshots]
                    execute_values(cursor, query, values)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save option snapshots: {e}")

    def save_signal_audit(self, audit: Dict[str, Any]):
        """Save or update signal candidate audit record"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if 'rejection_reasons' in audit:
                        audit['rejection_reasons'] = json.dumps(audit['rejection_reasons'], cls=NumpyEncoder)
                    if 'score_breakdown' in audit:
                        audit['score_breakdown'] = json.dumps(audit['score_breakdown'], cls=NumpyEncoder)
                    columns = list(audit.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    
                    query = f"""
                        INSERT INTO signal_audit ({','.join(columns)}) 
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (candidate_id, timestamp) DO UPDATE SET
                        accepted = EXCLUDED.accepted,
                        setup_type = EXCLUDED.setup_type,
                        rejection_reasons = EXCLUDED.rejection_reasons,
                        score_breakdown = EXCLUDED.score_breakdown,
                        daily_bias = EXCLUDED.daily_bias,
                        hourly_bias = EXCLUDED.hourly_bias,
                        market_regime = EXCLUDED.market_regime,
                        signal_logic_version = EXCLUDED.signal_logic_version,
                        position_logic_version = EXCLUDED.position_logic_version,
                        risk_logic_version = EXCLUDED.risk_logic_version,
                        entry_price = EXCLUDED.entry_price,
                        stop_loss = EXCLUDED.stop_loss,
                        take_profit = EXCLUDED.take_profit,
                        rr_ratio = EXCLUDED.rr_ratio
                    """
                    cursor.execute(query, audit)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save signal audit: {e}")

    def save_signal(self, signal: Dict[str, Any]):
        """Save or update signal snapshot"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if 'score_breakdown' in signal:
                        signal['score_breakdown'] = json.dumps(signal['score_breakdown'], cls=NumpyEncoder)
                    if 'context' in signal:
                        signal['context'] = json.dumps(signal['context'], cls=NumpyEncoder)
                    columns = list(signal.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    
                    query = f"""
                        INSERT INTO signals ({','.join(columns)}) 
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (signal_id, timestamp) DO UPDATE SET
                        executed = EXCLUDED.executed,
                        context = EXCLUDED.context,
                        setup_type = EXCLUDED.setup_type,
                        score_breakdown = EXCLUDED.score_breakdown,
                        daily_bias = EXCLUDED.daily_bias,
                        hourly_bias = EXCLUDED.hourly_bias,
                        market_regime = EXCLUDED.market_regime,
                        signal_logic_version = EXCLUDED.signal_logic_version,
                        position_logic_version = EXCLUDED.position_logic_version,
                        risk_logic_version = EXCLUDED.risk_logic_version
                    """
                    cursor.execute(query, signal)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save signal: {e}")

    def save_trade_performance(self, perf: Dict[str, Any]):
        """Save or update trade performance metadata"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if 'features' in perf:
                        perf['features'] = json.dumps(perf['features'], cls=NumpyEncoder)
                    columns = list(perf.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    
                    query = f"""
                        INSERT INTO trade_performance ({','.join(columns)}) 
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (trade_id, entry_time) DO UPDATE SET
                        exit_time = EXCLUDED.exit_time,
                        exit_price = EXCLUDED.exit_price,
                        mfe = EXCLUDED.mfe,
                        mae = EXCLUDED.mae,
                        pnl = EXCLUDED.pnl,
                        exit_reason = EXCLUDED.exit_reason,
                        setup_type = EXCLUDED.setup_type,
                        mfe_r = EXCLUDED.mfe_r,
                        mae_r = EXCLUDED.mae_r,
                        max_closed_profit_r = EXCLUDED.max_closed_profit_r,
                        final_pnl_r = EXCLUDED.final_pnl_r,
                        duration_minutes = EXCLUDED.duration_minutes,
                        bars_held = EXCLUDED.bars_held,
                        market_regime = EXCLUDED.market_regime,
                        signal_logic_version = EXCLUDED.signal_logic_version,
                        position_logic_version = EXCLUDED.position_logic_version,
                        risk_logic_version = EXCLUDED.risk_logic_version,
                        stop_loss = EXCLUDED.stop_loss,
                        take_profit = EXCLUDED.take_profit,
                        initial_stop_loss = EXCLUDED.initial_stop_loss,
                        initial_take_profit = EXCLUDED.initial_take_profit,
                        highest_price = EXCLUDED.highest_price,
                        lowest_price = EXCLUDED.lowest_price,
                        stop_loss_distance = EXCLUDED.stop_loss_distance,
                        signal_type = EXCLUDED.signal_type,
                        capture_rate = EXCLUDED.capture_rate
                    """
                    cursor.execute(query, perf)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save trade performance: {e}")

    def save_trade_event(self, event: Dict[str, Any]):
        """Save trade event lifecycle tracking record"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if 'payload' in event:
                        event['payload'] = json.dumps(event['payload'], cls=NumpyEncoder)
                    columns = list(event.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    
                    query = f"""
                        INSERT INTO trade_events ({','.join(columns)}) 
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (event_id, timestamp) DO NOTHING
                    """
                    cursor.execute(query, event)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save trade event: {e}")

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch all currently open real positions for recovery"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT * FROM trade_performance 
                        WHERE exit_time IS NULL
                    """)
                    return list(cursor.fetchall())
        except Exception as e:
            logger.error(f"❌ Failed to fetch open positions: {e}")
            return []

    def get_open_counterfactuals(self) -> List[Dict[str, Any]]:
        """Fetch all currently open counterfactual positions for recovery"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("""
                        SELECT * FROM counterfactual_results 
                        WHERE exit_time IS NULL
                    """)
                    return list(cursor.fetchall())
        except Exception as e:
            logger.error(f"❌ Failed to fetch open counterfactuals: {e}")
            return []

    def save_counterfactual_result(self, result: Dict[str, Any]):
        """Save or update counterfactual research trade result"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if 'rejection_reasons' in result:
                        result['rejection_reasons'] = json.dumps(result['rejection_reasons'], cls=NumpyEncoder)
                    columns = list(result.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    
                    query = f"""
                        INSERT INTO counterfactual_results ({','.join(columns)}) 
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (candidate_id, timestamp) DO UPDATE SET
                        exit_time = EXCLUDED.exit_time,
                        exit_price = EXCLUDED.exit_price,
                        mfe_r = EXCLUDED.mfe_r,
                        mae_r = EXCLUDED.mae_r,
                        final_pnl_r = EXCLUDED.final_pnl_r,
                        duration_minutes = EXCLUDED.duration_minutes,
                        bars_held = EXCLUDED.bars_held,
                        exit_reason = EXCLUDED.exit_reason,
                        stop_loss = EXCLUDED.stop_loss,
                        take_profit = EXCLUDED.take_profit,
                        highest_price = EXCLUDED.highest_price,
                        lowest_price = EXCLUDED.lowest_price,
                        capture_rate = EXCLUDED.capture_rate
                    """
                    cursor.execute(query, result)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save counterfactual result: {e}")

    def save_counterfactual_event(self, event: Dict[str, Any]):
        """Save counterfactual event lifecycle tracking record"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if 'payload' in event:
                        event['payload'] = json.dumps(event['payload'], cls=NumpyEncoder)
                    columns = list(event.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    
                    query = f"""
                        INSERT INTO counterfactual_trade_events ({','.join(columns)}) 
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (event_id, timestamp) DO NOTHING
                    """
                    cursor.execute(query, event)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save counterfactual event: {e}")

    def refresh_research_marts(self):
        """Refresh materialized views for research"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("REFRESH MATERIALIZED VIEW research_trade_mart")
                    cursor.execute("REFRESH MATERIALIZED VIEW research_signal_mart")
                conn.commit()
                logger.info("✨ Research Marts refreshed")
        except Exception as e:
            logger.error(f"❌ Failed to refresh Research Marts: {e}")
