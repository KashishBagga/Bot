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
                            holding_efficiency REAL,
                            valid BOOLEAN DEFAULT TRUE,
                            validation_errors TEXT,
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
                            holding_efficiency REAL,
                            valid BOOLEAN DEFAULT TRUE,
                            validation_errors TEXT,
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

                    # 7b. Execution Events (NEW)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS execution_events (
                            event_id TEXT NOT NULL,
                            trade_id TEXT,
                            candidate_id TEXT,
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

                    # ── Experiment Framework Migrations (v1) ────────────────────────────
                    # Add experiment_name, strategy_id, version to all research tables.
                    # Existing rows default to the production experiment name.
                    _EXP_DEFAULT = 'Structural_v3.2_RVOL1.0'
                    _SID_DEFAULT = 'structural'
                    _VER_DEFAULT = 'v3.2'

                    for table in ('signal_audit', 'signals', 'trade_performance',
                                  'trade_events', 'counterfactual_results',
                                  'counterfactual_trade_events'):
                        cursor.execute(
                            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
                            f"experiment_name TEXT DEFAULT '{_EXP_DEFAULT}'"
                        )

                    for table in ('signal_audit', 'signals', 'trade_performance',
                                  'counterfactual_results'):
                        cursor.execute(
                            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
                            f"strategy_id TEXT DEFAULT '{_SID_DEFAULT}'"
                        )
                        cursor.execute(
                            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
                            f"version TEXT DEFAULT '{_VER_DEFAULT}'"
                        )

                    # Experiment metadata table (one row per registered Experiment)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS experiments (
                            name        TEXT PRIMARY KEY,
                            strategy_id TEXT NOT NULL,
                            version     TEXT NOT NULL,
                            config_hash TEXT NOT NULL,
                            git_commit  TEXT,
                            params      JSONB,
                            description TEXT,
                            created_at  TIMESTAMPTZ DEFAULT NOW(),
                            status      TEXT DEFAULT 'active',
                            notes       TEXT,
                            strategy_metadata JSONB
                        )
                    ''')

                    # Research Decisions table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS research_decisions (
                            decision_id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMPTZ DEFAULT NOW(),
                            author TEXT,
                            strategy_id TEXT,
                            experiment_name TEXT,
                            parameter_name TEXT,
                            old_value TEXT,
                            new_value TEXT,
                            evidence TEXT,
                            expected_pnl_change REAL,
                            notes TEXT
                        )
                    ''')

                    # confidence and diagnostics columns
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS confidence REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS diagnostics JSONB")
                    cursor.execute("ALTER TABLE counterfactual_results ADD COLUMN IF NOT EXISTS confidence REAL")
                    cursor.execute("ALTER TABLE counterfactual_results ADD COLUMN IF NOT EXISTS diagnostics JSONB")
                    cursor.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS strategy_metadata JSONB")

                    # holding_efficiency = final_pnl_r / bars_held (R per bar)
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS holding_efficiency REAL")
                    cursor.execute("ALTER TABLE counterfactual_results ADD COLUMN IF NOT EXISTS holding_efficiency REAL")

                    # Data validation columns
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS valid BOOLEAN DEFAULT TRUE")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS validation_errors TEXT")
                    cursor.execute("ALTER TABLE counterfactual_results ADD COLUMN IF NOT EXISTS valid BOOLEAN DEFAULT TRUE")
                    cursor.execute("ALTER TABLE counterfactual_results ADD COLUMN IF NOT EXISTS validation_errors TEXT")

                    # research_id columns and indexes (M2B)
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS research_id VARCHAR(24)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_research_id ON trade_performance(research_id) WHERE research_id IS NOT NULL")

                    # Position sizing — was only ever kept in-memory, so a restart
                    # lost it and _deployed_capital() undercounted recovered real
                    # positions, silently weakening the exposure gate.
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS position_size_inr REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS lots REAL")

                    # Live dashboard support: per-candle heartbeat for open real
                    # positions (previously the row was only updated on a SL trail
                    # or TP expansion, so it went stale between those events), and
                    # tp1 — the 1.5R partial target every strategy already computes
                    # but which was never persisted or surfaced anywhere.
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS current_price REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS unrealized_pnl_r REAL")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS last_heartbeat_at TIMESTAMPTZ")
                    cursor.execute("ALTER TABLE trade_performance ADD COLUMN IF NOT EXISTS tp1 REAL")
                    cursor.execute("ALTER TABLE counterfactual_results ADD COLUMN IF NOT EXISTS tp1 REAL")

                    # Signals that passed every strategy filter (accepted=True) but
                    # were still blocked from becoming a real trade by the portfolio
                    # risk governor (daily-loss halt / max concurrent / max deployed
                    # capital). Previously these were just a warning log line — saved
                    # nowhere, so "why are real trades near-zero" couldn't be
                    # distinguished from "strategy filters are just strict".
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS risk_governor_blocks (
                            block_id        TEXT        NOT NULL,
                            timestamp       TIMESTAMPTZ NOT NULL,
                            symbol          TEXT        NOT NULL,
                            experiment_name TEXT        NOT NULL,
                            setup_type      TEXT,
                            signal_type     TEXT,
                            candidate_id    TEXT,
                            gate_reason     TEXT        NOT NULL,
                            entry_price     REAL,
                            stop_loss       REAL,
                            take_profit     REAL,
                            rr_ratio        REAL,
                            PRIMARY KEY (block_id, timestamp)
                        )
                    ''')
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS idx_risk_blocks_reason "
                        "ON risk_governor_blocks(gate_reason, timestamp)"
                    )

                    # One row per symbol, overwritten every candle — "what does the
                    # system currently believe about this market": bias, regime,
                    # RVOL/ATR/efficiency, active S/R zones, and in-progress/ready
                    # chart patterns. Previously none of this was persisted anywhere
                    # queryable; MarketSnapshot lived only in-memory inside the
                    # trader process for the duration of one candle. Feeds the
                    # dashboard's Market State page.
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS market_state (
                            symbol               TEXT        PRIMARY KEY,
                            updated_at           TIMESTAMPTZ NOT NULL,
                            current_price        REAL,
                            daily_bias           TEXT,
                            market_regime        TEXT,
                            rvol                 REAL,
                            atr                  REAL,
                            move_efficiency      REAL,
                            wickiness            REAL,
                            narrative_bias       TEXT,
                            narrative_confidence REAL,
                            zones                JSONB,
                            patterns             JSONB
                        )
                    ''')

                    # Multi-leg options combos (vertical spreads, straddle/strangle).
                    # Deliberately a SEPARATE schema from trade_performance, not a
                    # retrofit — a combo's risk/PnL is combined-premium-based, not a
                    # single directional index-price R-multiple, and forcing it into
                    # the single-leg schema would corrupt both. `legs` stores each
                    # leg's resolved contract + entry/exit premiums as JSONB (same
                    # convention as diagnostics/zones/patterns elsewhere) rather than
                    # a child table — legs are fixed at entry and don't need
                    # independent relational queries.
                    #
                    # final_pnl_r is combined premium P&L divided by max_loss (the
                    # premium paid, for every combo type built so far — vertical
                    # spreads and long straddle/strangle are both debit/defined-risk
                    # structures) — this keeps combos comparable in R-multiple terms
                    # to every other strategy's expectancy stats.
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS combo_trades (
                            combo_id           TEXT        NOT NULL,
                            entry_time         TIMESTAMPTZ NOT NULL,
                            exit_time          TIMESTAMPTZ,
                            symbol             TEXT        NOT NULL,
                            experiment_name    TEXT,
                            strategy_id        TEXT,
                            version            TEXT,
                            combo_type         TEXT        NOT NULL,
                            setup_type         TEXT,
                            underlying_entry_price REAL,
                            underlying_exit_price  REAL,
                            legs               JSONB,
                            net_premium_paid   REAL,
                            max_loss           REAL,
                            max_profit         REAL,
                            target_r           REAL,
                            stop_r             REAL,
                            current_pnl_r      REAL,
                            final_pnl_r        REAL,
                            exit_reason        TEXT,
                            duration_minutes   REAL,
                            confidence         REAL,
                            diagnostics        JSONB,
                            valid              BOOLEAN     DEFAULT TRUE,
                            validation_errors  TEXT,
                            PRIMARY KEY (combo_id, entry_time)
                        )
                    ''')
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS combo_trade_events (
                            event_id   TEXT        NOT NULL,
                            combo_id   TEXT,
                            timestamp  TIMESTAMPTZ NOT NULL,
                            event_type TEXT        NOT NULL,
                            payload    JSONB,
                            PRIMARY KEY (event_id, timestamp)
                        )
                    ''')
                    # Counterfactual mirror — same "track every rejection" guarantee
                    # the single-leg system already has, same combined-premium engine,
                    # separate storage only.
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS counterfactual_combo_results (
                            combo_id           TEXT        NOT NULL,
                            entry_time         TIMESTAMPTZ NOT NULL,
                            exit_time          TIMESTAMPTZ,
                            symbol             TEXT        NOT NULL,
                            experiment_name    TEXT,
                            strategy_id        TEXT,
                            version            TEXT,
                            combo_type         TEXT        NOT NULL,
                            setup_type         TEXT,
                            rejection_reasons  JSONB,
                            primary_rejection_reason TEXT,
                            underlying_entry_price REAL,
                            underlying_exit_price  REAL,
                            legs               JSONB,
                            net_premium_paid   REAL,
                            max_loss           REAL,
                            max_profit         REAL,
                            target_r           REAL,
                            stop_r             REAL,
                            current_pnl_r      REAL,
                            final_pnl_r        REAL,
                            exit_reason        TEXT,
                            duration_minutes   REAL,
                            confidence         REAL,
                            diagnostics        JSONB,
                            valid              BOOLEAN     DEFAULT TRUE,
                            validation_errors  TEXT,
                            PRIMARY KEY (combo_id, entry_time)
                        )
                    ''')
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS counterfactual_combo_events (
                            event_id   TEXT        NOT NULL,
                            combo_id   TEXT,
                            timestamp  TIMESTAMPTZ NOT NULL,
                            event_type TEXT        NOT NULL,
                            payload    JSONB,
                            PRIMARY KEY (event_id, timestamp)
                        )
                    ''')

                    # NOTE: the market_events research_id ALTER/INDEX was moved to
                    # AFTER `CREATE TABLE market_events` below. On a fresh database the
                    # table does not yet exist here, so running them at this point
                    # raised, rolled back the whole init transaction, and left the DB
                    # with NO tables while the error was swallowed.


                    # Per-experiment daily summary: one row per (date, experiment) written at session end
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS experiment_daily_metrics (
                            date                DATE        NOT NULL,
                            experiment_name     TEXT        NOT NULL,
                            real_trades         INT         DEFAULT 0,
                            cf_trades           INT         DEFAULT 0,
                            wins                INT         DEFAULT 0,
                            losses              INT         DEFAULT 0,
                            expectancy          REAL,
                            total_pnl_r         REAL,
                            avg_capture_rate    REAL,
                            avg_holding_eff     REAL,
                            avg_mfe             REAL,
                            avg_mae             REAL,
                            max_drawdown        REAL,
                            config_hash         TEXT
                        )
                    ''')

                    # 9. Market Events (Hypertable)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS market_events (
                            event_id TEXT NOT NULL,
                            timestamp TIMESTAMPTZ NOT NULL,
                            occurrence_timestamp TIMESTAMPTZ NOT NULL,
                            symbol TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            engine_version TEXT NOT NULL,
                            payload JSONB NOT NULL,
                            PRIMARY KEY (event_id, timestamp)
                        )
                    ''')

                    # market_events research_id migration (moved here from above so
                    # it runs only after the table is guaranteed to exist).
                    cursor.execute("ALTER TABLE market_events ADD COLUMN IF NOT EXISTS research_id VARCHAR(24)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_events_research_id ON market_events(research_id) WHERE research_id IS NOT NULL")

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
                    try:
                        cursor.execute("SELECT create_hypertable('market_events', 'timestamp', if_not_exists => TRUE)")
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
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_events_sym ON market_events(symbol)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_events_type ON market_events(event_type)")

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
                    audit = dict(audit)  # copy — don't mutate caller's dict (double-encode on retry)
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
                    signal = dict(signal)  # copy — don't mutate caller's dict (double-encode on retry)
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
            # Validate trade data before save
            from src.core.data_quality import validate_trade_data
            is_valid, errs = validate_trade_data(perf)
            perf['valid'] = is_valid
            perf['validation_errors'] = "; ".join(errs) if errs else None

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    perf_copy = dict(perf)
                    if 'features' in perf_copy:
                        perf_copy['features'] = json.dumps(perf_copy['features'], cls=NumpyEncoder)
                    if 'diagnostics' in perf_copy:
                        perf_copy['diagnostics'] = json.dumps(perf_copy['diagnostics'], cls=NumpyEncoder)
                    columns = list(perf_copy.keys())
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
                        capture_rate = EXCLUDED.capture_rate,
                        holding_efficiency = EXCLUDED.holding_efficiency,
                        valid = EXCLUDED.valid,
                        validation_errors = EXCLUDED.validation_errors,
                        confidence = EXCLUDED.confidence,
                        diagnostics = EXCLUDED.diagnostics,
                        position_size_inr = EXCLUDED.position_size_inr,
                        lots = EXCLUDED.lots,
                        tp1 = EXCLUDED.tp1
                    """
                    cursor.execute(query, perf_copy)
                conn.commit()
        except Exception as e:
            # CRITICAL, not error: a failed real-trade write means P&L accounting
            # and DB state silently diverge from reality, and the position will be
            # "recovered" as still-open on restart. Must be alertable.
            logger.critical(f"🚨 DATA LOSS: failed to save trade_performance {perf.get('trade_id')}: {e}", exc_info=True)

    def save_trade_event(self, event: Dict[str, Any]):
        """Save trade event lifecycle tracking record"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Copy before serializing — mutating the caller's dict in place
                    # double-encodes the payload if the same dict is ever retried.
                    event_copy = dict(event)
                    if 'payload' in event_copy:
                        event_copy['payload'] = json.dumps(event_copy['payload'], cls=NumpyEncoder)
                    columns = list(event_copy.keys())
                    placeholders = [f"%({col})s" for col in columns]

                    query = f"""
                        INSERT INTO trade_events ({','.join(columns)})
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (event_id, timestamp) DO NOTHING
                    """
                    cursor.execute(query, event_copy)
                conn.commit()
        except Exception as e:
            logger.critical(f"🚨 DATA LOSS: failed to save trade_event {event.get('event_id')}: {e}", exc_info=True)

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

    def update_live_heartbeat(
        self, trade_id: str, current_price: float, unrealized_pnl_r: float,
        mfe_r: float, mae_r: float, bars_held: int, stop_loss: float,
        take_profit: float, timestamp,
    ) -> None:
        """Lightweight per-candle refresh for an OPEN real position.

        Separate from save_trade_performance's upsert: this fires every market
        pulse (not just on a SL trail / TP expansion), so the live dashboard has
        a current price and unrealized PnL instead of data that's only as fresh
        as the last stop/target change.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE trade_performance
                        SET current_price = %s,
                            unrealized_pnl_r = %s,
                            mfe_r = %s,
                            mae_r = %s,
                            bars_held = %s,
                            stop_loss = %s,
                            take_profit = %s,
                            last_heartbeat_at = %s
                        WHERE trade_id = %s AND exit_time IS NULL
                        """,
                        (current_price, unrealized_pnl_r, mfe_r, mae_r, bars_held,
                         stop_loss, take_profit, timestamp, trade_id),
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"⚠️ Failed to write live heartbeat for {trade_id}: {e}")

    def upsert_market_state(self, record: Dict[str, Any]) -> None:
        """Overwrite the single current-state row for `record['symbol']`.

        Insert-or-replace, not append — this table answers "what does the
        system currently believe", not "what has it believed historically"
        (pattern/zone history already lives in market_events / signal_audit).
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    rec = dict(record)
                    rec['zones'] = json.dumps(rec.get('zones', []), cls=NumpyEncoder)
                    rec['patterns'] = json.dumps(rec.get('patterns', []), cls=NumpyEncoder)
                    cursor.execute(
                        """
                        INSERT INTO market_state (
                            symbol, updated_at, current_price, daily_bias, market_regime,
                            rvol, atr, move_efficiency, wickiness,
                            narrative_bias, narrative_confidence, zones, patterns
                        ) VALUES (
                            %(symbol)s, %(updated_at)s, %(current_price)s, %(daily_bias)s, %(market_regime)s,
                            %(rvol)s, %(atr)s, %(move_efficiency)s, %(wickiness)s,
                            %(narrative_bias)s, %(narrative_confidence)s, %(zones)s, %(patterns)s
                        )
                        ON CONFLICT (symbol) DO UPDATE SET
                            updated_at = EXCLUDED.updated_at,
                            current_price = EXCLUDED.current_price,
                            daily_bias = EXCLUDED.daily_bias,
                            market_regime = EXCLUDED.market_regime,
                            rvol = EXCLUDED.rvol,
                            atr = EXCLUDED.atr,
                            move_efficiency = EXCLUDED.move_efficiency,
                            wickiness = EXCLUDED.wickiness,
                            narrative_bias = EXCLUDED.narrative_bias,
                            narrative_confidence = EXCLUDED.narrative_confidence,
                            zones = EXCLUDED.zones,
                            patterns = EXCLUDED.patterns
                        """,
                        rec,
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"⚠️ Failed to upsert market_state for {record.get('symbol')}: {e}")

    def save_risk_governor_block(self, record: Dict[str, Any]) -> None:
        """Log a signal that passed strategy filters but was blocked by the
        portfolio risk governor (daily-loss halt / max concurrent / max deployed).
        Insert-only — there's nothing to update later, unlike a live position.
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    columns = list(record.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    cursor.execute(
                        f"""
                        INSERT INTO risk_governor_blocks ({','.join(columns)})
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (block_id, timestamp) DO NOTHING
                        """,
                        record,
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"⚠️ Failed to log risk governor block: {e}")

    def get_realized_r_today(self, date_str: str) -> float:
        """Sum of final_pnl_r for real trades already closed today (IST).

        Used to reconstruct the daily-loss kill switch on startup so a same-day
        restart can't silently undo an already-tripped halt (daily_realized_r
        was previously in-memory only and reset to 0 on every process start).
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT COALESCE(SUM(final_pnl_r), 0)
                        FROM trade_performance
                        WHERE DATE(entry_time AT TIME ZONE 'Asia/Kolkata') = %s
                          AND exit_time IS NOT NULL
                          AND valid = TRUE
                        """,
                        (date_str,),
                    )
                    row = cursor.fetchone()
                    return float(row[0]) if row and row[0] is not None else 0.0
        except Exception as e:
            logger.error(f"❌ Failed to fetch today's realized R: {e}")
            return 0.0

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
            # Validate trade data before save
            from src.core.data_quality import validate_trade_data
            is_valid, errs = validate_trade_data(result)
            result['valid'] = is_valid
            result['validation_errors'] = "; ".join(errs) if errs else None

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    result_copy = dict(result)
                    if 'rejection_reasons' in result_copy:
                        result_copy['rejection_reasons'] = json.dumps(result_copy['rejection_reasons'], cls=NumpyEncoder)
                    if 'diagnostics' in result_copy:
                        result_copy['diagnostics'] = json.dumps(result_copy['diagnostics'], cls=NumpyEncoder)
                    columns = list(result_copy.keys())
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
                        capture_rate = EXCLUDED.capture_rate,
                        holding_efficiency = EXCLUDED.holding_efficiency,
                        valid = EXCLUDED.valid,
                        validation_errors = EXCLUDED.validation_errors,
                        confidence = EXCLUDED.confidence,
                        diagnostics = EXCLUDED.diagnostics
                    """
                    cursor.execute(query, result_copy)
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

    # ── Multi-leg options combos (vertical spreads, straddle/strangle) ──────

    def save_combo_trade(self, combo: Dict[str, Any]):
        """Save or update a real multi-leg combo position."""
        try:
            from src.core.data_quality import validate_combo_data
            is_valid, errs = validate_combo_data(combo)
            combo['valid'] = is_valid
            combo['validation_errors'] = "; ".join(errs) if errs else None

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    c = dict(combo)
                    if 'legs' in c:
                        c['legs'] = json.dumps(c['legs'], cls=NumpyEncoder)
                    if 'diagnostics' in c:
                        c['diagnostics'] = json.dumps(c['diagnostics'], cls=NumpyEncoder)
                    columns = list(c.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    query = f"""
                        INSERT INTO combo_trades ({','.join(columns)})
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (combo_id, entry_time) DO UPDATE SET
                        exit_time = EXCLUDED.exit_time,
                        underlying_exit_price = EXCLUDED.underlying_exit_price,
                        legs = EXCLUDED.legs,
                        current_pnl_r = EXCLUDED.current_pnl_r,
                        final_pnl_r = EXCLUDED.final_pnl_r,
                        exit_reason = EXCLUDED.exit_reason,
                        duration_minutes = EXCLUDED.duration_minutes,
                        diagnostics = EXCLUDED.diagnostics,
                        valid = EXCLUDED.valid,
                        validation_errors = EXCLUDED.validation_errors
                    """
                    cursor.execute(query, c)
                conn.commit()
        except Exception as e:
            logger.critical(f"🚨 DATA LOSS: failed to save combo_trade {combo.get('combo_id')}: {e}", exc_info=True)

    def save_combo_event(self, event: Dict[str, Any]):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    e = dict(event)
                    if 'payload' in e:
                        e['payload'] = json.dumps(e['payload'], cls=NumpyEncoder)
                    columns = list(e.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    cursor.execute(
                        f"""
                        INSERT INTO combo_trade_events ({','.join(columns)})
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (event_id, timestamp) DO NOTHING
                        """,
                        e,
                    )
                conn.commit()
        except Exception as ex:
            logger.critical(f"🚨 DATA LOSS: failed to save combo_trade_event: {ex}", exc_info=True)

    def save_counterfactual_combo_result(self, combo: Dict[str, Any]):
        """Save or update a shadow (rejected) multi-leg combo candidate."""
        try:
            from src.core.data_quality import validate_combo_data
            is_valid, errs = validate_combo_data(combo)
            combo['valid'] = is_valid
            combo['validation_errors'] = "; ".join(errs) if errs else None

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    c = dict(combo)
                    if 'legs' in c:
                        c['legs'] = json.dumps(c['legs'], cls=NumpyEncoder)
                    if 'rejection_reasons' in c:
                        c['rejection_reasons'] = json.dumps(c['rejection_reasons'], cls=NumpyEncoder)
                    if 'diagnostics' in c:
                        c['diagnostics'] = json.dumps(c['diagnostics'], cls=NumpyEncoder)
                    columns = list(c.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    query = f"""
                        INSERT INTO counterfactual_combo_results ({','.join(columns)})
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (combo_id, entry_time) DO UPDATE SET
                        exit_time = EXCLUDED.exit_time,
                        underlying_exit_price = EXCLUDED.underlying_exit_price,
                        legs = EXCLUDED.legs,
                        current_pnl_r = EXCLUDED.current_pnl_r,
                        final_pnl_r = EXCLUDED.final_pnl_r,
                        exit_reason = EXCLUDED.exit_reason,
                        duration_minutes = EXCLUDED.duration_minutes,
                        diagnostics = EXCLUDED.diagnostics,
                        valid = EXCLUDED.valid,
                        validation_errors = EXCLUDED.validation_errors
                    """
                    cursor.execute(query, c)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save counterfactual combo result: {e}")

    def save_counterfactual_combo_event(self, event: Dict[str, Any]):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    e = dict(event)
                    if 'payload' in e:
                        e['payload'] = json.dumps(e['payload'], cls=NumpyEncoder)
                    columns = list(e.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    cursor.execute(
                        f"""
                        INSERT INTO counterfactual_combo_events ({','.join(columns)})
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (event_id, timestamp) DO NOTHING
                        """,
                        e,
                    )
                conn.commit()
        except Exception as ex:
            logger.error(f"❌ Failed to save counterfactual combo event: {ex}")

    def get_open_combo_positions(self) -> List[Dict[str, Any]]:
        """Fetch all currently open real combo positions for restart recovery."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM combo_trades WHERE exit_time IS NULL")
                    return list(cursor.fetchall())
        except Exception as e:
            logger.error(f"❌ Failed to fetch open combo positions: {e}")
            return []

    def get_open_counterfactual_combos(self) -> List[Dict[str, Any]]:
        """Fetch all currently open shadow combo positions for restart recovery."""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SELECT * FROM counterfactual_combo_results WHERE exit_time IS NULL")
                    return list(cursor.fetchall())
        except Exception as e:
            logger.error(f"❌ Failed to fetch open counterfactual combos: {e}")
            return []

    def save_execution_event(self, event: Dict[str, Any]):
        """Save execution trace event record"""
        try:
            event_copy = event.copy()
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    if 'payload' in event_copy:
                        event_copy['payload'] = json.dumps(event_copy['payload'], cls=NumpyEncoder)
                    columns = list(event_copy.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    
                    query = f"""
                        INSERT INTO execution_events ({','.join(columns)}) 
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (event_id, timestamp) DO NOTHING
                    """
                    cursor.execute(query, event_copy)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save execution event: {e}")

    def save_market_event(self, event: Dict[str, Any]):
        """Save a persistent market context research event (hypertable)"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    event_copy = dict(event)
                    if 'payload' in event_copy:
                        event_copy['payload'] = json.dumps(event_copy['payload'], cls=NumpyEncoder)
                    columns = list(event_copy.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    
                    query = f"""
                        INSERT INTO market_events ({','.join(columns)}) 
                        VALUES ({','.join(placeholders)})
                        ON CONFLICT (event_id, timestamp) DO NOTHING
                    """
                    cursor.execute(query, event_copy)
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to save market event: {e}")

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

    def save_experiment(self, exp_dict: Dict[str, Any]) -> None:
        """
        Upsert an experiment record into the experiments metadata table.
        Call this when registering an Experiment with ExperimentRegistry.
        exp_dict should come from Experiment.to_db_dict().
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO experiments
                            (name, strategy_id, version, config_hash, git_commit,
                             params, description, created_at, status, notes, strategy_metadata)
                        VALUES
                            (%(name)s, %(strategy_id)s, %(version)s, %(config_hash)s,
                             %(git_commit)s, %(params)s, %(description)s,
                             %(created_at)s, %(status)s, %(notes)s, %(strategy_metadata)s)
                        ON CONFLICT (name) DO UPDATE SET
                            status      = EXCLUDED.status,
                            notes       = EXCLUDED.notes,
                            git_commit  = EXCLUDED.git_commit,
                            strategy_metadata = EXCLUDED.strategy_metadata
                    """, exp_dict)
                conn.commit()
                logger.info(f"📋 Experiment saved: {exp_dict.get('name')} "
                            f"[hash={exp_dict.get('config_hash')} git={exp_dict.get('git_commit')}]")
        except Exception as e:
            logger.error(f"❌ Failed to save experiment: {e}")

    def save_research_decision(self, decision: Dict[str, Any]):
        """Save a configuration update or hypothesis update event log for audit trails"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    columns = list(decision.keys())
                    placeholders = [f"%({col})s" for col in columns]
                    query = f"""
                        INSERT INTO research_decisions ({','.join(columns)})
                        VALUES ({','.join(placeholders)})
                    """
                    cursor.execute(query, decision)
                conn.commit()
                logger.info(f"💾 Research decision logged for strategy_id={decision.get('strategy_id')}")
        except Exception as e:
            logger.error(f"❌ Failed to save research decision: {e}")

    def save_experiment_daily_metrics(self, date_str: str, experiment_name: str,
                                      config_hash: str = None) -> None:
        """
        Aggregate today's CF and real trade data for an experiment and upsert
        one summary row into experiment_daily_metrics.
        Call once per experiment at session end (15:25 IST).
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # CF aggregates from counterfactual_results
                    cursor.execute("""
                        SELECT
                            COUNT(*)                                         AS cf_trades,
                            SUM(CASE WHEN final_pnl_r > 0 THEN 1 ELSE 0 END) AS wins,
                            SUM(CASE WHEN final_pnl_r <= 0 THEN 1 ELSE 0 END) AS losses,
                            AVG(final_pnl_r)                                  AS expectancy,
                            SUM(final_pnl_r)                                  AS total_pnl_r,
                            AVG(capture_rate)                                 AS avg_capture_rate,
                            AVG(holding_efficiency)                           AS avg_holding_eff,
                            AVG(mfe_r)                                        AS avg_mfe,
                            AVG(mae_r)                                        AS avg_mae,
                            MIN(final_pnl_r)                                  AS max_drawdown
                        FROM counterfactual_results
                        WHERE exit_time IS NOT NULL
                          AND experiment_name = %s
                          AND DATE(exit_time AT TIME ZONE 'Asia/Kolkata') = %s
                          AND valid = TRUE
                    """, (experiment_name, date_str))
                    cf_row = cursor.fetchone()

                    # Real trade aggregates from trade_performance
                    cursor.execute("""
                        SELECT COUNT(*) AS real_trades
                        FROM trade_performance
                        WHERE experiment_name = %s
                          AND DATE(entry_time AT TIME ZONE 'Asia/Kolkata') = %s
                          AND valid = TRUE
                    """, (experiment_name, date_str))
                    real_row = cursor.fetchone()

                    metrics = {
                        'date': date_str,
                        'experiment_name': experiment_name,
                        'real_trades': real_row[0] if real_row else 0,
                        'cf_trades': cf_row[0] if cf_row else 0,
                        'wins': cf_row[1] if cf_row else 0,
                        'losses': cf_row[2] if cf_row else 0,
                        'expectancy': float(cf_row[3]) if cf_row and cf_row[3] is not None else None,
                        'total_pnl_r': float(cf_row[4]) if cf_row and cf_row[4] is not None else None,
                        'avg_capture_rate': float(cf_row[5]) if cf_row and cf_row[5] is not None else None,
                        'avg_holding_eff': float(cf_row[6]) if cf_row and cf_row[6] is not None else None,
                        'avg_mfe': float(cf_row[7]) if cf_row and cf_row[7] is not None else None,
                        'avg_mae': float(cf_row[8]) if cf_row and cf_row[8] is not None else None,
                        'max_drawdown': float(cf_row[9]) if cf_row and cf_row[9] is not None else None,
                        'config_hash': config_hash,
                    }

                    cursor.execute("""
                        INSERT INTO experiment_daily_metrics
                            (date, experiment_name, real_trades, cf_trades, wins, losses,
                             expectancy, total_pnl_r, avg_capture_rate, avg_holding_eff,
                             avg_mfe, avg_mae, max_drawdown, config_hash)
                        VALUES
                            (%(date)s, %(experiment_name)s, %(real_trades)s, %(cf_trades)s,
                             %(wins)s, %(losses)s, %(expectancy)s, %(total_pnl_r)s,
                             %(avg_capture_rate)s, %(avg_holding_eff)s,
                             %(avg_mfe)s, %(avg_mae)s, %(max_drawdown)s, %(config_hash)s)
                        ON CONFLICT (date, experiment_name) DO UPDATE SET
                            real_trades     = EXCLUDED.real_trades,
                            cf_trades       = EXCLUDED.cf_trades,
                            wins            = EXCLUDED.wins,
                            losses          = EXCLUDED.losses,
                            expectancy      = EXCLUDED.expectancy,
                            total_pnl_r     = EXCLUDED.total_pnl_r,
                            avg_capture_rate = EXCLUDED.avg_capture_rate,
                            avg_holding_eff = EXCLUDED.avg_holding_eff,
                            avg_mfe         = EXCLUDED.avg_mfe,
                            avg_mae         = EXCLUDED.avg_mae,
                            max_drawdown    = EXCLUDED.max_drawdown,
                            config_hash     = EXCLUDED.config_hash
                    """, metrics)
                conn.commit()
                logger.info(
                    f"📊 Daily metrics saved: [{experiment_name}] {date_str} "
                    f"cf={metrics['cf_trades']} expectancy={metrics['expectancy']}"
                )
        except Exception as e:
            logger.error(f"❌ Failed to save experiment daily metrics for {experiment_name}: {e}")
