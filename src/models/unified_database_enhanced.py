"""
Enhanced Unified Database - Single Database for All Options Data
Consolidates raw options data, analytics, alerts, and trading signals in one database.
"""

import sqlite3
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

logger = logging.getLogger('unified_database_enhanced')

class UnifiedDatabaseEnhanced:
    """Enhanced unified database with comprehensive options data and analytics."""
    
    def __init__(self, db_path: str = 'unified_trading.db'):
        self.db_path = db_path
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self.init_enhanced_database()
    
    def get_connection(self):
        """Get database connection with proper timeout and isolation."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        conn.execute("PRAGMA cache_size=10000")  # Larger cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
        return conn
    
    def init_enhanced_database(self):
        """Initialize enhanced database schema with all tables."""
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 1. Enhanced Raw Options Chain Table (existing table with improvements)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_options_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    raw_data TEXT NOT NULL,
                    call_oi INTEGER DEFAULT 0,
                    put_oi INTEGER DEFAULT 0,
                    indiavix REAL DEFAULT 0.0,
                    total_options INTEGER DEFAULT 0,
                    total_strikes INTEGER DEFAULT 0,
                    api_response_code INTEGER DEFAULT 0,
                    api_message TEXT,
                    api_status TEXT,
                    data_quality_score REAL DEFAULT 0.0,
                    is_market_open BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_symbol_timestamp ON raw_options_chain(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_options_chain(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_quality ON raw_options_chain(data_quality_score)")
            
            # 2. Individual Options Data Table (parsed from raw data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS options_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_chain_id INTEGER,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    option_symbol TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    strike_price INTEGER NOT NULL,
                    expiry_date TEXT NOT NULL,
                    ltp REAL DEFAULT 0.0,
                    bid REAL DEFAULT 0.0,
                    ask REAL DEFAULT 0.0,
                    volume INTEGER DEFAULT 0,
                    oi INTEGER DEFAULT 0,
                    oi_change INTEGER DEFAULT 0,
                    oi_change_pct REAL DEFAULT 0.0,
                    prev_oi INTEGER DEFAULT 0,
                    implied_volatility REAL DEFAULT 0.0,
                    delta REAL DEFAULT 0.0,
                    gamma REAL DEFAULT 0.0,
                    theta REAL DEFAULT 0.0,
                    vega REAL DEFAULT 0.0,
                    bid_ask_spread REAL DEFAULT 0.0,
                    spread_pct REAL DEFAULT 0.0,
                    is_atm BOOLEAN DEFAULT FALSE,
                    is_itm BOOLEAN DEFAULT FALSE,
                    is_otm BOOLEAN DEFAULT FALSE,
                    underlying_price REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (raw_chain_id) REFERENCES raw_options_chain(id)
                )
            """)
            
            # Create indexes for options_data
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_options_symbol_timestamp ON options_data(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_options_strike_type ON options_data(strike_price, option_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_options_iv ON options_data(implied_volatility)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_options_volume ON options_data(volume)")
            
            # 3. Market Summary Table (aggregated data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    underlying_price REAL NOT NULL,
                    call_oi INTEGER DEFAULT 0,
                    put_oi INTEGER DEFAULT 0,
                    total_oi INTEGER DEFAULT 0,
                    pcr REAL DEFAULT 0.0,
                    indiavix REAL DEFAULT 0.0,
                    total_volume INTEGER DEFAULT 0,
                    total_options INTEGER DEFAULT 0,
                    total_strikes INTEGER DEFAULT 0,
                    atm_strike INTEGER DEFAULT 0,
                    atm_call_ltp REAL DEFAULT 0.0,
                    atm_put_ltp REAL DEFAULT 0.0,
                    atm_call_iv REAL DEFAULT 0.0,
                    atm_put_iv REAL DEFAULT 0.0,
                    is_market_open BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for market_summary
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_symbol_timestamp ON market_summary(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_pcr ON market_summary(pcr)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_vix ON market_summary(indiavix)")
            
            # 4. OHLC Candles Table (minute-level data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlc_candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for OHLC
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_timestamp ON ohlc_candles(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_timeframe ON ohlc_candles(timeframe)")
            
            # 5. Alerts Table (system alerts and notifications)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    symbol TEXT,
                    message TEXT NOT NULL,
                    details TEXT,
                    is_acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for alerts
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(is_acknowledged)")
            
            # 6. Data Quality Log Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quality_score REAL DEFAULT 0.0,
                    issues TEXT,
                    warnings TEXT,
                    errors TEXT,
                    data_freshness_minutes INTEGER DEFAULT 0,
                    api_latency_ms INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for data quality
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_symbol_timestamp ON data_quality_log(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_score ON data_quality_log(quality_score)")
            
            # 7. Greeks Analysis Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS greeks_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    option_symbol TEXT NOT NULL,
                    delta REAL DEFAULT 0.0,
                    gamma REAL DEFAULT 0.0,
                    theta REAL DEFAULT 0.0,
                    vega REAL DEFAULT 0.0,
                    rho REAL DEFAULT 0.0,
                    implied_volatility REAL DEFAULT 0.0,
                    historical_volatility REAL DEFAULT 0.0,
                    iv_percentile REAL DEFAULT 0.0,
                    iv_rank REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for Greeks
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_greeks_symbol_timestamp ON greeks_analysis(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_greeks_iv ON greeks_analysis(implied_volatility)")
            
            # 8. Volatility Surface Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS volatility_surface (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strike_price INTEGER NOT NULL,
                    expiry_date TEXT NOT NULL,
                    option_type TEXT NOT NULL,
                    implied_volatility REAL DEFAULT 0.0,
                    moneyness REAL DEFAULT 0.0,
                    days_to_expiry INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for volatility surface
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vol_surface_symbol_timestamp ON volatility_surface(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vol_surface_strike_expiry ON volatility_surface(strike_price, expiry_date)")
            
            # 9. Strategy Signals Table (enhanced)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    signal_strength REAL DEFAULT 0.0,
                    entry_price REAL DEFAULT 0.0,
                    stop_loss REAL DEFAULT 0.0,
                    take_profit REAL DEFAULT 0.0,
                    option_symbol TEXT,
                    strike_price INTEGER,
                    option_type TEXT,
                    implied_volatility REAL DEFAULT 0.0,
                    confidence_score REAL DEFAULT 0.0,
                    market_conditions TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for strategy signals
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp ON strategy_signals(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_strategy ON strategy_signals(strategy_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON strategy_signals(signal_type)")
            
            # 10. Performance Metrics Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    losing_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    total_pnl REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    sharpe_ratio REAL DEFAULT 0.0,
                    avg_trade_pnl REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_symbol_strategy ON performance_metrics(symbol, strategy_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)")
            
            conn.commit()
            conn.close()
            logger.info("✅ Enhanced unified database schema created successfully")
    
    def save_raw_options_chain(self, raw_data: Dict) -> bool:
        """Save raw options chain data with enhanced processing."""
        with self._lock:
            try:
                timestamp = datetime.now(timezone.utc).isoformat()
                symbol = raw_data['_metadata']['symbol']
                
                # Extract key metrics
                data = raw_data.get('data', {})
                options_chain = data.get('optionsChain', [])
                call_oi = data.get('callOi', 0)
                put_oi = data.get('putOi', 0)
                indiavix = data.get('indiavixData', {}).get('ltp', 0)
                
                total_options = len(options_chain)
                real_strikes = set()
                for option in options_chain:
                    if option.get('option_type') in ['CE', 'PE']:
                        strike = option.get('strike_price', -1)
                        if strike > 0:
                            real_strikes.add(strike)
                total_strikes = len(real_strikes)
                
                # Calculate data quality score
                quality_score = self._calculate_data_quality_score(raw_data)
                
                # Check if market is open
                is_market_open = self._is_market_open()
                
                conn = self.get_connection()
                cursor = conn.cursor()
                
                # Insert raw data
                cursor.execute("""
                    INSERT INTO raw_options_chain (
                        timestamp, symbol, raw_data, call_oi, put_oi, indiavix,
                        total_options, total_strikes, api_response_code, api_message, 
                        api_status, data_quality_score, is_market_open, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, symbol, json.dumps(raw_data), call_oi, put_oi, indiavix,
                    total_options, total_strikes, 
                    raw_data['_metadata'].get('api_response_code', 0),
                    raw_data['_metadata'].get('api_message', ''),
                    raw_data['_metadata'].get('api_status', ''),
                    quality_score, is_market_open, timestamp
                ))
                
                raw_chain_id = cursor.lastrowid
                
                # Parse and save individual options data
                self._save_individual_options(raw_chain_id, timestamp, symbol, options_chain)
                
                # Save market summary
                self._save_market_summary(timestamp, symbol, data, raw_chain_id)
                
                # Save data quality log
                self._save_data_quality_log(timestamp, symbol, quality_score, raw_data)
                
                # Save volatility surface data
                self._save_volatility_surface(timestamp, symbol, options_chain)
                
                # Check for alerts
                self._check_and_create_alerts(timestamp, symbol, raw_data, quality_score)
                
                conn.commit()
                conn.close()
                
                logger.info(f"✅ Enhanced data saved for {symbol}: {total_options} options, {total_strikes} strikes, Quality: {quality_score:.2f}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error saving enhanced options data: {e}")
                return False
    
    def _calculate_data_quality_score(self, raw_data: Dict) -> float:
        """Calculate data quality score (0-100)."""
        try:
            score = 100.0
            data = raw_data.get('data', {})
            options_chain = data.get('optionsChain', [])
            
            # Check for basic structure
            if not options_chain:
                score -= 30
            
            # Check for active options
            active_options = 0
            for option in options_chain:
                if option.get('option_type') in ['CE', 'PE']:
                    if option.get('bid', 0) > 0 and option.get('ask', 0) > 0:
                        active_options += 1
            
            if active_options == 0:
                score -= 25
            elif active_options < len(options_chain) * 0.5:
                score -= 10
            
            # Check for required fields
            required_fields = ['callOi', 'putOi', 'indiavixData']
            for field in required_fields:
                if not data.get(field):
                    score -= 5
            
            # Check API response
            if raw_data['_metadata'].get('api_response_code') != 200:
                score -= 20
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            from options_data_accumulator import MarketHoursChecker
            checker = MarketHoursChecker()
            return checker.is_market_open()
        except:
            return False
    
    def _save_individual_options(self, raw_chain_id: int, timestamp: str, symbol: str, options_chain: List[Dict]):
        """Save individual options data."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for option in options_chain:
                if option.get('option_type') in ['CE', 'PE']:
                    # Calculate Greeks (simplified)
                    ltp = option.get('ltp', 0)
                    bid = option.get('bid', 0)
                    ask = option.get('ask', 0)
                    strike = option.get('strike_price', 0)
                    underlying_price = options_chain[0].get('ltp', 0) if options_chain else 0
                    
                    # Determine option moneyness
                    is_atm = abs(strike - underlying_price) <= 50
                    is_itm = (option.get('option_type') == 'CE' and strike < underlying_price) or \
                            (option.get('option_type') == 'PE' and strike > underlying_price)
                    is_otm = not is_itm and not is_atm
                    
                    cursor.execute("""
                        INSERT INTO options_data (
                            raw_chain_id, timestamp, symbol, option_symbol, option_type,
                            strike_price, expiry_date, ltp, bid, ask, volume, oi, oi_change,
                            oi_change_pct, prev_oi, implied_volatility, delta, gamma, theta, vega,
                            bid_ask_spread, spread_pct, is_atm, is_itm, is_otm, underlying_price, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        raw_chain_id, timestamp, symbol, option.get('symbol', ''), option.get('option_type', ''),
                        strike, option.get('expiry_date', ''), ltp, bid, ask, option.get('volume', 0),
                        option.get('oi', 0), option.get('oich', 0), option.get('oichp', 0),
                        option.get('prev_oi', 0), option.get('implied_volatility', 0), 0, 0, 0, 0,
                        ask - bid if ask > 0 and bid > 0 else 0,
                        ((ask - bid) / ltp * 100) if ltp > 0 else 0,
                        is_atm, is_itm, is_otm, underlying_price, timestamp
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving individual options: {e}")
    
    def _save_market_summary(self, timestamp: str, symbol: str, data: Dict, raw_chain_id: int):
        """Save market summary data."""
        try:
            call_oi = data.get('callOi', 0)
            put_oi = data.get('putOi', 0)
            total_oi = call_oi + put_oi
            pcr = put_oi / call_oi if call_oi > 0 else 0
            indiavix = data.get('indiavixData', {}).get('ltp', 0)
            underlying_price = data.get('optionsChain', [{}])[0].get('ltp', 0)
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO market_summary (
                    timestamp, symbol, underlying_price, call_oi, put_oi, total_oi, pcr,
                    indiavix, total_volume, total_options, total_strikes, is_market_open, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, symbol, underlying_price, call_oi, put_oi, total_oi, pcr,
                indiavix, 0, len(data.get('optionsChain', [])), 0,
                self._is_market_open(), timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving market summary: {e}")
    
    def _save_volatility_surface(self, timestamp: str, symbol: str, options_chain: List[Dict]):
        """Save volatility surface data."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            for option in options_chain:
                if option.get('option_type') in ['CE', 'PE']:
                    strike = option.get('strike_price', 0)
                    expiry = option.get('expiry_date', '')
                    option_type = option.get('option_type', '')
                    iv = option.get('implied_volatility', 0)
                    
                    # Calculate moneyness and days to expiry (simplified)
                    underlying_price = options_chain[0].get('ltp', 0) if options_chain else 0
                    moneyness = (strike - underlying_price) / underlying_price if underlying_price > 0 else 0
                    days_to_expiry = 30  # Simplified calculation
                    
                    cursor.execute("""
                        INSERT INTO volatility_surface (
                            timestamp, symbol, strike_price, expiry_date, option_type,
                            implied_volatility, moneyness, days_to_expiry, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp, symbol, strike, expiry, option_type,
                        iv, moneyness, days_to_expiry, timestamp
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving volatility surface: {e}")
    
    def _save_data_quality_log(self, timestamp: str, symbol: str, quality_score: float, raw_data: Dict):
        """Save data quality log."""
        try:
            issues = []
            warnings = []
            errors = []
            
            # Check for common issues
            data = raw_data.get('data', {})
            if not data.get('optionsChain'):
                issues.append("No options chain data")
            
            if raw_data['_metadata'].get('api_response_code') != 200:
                errors.append(f"API error: {raw_data['_metadata'].get('api_message', 'Unknown')}")
            
            if quality_score < 50:
                warnings.append("Low data quality score")
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO data_quality_log (
                    timestamp, symbol, quality_score, issues, warnings, errors, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, symbol, quality_score,
                json.dumps(issues), json.dumps(warnings), json.dumps(errors), timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving data quality log: {e}")
    
    def _check_and_create_alerts(self, timestamp: str, symbol: str, raw_data: Dict, quality_score: float):
        """Check for conditions and create alerts."""
        try:
            alerts = []
            
            # Check data quality
            if quality_score < 50:
                alerts.append({
                    'type': 'DATA_QUALITY',
                    'severity': 'HIGH',
                    'message': f'Low data quality score: {quality_score:.2f}',
                    'details': f'Symbol: {symbol}, Quality: {quality_score:.2f}'
                })
            
            # Check API errors
            if raw_data['_metadata'].get('api_response_code') != 200:
                alerts.append({
                    'type': 'API_ERROR',
                    'severity': 'CRITICAL',
                    'message': f'API error for {symbol}',
                    'details': f'Code: {raw_data["_metadata"].get("api_response_code")}, Message: {raw_data["_metadata"].get("api_message")}'
                })
            
            # Check for missing data
            data = raw_data.get('data', {})
            if not data.get('optionsChain'):
                alerts.append({
                    'type': 'MISSING_DATA',
                    'severity': 'HIGH',
                    'message': f'No options chain data for {symbol}',
                    'details': 'Options chain is empty or missing'
                })
            
            # Save alerts
            if alerts:
                conn = self.get_connection()
                cursor = conn.cursor()
                
                for alert in alerts:
                    cursor.execute("""
                        INSERT INTO alerts (
                            timestamp, alert_type, severity, symbol, message, details, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp, alert['type'], alert['severity'], symbol,
                        alert['message'], alert['details'], timestamp
                    ))
                
                conn.commit()
                conn.close()
                
                logger.warning(f"⚠️ Created {len(alerts)} alerts for {symbol}")
            
        except Exception as e:
            logger.error(f"Error creating alerts: {e}")
    
    def get_database_summary(self) -> Dict:
        """Get comprehensive database summary."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            summary = {}
            
            # Table statistics
            tables = ['raw_options_chain', 'options_data', 'market_summary', 'ohlc_candles',
                     'alerts', 'data_quality_log', 'greeks_analysis', 'volatility_surface',
                     'strategy_signals', 'performance_metrics']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                summary[f'{table}_count'] = count
            
            # Symbol coverage
            cursor.execute("SELECT DISTINCT symbol FROM raw_options_chain")
            symbols = [row[0] for row in cursor.fetchall()]
            summary['symbols'] = symbols
            
            # Date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM raw_options_chain")
            date_range = cursor.fetchone()
            summary['date_range'] = {
                'start': date_range[0] if date_range[0] else None,
                'end': date_range[1] if date_range[1] else None
            }
            
            # Quality metrics
            cursor.execute("SELECT AVG(data_quality_score) FROM raw_options_chain")
            avg_quality = cursor.fetchone()[0]
            summary['avg_quality_score'] = avg_quality or 0
            
            # Market hours data
            cursor.execute("SELECT COUNT(*) FROM raw_options_chain WHERE is_market_open = 1")
            market_open_count = cursor.fetchone()[0]
            summary['market_open_records'] = market_open_count
            
            # Alert statistics
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE is_acknowledged = 0")
            unacknowledged_alerts = cursor.fetchone()[0]
            summary['unacknowledged_alerts'] = unacknowledged_alerts
            
            conn.close()
            return summary
            
        except Exception as e:
            logger.error(f"Error getting database summary: {e}")
            return {}
    
    def get_analytics_data(self, symbol: str, start_date: str = None, end_date: str = None) -> Dict:
        """Get analytics data for a symbol."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Build date filter
            date_filter = ""
            params = [symbol]
            if start_date and end_date:
                date_filter = "AND timestamp BETWEEN ? AND ?"
                params.extend([start_date, end_date])
            
            # Get market summary data
            cursor.execute(f"""
                SELECT timestamp, underlying_price, call_oi, put_oi, pcr, indiavix
                FROM market_summary 
                WHERE symbol = ? {date_filter}
                ORDER BY timestamp DESC
                LIMIT 100
            """, params)
            
            market_data = cursor.fetchall()
            
            # Get options data
            cursor.execute(f"""
                SELECT option_type, strike_price, ltp, volume, oi, implied_volatility
                FROM options_data 
                WHERE symbol = ? {date_filter}
                ORDER BY timestamp DESC
                LIMIT 1000
            """, params)
            
            options_data = cursor.fetchall()
            
            # Get alerts
            cursor.execute(f"""
                SELECT alert_type, severity, message, created_at
                FROM alerts 
                WHERE symbol = ? {date_filter}
                ORDER BY timestamp DESC
                LIMIT 50
            """, params)
            
            alerts = cursor.fetchall()
            
            conn.close()
            
            return {
                'market_data': market_data,
                'options_data': options_data,
                'alerts': alerts,
                'symbol': symbol,
                'date_range': {'start': start_date, 'end': end_date}
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {} 