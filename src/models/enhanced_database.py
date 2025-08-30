"""
Enhanced Database Schema for Options Data Analysis
Comprehensive database with improved tables, indexes, and analytics capabilities.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

logger = logging.getLogger('enhanced_database')

class EnhancedOptionsDatabase:
    """Enhanced database for comprehensive options data analysis."""
    
    def __init__(self, db_path: str = 'enhanced_options.db'):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize enhanced database schema."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. Enhanced Raw Options Chain Table
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
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON raw_options_chain(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON raw_options_chain(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON raw_options_chain(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_open ON raw_options_chain(is_market_open)")
        
        # 2. Individual Options Data Table
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_option_symbol_timestamp ON options_data(option_symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strike_type ON options_data(strike_price, option_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_expiry ON options_data(expiry_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_volume ON options_data(volume)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_oi ON options_data(oi)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_iv ON options_data(implied_volatility)")
        
        # 3. Market Summary Table
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_timestamp ON market_summary(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pcr ON market_summary(pcr)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_vix ON market_summary(indiavix)")
        
        # 4. Data Quality Log Table
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
        
        # Create indexes for data_quality_log
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_symbol_timestamp ON data_quality_log(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_score ON data_quality_log(quality_score)")
        
        # 5. Alerts Table
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
        
        conn.commit()
        conn.close()
        logger.info("✅ Enhanced database schema created successfully")
    
    def save_raw_options_chain(self, raw_data: Dict) -> bool:
        """Save raw options chain data with enhanced processing."""
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
    
    def get_database_summary(self) -> Dict:
        """Get comprehensive database summary."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            summary = {}
            
            # Table statistics
            tables = ['raw_options_chain', 'options_data', 'market_summary', 'data_quality_log', 'alerts']
            
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
            
            conn.close()
            return summary
            
        except Exception as e:
            logger.error(f"Error getting database summary: {e}")
            return {} 