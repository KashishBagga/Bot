"""
Connection pooling system for database and API connections.
"""

import sqlite3
import threading
import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from queue import Queue, Empty
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    """Thread-safe SQLite connection pool."""
    
    def __init__(self, database_path: str, max_connections: int = 10, 
                 timeout: float = 30.0):
        self.database_path = database_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool = Queue(maxsize=max_connections)
        self._lock = threading.RLock()
        self._active_connections = 0
        self._created_connections = 0
        
        # Initialize pool with some connections
        for _ in range(min(3, max_connections)):
            self._create_connection()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        try:
            conn = sqlite3.connect(
                self.database_path,
                timeout=self.timeout,
                check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.row_factory = sqlite3.Row
            
            self._created_connections += 1
            logger.debug(f"Created database connection #{self._created_connections}")
            return conn
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        conn = None
        try:
            # Try to get existing connection
            try:
                conn = self._pool.get(timeout=5.0)
            except Empty:
                # Create new connection if pool is empty and under limit
                with self._lock:
                    if self._active_connections < self.max_connections:
                        conn = self._create_connection()
                        self._active_connections += 1
                    else:
                        raise Exception("Connection pool exhausted")
            
            yield conn
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
            
        finally:
            if conn:
                try:
                    conn.commit()
                    self._pool.put(conn)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()
                    except:
                        pass
                    with self._lock:
                        self._active_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except:
                    pass
            self._active_connections = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'max_connections': self.max_connections,
            'active_connections': self._active_connections,
            'available_connections': self._pool.qsize(),
            'created_connections': self._created_connections
        }

class APIConnectionPool:
    """HTTP connection pool for API requests."""
    
    def __init__(self, max_connections: int = 10, max_retries: int = 3):
        self.max_connections = max_connections
        self.max_retries = max_retries
        self._sessions = {}
        self._lock = threading.RLock()
    
    def get_session(self, base_url: str) -> requests.Session:
        """Get or create a session for the given base URL."""
        with self._lock:
            if base_url not in self._sessions:
                session = requests.Session()
                
                # Configure retry strategy
                retry_strategy = Retry(
                    total=self.max_retries,
                    backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
                )
                
                adapter = HTTPAdapter(
                    max_retries=retry_strategy,
                    pool_connections=self.max_connections,
                    pool_maxsize=self.max_connections
                )
                
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                # Set default headers
                session.headers.update({
                    'User-Agent': 'TradingBot/1.0',
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                })
                
                self._sessions[base_url] = session
                logger.debug(f"Created HTTP session for {base_url}")
            
            return self._sessions[base_url]
    
    def close_all(self):
        """Close all sessions."""
        with self._lock:
            for session in self._sessions.values():
                try:
                    session.close()
                except:
                    pass
            self._sessions.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'active_sessions': len(self._sessions),
            'max_connections': self.max_connections,
            'max_retries': self.max_retries
        }

# Global connection pools
db_pool = None
api_pool = None

def initialize_connection_pools(database_path: str = "trading.db"):
    """Initialize global connection pools."""
    global db_pool, api_pool
    
    db_pool = DatabaseConnectionPool(database_path)
    api_pool = APIConnectionPool()
    
    logger.info("✅ Connection pools initialized")

def get_db_connection():
    """Get a database connection from the global pool."""
    if db_pool is None:
        raise Exception("Database connection pool not initialized")
    return db_pool.get_connection()

def get_api_session(base_url: str):
    """Get an API session from the global pool."""
    if api_pool is None:
        raise Exception("API connection pool not initialized")
    return api_pool.get_session(base_url)

def close_connection_pools():
    """Close all connection pools."""
    global db_pool, api_pool
    
    if db_pool:
        db_pool.close_all()
        db_pool = None
    
    if api_pool:
        api_pool.close_all()
        api_pool = None
    
    logger.info("✅ Connection pools closed")
