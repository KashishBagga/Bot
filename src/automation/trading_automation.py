#!/usr/bin/env python3
"""
Advanced Trading Automation System
"""

import os
import sys
import time
import logging
import subprocess
import threading
import json
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import psutil

logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    RESTARTING = "restarting"

@dataclass
class TradingProcess:
    name: str
    script_path: str
    args: List[str]
    working_dir: str
    max_restarts: int = 5
    restart_delay: int = 10
    enabled: bool = True

class TradingAutomation:
    """Trading automation system."""
    
    def __init__(self):
        self.processes = {
            'crypto_trader': TradingProcess(
                name='crypto_trader',
                script_path='./run_crypto_trader.sh',
                args=['--market', 'crypto', '--capital', '10000'],
                working_dir='.'
            ),
            'indian_trader': TradingProcess(
                name='indian_trader',
                script_path='./run_indian_trader.sh',
                args=['--capital', '10000'],
                working_dir='.'
            )
        }
        
        self.process_pids = {}
        self.restart_counts = {}
        self.is_running = False
        
    def start_automation(self):
        """Start automation system."""
        logger.info("🚀 Starting automation system")
        self.is_running = True
        
        # Start all processes
        for name, process in self.processes.items():
            if process.enabled:
                self.start_process(name)
    
    def start_process(self, name: str):
        """Start a trading process."""
        try:
            process = self.processes[name]
            cmd = [process.script_path] + process.args
            
            proc = subprocess.Popen(
                cmd,
                cwd=process.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.process_pids[name] = proc.pid
            logger.info(f"✅ Started {name} with PID {proc.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
    
    def stop_process(self, name: str):
        """Stop a trading process."""
        try:
            if name in self.process_pids:
                pid = self.process_pids[name]
                os.kill(pid, signal.SIGTERM)
                del self.process_pids[name]
                logger.info(f"🛑 Stopped {name}")
        except Exception as e:
            logger.error(f"Failed to stop {name}: {e}")
    
    def stop_automation(self):
        """Stop automation system."""
        logger.info("🛑 Stopping automation system")
        self.is_running = False
        
        for name in list(self.process_pids.keys()):
            self.stop_process(name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            'running': self.is_running,
            'processes': {
                name: {
                    'enabled': proc.enabled,
                    'running': name in self.process_pids,
                    'pid': self.process_pids.get(name)
                }
                for name, proc in self.processes.items()
            }
        }

# Global instance
automation_system = TradingAutomation()
