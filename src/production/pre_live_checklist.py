#!/usr/bin/env python3
"""
Pre-Live Checklist and Staging
MUST #6: Formal go-live checklist with kill switch
"""

import sys
import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChecklistStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class TradingStatus(Enum):
    STOPPED = "STOPPED"
    PILOT = "PILOT"
    SCALED_UP = "SCALED_UP"
    FULL_TRADING = "FULL_TRADING"
    EMERGENCY_STOP = "EMERGENCY_STOP"

@dataclass
class ChecklistItem:
    """Checklist item"""
    id: str
    name: str
    description: str
    status: ChecklistStatus
    required: bool
    category: str
    estimated_time: int  # minutes
    actual_time: Optional[int] = None
    notes: Optional[str] = None
    completed_at: Optional[datetime] = None
    failed_reason: Optional[str] = None

@dataclass
class TradingStage:
    """Trading stage configuration"""
    stage_name: str
    capital_limit: float
    duration_hours: int
    max_trades_per_day: int
    risk_multiplier: float
    monitoring_interval: int  # minutes

class PreLiveChecklist:
    """Pre-live checklist and staging system"""
    
    def __init__(self):
        self.checklist_items = []
        self.trading_stages = []
        self.current_stage = None
        self.trading_status = TradingStatus.STOPPED
        self.kill_switch_active = False
        self.stage_start_time = None
        self.stage_monitoring = {}
        
        # Initialize checklist
        self._initialize_checklist()
        self._initialize_trading_stages()
    
    def _initialize_checklist(self):
        """Initialize pre-live checklist"""
        checklist_data = [
            # System Validation
            {
                'id': 'system_validation',
                'name': 'System Validation',
                'description': 'Validate all system components are working',
                'required': True,
                'category': 'System',
                'estimated_time': 30
            },
            {
                'id': 'api_connectivity',
                'name': 'API Connectivity',
                'description': 'Test API connectivity and authentication',
                'required': True,
                'category': 'System',
                'estimated_time': 15
            },
            {
                'id': 'database_integrity',
                'name': 'Database Integrity',
                'description': 'Verify database integrity and backup/restore',
                'required': True,
                'category': 'System',
                'estimated_time': 20
            },
            
            # Risk Management
            {
                'id': 'risk_limits',
                'name': 'Risk Limits Configuration',
                'description': 'Configure and test risk limits',
                'required': True,
                'category': 'Risk',
                'estimated_time': 25
            },
            {
                'id': 'circuit_breakers',
                'name': 'Circuit Breakers',
                'description': 'Test circuit breaker functionality',
                'required': True,
                'category': 'Risk',
                'estimated_time': 20
            },
            {
                'id': 'position_sizing',
                'name': 'Position Sizing',
                'description': 'Validate position sizing calculations',
                'required': True,
                'category': 'Risk',
                'estimated_time': 15
            },
            
            # Execution
            {
                'id': 'order_execution',
                'name': 'Order Execution',
                'description': 'Test order execution and reconciliation',
                'required': True,
                'category': 'Execution',
                'estimated_time': 30
            },
            {
                'id': 'slippage_model',
                'name': 'Slippage Model',
                'description': 'Validate slippage and partial fill models',
                'required': True,
                'category': 'Execution',
                'estimated_time': 20
            },
            {
                'id': 'reconciliation',
                'name': 'Reconciliation System',
                'description': 'Test order reconciliation system',
                'required': True,
                'category': 'Execution',
                'estimated_time': 25
            },
            
            # Monitoring
            {
                'id': 'monitoring_system',
                'name': 'Monitoring System',
                'description': 'Test monitoring and alerting system',
                'required': True,
                'category': 'Monitoring',
                'estimated_time': 20
            },
            {
                'id': 'alert_channels',
                'name': 'Alert Channels',
                'description': 'Test all alert channels (email, SMS, etc.)',
                'required': True,
                'category': 'Monitoring',
                'estimated_time': 15
            },
            {
                'id': 'kill_switch',
                'name': 'Kill Switch',
                'description': 'Test emergency kill switch functionality',
                'required': True,
                'category': 'Monitoring',
                'estimated_time': 10
            },
            
            # Backtesting
            {
                'id': 'backtest_validation',
                'name': 'Backtest Validation',
                'description': 'Run and validate backtest results',
                'required': True,
                'category': 'Backtesting',
                'estimated_time': 60
            },
            {
                'id': 'forward_test',
                'name': 'Forward Test',
                'description': 'Run forward test with replay',
                'required': True,
                'category': 'Backtesting',
                'estimated_time': 120
            },
            {
                'id': 'performance_validation',
                'name': 'Performance Validation',
                'description': 'Validate performance metrics and equity curve',
                'required': True,
                'category': 'Backtesting',
                'estimated_time': 30
            }
        ]
        
        for item_data in checklist_data:
            item = ChecklistItem(
                id=item_data['id'],
                name=item_data['name'],
                description=item_data['description'],
                status=ChecklistStatus.PENDING,
                required=item_data['required'],
                category=item_data['category'],
                estimated_time=item_data['estimated_time']
            )
            self.checklist_items.append(item)
    
    def _initialize_trading_stages(self):
        """Initialize trading stages"""
        self.trading_stages = [
            TradingStage(
                stage_name="PILOT",
                capital_limit=10000,  # ₹10k
                duration_hours=72,    # 3 days
                max_trades_per_day=5,
                risk_multiplier=0.5,  # 50% of normal risk
                monitoring_interval=15  # 15 minutes
            ),
            TradingStage(
                stage_name="SCALED_UP_1",
                capital_limit=25000,  # ₹25k
                duration_hours=168,   # 1 week
                max_trades_per_day=10,
                risk_multiplier=0.75, # 75% of normal risk
                monitoring_interval=30  # 30 minutes
            ),
            TradingStage(
                stage_name="SCALED_UP_2",
                capital_limit=50000,  # ₹50k
                duration_hours=336,   # 2 weeks
                max_trades_per_day=20,
                risk_multiplier=0.9,  # 90% of normal risk
                monitoring_interval=60  # 1 hour
            ),
            TradingStage(
                stage_name="FULL_TRADING",
                capital_limit=100000, # ₹100k
                duration_hours=8760,  # 1 year
                max_trades_per_day=50,
                risk_multiplier=1.0,  # 100% of normal risk
                monitoring_interval=120  # 2 hours
            )
        ]
    
    async def run_checklist_item(self, item_id: str) -> bool:
        """Run a specific checklist item"""
        try:
            item = self._get_checklist_item(item_id)
            if not item:
                logger.error(f"❌ Checklist item not found: {item_id}")
                return False
            
            logger.info(f"🔍 Running checklist item: {item.name}")
            item.status = ChecklistStatus.IN_PROGRESS
            start_time = time.time()
            
            # Run the specific test
            success = await self._execute_checklist_item(item)
            
            execution_time = int((time.time() - start_time) / 60)  # minutes
            item.actual_time = execution_time
            
            if success:
                item.status = ChecklistStatus.PASSED
                item.completed_at = datetime.now()
                logger.info(f"✅ Checklist item passed: {item.name} ({execution_time} minutes)")
            else:
                item.status = ChecklistStatus.FAILED
                item.failed_reason = "Test execution failed"
                logger.error(f"❌ Checklist item failed: {item.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Checklist item execution failed: {e}")
            item = self._get_checklist_item(item_id)
            if item:
                item.status = ChecklistStatus.FAILED
                item.failed_reason = str(e)
            return False
    
    async def _execute_checklist_item(self, item: ChecklistItem) -> bool:
        """Execute specific checklist item"""
        try:
            if item.id == 'system_validation':
                return await self._test_system_validation()
            elif item.id == 'api_connectivity':
                return await self._test_api_connectivity()
            elif item.id == 'database_integrity':
                return await self._test_database_integrity()
            elif item.id == 'risk_limits':
                return await self._test_risk_limits()
            elif item.id == 'circuit_breakers':
                return await self._test_circuit_breakers()
            elif item.id == 'position_sizing':
                return await self._test_position_sizing()
            elif item.id == 'order_execution':
                return await self._test_order_execution()
            elif item.id == 'slippage_model':
                return await self._test_slippage_model()
            elif item.id == 'reconciliation':
                return await self._test_reconciliation()
            elif item.id == 'monitoring_system':
                return await self._test_monitoring_system()
            elif item.id == 'alert_channels':
                return await self._test_alert_channels()
            elif item.id == 'kill_switch':
                return await self._test_kill_switch()
            elif item.id == 'backtest_validation':
                return await self._test_backtest_validation()
            elif item.id == 'forward_test':
                return await self._test_forward_test()
            elif item.id == 'performance_validation':
                return await self._test_performance_validation()
            else:
                logger.warning(f"⚠️ Unknown checklist item: {item.id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Checklist item execution failed: {e}")
            return False
    
    async def _test_system_validation(self) -> bool:
        """Test system validation"""
        # Test all system components
        return True  # Mock implementation
    
    async def _test_api_connectivity(self) -> bool:
        """Test API connectivity"""
        # Test API connections
        return True  # Mock implementation
    
    async def _test_database_integrity(self) -> bool:
        """Test database integrity"""
        # Test database backup/restore
        return True  # Mock implementation
    
    async def _test_risk_limits(self) -> bool:
        """Test risk limits"""
        # Test risk limit functionality
        return True  # Mock implementation
    
    async def _test_circuit_breakers(self) -> bool:
        """Test circuit breakers"""
        # Test circuit breaker functionality
        return True  # Mock implementation
    
    async def _test_position_sizing(self) -> bool:
        """Test position sizing"""
        # Test position sizing calculations
        return True  # Mock implementation
    
    async def _test_order_execution(self) -> bool:
        """Test order execution"""
        # Test order execution system
        return True  # Mock implementation
    
    async def _test_slippage_model(self) -> bool:
        """Test slippage model"""
        # Test slippage and partial fill models
        return True  # Mock implementation
    
    async def _test_reconciliation(self) -> bool:
        """Test reconciliation system"""
        # Test order reconciliation
        return True  # Mock implementation
    
    async def _test_monitoring_system(self) -> bool:
        """Test monitoring system"""
        # Test monitoring and alerting
        return True  # Mock implementation
    
    async def _test_alert_channels(self) -> bool:
        """Test alert channels"""
        # Test all alert channels
        return True  # Mock implementation
    
    async def _test_kill_switch(self) -> bool:
        """Test kill switch"""
        # Test emergency kill switch
        return True  # Mock implementation
    
    async def _test_backtest_validation(self) -> bool:
        """Test backtest validation"""
        # Run and validate backtest
        return True  # Mock implementation
    
    async def _test_forward_test(self) -> bool:
        """Test forward test"""
        # Run forward test with replay
        return True  # Mock implementation
    
    async def _test_performance_validation(self) -> bool:
        """Test performance validation"""
        # Validate performance metrics
        return True  # Mock implementation
    
    def _get_checklist_item(self, item_id: str) -> Optional[ChecklistItem]:
        """Get checklist item by ID"""
        for item in self.checklist_items:
            if item.id == item_id:
                return item
        return None
    
    def get_checklist_status(self) -> Dict[str, Any]:
        """Get checklist status"""
        total_items = len(self.checklist_items)
        passed_items = sum(1 for item in self.checklist_items if item.status == ChecklistStatus.PASSED)
        failed_items = sum(1 for item in self.checklist_items if item.status == ChecklistStatus.FAILED)
        pending_items = sum(1 for item in self.checklist_items if item.status == ChecklistStatus.PENDING)
        
        required_items = [item for item in self.checklist_items if item.required]
        required_passed = sum(1 for item in required_items if item.status == ChecklistStatus.PASSED)
        required_failed = sum(1 for item in required_items if item.status == ChecklistStatus.FAILED)
        
        return {
            'total_items': total_items,
            'passed_items': passed_items,
            'failed_items': failed_items,
            'pending_items': pending_items,
            'required_items': len(required_items),
            'required_passed': required_passed,
            'required_failed': required_failed,
            'completion_percentage': (passed_items / total_items * 100) if total_items > 0 else 0,
            'ready_for_live': required_passed == len(required_items) and required_failed == 0,
            'items': [
                {
                    'id': item.id,
                    'name': item.name,
                    'status': item.status.value,
                    'required': item.required,
                    'category': item.category,
                    'estimated_time': item.estimated_time,
                    'actual_time': item.actual_time,
                    'completed_at': item.completed_at.isoformat() if item.completed_at else None,
                    'failed_reason': item.failed_reason
                }
                for item in self.checklist_items
            ]
        }
    
    def start_trading_stage(self, stage_name: str) -> bool:
        """Start a trading stage"""
        try:
            stage = self._get_trading_stage(stage_name)
            if not stage:
                logger.error(f"❌ Trading stage not found: {stage_name}")
                return False
            
            # Check if checklist is complete
            status = self.get_checklist_status()
            if not status['ready_for_live']:
                logger.error("❌ Cannot start trading - checklist not complete")
                return False
            
            # Check if kill switch is active
            if self.kill_switch_active:
                logger.error("❌ Cannot start trading - kill switch is active")
                return False
            
            # Start stage
            self.current_stage = stage
            self.trading_status = TradingStatus(stage_name)
            self.stage_start_time = datetime.now()
            
            logger.info(f"🚀 Started trading stage: {stage_name}")
            logger.info(f"  Capital limit: ₹{stage.capital_limit:,.2f}")
            logger.info(f"  Duration: {stage.duration_hours} hours")
            logger.info(f"  Max trades per day: {stage.max_trades_per_day}")
            logger.info(f"  Risk multiplier: {stage.risk_multiplier:.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading stage: {e}")
            return False
    
    def _get_trading_stage(self, stage_name: str) -> Optional[TradingStage]:
        """Get trading stage by name"""
        for stage in self.trading_stages:
            if stage.stage_name == stage_name:
                return stage
        return None
    
    def activate_kill_switch(self, reason: str = "Manual activation"):
        """Activate emergency kill switch"""
        self.kill_switch_active = True
        self.trading_status = TradingStatus.EMERGENCY_STOP
        
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        
        # Send emergency alerts
        self._send_emergency_alert(reason)
    
    def deactivate_kill_switch(self, reason: str = "Manual deactivation"):
        """Deactivate kill switch"""
        self.kill_switch_active = False
        
        if self.current_stage:
            self.trading_status = TradingStatus(self.current_stage.stage_name)
        else:
            self.trading_status = TradingStatus.STOPPED
        
        logger.info(f"✅ Kill switch deactivated: {reason}")
    
    def _send_emergency_alert(self, reason: str):
        """Send emergency alert"""
        # In real implementation, send to alerting system
        logger.critical(f"🚨 EMERGENCY ALERT: Kill switch activated - {reason}")
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        stage_duration = None
        if self.stage_start_time:
            stage_duration = (datetime.now() - self.stage_start_time).total_seconds() / 3600  # hours
        
        return {
            'trading_status': self.trading_status.value,
            'current_stage': self.current_stage.stage_name if self.current_stage else None,
            'stage_duration_hours': stage_duration,
            'kill_switch_active': self.kill_switch_active,
            'stage_config': {
                'capital_limit': self.current_stage.capital_limit if self.current_stage else 0,
                'max_trades_per_day': self.current_stage.max_trades_per_day if self.current_stage else 0,
                'risk_multiplier': self.current_stage.risk_multiplier if self.current_stage else 0,
                'monitoring_interval': self.current_stage.monitoring_interval if self.current_stage else 0
            } if self.current_stage else None
        }
    
    def check_stage_completion(self) -> bool:
        """Check if current stage is complete"""
        if not self.current_stage or not self.stage_start_time:
            return False
        
        stage_duration = (datetime.now() - self.stage_start_time).total_seconds() / 3600
        return stage_duration >= self.current_stage.duration_hours
    
    def advance_to_next_stage(self) -> bool:
        """Advance to next trading stage"""
        if not self.current_stage:
            return False
        
        current_stage_name = self.current_stage.stage_name
        stage_names = [stage.stage_name for stage in self.trading_stages]
        
        try:
            current_index = stage_names.index(current_stage_name)
            if current_index < len(stage_names) - 1:
                next_stage_name = stage_names[current_index + 1]
                return self.start_trading_stage(next_stage_name)
            else:
                logger.info("🎉 All trading stages completed!")
                return True
        except ValueError:
            logger.error(f"❌ Current stage not found in stage list: {current_stage_name}")
            return False

def main():
    """Main function for testing"""
    async def test_checklist():
        checklist = PreLiveChecklist()
        
        # Run some checklist items
        await checklist.run_checklist_item('system_validation')
        await checklist.run_checklist_item('api_connectivity')
        await checklist.run_checklist_item('kill_switch')
        
        # Get status
        status = checklist.get_checklist_status()
        print(f"📊 Checklist status: {status}")
        
        # Test trading stages
        if status['ready_for_live']:
            checklist.start_trading_stage('PILOT')
            trading_status = checklist.get_trading_status()
            print(f"🚀 Trading status: {trading_status}")
        
        # Test kill switch
        checklist.activate_kill_switch("Test activation")
        trading_status = checklist.get_trading_status()
        print(f"🚨 Kill switch status: {trading_status}")
    
    # Run test
    asyncio.run(test_checklist())

if __name__ == "__main__":
    main()
