#!/usr/bin/env python3
"""
Live Options Trading System
Bridges the gap between backtesting and real trading with comprehensive risk management
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.realtime_data_manager import RealTimeDataManager, create_data_provider
from src.execution.broker_execution import BrokerExecution, create_broker_api, OrderSide, OrderType
from src.risk.risk_manager import RiskManager, RiskConfig, RiskLevel
from src.core.option_signal_mapper import OptionSignalMapper
from src.models.option_contract import StrikeSelection
from src.strategies.ema_crossover_enhanced import EmaCrossoverEnhanced
from src.strategies.supertrend_ema import SupertrendEma
from src.strategies.supertrend_macd_rsi_ema import SupertrendMacdRsiEma
from src.models.unified_database import UnifiedDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_options_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LiveOptionsTradingSystem:
    """Comprehensive live options trading system."""
    
    def __init__(
        self,
        # Data configuration
        data_provider_type: str = 'historical',
        data_provider_config: Dict = None,
        
        # Broker configuration
        broker_type: str = 'paper',
        broker_config: Dict = None,
        
        # Risk configuration
        risk_config: RiskConfig = None,
        
        # Trading configuration
        symbols: List[str] = None,
        strategies: List[str] = None,
        confidence_cutoff: float = 40.0,
        max_risk_per_trade: float = 0.02,
        expiry_type: str = "weekly",
        strike_selection: StrikeSelection = StrikeSelection.ATM,
        delta_target: float = 0.30,
        
        # System configuration
        trading_mode: str = 'paper',  # 'paper' or 'live'
        auto_execution: bool = False,
        slippage_model: bool = True
    ):
        """Initialize live trading system."""
        
        # Default configurations
        if data_provider_config is None:
            data_provider_config = {}
        if broker_config is None:
            broker_config = {}
        if risk_config is None:
            risk_config = RiskConfig()
        if symbols is None:
            symbols = ['NSE:NIFTY50-INDEX']
        if strategies is None:
            strategies = ['ema_crossover_enhanced']
        
        # Store configuration
        self.data_provider_type = data_provider_type
        self.data_provider_config = data_provider_config
        self.broker_type = broker_type
        self.broker_config = broker_config
        self.risk_config = risk_config
        self.symbols = symbols
        self.strategies = strategies
        self.confidence_cutoff = confidence_cutoff
        self.max_risk_per_trade = max_risk_per_trade
        self.expiry_type = expiry_type
        self.strike_selection = strike_selection
        self.delta_target = delta_target
        self.trading_mode = trading_mode
        self.auto_execution = auto_execution
        self.slippage_model = slippage_model
        
        # Initialize components
        self._initialize_components()
        
        # Trading state
        self.running = False
        self.trading_thread = None
        self.signal_queue = []
        self.execution_queue = []
        self.trade_history = []
        self.performance_metrics = {}
        
        logger.info("🚀 Live Options Trading System initialized")
        logger.info(f"📊 Mode: {trading_mode.upper()}")
        logger.info(f"🔄 Auto Execution: {auto_execution}")
        logger.info(f"📈 Symbols: {symbols}")
        logger.info(f"🎯 Strategies: {strategies}")
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # 1. Initialize data provider
            logger.info("📡 Initializing data provider...")
            data_provider = create_data_provider(self.data_provider_type, **self.data_provider_config)
            self.data_manager = RealTimeDataManager(data_provider)
            
            # 2. Initialize broker execution
            logger.info("🎯 Initializing broker execution...")
            broker_api = create_broker_api(self.broker_type, **self.broker_config)
            self.broker_execution = BrokerExecution(broker_api)
            
            # 3. Initialize risk manager
            logger.info("🛡️ Initializing risk manager...")
            self.risk_manager = RiskManager(self.risk_config)
            
            # 4. Initialize signal mapper
            logger.info("🎯 Initializing signal mapper...")
            self.signal_mapper = OptionSignalMapper(self.data_manager)
            self.signal_mapper.set_parameters(
                expiry_type=self.expiry_type,
                strike_selection=self.strike_selection,
                delta_target=self.delta_target
            )
            
            # 5. Initialize strategies
            logger.info("📊 Initializing strategies...")
            self.strategy_instances = {}
            strategy_map = {
                'ema_crossover_enhanced': EmaCrossoverEnhanced(),
                'supertrend_ema': SupertrendEma(),
                'supertrend_macd_rsi_ema': SupertrendMacdRsiEma()
            }
            
            for strategy_name in self.strategies:
                if strategy_name in strategy_map:
                    self.strategy_instances[strategy_name] = strategy_map[strategy_name]
                else:
                    logger.warning(f"⚠️ Strategy {strategy_name} not found")
            
            # 6. Initialize database
            logger.info("💾 Initializing database...")
            self.db = UnifiedDatabase()
            
            # 7. Add risk callbacks
            self.risk_manager.add_risk_callback(self._on_risk_alert)
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            raise
    
    def start(self):
        """Start the live trading system."""
        if self.running:
            logger.warning("⚠️ System is already running")
            return
        
        try:
            # Start data manager
            self.data_manager.start()
            
            # Start risk monitoring
            self.risk_manager.start_monitoring()
            
            # Start trading thread
            self.running = True
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            
            logger.info("🚀 Live trading system started")
            logger.info(f"📊 Trading mode: {self.trading_mode.upper()}")
            logger.info(f"🔄 Auto execution: {self.auto_execution}")
            
        except Exception as e:
            logger.error(f"❌ Error starting system: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the live trading system."""
        if not self.running:
            return
        
        logger.info("🛑 Stopping live trading system...")
        
        # Stop trading thread
        self.running = False
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
        
        # Stop risk monitoring
        self.risk_manager.stop_monitoring()
        
        # Stop data manager
        self.data_manager.stop()
        
        logger.info("✅ Live trading system stopped")
    
    def _trading_loop(self):
        """Main trading loop."""
        logger.info("🔄 Starting trading loop...")
        
        while self.running:
            try:
                # Check if trading is allowed
                if not self._is_trading_allowed():
                    time.sleep(60)  # Wait 1 minute
                    continue
                
                # Generate signals for each symbol
                for symbol in self.symbols:
                    if not self.running:
                        break
                    
                    self._process_symbol(symbol)
                
                # Process execution queue
                self._process_execution_queue()
                
                # Sleep between iterations
                time.sleep(30)  # 30 seconds between iterations
                
            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                time.sleep(60)
    
    def _process_symbol(self, symbol: str):
        """Process a single symbol for signals."""
        try:
            # Get current underlying price
            current_price = self.data_manager.get_underlying_price(symbol)
            if current_price is None:
                logger.warning(f"⚠️ No price data for {symbol}")
                return
            
            # Get option chain
            option_chain = self.data_manager.get_option_chain(symbol)
            if option_chain is None or not option_chain.contracts:
                logger.warning(f"⚠️ No option chain for {symbol}")
                return
            
            # Generate signals from each strategy
            for strategy_name, strategy in self.strategy_instances.items():
                if not self.running:
                    break
                
                # Get historical data for strategy
                df = self._get_strategy_data(symbol, strategy_name)
                if df is None or df.empty:
                    continue
                
                # Generate signals
                signals = self._generate_strategy_signals(df, strategy, strategy_name, symbol, current_price)
                
                # Process signals
                for signal in signals:
                    if not self.running:
                        break
                    
                    self._process_signal(signal, option_chain)
                
        except Exception as e:
            logger.error(f"❌ Error processing symbol {symbol}: {e}")
    
    def _get_strategy_data(self, symbol: str, strategy_name: str) -> Optional[pd.DataFrame]:
        """Get historical data for strategy analysis."""
        try:
            # This would need to be implemented based on your data structure
            # For now, return None to indicate no data
            return None
        except Exception as e:
            logger.error(f"❌ Error getting strategy data: {e}")
            return None
    
    def _generate_strategy_signals(self, df: pd.DataFrame, strategy, strategy_name: str, 
                                 symbol: str, current_price: float) -> List[Dict]:
        """Generate signals from a strategy."""
        signals = []
        
        try:
            # Generate signals using strategy
            if hasattr(strategy, 'analyze_vectorized'):
                signals_df = strategy.analyze_vectorized(df)
            else:
                logger.warning(f"⚠️ Strategy {strategy_name} doesn't support vectorized analysis")
                return signals
            
            if signals_df.empty:
                return signals
            
            # Convert to signal format
            for idx, row in signals_df.iterrows():
                if row['signal'] != 'NO TRADE':
                    signal = {
                        'timestamp': df.loc[idx, 'timestamp'],
                        'strategy': strategy_name,
                        'signal': row['signal'],
                        'price': float(row['price']),
                        'confidence': float(row.get('confidence_score', 50)),
                        'reasoning': str(row.get('reasoning', ''))[:200],
                        'symbol': symbol,
                        'current_price': current_price
                    }
                    signals.append(signal)
            
        except Exception as e:
            logger.error(f"❌ Error generating signals for {strategy_name}: {e}")
        
        return signals
    
    def _process_signal(self, signal: Dict, option_chain):
        """Process a trading signal."""
        try:
            # Check confidence threshold
            if signal['confidence'] < self.confidence_cutoff:
                return
            
            # Map signal to options
            option_signals = self.signal_mapper.map_multiple_signals(
                [signal], signal['current_price'], signal['timestamp'], option_chain
            )
            
            # Process each option signal
            for option_signal in option_signals:
                if not self.running:
                    break
                
                self._process_option_signal(option_signal)
                
        except Exception as e:
            logger.error(f"❌ Error processing signal: {e}")
    
    def _process_option_signal(self, option_signal: Dict):
        """Process an option signal."""
        try:
            contract = option_signal.get('contract')
            if not contract:
                return
            
            # Check risk management
            can_place, reason = self.risk_manager.check_can_place_order(
                contract, option_signal['quantity'], OrderSide.BUY, option_signal['entry_price']
            )
            
            if not can_place:
                logger.info(f"⚠️ Order blocked by risk management: {reason}")
                return
            
            # Check margin
            if not self.broker_execution.check_margin_before_order(
                contract, option_signal['quantity'], OrderSide.BUY, option_signal['entry_price']
            ):
                logger.warning("⚠️ Insufficient margin for order")
                return
            
            # Add to execution queue
            execution_request = {
                'contract': contract,
                'quantity': option_signal['quantity'],
                'side': OrderSide.BUY,
                'order_type': OrderType.MARKET,
                'price': option_signal['entry_price'],
                'signal': option_signal,
                'timestamp': datetime.now()
            }
            
            self.execution_queue.append(execution_request)
            logger.info(f"📊 Signal queued for execution: {contract.symbol}")
            
        except Exception as e:
            logger.error(f"❌ Error processing option signal: {e}")
    
    def _process_execution_queue(self):
        """Process the execution queue."""
        if not self.auto_execution:
            return
        
        while self.execution_queue and self.running:
            try:
                request = self.execution_queue.pop(0)
                
                # Place order
                response = self.broker_execution.place_option_order(
                    contract=request['contract'],
                    quantity=request['quantity'],
                    side=request['side'],
                    order_type=request['order_type'],
                    price=request['price']
                )
                
                if response.status.value in ['COMPLETE', 'PARTIALLY_FILLED']:
                    # Record trade
                    self.risk_manager.record_trade(
                        contract=request['contract'],
                        quantity=response.filled_quantity,
                        side=request['side'],
                        price=response.average_price
                    )
                    
                    # Store trade history
                    trade_record = {
                        'order_id': response.order_id,
                        'contract': request['contract'],
                        'quantity': response.filled_quantity,
                        'side': request['side'],
                        'price': response.average_price,
                        'signal': request['signal'],
                        'timestamp': response.timestamp,
                        'status': response.status.value
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"✅ Order executed: {response.order_id}")
                    
                else:
                    logger.warning(f"⚠️ Order failed: {response.message}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing execution request: {e}")
    
    def _is_trading_allowed(self) -> bool:
        """Check if trading is allowed."""
        # Check if system is running
        if not self.running:
            return False
        
        # Check if data is connected
        if not self.data_manager.is_connected():
            return False
        
        # Check if broker is connected
        if not self.broker_execution.is_connected():
            return False
        
        # Check risk manager
        metrics = self.risk_manager.get_risk_metrics()
        if metrics.risk_level == RiskLevel.CRITICAL:
            return False
        
        return True
    
    def _on_risk_alert(self, alert: Dict):
        """Handle risk alerts."""
        logger.warning(f"🚨 Risk Alert: {alert['message']} (Level: {alert['level'].value})")
        
        # Take action based on risk level
        if alert['level'] == RiskLevel.CRITICAL:
            logger.error("🚨 CRITICAL RISK - Stopping trading")
            self.stop()
        elif alert['level'] == RiskLevel.HIGH:
            logger.warning("⚠️ HIGH RISK - Pausing new orders")
            # Could implement pausing logic here
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report."""
        try:
            # Get risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            risk_report = self.risk_manager.get_risk_report()
            
            # Get slippage stats
            slippage_stats = self.broker_execution.get_slippage_stats()
            
            # Calculate performance metrics
            total_trades = len(self.trade_history)
            winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
            
            return {
                'timestamp': datetime.now(),
                'trading_mode': self.trading_mode,
                'auto_execution': self.auto_execution,
                'risk_metrics': risk_metrics,
                'risk_report': risk_report,
                'slippage_stats': slippage_stats,
                'trade_history': self.trade_history,
                'performance': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'initial_capital': self.risk_config.initial_capital,
                    'current_capital': risk_metrics.current_capital,
                    'returns_pct': ((risk_metrics.current_capital - self.risk_config.initial_capital) / 
                                  self.risk_config.initial_capital * 100)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")
            return {}
    
    def manual_order(self, contract_symbol: str, quantity: int, side: str, 
                    order_type: str = 'MARKET', price: Optional[float] = None) -> Dict:
        """Place a manual order."""
        try:
            # Find contract
            contract = None
            for symbol in self.symbols:
                option_chain = self.data_manager.get_option_chain(symbol)
                if option_chain:
                    for c in option_chain.contracts:
                        if c.symbol == contract_symbol:
                            contract = c
                            break
                    if contract:
                        break
            
            if not contract:
                return {'success': False, 'message': 'Contract not found'}
            
            # Convert side string to enum
            side_enum = OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL
            order_type_enum = OrderType(order_type.upper())
            
            # Check risk management
            can_place, reason = self.risk_manager.check_can_place_order(
                contract, quantity, side_enum, price
            )
            
            if not can_place:
                return {'success': False, 'message': f'Risk check failed: {reason}'}
            
            # Place order
            response = self.broker_execution.place_option_order(
                contract=contract,
                quantity=quantity,
                side=side_enum,
                order_type=order_type_enum,
                price=price
            )
            
            if response.status.value in ['COMPLETE', 'PARTIALLY_FILLED']:
                # Record trade
                self.risk_manager.record_trade(
                    contract=contract,
                    quantity=response.filled_quantity,
                    side=side_enum,
                    price=response.average_price
                )
                
                return {
                    'success': True,
                    'order_id': response.order_id,
                    'filled_quantity': response.filled_quantity,
                    'average_price': response.average_price,
                    'status': response.status.value
                }
            else:
                return {
                    'success': False,
                    'message': response.message,
                    'status': response.status.value
                }
                
        except Exception as e:
            logger.error(f"❌ Error placing manual order: {e}")
            return {'success': False, 'message': str(e)}


def main():
    """Main function for live trading system."""
    parser = argparse.ArgumentParser(description='Live Options Trading System')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper', help='Trading mode')
    parser.add_argument('--auto-execution', action='store_true', help='Enable auto execution')
    parser.add_argument('--symbols', nargs='+', default=['NSE:NIFTY50-INDEX'], help='Trading symbols')
    parser.add_argument('--strategies', nargs='+', default=['ema_crossover_enhanced'], help='Trading strategies')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--max-daily-loss', type=float, default=0.03, help='Max daily loss percentage')
    parser.add_argument('--confidence-cutoff', type=float, default=40.0, help='Confidence cutoff')
    parser.add_argument('--data-provider', default='historical', help='Data provider type')
    parser.add_argument('--broker', default='paper', help='Broker type')
    
    args = parser.parse_args()
    
    # Create risk configuration
    risk_config = RiskConfig(
        initial_capital=args.capital,
        max_daily_loss_pct=args.max_daily_loss
    )
    
    # Create trading system
    trading_system = LiveOptionsTradingSystem(
        trading_mode=args.mode,
        auto_execution=args.auto_execution,
        symbols=args.symbols,
        strategies=args.strategies,
        confidence_cutoff=args.confidence_cutoff,
        risk_config=risk_config,
        data_provider_type=args.data_provider,
        broker_type=args.broker
    )
    
    try:
        # Start system
        trading_system.start()
        
        # Keep running
        while True:
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        # Stop system
        trading_system.stop()
        
        # Print final report
        report = trading_system.get_performance_report()
        if report:
            print("\n" + "=" * 60)
            print("📊 FINAL PERFORMANCE REPORT")
            print("=" * 60)
            print(f"💰 Initial Capital: ₹{report['performance']['initial_capital']:,.2f}")
            print(f"💰 Current Capital: ₹{report['performance']['current_capital']:,.2f}")
            print(f"📈 Total Returns: {report['performance']['returns_pct']:+.2f}%")
            print(f"📊 Total Trades: {report['performance']['total_trades']}")
            print(f"🎯 Win Rate: {report['performance']['win_rate']:.1f}%")
            print(f"💵 Total P&L: ₹{report['performance']['total_pnl']:+,.2f}")
            print("=" * 60)


if __name__ == "__main__":
    main() 