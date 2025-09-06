#!/usr/bin/env python3
"""
Shadow Mode with Broker Integration
Run system live with mock trades alongside real broker quotes to detect slippage/latency
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ShadowTrade:
    """Shadow trade for comparison with real broker"""
    id: str
    timestamp: datetime
    symbol: str
    signal_type: str
    strategy: str
    mock_entry_price: float
    mock_quantity: int
    broker_entry_price: Optional[float] = None
    broker_quantity: Optional[int] = None
    slippage: Optional[float] = None
    latency_ms: Optional[float] = None
    status: str = 'PENDING'
    mock_pnl: Optional[float] = None
    broker_pnl: Optional[float] = None
    pnl_difference: Optional[float] = None

class ShadowModeBroker:
    """Shadow mode broker for comparing mock vs real trades"""
    
    def __init__(self, trading_system, broker_api=None):
        self.trading_system = trading_system
        self.broker_api = broker_api
        self.shadow_trades = []
        self.comparison_metrics = {
            'total_trades': 0,
            'slippage_sum': 0.0,
            'latency_sum': 0.0,
            'pnl_difference_sum': 0.0,
            'mock_pnl_sum': 0.0,
            'broker_pnl_sum': 0.0
        }
        self.is_running = False
        self._lock = threading.RLock()
        
    def start_shadow_mode(self):
        """Start shadow mode trading"""
        logger.info("🎭 Starting shadow mode trading...")
        self.is_running = True
        
        # Override trading system's trade opening method
        original_open_trade = self.trading_system._open_paper_trade
        self.trading_system._open_paper_trade = self._shadow_open_trade
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_shadow_trades, daemon=True)
        monitor_thread.start()
        
        logger.info("✅ Shadow mode started - monitoring mock vs real trades")
    
    def stop_shadow_mode(self):
        """Stop shadow mode trading"""
        logger.info("🛑 Stopping shadow mode trading...")
        self.is_running = False
        
        # Restore original method
        # Note: In production, you'd want to properly restore the original method
        
        # Generate final comparison report
        self._generate_comparison_report()
        
        logger.info("✅ Shadow mode stopped")
    
    def _shadow_open_trade(self, signal: Dict, option_contract, entry_price: float, timestamp: datetime, last_prices: Dict[str, float] = None) -> Optional[str]:
        """Shadow version of open trade that compares with real broker"""
        try:
            # Create shadow trade record
            shadow_trade = ShadowTrade(
                id=f"shadow_{int(time.time() * 1000)}",
                timestamp=timestamp,
                symbol=signal['symbol'],
                signal_type=signal['signal'],
                strategy=signal['strategy'],
                mock_entry_price=entry_price,
                mock_quantity=option_contract.lot_size,
                status='PENDING'
            )
            
            # Record mock trade
            with self._lock:
                self.shadow_trades.append(shadow_trade)
                self.comparison_metrics['total_trades'] += 1
            
            # Get real broker quote (if available)
            broker_start_time = time.time()
            broker_price = self._get_broker_quote(signal['symbol'], signal['signal'])
            broker_latency = (time.time() - broker_start_time) * 1000  # ms
            
            if broker_price:
                shadow_trade.broker_entry_price = broker_price
                shadow_trade.latency_ms = broker_latency
                shadow_trade.slippage = abs(broker_price - entry_price) / entry_price * 100
                
                # Update metrics
                with self._lock:
                    self.comparison_metrics['slippage_sum'] += shadow_trade.slippage
                    self.comparison_metrics['latency_sum'] += broker_latency
                
                logger.info(f"🎭 Shadow Trade: Mock ₹{entry_price:.2f} vs Broker ₹{broker_price:.2f} "
                           f"(Slippage: {shadow_trade.slippage:.2f}%, Latency: {broker_latency:.1f}ms)")
            
            # Execute mock trade (call original method)
            trade_id = self.trading_system._open_paper_trade_original(signal, option_contract, entry_price, timestamp, last_prices)
            
            if trade_id:
                shadow_trade.status = 'EXECUTED'
                logger.info(f"✅ Shadow trade executed: {trade_id[:8]}...")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"❌ Error in shadow trade: {e}")
            return None
    
    def _get_broker_quote(self, symbol: str, signal_type: str) -> Optional[float]:
        """Get real broker quote for comparison"""
        try:
            if not self.broker_api:
                # Simulate broker quote with some realistic variation
                base_price = self.trading_system._get_price_cached(symbol)
                if base_price:
                    # Add realistic slippage simulation
                    slippage_factor = 0.001 if 'BUY' in signal_type else -0.001  # 0.1% slippage
                    return base_price * (1 + slippage_factor)
                return None
            
            # In production, this would call your real broker API
            # return self.broker_api.get_option_quote(symbol, signal_type)
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting broker quote: {e}")
            return None
    
    def _monitor_shadow_trades(self):
        """Monitor shadow trades and update PnL comparisons"""
        while self.is_running:
            try:
                with self._lock:
                    for trade in self.shadow_trades:
                        if trade.status == 'EXECUTED' and trade.mock_pnl is None:
                            # Calculate mock PnL
                            trade.mock_pnl = self._calculate_mock_pnl(trade)
                            
                            # Calculate broker PnL if we have broker price
                            if trade.broker_entry_price:
                                trade.broker_pnl = self._calculate_broker_pnl(trade)
                                trade.pnl_difference = trade.broker_pnl - trade.mock_pnl
                                
                                # Update metrics
                                self.comparison_metrics['pnl_difference_sum'] += trade.pnl_difference
                                self.comparison_metrics['mock_pnl_sum'] += trade.mock_pnl
                                self.comparison_metrics['broker_pnl_sum'] += trade.broker_pnl
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"❌ Error monitoring shadow trades: {e}")
                time.sleep(5)
    
    def _calculate_mock_pnl(self, trade: ShadowTrade) -> float:
        """Calculate PnL for mock trade"""
        try:
            # Get current price
            current_price = self.trading_system._get_price_cached(trade.symbol)
            if not current_price:
                return 0.0
            
            # Simple PnL calculation
            if 'BUY' in trade.signal_type:
                return (current_price - trade.mock_entry_price) * trade.mock_quantity
            else:
                return (trade.mock_entry_price - current_price) * trade.mock_quantity
                
        except Exception as e:
            logger.error(f"❌ Error calculating mock PnL: {e}")
            return 0.0
    
    def _calculate_broker_pnl(self, trade: ShadowTrade) -> float:
        """Calculate PnL for broker trade"""
        try:
            if not trade.broker_entry_price:
                return 0.0
            
            # Get current price
            current_price = self.trading_system._get_price_cached(trade.symbol)
            if not current_price:
                return 0.0
            
            # Simple PnL calculation
            if 'BUY' in trade.signal_type:
                return (current_price - trade.broker_entry_price) * trade.mock_quantity
            else:
                return (trade.broker_entry_price - current_price) * trade.mock_quantity
                
        except Exception as e:
            logger.error(f"❌ Error calculating broker PnL: {e}")
            return 0.0
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        try:
            with self._lock:
                total_trades = self.comparison_metrics['total_trades']
                if total_trades == 0:
                    logger.info("📊 No shadow trades to compare")
                    return
                
                avg_slippage = self.comparison_metrics['slippage_sum'] / total_trades
                avg_latency = self.comparison_metrics['latency_sum'] / total_trades
                total_pnl_diff = self.comparison_metrics['pnl_difference_sum']
                mock_pnl_total = self.comparison_metrics['mock_pnl_sum']
                broker_pnl_total = self.comparison_metrics['broker_pnl_sum']
                
                logger.info("=" * 80)
                logger.info("🎭 SHADOW MODE COMPARISON REPORT")
                logger.info("=" * 80)
                logger.info(f"📊 Total Trades Compared: {total_trades}")
                logger.info(f"📈 Average Slippage: {avg_slippage:.2f}%")
                logger.info(f"⏱️ Average Latency: {avg_latency:.1f}ms")
                logger.info(f"💰 Mock PnL Total: ₹{mock_pnl_total:+,.2f}")
                logger.info(f"💰 Broker PnL Total: ₹{broker_pnl_total:+,.2f}")
                logger.info(f"📊 PnL Difference: ₹{total_pnl_diff:+,.2f}")
                logger.info(f"📊 PnL Difference %: {(total_pnl_diff/mock_pnl_total*100):+.2f}%" if mock_pnl_total != 0 else "N/A")
                
                # Save detailed report
                report_data = {
                    'timestamp': datetime.now().isoformat(),
                    'total_trades': total_trades,
                    'average_slippage_pct': avg_slippage,
                    'average_latency_ms': avg_latency,
                    'mock_pnl_total': mock_pnl_total,
                    'broker_pnl_total': broker_pnl_total,
                    'pnl_difference': total_pnl_diff,
                    'pnl_difference_pct': (total_pnl_diff/mock_pnl_total*100) if mock_pnl_total != 0 else 0,
                    'trades': [asdict(trade) for trade in self.shadow_trades]
                }
                
                filename = f"shadow_mode_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                
                logger.info(f"📄 Detailed report saved to: {filename}")
                logger.info("=" * 80)
                
        except Exception as e:
            logger.error(f"❌ Error generating comparison report: {e}")
    
    def get_shadow_metrics(self) -> Dict:
        """Get current shadow mode metrics"""
        with self._lock:
            return {
                'total_trades': self.comparison_metrics['total_trades'],
                'average_slippage': self.comparison_metrics['slippage_sum'] / max(self.comparison_metrics['total_trades'], 1),
                'average_latency': self.comparison_metrics['latency_sum'] / max(self.comparison_metrics['total_trades'], 1),
                'pnl_difference': self.comparison_metrics['pnl_difference_sum'],
                'is_running': self.is_running
            }

def main():
    """Main function to run shadow mode"""
    try:
        from live_paper_trading import LivePaperTradingSystem
        
        # Initialize trading system
        logger.info("🚀 Initializing trading system for shadow mode...")
        trading_system = LivePaperTradingSystem(initial_capital=100000)
        
        # Initialize shadow mode broker
        shadow_broker = ShadowModeBroker(trading_system)
        
        # Start shadow mode
        shadow_broker.start_shadow_mode()
        
        # Start trading system
        trading_system.start_trading()
        
        logger.info("🎭 Shadow mode running - press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Stopping shadow mode...")
            shadow_broker.stop_shadow_mode()
            trading_system.stop_trading()
            logger.info("✅ Shadow mode stopped")
        
    except Exception as e:
        logger.error(f"❌ Shadow mode failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
