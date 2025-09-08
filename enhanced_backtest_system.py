#!/usr/bin/env python3
"""
Enhanced Backtest System with Unified Strategy Engine
"""

import os
import sys
import pandas as pd
import logging
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional
import json

# Add src to path
sys.path.append('src')

from src.core.unified_strategy_engine import UnifiedStrategyEngine
from src.api.fyers import FyersClient
from src.models.unified_database import UnifiedTradingDatabase
from src.models.option_contract import OptionContract, OptionType

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_backtest.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class EnhancedBacktestSystem:
    """Enhanced backtest system with unified strategy engine"""
    
    def __init__(self, 
                 symbols: List[str],
                 initial_capital: float = 20000.0,
                 commission_rate: float = 0.0001,
                 confidence_cutoff: float = 0.6):
        
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.confidence_cutoff = confidence_cutoff
        self.tz = ZoneInfo("Asia/Kolkata")
        
        # Initialize components
        self.strategy_engine = UnifiedStrategyEngine(symbols, confidence_cutoff)
        self.data_provider = FyersClient()
        self.db = UnifiedTradingDatabase("unified_trading.db")
        
        # Trading state
        self.cash = initial_capital
        self.trades = []
        self.open_positions = {}
        self.equity_curve = []
        
        # Performance metrics
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        self.total_fees = 0.0
        
        logger.info(f"✅ Enhanced Backtest System initialized with ₹{initial_capital:,.2f} capital")
    
    def fetch_historical_data(self, start_date: str, end_date: str, resolution: str = '1') -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols"""
        
        if not self.data_provider.initialize_client():
            logger.error("❌ Failed to initialize Fyers client")
            return {}
        
        data = {}
        
        for symbol in self.symbols:
            try:
                logger.info(f"📊 Fetching data for {symbol} from {start_date} to {end_date}")
                
                # Fetch historical data
                raw_data = self.data_provider.get_historical_data(
                    symbol=symbol,
                    resolution=resolution,
                    range_from=start_date,
                    range_to=end_date
                )
                
                if raw_data and 'candles' in raw_data and len(raw_data['candles']) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(raw_data['candles'], 
                                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('timestamp', inplace=True)
                    
                    # Remove duplicate timestamps
                    df = df[~df.index.duplicated(keep='last')]
                    
                    data[symbol] = df
                    logger.info(f"✅ Loaded {len(df)} candles for {symbol}")
                else:
                    logger.warning(f"⚠️ No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching data for {symbol}: {e}")
                continue
        
        return data
    
    def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Run backtest on historical data"""
        
        logger.info("🚀 Starting enhanced backtest...")
        
        # Process data chronologically
        all_timestamps = set()
        for symbol_data in data.values():
            all_timestamps.update(symbol_data.index)
        
        timestamps = sorted(all_timestamps)
        
        # Find the minimum required candles across all strategies
        min_candles_required = max(
            self.strategy_engine.strategies['ema_crossover_enhanced'].min_candles,
            self.strategy_engine.strategies['supertrend_ema'].min_candles,
            self.strategy_engine.strategies['supertrend_macd_rsi_ema'].min_candles
        )
        
        logger.info(f"📊 Minimum candles required: {min_candles_required}")
        
        # Generate signals using the full dataset for each timestamp
        for i, timestamp in enumerate(timestamps):
            # Get current prices for all symbols
            current_prices = {}
            current_data = {}
            
            for symbol in self.symbols:
                if symbol in data:
                    symbol_data = data[symbol]
                    if timestamp in symbol_data.index:
                        current_prices[symbol] = symbol_data.loc[timestamp, 'close']
                        
                        # Use data up to current timestamp for signal generation
                        # This ensures we have enough historical data for indicators
                        current_data[symbol] = symbol_data.loc[:timestamp]
            
            # Only generate signals if we have enough data
            if i >= min_candles_required:
                # Generate signals using unified engine with full historical data
                signals = self.strategy_engine.generate_signals(current_data, current_prices)
                
                # Process signals
                for signal in signals:
                    if self.strategy_engine.validate_signal(signal):
                        self._process_signal(signal, current_prices)
            
            # Check for exits
            self._check_exits(current_prices, timestamp)
            
            # Update equity curve
            self._update_equity_curve(timestamp, current_prices)
        
        # Close any remaining positions
        self._close_all_positions(current_prices, timestamps[-1])
        
        # Calculate final metrics
        results = self._calculate_results()
        
        logger.info("✅ Enhanced backtest completed")
        return results
    
    def _process_signal(self, signal: Dict, current_prices: Dict[str, float]):
        """Process a trading signal"""
        
        symbol = signal['symbol']
        strategy = signal['strategy']
        signal_type = signal['signal']
        confidence = signal['confidence']
        
        # Check if we can afford the trade
        if symbol not in current_prices:
            return
        
        current_price = current_prices[symbol]
        
        # Create option contract (simplified for backtest)
        option_contract = self._create_option_contract(signal, current_price)
        if not option_contract:
            return
        
        # Calculate position size (1 lot)
        lot_size = option_contract.lot_size
        premium_per_lot = option_contract.last * lot_size
        commission = premium_per_lot * self.commission_rate
        total_cost = premium_per_lot + commission
        
        if total_cost > self.cash:
            logger.debug(f"⚠️ Insufficient capital for {strategy} - need ₹{total_cost:,.2f}, have ₹{self.cash:,.2f}")
            return
        
        # Open position
        position_id = f"{symbol}_{strategy}_{signal_type}_{len(self.trades)}"
        
        trade = {
            'id': position_id,
            'symbol': symbol,
            'strategy': strategy,
            'signal_type': signal_type,
            'confidence': confidence,
            'entry_time': signal['timestamp'],
            'entry_price': option_contract.last,
            'quantity': lot_size,
            'commission': commission,
            'status': 'open'
        }
        
        self.open_positions[position_id] = trade
        self.cash -= total_cost
        self.total_fees += commission
        
        logger.info(f"✅ Opened {signal_type} position: {strategy} on {symbol} at ₹{option_contract.last:.2f}")
    
    def _create_option_contract(self, signal: Dict, current_price: float) -> Optional[OptionContract]:
        """Create option contract for backtest"""
        
        symbol = signal['symbol']
        signal_type = signal['signal']
        
        # Calculate ATM strike
        atm_strike = round(current_price / 50) * 50
        
        # Determine option type
        if 'CALL' in signal_type:
            option_type = OptionType.CALL
            premium = current_price * 0.008  # 0.8% for ATM options
        elif 'PUT' in signal_type:
            option_type = OptionType.PUT
            premium = current_price * 0.008  # 0.8% for ATM options
        else:
            return None
        
        # Get lot size
        if 'NIFTY50' in symbol:
            lot_size = 50
        elif 'NIFTYBANK' in symbol:
            lot_size = 25
        else:
            lot_size = 50
        
        # Create contract
        contract = OptionContract(
            symbol=f"{symbol.replace(':', '')}{self.now_kolkata().strftime('%d%m%y')}{atm_strike}{'CE' if option_type == OptionType.CALL else 'PE'}",
            underlying=symbol,
            strike=atm_strike,
            expiry=self.now_kolkata() + timedelta(days=7),  # Next week
            option_type=option_type,
            lot_size=lot_size,
            bid=premium * 0.95,
            ask=premium * 1.05,
            last=premium,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            delta=0.5 if option_type == OptionType.CALL else -0.5,
            gamma=0.01,
            theta=-premium * 0.1,
            vega=premium * 0.5
        )
        
        return contract
    
    def _check_exits(self, current_prices: Dict[str, float], timestamp: datetime):
        """Check for position exits"""
        
        positions_to_close = []
        
        for position_id, position in self.open_positions.items():
            symbol = position['symbol']
            
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            entry_price = position['entry_price']
            
            # Simple exit logic (can be enhanced)
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Exit conditions
            exit_reason = None
            
            if pnl_pct >= 0.30:  # 30% profit
                exit_reason = "take_profit"
            elif pnl_pct <= -0.15:  # 15% loss
                exit_reason = "stop_loss"
            elif (timestamp - position['entry_time']).total_seconds() > 3600:  # 1 hour
                exit_reason = "time_exit"
            
            if exit_reason:
                positions_to_close.append((position_id, current_price, exit_reason))
        
        # Close positions
        for position_id, exit_price, reason in positions_to_close:
            self._close_position(position_id, exit_price, reason, timestamp)
    
    def _close_position(self, position_id: str, exit_price: float, reason: str, timestamp: datetime):
        """Close a position"""
        
        position = self.open_positions[position_id]
        
        # Calculate P&L
        entry_price = position['entry_price']
        quantity = position['quantity']
        entry_commission = position['commission']
        
        exit_value = exit_price * quantity
        exit_commission = exit_value * self.commission_rate
        
        pnl = (exit_price - entry_price) * quantity - entry_commission - exit_commission
        
        # Update cash
        self.cash += exit_value - exit_commission
        self.total_fees += exit_commission
        
        # Record trade
        trade = {
            'id': position_id,
            'symbol': position['symbol'],
            'strategy': position['strategy'],
            'signal_type': position['signal_type'],
            'confidence': position['confidence'],
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'fees': entry_commission + exit_commission,
            'exit_reason': reason
        }
        
        self.trades.append(trade)
        del self.open_positions[position_id]
        
        logger.info(f"🔒 Closed {position['signal_type']} position: {position['strategy']} on {position['symbol']} | P&L: ₹{pnl:,.2f} ({reason})")
    
    def _close_all_positions(self, current_prices: Dict[str, float], timestamp: datetime):
        """Close all remaining positions"""
        
        for position_id in list(self.open_positions.keys()):
            symbol = self.open_positions[position_id]['symbol']
            exit_price = current_prices.get(symbol, self.open_positions[position_id]['entry_price'])
            self._close_position(position_id, exit_price, "eod_exit", timestamp)
    
    def _update_equity_curve(self, timestamp: datetime, current_prices: Dict[str, float]):
        """Update equity curve"""
        
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for position in self.open_positions.values():
            symbol = position['symbol']
            if symbol in current_prices:
                current_price = current_prices[symbol]
                entry_price = position['entry_price']
                quantity = position['quantity']
                unrealized_pnl += (current_price - entry_price) * quantity
        
        equity = self.cash + unrealized_pnl
        
        # Update peak capital and drawdown
        if equity > self.peak_capital:
            self.peak_capital = equity
        
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - equity) / self.peak_capital
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.cash,
            'unrealized_pnl': unrealized_pnl,
            'open_positions': len(self.open_positions)
        })
    
    def _calculate_results(self) -> Dict:
        """Calculate final backtest results"""
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] < 0])
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = (self.cash - self.initial_capital) / self.initial_capital * 100
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Strategy performance
        strategy_performance = self.strategy_engine.get_strategy_performance(self.trades)
        
        results = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.cash,
                'total_return_pct': total_return,
                'total_pnl': total_pnl,
                'total_fees': self.total_fees,
                'max_drawdown_pct': self.max_drawdown * 100,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate_pct': win_rate
            },
            'strategy_performance': strategy_performance,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }
        
        return results
    
    def now_kolkata(self) -> datetime:
        """Get current time in Kolkata timezone"""
        return datetime.now(self.tz)
    
    def generate_report(self, results: Dict, output_file: str = None) -> str:
        """Generate comprehensive backtest report"""
        
        if output_file is None:
            timestamp = self.now_kolkata().strftime('%Y%m%d_%H%M%S')
            output_file = f"backtest_report_{timestamp}.html"
        
        summary = results['summary']
        strategy_performance = results['strategy_performance']
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .strategy {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .trades {{ background-color: #fff8f0; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Enhanced Backtest Report</h1>
                <p>Generated on: {self.now_kolkata().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <table>
                    <tr><td>Initial Capital</td><td>₹{summary['initial_capital']:,.2f}</td></tr>
                    <tr><td>Final Capital</td><td>₹{summary['final_capital']:,.2f}</td></tr>
                    <tr><td>Total Return</td><td class="{'positive' if summary['total_return_pct'] > 0 else 'negative'}">{summary['total_return_pct']:+.2f}%</td></tr>
                    <tr><td>Total P&L</td><td class="{'positive' if summary['total_pnl'] > 0 else 'negative'}">₹{summary['total_pnl']:+,.2f}</td></tr>
                    <tr><td>Total Fees</td><td>₹{summary['total_fees']:,.2f}</td></tr>
                    <tr><td>Max Drawdown</td><td class="negative">{summary['max_drawdown_pct']:.2f}%</td></tr>
                    <tr><td>Total Trades</td><td>{summary['total_trades']}</td></tr>
                    <tr><td>Win Rate</td><td>{summary['win_rate_pct']:.1f}%</td></tr>
                </table>
            </div>
            
            <div class="strategy">
                <h2>Strategy Performance</h2>
                <table>
                    <tr>
                        <th>Strategy</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Total P&L</th>
                        <th>Avg P&L</th>
                        <th>Net P&L</th>
                    </tr>
        """
        
        for strategy, stats in strategy_performance.items():
            html_content += f"""
                    <tr>
                        <td>{strategy}</td>
                        <td>{stats['total_trades']}</td>
                        <td>{stats['win_rate']:.1f}%</td>
                        <td class="{'positive' if stats['total_pnl'] > 0 else 'negative'}">₹{stats['total_pnl']:+,.2f}</td>
                        <td class="{'positive' if stats['avg_pnl'] > 0 else 'negative'}">₹{stats['avg_pnl']:+,.2f}</td>
                        <td class="{'positive' if stats['net_pnl'] > 0 else 'negative'}">₹{stats['net_pnl']:+,.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="trades">
                <h2>Trade Details</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Symbol</th>
                        <th>Strategy</th>
                        <th>Signal</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>P&L</th>
                        <th>Exit Reason</th>
                    </tr>
        """
        
        for trade in results['trades']:
            html_content += f"""
                    <tr>
                        <td>{trade['id']}</td>
                        <td>{trade['symbol']}</td>
                        <td>{trade['strategy']}</td>
                        <td>{trade['signal_type']}</td>
                        <td>{trade['entry_time'].strftime('%H:%M:%S')}</td>
                        <td>{trade['exit_time'].strftime('%H:%M:%S')}</td>
                        <td>₹{trade['entry_price']:.2f}</td>
                        <td>₹{trade['exit_price']:.2f}</td>
                        <td class="{'positive' if trade['pnl'] > 0 else 'negative'}">₹{trade['pnl']:+,.2f}</td>
                        <td>{trade['exit_reason']}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"📊 Report generated: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description='Enhanced Backtest System')
    parser.add_argument('--symbols', nargs='+', default=['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX'],
                       help='Trading symbols')
    parser.add_argument('--capital', type=float, default=20000.0, help='Initial capital')
    parser.add_argument('--start_date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--confidence', type=float, default=0.6, help='Confidence cutoff')
    parser.add_argument('--output', help='Output report file')
    
    args = parser.parse_args()
    
    # Initialize backtest system
    backtest = EnhancedBacktestSystem(
        symbols=args.symbols,
        initial_capital=args.capital,
        confidence_cutoff=args.confidence
    )
    
    # Fetch data
    data = backtest.fetch_historical_data(args.start_date, args.end_date)
    
    if not data:
        logger.error("❌ No data available for backtest")
        return
    
    # Run backtest
    results = backtest.run_backtest(data)
    
    # Generate report
    report_file = backtest.generate_report(results, args.output)
    
    # Print summary
    summary = results['summary']

if __name__ == "__main__":
    main() 