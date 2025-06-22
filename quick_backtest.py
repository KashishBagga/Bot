import sys
import os
sys.path.append('src')
import pandas as pd
from strategies.ema_crossover import EmaCrossover

def quick_backtest():
    """Quick backtest of EMA crossover strategy"""
    
    # Load 5min data for faster processing
    data_path = 'data/parquet/NSE_NIFTYBANK_INDEX/5min.parquet'
    data = pd.read_parquet(data_path)
    print(f"Data loaded: {data.shape}")
    
    # Initialize strategy
    strategy = EmaCrossover()
    
    # Add indicators to the data
    data_with_indicators = strategy.add_indicators(data.copy())
    
    # Skip initial rows where EMAs are still stabilizing
    start_idx = 50
    signals = []
    test_threshold = 0.2
    
    print(f"Scanning {len(data_with_indicators)} rows for signals with threshold {test_threshold}...")
    
    for i in range(start_idx, len(data_with_indicators)):
        candle = data_with_indicators.iloc[i]
        
        # Skip if EMAs are zero or invalid
        if (candle['ema_fast'] == 0 or candle['ema_slow'] == 0 or 
            pd.isna(candle['ema_fast']) or pd.isna(candle['ema_slow'])):
            continue
        
        # Manual signal generation with lower threshold
        crossover_strength = abs(candle['crossover_strength']) if not pd.isna(candle['crossover_strength']) and candle['crossover_strength'] != float('inf') else 0
        
        signal = None
        if (candle['ema_fast'] > candle['ema_slow'] and 
            candle['close'] > candle['ema_fast'] and
            crossover_strength > test_threshold):
            signal = 'BUY'
        elif (candle['ema_fast'] < candle['ema_slow'] and 
              candle['close'] < candle['ema_fast'] and
              crossover_strength > test_threshold):
            signal = 'SELL'
        
        if signal:
            signals.append({
                'time': candle.name if hasattr(candle, 'name') else data_with_indicators.index[i],
                'signal': signal,
                'price': candle['close'],
                'strength': crossover_strength
            })
    
    print(f"Signals generated: {len(signals)}")
    
    if signals:
        print("\nFirst 5 signals:")
        for i, signal in enumerate(signals[:5]):
            print(f"  {i+1}. {signal['signal']} at {signal['price']:.2f}, strength: {signal['strength']:.4f}")
    
    # Simple trading simulation
    initial_capital = 100000
    capital = initial_capital
    trades = []
    position = None
    
    for signal_data in signals:
        if position is None:  # Enter position
            position = {
                'type': signal_data['signal'],
                'entry_price': signal_data['price'],
                'entry_time': signal_data['time'],
                'quantity': int(capital / signal_data['price'])
            }
        else:  # Exit position
            exit_price = signal_data['price']
            if position['type'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['quantity']
            else:  # SELL
                pnl = (position['entry_price'] - exit_price) * position['quantity']
            
            trades.append({
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'success': pnl > 0
            })
            
            capital += pnl
            position = None
    
    # Calculate metrics
    total_return = ((capital - initial_capital) / initial_capital) * 100
    num_trades = len(trades)
    win_rate = (sum(1 for t in trades if t['success']) / num_trades * 100) if num_trades > 0 else 0
    
    print(f"\n🎯 EMA Crossover Strategy Results:")
    print(f"   Return: {total_return:.2f}%")
    print(f"   Trades: {num_trades}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Final Capital: ${capital:,.2f}")
    
    if trades:
        avg_pnl = sum(t['pnl'] for t in trades) / len(trades)
        max_profit = max(t['pnl'] for t in trades)
        max_loss = min(t['pnl'] for t in trades)
        print(f"   Avg PnL per trade: ${avg_pnl:.2f}")
        print(f"   Max Profit: ${max_profit:.2f}")
        print(f"   Max Loss: ${max_loss:.2f}")

if __name__ == "__main__":
    quick_backtest() 