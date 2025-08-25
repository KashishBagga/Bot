#!/usr/bin/env python3
"""
Trading System Optimization Plan
Based on performance analysis - focus on profitability
"""

# 1. STRATEGY PRIORITIZATION (based on 30-day performance)
PROFITABLE_STRATEGIES = {
    'supertrend_ema': {
        'symbols': ['NSE:NIFTY50-INDEX'],
        'min_confidence': 75,  # Increase from 60
        'position_size': 1.0,
        'active': True
    },
    'supertrend_macd_rsi_ema': {
        'symbols': ['NSE:NIFTYBANK-INDEX'], 
        'min_confidence': 80,  # Increase from 60
        'position_size': 1.0,
        'active': True
    }
}

# 2. STRATEGIES TO OPTIMIZE (currently losing money)
STRATEGIES_TO_FIX = {
    'ema_crossover': {
        'issue': 'High volume (1000+ trades), low win rate (8-13%)',
        'fixes': [
            'Increase confidence threshold 60→85',
            'Add volume filter (ratio > 1.5)',
            'Add MACD alignment filter', 
            'Reduce position size to 0.3x for testing'
        ],
        'target_trades_per_month': '<200',
        'target_win_rate': '>25%'
    },
    'macd_cross_rsi_filter': {
        'issue': 'High volume, 41-44% win rate but net negative',
        'fixes': [
            'Improve R:R ratio (currently 1.5:1 → target 2:1)',
            'Add market regime filter (low VIX proxy)',
            'Tighten entry criteria (price vs multiple EMAs)',
            'Add time-based exits (EOD)'
        ]
    },
    'rsi_mean_reversion_bb': {
        'issue': '46-52% win rate but still net negative',
        'fixes': [
            'Only trade in ranging markets (EMA proximity)',
            'Add volume spike confirmation',
            'Improve position sizing based on volatility',
            'Add profit target scaling'
        ]
    }
}

# 3. RISK MANAGEMENT IMPROVEMENTS
RISK_OPTIMIZATIONS = {
    'dynamic_position_sizing': {
        'high_confidence': 1.5,  # 85+ confidence
        'medium_confidence': 1.0,  # 70-84 confidence  
        'low_confidence': 0.5   # 60-69 confidence
    },
    'market_regime_filters': {
        'vix_proxy_max': 0.015,  # ATR/Price ratio
        'trend_strength_min': 0.005,  # EMA separation
        'volume_confirmation': 1.2  # Above average volume
    },
    'time_management': {
        'no_trades_first_30min': True,
        'no_trades_last_30min': True, 
        'eod_exit_time': '15:00',
        'max_trade_duration': '4 hours'
    }
}

# 4. BACKTESTING PLAN
OPTIMIZATION_TESTS = [
    {
        'name': 'profitable_only',
        'strategies': ['supertrend_ema', 'supertrend_macd_rsi_ema'],
        'period': '90 days',
        'target': 'Maintain profitability, reduce drawdown'
    },
    {
        'name': 'ema_crossover_fix',
        'strategies': ['ema_crossover'], 
        'modifications': 'High confidence + volume filter',
        'target': 'Reduce trades to <200/month, win rate >25%'
    },
    {
        'name': 'improved_rr',
        'strategies': ['macd_cross_rsi_filter'],
        'modifications': 'Dynamic R:R based on volatility',
        'target': 'Achieve net positive with <1000 trades/month'
    }
]

# 5. MONITORING METRICS
SUCCESS_METRICS = {
    'monthly_pnl': '>0',  # Net positive
    'max_drawdown': '<2000',  # Limit max loss
    'win_rate': '>40%',  # Minimum acceptable
    'avg_trade': '>5',  # Positive expectancy
    'trades_per_month': '<500',  # Quality over quantity
    'sharpe_ratio': '>0.5'  # Risk-adjusted returns
}

if __name__ == '__main__':
    print("🎯 TRADING SYSTEM OPTIMIZATION PLAN")
    print("="*50)
    print("\n📈 PROFITABLE STRATEGIES TO MAINTAIN:")
    for strategy, config in PROFITABLE_STRATEGIES.items():
        print(f"  ✅ {strategy}: {config}")
    
    print("\n🔧 STRATEGIES TO FIX:")
    for strategy, details in STRATEGIES_TO_FIX.items():
        print(f"  ❌ {strategy}: {details['issue']}")
        print(f"      Fixes: {details['fixes']}")
    
    print("\n🎯 SUCCESS TARGETS:")
    for metric, target in SUCCESS_METRICS.items():
        print(f"  📊 {metric}: {target}") 