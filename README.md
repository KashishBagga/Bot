# 🚀 Simple Trading System

A clean, modular trading system for crypto and Indian stocks.

## ✨ Features

- **Crypto Trading**: Binance API integration
- **Indian Stock Trading**: Fyers API integration  
- **4 Trading Strategies**: EMA Crossover, SuperTrend, MACD RSI
- **Risk Management**: Stop-loss, position sizing
- **Real-time Trading**: Live market data and execution

## 🚀 Quick Start

### Crypto Trading
```bash
python3 crypto_trader.py
```

### Indian Stock Trading
```bash
python3 indian_trader.py
```

### Test Fyers API
```bash
python3 test_fyers.py
```

## 📁 Structure

```
├── crypto_trader.py          # Crypto trading system
├── indian_trader.py          # Indian stock trading system
├── test_fyers.py            # Fyers API testing
├── src/                     # Core system components
│   ├── markets/             # Market implementations
│   ├── strategies/          # Trading strategies
│   ├── core/               # Core utilities
│   └── api/                # API clients
└── logs/                   # Trading logs
```

## 🔧 Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp sample.env .env
# Edit .env with your API credentials
```

3. Start trading:
```bash
python3 crypto_trader.py    # For crypto
python3 indian_trader.py    # For Indian stocks
```

## 📊 Monitoring

Check logs:
```bash
tail -f logs/crypto/crypto_trading.log
tail -f logs/indian/indian_trading.log
```

That's it! Simple and clean. 🎯
