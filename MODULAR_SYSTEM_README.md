# 🚀 Modular Multi-Asset Trading System

A flexible, modular trading system that supports multiple markets including Indian stocks, cryptocurrency, and more.

## ✨ Features

- **Multi-Market Support**: Trade Indian stocks, crypto, and other markets
- **Modular Architecture**: Easy to extend with new markets and data providers
- **Risk Management**: Built-in position sizing and risk controls
- **Real-time Trading**: Live market data and execution
- **Strategy Engine**: Unified strategy system across all markets
- **Database Integration**: Persistent trade tracking and analytics

## 🏗️ Architecture

### Core Components

1. **Market Interface**: Abstract base class for all market implementations
2. **Market Factory**: Creates market instances based on type
3. **Data Providers**: Market-specific data sources (Fyers, Binance, etc.)
4. **Strategy Engine**: Unified signal generation across markets
5. **Trading System**: Core trading logic with risk management

### Directory Structure

```
src/
├── adapters/
│   ├── market_interface.py      # Core market abstractions
│   ├── market_factory.py        # Market factory
│   ├── data/                    # Data provider implementations
│   └── execution/               # Execution provider implementations
├── markets/
│   ├── indian/                  # Indian market implementation
│   ├── crypto/                  # Crypto market implementation
│   └── us/                      # US market implementation (future)
└── core/                        # Existing core components
```

## 🚀 Quick Start

### 1. Indian Stocks Trading

```python
from modular_trading_system import ModularTradingSystem
from src.adapters.market_interface import MarketType

# Create Indian stocks trading system
system = ModularTradingSystem(
    market_type=MarketType.INDIAN_STOCKS,
    initial_capital=50000.0,  # ₹50,000
    max_risk_per_trade=0.02,  # 2% risk per trade
    confidence_cutoff=25.0,
    symbols=['NSE:NIFTY50-INDEX', 'NSE:NIFTYBANK-INDEX']
)

# Start trading
system.start_trading()
```

### 2. Crypto Trading

```python
# Create crypto trading system
system = ModularTradingSystem(
    market_type=MarketType.CRYPTO,
    initial_capital=10000.0,  # $10,000
    max_risk_per_trade=0.02,  # 2% risk per trade
    confidence_cutoff=25.0,
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
)

# Start trading
system.start_trading()
```

### 3. Command Line Usage

```bash
# Trade Indian stocks
python modular_trading_system.py --market indian --capital 50000 --test

# Trade crypto
python modular_trading_system.py --market crypto --capital 10000 --test

# Run examples
python modular_trading_example.py
```

## 📊 Supported Markets

### Indian Stocks (NSE/BSE)
- **Currency**: INR
- **Trading Hours**: 9:15 AM - 3:30 PM IST
- **Symbols**: NSE:NIFTY50-INDEX, NSE:NIFTYBANK-INDEX, NSE:RELIANCE-EQ, etc.
- **Data Provider**: Fyers API
- **Lot Sizes**: 50 (Nifty), 25 (Bank Nifty), 1 (Stocks)

### Cryptocurrency
- **Currency**: USDT
- **Trading Hours**: 24/7
- **Symbols**: BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, etc.
- **Data Provider**: Binance API
- **Lot Sizes**: 0.001 (BTC), 0.01 (ETH), 1.0 (ADA)

## 🔧 Configuration

### Market Configuration

Each market has its own configuration:

```python
MarketConfig(
    market_type=MarketType.CRYPTO,
    timezone="UTC",
    trading_hours={"start": "00:00", "end": "23:59"},
    trading_days=[0, 1, 2, 3, 4, 5, 6],  # All days
    lot_sizes={"BTCUSDT": 0.001, "ETHUSDT": 0.01},
    tick_sizes={"BTCUSDT": 0.01, "ETHUSDT": 0.01},
    commission_rates={"BTCUSDT": 0.001, "ETHUSDT": 0.001},
    margin_requirements={"BTCUSDT": 0.05, "ETHUSDT": 0.05},
    currency="USDT"
)
```

### Risk Management

- **Position Sizing**: Based on risk percentage and confidence
- **Stop Loss**: Configurable percentage-based stops
- **Take Profit**: Configurable percentage-based targets
- **Time Stops**: Automatic exit after specified time
- **Exposure Limits**: Maximum portfolio exposure controls

## 🛠️ Extending the System

### Adding a New Market

1. **Create Market Implementation**:
```python
class NewMarket(MarketInterface):
    def __init__(self):
        config = MarketConfig(...)
        super().__init__(config)
    
    def is_market_open(self, timestamp=None):
        # Implement market hours logic
        pass
    
    # Implement other required methods
```

2. **Register with Factory**:
```python
MarketFactory.register_market(MarketType.NEW_MARKET, NewMarket)
```

3. **Create Data Provider**:
```python
class NewDataProvider(DataProviderInterface):
    def get_historical_data(self, symbol, start_date, end_date, resolution):
        # Implement data fetching
        pass
    
    # Implement other required methods
```

### Adding a New Data Provider

1. **Implement DataProviderInterface**
2. **Add to market-specific data provider selection**
3. **Handle authentication and rate limiting**

## 📈 Performance Monitoring

The system provides comprehensive performance tracking:

```python
summary = system.get_performance_summary()
print(f"Total P&L: {summary['total_pnl']}")
print(f"Win Rate: {summary['win_rate']:.1f}%")
print(f"Max Drawdown: {summary['max_drawdown']:.1f}%")
```

## 🔒 Security & Risk

- **Paper Trading**: Test strategies without real money
- **Position Limits**: Maximum positions per symbol
- **Capital Limits**: Maximum capital utilization
- **Error Handling**: Comprehensive error recovery
- **Transaction Rollback**: Database consistency guarantees

## 🚀 Future Enhancements

- **US Stock Market**: NYSE/NASDAQ support
- **Forex Trading**: Major currency pairs
- **Commodities**: Gold, oil, agricultural products
- **Options Trading**: Advanced derivatives support
- **Portfolio Management**: Multi-asset portfolio optimization
- **Machine Learning**: AI-powered signal generation

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions and support, please open an issue on GitHub.
