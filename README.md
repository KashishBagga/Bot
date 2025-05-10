# Trading Bot

A modular, extensible trading bot designed for technical analysis and automated trading with the Fyers API.

## Features

- 📈 Multiple technical trading strategies
- 🤖 Real-time and backtest modes
- 📊 Visualization of strategies and signals
- 📱 Telegram notifications for trade signals
- 🔄 Easily extensible architecture for adding new strategies

## Getting Started

### Prerequisites

- Python 3.9+
- Fyers Trading Account
- API credentials from Fyers

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. Run the setup script to create directories and install dependencies:
```bash
python setup.py
```

3. Configure your environment variables by copying the `.env.example` file:
```bash
cp .env.example .env
```

4. Edit the `.env` file with your Fyers API credentials and other settings:
```
# Fyers API Configuration
FYERS_CLIENT_ID=your_client_id
FYERS_SECRET_KEY=your_secret_key
FYERS_REDIRECT_URI=your_redirect_uri
FYERS_GRANT_TYPE=authorization_code
FYERS_RESPONSE_TYPE=code
FYERS_STATE=state
```

## Usage

### Testing Strategies

You can test strategies with sample data using the `test_strategy.py` script:

```bash
# List available strategies
python test_strategy.py --list

# Test a specific strategy with visualization
python test_strategy.py --strategy ema_crossover

# Test without visualization
python test_strategy.py --strategy supertrend_macd_rsi_ema --no-visual
```

### Running the Bot

The bot can run in different modes:

```bash
# Test mode (uses sample data)
python -m src.main --mode test

# Backtest mode (uses historical data)
python -m src.main --mode backtest --start-date 2023-01-01 --end-date 2023-12-31

# Real-time mode (trades with live market data)
python -m src.main --mode realtime
```

You can also specify which symbols to trade and which strategies to use:

```bash
python -m src.main --mode realtime --symbols NSE:NIFTY50-INDEX NSE:BANKNIFTY-INDEX --strategies ema_crossover supertrend_macd_rsi_ema
```

### Adding a New Strategy

1. Create a new strategy file in `src/strategies/`:

```bash
python migrate_strategy.py --create my_new_strategy
```

2. Edit the created files:
   - `src/strategies/my_new_strategy.py`: Implement your strategy logic
   - `src/models/schema/my_new_strategy.py`: Define the database schema for your strategy

## Project Structure

```
trading-bot/
├── src/
│   ├── api/             # API client implementations
│   ├── core/            # Core functionality and base classes
│   │   ├── strategy.py  # Base Strategy class
│   │   └── indicators.py # Technical indicators
│   ├── models/          # Database models
│   │   ├── database.py  # Database interface
│   │   └── schema/      # Schema definitions for strategies
│   ├── services/        # External services (Telegram, etc.)
│   └── strategies/      # Trading strategy implementations
├── test_strategy.py     # Strategy testing script
├── migrate_strategy.py  # Tool for migrating/creating strategies
├── setup.py             # Setup script
└── .env.example         # Example environment configuration
```

## Available Strategies

1. **EMA Crossover**
   - Uses exponential moving average crossovers to generate signals
   - Configurable fast and slow EMA periods

2. **Supertrend with MACD, RSI, and EMA**
   - Combines multiple indicators for higher-confidence signals
   - Uses Supertrend for trend direction
   - Confirms with MACD, RSI, and EMA

## Backtesting with Real Fyers Data

To backtest your trading strategies using real market data from Fyers, follow these steps:

### Prerequisites

1. Make sure you have a Fyers account with API access
2. Set up your API credentials in the `.env` file:
   ```
   FYERS_REDIRECT_URI="https://trade.fyers.in/"
   FYERS_CLIENT_ID="your_client_id"
   FYERS_SECRET_KEY="your_secret_key"
   FYERS_GRANT_TYPE="authorization_code"
   FYERS_RESPONSE_TYPE="code"
   FYERS_STATE="sample"
   ```

3. Generate an auth code and access token:
   ```
   python test_fyers.py
   ```
   Follow the instructions to authorize the app and get an access token.

### Running Backtests

Use the `run_backtesting.py` script to test your strategies with real market data:

```bash
# Basic usage - test a specific strategy
python run_backtesting.py --strategy strategy_name

# Test with specific timeframe (days back)
python run_backtesting.py --strategy strategy_name --days 60

# Test with specific candle resolution (1, 5, 15, 30, 60, D)
python run_backtesting.py --strategy strategy_name --resolution 15

# Test without saving results to database
python run_backtesting.py --strategy strategy_name --no-save

# Test on specific symbols
python run_backtesting.py --strategy strategy_name --symbols NIFTY50,BANKNIFTY,RELIANCE

# Combined example
python run_backtesting.py --strategy supertrend_ema --days 60 --resolution 15
```

### Available Strategies

The following strategies can be backtested:

- `supertrend_ema`: Combines Supertrend indicator with EMA confirmation
- `strategy_range_breakout_volatility`: Identifies breakouts from a price range when volatility is high
- `insidebar_rsi`: Detects inside bar patterns with RSI confirmation
- `strategy_ema_crossover`: Generates signals based on EMA crossovers
- `strategy_donchian_breakout`: Identifies breakouts from Donchian channels
- `strategy_insidebar_bollinger`: Combines inside bar patterns with Bollinger Bands
- `strategy_breakout_rsi`: Detects price breakouts with RSI confirmation

### Backtesting Results

Results are saved in the SQLite database (`trading_signals.db`), with each strategy having its own dedicated table containing:
- Timestamp of each candle
- Symbol/index being analyzed
- Signal generated (BUY CALL, BUY PUT, NO TRADE)
- Price at the time of the signal
- Strategy-specific indicators and values
- Confidence level of the signal
- Other metadata like targets, stop-loss, etc.

The script also provides a summary of signal distribution at the end of the backtest.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Fyers API](https://fyers.in/api-module/) for trading interface
- [ta-lib](https://github.com/mrjbq7/ta-lib) for technical indicators
- [pandas](https://pandas.pydata.org/) for data analysis 