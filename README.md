# Crypto Short-Term Trading Bot

A comprehensive, modular trading bot for short-term crypto strategies using hybrid ML approaches (GRU, LightGBM, PPO). Designed for 15-minute OHLCV data with Bitvavo exchange integration and Telegram notifications.

## 🚀 Features

- **Hybrid ML Architecture**: GRU for sequence prediction, LightGBM for feature-based predictions, PPO for trade execution
- **Comprehensive Data Pipeline**: 15-min interval crypto data with advanced feature engineering
- **Advanced Feature Engineering**: 50+ technical indicators, volatility metrics, momentum features
- **Reinforcement Learning**: PPO agent for intelligent trade execution and position sizing
- **Paper & Live Trading**: Safe testing environment with real exchange integration
- **Real-time Notifications**: Telegram alerts for trades, portfolio updates, and system status
- **Robust Backtesting**: Walk-forward validation with comprehensive performance metrics
- **GPU Optimized**: Ready for Paperspace Gradient and other GPU platforms
- **Modular Design**: Easy to extend and customize

## 📊 Performance Targets

- **Sharpe Ratio**: > 1.0 in backtests
- **Telegram Notifications**: < 5s latency
- **Data Coverage**: BTCEUR, SOLEUR, ADAEUR, XRPEUR, ETHEUR
- **Interval**: 15-minute OHLCV data

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │   ML Models     │    │ Trading Engine  │
│                 │    │                 │    │                 │
│ • Data Loader   │───▶│ • GRU Trainer   │───▶│ • Paper Trader  │
│ • Feature Eng.  │    │ • LightGBM      │    │ • Backtester    │
│ • Preprocessor  │    │ • PPO Agent     │    │ • Risk Mgmt     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Storage  │    │   Experiment    │    │  Notifications  │
│                 │    │   Tracking      │    │                 │
│ • SQLite DBs    │    │ • MLflow        │    │ • Telegram Bot  │
│ • 190K+ records │    │ • Model Versioning   │ • Trade Alerts  │
│ • 5 Crypto Pairs│    │ • Performance   │    │ • System Status │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- GPU support (CUDA) recommended for training
- Bitvavo API credentials
- Telegram Bot Token (optional)

### Quick Setup

1. **Clone and Install**:
```bash
git clone <repository-url>
cd Bot_kilo
pip install -r requirements.txt
```

2. **Environment Configuration**:
```bash
# Create .env file
cp .env.example .env

# Add your credentials
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
BITVAVO_API_KEY=your_bitvavo_api_key
BITVAVO_API_SECRET=your_bitvavo_api_secret
```

3. **Configuration**:
```bash
# Edit configuration
nano src/config/config.yaml
```

## 🚀 Quick Start

### 1. Data Collection (Already Available)
Your existing data collection system has gathered 190K+ records:
```bash
# Data is already available in ./data/
ls data/
# btceur_15m.db  etheur_15m.db  soleur_15m.db  adaeur_15m.db  xrpeur_15m.db
```

### 2. Train Models
```bash
# Train all models (GRU, LightGBM, PPO)
python scripts/trainer.py --model all

# Train specific model
python scripts/trainer.py --model gru
python scripts/trainer.py --model lightgbm
python scripts/trainer.py --model ppo
```

### 3. Run Paper Trading
```bash
# Start paper trading with 15-minute intervals
python scripts/trader.py --interval 15

# Custom configuration
python scripts/trader.py --config custom_config.yaml --models-dir ./trained_models
```

### 4. Backtesting
```python
from src.backtesting.backtest import Backtester
from src.data_pipeline.loader import DataLoader

# Load data
loader = DataLoader('./data')
data = loader.load_symbol_data('BTCEUR')

# Run backtest
backtester = Backtester(initial_capital=10000)
results = backtester.run_backtest(data, your_strategy_function)

# View results
print(backtester.generate_report(results))
backtester.plot_results(results)
```

## 📈 Model Details

### GRU Sequence Model
- **Purpose**: Short-term price prediction using historical sequences
- **Architecture**: 2-layer GRU with dropout, 128 hidden units
- **Input**: 20-period sequences of engineered features
- **Output**: Next period price/return prediction
- **Optimization**: GPU-accelerated training with mixed precision

### LightGBM Feature Model
- **Purpose**: Feature-based prediction refinement
- **Features**: 50+ technical indicators and market features
- **Optimization**: Hyperparameter tuning with cross-validation
- **Output**: Refined predictions with feature importance

### PPO Reinforcement Learning Agent
- **Purpose**: Intelligent trade execution and position sizing
- **Environment**: Realistic trading simulation with fees and slippage
- **Action Space**: Position changes (-1 to 1)
- **Reward**: Risk-adjusted returns with transaction cost penalties

## 🔧 Configuration

### Main Configuration (`src/config/config.yaml`)

```yaml
# Data Configuration
data:
  symbols: ["BTCEUR", "SOLEUR", "ADAEUR", "XRPEUR", "ETHEUR"]
  interval: "15m"
  data_dir: "./data"

# Model Configuration
models:
  gru:
    sequence_length: 20
    hidden_size: 128
    learning_rate: 0.001
    epochs: 100
  
  lightgbm:
    num_leaves: 31
    max_depth: 6
    learning_rate: 0.1
  
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    gamma: 0.99

# Trading Configuration
trading:
  initial_balance: 10000.0
  max_position_size: 0.1
  transaction_fee: 0.001

# Notifications
notifications:
  telegram:
    enabled: true
    # Set via environment variables
```

## 📊 Features Engineering

The system generates 50+ features including:

### Technical Indicators
- **Moving Averages**: SMA, EMA (5, 10, 20, 50 periods)
- **Momentum**: RSI, MACD, Stochastic, CCI
- **Volatility**: Bollinger Bands, ATR, Historical Volatility
- **Volume**: OBV, Volume ratios, VWAP

### Price Features
- **Returns**: 1, 5, 15-period returns and log returns
- **Price Patterns**: OHLC relationships, candlestick patterns
- **Volatility**: Rolling standard deviation, Parkinson volatility

### Time Features
- **Cyclical**: Hour, day of week, month (sine/cosine encoded)
- **Market Sessions**: Weekend/night indicators

### Custom Features
- **Signal Combinations**: MA crosses, RSI signals, MACD signals
- **Relative Positioning**: Price vs. moving averages
- **Volume Confirmation**: High/low volume periods

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_data_pipeline.py
pytest tests/test_models.py
pytest tests/test_backtesting.py
```

## 📱 Telegram Notifications

The bot sends real-time notifications for:

- **Trade Executions**: Buy/sell orders with details
- **Portfolio Updates**: Daily P&L and performance
- **System Status**: Bot start/stop, errors, model updates
- **Alerts**: Risk management triggers, system errors

### Setup Telegram Bot

1. Create bot with [@BotFather](https://t.me/botfather)
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Add credentials to `.env` file

## 🔒 Security & Risk Management

### API Security
- Credentials stored in environment variables
- No hardcoded secrets in code
- Secure API key management

### Risk Management
- Maximum position size limits (10% default)
- Stop-loss and take-profit levels
- Drawdown monitoring
- Transaction cost accounting

### Error Handling
- Comprehensive logging system
- Automatic error notifications
- Graceful failure recovery
- Circuit breaker patterns

## 📈 Performance Monitoring

### MLflow Integration
- Experiment tracking
- Model versioning
- Performance metrics logging
- Hyperparameter optimization

### Logging System
- Structured logging with multiple levels
- Performance metrics tracking
- Trade execution logs
- Error tracking and alerts

## 🚀 Deployment

### Local Development
```bash
# Development mode
python scripts/trainer.py --model all
python scripts/trader.py --interval 15
```

### Paperspace Gradient (GPU Training)
```bash
# Optimized for GPU training
# Mixed precision training enabled
# Efficient data loading with multiple workers
```

### Docker Deployment
```bash
# Build container
docker build -t crypto-trading-bot .

# Run container
docker run -d --name trading-bot \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  crypto-trading-bot
```

## 📊 Performance Metrics

The system tracks comprehensive metrics:

- **Returns**: Total, annualized, risk-adjusted
- **Risk**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Trading**: Win rate, profit factor, number of trades
- **Costs**: Transaction fees, slippage costs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## 🆘 Support

- **Documentation**: Check the `/docs` folder for detailed guides
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join our community discussions

## 🎯 Roadmap

- [ ] Multi-exchange support (Binance, Coinbase)
- [ ] Advanced portfolio optimization
- [ ] Sentiment analysis integration
- [ ] Web dashboard interface
- [ ] Mobile app notifications
- [ ] Advanced risk management strategies

---

**Built with ❤️ for the crypto trading community**

*Ready for GPU acceleration on Paperspace Gradient and other cloud platforms*