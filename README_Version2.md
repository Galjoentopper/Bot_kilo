# Crypto Short-Term Trading Bot

## Overview
A modular trading bot for short-term crypto strategies using GRU, LightGBM, and PPO reinforcement learning. Designed for 15-minute OHLCV data, with Bitvavo exchange integration and Telegram notifications.

## Features
- Data pipeline for 15-min interval crypto data
- Feature engineering (technical indicators, order book stats, etc.)
- GRU and LightGBM hybrid models for prediction
- PPO RL agent for trade execution
- Paper and live trading support
- Experiment tracking (MLflow)
- Modular design for easy extension
- Automated notifications (Telegram)

## Technologies
- Python 3.10+
- PyTorch, LightGBM, Stable Baselines3
- pandas, numpy
- bitvavo-python
- python-telegram-bot
- MLflow
- pytest
- Docker

## Setup
1. Clone repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Add your API keys in a `.env` file (never commit this file).
3. Configure settings in `src/config/config.yaml`.
4. Run data pipeline and training scripts from `scripts/`.

## Quickstart
1. Prepare data in `data/raw/`.
2. Train models:
   ```
   python scripts/trainer.py
   ```
3. Start paper trader:
   ```
   python scripts/trader.py
   ```

## Project Structure
See [plan.txt](plan.txt) for full architecture and development plan.

## Documentation
- All modules and scripts are documented with docstrings.
- See `notebooks/` for EDA and prototyping.

## Testing
Run tests with:
```
pytest tests/
```

## Contribution & License
Open to contributors. MIT License.
