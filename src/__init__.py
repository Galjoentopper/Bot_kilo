"""
Bot Kilo - Cryptocurrency Trading Bot
====================================

A sophisticated cryptocurrency trading bot with machine learning and reinforcement learning capabilities.

Modules:
- data_pipeline: Data collection, preprocessing, and feature engineering
- models: Machine learning models (GRU, LightGBM, PPO)
- rl_env: Reinforcement learning trading environment
- backtesting: Backtesting framework
- notifier: Notification system (Telegram)
- utils: Utility functions and logging
- config: Configuration management
"""

__version__ = "1.0.0"
__author__ = "Bot Kilo Team"
__email__ = "contact@botkilo.com"

# Import main components for easy access
try:
    from .data_pipeline.loader import DataLoader
    from .data_pipeline.features import FeatureEngine
    from .data_pipeline.preprocess import DataPreprocessor
    from .models.gru_trainer import GRUTrainer
    from .models.lgbm_trainer import LightGBMTrainer
    from .models.ppo_trainer import PPOTrainer
    from .rl_env.trading_env import TradingEnvironment
    from .backtesting.backtest import Backtester
    from .notifier.telegram import TelegramNotifier
    from .utils.logger import setup_logging, TradingBotLogger
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}", ImportWarning)

__all__ = [
    "DataLoader",
    "FeatureEngine", 
    "DataPreprocessor",
    "GRUTrainer",
    "LightGBMTrainer",
    "PPOTrainer",
    "TradingEnvironment",
    "Backtester",
    "TelegramNotifier",
    "setup_logging",
    "TradingBotLogger",
]
