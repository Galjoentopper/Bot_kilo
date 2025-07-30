"""
Main Training Script
===================

Orchestrates the training of all models (GRU, LightGBM, PPO).
Optimized for GPU training on Paperspace Gradient.
"""

import sys
import os
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Import from installed package
from src.utils.logger import setup_logging, TradingBotLogger
from src.utils.config import prepare_feature_config
from src.data_pipeline.loader import DataLoader
from src.data_pipeline.features import FeatureEngine
from src.data_pipeline.preprocess import DataPreprocessor
from src.rl_env.trading_env import TradingEnvironment
from src.notifier.telegram import TelegramNotifier

def load_config(config_path: str = "src/config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def main() -> None:
    """Main training orchestration function."""
    parser = argparse.ArgumentParser(description='Crypto Trading Bot Trainer')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, choices=['gru', 'lightgbm', 'ppo', 'all'],
                       default='all', help='Model to train')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory path')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for trained models')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration. Exiting.")
        return
    
    # Setup logging
    logger = setup_logging(config)
    bot_logger = TradingBotLogger()
    
    logger.info("=" * 60)
    logger.info("CRYPTO TRADING BOT - MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Training model(s): {args.model}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize notification system
    try:
        notifier = TelegramNotifier.from_config(config)
        if notifier.enabled:
            notifier.send_message_sync("🚀 Starting model training...")
    except Exception as e:
        logger.warning(f"Failed to initialize notifications: {e}")
        notifier = None
    
    try:
        # Load data
        logger.info("Loading market data...")
        data_loader = DataLoader(args.data_dir)
        
        # Check data availability
        availability = data_loader.check_data_availability()
        available_symbols = [symbol for symbol, available in availability.items() if available]
        
        if not available_symbols:
            logger.error("No data available for training!")
            return
        
        logger.info(f"Available symbols: {available_symbols}")
        
        # Load data for available symbols
        data_dict = data_loader.load_multiple_symbols(available_symbols)
        
        if not data_dict:
            logger.error("Failed to load any data!")
            return
        
        # Feature engineering
        logger.info("Generating features...")
        feature_config = prepare_feature_config(config)
        feature_engine = FeatureEngine(feature_config)
        
        processed_data = {}
        for symbol, df in data_dict.items():
            logger.info(f"Processing features for {symbol}...")
            features_df = feature_engine.generate_all_features(df)
            processed_data[symbol] = features_df
            logger.info(f"Generated {len(feature_engine.get_feature_names(features_df))} features for {symbol}")
        
        # Data preprocessing
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor()
        
        # Use the first symbol's data for training (can be extended for multi-asset)
        main_symbol = available_symbols[0]
        main_data = processed_data[main_symbol]
        
        logger.info(f"Using {main_symbol} as primary training asset")
        
        # Prepare features and targets
        feature_names = feature_engine.get_feature_names(main_data)
        X = main_data[feature_names].dropna()
        
        # Prepare target variable (future returns)
        y = preprocessor.prepare_target_variable(main_data, target_type="return", horizon=1)
        
        # Align X and y
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y[:min_len]
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Train/validation/test split
        train_df, val_df, test_df = data_loader.create_train_test_split(main_data)
        
        # Train models based on selection
        experiment_name = args.experiment_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if args.model in ['gru', 'all']:
            train_gru_model(config, X, y, preprocessor, args.output_dir, experiment_name, logger, bot_logger, notifier)
        
        if args.model in ['lightgbm', 'all']:
            train_lightgbm_model(config, X, y, args.output_dir, experiment_name, logger, bot_logger, notifier)
        
        if args.model in ['ppo', 'all']:
            train_ppo_model(config, processed_data, args.output_dir, experiment_name, logger, bot_logger, notifier)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        if notifier and notifier.enabled:
            notifier.send_message_sync("✅ Model training completed successfully!")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        bot_logger.log_error("TRAINING_FAILURE", str(e))
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(f"❌ Training failed: {str(e)}")
        
        raise

def train_gru_model(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: Any,
    output_dir: str,
    experiment_name: str,
    logger: Any,
    bot_logger: Any,
    notifier: Optional[Any]
) -> None:
    """Train GRU model."""
    logger.info("Training GRU model...")
    
    try:
        # Import GRU trainer (lazy import)
        try:
            from src.models.gru_trainer import GRUTrainer
        except ImportError as e:
            logger.error("Failed to import GRUTrainer. Please install required dependencies.")
            raise ImportError("GRU model dependencies not installed. Run 'pip install -e .' to install all dependencies.") from e
        
        # Prepare sequences for GRU
        sequence_length = config.get('models', {}).get('gru', {}).get('sequence_length', 20)
        
        # Scale features
        X_scaled = preprocessor.fit_transform(X)
        
        # Create sequences
        X_sequences, y_sequences = preprocessor.create_sequences(
            X_scaled, y, sequence_length=sequence_length
        )
        
        if X_sequences.size == 0:
            logger.warning("No sequences created for GRU training")
            return
        
        # Split sequences manually since split_sequences method doesn't exist
        train_size = int(len(X_sequences) * 0.7)
        val_size = int(len(X_sequences) * 0.15)
        
        X_train, y_train = X_sequences[:train_size], y_sequences[:train_size]
        X_val, y_val = X_sequences[train_size:train_size+val_size], y_sequences[train_size:train_size+val_size]
        X_test, y_test = X_sequences[train_size+val_size:], y_sequences[train_size+val_size:]
        
        # Initialize and train GRU
        gru_trainer = GRUTrainer(config)
        results = gru_trainer.train(X_train, y_train, X_val, y_val, experiment_name)
        
        # Evaluate on test set
        test_metrics = gru_trainer.evaluate(X_test, y_test)
        
        # Save model
        model_path = os.path.join(output_dir, f"gru_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        gru_trainer.save_model(model_path)
        
        logger.info(f"GRU model saved to {model_path}")
        logger.info(f"GRU test metrics: {test_metrics}")
        
        bot_logger.log_model_update("GRU", "TRAINED", test_metrics)
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(
                f"🎓 GRU model trained successfully!\n"
                f"RMSE: {test_metrics.get('rmse', 0):.6f}\n"
                f"R²: {test_metrics.get('r2', 0):.4f}\n"
                f"Directional Accuracy: {test_metrics.get('directional_accuracy', 0):.2%}"
            )
    
    except Exception as e:
        logger.error(f"GRU training failed: {e}")
        raise

def train_lightgbm_model(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    output_dir: str,
    experiment_name: str,
    logger: Any,
    bot_logger: Any,
    notifier: Optional[Any]
) -> None:
    """Train LightGBM model."""
    logger.info("Training LightGBM model...")
    
    try:
        # Import LightGBM trainer (lazy import)
        try:
            from src.models.lgbm_trainer import LightGBMTrainer
        except ImportError as e:
            logger.error("Failed to import LightGBMTrainer. Please install required dependencies.")
            raise ImportError("LightGBM model dependencies not installed. Run 'pip install -e .' to install all dependencies.") from e
        
        # Initialize and train LightGBM
        lgbm_trainer = LightGBMTrainer(config, task_type="regression")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        results = lgbm_trainer.train(X_train, y_train, X_val, y_val, experiment_name=experiment_name)
        
        # Evaluate
        test_metrics = lgbm_trainer.evaluate(X_val, y_val)
        
        # Save model
        model_path = os.path.join(output_dir, f"lightgbm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        lgbm_trainer.save_model(model_path)
        
        logger.info(f"LightGBM model saved to {model_path}")
        logger.info(f"LightGBM test metrics: {test_metrics}")
        
        # Log feature importance
        importance_df = lgbm_trainer.get_feature_importance(top_n=10)
        logger.info("Top 10 important features:")
        for _, row in importance_df.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        bot_logger.log_model_update("LightGBM", "TRAINED", test_metrics)
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(
                f"🎓 LightGBM model trained successfully!\n"
                f"RMSE: {test_metrics.get('rmse', 0):.6f}\n"
                f"R²: {test_metrics.get('r2', 0):.4f}\n"
                f"Directional Accuracy: {test_metrics.get('directional_accuracy', 0):.2%}"
            )
    
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        raise

def train_ppo_model(
    config: Dict[str, Any],
    processed_data: Dict[str, pd.DataFrame],
    output_dir: str,
    experiment_name: str,
    logger: Any,
    bot_logger: Any,
    notifier: Optional[Any]
) -> None:
    """Train PPO model."""
    logger.info("Training PPO model...")
    
    try:
        # Import PPO trainer (lazy import)
        try:
            from src.models.ppo_trainer import PPOTrainer
        except ImportError as e:
            logger.error("Failed to import PPOTrainer. Please install required dependencies.")
            raise ImportError("PPO model dependencies not installed. Run 'pip install -e .' to install all dependencies.") from e
        
        # Use the first symbol's data for RL training
        main_symbol = list(processed_data.keys())[0]
        main_data = processed_data[main_symbol]
        
        # Split data for training and evaluation
        split_idx = int(len(main_data) * 0.8)
        train_data = main_data.iloc[:split_idx]
        eval_data = main_data.iloc[split_idx:]
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(config)
        
        # Train PPO agent
        total_timesteps = config.get('models', {}).get('ppo', {}).get('total_timesteps', 100000)
        results = ppo_trainer.train(train_data, eval_data, total_timesteps, experiment_name)
        
        # Evaluate
        eval_results = ppo_trainer.evaluate(eval_data, n_episodes=10)
        
        # Save model
        model_path = os.path.join(output_dir, f"ppo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        ppo_trainer.save_model(model_path)
        
        logger.info(f"PPO model saved to {model_path}")
        logger.info(f"PPO evaluation results: {eval_results}")
        
        bot_logger.log_model_update("PPO", "TRAINED", {
            'mean_reward': eval_results.get('mean_episode_reward', 0),
            'mean_return': eval_results.get('mean_total_return', 0),
            'sharpe_ratio': eval_results.get('mean_sharpe_ratio', 0)
        })
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(
                f"🎓 PPO model trained successfully!\n"
                f"Mean Episode Reward: {eval_results.get('mean_episode_reward', 0):.4f}\n"
                f"Mean Total Return: {eval_results.get('mean_total_return', 0):.2%}\n"
                f"Mean Sharpe Ratio: {eval_results.get('mean_sharpe_ratio', 0):.2f}"
            )
    
    except Exception as e:
        logger.error(f"PPO training failed: {e}")
        raise

if __name__ == "__main__":
    main()