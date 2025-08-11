"""
Enhanced Training Script for Per-Symbol Models
==============================================

Orchestrates the training of separate models for each symbol (GRU, LightGBM, PPO).
Creates per-symbol models with metadata for feature consistency.
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
    parser = argparse.ArgumentParser(description='Crypto Trading Bot Trainer - Per Symbol Models')
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
    parser.add_argument('--symbol', type=str, default=None,
                       help='Train models for specific symbol only')
    
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
    logger.info("CRYPTO TRADING BOT - PER-SYMBOL MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Training model(s): {args.model}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "metadata"), exist_ok=True)
    
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
        
        # Filter symbols if specific symbol requested
        if args.symbol:
            if args.symbol in available_symbols:
                available_symbols = [args.symbol]
                logger.info(f"Training models for single symbol: {args.symbol}")
            else:
                logger.error(f"Symbol {args.symbol} not available in data!")
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
        
        # Create metadata directory for feature tracking
        os.makedirs(os.path.join(args.output_dir, "metadata"), exist_ok=True)
        
        # Train models for each symbol individually
        experiment_name = args.experiment_name or f"per_symbol_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for symbol in available_symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"TRAINING MODELS FOR {symbol}")
            logger.info(f"{'='*50}")
            
            try:
                # Process symbol data
                symbol_data = processed_data[symbol]
                logger.info(f"Processing features for {symbol}...")
                
                # Get feature names
                feature_names = feature_engine.get_feature_names(symbol_data)
                logger.info(f"Generated {len(feature_names)} features for {symbol}")
                
                # Comprehensive NaN cleaning
                symbol_data = clean_features_robust(symbol_data, feature_names, logger)
                
                # Data preprocessing - create separate preprocessor for each symbol
                preprocessor = DataPreprocessor()
                
                # Prepare features and targets
                X = symbol_data[feature_names].copy()
                
                # Remove any remaining NaN/inf values
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.ffill().bfill().fillna(0)
                
                # Prepare target variable (future returns)
                y = preprocessor.prepare_target_variable(symbol_data, target_type="return", horizon=1)
                
                # Align X and y
                min_len = min(len(X), len(y))
                X = X.iloc[:min_len]
                y = y[:min_len]
                
                logger.info(f"Training data shape for {symbol}: X={X.shape}, y={y.shape}")
                
                # Validate data quality
                if X.shape[0] < 100:
                    logger.warning(f"Insufficient data for {symbol}: {X.shape[0]} samples. Skipping.")
                    continue
                
                # Check for NaN/inf in final datasets
                if X.isna().any().any() or np.isinf(X.values).any():
                    logger.warning(f"Still have NaN/inf values in {symbol} features after cleaning")
                
                if np.isnan(y).any() or np.isinf(y).any():
                    logger.warning(f"NaN/inf values in {symbol} targets")
                    y = np.nan_to_num(y)
                
                # Save feature metadata for this symbol
                save_feature_metadata(symbol, feature_names, X.columns.tolist(), args.output_dir, logger)
                
                # Save preprocessor for this symbol
                import pickle
                preprocessor_path = os.path.join(args.output_dir, f"preprocessor_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                with open(preprocessor_path, 'wb') as f:
                    pickle.dump(preprocessor, f)
                logger.info(f"Saved preprocessor for {symbol}: {preprocessor_path}")
                
                # Train models for this symbol
                if args.model in ['gru', 'all']:
                    train_gru_model_for_symbol(config, symbol, X, y, preprocessor, args.output_dir, 
                                             experiment_name, logger, bot_logger, notifier)
                
                if args.model in ['lightgbm', 'all']:
                    train_lightgbm_model_for_symbol(config, symbol, X, y, args.output_dir, 
                                                  experiment_name, logger, bot_logger, notifier)
                
                if args.model in ['ppo', 'all']:
                    train_ppo_model_for_symbol(config, symbol, symbol_data, args.output_dir, 
                                             experiment_name, logger, bot_logger, notifier)
                
            except Exception as e:
                logger.error(f"Failed to train models for {symbol}: {e}")
                continue
        
        logger.info("=" * 60)
        logger.info("PER-SYMBOL TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        if notifier and notifier.enabled:
            notifier.send_message_sync("✅ Per-symbol model training completed successfully!")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        bot_logger.log_error("TRAINING_FAILURE", str(e))
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(f"❌ Training failed: {str(e)}")
        
        raise

def clean_features_robust(df: pd.DataFrame, feature_names: list, logger) -> pd.DataFrame:
    """Robust feature cleaning with multiple strategies."""
    logger.info("Applying robust feature cleaning...")
    
    # Make a copy
    cleaned_df = df.copy()
    
    for col in feature_names:
        if col in cleaned_df.columns:
            original_nans = cleaned_df[col].isna().sum()
            
            if original_nans > 0:
                # Strategy 1: Forward fill
                cleaned_df[col] = cleaned_df[col].ffill()
                
                # Strategy 2: Backward fill
                cleaned_df[col] = cleaned_df[col].bfill()
                
                # Strategy 3: Mean fill
                if cleaned_df[col].isna().any():
                    mean_val = cleaned_df[col].mean()
                    if pd.isna(mean_val):
                        mean_val = 0.0
                    cleaned_df[col].fillna(mean_val, inplace=True)
                
                remaining_nans = cleaned_df[col].isna().sum()
                if remaining_nans > 0:
                    logger.warning(f"Column {col}: {original_nans} -> {remaining_nans} NaN values after cleaning")
    
    # Replace infinite values
    cleaned_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cleaned_df.fillna(0, inplace=True)
    
    # Final check
    total_nans = cleaned_df[feature_names].isna().sum().sum()
    if total_nans > 0:
        logger.warning(f"Still have {total_nans} NaN values after robust cleaning")
    
    return cleaned_df

def save_feature_metadata(symbol: str, feature_names: list, feature_columns: list, output_dir: str, logger):
    """Save feature metadata for this symbol."""
    import json
    metadata = {
        'symbol': symbol,
        'feature_names': feature_names,
        'feature_columns': feature_columns,
        'num_features': len(feature_names),
        'created_at': datetime.now().isoformat(),
        'feature_order': {name: idx for idx, name in enumerate(feature_names)}
    }
    
    metadata_path = os.path.join(output_dir, "metadata", f"features_{symbol}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved feature metadata for {symbol}: {metadata_path}")

def train_gru_model_for_symbol(
    config: Dict[str, Any],
    symbol: str,
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: Any,
    output_dir: str,
    experiment_name: str,
    logger: Any,
    bot_logger: Any,
    notifier: Optional[Any]
) -> None:
    """Train GRU model for specific symbol."""
    logger.info(f"Training GRU model for {symbol}...")
    
    try:
        # Import GRU trainer (lazy import)
        try:
            from src.models.gru_trainer import GRUTrainer
        except ImportError as e:
            logger.error("Failed to import GRUTrainer. Please install required dependencies.")
            raise ImportError("GRU model dependencies not installed. Run 'pip install -e .' to install all dependencies.") from e
        
        # Prepare sequences for GRU
        sequence_length = config.get('models', {}).get('gru', {}).get('sequence_length', 20)
        
        # Scale features using the symbol-specific preprocessor
        X_scaled = preprocessor.fit_transform(X)
        
        # Create sequences
        X_sequences, y_sequences = preprocessor.create_sequences(
            X_scaled, y, sequence_length=sequence_length
        )
        
        if X_sequences.size == 0:
            logger.warning(f"No sequences created for GRU training for {symbol}")
            return
        
        # Split sequences
        train_size = int(len(X_sequences) * 0.7)
        val_size = int(len(X_sequences) * 0.15)
        
        X_train, y_train = X_sequences[:train_size], y_sequences[:train_size]
        X_val, y_val = X_sequences[train_size:train_size+val_size], y_sequences[train_size:train_size+val_size]
        X_test, y_test = X_sequences[train_size+val_size:], y_sequences[train_size+val_size:]
        
        # Initialize and train GRU
        gru_trainer = GRUTrainer(config)
        results = gru_trainer.train(X_train, y_train, X_val, y_val, f"{experiment_name}_{symbol}")
        
        # Evaluate on test set
        test_metrics = gru_trainer.evaluate(X_test, y_test)
        
        # Save model with symbol name
        model_path = os.path.join(output_dir, f"gru_model_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        gru_trainer.save_model(model_path)
        
        logger.info(f"GRU model for {symbol} saved to {model_path}")
        logger.info(f"GRU test metrics for {symbol}: {test_metrics}")
        
        bot_logger.log_model_update(f"GRU_{symbol}", "TRAINED", test_metrics)
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(
                f"🎓 GRU model trained for {symbol}!\n"
                f"RMSE: {test_metrics.get('rmse', 0):.6f}\n"
                f"R²: {test_metrics.get('r2', 0):.4f}\n"
                f"Directional Accuracy: {test_metrics.get('directional_accuracy', 0):.2%}"
            )
    
    except Exception as e:
        logger.error(f"GRU training failed for {symbol}: {e}")
        raise

def train_lightgbm_model_for_symbol(
    config: Dict[str, Any],
    symbol: str,
    X: pd.DataFrame,
    y: np.ndarray,
    output_dir: str,
    experiment_name: str,
    logger: Any,
    bot_logger: Any,
    notifier: Optional[Any]
) -> None:
    """Train LightGBM model for specific symbol."""
    logger.info(f"Training LightGBM model for {symbol}...")
    
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
        
        results = lgbm_trainer.train(X_train, y_train, X_val, y_val, experiment_name=f"{experiment_name}_{symbol}")
        
        # Evaluate
        test_metrics = lgbm_trainer.evaluate(X_val, y_val)
        
        # Save model with symbol name
        model_path = os.path.join(output_dir, f"lightgbm_model_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        lgbm_trainer.save_model(model_path)
        
        logger.info(f"LightGBM model for {symbol} saved to {model_path}")
        logger.info(f"LightGBM test metrics for {symbol}: {test_metrics}")
        
        # Log feature importance
        importance_df = lgbm_trainer.get_feature_importance(top_n=10)
        logger.info(f"Top 10 important features for {symbol}:")
        for _, row in importance_df.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        bot_logger.log_model_update(f"LightGBM_{symbol}", "TRAINED", test_metrics)
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(
                f"🎓 LightGBM model trained for {symbol}!\n"
                f"RMSE: {test_metrics.get('rmse', 0):.6f}\n"
                f"R²: {test_metrics.get('r2', 0):.4f}\n"
                f"Directional Accuracy: {test_metrics.get('directional_accuracy', 0):.2%}"
            )
    
    except Exception as e:
        logger.error(f"LightGBM training failed for {symbol}: {e}")
        raise

def train_ppo_model_for_symbol(
    config: Dict[str, Any],
    symbol: str,
    processed_data: pd.DataFrame,
    output_dir: str,
    experiment_name: str,
    logger: Any,
    bot_logger: Any,
    notifier: Optional[Any]
) -> None:
    """Train PPO model for specific symbol."""
    logger.info(f"Training PPO model for {symbol}...")
    
    try:
        # Import PPO trainer (lazy import)
        try:
            from src.models.ppo_trainer import PPOTrainer
        except ImportError as e:
            logger.error("Failed to import PPOTrainer. Please install required dependencies.")
            raise ImportError("PPO model dependencies not installed. Run 'pip install -e .' to install all dependencies.") from e
        
        # Split data for training and evaluation
        split_idx = int(len(processed_data) * 0.8)
        train_data = processed_data.iloc[:split_idx]
        eval_data = processed_data.iloc[split_idx:]
        
        # Initialize PPO trainer
        ppo_trainer = PPOTrainer(config)
        
        # Train PPO agent
        total_timesteps = config.get('models', {}).get('ppo', {}).get('total_timesteps', 100000)
        results = ppo_trainer.train(train_data, eval_data, total_timesteps, f"{experiment_name}_{symbol}")
        
        # Evaluate
        eval_results = ppo_trainer.evaluate(eval_data, n_episodes=10)
        
        # Save model with symbol name
        model_path = os.path.join(output_dir, f"ppo_model_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        ppo_trainer.save_model(model_path)
        
        logger.info(f"PPO model for {symbol} saved to {model_path}")
        logger.info(f"PPO evaluation results for {symbol}: {eval_results}")
        
        bot_logger.log_model_update(f"PPO_{symbol}", "TRAINED", {
            'mean_reward': eval_results.get('mean_episode_reward', 0),
            'mean_return': eval_results.get('mean_total_return', 0),
            'sharpe_ratio': eval_results.get('mean_sharpe_ratio', 0)
        })
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(
                f"🎓 PPO model trained for {symbol}!\n"
                f"Mean Episode Reward: {eval_results.get('mean_episode_reward', 0):.4f}\n"
                f"Mean Total Return: {eval_results.get('mean_total_return', 0):.2%}\n"
                f"Mean Sharpe Ratio: {eval_results.get('mean_sharpe_ratio', 0):.2f}"
            )
    
    except Exception as e:
        logger.error(f"PPO training failed for {symbol}: {e}")
        raise

if __name__ == "__main__":
    main()