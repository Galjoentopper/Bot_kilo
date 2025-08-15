"""
Enhanced Training Script for Per-Symbol Models
==============================================

Implements all improvements:
- Centralized dataset building with caching
- Time-series cross-validation with leakage guards
- Cost-aware metrics and optimization
- Unified model adapters
- Parallel training capability
- Unified artifact management
"""

import sys
import os
import argparse
import yaml
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import logging
from functools import partial

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import enhanced modules
from src.utils.logger import setup_logging, TradingBotLogger
from src.utils.config import prepare_feature_config
from src.data_pipeline.dataset_builder import DatasetBuilder
from src.utils.cross_validation import PurgedTimeSeriesSplit, create_time_series_splits
from src.utils.metrics import TradingMetrics, find_optimal_threshold
from src.utils.calibration import ProbabilityCalibrator, calibrate_probabilities_cv
from src.models.adapters import create_model_adapter
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


def train_symbol_models(
    symbol: str,
    args: argparse.Namespace,
    config: Dict[str, Any],
    dataset_builder: DatasetBuilder,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Train all models for a single symbol.
    
    Args:
        symbol: Trading symbol
        args: Command line arguments
        config: Configuration dictionary
        dataset_builder: DatasetBuilder instance
        logger: Logger instance
        
    Returns:
        Dictionary of training results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MODELS FOR {symbol}")
    logger.info(f"{'='*60}")
    
    results = {
        'symbol': symbol,
        'models': {},
        'errors': []
    }
    
    try:
        # Build dataset with caching
        logger.info(f"Building dataset for {symbol}...")
        X, y, timestamps, feature_names, metadata = dataset_builder.build_dataset(
            symbol=symbol,
            interval=args.interval,
            use_cache=args.cache,
            target_type=args.target_type,
            target_horizon=args.target_horizon
        )
        
        # Validate dataset
        is_valid, errors = dataset_builder.validate_dataset(X, y, metadata)
        if not is_valid:
            logger.error(f"Dataset validation failed for {symbol}: {errors}")
            results['errors'].extend(errors)
            return results
        
        logger.info(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Create time-series cross-validation splits
        cv_splitter = PurgedTimeSeriesSplit(
            n_splits=args.n_splits,
            gap=args.embargo,
            embargo=args.embargo
        )
        
        # Initialize metrics calculator
        metrics_calc = TradingMetrics(
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps
        )
        
        # Train each model type
        models_to_train = args.models if args.models != ['all'] else ['gru', 'lightgbm', 'ppo']
        
        for model_type in models_to_train:
            if model_type not in ['gru', 'lightgbm', 'ppo']:
                logger.warning(f"Unknown model type: {model_type}")
                continue
                
            logger.info(f"\nTraining {model_type.upper()} model for {symbol}...")
            
            try:
                # Create model adapter
                task_type = 'classification' if args.target_type == 'direction' else 'regression'
                adapter = create_model_adapter(model_type, config, task_type)
                
                # Cross-validation training
                cv_results = []
                calibrators = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
                    logger.info(f"Training fold {fold_idx + 1}/{args.n_splits}")
                    
                    # Train model
                    if model_type == 'ppo':
                        # PPO needs full DataFrame
                        fold_results = adapter.fit(
                            X=metadata.get('full_data', X),  # Pass full DataFrame if available
                            y=y,
                            train_idx=train_idx,
                            valid_idx=val_idx,
                            experiment_name=f"{args.experiment_name}_{symbol}_{model_type}_fold{fold_idx}"
                        )
                    else:
                        fold_results = adapter.fit(
                            X=X,
                            y=y,
                            train_idx=train_idx,
                            valid_idx=val_idx,
                            experiment_name=f"{args.experiment_name}_{symbol}_{model_type}_fold{fold_idx}"
                        )
                    
                    # Evaluate on validation set
                    X_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
                    y_val = y[val_idx]
                    
                    if model_type != 'ppo':
                        y_pred = adapter.predict(X_val)
                        
                        # Calculate metrics
                        if task_type == 'classification' and hasattr(adapter, 'predict_proba'):
                            y_prob = adapter.predict_proba(X_val)[:, 1]
                            
                            # Calibrate probabilities
                            calibrator = ProbabilityCalibrator(method='isotonic')
                            calibrator.fit(y_val, y_prob)
                            calibrators.append(calibrator)
                            
                            # Find optimal threshold
                            if 'prices' in metadata:
                                prices_val = metadata['prices'][val_idx]
                                optimal_threshold, threshold_metrics = find_optimal_threshold(
                                    y_prob, prices_val, metrics_calc, metric='sharpe_ratio'
                                )
                                fold_results['optimal_threshold'] = optimal_threshold
                                fold_results['threshold_metrics'] = threshold_metrics
                    
                    cv_results.append(fold_results)
                
                # Save model and artifacts
                model_dir = os.path.join(
                    args.output_dir,
                    model_type,
                    symbol,
                    datetime.now().strftime('%Y%m%d_%H%M%S')
                )
                
                # Train final model on all data
                logger.info(f"Training final {model_type} model on all data...")
                final_adapter = create_model_adapter(model_type, config, task_type)
                
                # Use all data for final training
                train_size = int(len(X) * 0.8)
                train_idx_final = np.arange(train_size)
                val_idx_final = np.arange(train_size, len(X))
                
                if model_type == 'ppo':
                    final_results = final_adapter.fit(
                        X=metadata.get('full_data', X),
                        y=y,
                        train_idx=train_idx_final,
                        valid_idx=val_idx_final,
                        experiment_name=f"{args.experiment_name}_{symbol}_{model_type}_final"
                    )
                else:
                    final_results = final_adapter.fit(
                        X=X,
                        y=y,
                        train_idx=train_idx_final,
                        valid_idx=val_idx_final,
                        experiment_name=f"{args.experiment_name}_{symbol}_{model_type}_final"
                    )
                
                # Save model
                saved_path = final_adapter.save(args.output_dir, run_id=model_dir.split('/')[-1])
                
                # Save additional artifacts
                artifacts_dir = Path(saved_path)
                
                # Save feature metadata
                with open(artifacts_dir / 'features.json', 'w') as f:
                    json.dump({
                        'feature_names': feature_names,
                        'feature_count': len(feature_names),
                        'metadata': metadata
                    }, f, indent=2)
                
                # Save calibrators if available
                if calibrators:
                    for i, cal in enumerate(calibrators):
                        cal.save(str(artifacts_dir / f'calibrator_fold{i}'))
                
                # Save CV results
                with open(artifacts_dir / 'cv_results.json', 'w') as f:
                    json.dump(cv_results, f, indent=2)
                
                # Create "latest" symlink
                latest_path = os.path.join(args.output_dir, model_type, symbol, 'latest')
                if os.path.exists(latest_path):
                    os.remove(latest_path) if os.path.islink(latest_path) else shutil.rmtree(latest_path)
                os.symlink(saved_path, latest_path)
                
                logger.info(f"{model_type.upper()} model saved to {saved_path}")
                
                results['models'][model_type] = {
                    'path': saved_path,
                    'cv_results': cv_results,
                    'final_results': final_results
                }
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} for {symbol}: {e}")
                results['errors'].append(f"{model_type}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Failed to process {symbol}: {e}")
        results['errors'].append(str(e))
    
    return results


def train_parallel(
    symbols: List[str],
    args: argparse.Namespace,
    config: Dict[str, Any],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Train models for multiple symbols in parallel.
    
    Args:
        symbols: List of symbols to train
        args: Command line arguments
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        List of training results
    """
    # Create dataset builder
    dataset_builder = DatasetBuilder(
        data_dir=args.data_dir,
        cache_dir=os.path.join(args.data_dir, 'features'),
        config=config
    )
    
    # Create partial function with fixed arguments
    train_func = partial(
        train_symbol_models,
        args=args,
        config=config,
        dataset_builder=dataset_builder,
        logger=logger
    )
    
    # Train in parallel
    with Pool(processes=args.max_workers) as pool:
        results = pool.map(train_func, symbols)
    
    return results


def main() -> None:
    """Main training orchestration function."""
    parser = argparse.ArgumentParser(
        description='Enhanced Crypto Trading Bot Trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic arguments
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory path')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Output directory for trained models')
    
    # Model selection
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['gru', 'lightgbm', 'ppo', 'all'],
                       default=['all'], help='Models to train')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                       help='Symbols to train (default: all available)')
    
    # Data parameters
    parser.add_argument('--interval', type=str, default='15m',
                       help='Data interval')
    parser.add_argument('--target-type', type=str, default='return',
                       choices=['return', 'direction', 'price'],
                       help='Target variable type')
    parser.add_argument('--target-horizon', type=int, default=1,
                       help='Prediction horizon in periods')
    
    # Cross-validation parameters
    parser.add_argument('--n-splits', type=int, default=5,
                       help='Number of CV splits')
    parser.add_argument('--embargo', type=int, default=100,
                       help='Embargo periods for CV')
    
    # Trading cost parameters
    parser.add_argument('--fee-bps', type=float, default=10.0,
                       help='Trading fee in basis points')
    parser.add_argument('--slippage-bps', type=float, default=5.0,
                       help='Slippage in basis points')
    
    # Performance parameters
    parser.add_argument('--cache', action='store_true', default=True,
                       help='Use feature cache')
    parser.add_argument('--no-cache', dest='cache', action='store_false',
                       help='Disable feature cache')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Max parallel workers (default: CPU count)')
    
    # Other parameters
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='MLflow experiment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    # Set max workers
    if args.max_workers is None:
        args.max_workers = min(cpu_count(), 4)  # Limit to 4 by default
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration. Exiting.")
        return
    
    # Setup logging
    logger = setup_logging(config)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    bot_logger = TradingBotLogger()
    
    logger.info("=" * 80)
    logger.info("ENHANCED CRYPTO TRADING BOT - MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Models: {args.models}")
    logger.info(f"Target: {args.target_type} (horizon={args.target_horizon})")
    logger.info(f"CV: {args.n_splits} splits, embargo={args.embargo}")
    logger.info(f"Costs: fee={args.fee_bps}bps, slippage={args.slippage_bps}bps")
    logger.info(f"Cache: {'enabled' if args.cache else 'disabled'}")
    logger.info(f"Parallel workers: {args.max_workers}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize notification system
    try:
        notifier = TelegramNotifier.from_config(config)
        if notifier.enabled:
            notifier.send_message_sync("🚀 Starting enhanced model training...")
    except Exception as e:
        logger.warning(f"Failed to initialize notifications: {e}")
        notifier = None
    
    try:
        # Create dataset builder
        dataset_builder = DatasetBuilder(
            data_dir=args.data_dir,
            cache_dir=os.path.join(args.data_dir, 'features'),
            config=config
        )
        
        # Get available symbols
        from src.data_pipeline.loader import DataLoader
        data_loader = DataLoader(args.data_dir)
        availability = data_loader.check_data_availability()
        available_symbols = [symbol for symbol, available in availability.items() if available]
        
        if not available_symbols:
            logger.error("No data available for training!")
            return
        
        # Filter symbols if specified
        if args.symbols:
            symbols_to_train = [s for s in args.symbols if s in available_symbols]
            if not symbols_to_train:
                logger.error(f"None of the specified symbols are available: {args.symbols}")
                return
        else:
            symbols_to_train = available_symbols
        
        logger.info(f"Training models for {len(symbols_to_train)} symbols: {symbols_to_train}")
        
        # Set experiment name
        if args.experiment_name is None:
            args.experiment_name = f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Train models
        if len(symbols_to_train) == 1 or args.max_workers == 1:
            # Sequential training
            results = []
            for symbol in symbols_to_train:
                result = train_symbol_models(
                    symbol, args, config, dataset_builder, logger
                )
                results.append(result)
        else:
            # Parallel training
            logger.info(f"Training in parallel with {args.max_workers} workers...")
            results = train_parallel(symbols_to_train, args, config, logger)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        
        successful_models = 0
        failed_models = 0
        
        for result in results:
            symbol = result['symbol']
            if result['errors']:
                logger.error(f"{symbol}: FAILED - {result['errors']}")
                failed_models += len(result['errors'])
            else:
                logger.info(f"{symbol}: SUCCESS - {len(result['models'])} models trained")
                successful_models += len(result['models'])
        
        logger.info(f"\nTotal: {successful_models} successful, {failed_models} failed")
        
        # Show cache info
        cache_info = dataset_builder.get_cache_info()
        logger.info(f"\nCache usage: {cache_info['total_cached_datasets']} datasets, "
                   f"{cache_info['total_size_mb']:.1f} MB")
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(
                f"✅ Training completed!\n"
                f"Successful: {successful_models}\n"
                f"Failed: {failed_models}"
            )
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        bot_logger.log_error("TRAINING_FAILURE", str(e))
        
        if notifier and notifier.enabled:
            notifier.send_message_sync(f"❌ Training failed: {str(e)}")
        
        raise


if __name__ == "__main__":
    main()