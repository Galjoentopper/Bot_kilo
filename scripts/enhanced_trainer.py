#!/usr/bin/env python3
"""
Enhanced Unified Training Script with Model Packaging
===================================================

This enhanced trainer extends the original trainer.py with:
- Automatic model packaging for easy transfer
- Enhanced export functionality
- Better model organization and metadata
- Transfer-ready artifacts
- Compatibility validation
"""

import sys
import os
import argparse
import yaml
import json
import shutil
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Critical dependencies with error handling
try:
    import numpy as np
except ImportError as e:
    print(f"Error: NumPy is required but not installed: {e}")
    print("Please install numpy: pip install numpy")
    sys.exit(1)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.logger import setup_logging, TradingBotLogger
from src.utils.mlflow_init import initialize_mlflow_from_config
from src.data_pipeline.dataset_builder import DatasetBuilder
from src.data_pipeline.loader import DataLoader
from src.utils.cross_validation import PurgedTimeSeriesSplit
from src.utils.metrics import TradingMetrics, optimize_threshold
from src.utils.calibration import ProbabilityCalibrator
from src.models.adapters import create_model_adapter
from src.notifier.telegram import TelegramNotifier

# Import our new model packaging utilities
from src.utils.model_packaging import ModelPackager
from src.utils.model_transfer import ModelTransferManager
from src.utils.training_checkpoint import TrainingCheckpoint, TrainingProgress, CheckpointMetadata

# Global variables for checkpoint management
checkpoint_manager = None
shutdown_requested = False
current_progress = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested, checkpoint_manager, current_progress
    
    print("\n🛑 Shutdown signal received. Saving checkpoint...")
    shutdown_requested = True
    
    if checkpoint_manager and current_progress:
        try:
            # Save current progress before shutdown
            checkpoint_manager.save_checkpoint(
                progress=current_progress,
                config={},  # Will be updated with actual config during training
                partial_results={}
            )
            print("✅ Checkpoint saved successfully. Training can be resumed later.")
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
    
    print("Exiting gracefully...")
    sys.exit(0)


def load_config(config_path: str = "src/config/config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def _make_jsonable(value: Any) -> Any:
    """Convert common non-JSON-serializable types into JSON-safe structures."""
    # Scalars
    if hasattr(np, 'generic') and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    # Datetime
    if isinstance(value, datetime):
        return value.isoformat()
    # Numpy arrays
    if hasattr(np, 'ndarray') and isinstance(value, np.ndarray):
        # Be cautious with huge arrays; convert to list
        return value.tolist()
    # Lists / tuples
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(v) for v in value]
    # Dicts
    if isinstance(value, dict):
        return {k: _make_jsonable(v) for k, v in value.items()}
    # Fallback: best-effort string
    try:
        json.dumps(value)  # type: ignore[arg-type]
        return value
    except Exception:
        return str(value)


def _sanitize_results(res: Dict[str, Any]) -> Dict[str, Any]:
    """Drop or convert non-serializable fields from training result dicts."""
    if not isinstance(res, dict):
        return {}
    blacklist = {"model", "model_state", "feature_importance"}
    out: Dict[str, Any] = {}
    for k, v in res.items():
        if k in blacklist:
            continue
        out[k] = _make_jsonable(v)
    return out


def package_and_export_models(output_dir: str, symbols: List[str], models: List[str], 
                             config: Dict[str, Any], logger: logging.Logger,
                             export_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Package trained models for easy transfer and deployment.
    
    Args:
        output_dir: Directory containing trained models
        symbols: List of symbols that were trained
        models: List of model types that were trained
        config: Configuration dictionary
        logger: Logger instance
        export_dir: Optional directory to export packaged models
    
    Returns:
        Dictionary with packaging results
    """
    logger.info("Starting model packaging and export...")
    
    packager = ModelPackager()
    transfer_manager = ModelTransferManager()
    
    packaging_results = {
        'packaged_models': [],
        'failed_models': [],
        'export_path': None,
        'transfer_ready': False
    }
    
    # Package individual models
    for symbol in symbols:
        for model_type in models:
            model_path = os.path.join(output_dir, model_type, symbol)
            
            # Find the latest model directory
            if os.path.exists(model_path):
                # Look for latest symlink or latest_pointer.txt
                latest_path = os.path.join(model_path, 'latest')
                latest_pointer_path = os.path.join(model_path, 'latest_pointer.txt')
                
                actual_model_path = None
                if os.path.exists(latest_path):
                    if os.path.islink(latest_path):
                        actual_model_path = os.readlink(latest_path)
                    elif os.path.isdir(latest_path):
                        actual_model_path = latest_path
                elif os.path.exists(latest_pointer_path):
                    with open(latest_pointer_path, 'r') as f:
                        actual_model_path = f.read().strip()
                
                if actual_model_path and os.path.exists(actual_model_path):
                    try:
                        # Package the model
                        package_info = packager.package_model(
                            model_path=actual_model_path,
                            model_type=model_type,
                            symbol=symbol,
                            config=config
                        )
                        
                        packaging_results['packaged_models'].append({
                            'symbol': symbol,
                            'model_type': model_type,
                            'package_path': package_info['package_path'],
                            'metadata': package_info['metadata']
                        })
                        
                        logger.info(f"Successfully packaged {model_type} model for {symbol}")
                        
                    except Exception as e:
                        logger.error(f"Failed to package {model_type} model for {symbol}: {e}")
                        packaging_results['failed_models'].append({
                            'symbol': symbol,
                            'model_type': model_type,
                            'error': str(e)
                        })
                else:
                    logger.warning(f"No trained model found for {model_type}/{symbol}")
    
    # Create transfer bundle if we have packaged models
    if packaging_results['packaged_models']:
        try:
            if export_dir is None:
                export_dir = os.path.join(output_dir, 'exports', datetime.now().strftime('%Y%m%d_%H%M%S'))
            
            os.makedirs(export_dir, exist_ok=True)
            
            # Prepare models for transfer
            transfer_info = transfer_manager.prepare_models_for_transfer(
                model_packages=[pkg['package_path'] for pkg in packaging_results['packaged_models']],
                destination_dir=export_dir,
                include_config=True,
                config_path="src/config/config.yaml"
            )
            
            packaging_results['export_path'] = export_dir
            packaging_results['transfer_ready'] = True
            packaging_results['transfer_info'] = transfer_info
            
            logger.info(f"Models packaged and ready for transfer at: {export_dir}")
            logger.info(f"Transfer bundle includes: {len(packaging_results['packaged_models'])} models")
            
        except Exception as e:
            logger.error(f"Failed to create transfer bundle: {e}")
            packaging_results['transfer_ready'] = False
    
    return packaging_results


def main() -> None:
    # If run without flags, default to the walk-forward + Optuna harness
    if len(sys.argv) == 1:
        try:
            # Load default config
            config = load_config('src/config/config.yaml')
            # Import early so names are bound before use
            from scripts.walk_forward_optuna import run_walk_forward_optuna  # type: ignore
            logger = setup_logging(config)

            # Determine available symbols
            data_loader = DataLoader('./data')
            availability = data_loader.check_data_availability()
            available_symbols = [s for s, ok in availability.items() if ok]
            if not available_symbols:
                logger.error('No data available for training!')
                return

            # Symbols to optimize from config or all available
            trainer_cfg = config.get('trainer', {}) if isinstance(config, dict) else {}
            symbols_cfg = trainer_cfg.get('symbols') if isinstance(trainer_cfg.get('symbols'), list) else None
            symbols = symbols_cfg or available_symbols
            symbols = [s for s in symbols if s in available_symbols]
            if not symbols:
                logger.error('No valid symbols found to optimize')
                return
            interval = (
                trainer_cfg.get('interval')
                or (config.get('data', {}) or {}).get('interval', '15m')
            )
            target_type = trainer_cfg.get('target_type', 'return')
            target_horizon = int(trainer_cfg.get('default_target_horizon', 1))
            n_splits = int(trainer_cfg.get('n_splits', 5))
            embargo = int(trainer_cfg.get('embargo', 100))
            fee_bps = float(trainer_cfg.get('fee_bps', 10.0))
            trials = int(trainer_cfg.get('optuna_trials', 30))
            
            # Build and optimize per symbol
            dataset_builder = DatasetBuilder(
                data_dir='./data',
                cache_dir='./models/metadata',
                config=config
            )
            
            trained_models = []
            for symbol in symbols:
                X, y, timestamps, feature_names, metadata = dataset_builder.build_dataset(
                    symbol=symbol,
                    interval=interval,
                    use_cache=True,
                    target_type=target_type,
                    target_horizon=target_horizon,
                    start_date=trainer_cfg.get('start_date')
                )
                is_valid, errors = dataset_builder.validate_dataset(X, y, metadata)
                if not is_valid:
                    logger.error(f"Dataset invalid for {symbol}: {errors}")
                    continue
                # Convert to arrays
                X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
                y_arr = np.asarray(y)
                
                # Train LightGBM
                save_best = os.path.join('models', 'metadata', f'best_wf_lightgbm_{symbol}.pkl')
                res = run_walk_forward_optuna(
                    model='lightgbm',
                    X=X_arr,
                    y=y_arr,
                    cfg=config,
                    n_folds=n_splits,
                    embargo=embargo,
                    trials=trials,
                    fees_bps=fee_bps,
                    save_best=save_best,
                )
                logger.info(f"{symbol} best Sharpe: {res.get('best_sharpe')}, saved: {res.get('saved_path')}")
                trained_models.append(('lightgbm', symbol))
                
                # Also run GRU auto by default
                save_best_gru = os.path.join('models', 'metadata', f'best_wf_gru_{symbol}.pt')
                res_gru = run_walk_forward_optuna(
                    model='gru',
                    X=X_arr,
                    y=y_arr,
                    cfg=config,
                    n_folds=n_splits,
                    embargo=embargo,
                    trials=max(10, trials//2),  # fewer trials by default for GRU
                    fees_bps=fee_bps,
                    save_best=save_best_gru,
                )
                logger.info(f"{symbol} GRU best Sharpe: {res_gru.get('best_sharpe')}, saved: {res_gru.get('saved_path')}")
                trained_models.append(('gru', symbol))
            
            # Package and export models after training
            if trained_models:
                symbols_trained = list(set([symbol for _, symbol in trained_models]))
                models_trained = list(set([model for model, _ in trained_models]))
                
                packaging_results = package_and_export_models(
                    output_dir='./models',
                    symbols=symbols_trained,
                    models=models_trained,
                    config=config,
                    logger=logger
                )
                
                if packaging_results['transfer_ready']:
                    logger.info(f"✅ Models packaged and ready for transfer at: {packaging_results['export_path']}")
                    logger.info("📦 Use the generated import_models.py script to transfer models to another machine")
                else:
                    logger.warning("⚠️ Model packaging completed but transfer bundle creation failed")
            
            return
        except Exception as e:
            print(f"Default walk-forward harness failed: {e}")
            # Fall through to the regular trainer if default path fails
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Enhanced Unified Trainer with Model Packaging',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='src/config/config.yaml')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./models')
    parser.add_argument('--export-dir', type=str, default=None, help='Directory to export packaged models')
    parser.add_argument('--models', type=str, nargs='+', choices=['gru','lightgbm','ppo','all'], default=None)
    parser.add_argument('--symbols', type=str, nargs='+', default=None)
    parser.add_argument('--interval', type=str, default=None)
    parser.add_argument('--target-type', type=str, choices=['return','direction','price'], default=None)
    parser.add_argument('--target-horizon', type=int, default=None)
    parser.add_argument('--n-splits', type=int, default=None)
    parser.add_argument('--embargo', type=int, default=None)
    parser.add_argument('--fee-bps', type=float, default=None)
    parser.add_argument('--slippage-bps', type=float, default=None)
    parser.add_argument('--turnover-lambda', type=float, default=None)
    parser.add_argument('--package-models', action='store_true', help='Package models after training')
    parser.add_argument('--create-transfer-bundle', action='store_true', help='Create transfer bundle after training')
    
    # Tri-state cache flag: None = use config, True/False if explicitly set
    parser.set_defaults(cache=None)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--no-cache', dest='cache', action='store_false')
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--objective', type=str, choices=['sharpe_ratio','sortino_ratio','calmar_ratio','profit_factor'], default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory for checkpoint files')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    config = load_config(args.config)
    if not config:
        print("Failed to load configuration. Exiting.")
        return

    # Initialize MLflow tracking with dynamic paths (fixes hardcoded path issues)
    try:
        if not initialize_mlflow_from_config(args.config):
            print("Warning: MLflow initialization failed, but continuing...")
    except Exception as e:
        print(f"Warning: MLflow initialization error: {e}")

    logger = setup_logging(config)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    bot_logger = TradingBotLogger()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        notifier = TelegramNotifier.from_config(config)
        logger.info(f"Telegram notifier initialized: enabled={getattr(notifier, 'enabled', False)}")
    except Exception as e:
        logger.error(f"Failed to initialize Telegram notifier: {e}")
        notifier = None

    dataset_builder = DatasetBuilder(
        data_dir=args.data_dir,
        cache_dir="./models/metadata",
        config=config
    )

    data_loader = DataLoader(args.data_dir)
    availability = data_loader.check_data_availability()
    available_symbols = [s for s, ok in availability.items() if ok]
    if not available_symbols:
        logger.error('No data available for training!')
        return
    
    # Resolve trainer defaults from config when CLI not provided
    trainer_cfg = config.get('trainer', {}) if isinstance(config, dict) else {}
    symbols_default = trainer_cfg.get('symbols') if isinstance(trainer_cfg.get('symbols'), list) else None
    symbols_to_train = args.symbols or symbols_default or available_symbols
    symbols_to_train = [s for s in symbols_to_train if s in available_symbols]

    if args.experiment_name is None:
        args.experiment_name = f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resolve trainer defaults from config when CLI not provided (continued)
    interval = (
        args.interval
        or trainer_cfg.get('interval')
        or (config.get('data', {}) or {}).get('interval', '15m')
    )
    target_type = args.target_type or trainer_cfg.get('target_type', 'return')
    target_horizon = args.target_horizon if args.target_horizon is not None else int(trainer_cfg.get('default_target_horizon', 1))
    n_splits = args.n_splits if args.n_splits is not None else int(trainer_cfg.get('n_splits', 5))
    embargo = args.embargo if args.embargo is not None else int(trainer_cfg.get('embargo', 100))
    fee_bps = args.fee_bps if args.fee_bps is not None else float(trainer_cfg.get('fee_bps', 10.0))
    slippage_bps = args.slippage_bps if args.slippage_bps is not None else float(trainer_cfg.get('slippage_bps', 5.0))
    turnover_lambda = args.turnover_lambda if args.turnover_lambda is not None else float(trainer_cfg.get('turnover_lambda', 0.05))
    max_workers = args.max_workers if args.max_workers is not None else int(trainer_cfg.get('max_workers', 1))
    objective = args.objective or trainer_cfg.get('objective', 'sharpe_ratio')
    cache = args.cache if args.cache is not None else bool(trainer_cfg.get('cache', True))
    seed = args.seed if args.seed is not None else int(trainer_cfg.get('seed', 42))
    
    # Explicit start_date support
    start_date = (
        args.start_date
        or trainer_cfg.get('start_date')
        or (config.get('data', {}) or {}).get('start_date')
    )

    # Apply random seeds
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    # Optional: seed torch/lightgbm/sb3 if installed
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        import lightgbm as lgb
        # LightGBM uses seed in params; adapters should pass this seed
        os.environ.setdefault('LGBM_RAND_SEED', str(seed))
    except Exception:
        pass

    # If using maker/taker fees and no CLI fee override, pick from order_type
    if args.fee_bps is None:
        order_type = str(trainer_cfg.get('order_type', '')).lower()
        maker_fee = float(trainer_cfg.get('maker_fee_bps', fee_bps))
        taker_fee = float(trainer_cfg.get('taker_fee_bps', fee_bps))
        if order_type in ('maker', 'taker'):
            fee_bps = maker_fee if order_type == 'maker' else taker_fee

    logger.info(f"Enhanced Trainer settings: interval={interval}, target={target_type}, splits={n_splits}, embargo={embargo}, fees={fee_bps}bps, slippage={slippage_bps}bps, turnover_lambda={turnover_lambda}, cache={cache}, objective={objective}, max_workers={max_workers}, start_date={start_date}")
    logger.info(f"Model packaging: enabled={args.package_models or args.create_transfer_bundle}")

    # Determine model list before checkpoint initialization
    default_models = trainer_cfg.get('default_models')
    if args.models is None:
        # If not specified in CLI or config, include PPO by default alongside GRU and LightGBM
        model_list = default_models if isinstance(default_models, list) else ['lightgbm','gru','ppo']
    else:
        model_list = ['gru','lightgbm','ppo'] if args.models == ['all'] else args.models

    # Initialize checkpoint system
    global checkpoint_manager, shutdown_requested, current_progress
    checkpoint_manager = TrainingCheckpoint(args.checkpoint_dir)
    shutdown_requested = False
    current_progress = TrainingProgress(
        symbol_index=0,
        model_index=0,
        fold_index=0,
        total_symbols=len(symbols_to_train),
        total_models=len(model_list),
        total_folds=n_splits,
        completed_models=[],
        partial_results={}
    )
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check for existing checkpoint and resume if requested
    if args.resume:
        try:
            checkpoint_data = checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                current_progress = checkpoint_data['progress']
                logger.info(f"Resuming from checkpoint: Symbol {current_progress.symbol_index+1}/{current_progress.total_symbols}, Model {current_progress.model_index+1}/{current_progress.total_models}")
                logger.info(f"Completed models: {len(current_progress.completed_models)}")
            else:
                logger.info("No valid checkpoint found, starting fresh training")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh training")
    else:
        # Clean up any existing checkpoints if not resuming
        checkpoint_manager.cleanup_checkpoints()
        logger.info("Starting fresh training (checkpoints cleared)")

    # CV splitter and metrics
    cv_splitter = PurgedTimeSeriesSplit(n_splits=n_splits, gap=embargo, embargo=embargo)
    metrics_calc = TradingMetrics(fee_bps=fee_bps, slippage_bps=slippage_bps)

    # Notify start of training (after model_list is known)
    if notifier and getattr(notifier, 'enabled', False):
        try:
            symbols_preview = ", ".join(symbols_to_train)
            models_preview = ", ".join(model_list)
            logger.info("Sending Telegram start notification...")
            notifier.send_message_sync(
                f"🚀 <b>Enhanced Training started</b>\n<b>Symbols:</b> {symbols_preview}\n<b>Models:</b> {models_preview}\n<b>Interval:</b> {interval}\n<b>Start:</b> {start_date or 'full history'}\n<b>Packaging:</b> {'✅' if args.package_models or args.create_transfer_bundle else '❌'}"
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram start notification: {e}")
    else:
        logger.warning(f"Telegram notifications disabled - notifier exists: {notifier is not None}, enabled: {getattr(notifier, 'enabled', False) if notifier else False}")

    trained_models = []
    
    # Restore trained_models from checkpoint if resuming
    if args.resume and current_progress.completed_models:
        for model_key in current_progress.completed_models:
            if '_' in model_key:
                symbol, model_type = model_key.rsplit('_', 1)
                trained_models.append((model_type, symbol))
        logger.info(f"Restored {len(trained_models)} completed models from checkpoint")
    
    # Main training loop with checkpoint support
    for symbol_idx, symbol in enumerate(symbols_to_train):
        # Skip symbols that are already completed (resume logic)
        if symbol_idx < current_progress.symbol_index:
            logger.info(f"Skipping already completed symbol: {symbol}")
            continue
            
        current_progress.symbol_index = symbol_idx
        logger.info(f"==== Training {symbol} ====")
        try:
            X, y, timestamps, feature_names, metadata = dataset_builder.build_dataset(
                symbol=symbol,
                interval=interval,
                use_cache=cache,
                target_type=target_type,
                target_horizon=target_horizon,
                start_date=start_date
            )
        except Exception as e:
            logger.error(f"Dataset build failed for {symbol}: {e}")
            continue

        is_valid, errors = dataset_builder.validate_dataset(X, y, metadata)
        if not is_valid:
            logger.error(f"Dataset invalid for {symbol}: {errors}")
            continue

        for model_idx, model_type in enumerate(model_list):
            # Skip models that are already completed for this symbol (resume logic)
            model_key = f"{symbol}_{model_type}"
            if model_key in current_progress.completed_models:
                logger.info(f"Skipping already completed model: {model_type} for {symbol}")
                continue
                
            # Check for shutdown signal
            if shutdown_requested:
                logger.info("Shutdown requested, saving checkpoint and exiting...")
                checkpoint_manager.save_checkpoint(
                    progress=current_progress,
                    config={
                        'symbols': symbols_to_train,
                        'models': model_list,
                        'interval': interval,
                        'target_type': target_type,
                        'target_horizon': target_horizon,
                        'n_splits': n_splits,
                        'embargo': embargo,
                        'fee_bps': fee_bps,
                        'slippage_bps': slippage_bps,
                        'turnover_lambda': turnover_lambda,
                        'cache': cache,
                        'objective': objective,
                        'seed': seed,
                        'start_date': start_date
                    }
                )
                return
                
            current_progress.model_index = model_idx
            logger.info(f"Training {model_type} for {symbol} (Progress: {len(current_progress.completed_models)}/{len(symbols_to_train) * len(model_list)} models completed)")
            
            try:
                task_type = 'classification' if target_type == 'direction' else 'regression'
                adapter = create_model_adapter(model_type, config, task_type)

                cv_results = []
                calibrators = []
                saved_threshold = None

                for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
                    logger.info(f"{model_type} fold {fold_idx+1}/{n_splits}")
                    if model_type == 'ppo':
                        # Use raw OHLCV data for PPO if available
                        ppo_X = metadata.get('_runtime', {}).get('full_data', metadata.get('full_data', X))
                        fold_results = adapter.fit(
                            X=ppo_X,
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

                    # Threshold and calibration for classifiers
                    if task_type == 'classification' and hasattr(adapter, 'predict_proba'):
                        X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                        y_val = y[val_idx]
                        y_prob = adapter.predict_proba(X_val)[:, 1]

                        calibrator = ProbabilityCalibrator(method='isotonic')
                        calibrator.fit(y_val, y_prob)
                        calibrators.append(calibrator)
                        y_prob_cal = calibrator.transform(y_prob)

                        prices = np.asarray(metadata.get('prices', []))
                        if prices.size == len(y_prob_cal):
                            best, by = optimize_threshold(
                                y_true=y_val,
                                y_proba=y_prob_cal,
                                prices=prices[val_idx],
                                metrics_calculator=metrics_calc,
                                turnover_lambda=turnover_lambda,
                                asymmetric=True,
                                objective=objective
                            )
                            fold_results['optimal_threshold'] = best
                            fold_results['threshold_scan'] = by
                            saved_threshold = best

                    cv_results.append(fold_results)

                # Final fit on all data (simple 80/20 for monitoring)
                final_adapter = create_model_adapter(model_type, config, task_type)
                n = len(X)
                train_idx_final = np.arange(int(n*0.8))
                val_idx_final = np.arange(int(n*0.8), n)

                if model_type == 'ppo':
                    ppo_X_final = metadata.get('_runtime', {}).get('full_data', metadata.get('full_data', X))
                    final_results = final_adapter.fit(
                        X=ppo_X_final,
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

                # Save artifacts layout
                run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_dir = os.path.join(args.output_dir, model_type, symbol, run_id)
                os.makedirs(model_dir, exist_ok=True)
                saved_path = final_adapter.save(os.path.join(args.output_dir, model_type, symbol), run_id=run_id)

                # Extra artifacts
                with open(os.path.join(saved_path, 'features.json'), 'w') as f:
                    json.dump({'feature_names': list(getattr(X, 'columns', [])), 'feature_count': int(getattr(X, 'shape', [0,0])[1])}, f, indent=2)
                with open(os.path.join(saved_path, 'cv_results.json'), 'w') as f:
                    json.dump([_sanitize_results(r) for r in cv_results], f, indent=2)
                if calibrators:
                    for i, cal in enumerate(calibrators):
                        cal.save(os.path.join(saved_path, f'calibrator_fold{i}'))
                if saved_threshold is not None:
                    with open(os.path.join(saved_path, 'threshold.json'), 'w') as f:
                        json.dump(saved_threshold, f, indent=2)

                latest_path = os.path.join(args.output_dir, model_type, symbol, 'latest')
                if os.path.exists(latest_path):
                    try:
                        os.remove(latest_path) if os.path.islink(latest_path) else shutil.rmtree(latest_path)
                    except Exception:
                        pass
                try:
                    os.symlink(saved_path, latest_path)
                except Exception:
                    # On Windows, create a copy of latest pointer info
                    with open(os.path.join(os.path.dirname(latest_path), 'latest_pointer.txt'), 'w') as f:
                        f.write(saved_path)

                logger.info(f"Saved {model_type} artifacts to {saved_path}")
                trained_models.append((model_type, symbol))
                
                # Update checkpoint progress
                model_key = f"{symbol}_{model_type}"
                current_progress.completed_models.append(model_key)
                current_progress.partial_results[model_key] = {
                    'saved_path': saved_path,
                    'run_id': run_id,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save checkpoint after each model completion
                try:
                    checkpoint_manager.save_checkpoint(
                        progress=current_progress,
                        config={
                            'symbols': symbols_to_train,
                            'models': model_list,
                            'interval': interval,
                            'target_type': target_type,
                            'target_horizon': target_horizon,
                            'n_splits': n_splits,
                            'embargo': embargo,
                            'fee_bps': fee_bps,
                            'slippage_bps': slippage_bps,
                            'turnover_lambda': turnover_lambda,
                            'cache': cache,
                            'objective': objective,
                            'seed': seed,
                            'start_date': start_date
                        }
                    )
                    logger.debug(f"Checkpoint saved after completing {model_type} for {symbol}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")
                
                # Notify model completion
                if notifier and getattr(notifier, 'enabled', False):
                    try:
                        notifier.send_message_sync(
                            f"✅ <b>{model_type.upper()} trained</b> for <b>{symbol}</b>\nArtifacts: {os.path.basename(saved_path)}"
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Failed training {model_type} for {symbol}: {e}")
                if notifier and getattr(notifier, 'enabled', False):
                    try:
                        notifier.send_message_sync(
                            f"🚨 <b>Training error</b>\n<b>Symbol:</b> {symbol}\n<b>Model:</b> {model_type}\n<b>Message:</b> {str(e)}"
                        )
                    except Exception:
                        pass
                continue

    # Package and export models after training if requested
    if (args.package_models or args.create_transfer_bundle) and trained_models:
        logger.info("Starting post-training model packaging...")
        
        symbols_trained = list(set([symbol for _, symbol in trained_models]))
        models_trained = list(set([model for model, _ in trained_models]))
        
        packaging_results = package_and_export_models(
            output_dir=args.output_dir,
            symbols=symbols_trained,
            models=models_trained,
            config=config,
            logger=logger,
            export_dir=args.export_dir
        )
        
        if packaging_results['transfer_ready']:
            logger.info(f"✅ Models packaged and ready for transfer at: {packaging_results['export_path']}")
            logger.info("📦 Use the generated import_models.py script to transfer models to another machine")
            
            # Notify about packaging completion
            if notifier and getattr(notifier, 'enabled', False):
                try:
                    notifier.send_message_sync(
                        f"📦 <b>Model packaging completed</b>\n<b>Export path:</b> {os.path.basename(packaging_results['export_path'])}\n<b>Models packaged:</b> {len(packaging_results['packaged_models'])}\n<b>Transfer ready:</b> ✅"
                    )
                except Exception:
                    pass
        else:
            logger.warning("⚠️ Model packaging completed but transfer bundle creation failed")
            if notifier and getattr(notifier, 'enabled', False):
                try:
                    notifier.send_message_sync(
                        f"⚠️ <b>Model packaging warning</b>\nPackaging completed but transfer bundle creation failed"
                    )
                except Exception:
                    pass

    # Clean up checkpoints after successful completion
    try:
        checkpoint_manager.cleanup_checkpoint()
        logger.info("Training completed successfully, checkpoints cleaned up")
    except Exception as e:
        logger.error(f"Failed to cleanup checkpoints: {e}")
    
    # Notify completion
    if notifier and getattr(notifier, 'enabled', False):
        try:
            notifier.send_message_sync("🏁 <b>Enhanced training run completed</b>")
        except Exception:
            pass


if __name__ == '__main__':
    main()