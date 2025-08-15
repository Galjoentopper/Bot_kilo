"""
Unified Enhanced Training Script
================================

This is the single trainer entrypoint. It integrates the advanced flow from
the previous trainer_enhanced.py, including DatasetBuilder, CV, calibration,
cost-aware metrics, adapters, parallel training, and unified artifacts.
"""

import sys
import os
import argparse
import yaml
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.logger import setup_logging, TradingBotLogger
from src.data_pipeline.dataset_builder import DatasetBuilder
from src.utils.cross_validation import PurgedTimeSeriesSplit
from src.utils.metrics import TradingMetrics, optimize_threshold
from src.utils.calibration import ProbabilityCalibrator
from src.models.adapters import create_model_adapter
from src.notifier.telegram import TelegramNotifier


def load_config(config_path: str = "src/config/config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Unified Enhanced Trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='src/config/config.yaml')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./models')
    parser.add_argument('--models', type=str, nargs='+', choices=['gru','lightgbm','ppo','all'], default=['all'])
    parser.add_argument('--symbols', type=str, nargs='+', default=None)
    parser.add_argument('--interval', type=str, default='15m')
    parser.add_argument('--target-type', type=str, choices=['return','direction','price'], default='return')
    parser.add_argument('--target-horizon', type=int, default=1)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--embargo', type=int, default=100)
    parser.add_argument('--fee-bps', type=float, default=10.0)
    parser.add_argument('--slippage-bps', type=float, default=5.0)
    parser.add_argument('--turnover-lambda', type=float, default=0.0)
    parser.add_argument('--cache', action='store_true', default=True)
    parser.add_argument('--no-cache', dest='cache', action='store_false')
    parser.add_argument('--max-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration. Exiting.")
        return

    logger = setup_logging(config)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    bot_logger = TradingBotLogger()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        notifier = TelegramNotifier.from_config(config)
    except Exception:
        notifier = None

    dataset_builder = DatasetBuilder(
        data_dir=args.data_dir,
        cache_dir=os.path.join(args.data_dir, 'features'),
        config=config
    )

    from src.data_pipeline.loader import DataLoader
    data_loader = DataLoader(args.data_dir)
    availability = data_loader.check_data_availability()
    available_symbols = [s for s, ok in availability.items() if ok]
    if not available_symbols:
        logger.error('No data available for training!')
        return
    symbols_to_train = args.symbols or available_symbols
    symbols_to_train = [s for s in symbols_to_train if s in available_symbols]

    if args.experiment_name is None:
        args.experiment_name = f"unified_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # CV splitter and metrics
    cv_splitter = PurgedTimeSeriesSplit(n_splits=args.n_splits, gap=args.embargo, embargo=args.embargo)
    metrics_calc = TradingMetrics(fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)

    model_list = args.models if args.models != ['all'] else ['gru','lightgbm','ppo']

    for symbol in symbols_to_train:
        logger.info(f"==== Training {symbol} ====")
        try:
            X, y, timestamps, feature_names, metadata = dataset_builder.build_dataset(
                symbol=symbol,
                interval=args.interval,
                use_cache=args.cache,
                target_type=args.target_type,
                target_horizon=args.target_horizon
            )
        except Exception as e:
            logger.error(f"Dataset build failed for {symbol}: {e}")
            continue

        is_valid, errors = dataset_builder.validate_dataset(X, y, metadata)
        if not is_valid:
            logger.error(f"Dataset invalid for {symbol}: {errors}")
            continue

        for model_type in model_list:
            try:
                task_type = 'classification' if args.target_type == 'direction' else 'regression'
                adapter = create_model_adapter(model_type, config, task_type)

                cv_results = []
                calibrators = []
                saved_threshold = None

                for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
                    logger.info(f"{model_type} fold {fold_idx+1}/{args.n_splits}")
                    if model_type == 'ppo':
                        fold_results = adapter.fit(
                            X=metadata.get('full_data', X),
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
                                turnover_lambda=args.turnover_lambda,
                                asymmetric=True,
                                objective='sharpe_ratio'
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

                # Save artifacts layout
                run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_dir = os.path.join(args.output_dir, model_type, symbol, run_id)
                os.makedirs(model_dir, exist_ok=True)
                saved_path = final_adapter.save(os.path.join(args.output_dir, model_type, symbol), run_id=run_id)

                # Extra artifacts
                with open(os.path.join(saved_path, 'features.json'), 'w') as f:
                    json.dump({'feature_names': list(getattr(X, 'columns', [])), 'feature_count': int(getattr(X, 'shape', [0,0])[1])}, f, indent=2)
                with open(os.path.join(saved_path, 'cv_results.json'), 'w') as f:
                    json.dump(cv_results, f, indent=2)
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

            except Exception as e:
                logger.error(f"Failed training {model_type} for {symbol}: {e}")
                continue


if __name__ == '__main__':
    main()