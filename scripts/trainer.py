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
from datetime import datetime, timedelta
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


def _make_jsonable(value: Any) -> Any:
    """Convert common non-JSON-serializable types into JSON-safe structures."""
    try:
        import numpy as _np  # local alias to avoid shadowing
        from datetime import datetime as _dt
    except Exception:
        _np = None
        _dt = None

    # Scalars
    if _np is not None and isinstance(value, _np.generic):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    # Datetime
    if _dt is not None and isinstance(value, _dt):
        return value.isoformat()
    # Numpy arrays
    if _np is not None and isinstance(value, _np.ndarray):
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


def main() -> None:
    # If run without flags, default to the walk-forward + Optuna harness
    if len(sys.argv) == 1:
        try:
            # Load default config
            config = load_config('src/config/config.yaml')
            # Import early so names are bound before use
            from src.data_pipeline.loader import DataLoader  # type: ignore
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
                import numpy as _np
                X_arr = X.values if hasattr(X, 'values') else _np.asarray(X)
                y_arr = _np.asarray(y)
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
            return
        except Exception as e:
            print(f"Default walk-forward harness failed: {e}")
            # Fall through to the regular trainer if default path fails
    parser = argparse.ArgumentParser(
        description='Unified Enhanced Trainer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='src/config/config.yaml')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./models')
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
    # Tri-state cache flag: None = use config, True/False if explicitly set
    parser.set_defaults(cache=None)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--no-cache', dest='cache', action='store_false')
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--objective', type=str, choices=['sharpe_ratio','sortino_ratio','calmar_ratio','profit_factor'], default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

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
        cache_dir="./models/metadata",
        config=config
    )

    from src.data_pipeline.loader import DataLoader
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
        args.experiment_name = f"unified_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    logger.info(f"Trainer settings: interval={interval}, target={target_type}, splits={n_splits}, embargo={embargo}, fees={fee_bps}bps, slippage={slippage_bps}bps, turnover_lambda={turnover_lambda}, cache={cache}, objective={objective}, max_workers={max_workers}, start_date={start_date}")

    # CV splitter and metrics
    cv_splitter = PurgedTimeSeriesSplit(n_splits=n_splits, gap=embargo, embargo=embargo)
    metrics_calc = TradingMetrics(fee_bps=fee_bps, slippage_bps=slippage_bps)

    default_models = trainer_cfg.get('default_models')
    if args.models is None:
        # If not specified in CLI or config, include PPO by default alongside GRU and LightGBM
        model_list = default_models if isinstance(default_models, list) else ['lightgbm','gru','ppo']
    else:
        model_list = ['gru','lightgbm','ppo'] if args.models == ['all'] else args.models

    # Notify start of training (after model_list is known)
    if notifier and getattr(notifier, 'enabled', False):
        try:
            symbols_preview = ", ".join(symbols_to_train)
            models_preview = ", ".join(model_list)
            notifier.send_message_sync(
                f"🚀 <b>Training started</b>\n<b>Symbols:</b> {symbols_preview}\n<b>Models:</b> {models_preview}\n<b>Interval:</b> {interval}\n<b>Start:</b> {start_date or 'full history'}"
            )
        except Exception:
            pass

    for symbol in symbols_to_train:
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

        for model_type in model_list:
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

    # Notify completion
    if notifier and getattr(notifier, 'enabled', False):
        try:
            notifier.send_message_sync("🏁 <b>Training run completed</b>")
        except Exception:
            pass


if __name__ == '__main__':
    main()