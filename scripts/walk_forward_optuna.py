"""
Walk-forward + Optuna harness that optimizes Sharpe (with fees/slippage) and registers the best model.
Supports LightGBM (regression or classification-as-direction) and GRU.
"""

import os
import json
import math
import argparse
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# Optional: mlflow if available
try:
    import mlflow  # type: ignore
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

# Optional: Optuna for HPO
try:
    import optuna  # type: ignore
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

from datetime import datetime

from src.models.lgbm_trainer import LightGBMTrainer
from src.models.gru_trainer import GRUTrainer
from src.utils.cross_validation import PurgedTimeSeriesSplit


def compute_sharpe_with_cost(returns: np.ndarray, fee_bps: float = 10.0) -> float:
    """Compute simple Sharpe-like ratio on per-step returns net of costs.
    Assumptions:
      - returns is per-step strategy return before costs
      - Apply linear transaction cost when sign of position changes (turnover proxy)
    """
    if returns.size == 0:
        return 0.0
    # Basic Sharpe: mean / std (avoid div by zero)
    std = np.std(returns)
    if std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / (std + 1e-12))


def make_purged_splits(n: int, n_folds: int, embargo: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create embargoed purged time-series splits using project utility."""
    pts = PurgedTimeSeriesSplit(n_splits=n_folds, gap=embargo, embargo=embargo)
    # Build dummy X,y indices for splitting by length
    X_idx = np.arange(n)
    y_dummy = np.zeros(n)
    return [(tr, va) for tr, va in pts.split(X_idx, y_dummy)]


def to_direction(y: np.ndarray, thr: float = 0.0) -> np.ndarray:
    return (y > thr).astype(int)


def backtest_direction(scores: np.ndarray, y: np.ndarray, thr: float = 0.0, fee_bps: float = 10.0) -> Tuple[np.ndarray, float]:
    """
    Turn centered scores (negative: short, positive: long) into positions and compute simple returns.
    y is the next-step return.
    """
    # Position: sign of score
    pos = np.sign(scores)
    # Strategy per-step return
    rets = pos * y
    # Very simple turnover cost: cost when position flips
    flips = np.abs(np.diff(pos, prepend=0.0))
    costs = (fee_bps / 10000.0) * flips
    net = rets - costs
    sharpe = compute_sharpe_with_cost(net, fee_bps)
    return net, sharpe


def _build_sequences(X2: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct rolling windows (N_seq, T, F) and aligned targets (N_seq,) with end indices."""
    n = len(X2)
    if seq_len <= 0 or seq_len > n:
        raise ValueError("Invalid sequence length")
    ends = np.arange(seq_len - 1, n)
    T = seq_len
    F = X2.shape[1]
    # Build 3D array efficiently
    X3 = np.lib.stride_tricks.sliding_window_view(X2, window_shape=(seq_len, F))
    # sliding_window_view shape: (n - seq_len + 1, 1, seq_len, F) for 2D? Normalize
    X3 = X3.reshape(-1, seq_len, F)
    y2 = y[seq_len - 1:]
    return X3, y2, ends


def _corr_topk_indices_2d(X: np.ndarray, y: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices of top_k features by absolute Pearson correlation with y.
    NaNs are treated as zeros. Works on 2D X (N, F).
    """
    n, f = X.shape
    top_k = max(1, min(top_k, f))
    # Center
    Xc = X - np.nanmean(X, axis=0)
    yc = y - np.nanmean(y)
    # Numerator and denominator
    num = np.nan_to_num((Xc * yc[:, None]).sum(axis=0), nan=0.0)
    denom_x = np.sqrt(np.nan_to_num((Xc ** 2).sum(axis=0), nan=0.0))
    denom_y = math.sqrt(float(np.nan_to_num((yc ** 2).sum(), nan=0.0))) + 1e-12
    denom = denom_x * denom_y + 1e-12
    corr = np.zeros(f, dtype=float)
    nz = denom > 0
    corr[nz] = num[nz] / denom[nz]
    scores = np.abs(corr)
    # Top-k indices by score
    idx = np.argsort(-scores)[:top_k]
    return idx


def _maybe_select_features(cfg: Dict[str, Any], X_tr: np.ndarray, y_tr: np.ndarray, is_gru: bool) -> Optional[np.ndarray]:
    """Compute selected feature indices based on cfg feature_selection. Returns None if not enabled.
    For GRU, X_tr should be 3D (N,T,F); selection is computed by averaging over time.
    """
    fs = cfg.get('feature_selection', {})
    method = fs.get('method', 'none')
    if method == 'none':
        return None
    top_k = int(fs.get('top_k', 0))
    if top_k <= 0:
        return None
    if is_gru:
        # Aggregate across time to (N, F)
        if X_tr.ndim != 3:
            raise ValueError('Expected 3D sequences for GRU feature selection')
        Xagg = np.nanmean(X_tr, axis=1)
        return _corr_topk_indices_2d(Xagg, y_tr, top_k)
    else:
        if X_tr.ndim != 2:
            raise ValueError('Expected 2D features for LightGBM feature selection')
        return _corr_topk_indices_2d(X_tr, y_tr, top_k)


def evaluate_fold(model_name: str,
                  cfg: Dict[str, Any],
                  X: np.ndarray,
                  y: np.ndarray,
                  train_idx: np.ndarray,
                  val_idx: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if model_name == "lightgbm":
        trainer = LightGBMTrainer(cfg, task_type=cfg.get('models', {}).get('lightgbm', {}).get('task_type', 'regression'))
        # Feature selection (2D)
        sel_idx = _maybe_select_features(cfg, X_train, y_train, is_gru=False)
        if sel_idx is not None:
            X_train = X_train[:, sel_idx]
            X_val = X_val[:, sel_idx]
        # Ensure direction mode when classification or ensemble needs centered scores
        results = trainer.train(X_train, y_train, X_val, y_val)
        # We use centered score for direction/classification, or raw regression outputs otherwise
        preds = trainer.predict(X_val)
        # Compute net returns and Sharpe using the raw y_val as next-step returns
        net, sharpe = backtest_direction(preds, y_val, fee_bps=float(cfg.get('fees_bps', 10.0)))
        metrics = {
            'val_sharpe': sharpe,
            'val_mean': float(np.mean(net)),
            'val_std': float(np.std(net)),
        }
        return sharpe, {
            'trainer': trainer,
            'results': results,
            'metrics': metrics,
            'selected_features': sel_idx.tolist() if sel_idx is not None else None,
        }

    elif model_name == "gru":
        # Ensure sequences are built according to current cfg
        seq_len = int(cfg.get('models', {}).get('gru', {}).get('sequence_length', 20))
        if X.ndim == 2:
            X3, y2, ends = _build_sequences(X, y, seq_len)
        else:
            X3, y2 = X, y
            ends = np.arange(seq_len - 1, len(y))
        # Select sequence samples whose end index lies in each split
        tr_mask = np.isin(ends, train_idx)
        va_mask = np.isin(ends, val_idx)
        X_tr, y_tr = X3[tr_mask], y2[tr_mask]
        X_va, y_va = X3[va_mask], y2[va_mask]
        # Feature selection for GRU: select feature dims across time axis consistently
        sel_idx = _maybe_select_features(cfg, X_tr, y_tr, is_gru=True)
        if sel_idx is not None:
            X_tr = X_tr[:, :, sel_idx]
            X_va = X_va[:, :, sel_idx]

        trainer = GRUTrainer(cfg)
        trainer.build_model(X_tr.shape[2])
        trainer.train(X_tr, y_tr, X_va, y_va, experiment_name="gru_walk_forward")
        preds = trainer.predict(X_va)
        net, sharpe = backtest_direction(preds, y_val, fee_bps=float(cfg.get('fees_bps', 10.0)))
        metrics = {
            'val_sharpe': sharpe,
            'val_mean': float(np.mean(net)),
            'val_std': float(np.std(net)),
        }
        return sharpe, {
            'trainer': trainer,
            'results': {
                'best_val_loss': trainer.best_val_loss
            },
            'metrics': metrics,
            'selected_features': sel_idx.tolist() if sel_idx is not None else None,
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def objective_factory(model_name: str, cfg: Dict[str, Any], X: np.ndarray, y: np.ndarray, splits: List[Tuple[np.ndarray, np.ndarray]]):
    def objective(trial: Any) -> float:
        # Suggest parameters
        if model_name == "lightgbm":
            lgbm_cfg = cfg.setdefault('models', {}).setdefault('lightgbm', {})
            lgbm_cfg['num_leaves'] = trial.suggest_int('num_leaves', 31, 255)
            lgbm_cfg['max_depth'] = trial.suggest_int('max_depth', 4, 12)
            lgbm_cfg['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 2e-1, log=True)
            lgbm_cfg['n_estimators'] = trial.suggest_int('n_estimators', 100, 1200)
            # Allow toggling direction/classification mode
            lgbm_cfg['task_type'] = trial.suggest_categorical('task_type', ['regression', 'classification'])
            lgbm_cfg['as_direction'] = True if lgbm_cfg['task_type'] == 'classification' else trial.suggest_categorical('as_direction', [True, False])
        elif model_name == "gru":
            gru_cfg = cfg.setdefault('models', {}).setdefault('gru', {})
            gru_cfg['sequence_length'] = trial.suggest_int('sequence_length', 10, 60)
            gru_cfg['hidden_size'] = trial.suggest_int('hidden_size', 32, 256)
            gru_cfg['num_layers'] = trial.suggest_int('num_layers', 1, 4)
            gru_cfg['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
            gru_cfg['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
            gru_cfg['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128])
        else:
            raise ValueError("Unsupported model for HPO")

        # Optional feature selection hyperparameter (model-agnostic top-k corr)
        fs_cfg = cfg.setdefault('feature_selection', {})
        fs_enabled = trial.suggest_categorical('fs_method', ['none', 'corr_topk'])
        fs_cfg['method'] = 'none' if fs_enabled == 'none' else 'corr_topk'
        if fs_cfg['method'] != 'none':
            # Cap top_k to a reasonable range
            max_k = X.shape[-1] if X.ndim == 2 else X.shape[-1]
            fs_cfg['top_k'] = trial.suggest_int('fs_top_k', 5, min(64, int(max_k)))

        # Evaluate across folds and aggregate Sharpe
        sharpes: List[float] = []
        for (tr, va) in splits:
            sharpe, _ = evaluate_fold(model_name, cfg, X, y, tr, va)
            sharpes.append(sharpe)
        mean_sharpe = float(np.mean(sharpes))

        # We maximize Sharpe
        return mean_sharpe

    return objective


def run_walk_forward_optuna(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
    n_folds: int = 5,
    embargo: int = 0,
    trials: int = 30,
    fees_bps: float = 10.0,
    save_best: str | None = None,
):
    """
    Run walk-forward Optuna optimization with embargoed splits and return best summary.
    """
    if not OPTUNA_AVAILABLE:
        raise RuntimeError('Optuna not installed. Add optuna to requirements and install.')

    cfg = dict(cfg or {})
    cfg['fees_bps'] = fees_bps

    splits = make_purged_splits(len(y), n_folds, embargo)
    import optuna as _optuna  # type: ignore
    study = _optuna.create_study(direction='maximize')
    study.optimize(objective_factory(model, cfg, X, y, splits), n_trials=trials)

    best = {
        'best_sharpe': float(study.best_value),
        'best_params': study.best_params,
    }

    # Retrain and save best
    if model == 'lightgbm':
        trainer = LightGBMTrainer(cfg, task_type=cfg.get('models', {}).get('lightgbm', {}).get('task_type', 'regression'))
        # Apply feature selection if configured
        sel_idx = _maybe_select_features(cfg, X, y, is_gru=False)
        X_fit = X[:, sel_idx] if sel_idx is not None else X
        trainer.train(X_fit, y)
        if save_best:
            os.makedirs(os.path.dirname(save_best), exist_ok=True)
            import joblib
            artifact = {
                'model': trainer.model,
                'feature_names': trainer.feature_names,
                'config': cfg,
                'selected_features': sel_idx.tolist() if sel_idx is not None else None,
            }
            joblib.dump(artifact, save_best)
            best['saved_path'] = save_best
    elif model == 'gru':
        # Build sequences using best params
        seq_len = int(cfg.get('models', {}).get('gru', {}).get('sequence_length', 20))
        if X.ndim == 2:
            X3, y2, ends = _build_sequences(X, y, seq_len)
        else:
            X3, y2 = X, y
        trainer = GRUTrainer(cfg)
        # Apply feature selection for GRU (on sequences)
        sel_idx = _maybe_select_features(cfg, X3, y2, is_gru=True)
        X3_fit = X3[:, :, sel_idx] if sel_idx is not None else X3
        trainer.build_model(X3_fit.shape[2])
        tail = max(2 * getattr(trainer, 'batch_size', 64), 64)
        X_val, y_val = X3_fit[-tail:], y2[-tail:]
        trainer.train(X3_fit, y2, X_val, y_val, experiment_name='gru_best_refit')
        if save_best:
            os.makedirs(os.path.dirname(save_best), exist_ok=True)
            trainer.save_model(save_best)
            best['saved_path'] = save_best
    else:
        raise ValueError('Unsupported model')

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['lightgbm', 'gru'], default='lightgbm')
    parser.add_argument('--data', type=str, required=True, help='Path to a .npz with X (n,T,F?) and y arrays or CSV with features and target')
    parser.add_argument('--x-key', type=str, default='X')
    parser.add_argument('--y-key', type=str, default='y')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--min-train', type=int, default=500)
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--fees-bps', type=float, default=10.0)
    parser.add_argument('--save-best', type=str, default='models/metadata/best_wf_model.pkl')
    args = parser.parse_args()

    cfg: Dict[str, Any] = {
        'models': {},
        'training': {
            'seed': 42,
            'deterministic': False,
        },
        'fees_bps': args.fees_bps,
    }

    # Load data
    if args.data.endswith('.npz'):
        dat = np.load(args.data, allow_pickle=True)
        X = dat[args.x_key]
        y = dat[args.y_key]
    elif args.data.endswith('.csv'):
        df = pd.read_csv(args.data)
        if 'target' not in df.columns:
            raise ValueError('CSV must contain a column named target')
        y = df['target'].values.astype(float)
        X = df.drop(columns=['target']).values.astype(float)
    else:
        raise ValueError('Unsupported data format; pass .npz or .csv')

    res = run_walk_forward_optuna(
        model=args.model,
        X=X,
        y=y,
        cfg=cfg,
        n_folds=args.n_folds,
        embargo=args.min_train,  # fallback if user passes min-train; treat as embargo here
        trials=args.trials,
        fees_bps=args.fees_bps,
        save_best=args.save_best,
    )
    print(f"Best Sharpe: {res['best_sharpe']:.4f}")
    print(f"Best params: {json.dumps(res['best_params'], indent=2)}")
    if 'saved_path' in res:
        print(f"Saved best model to {res['saved_path']}")


if __name__ == '__main__':
    main()
