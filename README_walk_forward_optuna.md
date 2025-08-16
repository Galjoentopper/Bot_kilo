# Walk-Forward + Optuna Harness

Script: `scripts/walk_forward_optuna.py`

- Optimizes model hyperparameters on walk-forward splits using Sharpe (with simple fees) as the objective.
- Supports `lightgbm` and `gru`.
- Saves the best model to a path you specify.
 - Optional feature selection: Optuna-tuned correlation top-k feature mask for both LightGBM (2D) and GRU (applied across time consistently).

## Data formats

- NPZ: contains arrays `X` and `y` (override keys via `--x-key` and `--y-key`).
  - LightGBM: `X` shape (N, F), `y` shape (N,)
  - GRU: `X` shape (N, T, F), `y` shape (N,)
- CSV: must include `target` column. All other columns are features (LightGBM only).

## Usage

```powershell
# Example for LightGBM with NPZ data
python scripts/walk_forward_optuna.py --model lightgbm --data data/evaluations.npz --n-folds 5 --min-train 500 --trials 40 --fees-bps 10 --save-best models/metadata/best_lgbm.pkl

# Example for GRU with NPZ data
python scripts/walk_forward_optuna.py --model gru --data data/evaluations.npz --n-folds 5 --min-train 500 --trials 25 --fees-bps 10 --save-best models/metadata/best_gru.pt
```

Notes:
- Installs: ensure `optuna` is installed (added to `requirements.txt`).
- The objective uses centered model scores (for LightGBM classification/direction) or raw GRU outputs as signals; returns are computed as sign(score) * y minus simple turnover cost.
- For production, consider richer cost models and a stricter walk-forward schedule.

## Feature selection

You can enable simple, fast feature selection driven by Optuna:

- Method: absolute Pearson correlation with the target (computed on the training fold).
- Hyperparameters tuned by Optuna:
  - `fs_method`: `none` or `corr_topk`
  - `fs_top_k`: number of features to keep (5..64 by default)

Behavior:
- LightGBM: selects top-k columns on (N, F) inputs per fold and during final refit; the saved artifact includes `selected_features` indices.
- GRU: computes a per-feature score by averaging features over time on (N, T, F), selects the same feature indices across all timesteps; applied per fold and during final refit.

To adjust ranges, edit `scripts/walk_forward_optuna.py` (search for `feature_selection`).
