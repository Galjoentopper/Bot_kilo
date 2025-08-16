import os
import sys
import numpy as np
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.gru_trainer import GRUTrainer  # type: ignore


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Minimal config
    config = {
        'models': {
            'gru': {
                'sequence_length': 20,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.1,
                'learning_rate': 0.001,
                'optimizer': 'rmsprop',
                'batch_size': 32,
                'epochs': 2,
                'early_stopping_patience': 5,
                'loss': 'mse',
            }
        },
        'training': {
            'mixed_precision': False,
            'num_workers': 0,
            'pin_memory': False,
            'max_grad_norm': 1.0,
            'seed': 42,
            'deterministic': True,
        }
    }

    # Synthetic data
    n_train = 512
    n_val = 128
    seq_len = config['models']['gru']['sequence_length']
    n_feat = 10

    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(n_train, seq_len, n_feat)).astype(np.float32)
    X_val = rng.normal(size=(n_val, seq_len, n_feat)).astype(np.float32)
    # Target: linear combo of last timestep features + noise
    w = rng.normal(size=(n_feat,)).astype(np.float32)
    y_train = (X_train[:, -1, :] @ w + 0.01 * rng.normal(size=(n_train,))).astype(np.float32)
    y_val = (X_val[:, -1, :] @ w + 0.01 * rng.normal(size=(n_val,))).astype(np.float32)

    trainer = GRUTrainer(config)
    res = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        experiment_name="smoke_gru"
    )

    print("RESULTS:", {k: v for k, v in res.items() if k in ('best_val_loss', 'total_epochs')})


if __name__ == "__main__":
    main()
