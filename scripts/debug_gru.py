#!/usr/bin/env python3
"""
Debug script for GRU trainer to identify gradient instability issues.
"""

import sys
import os
import numpy as np
import yaml
import logging
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.logger import setup_logging
from src.models.gru_trainer import GRUTrainer

def load_config():
    """Load configuration."""
    config_path = os.path.join(project_root, "src/config/config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_synthetic_data():
    """Create simple synthetic data for testing."""
    np.random.seed(42)
    
    # Simple, well-behaved data
    n_samples = 1000
    sequence_length = 20
    n_features = 10
    
    # Generate features with small, controlled values
    X = np.random.normal(0, 0.1, (n_samples, sequence_length, n_features))
    
    # Simple target: sum of last 3 features with small noise
    y = np.sum(X[:, -1, -3:], axis=1) + np.random.normal(0, 0.01, n_samples)
    
    # Split into train/val
    split_idx = int(0.8 * n_samples)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Data ranges - X: [{X.min():.4f}, {X.max():.4f}], y: [{y.min():.4f}, {y.max():.4f}]")
    
    return X_train, y_train, X_val, y_val

def main():
    """Main debug function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting GRU debug session...")
    
    # Load config
    config = load_config()
    
    # Override config for debugging
    config['models']['gru']['learning_rate'] = 0.0001
    config['models']['gru']['batch_size'] = 16
    config['models']['gru']['epochs'] = 10
    config['models']['gru']['hidden_size'] = 16  # Smaller model
    config['training']['mixed_precision'] = False
    
    # Create synthetic data
    X_train, y_train, X_val, y_val = create_synthetic_data()
    
    # Initialize trainer
    trainer = GRUTrainer(config)
    
    # Build model
    input_size = X_train.shape[-1]
    trainer.build_model(input_size)
    
    logger.info("Starting training with synthetic data...")
    
    try:
        # Train model
        results = trainer.train(X_train, y_train, X_val, y_val)
        logger.info(f"Training completed successfully! Best loss: {results.get('best_val_loss', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
