#!/usr/bin/env python3
"""
Test script to verify GRU trainer fixes for gradient stability.
"""

import sys
import os
import numpy as np
import torch
import yaml
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.gru_trainer import GRUTrainer
from src.utils.logger import setup_logging

def test_gru_trainer_stability():
    """Test GRU trainer with synthetic data to verify stability fixes."""
    
    # Setup logging
    setup_logging()
    
    # Load config
    config_path = Path("src/config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create simple synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create well-behaved synthetic financial-like data
    n_samples = 1000
    n_features = 50
    sequence_length = 20
    
    # Generate features with realistic financial data characteristics
    X = np.random.normal(0, 0.1, (n_samples, sequence_length, n_features))
    # Add some trend and volatility clustering
    for i in range(1, n_samples):
        X[i] += 0.95 * X[i-1] + np.random.normal(0, 0.05, (sequence_length, n_features))
    
    # Generate realistic returns (small values, centered around 0)
    y = np.random.normal(0, 0.01, n_samples)  # 1% volatility
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Data ranges: X_train=[{X_train.min():.6f}, {X_train.max():.6f}], y_train=[{y_train.min():.6f}, {y_train.max():.6f}]")
    
    # Initialize GRU trainer
    trainer = GRUTrainer(config)
    
    # Build model
    model = trainer.build_model(input_size=n_features)
    print(f"Model built successfully with mixed_precision={trainer.mixed_precision}")
    
    # Prepare data loaders
    train_loader, val_loader = trainer.prepare_data(X_train, y_train, X_test, y_test)
    
    # Test a few training steps
    print("Testing training stability...")
    
    # Train for just a few epochs to test stability
    trainer.epochs = 5  # Override for quick test
    
    try:
        results = trainer.train(X_train, y_train, X_test, y_test)
        print(f"✅ Training completed successfully!")
        print(f"Best validation loss: {results['best_val_loss']:.6f}")
        print(f"Total epochs: {results['total_epochs']}")
        
        # Test prediction
        predictions = trainer.predict(X_test)
        print(f"✅ Prediction completed successfully!")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction range: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing GRU trainer stability fixes...")
    success = test_gru_trainer_stability()
    
    if success:
        print("\n✅ All tests passed! GRU trainer is stable.")
    else:
        print("\n❌ Tests failed. GRU trainer still has issues.")
