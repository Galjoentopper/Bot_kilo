"""
Test script to verify all fixes for the crypto trading bot.
"""

import os
import sys
import pandas as pd
import numpy as np
from src.models.ppo_trainer import PPOTrainer
from src.models.gru_trainer import GRUTrainer
from src.models.lgbm_trainer import LightGBMTrainer
from src.data_pipeline.preprocess import DataPreprocessor
from src.data_pipeline.features import FeatureEngine
from src.data_pipeline.loader import DataLoader

def test_model_loading():
    """Test that all models can be loaded correctly."""
    print("Testing model loading...")
    
    # Check if model files exist
    model_files = [
        "models/gru_model_20250807_125954.pth",
        "models/lightgbm_model_20250807_130003.pkl",
        "models/ppo_model_20250807_132238.zip"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"PASS: Found model file: {model_file}")
        else:
            print(f"FAIL: Missing model file: {model_file}")
    
    # Test PPO model loading (with our fix)
    try:
        # This should work now with our fix
        ppo_trainer = PPOTrainer.load_model("models/ppo_model_20250807_132238", {})
        print("PASS: PPO model loaded successfully")
    except FileNotFoundError as e:
        print(f"FAIL: PPO model loading failed: {e}")
    except Exception as e:
        print(f"FAIL: PPO model loading failed with unexpected error: {e}")
    
    print()

def test_preprocessing():
    """Test data preprocessing improvements."""
    print("Testing data preprocessing...")
    
    # Create sample data with NaN values
    sample_data = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'feature2': [np.nan, 2.0, 3.0, np.nan, 5.0],
        'feature3': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    
    preprocessor = DataPreprocessor()
    
    try:
        # This should work with our improved error handling
        transformed = preprocessor.fit_transform(sample_data)
        print("PASS: Preprocessing handled NaN values successfully")
        print(f"  Transformed data shape: {transformed.shape}")
    except Exception as e:
        print(f"FAIL: Preprocessing failed: {e}")
    
    print()

def test_feature_engine():
    """Test feature engineering improvements."""
    print("Testing feature engineering...")
    
    # Create sample OHLCV data
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    sample_data = pd.DataFrame({
        'open': np.random.rand(100) * 100 + 50,
        'high': np.random.rand(100) * 100 + 51,
        'low': np.random.rand(100) * 100 + 49,
        'close': np.random.rand(100) * 100 + 50,
        'volume': np.random.rand(100) * 1000 + 100,
        'quote_volume': np.random.rand(100) * 10000 + 1000,
        'trades': np.random.rand(100) * 100 + 10,
        'taker_buy_base': np.random.rand(100) * 500 + 50,
        'taker_buy_quote': np.random.rand(100) * 5000 + 500
    }, index=dates)
    
    feature_engine = FeatureEngine()
    
    try:
        # Generate features
        features_df = feature_engine.generate_all_features(sample_data)
        print("PASS: Feature engineering completed successfully")
        print(f"  Original columns: {len(sample_data.columns)}")
        print(f"  Generated features: {len(features_df.columns) - len(sample_data.columns)}")
        print(f"  Final columns: {len(features_df.columns)}")
        
        # Check for NaN values
        nan_count = features_df.isnull().sum().sum()
        print(f"  NaN values in features: {nan_count}")
        
    except Exception as e:
        print(f"FAIL: Feature engineering failed: {e}")
    
    print()

def main():
    """Run all tests."""
    print("Running Crypto Trading Bot Fixes Verification")
    print("=" * 50)
    
    test_model_loading()
    test_preprocessing()
    test_feature_engine()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()