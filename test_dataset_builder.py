"""
Test Dataset Builder
==================

Simple test to verify the DatasetBuilder works correctly and produces consistent results.
"""

import sys
import os
sys.path.append('/home/runner/work/Bot_kilo/Bot_kilo')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Mock data for testing since we don't have real data available
def create_mock_data(symbol: str = "BTCEUR", n_samples: int = 1000) -> pd.DataFrame:
    """Create mock OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15T')
    
    # Generate mock price data
    base_price = 30000 if symbol == "BTCEUR" else 2000
    price_changes = np.random.randn(n_samples) * 0.01  # 1% volatility
    prices = base_price * np.cumprod(1 + price_changes)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_samples)) * 0.005),
        'low': prices * (1 - np.abs(np.random.randn(n_samples)) * 0.005),
        'close': prices,
        'volume': np.random.rand(n_samples) * 1000 + 100,
        'quote_volume': np.random.rand(n_samples) * 10000 + 1000,
        'trades': np.random.randint(50, 500, n_samples),
        'taker_buy_base': np.random.rand(n_samples) * 500 + 50,
        'taker_buy_quote': np.random.rand(n_samples) * 5000 + 500,
    }, index=dates)
    
    return data

def test_dataset_builder():
    """Test the DatasetBuilder functionality."""
    print("Testing DatasetBuilder...")
    
    # Mock config
    config = {
        'data': {
            'data_dir': './test_data',
            'symbols': ['BTCEUR']
        },
        'features': {
            'technical_indicators': {
                'sma_periods': [5, 10, 20],
                'ema_periods': [5, 10, 20],
                'rsi_period': 14
            },
            'price_features': {
                'returns_periods': [1, 5],
                'volatility_periods': [10, 20]
            },
            'time_features': {
                'include_hour': True,
                'include_day_of_week': True
            }
        }
    }
    
    # Create test directory and mock data
    os.makedirs('./test_data', exist_ok=True)
    
    # Create mock database file with test data
    import sqlite3
    mock_data = create_mock_data("BTCEUR", 500)
    
    db_path = './test_data/btceur_15m.db'
    conn = sqlite3.connect(db_path)
    
    # Create table structure
    conn.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp INTEGER,
            datetime TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            quote_volume REAL,
            trades INTEGER,
            taker_buy_base REAL,
            taker_buy_quote REAL
        )
    ''')
    
    # Insert mock data
    for idx, row in mock_data.iterrows():
        conn.execute('''
            INSERT OR REPLACE INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(idx.timestamp()),
            idx.isoformat(),
            row['open'], row['high'], row['low'], row['close'],
            row['volume'], row['quote_volume'], row['trades'],
            row['taker_buy_base'], row['taker_buy_quote']
        ))
    
    conn.commit()
    conn.close()
    
    try:
        # Now test DatasetBuilder
        from src.data_pipeline.dataset_builder import DatasetBuilder, ModelType, TargetType
        
        # Initialize builder
        builder = DatasetBuilder(config)
        
        # Test 1: Build LightGBM regression dataset
        print("\n1. Testing LightGBM regression dataset...")
        lgb_dataset, lgb_metadata = builder.build_regression_dataset(
            symbol="BTCEUR",
            model_type=ModelType.LIGHTGBM
        )
        
        print(f"   - Training data shape: {lgb_dataset['train']['X'].shape}")
        print(f"   - Features: {lgb_metadata.feature_count}")
        print(f"   - Target type: {lgb_metadata.target_type}")
        
        # Test 2: Build GRU dataset
        print("\n2. Testing GRU dataset...")
        gru_dataset, gru_metadata = builder.build_gru_dataset(
            symbol="BTCEUR",
            sequence_length=20
        )
        
        print(f"   - Training data shape: {gru_dataset['train']['X'].shape}")
        print(f"   - Features: {gru_metadata.feature_count}")
        print(f"   - Target type: {gru_metadata.target_type}")
        
        # Test 3: Build PPO dataset
        print("\n3. Testing PPO dataset...")
        ppo_dataset, ppo_metadata = builder.build_ppo_dataset(
            symbol="BTCEUR"
        )
        
        print(f"   - Training data shape: {ppo_dataset['train']['X'].shape}")
        print(f"   - Features: {ppo_metadata.feature_count}")
        print(f"   - Target type: {ppo_metadata.target_type}")
        
        # Test 4: Feature consistency check
        print("\n4. Testing feature consistency...")
        consistency_report = builder.get_feature_consistency_report([
            (lgb_dataset, lgb_metadata),
            (ppo_dataset, ppo_metadata)
        ])
        
        print(f"   - Consistent feature counts: {consistency_report['consistent_feature_count']}")
        print(f"   - Consistent feature names: {consistency_report['consistent_feature_names']}")
        print(f"   - Feature counts: {consistency_report['feature_counts']}")
        
        # Test 5: Data validation
        print("\n5. Testing data validation...")
        
        # Check for NaN/inf values
        def check_data_quality(dataset, name):
            X = dataset['train']['X']
            y = dataset['train']['y']
            
            if hasattr(X, 'isna'):  # DataFrame
                nan_count = X.isna().sum().sum()
                inf_count = np.isinf(X.values).sum()
            else:  # NumPy array
                nan_count = np.isnan(X).sum()
                inf_count = np.isinf(X).sum()
            
            y_nan = np.isnan(y).sum() if hasattr(y, '__len__') else 0
            y_inf = np.isinf(y).sum() if hasattr(y, '__len__') else 0
            
            print(f"   - {name}: NaN in X: {nan_count}, Inf in X: {inf_count}, NaN in y: {y_nan}, Inf in y: {y_inf}")
        
        check_data_quality(lgb_dataset, "LightGBM")
        check_data_quality(gru_dataset, "GRU")
        check_data_quality(ppo_dataset, "PPO")
        
        print("\n✅ All tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup test files
        import shutil
        if os.path.exists('./test_data'):
            shutil.rmtree('./test_data')

if __name__ == "__main__":
    success = test_dataset_builder()
    exit(0 if success else 1)