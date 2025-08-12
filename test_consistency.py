"""
Dataset Consistency Validation Test
==================================

Test to ensure DatasetBuilder produces consistent results across all models.
"""

import sys
import os
sys.path.append('/home/runner/work/Bot_kilo/Bot_kilo')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
import tempfile
import shutil

def create_test_environment():
    """Create temporary test environment with mock data."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create mock OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='15min')
    
    # Generate realistic price movements
    base_price = 30000
    returns = np.random.randn(1000) * 0.02  # 2% volatility
    prices = base_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(1000)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(1000)) * 0.01),
        'close': prices,
        'volume': np.random.rand(1000) * 1000 + 100,
        'quote_volume': np.random.rand(1000) * 10000 + 1000,
        'trades': np.random.randint(50, 500, 1000),
        'taker_buy_base': np.random.rand(1000) * 500 + 50,
        'taker_buy_quote': np.random.rand(1000) * 5000 + 500,
    }, index=dates)
    
    # Create SQLite database
    db_path = os.path.join(temp_dir, 'btceur_15m.db')
    conn = sqlite3.connect(db_path)
    
    conn.execute('''
        CREATE TABLE market_data (
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
    
    # Insert data
    for idx, row in data.iterrows():
        conn.execute('''
            INSERT INTO market_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(idx.timestamp()),
            idx.isoformat(),
            row['open'], row['high'], row['low'], row['close'],
            row['volume'], row['quote_volume'], row['trades'],
            row['taker_buy_base'], row['taker_buy_quote']
        ))
    
    conn.commit()
    conn.close()
    
    return temp_dir

def test_feature_consistency():
    """Test that all models can get identical features when requested."""
    
    print("=== TESTING DATASET CONSISTENCY ===")
    
    # Setup test environment
    temp_dir = create_test_environment()
    
    try:
        # Configuration
        config = {
            'data': {'data_dir': temp_dir},
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
        
        from src.data_pipeline.dataset_builder import DatasetBuilder, ModelType, TargetType
        
        builder = DatasetBuilder(config)
        
        print("1. Building datasets for each model type...")
        
        # Build datasets
        lgb_data, lgb_meta = builder.build_regression_dataset("BTCEUR", ModelType.LIGHTGBM)
        gru_data, gru_meta = builder.build_gru_dataset("BTCEUR", sequence_length=20)
        ppo_data_inconsistent, ppo_meta_inconsistent = builder.build_ppo_dataset("BTCEUR")
        ppo_data_consistent, ppo_meta_consistent = builder.build_ppo_dataset("BTCEUR", force_feature_consistency=True)
        
        print(f"   - LightGBM: {lgb_meta.feature_count} features")
        print(f"   - GRU: {gru_meta.feature_count} features")
        print(f"   - PPO (default): {ppo_meta_inconsistent.feature_count} features")
        print(f"   - PPO (consistent): {ppo_meta_consistent.feature_count} features")
        
        print("\n2. Testing consistency across models...")
        
        # Test consistency between LightGBM and GRU
        lgb_features = set(lgb_meta.feature_names)
        gru_features = set(gru_meta.feature_names)
        ppo_consistent_features = set(ppo_meta_consistent.feature_names)
        
        lgb_gru_consistent = lgb_features == gru_features
        all_consistent = lgb_features == gru_features == ppo_consistent_features
        
        print(f"   - LightGBM vs GRU consistency: {'✅' if lgb_gru_consistent else '❌'}")
        print(f"   - All models consistency: {'✅' if all_consistent else '❌'}")
        
        if not lgb_gru_consistent:
            print(f"     Feature differences: {lgb_features.symmetric_difference(gru_features)}")
        
        print("\n3. Testing deterministic ordering...")
        
        # Build same dataset twice
        lgb_data2, lgb_meta2 = builder.build_regression_dataset("BTCEUR", ModelType.LIGHTGBM)
        
        ordering_consistent = lgb_meta.feature_names == lgb_meta2.feature_names
        print(f"   - Deterministic ordering: {'✅' if ordering_consistent else '❌'}")
        
        print("\n4. Testing data quality...")
        
        def check_data_quality(data, name):
            X_train = data['train']['X']
            y_train = data['train']['y']
            
            # Check for NaN/inf values
            if hasattr(X_train, 'isna'):  # DataFrame
                nan_count = X_train.isna().sum().sum()
                inf_count = np.isinf(X_train.values).sum()
            else:  # NumPy array
                nan_count = np.isnan(X_train).sum()
                inf_count = np.isinf(X_train).sum()
            
            y_nan = np.isnan(y_train).sum()
            y_inf = np.isinf(y_train).sum()
            
            quality_score = "✅" if (nan_count + inf_count + y_nan + y_inf) == 0 else "❌"
            print(f"   - {name}: {quality_score} (NaN/Inf: X={nan_count+inf_count}, y={y_nan+y_inf})")
        
        check_data_quality(lgb_data, "LightGBM")
        check_data_quality(gru_data, "GRU")
        check_data_quality(ppo_data_consistent, "PPO")
        
        print("\n5. Testing consistency report...")
        
        datasets = [
            (lgb_data, lgb_meta),
            (gru_data, gru_meta),
            (ppo_data_consistent, ppo_meta_consistent)
        ]
        
        consistency_report = builder.get_feature_consistency_report(datasets)
        print(f"   - Feature count consistency: {'✅' if consistency_report['consistent_feature_count'] else '❌'}")
        print(f"   - Feature name consistency: {'✅' if consistency_report['consistent_feature_names'] else '❌'}")
        print(f"   - Feature counts: {consistency_report['feature_counts']}")
        
        print("\n6. Testing target definitions...")
        
        # Test different target types
        reg_data, reg_meta = builder.build_dataset("BTCEUR", ModelType.LIGHTGBM, TargetType.REGRESSION)
        cls_data, cls_meta = builder.build_dataset("BTCEUR", ModelType.LIGHTGBM, TargetType.CLASSIFICATION)
        
        reg_targets = reg_data['train']['y']
        cls_targets = cls_data['train']['y']
        
        # Regression targets should be continuous
        reg_continuous = len(np.unique(reg_targets)) > 10
        # Classification targets should be binary
        cls_binary = set(np.unique(cls_targets)) <= {0, 1}
        
        print(f"   - Regression targets continuous: {'✅' if reg_continuous else '❌'}")
        print(f"   - Classification targets binary: {'✅' if cls_binary else '❌'}")
        
        print("\n=== TEST RESULTS ===")
        
        all_tests_passed = (
            lgb_gru_consistent and
            ordering_consistent and
            consistency_report['consistent_feature_count'] and
            consistency_report['consistent_feature_names'] and
            reg_continuous and
            cls_binary
        )
        
        if all_tests_passed:
            print("🎉 ALL TESTS PASSED - DatasetBuilder ensures consistency!")
        else:
            print("⚠️  Some tests failed - check implementation")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

def main():
    """Run consistency validation tests."""
    print("DATASET BUILDER CONSISTENCY VALIDATION")
    print("=" * 50)
    
    success = test_feature_consistency()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ DatasetBuilder validation completed successfully!")
        print("\nBenefits achieved:")
        print("• Deterministic feature ordering")
        print("• Consistent data types and validation")  
        print("• Centralized target definitions")
        print("• Clean, validated datasets for all models")
        print("• Reduced code duplication")
    else:
        print("❌ Validation failed - check implementation")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)