"""
Test Script for Trainer Improvements
====================================

Tests all the implemented improvements to ensure they work correctly.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import modules to test
from src.data_pipeline.dataset_builder import DatasetBuilder
from src.utils.cross_validation import PurgedTimeSeriesSplit, create_time_series_splits
from src.utils.metrics import TradingMetrics, find_optimal_threshold
from src.utils.calibration import ProbabilityCalibrator
from src.models.adapters import create_model_adapter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
    """Create synthetic test data."""
    # Create time index
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        periods=n_samples,
        freq='15min'
    )
    
    # Create OHLCV data
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    
    data = pd.DataFrame({
        'open': close + np.random.randn(n_samples) * 0.1,
        'high': close + np.abs(np.random.randn(n_samples) * 0.2),
        'low': close - np.abs(np.random.randn(n_samples) * 0.2),
        'close': close,
        'volume': np.random.lognormal(10, 1, n_samples),
        'quote_volume': np.random.lognormal(12, 1, n_samples),
        'trades': np.random.poisson(100, n_samples),
        'taker_buy_base': np.random.lognormal(9, 1, n_samples),
        'taker_buy_quote': np.random.lognormal(11, 1, n_samples)
    }, index=dates)
    
    return data


def test_dataset_builder():
    """Test DatasetBuilder functionality."""
    logger.info("\n" + "="*60)
    logger.info("Testing DatasetBuilder")
    logger.info("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_data = create_test_data()
        
        # Save test data in the expected format
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        symbol = 'TESTUSDT'
        
        # Save as SQLite database (expected format)
        import sqlite3
        db_path = os.path.join(data_dir, f"{symbol.lower()}_15m.db")
        conn = sqlite3.connect(db_path)
        
        # Reset index to have timestamp column
        test_data_db = test_data.reset_index()
        test_data_db.rename(columns={'index': 'timestamp'}, inplace=True)
        test_data_db['datetime'] = test_data_db['timestamp']
        
        # Save with correct table name
        test_data_db.to_sql('market_data', conn, if_exists='replace', index=False)
        conn.close()
        
        # Create DatasetBuilder with empty config
        builder = DatasetBuilder(
            data_dir=data_dir,
            cache_dir=os.path.join(temp_dir, 'cache'),
            config={}
        )
        
        # Test building dataset
        X, y, timestamps, feature_names, metadata = builder.build_dataset(
            symbol=symbol,
            interval='15m',
            use_cache=True,
            target_type='return',
            target_horizon=1
        )
        
        logger.info(f"✓ Built dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"✓ Feature names: {len(feature_names)} features")
        logger.info(f"✓ Metadata keys: {list(metadata.keys())}")
        
        # Test caching
        X2, y2, _, _, _ = builder.build_dataset(
            symbol=symbol,
            interval='15m',
            use_cache=True,
            target_type='return',
            target_horizon=1
        )
        
        assert X.shape == X2.shape, "Cached data shape mismatch"
        logger.info("✓ Cache working correctly")
        
        # Test validation
        is_valid, errors = builder.validate_dataset(X, y, metadata)
        logger.info(f"✓ Dataset validation: {'PASSED' if is_valid else 'FAILED'}")
        if errors:
            logger.warning(f"  Validation errors: {errors}")
        
        # Test cache info
        cache_info = builder.get_cache_info()
        logger.info(f"✓ Cache info: {cache_info['total_cached_datasets']} datasets, "
                   f"{cache_info['total_size_mb']:.2f} MB")
    
    logger.info("✓ DatasetBuilder tests PASSED")


def test_cross_validation():
    """Test cross-validation functionality."""
    logger.info("\n" + "="*60)
    logger.info("Testing Cross-Validation")
    logger.info("="*60)
    
    n_samples = 1000
    
    # Test PurgedTimeSeriesSplit
    cv = PurgedTimeSeriesSplit(n_splits=5, gap=10, embargo=10)
    X_dummy = np.zeros((n_samples, 1))
    
    splits = list(cv.split(X_dummy))
    logger.info(f"✓ Created {len(splits)} CV splits")
    
    # Verify no overlap
    for i, (train_idx, test_idx) in enumerate(splits):
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Overlap found in split {i}"
        
        # Verify temporal order with gap
        if len(train_idx) > 0 and len(test_idx) > 0:
            assert max(train_idx) < min(test_idx) - cv.gap, \
                f"Insufficient gap in split {i}"
    
    logger.info("✓ No data leakage detected")
    
    # Test automatic split creation
    splits_auto = create_time_series_splits(
        n_samples=n_samples,
        n_splits=5,
        test_ratio=0.2,
        gap_ratio=0.02,
        embargo_ratio=0.02
    )
    
    logger.info(f"✓ Auto-created {len(splits_auto)} splits")
    logger.info("✓ Cross-validation tests PASSED")


def test_metrics():
    """Test cost-aware metrics."""
    logger.info("\n" + "="*60)
    logger.info("Testing Cost-Aware Metrics")
    logger.info("="*60)
    
    # Create test data
    n_samples = 100
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    positions = np.random.choice([-1, 0, 1], size=n_samples)
    
    # Test metrics calculation
    metrics_calc = TradingMetrics(fee_bps=10, slippage_bps=5)
    
    # Calculate metrics with costs
    metrics_with_costs = metrics_calc.calculate_all_metrics(prices, positions, include_costs=True)
    metrics_without_costs = metrics_calc.calculate_all_metrics(prices, positions, include_costs=False)
    
    logger.info("✓ Metrics with costs:")
    for key, value in metrics_with_costs.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
    
    # Verify costs are applied
    assert metrics_with_costs['net_return'] < metrics_without_costs['net_return'], \
        "Costs not properly applied"
    logger.info("✓ Transaction costs properly applied")
    
    # Test threshold optimization
    probabilities = np.random.rand(n_samples)
    optimal_threshold, optimal_metrics = find_optimal_threshold(
        probabilities, prices, metrics_calc, metric='sharpe_ratio'
    )
    
    logger.info(f"✓ Optimal threshold: {optimal_threshold:.3f}")
    logger.info(f"✓ Sharpe ratio at optimal: {optimal_metrics['sharpe_ratio']:.3f}")
    logger.info("✓ Metrics tests PASSED")


def test_calibration():
    """Test probability calibration."""
    logger.info("\n" + "="*60)
    logger.info("Testing Probability Calibration")
    logger.info("="*60)
    
    # Create test data
    n_samples = 1000
    np.random.seed(42)
    
    # Create miscalibrated probabilities
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_prob_raw = np.random.beta(2, 5, n_samples)  # Miscalibrated
    
    # Test isotonic calibration
    calibrator = ProbabilityCalibrator(method='isotonic')
    calibrator.fit(y_true, y_prob_raw)
    y_prob_cal = calibrator.transform(y_prob_raw)
    
    logger.info(f"✓ Calibration stats: {calibrator.calibration_stats}")
    logger.info(f"✓ ECE before: {calibrator.calibration_stats['ece']:.4f}")
    
    # Verify calibration improves probabilities
    assert np.mean(np.abs(y_prob_cal - y_true)) <= np.mean(np.abs(y_prob_raw - y_true)), \
        "Calibration did not improve probabilities"
    
    logger.info("✓ Calibration improved probability estimates")
    
    # Test save/load
    with tempfile.TemporaryDirectory() as temp_dir:
        cal_path = os.path.join(temp_dir, 'calibrator')
        calibrator.save(cal_path)
        
        loaded_cal = ProbabilityCalibrator.load(cal_path)
        y_prob_loaded = loaded_cal.transform(y_prob_raw)
        
        assert np.allclose(y_prob_cal, y_prob_loaded), "Loaded calibrator produces different results"
        logger.info("✓ Calibrator save/load working")
    
    logger.info("✓ Calibration tests PASSED")


def test_model_adapters():
    """Test model adapter interface."""
    logger.info("\n" + "="*60)
    logger.info("Testing Model Adapters")
    logger.info("="*60)
    
    # Create test data
    n_samples = 500
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    train_idx = np.arange(400)
    val_idx = np.arange(400, 500)
    
    # Test config
    config = {
        'models': {
            'lightgbm': {
                'n_estimators': 10,
                'max_depth': 3
            }
        }
    }
    
    # Test LightGBM adapter (fastest to test)
    adapter = create_model_adapter('lightgbm', config, 'regression')
    
    # Train
    results = adapter.fit(X, y, train_idx, val_idx)
    logger.info("✓ Model training completed")
    
    # Predict
    y_pred = adapter.predict(X[val_idx])
    assert len(y_pred) == len(val_idx), "Prediction length mismatch"
    logger.info("✓ Model prediction working")
    
    # Test save/load
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save
        saved_path = adapter.save(temp_dir)
        logger.info(f"✓ Model saved to {saved_path}")
        
        # Load
        new_adapter = create_model_adapter('lightgbm', config, 'regression')
        new_adapter.load(saved_path)
        
        # Verify predictions are same
        y_pred_loaded = new_adapter.predict(X[val_idx])
        assert np.allclose(y_pred, y_pred_loaded), "Loaded model produces different predictions"
        logger.info("✓ Model load/save working correctly")
    
    logger.info("✓ Model adapter tests PASSED")


def test_integration():
    """Test integrated workflow."""
    logger.info("\n" + "="*60)
    logger.info("Testing Integrated Workflow")
    logger.info("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_data = create_test_data(n_samples=500)
        
        # Save test data in the expected format
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        symbol = 'TESTUSDT'
        
        # Save as SQLite database (expected format)
        import sqlite3
        db_path = os.path.join(data_dir, f"{symbol.lower()}_15m.db")
        conn = sqlite3.connect(db_path)
        
        # Reset index to have timestamp column
        test_data_db = test_data.reset_index()
        test_data_db.rename(columns={'index': 'timestamp'}, inplace=True)
        test_data_db['datetime'] = test_data_db['timestamp']
        
        # Save with correct table name
        test_data_db.to_sql('market_data', conn, if_exists='replace', index=False)
        conn.close()
        
        # 1. Build dataset
        builder = DatasetBuilder(data_dir=data_dir, config={})
        X, y, timestamps, feature_names, metadata = builder.build_dataset(
            symbol=symbol,
            interval='15m',
            target_type='direction'  # Classification task
        )
        
        logger.info(f"✓ Dataset built: {X.shape}")
        
        # 2. Create CV splits
        cv = PurgedTimeSeriesSplit(n_splits=3, gap=10, embargo=10)
        
        # 3. Train model with cross-validation
        config = {'models': {'lightgbm': {'n_estimators': 10}}}
        adapter = create_model_adapter('lightgbm', config, 'classification')
        
        metrics_calc = TradingMetrics(fee_bps=10, slippage_bps=5)
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            # Train
            adapter.fit(X, y, train_idx, val_idx)
            
            # Predict probabilities
            y_prob = adapter.predict_proba(X.iloc[val_idx])[:, 1]
            
            # Calibrate
            calibrator = ProbabilityCalibrator(method='isotonic')
            y_prob_cal = calibrator.fit_transform(y[val_idx], y_prob)
            
            logger.info(f"✓ Fold {fold_idx + 1} completed")
        
        logger.info("✓ Integrated workflow completed successfully")
    
    logger.info("✓ Integration tests PASSED")


def main():
    """Run all tests."""
    logger.info("Starting trainer improvement tests...")
    
    try:
        test_dataset_builder()
        test_cross_validation()
        test_metrics()
        test_calibration()
        test_model_adapters()
        test_integration()
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED! ✅")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()