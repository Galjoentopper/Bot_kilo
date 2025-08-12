"""
Integration Example: Using DatasetBuilder with Existing Trainers
===============================================================

This example shows how to modify existing model trainers to use the centralized DatasetBuilder.
"""

import sys
import os
sys.path.append('/home/runner/work/Bot_kilo/Bot_kilo')

import numpy as np
import pandas as pd
from typing import Dict, Any

def example_lightgbm_trainer_integration():
    """
    Example showing how to integrate DatasetBuilder with LightGBM trainer.
    
    BEFORE: Each trainer loads data, generates features, preprocesses separately
    AFTER: Use centralized DatasetBuilder for consistent data preparation
    """
    
    print("=== LightGBM Trainer Integration Example ===")
    
    # Mock configuration (would normally come from config file)
    config = {
        'data': {'data_dir': './data'},
        'features': {
            'technical_indicators': {
                'sma_periods': [5, 10, 20, 50],
                'ema_periods': [5, 10, 20, 50],
                'rsi_period': 14
            }
        },
        'models': {
            'lightgbm': {
                'num_leaves': 31,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
        }
    }
    
    # BEFORE (old way - scattered across trainer):
    print("\nOLD APPROACH:")
    print("1. data_loader = DataLoader()")
    print("2. raw_data = data_loader.load_symbol_data(symbol)")
    print("3. feature_engine = FeatureEngine()")
    print("4. features = feature_engine.generate_all_features(raw_data)")
    print("5. preprocessor = DataPreprocessor()")
    print("6. X = preprocessor.fit_transform(features)")
    print("7. y = prepare_targets(...)")
    print("8. model.train(X, y)")
    
    # AFTER (new way - centralized):
    print("\nNEW APPROACH:")
    print("1. builder = DatasetBuilder(config)")
    print("2. dataset, metadata = builder.build_regression_dataset(symbol, ModelType.LIGHTGBM)")
    print("3. model.train(dataset['train']['X'], dataset['train']['y'])")
    
    print("\nBENEFITS:")
    print("- Consistent feature engineering across all models")
    print("- Deterministic column ordering")
    print("- Centralized target definitions")
    print("- Reduced code duplication")
    print("- Better data validation")
    
def example_trainer_refactor():
    """
    Example of refactoring an existing trainer method to use DatasetBuilder.
    """
    
    print("\n=== Trainer Refactor Example ===")
    
    # BEFORE: Old trainer method structure
    print("\nBEFORE (scattered data processing):")
    old_code = '''
def train_model(self, symbol: str, config: Dict[str, Any]):
    # Each trainer repeats this pattern
    data_loader = DataLoader()
    raw_data = data_loader.load_symbol_data(symbol)
    
    feature_engine = FeatureEngine()
    features_df = feature_engine.generate_all_features(raw_data)
    
    # Model-specific feature adjustments
    feature_names = feature_engine.get_feature_names(features_df)
    X = features_df[feature_names]
    
    preprocessor = DataPreprocessor()
    y = preprocessor.prepare_target_variable(features_df, "return")
    
    # Manual data cleaning and validation
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Train/val/test split
    train_X, val_X, test_X = create_splits(X)
    train_y, val_y, test_y = create_splits(y)
    
    # Preprocessing
    train_X_scaled = preprocessor.fit_transform(train_X)
    val_X_scaled = preprocessor.transform(val_X)
    
    self.model.train(train_X_scaled, train_y, val_X_scaled, val_y)
    '''
    print(old_code)
    
    # AFTER: Simplified trainer using DatasetBuilder
    print("\nAFTER (centralized dataset assembly):")
    new_code = '''
def train_model(self, symbol: str, config: Dict[str, Any]):
    # Single call to get consistent, validated data
    builder = DatasetBuilder(config)
    dataset, metadata = builder.build_regression_dataset(
        symbol=symbol, 
        model_type=self.model_type
    )
    
    # Data is already preprocessed, validated, and split
    self.model.train(
        dataset['train']['X'], dataset['train']['y'],
        dataset['validation']['X'], dataset['validation']['y']
    )
    
    # Access metadata for logging/validation
    self.log_training_info(metadata)
    '''
    print(new_code)
    
def example_feature_consistency_validation():
    """
    Example showing how to validate feature consistency across models.
    """
    
    print("\n=== Feature Consistency Validation Example ===")
    
    validation_code = '''
# Validate that all models get identical features
def validate_model_consistency(symbols: List[str], config: Dict[str, Any]):
    builder = DatasetBuilder(config)
    datasets = []
    
    for symbol in symbols:
        # Build datasets for different model types
        lgb_data, lgb_meta = builder.build_regression_dataset(symbol, ModelType.LIGHTGBM)
        gru_data, gru_meta = builder.build_gru_dataset(symbol, sequence_length=20)
        ppo_data, ppo_meta = builder.build_ppo_dataset(symbol, force_feature_consistency=True)
        
        datasets.extend([(lgb_data, lgb_meta), (gru_data, gru_meta), (ppo_data, ppo_meta)])
    
    # Generate consistency report
    consistency_report = builder.get_feature_consistency_report(datasets)
    
    if not consistency_report['consistent_feature_names']:
        raise ValueError("Feature inconsistency detected across models!")
    
    print(f"✅ All {len(datasets)} datasets have consistent features")
    return consistency_report
    '''
    print(validation_code)

def example_migration_steps():
    """
    Step-by-step migration guide for existing trainers.
    """
    
    print("\n=== Migration Steps for Existing Trainers ===")
    
    steps = [
        "1. IDENTIFY DATA PROCESSING CODE",
        "   - Find where each trainer loads data",
        "   - Locate feature generation calls", 
        "   - Find preprocessing and target preparation",
        "",
        "2. REPLACE WITH DATASET BUILDER",
        "   - Import DatasetBuilder and ModelType enums",
        "   - Replace data loading with builder.build_*_dataset() calls",
        "   - Remove redundant preprocessing code",
        "",
        "3. UPDATE MODEL TRAINING CALLS",
        "   - Use dataset['train']['X'] and dataset['train']['y']",
        "   - Access validation data via dataset['validation']",
        "   - Use preprocessor from dataset['preprocessor'] for predictions",
        "",
        "4. ADD METADATA LOGGING",
        "   - Log feature counts and names for validation",
        "   - Track data quality metrics from metadata",
        "   - Store dataset metadata with trained models",
        "",
        "5. TEST CONSISTENCY",
        "   - Run consistency validation across all models",
        "   - Verify identical feature sets (if desired)",
        "   - Check for data quality improvements"
    ]
    
    for step in steps:
        print(step)

def main():
    """Run all integration examples."""
    print("DATASET BUILDER INTEGRATION EXAMPLES")
    print("=" * 50)
    
    example_lightgbm_trainer_integration()
    example_trainer_refactor()
    example_feature_consistency_validation()
    example_migration_steps()
    
    print("\n" + "=" * 50)
    print("Integration examples completed!")
    print("\nNext steps:")
    print("1. Update actual trainer classes to use DatasetBuilder")
    print("2. Run consistency validation tests") 
    print("3. Verify model performance is maintained")
    print("4. Add dataset metadata tracking to MLflow")

if __name__ == "__main__":
    main()