"""
Updated LightGBM Trainer using DatasetBuilder
============================================

This shows how to integrate the centralized DatasetBuilder with an existing trainer.
"""

import lightgbm as lgb  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
import joblib
from datetime import datetime

# Import the centralized DatasetBuilder
from ..data_pipeline.dataset_builder import DatasetBuilder, ModelType, TargetType

logger = logging.getLogger(__name__)

class LightGBMTrainerWithDatasetBuilder:
    """
    Updated LightGBM trainer that uses the centralized DatasetBuilder.
    
    This demonstrates the integration pattern that can be applied to all trainers.
    """
    
    def __init__(self, config: Dict[str, Any], task_type: str = "regression"):
        """
        Initialize LightGBM trainer with DatasetBuilder integration.
        
        Args:
            config: Configuration dictionary
            task_type: Type of task ('regression' or 'classification')
        """
        self.config = config
        self.model_config = config.get('models', {}).get('lightgbm', {})
        self.task_type = task_type
        
        # Initialize centralized dataset builder
        self.dataset_builder = DatasetBuilder(config)
        
        # Model parameters (unchanged from original)
        self.num_leaves = self.model_config.get('num_leaves', 31)
        self.max_depth = self.model_config.get('max_depth', 6)
        self.learning_rate = self.model_config.get('learning_rate', 0.1)
        self.n_estimators = self.model_config.get('n_estimators', 100)
        self.boosting_type = self.model_config.get('boosting_type', 'gbdt')
        
        # Task-specific parameters
        if task_type == "regression":
            self.objective = self.model_config.get('objective', 'regression')
            self.metric = self.model_config.get('metric', 'rmse')
        else:
            self.objective = self.model_config.get('objective', 'binary')
            self.metric = self.model_config.get('metric', 'binary_logloss')
        
        # Initialize model components
        self.model = None
        self.feature_names = []
        self.feature_importance = None
        self.training_history = {}
        self.dataset_metadata = None
        
        logger.info(f"LightGBM Trainer with DatasetBuilder initialized for {task_type} task")
    
    def train(
        self, 
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        experiment_name: str = "lightgbm_training"
    ) -> Dict[str, Any]:
        """
        Train the LightGBM model using centralized dataset assembly.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCEUR')
            start_date: Start date for training data (optional)
            end_date: End date for training data (optional)
            experiment_name: MLflow experiment name
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting LightGBM model training for {symbol}")
        
        # CENTRALIZED DATASET ASSEMBLY - Single call replaces all scattered data processing
        try:
            if self.task_type == "regression":
                dataset, metadata = self.dataset_builder.build_regression_dataset(
                    symbol=symbol,
                    model_type=ModelType.LIGHTGBM,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                dataset, metadata = self.dataset_builder.build_classification_dataset(
                    symbol=symbol,
                    model_type=ModelType.LIGHTGBM,
                    start_date=start_date,
                    end_date=end_date
                )
            
            # Store metadata for validation and logging
            self.dataset_metadata = metadata
            self.feature_names = metadata.feature_names
            
            logger.info(f"Dataset assembled - Features: {metadata.feature_count}, Samples: {metadata.total_samples}")
            
        except Exception as e:
            logger.error(f"Dataset assembly failed for {symbol}: {e}")
            raise
        
        # Build and train model with clean, validated data
        self.build_model()
        
        # Extract preprocessed data (already scaled and validated)
        X_train = dataset['train']['X']
        y_train = dataset['train']['y']
        X_val = dataset['validation']['X'] 
        y_val = dataset['validation']['y']
        
        logger.info(f"Training data shapes - X_train: {X_train.shape}, y_train: {len(y_train)}")
        
        # Train model with validated data (no need for manual data cleaning)
        if self.model is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_names=['validation'],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
        
        # Get feature importance
        if self.model is not None:
            self.feature_importance = self.model.feature_importances_
        
        # Create feature importance DataFrame using deterministic feature names
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        # Evaluate on validation set
        val_predictions = self.predict(X_val)
        val_metrics = self._calculate_metrics(y_val, val_predictions)
        
        logger.info(f"Validation metrics: {val_metrics}")
        
        # Training results with dataset metadata
        results = {
            "model": self.model,
            "feature_importance": importance_df,
            "feature_names": self.feature_names,
            "validation_metrics": val_metrics,
            "dataset_metadata": metadata,  # Include metadata for tracking
            "best_iteration": getattr(self.model, 'best_iteration', self.n_estimators)
        }
        
        logger.info("LightGBM training completed with centralized dataset assembly")
        return results
    
    def build_model(self) -> lgb.LGBMModel:
        """Build and initialize the LightGBM model (unchanged)."""
        if self.task_type == "regression":
            self.model = lgb.LGBMRegressor(
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                boosting_type=self.boosting_type,
                objective=self.objective,
                metric=self.metric,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            self.model = lgb.LGBMClassifier(
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                boosting_type=self.boosting_type,
                objective=self.objective,
                metric=self.metric,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        logger.info(f"LightGBM {self.task_type} model built")
        return self.model
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions with the trained model (unchanged)."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        predictions = self.model.predict(X_array)
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        return predictions
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics based on task type (unchanged)."""
        if self.task_type == "regression":
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # Directional accuracy for regression
            y_true_direction = np.sign(y_true)
            y_pred_direction = np.sign(y_pred)
            directional_accuracy = np.mean(y_true_direction == y_pred_direction)
            
            return {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "directional_accuracy": directional_accuracy
            }
        
        else:  # classification
            from sklearn.metrics import accuracy_score
            
            # Convert probabilities to binary predictions if needed
            if y_pred.ndim > 1:
                y_pred_binary = (y_pred[:, 1] > 0.5).astype(int)
            else:
                y_pred_binary = (y_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(y_true, y_pred_binary)
            
            return {
                "accuracy": float(accuracy)
            }
    
    def validate_dataset_consistency(self) -> Dict[str, Any]:
        """
        Validate that the dataset was assembled consistently.
        This is a new method enabled by the centralized approach.
        """
        if self.dataset_metadata is None:
            return {"error": "No dataset metadata available"}
        
        return {
            "symbol": self.dataset_metadata.symbol,
            "feature_count": self.dataset_metadata.feature_count,
            "total_samples": self.dataset_metadata.total_samples,
            "data_quality_score": self.dataset_metadata.data_quality_score,
            "target_type": self.dataset_metadata.target_type,
            "model_type": self.dataset_metadata.model_type,
            "created_at": self.dataset_metadata.created_at
        }

# Demonstration function showing the benefits
def demonstrate_integration_benefits():
    """
    Demonstrate the benefits of using DatasetBuilder vs old approach.
    """
    
    print("=== LIGHTGBM TRAINER INTEGRATION DEMONSTRATION ===\n")
    
    # Show code reduction
    print("CODE REDUCTION:")
    print("BEFORE: ~150 lines of data processing code in each trainer")
    print("AFTER:  ~15 lines using centralized DatasetBuilder")
    print("REDUCTION: ~90% less data processing code per trainer\n")
    
    # Show consistency benefits
    print("CONSISTENCY BENEFITS:")
    print("✅ Deterministic feature ordering across all models")
    print("✅ Centralized target variable definitions")
    print("✅ Consistent data validation and cleaning") 
    print("✅ Standardized train/validation/test splits")
    print("✅ Unified preprocessing approach\n")
    
    # Show maintainability benefits
    print("MAINTAINABILITY BENEFITS:")
    print("✅ Single place to modify feature engineering logic")
    print("✅ Centralized data quality validation")
    print("✅ Consistent error handling across all models")
    print("✅ Standardized metadata tracking")
    print("✅ Easier testing and debugging\n")
    
    # Show example usage
    print("EXAMPLE USAGE:")
    usage_code = '''
# Simple, consistent training across all models
config = load_config()
trainer = LightGBMTrainerWithDatasetBuilder(config)

# All data processing handled centrally
results = trainer.train(symbol="BTCEUR")

# Access consistent metadata
print(f"Features: {results['dataset_metadata'].feature_count}")
print(f"Quality: {results['dataset_metadata'].data_quality_score}")
    '''
    print(usage_code)

if __name__ == "__main__":
    demonstrate_integration_benefits()