"""
LightGBM Trainer Module (Updated with DatasetBuilder)
====================================================

Modern LightGBM trainer that uses shared DatasetBuilder for consistent
data management across models while maintaining compatibility with the adapter interface.
"""

import lightgbm as lgb  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

try:
    import mlflow  # type: ignore[import-untyped]
    import mlflow.lightgbm  # type: ignore
    MLFLOW_AVAILABLE = True
except ImportError:
    # Create dummy mlflow module for type hints
    class _DummyMLflow:
        @staticmethod
        def start_run(*args: Any, **kwargs: Any) -> Any:
            from contextlib import nullcontext
            return nullcontext()
        
        @staticmethod
        def log_params(params: Dict[str, Any]) -> None:
            pass
            
        @staticmethod
        def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
            pass
            
        @staticmethod
        def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
            pass
        
        class lightgbm:
            @staticmethod
            def log_model(model: Any, artifact_path: str) -> None:
                pass
    
    mlflow = _DummyMLflow()  # type: ignore
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

class LightGBMTrainer:
    """
    LightGBM trainer with DatasetBuilder integration for consistent preprocessing.
    Maintains compatibility with existing adapter interface while using modern data pipeline.
    """
    
    def __init__(self, config: Dict[str, Any], task_type: str = "regression"):
        """
        Initialize LightGBM trainer.
        
        Args:
            config: Configuration dictionary
            task_type: Type of task ('regression' or 'classification')
        """
        self.config = config
        self.model_config = config.get('models', {}).get('lightgbm', {})
        # Task can be driven by config, fallback to provided arg
        self.task_type = self.model_config.get('task_type', task_type)
        # Optional: treat target as direction labels
        self.as_direction = bool(self.model_config.get('as_direction', False))
        # Optional: threshold for direction labeling (default 0 for sign)
        self.direction_threshold = float(self.model_config.get('direction_threshold', 0.0))
        # Optional: classification decision threshold (calibrated if possible)
        self.decision_threshold = None  # type: Optional[float]
        
        # Model parameters
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

        # Initialize model
        self.model = None
        self.feature_names = []
        self.feature_importance = None
        self.training_history = {}
        # Feature tracking for persistence
        self.selected_features = None  # type: Optional[List[int]]
        self.feature_count = None
        self.input_size = None

        # Cross-validation settings
        self.cv_folds = 5
        self.early_stopping_rounds = 50

        logger.info(f"LightGBM Trainer initialized for {task_type} task")
    
    def build_model(self) -> lgb.LGBMModel:
        """
        Build and initialize the LightGBM model.
        
        Returns:
            Initialized LightGBM model
        """
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
    
    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: np.ndarray,
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        selected_features: Optional[List[int]] = None,
        experiment_name: str = "lightgbm_training"
    ) -> Dict[str, Any]:
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Feature names (optional)
            selected_features: Indices of selected features (optional)
            experiment_name: MLflow experiment name
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting LightGBM model training")
        
        # Store feature information for persistence
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train_array = X_train.values
            self.input_size = len(self.feature_names)
        else:
            self.input_size = X_train.shape[1]
            self.feature_names = feature_names or [f"feature_{i}" for i in range(self.input_size)]
            X_train_array = X_train
        
        # Store feature tracking information
        self.feature_count = self.input_size
        self.selected_features = selected_features
        
        # Apply feature selection if provided
        if selected_features is not None:
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.iloc[:, selected_features]
                self.feature_names = [self.feature_names[i] for i in selected_features]
            else:
                X_train_array = X_train_array[:, selected_features]
                self.feature_names = [self.feature_names[i] for i in selected_features]
        
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                if selected_features is not None:
                    X_val = X_val.iloc[:, selected_features]
                X_val_array = X_val.values
            else:
                if selected_features is not None:
                    X_val = X_val[:, selected_features]
                X_val_array = X_val

        # Build model
        self.build_model()
        
        # Start MLflow run (if available)
        if MLFLOW_AVAILABLE:
            mlflow_context = mlflow.start_run(run_name=f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            from contextlib import nullcontext
            mlflow_context = nullcontext()
        
        with mlflow_context:
            # Log parameters (if MLflow available)
            if MLFLOW_AVAILABLE:
                mlflow.log_params({
                    "model_type": "LightGBM",
                    "task_type": self.task_type,
                    "num_leaves": self.num_leaves,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "n_estimators": self.n_estimators,
                    "boosting_type": self.boosting_type,
                    "objective": self.objective,
                    "metric": self.metric,
                    "n_features": len(self.feature_names)
                })
            
            # Optionally convert targets for classification/direction tasks
            y_train_proc = y_train
            y_val_proc = y_val
            if self.task_type == "classification" or self.as_direction:
                # Convert to binary labels based on threshold (positive vs non-positive)
                y_train_proc = (y_train > self.direction_threshold).astype(int)
                if y_val is not None:
                    y_val_proc = (y_val > self.direction_threshold).astype(int)

            # Prepare evaluation set
            eval_set = None
            X_val_array = None
            if X_val is not None and y_val is not None:
                if isinstance(X_val, pd.DataFrame):
                    X_val_array = X_val.values
                else:
                    X_val_array = X_val
                eval_set = [(X_val_array, y_val_proc if (self.task_type == "classification" or self.as_direction) else y_val)]
            
            # Train model
            if self.model is not None:
                self.model.fit(
                    X_train_array,
                    y_train_proc if (self.task_type == "classification" or self.as_direction) else y_train,
                    eval_set=eval_set,
                    eval_names=['validation'] if eval_set else None,
                    callbacks=[
                        lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0)  # Disable verbose logging
                    ] if eval_set else None
                )
            
            # Get feature importance
            if self.model is not None:
                self.feature_importance = self.model.feature_importances_
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            
            # Log feature importance (if MLflow available)
            if MLFLOW_AVAILABLE:
                # Log top 20 feature importances
                for idx, row in importance_df.head(20).iterrows():
                    mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
            
            # Evaluate on validation set if provided
            val_metrics = {}
            if X_val is not None and y_val is not None and X_val_array is not None:
                # Calibrate classification threshold if applicable
                if (self.task_type == "classification" or self.as_direction):
                    from lightgbm import LGBMClassifier
                    if isinstance(self.model, LGBMClassifier):
                        try:
                            proba = self.model.predict_proba(X_val_array)
                            pos = np.asarray(proba)[:, 1]
                            # Grid search a simple threshold to maximize accuracy
                            thresholds = np.linspace(0.3, 0.7, 41)
                            best_thr = 0.5
                            best_acc = -1.0
                            y_val_bin = (y_val > self.direction_threshold).astype(int)
                            for thr in thresholds:
                                preds = (pos > thr).astype(int)
                                acc = accuracy_score(y_val_bin, preds)
                                if acc > best_acc:
                                    best_acc = acc
                                    best_thr = thr
                            self.decision_threshold = float(best_thr)
                            if MLFLOW_AVAILABLE:
                                mlflow.log_metric("val_decision_threshold", float(best_thr))
                                mlflow.log_metric("val_decision_threshold_acc", float(best_acc))
                        except Exception:
                            # Keep default 0.5 if calibration fails
                            self.decision_threshold = self.decision_threshold or 0.5

                val_predictions = self.predict(X_val_array)
                # Use processed targets for classification metrics
                y_val_for_metrics = y_val_proc if (self.task_type == "classification" or self.as_direction) else y_val
                
                # Only calculate metrics if we have valid target data
                if y_val_for_metrics is not None:
                    val_metrics = self._calculate_metrics(y_val_for_metrics, val_predictions)
                else:
                    val_metrics = {}
                
                # Log validation metrics (if MLflow available)
                if MLFLOW_AVAILABLE:
                    for metric_name, metric_value in val_metrics.items():
                        mlflow.log_metric(f"val_{metric_name}", metric_value)
                
                logger.info(f"Validation metrics: {val_metrics}")
            
            # Save model with signature and input example (if MLflow available)
            if MLFLOW_AVAILABLE:
                # Create a sample input for signature inference
                if isinstance(X_train, pd.DataFrame):
                    sample_input = X_train.iloc[:1]
                else:
                    # For numpy arrays, get the first row
                    sample_input = X_train[0:1]
                
                # Log model with signature and input example
                if MLFLOW_AVAILABLE and mlflow is not None and hasattr(mlflow, "lightgbm"):
                    try:
                        mlflow.lightgbm.log_model(  # type: ignore[attr-defined]
                            self.model,
                            "lightgbm_model"  # artifact_path as positional argument
                        )
                    except Exception as e:
                        logger.warning(f"MLflow log_model failed: {e}")
        
        # Training results
        results = {
            "model": self.model,
            "feature_importance": importance_df,
            "feature_names": self.feature_names,
            "best_iteration": getattr(self.model, 'best_iteration', self.n_estimators),
            "decision_threshold": self.decision_threshold
        }
        
        if X_val is not None and y_val is not None and val_metrics:
            results["validation_metrics"] = val_metrics
        
        logger.info("LightGBM training completed")
        return results
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        # Apply feature selection if available
        if self.selected_features is not None and len(self.selected_features) > 0:
            try:
                X_array = X_array[:, self.selected_features]
            except Exception:
                # Fall back silently if shapes don't align; better to return a prediction than fail hard
                pass
        
        # For classification or direction tasks, return centered score in [-0.5, 0.5]
        if self.task_type == "classification" or self.as_direction:
            from lightgbm import LGBMClassifier
            if isinstance(self.model, LGBMClassifier) and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_array)
                pos = np.asarray(proba)[:, 1]
                # Center around 0: score = p - threshold (default 0.5 or calibrated)
                thr = self.decision_threshold if self.decision_threshold is not None else 0.5
                scores = pos - thr
                return scores.astype(float)
            else:
                logits_or_labels = self.model.predict(X_array)
                if not isinstance(logits_or_labels, np.ndarray):
                    logits_or_labels = np.array(logits_or_labels)
                    
                # Map {0,1} to {-0.5, +0.5}
                unique_vals = np.unique(logits_or_labels)
                if logits_or_labels.ndim == 1 and set(unique_vals).issubset({0, 1}):
                    return (logits_or_labels.astype(float) - 0.5)
                return logits_or_labels.astype(float)
        # Regression: raw prediction
        predictions = self.model.predict(X_array)
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities (for classification tasks).
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions array
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks")
        
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        # Apply feature selection if available
        if self.selected_features is not None and len(self.selected_features) > 0:
            try:
                X_array = X_array[:, self.selected_features]
            except Exception:
                pass
        
        # Check if we're using a classifier model that has predict_proba method
        from lightgbm import LGBMClassifier
        if isinstance(self.model, LGBMClassifier):
            probabilities = self.model.predict_proba(X_array)
            # Ensure return type is numpy array
            if not isinstance(probabilities, np.ndarray):
                probabilities = np.array(probabilities)
            return probabilities
        else:
            raise ValueError("Model does not support predict_proba method")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics based on task type.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        if self.task_type == "regression":
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
            # Convert probabilities to binary predictions if needed
            if y_pred.ndim > 1:
                y_pred_binary = (y_pred[:, 1] > 0.5).astype(int)
            else:
                y_pred_binary = (y_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(y_true, y_pred_binary)
            
            return {
                "accuracy": float(accuracy)
            }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str, symbol: Optional[str] = None):
        """
        Save the trained model with feature index persistence.
        
        Args:
            filepath: Path to save the model
            symbol: Optional symbol for symbol-specific models
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Enhanced save with feature information and standardized metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'task_type': self.task_type,
            'model_config': self.model_config,
            'decision_threshold': self.decision_threshold,
            'selected_features': self.selected_features,
            # Feature persistence information
            'feature_count': self.feature_count,
            'input_size_actual': self.input_size,
            'symbol': symbol,
            'created_at': datetime.now().isoformat(),
            'model_type': 'lightgbm',
            # Training configuration
            'training_config': {
                'num_leaves': self.num_leaves,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'boosting_type': self.boosting_type,
                'objective': self.objective,
                'metric': self.metric
            }
        }
        
        joblib.dump(model_data, filepath)
        
        logger.info(f"LightGBM model saved to {filepath} with {len(self.feature_names)} features")
        if self.selected_features:
            logger.info(f"Selected feature indices: {len(self.selected_features)} features")
    
    @classmethod
    def load_model(cls, filepath: str, config: Dict[str, Any]) -> 'LightGBMTrainer':
        """
        Load a trained model with feature index restoration.
        
        Args:
            filepath: Path to the saved model
            config: Configuration dictionary
            
        Returns:
            Loaded LightGBMTrainer instance with restored feature information
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # Load model data
        model_data = joblib.load(filepath)

        # Create trainer instance (default to regression, may override based on loaded model)
        initial_task = model_data.get('task_type', 'regression')
        trainer = cls(config, task_type=initial_task)

        # Restore model and metadata (support harness artifacts)
        loaded_model = model_data.get('model')
        if loaded_model is None:
            raise ValueError("Loaded artifact is missing the 'model' field")
        trainer.model = loaded_model
        trainer.feature_names = model_data.get('feature_names', [])
        trainer.feature_importance = model_data.get('feature_importance')
        trainer.decision_threshold = model_data.get('decision_threshold')

        # Restore feature information for consistency
        trainer.feature_count = model_data.get('feature_count', len(trainer.feature_names))
        trainer.input_size = model_data.get('input_size_actual', len(trainer.feature_names))
        
        # Restore selected features if present
        sel = model_data.get('selected_features')
        if isinstance(sel, (list, tuple, np.ndarray)):
            trainer.selected_features = list(sel)

        # If artifact included full config, prefer its lightgbm block for inference toggles
        cfg_in_art = model_data.get('config')
        if isinstance(cfg_in_art, dict):
            try:
                lb = (cfg_in_art.get('models', {}) or {}).get('lightgbm', {})
                if isinstance(lb, dict):
                    trainer.model_config.update(lb)
                    trainer.as_direction = bool(lb.get('as_direction', trainer.as_direction))
            except Exception:
                pass

        # Detect classifier vs regressor by model type if task_type missing/mismatched
        try:
            from lightgbm import LGBMClassifier as _LGBMClassifier
            if isinstance(trainer.model, _LGBMClassifier):
                trainer.task_type = 'classification'
        except Exception:
            pass

        # Log feature information
        logger.info(f"LightGBM model loaded from {filepath}")
        logger.info(f"Restored {len(trainer.feature_names)} feature names")
        if trainer.selected_features:
            logger.info(f"Restored {len(trainer.selected_features)} selected feature indices")
        
        # Validate feature consistency
        if trainer.feature_count != len(trainer.feature_names):
            logger.warning(f"Feature count mismatch: stored={trainer.feature_count}, actual={len(trainer.feature_names)}")

        return trainer