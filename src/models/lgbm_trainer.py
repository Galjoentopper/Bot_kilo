"""
LightGBM Trainer Module
======================

LightGBM-based model for refined predictions using engineered features.
Optimized for fast training and feature importance analysis.
"""

import lightgbm as lgb  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
import joblib
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

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
    Trainer class for LightGBM model with feature importance analysis.
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
        self.task_type = task_type
        
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
            experiment_name: MLflow experiment name
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting LightGBM model training")
        
        # Handle feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train_array = X_train.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
            X_train_array = X_train
        
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val_array = X_val.values
            else:
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
            
            # Prepare evaluation set
            eval_set = None
            X_val_array = None
            if X_val is not None and y_val is not None:
                if isinstance(X_val, pd.DataFrame):
                    X_val_array = X_val.values
                else:
                    X_val_array = X_val
                eval_set = [(X_val_array, y_val)]
            
            # Train model
            if self.model is not None:
                self.model.fit(
                    X_train_array,
                    y_train,
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
                val_predictions = self.predict(X_val_array)
                val_metrics = self._calculate_metrics(y_val, val_predictions)
                
                # Log validation metrics (if MLflow available)
                if MLFLOW_AVAILABLE:
                    for metric_name, metric_value in val_metrics.items():
                        mlflow.log_metric(f"val_{metric_name}", metric_value)
                
                logger.info(f"Validation metrics: {val_metrics}")
            
            # Save model (if MLflow available)
            if MLFLOW_AVAILABLE:
                mlflow.lightgbm.log_model(self.model, "model")
        
        # Training results
        results = {
            "model": self.model,
            "feature_importance": importance_df,
            "feature_names": self.feature_names,
            "best_iteration": getattr(self.model, 'best_iteration', self.n_estimators)
        }
        
        if X_val is not None and y_val is not None and val_metrics:
            results["validation_metrics"] = val_metrics
        
        logger.info("LightGBM training completed")
        return results
    
    def train_with_cross_validation(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train model with time series cross-validation.
        
        Args:
            X: Features
            y: Targets
            feature_names: Feature names (optional)
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Starting {cv_folds}-fold cross-validation")
        
        # Handle feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        # Build model
        self.build_model()
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Perform cross-validation
        if self.model is not None:
            if self.task_type == "regression":
                cv_scores = cross_val_score(
                    self.model, X_array, y,
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                cv_scores = -cv_scores  # Convert to positive RMSE
                cv_scores = np.sqrt(cv_scores)  # Convert MSE to RMSE
            else:
                cv_scores = cross_val_score(
                    self.model, X_array, y,
                    cv=tscv,
                    scoring='accuracy',
                    n_jobs=-1
                )
            
            # Train final model on full dataset
            self.model.fit(X_array, y)
            self.feature_importance = self.model.feature_importances_
        else:
            cv_scores = np.array([0.0])
        
        # Results
        results = {
            "cv_scores": cv_scores,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "model": self.model,
            "feature_importance": pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
        }
        
        logger.info(f"Cross-validation completed - Mean score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
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
        
        return self.model.predict(X_array)
    
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
        
        return self.model.predict_proba(X_array)
    
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
    
    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray], y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics dictionary
        """
        predictions = self.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)
        
        logger.info(f"Model evaluation: {metrics}")
        return metrics
    
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
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained to plot feature importance")
        
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'task_type': self.task_type,
            'model_config': self.model_config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, config: Dict[str, Any]) -> 'LightGBMTrainer':
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            config: Configuration dictionary
            
        Returns:
            Loaded LightGBMTrainer instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Create trainer instance
        trainer = cls(config, task_type=model_data['task_type'])
        
        # Restore model and metadata
        trainer.model = model_data['model']
        trainer.feature_names = model_data['feature_names']
        trainer.feature_importance = model_data['feature_importance']
        
        logger.info(f"Model loaded from {filepath}")
        return trainer
    
    def hyperparameter_tuning(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: np.ndarray,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using cross-validation.
        
        Args:
            X: Features
            y: Targets
            param_grid: Parameter grid for tuning
            cv_folds: Number of CV folds
            
        Returns:
            Best parameters and results
        """
        if param_grid is None:
            param_grid = {
                'num_leaves': [31, 50, 100],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 300]
            }
        
        logger.info("Starting hyperparameter tuning")
        
        # Handle feature names
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        best_score = float('-inf') if self.task_type == "classification" else float('inf')
        best_params = {}
        results = []
        
        # Simple grid search (can be replaced with more sophisticated methods)
        from itertools import product
        
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        for params in param_combinations:
            param_dict = dict(zip(param_names, params))
            
            # Update model parameters
            for param, value in param_dict.items():
                setattr(self, param, value)
            
            # Build model with new parameters
            self.build_model()
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            if self.model is not None:
                if self.task_type == "regression":
                    cv_scores = cross_val_score(
                        self.model, X_array, y,
                        cv=tscv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                    mean_score = -cv_scores.mean()  # Convert to positive
                else:
                    cv_scores = cross_val_score(
                        self.model, X_array, y,
                        cv=tscv,
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    mean_score = cv_scores.mean()
            else:
                cv_scores = np.array([0.0])
                mean_score = 0.0
            
            results.append({
                'params': param_dict.copy(),
                'mean_score': mean_score,
                'std_score': cv_scores.std()
            })
            
            # Update best parameters
            if self.task_type == "classification":
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = param_dict.copy()
            else:
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = param_dict.copy()
        
        # Set best parameters
        for param, value in best_params.items():
            setattr(self, param, value)
        
        logger.info(f"Hyperparameter tuning completed - Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }