"""
Model Adapter Implementations
============================

Concrete implementations of the BaseModelAdapter for each model type.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import torch
import joblib
import os

from .base_adapter import BaseModelAdapter
from .gru_trainer import GRUTrainer
from .lgbm_trainer import LightGBMTrainer
from .ppo_trainer import PPOTrainer
from ..data_pipeline.preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class GRUAdapter(BaseModelAdapter):
    """
    Adapter for GRU model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GRU adapter."""
        super().__init__(config, 'gru')
        self.trainer = GRUTrainer(config)
        self.preprocessor = DataPreprocessor()
        self.sequence_length = self.model_config.get('sequence_length', 20)
        self.input_size = None
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
        experiment_name: str = "gru_training",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit GRU model.
        
        Args:
            X: Feature matrix
            y: Target array
            train_idx: Training indices
            valid_idx: Validation indices
            experiment_name: MLflow experiment name
            **kwargs: Additional arguments
            
        Returns:
            Training results
        """
        # Validate input
        X_array = self.validate_input(X)
        self.input_size = X_array.shape[1]
        
        # Split data
        X_train, y_train = X_array[train_idx], y[train_idx]
        X_val, y_val = X_array[valid_idx], y[valid_idx]
        
        # Scale features
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        X_val_scaled = self.preprocessor.transform(X_val)
        
        # Create sequences
        X_train_seq, y_train_seq = self.preprocessor.create_sequences(
            X_train_scaled, y_train, self.sequence_length
        )
        X_val_seq, y_val_seq = self.preprocessor.create_sequences(
            X_val_scaled, y_val, self.sequence_length
        )
        
        # Train model with feature information
        feature_names = getattr(X, 'columns', None)
        if feature_names is not None:
            feature_names = list(feature_names)
        
        results = self.trainer.train(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            experiment_name=experiment_name,
            feature_names=feature_names
        )
        
        self.model = self.trainer.model
        self.is_fitted = True
        
        # Store metadata
        self.metadata.update({
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'train_samples': len(X_train_seq),
            'val_samples': len(X_val_seq)
        })
        
        return results
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> np.ndarray:
        """Make predictions with GRU model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Validate input
        X_array = self.validate_input(X, self.input_size)
        
        # Scale features
        X_scaled = self.preprocessor.transform(X_array)
        
        # Create sequences
        X_seq, _ = self.preprocessor.create_sequences(X_scaled, sequence_length=self.sequence_length)
        
        # Predict
        predictions = self.trainer.predict(X_seq)
        
        return predictions
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get GRU artifacts for saving."""
        return {
            'model': self.trainer,
            'preprocessor': self.preprocessor,
            'config': self.model_config
        }
    
    def load_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Load GRU from artifacts."""
        self.trainer = artifacts['model']
        self.preprocessor = artifacts['preprocessor']
        self.model_config = artifacts.get('config', {})
        self.model = self.trainer.model
        
    def _save_model_artifact(self, artifact: Any, path: str) -> None:
        """Save GRU-specific artifacts."""
        if isinstance(artifact, GRUTrainer):
            # Save PyTorch model with symbol from metadata
            symbol = self.metadata.get('symbol')
            artifact.save_model(path + '.pth', symbol=symbol)
        elif isinstance(artifact, DataPreprocessor):
            # Save preprocessor
            artifact.save_preprocessor(path + '.pkl')
        else:
            # Default to joblib
            joblib.dump(artifact, path + '.pkl')
    
    def _load_model_artifact(self, path: str) -> Any:
        """Load GRU-specific artifacts."""
        if path.suffix == '.pth':
            # Load PyTorch model
            trainer = GRUTrainer(self.config)
            trainer = GRUTrainer.load_model(str(path), self.config)
            return trainer
        elif path.suffix == '.pkl':
            # Load with joblib
            return joblib.load(path)
        else:
            raise ValueError(f"Unknown artifact type: {path}")
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """Get GRU training history."""
        if hasattr(self.trainer, 'train_losses') and hasattr(self.trainer, 'val_losses'):
            return {
                'train_loss': self.trainer.train_losses,
                'val_loss': self.trainer.val_losses
            }
        return None


class LightGBMAdapter(BaseModelAdapter):
    """
    Adapter for LightGBM model.
    """
    
    def __init__(self, config: Dict[str, Any], task_type: str = "regression"):
        """Initialize LightGBM adapter."""
        super().__init__(config, 'lightgbm')
        self.task_type = task_type
        self.trainer = LightGBMTrainer(config, task_type)
        self.feature_names = []
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
        experiment_name: str = "lightgbm_training",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit LightGBM model.
        
        Args:
            X: Feature matrix
            y: Target array
            train_idx: Training indices
            valid_idx: Validation indices
            experiment_name: MLflow experiment name
            **kwargs: Additional arguments
            
        Returns:
            Training results
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_train = X.iloc[train_idx]
            X_val = X.iloc[valid_idx]
        else:
            X_train = X[train_idx]
            X_val = X[valid_idx]
            
        y_train = y[train_idx]
        y_val = y[valid_idx]
        
        # Train model
        results = self.trainer.train(
            X_train, y_train,
            X_val, y_val,
            feature_names=self.feature_names,
            experiment_name=experiment_name
        )
        
        self.model = self.trainer.model
        self.is_fitted = True
        
        # Store metadata
        self.metadata.update({
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'best_iteration': results.get('best_iteration', 0)
        })
        
        return results
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> np.ndarray:
        """Make predictions with LightGBM model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.trainer.predict(X)
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> np.ndarray:
        """Predict probabilities with LightGBM classifier."""
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")
        
        return self.trainer.predict_proba(X)
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get LightGBM artifacts for saving."""
        return {
            'model': self.trainer,
            'config': self.model_config,
            'feature_importance': self.trainer.get_feature_importance() if self.is_fitted else None
        }
    
    def load_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Load LightGBM from artifacts."""
        self.trainer = artifacts['model']
        self.model_config = artifacts.get('config', {})
        self.model = self.trainer.model
        
    def _save_model_artifact(self, artifact: Any, path: str) -> None:
        """Save LightGBM-specific artifacts."""
        if isinstance(artifact, LightGBMTrainer):
            artifact.save_model(path + '.pkl')
        elif isinstance(artifact, pd.DataFrame):
            artifact.to_parquet(path + '.parquet')
        else:
            joblib.dump(artifact, path + '.pkl')
    
    def _load_model_artifact(self, path: str) -> Any:
        """Load LightGBM-specific artifacts."""
        if path.suffix == '.pkl':
            if 'model' in path.stem:
                return LightGBMTrainer.load_model(str(path), self.config)
            else:
                return joblib.load(path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unknown artifact type: {path}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get LightGBM feature importance."""
        if self.is_fitted:
            return self.trainer.get_feature_importance()
        return None


class PPOAdapter(BaseModelAdapter):
    """
    Adapter for PPO model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PPO adapter."""
        super().__init__(config, 'ppo')
        self.trainer = PPOTrainer(config)
        self.env_kwargs = {}
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
        experiment_name: str = "ppo_training",
        total_timesteps: int = 100000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit PPO model.
        
        Note: PPO requires the full DataFrame with OHLCV data, not just features.
        The X parameter should be the full market data DataFrame.
        
        Args:
            X: Full market data DataFrame (with OHLCV columns)
            y: Not used for PPO (included for interface compatibility)
            train_idx: Training indices
            valid_idx: Validation indices
            experiment_name: MLflow experiment name
            total_timesteps: Total training timesteps
            **kwargs: Additional arguments (e.g., env_kwargs)
            
        Returns:
            Training results
        """
        # PPO needs full DataFrame, not just features
        if not isinstance(X, pd.DataFrame):
            raise ValueError("PPO requires DataFrame input with OHLCV data")
        
        # Split data
        train_data = X.iloc[train_idx]
        eval_data = X.iloc[valid_idx] if len(valid_idx) > 0 else None
        
        # Get environment kwargs
        self.env_kwargs = kwargs.get('env_kwargs', {})
        
        # Train model
        results = self.trainer.train(
            train_data,
            eval_data,
            total_timesteps=total_timesteps,
            experiment_name=experiment_name
        )
        
        self.model = self.trainer.model
        self.is_fitted = True
        
        # Store metadata
        self.metadata.update({
            'total_timesteps': total_timesteps,
            'train_samples': len(train_data),
            'eval_samples': len(eval_data) if eval_data is not None else 0,
            'env_kwargs': self.env_kwargs
        })
        
        return results
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        deterministic: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions with PPO model.
        
        Note: For PPO, this returns actions, not price predictions.
        
        Args:
            X: Observation (current state features)
            deterministic: Whether to use deterministic policy
            **kwargs: Additional arguments
            
        Returns:
            Actions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # PPO expects observation, not full DataFrame
        if isinstance(X, pd.DataFrame):
            # Extract observation features (this depends on your environment)
            # For now, we'll assume X is already the observation
            obs = X.values if hasattr(X, 'values') else X
        else:
            obs = X
        
        # Get actions
        actions, _ = self.trainer.predict(obs, deterministic=deterministic)
        
        return actions
    
    def get_artifacts(self) -> Dict[str, Any]:
        """Get PPO artifacts for saving."""
        return {
            'model': self.trainer,
            'config': self.model_config,
            'env_kwargs': self.env_kwargs
        }
    
    def load_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Load PPO from artifacts."""
        self.trainer = artifacts['model']
        self.model_config = artifacts.get('config', {})
        self.env_kwargs = artifacts.get('env_kwargs', {})
        self.model = self.trainer.model
        
    def _save_model_artifact(self, artifact: Any, path: str) -> None:
        """Save PPO-specific artifacts."""
        if isinstance(artifact, PPOTrainer):
            artifact.save_model(path)
        else:
            joblib.dump(artifact, path + '.pkl')
    
    def _load_model_artifact(self, path: str) -> Any:
        """Load PPO-specific artifacts."""
        if 'model' in str(path) and path.suffix == '':
            # PPO model saved without extension
            return PPOTrainer.load_model(str(path), self.config)
        elif path.suffix == '.pkl':
            return joblib.load(path)
        else:
            raise ValueError(f"Unknown artifact type: {path}")


def create_model_adapter(
    model_type: str,
    config: Dict[str, Any],
    task_type: str = "regression"
) -> BaseModelAdapter:
    """
    Factory function to create model adapters.
    
    Args:
        model_type: Type of model ('gru', 'lightgbm', 'ppo')
        config: Configuration dictionary
        task_type: Task type for LightGBM ('regression' or 'classification')
        
    Returns:
        Model adapter instance
    """
    adapters = {
        'gru': lambda: GRUAdapter(config),
        'lightgbm': lambda: LightGBMAdapter(config, task_type),
        'ppo': lambda: PPOAdapter(config)
    }
    
    if model_type not in adapters:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return adapters[model_type]()