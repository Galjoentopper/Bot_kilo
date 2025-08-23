"""
Model Adapter Base Module
========================

Provides a uniform interface for all models to ensure consistency
in training, prediction, and artifact management.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    Provides a uniform interface for training, prediction, and
    artifact management across different model types.
    """
    
    def __init__(self, config: Dict[str, Any], model_type: str):
        """
        Initialize base model adapter.
        
        Args:
            config: Configuration dictionary
            model_type: Type of model (e.g., 'gru', 'lightgbm', 'ppo')
        """
        self.config = config
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.metadata = {}
        
        # Extract model-specific config
        self.model_config = config.get('models', {}).get(model_type, {})
        
        logger.info(f"Initialized {model_type} adapter")
    
    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit the model on training data.
        
        Args:
            X: Feature matrix
            y: Target array
            train_idx: Training indices
            valid_idx: Validation indices
            **kwargs: Additional model-specific arguments
            
        Returns:
            Dictionary of training results
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            **kwargs: Additional model-specific arguments
            
        Returns:
            Predictions array
        """
        pass
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        **kwargs
    ) -> np.ndarray:
        """
        Predict class probabilities (for classification models).
        
        Args:
            X: Feature matrix
            **kwargs: Additional model-specific arguments
            
        Returns:
            Probability predictions array
        """
        # Default implementation for models that don't support probabilities
        raise NotImplementedError(f"{self.model_type} does not support probability predictions")
    
    @abstractmethod
    def get_artifacts(self) -> Dict[str, Any]:
        """
        Get model artifacts for saving.
        
        Returns:
            Dictionary of artifacts to save
        """
        pass
    
    @abstractmethod
    def load_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """
        Load model from artifacts.
        
        Args:
            artifacts: Dictionary of saved artifacts
        """
        pass
    
    def save(self, output_dir: str, run_id: Optional[str] = None, symbol: Optional[str] = None) -> str:
        """
        Save model and artifacts to directory.
        
        Args:
            output_dir: Base output directory
            run_id: Optional run ID (generated if not provided)
            symbol: Optional symbol for symbol-specific models
            
        Returns:
            Path to saved model directory
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Generate run ID if not provided
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create model directory with symbol support
        if symbol:
            model_dir = os.path.join(output_dir, self.model_type, symbol, run_id)
        else:
            model_dir = os.path.join(output_dir, self.model_type, run_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Get artifacts
        artifacts = self.get_artifacts()
        
        # Save each artifact
        for name, artifact in artifacts.items():
            artifact_path = os.path.join(model_dir, name)
            
            if isinstance(artifact, dict):
                # Save as JSON
                with open(artifact_path + '.json', 'w') as f:
                    json.dump(artifact, f, indent=2)
            elif isinstance(artifact, (np.ndarray, pd.DataFrame)):
                # Save as numpy/parquet
                if isinstance(artifact, pd.DataFrame):
                    artifact.to_parquet(artifact_path + '.parquet')
                else:
                    np.save(artifact_path + '.npy', artifact)
            else:
                # Model-specific saving (implemented in subclasses)
                self._save_model_artifact(artifact, artifact_path)
        
        # Save metadata with enhanced information
        metadata = {
            'model_type': self.model_type,
            'run_id': run_id,
            'symbol': symbol,
            'created_at': datetime.now().isoformat(),
            'config': self.model_config,
            'is_fitted': self.is_fitted,
            'feature_names': getattr(self, 'feature_names', []),
            'selected_features': getattr(self, 'selected_features', None),
            'feature_count': getattr(self, 'feature_count', None),
            **self.metadata
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create "latest" symlink for easy access
        if symbol:
            latest_path = os.path.join(output_dir, self.model_type, symbol, 'latest')
            self._create_latest_symlink(model_dir, latest_path)
        
        logger.info(f"Model saved to {model_dir}")
        return model_dir
    
    def _create_latest_symlink(self, target_dir: str, symlink_path: str) -> None:
        """Create or update 'latest' symlink to point to the most recent model."""
        # Remove existing symlink if it exists
        if os.path.exists(symlink_path) or os.path.islink(symlink_path):
            try:
                if os.path.islink(symlink_path):
                    os.unlink(symlink_path)
                elif os.path.isdir(symlink_path):
                    import shutil
                    shutil.rmtree(symlink_path)
                else:
                    os.remove(symlink_path)
            except Exception as e:
                logger.warning(f"Failed to remove existing latest symlink: {e}")
        
        # Create new symlink
        try:
            # Use relative path for the symlink target
            relative_target = os.path.relpath(target_dir, os.path.dirname(symlink_path))
            os.symlink(relative_target, symlink_path)
            logger.debug(f"Created latest symlink: {symlink_path} -> {relative_target}")
        except Exception as e:
            # On Windows or if symlink fails, create a text file with the path
            logger.warning(f"Symlink creation failed, creating pointer file instead: {e}")
            try:
                # Create pointer file in the parent directory with the expected name
                pointer_path = os.path.join(os.path.dirname(symlink_path), 'latest_pointer.txt')
                with open(pointer_path, 'w') as f:
                    f.write(target_dir)
            except Exception as e2:
                logger.warning(f"Failed to create pointer file: {e2}")
    
    def load(self, model_dir: str) -> None:
        """
        Load model from directory.
        
        Args:
            model_dir: Path to model directory
        """
        # Load metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify model type
        if metadata['model_type'] != self.model_type:
            raise ValueError(
                f"Model type mismatch: expected {self.model_type}, "
                f"got {metadata['model_type']}"
            )
        
        # Update metadata
        self.metadata = metadata
        self.model_config = metadata.get('config', {})
        
        # Load artifacts
        artifacts = {}
        for file_path in Path(model_dir).iterdir():
            if file_path.name == 'metadata.json':
                continue
            
            name = file_path.stem
            
            if file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    artifacts[name] = json.load(f)
            elif file_path.suffix == '.parquet':
                artifacts[name] = pd.read_parquet(file_path)
            elif file_path.suffix == '.npy':
                artifacts[name] = np.load(file_path)
            else:
                # Model-specific loading
                artifacts[name] = self._load_model_artifact(file_path)
        
        # Load model from artifacts
        self.load_artifacts(artifacts)
        self.is_fitted = True
        
        logger.info(f"Model loaded from {model_dir}")
    
    @abstractmethod
    def _save_model_artifact(self, artifact: Any, path: str) -> None:
        """
        Save model-specific artifact (implemented in subclasses).
        
        Args:
            artifact: Artifact to save
            path: Path to save to
        """
        pass
    
    @abstractmethod
    def _load_model_artifact(self, path: str) -> Any:
        """
        Load model-specific artifact (implemented in subclasses).
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded artifact
        """
        pass
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.
        
        Returns:
            DataFrame with feature importance or None
        """
        return None
    
    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """
        Get training history if available.
        
        Returns:
            Dictionary of training metrics over time or None
        """
        return None
    
    def validate_input(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        expected_features: Optional[int] = None
    ) -> np.ndarray:
        """
        Validate and prepare input data.
        
        Args:
            X: Input features
            expected_features: Expected number of features
            
        Returns:
            Validated numpy array
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Check shape
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        
        # Check feature count
        if expected_features is not None:
            if X_array.shape[1] != expected_features:
                raise ValueError(
                    f"Feature count mismatch: expected {expected_features}, "
                    f"got {X_array.shape[1]}"
                )
        
        # Check for NaN/inf
        if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
            logger.warning("Input contains NaN or inf values")
            # Replace with zeros (or could raise error)
            X_array = np.nan_to_num(X_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X_array