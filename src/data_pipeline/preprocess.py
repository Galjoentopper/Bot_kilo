"""
Data Preprocessing Module
========================

Handles data preprocessing, scaling, and preparation for model training.
Optimized for GPU training with Paperspace Gradient.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import joblib
import os

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class for preparing features for model training.
    """
    
    def __init__(self, scaler_type: str = "standard", imputer_strategy: str = "median"):
        """
        Initialize DataPreprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            imputer_strategy: Strategy for imputing missing values ('mean', 'median', 'most_frequent')
        """
        self.scaler_type = scaler_type
        self.imputer_strategy = imputer_strategy
        
        # Initialize scalers and imputers
        self.scaler = self._get_scaler(scaler_type)
        self.imputer = SimpleImputer(strategy=imputer_strategy)
        
        # Track fitted status
        self.is_fitted = False
        self.feature_names = []
        
        logger.info(f"DataPreprocessor initialized with {scaler_type} scaler and {imputer_strategy} imputer")
    
    def _get_scaler(self, scaler_type: str):
        """Get the appropriate scaler based on type."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if scaler_type not in scalers:
            raise ValueError(f"Unknown scaler type: {scaler_type}. Choose from {list(scalers.keys())}")
        
        return scalers[scaler_type]
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features
            feature_names: Names of features (if X is numpy array)
            
        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X_array)
        
        # Fit scaler
        self.scaler.fit(X_imputed)
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted on {X_array.shape[0]} samples with {X_array.shape[1]} features")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Handle missing values
        X_imputed = self.imputer.transform(X_array)
        
        # Scale features
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            X: Training features
            feature_names: Names of features (if X is numpy array)
            
        Returns:
            Transformed features as numpy array
        """
        return self.fit(X, feature_names).transform(X)
    
    def create_sequences(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        sequence_length: int = 20,
        step_size: int = 1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for time series models (GRU).
        
        Args:
            X: Input features
            y: Target values (optional)
            sequence_length: Length of each sequence
            step_size: Step size between sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        if len(X) < sequence_length:
            raise ValueError(f"Data length ({len(X)}) is less than sequence_length ({sequence_length})")
        
        # Calculate number of sequences
        n_sequences = (len(X) - sequence_length) // step_size + 1
        
        # Create sequences
        X_sequences = np.zeros((n_sequences, sequence_length, X.shape[1]))
        
        for i in range(n_sequences):
            start_idx = i * step_size
            end_idx = start_idx + sequence_length
            X_sequences[i] = X[start_idx:end_idx]
        
        y_sequences = None
        if y is not None:
            y_sequences = np.zeros(n_sequences)
            for i in range(n_sequences):
                target_idx = i * step_size + sequence_length - 1
                if target_idx < len(y):
                    y_sequences[i] = y[target_idx]
        
        logger.info(f"Created {n_sequences} sequences of length {sequence_length}")
        
        return X_sequences, y_sequences
    
    def prepare_target_variable(
        self, 
        df: pd.DataFrame, 
        target_type: str = "return",
        horizon: int = 1,
        threshold: float = 0.001
    ) -> np.ndarray:
        """
        Prepare target variable for prediction.
        
        Args:
            df: DataFrame with price data
            target_type: Type of target ('return', 'price', 'direction', 'volatility')
            horizon: Prediction horizon (number of periods ahead)
            threshold: Threshold for direction classification
            
        Returns:
            Target variable array
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        if target_type == "return":
            # Future return
            target = df['close'].pct_change(horizon).shift(-horizon)
            
        elif target_type == "price":
            # Future price
            target = df['close'].shift(-horizon)
            
        elif target_type == "direction":
            # Future direction (1 for up, 0 for down)
            future_return = df['close'].pct_change(horizon).shift(-horizon)
            target = (future_return > threshold).astype(int)
            
        elif target_type == "volatility":
            # Future volatility (rolling std of returns)
            returns = df['close'].pct_change()
            target = returns.rolling(window=horizon).std().shift(-horizon)
            
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        # Remove NaN values
        target = target.dropna()
        
        logger.info(f"Prepared {target_type} target with {len(target)} samples")
        
        return np.array(target.values)
    
    def save_preprocessor(self, filepath: str):
        """
        Save fitted preprocessor to file.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save preprocessor components
        preprocessor_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'scaler_type': self.scaler_type,
            'imputer_strategy': self.imputer_strategy,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str) -> 'DataPreprocessor':
        """
        Load fitted preprocessor from file.
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded DataPreprocessor instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        preprocessor_data = joblib.load(filepath)
        
        # Create new instance
        preprocessor = cls(
            scaler_type=preprocessor_data['scaler_type'],
            imputer_strategy=preprocessor_data['imputer_strategy']
        )
        
        # Restore fitted components
        preprocessor.scaler = preprocessor_data['scaler']
        preprocessor.imputer = preprocessor_data['imputer']
        preprocessor.feature_names = preprocessor_data['feature_names']
        preprocessor.is_fitted = preprocessor_data['is_fitted']
        
        logger.info(f"Preprocessor loaded from {filepath}")
        
        return preprocessor
