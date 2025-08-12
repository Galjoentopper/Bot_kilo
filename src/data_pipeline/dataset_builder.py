"""
Dataset Builder Module
=====================

Centralized dataset assembly for all models to prevent duplication and ensure consistency.
Returns standardized (X, y, timestamps, feature_names, meta) tuples for all model types.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .loader import DataLoader
from .features import FeatureEngine
from .preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class TargetType(Enum):
    """Enumeration of supported target types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification" 
    DIRECTION = "direction"
    VOLATILITY = "volatility"


class ModelType(Enum):
    """Enumeration of supported model types."""
    GRU = "gru"
    LIGHTGBM = "lightgbm"
    PPO = "ppo"


@dataclass
class DatasetMetadata:
    """Metadata container for dataset information."""
    symbol: str
    start_date: str
    end_date: str
    total_samples: int
    feature_count: int
    target_type: str
    model_type: str
    data_quality_score: float
    missing_values_filled: int
    infinite_values_clipped: int
    feature_names: List[str]
    preprocessing_config: Dict[str, Any]
    created_at: str


class DatasetBuilder:
    """
    Centralized dataset builder for all model types.
    
    Ensures deterministic column order, consistent data types, and 
    centralized target variable definitions.
    """
    
    # Central target variable definitions
    TARGET_DEFINITIONS = {
        TargetType.REGRESSION: {
            'description': 'Future price return (pct_change)',
            'horizon': 1,
            'transformation': 'pct_change'
        },
        TargetType.CLASSIFICATION: {
            'description': 'Binary up/down direction',
            'horizon': 1,
            'threshold': 0.001,
            'transformation': 'direction'
        },
        TargetType.DIRECTION: {
            'description': 'Directional movement with threshold',
            'horizon': 1,
            'threshold': 0.0005,
            'transformation': 'direction'
        },
        TargetType.VOLATILITY: {
            'description': 'Future volatility (rolling std)',
            'horizon': 5,
            'transformation': 'volatility'
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DatasetBuilder.
        
        Args:
            config: Configuration dictionary with data and model parameters
        """
        self.config = config
        self.data_config = config.get('data', {})
        self.feature_config = config.get('features', {})
        
        # Initialize components
        self.data_loader = DataLoader(self.data_config.get('data_dir', './data'))
        self.feature_engine = FeatureEngine(self.feature_config)
        
        # Create preprocessors for different model types
        self.preprocessors = {}
        
        logger.info("DatasetBuilder initialized")
    
    def build_dataset(
        self,
        symbol: str,
        model_type: ModelType,
        target_type: TargetType = TargetType.REGRESSION,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        sequence_length: Optional[int] = None
    ) -> Tuple[Dict[str, Any], DatasetMetadata]:
        """
        Build complete dataset for specified model and target type.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCEUR')
            model_type: Type of model (GRU, LightGBM, PPO)
            target_type: Type of target variable
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            train_ratio: Proportion for training set
            validation_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            sequence_length: Sequence length for GRU (optional)
            
        Returns:
            Tuple of (dataset_dict, metadata)
        """
        logger.info(f"Building dataset for {symbol} - Model: {model_type.value}, Target: {target_type.value}")
        
        # Load raw data
        raw_data = self._load_and_validate_data(symbol, start_date, end_date)
        
        # Generate features
        features_data = self._generate_features(raw_data, model_type)
        
        # Prepare target variable
        targets = self._prepare_targets(features_data, target_type)
        
        # Align features and targets
        X, y, timestamps = self._align_data(features_data, targets)
        
        # Get deterministic feature names
        feature_names = self._get_deterministic_feature_names(X, model_type)
        
        # Enforce data consistency
        X, y = self._enforce_data_consistency(X, y, feature_names, model_type)
        
        # Prepare preprocessor
        preprocessor_key = f"{model_type.value}_{target_type.value}"
        if preprocessor_key not in self.preprocessors:
            self.preprocessors[preprocessor_key] = DataPreprocessor()
        
        # Split data chronologically
        train_data, val_data, test_data = self._create_splits(
            X, y, timestamps, train_ratio, validation_ratio, test_ratio
        )
        
        # Preprocess data
        processed_data = self._preprocess_splits(
            train_data, val_data, test_data, 
            self.preprocessors[preprocessor_key], 
            model_type, sequence_length
        )
        
        # Create metadata
        metadata = self._create_metadata(
            symbol, X, y, feature_names, model_type, target_type,
            raw_data.index[0].isoformat(), raw_data.index[-1].isoformat()
        )
        
        logger.info(f"Dataset built successfully - Shape: {X.shape}, Features: {len(feature_names)}")
        
        return processed_data, metadata
    
    def _load_and_validate_data(
        self, 
        symbol: str, 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Load and validate raw market data."""
        try:
            data = self.data_loader.load_symbol_data(symbol, start_date, end_date)
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Validate data integrity
            validation_result = self.data_loader.validate_data_integrity(symbol)
            if not validation_result.get('valid', False):
                raise ValueError(f"Data validation failed for {symbol}: {validation_result.get('error', 'Unknown error')}")
            
            logger.info(f"Loaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            raise
    
    def _generate_features(self, data: pd.DataFrame, model_type: ModelType) -> pd.DataFrame:
        """Generate technical features using the feature engine."""
        try:
            # Generate all features
            features_data = self.feature_engine.generate_all_features(data)
            
            # Model-specific feature adjustments
            if model_type == ModelType.PPO:
                # PPO needs exactly 113 features (exclude ADX for compatibility)
                features_data = self.feature_engine.pad_features_for_model(features_data, 'ppo')
            else:
                # GRU and LightGBM use all 114 features
                features_data = self.feature_engine.pad_features_for_model(features_data, model_type.value)
            
            logger.info(f"Generated features - Shape: {features_data.shape}")
            return features_data
            
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            raise
    
    def _prepare_targets(self, data: pd.DataFrame, target_type: TargetType) -> pd.Series:
        """Prepare target variable based on centralized definitions."""
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column for target preparation")
        
        target_def = self.TARGET_DEFINITIONS[target_type]
        horizon = target_def['horizon']
        
        if target_type == TargetType.REGRESSION:
            # Future return
            targets = data['close'].pct_change(horizon).shift(-horizon)
            
        elif target_type in [TargetType.CLASSIFICATION, TargetType.DIRECTION]:
            # Future direction (1 for up, 0 for down)
            future_return = data['close'].pct_change(horizon).shift(-horizon)
            threshold = target_def['threshold']
            targets = (future_return > threshold).astype(int)
            
        elif target_type == TargetType.VOLATILITY:
            # Future volatility (rolling std of returns)
            returns = data['close'].pct_change()
            targets = returns.rolling(window=horizon).std().shift(-horizon)
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        # Remove NaN values and ensure proper data type
        targets = targets.dropna()
        
        logger.info(f"Prepared {target_type.value} targets - {len(targets)} samples")
        return targets
    
    def _align_data(
        self, 
        features_data: pd.DataFrame, 
        targets: pd.Series
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
        """Align features and targets by removing mismatched indices."""
        # Find common index
        common_index = features_data.index.intersection(targets.index)
        
        if len(common_index) == 0:
            raise ValueError("No common timestamps between features and targets")
        
        # Align data
        X = features_data.loc[common_index].copy()
        y = targets.loc[common_index].values
        timestamps = common_index
        
        logger.info(f"Aligned data - {len(common_index)} samples")
        return X, y, timestamps
    
    def _get_deterministic_feature_names(
        self, 
        X: pd.DataFrame, 
        model_type: ModelType
    ) -> List[str]:
        """Get deterministic feature names with consistent ordering."""
        # Get feature names (excluding original OHLCV columns)
        feature_names = self.feature_engine.get_feature_names(X)
        
        # Sort feature names for deterministic ordering
        feature_names_sorted = sorted(feature_names)
        
        logger.info(f"Deterministic feature order established - {len(feature_names_sorted)} features")
        return feature_names_sorted
    
    def _enforce_data_consistency(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        feature_names: List[str],
        model_type: ModelType
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Enforce consistent data types and handle missing values."""
        # Select features in deterministic order
        X_ordered = X[feature_names].copy()
        
        # Enforce float64 for all features
        X_ordered = X_ordered.astype(np.float64)
        
        # Count and handle NaN/inf values
        nan_count = X_ordered.isna().sum().sum()
        inf_count = np.isinf(X_ordered.values).sum()
        
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values, filling with forward/backward fill then 0")
            X_ordered = X_ordered.ffill().bfill().fillna(0.0)
        
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values, clipping to finite range")
            X_ordered = X_ordered.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Handle target consistency
        y_clean = np.array(y, dtype=np.float64)
        y_clean = np.where(np.isnan(y_clean), 0.0, y_clean)
        y_clean = np.where(np.isinf(y_clean), 0.0, y_clean)
        
        # Final validation
        assert not X_ordered.isna().any().any(), "NaN values remain in features"
        assert not np.isnan(y_clean).any(), "NaN values remain in targets"
        assert not np.isinf(X_ordered.values).any(), "Infinite values remain in features"
        assert not np.isinf(y_clean).any(), "Infinite values remain in targets"
        
        logger.info("Data consistency enforced successfully")
        return X_ordered, y_clean
    
    def _create_splits(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray, 
        timestamps: pd.DatetimeIndex,
        train_ratio: float, 
        validation_ratio: float, 
        test_ratio: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Create chronological train/validation/test splits."""
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        
        train_data = {
            'X': X.iloc[:train_end],
            'y': y[:train_end],
            'timestamps': timestamps[:train_end]
        }
        
        val_data = {
            'X': X.iloc[train_end:val_end],
            'y': y[train_end:val_end],
            'timestamps': timestamps[train_end:val_end]
        }
        
        test_data = {
            'X': X.iloc[val_end:],
            'y': y[val_end:],
            'timestamps': timestamps[val_end:]
        }
        
        logger.info(f"Created splits - Train: {len(train_data['X'])}, Val: {len(val_data['X'])}, Test: {len(test_data['X'])}")
        
        return train_data, val_data, test_data
    
    def _preprocess_splits(
        self,
        train_data: Dict[str, Any],
        val_data: Dict[str, Any],
        test_data: Dict[str, Any],
        preprocessor: DataPreprocessor,
        model_type: ModelType,
        sequence_length: Optional[int]
    ) -> Dict[str, Any]:
        """Preprocess data splits according to model requirements."""
        # Fit preprocessor on training data
        X_train_scaled = preprocessor.fit_transform(train_data['X'])
        X_val_scaled = preprocessor.transform(val_data['X'])
        X_test_scaled = preprocessor.transform(test_data['X'])
        
        # Model-specific preprocessing
        if model_type == ModelType.GRU and sequence_length:
            # Create sequences for GRU
            X_train_seq, y_train_seq = preprocessor.create_sequences(
                X_train_scaled, train_data['y'], sequence_length
            )
            X_val_seq, y_val_seq = preprocessor.create_sequences(
                X_val_scaled, val_data['y'], sequence_length
            )
            X_test_seq, y_test_seq = preprocessor.create_sequences(
                X_test_scaled, test_data['y'], sequence_length
            )
            
            return {
                'train': {'X': X_train_seq, 'y': y_train_seq},
                'validation': {'X': X_val_seq, 'y': y_val_seq},
                'test': {'X': X_test_seq, 'y': y_test_seq},
                'preprocessor': preprocessor,
                'raw_train': train_data,
                'raw_validation': val_data,
                'raw_test': test_data
            }
        
        else:
            # Standard flat arrays for LightGBM and PPO
            return {
                'train': {'X': X_train_scaled, 'y': train_data['y']},
                'validation': {'X': X_val_scaled, 'y': val_data['y']},
                'test': {'X': X_test_scaled, 'y': test_data['y']},
                'preprocessor': preprocessor,
                'raw_train': train_data,
                'raw_validation': val_data,
                'raw_test': test_data
            }
    
    def _create_metadata(
        self,
        symbol: str,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: List[str],
        model_type: ModelType,
        target_type: TargetType,
        start_date: str,
        end_date: str
    ) -> DatasetMetadata:
        """Create comprehensive dataset metadata."""
        return DatasetMetadata(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_samples=len(X),
            feature_count=len(feature_names),
            target_type=target_type.value,
            model_type=model_type.value,
            data_quality_score=1.0,  # Could be calculated based on data validation
            missing_values_filled=0,  # Could be tracked during preprocessing
            infinite_values_clipped=0,  # Could be tracked during preprocessing
            feature_names=feature_names,
            preprocessing_config={
                'scaler_type': 'standard',
                'target_definition': self.TARGET_DEFINITIONS[target_type]
            },
            created_at=datetime.now().isoformat()
        )
    
    def build_regression_dataset(
        self,
        symbol: str,
        model_type: ModelType = ModelType.LIGHTGBM,
        **kwargs
    ) -> Tuple[Dict[str, Any], DatasetMetadata]:
        """Build dataset for regression tasks."""
        return self.build_dataset(
            symbol=symbol,
            model_type=model_type,
            target_type=TargetType.REGRESSION,
            **kwargs
        )
    
    def build_classification_dataset(
        self,
        symbol: str,
        model_type: ModelType = ModelType.LIGHTGBM,
        **kwargs
    ) -> Tuple[Dict[str, Any], DatasetMetadata]:
        """Build dataset for classification tasks."""
        return self.build_dataset(
            symbol=symbol,
            model_type=model_type,
            target_type=TargetType.CLASSIFICATION,
            **kwargs
        )
    
    def build_gru_dataset(
        self,
        symbol: str,
        sequence_length: int = 20,
        **kwargs
    ) -> Tuple[Dict[str, Any], DatasetMetadata]:
        """Build dataset specifically for GRU models."""
        return self.build_dataset(
            symbol=symbol,
            model_type=ModelType.GRU,
            target_type=TargetType.REGRESSION,
            sequence_length=sequence_length,
            **kwargs
        )
    
    def build_ppo_dataset(
        self,
        symbol: str,
        force_feature_consistency: bool = False,
        **kwargs
    ) -> Tuple[Dict[str, Any], DatasetMetadata]:
        """
        Build dataset specifically for PPO models.
        
        Args:
            symbol: Trading symbol
            force_feature_consistency: If True, uses 114 features like other models
            **kwargs: Additional arguments
        """
        # Temporarily override model type for feature consistency if requested
        if force_feature_consistency:
            dataset, metadata = self.build_dataset(
                symbol=symbol,
                model_type=ModelType.LIGHTGBM,  # Use LightGBM feature set (114 features)
                target_type=TargetType.REGRESSION,
                **kwargs
            )
            # Update metadata to reflect PPO usage
            metadata.model_type = "ppo"
            return dataset, metadata
        else:
            return self.build_dataset(
                symbol=symbol,
                model_type=ModelType.PPO,
                target_type=TargetType.REGRESSION,  # PPO uses environment rewards, not supervised targets
                **kwargs
            )
    
    def get_feature_consistency_report(self, datasets: List[Tuple[Dict[str, Any], DatasetMetadata]]) -> Dict[str, Any]:
        """Generate consistency report across multiple datasets."""
        if not datasets:
            return {"error": "No datasets provided"}
        
        feature_counts = []
        feature_name_sets = []
        symbols = []
        
        for dataset, metadata in datasets:
            feature_counts.append(metadata.feature_count)
            feature_name_sets.append(set(metadata.feature_names))
            symbols.append(metadata.symbol)
        
        # Check consistency
        consistent_count = len(set(feature_counts)) == 1
        consistent_names = len(set(tuple(sorted(names)) for names in feature_name_sets)) == 1
        
        return {
            "symbols": symbols,
            "feature_counts": feature_counts,
            "consistent_feature_count": consistent_count,
            "consistent_feature_names": consistent_names,
            "unique_feature_counts": list(set(feature_counts)),
            "total_datasets": len(datasets)
        }