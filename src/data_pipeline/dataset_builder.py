"""
Dataset Builder Module
=====================

Centralized dataset assembly for consistent data preparation across all models.
Ensures deterministic feature engineering and prevents data mismatches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

from .loader import DataLoader
from .features import FeatureEngine
from .preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Centralized dataset builder that ensures consistent data preparation
    for all models with caching support.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        cache_dir: str = "./models/metadata",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize DatasetBuilder.
        
        Args:
            data_dir: Directory containing raw market data
            cache_dir: Directory for cached features
            config: Configuration dictionary
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.config = config or {}

        # Initialize components
        self.data_loader = DataLoader(data_dir)
        self.feature_engine = FeatureEngine(self.config.get('features', {}))

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"DatasetBuilder initialized with cache at {cache_dir}")
    
    def build_dataset(
        self,
        symbol: str,
        interval: str = "15m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        target_type: str = "return",
        target_horizon: int = 1
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex, List[str], Dict[str, Any]]:
        """
        Build dataset for a specific symbol with caching support.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval for data
            start_date: Start date for data (optional)
            end_date: End date for data (optional)
            use_cache: Whether to use cached features if available
            target_type: Type of target variable ('return', 'direction', 'price')
            target_horizon: Prediction horizon in periods
            
        Returns:
            Tuple of (X, y, timestamps, feature_names, metadata)
        """
        logger.info(f"Building dataset for {symbol} ({interval})")
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            symbol, interval, start_date, end_date, target_type, target_horizon
        )
        cache_path = os.path.join(self.cache_dir, f"{symbol}_{interval}_{cache_key}.parquet")
        metadata_path = os.path.join(self.cache_dir, f"{symbol}_{interval}_{cache_key}_metadata.json")
        
        # Try to load from cache
        if use_cache and os.path.exists(cache_path) and os.path.exists(metadata_path):
            logger.info(f"Loading cached features for {symbol}")
            try:
                # Load cached data
                features_df = pd.read_parquet(cache_path)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Extract components
                feature_names = metadata['feature_names']
                X = features_df[feature_names]
                if 'target' in features_df:
                    y = features_df['target'].to_numpy()
                else:
                    logger.warning("Cached features missing 'target' column; using zeros as placeholder target.")
                    y = np.zeros(len(features_df), dtype=float)
                timestamps = pd.to_datetime(features_df.index)

                # Reconstruct runtime raw OHLCV data for PPO and downstream consumers
                try:
                    raw_df = self.data_loader.load_symbol_data(symbol, interval=interval)
                    if raw_df is not None and not raw_df.empty:
                        # Apply date filters to raw data to match cached features
                        if start_date:
                            raw_df = raw_df[raw_df.index >= pd.to_datetime(start_date)]
                        if end_date:
                            raw_df = raw_df[raw_df.index <= pd.to_datetime(end_date)]
                        # Align to feature index
                        raw_df = raw_df.reindex(features_df.index).dropna(how='all')
                        # Attach non-serializable runtime section
                        metadata.setdefault('_runtime', {})['full_data'] = raw_df
                except Exception as e:
                    logger.warning(f"Failed to reconstruct runtime raw data for {symbol}: {e}")
                
                logger.info(f"Loaded {len(X)} samples from cache")
                return X, y, timestamps, feature_names, metadata
                
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding dataset.")
        
        # Load raw data
        logger.info(f"Loading raw data for {symbol}")
        df = self.data_loader.load_symbol_data(symbol, interval=interval)
        
        if df is None or df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Apply date filtering if specified
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        logger.info(f"Loaded {len(df)} raw data points")
        
        # Generate features
        logger.info("Generating features...")
        features_df = self.feature_engine.generate_all_features(df)
        
        # Get feature names (excluding original OHLCV columns)
        feature_names = self.feature_engine.get_feature_names(features_df)
        logger.info(f"Generated {len(feature_names)} features")
        
        # Prepare target variable
        logger.info(f"Preparing {target_type} target with horizon {target_horizon}")
        preprocessor = DataPreprocessor()
        y = preprocessor.prepare_target_variable(
            features_df, 
            target_type=target_type,
            horizon=target_horizon
        )
        
        # Align features and target
        min_len = min(len(features_df), len(y))
        features_df = features_df.iloc[:min_len]
        y = y[:min_len]
        
        # Add target to dataframe for caching
        features_df['target'] = y
        
        # Extract components
        X = features_df[feature_names]
        timestamps = pd.to_datetime(features_df.index)

        # Align raw close prices for downstream cost-aware metrics/thresholding
        try:
            prices_aligned = df['close'].iloc[:min_len].astype(float).values
        except Exception:
            # Fallback: try typical OHLCV naming
            price_col = next((c for c in df.columns if c.lower() in ("close", "Close")), None)
            prices_aligned = df[price_col].iloc[:min_len].astype(float).values if price_col else np.asarray([])
        
        # Create metadata
        metadata = self._create_metadata(
            symbol, interval, feature_names, X, y,
            target_type, target_horizon, cache_key
        )

        # Inject prices (serializable) for later threshold optimization
        prices_aligned = np.asarray(prices_aligned)
        metadata['prices'] = prices_aligned.tolist() if prices_aligned.size > 0 else []

        # Attach non-serializable runtime raw data for PPO
        try:
            runtime_df = df.reindex(features_df.index).iloc[:min_len]
            metadata['_runtime'] = {'full_data': runtime_df}
        except Exception:
            pass
        
        # Save to cache
        if use_cache:
            try:
                logger.info(f"Saving features to cache: {cache_path}")
                features_df.to_parquet(cache_path, compression='snappy')
                # Remove non-serializable runtime section before saving
                metadata_to_save = dict(metadata)
                if '_runtime' in metadata_to_save:
                    metadata_to_save.pop('_runtime', None)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata_to_save, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        return X, y, timestamps, feature_names, metadata
    
    def build_multi_symbol_dataset(
        self,
        symbols: List[str],
        **kwargs
    ) -> Dict[str, Tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex, List[str], Dict[str, Any]]]:
        """
        Build datasets for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            **kwargs: Additional arguments passed to build_dataset
            
        Returns:
            Dictionary mapping symbols to their dataset tuples
        """
        datasets = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Building dataset for {symbol}")
                dataset = self.build_dataset(symbol, **kwargs)
                datasets[symbol] = dataset
            except Exception as e:
                logger.error(f"Failed to build dataset for {symbol}: {e}")
                continue
        
        logger.info(f"Built datasets for {len(datasets)} symbols")
        return datasets
    
    def validate_dataset(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate dataset quality and consistency.
        
        Args:
            X: Feature matrix
            y: Target array
            metadata: Dataset metadata
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for NaN values
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            errors.append(f"Found {nan_count} NaN values in features")
        
        # Check for infinite values
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            errors.append(f"Found {inf_count} infinite values in features")
        
        # Check feature count
        expected_features = len(metadata.get('feature_names', []))
        actual_features = X.shape[1]
        if expected_features != actual_features:
            errors.append(f"Feature count mismatch: expected {expected_features}, got {actual_features}")
        
        # Check target alignment
        if len(X) != len(y):
            errors.append(f"Feature/target length mismatch: X={len(X)}, y={len(y)}")
        
        # Check for sufficient data
        min_samples = 1000  # Minimum samples for reliable training
        if len(X) < min_samples:
            errors.append(f"Insufficient samples: {len(X)} < {min_samples}")
        
        # Check feature variance
        low_variance_features = []
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_std = X[col].std()
            if col_std < 1e-6:
                low_variance_features.append(col)
        
        if low_variance_features:
            errors.append(f"Low variance features: {low_variance_features[:5]}...")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _generate_cache_key(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str],
        end_date: Optional[str],
        target_type: str,
        target_horizon: int
    ) -> str:
        """Generate unique cache key based on parameters and feature configuration."""
        # Create a dictionary of all parameters
        params = {
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date or 'none',
            'end_date': end_date or 'none',
            'target_type': target_type,
            'target_horizon': target_horizon,
            'feature_config': self.feature_engine.config
        }
        
        # Convert to stable string representation
        params_str = json.dumps(params, sort_keys=True)
        
        # Generate hash
        cache_key = hashlib.md5(params_str.encode()).hexdigest()[:12]
        
        return cache_key
    
    def _create_metadata(
        self,
        symbol: str,
        interval: str,
        feature_names: List[str],
        X: pd.DataFrame,
        y: np.ndarray,
        target_type: str,
        target_horizon: int,
        cache_key: str
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for the dataset."""
        # Calculate feature statistics
        feature_stats = {}
        for col in feature_names:
            if col in X.columns:
                feature_stats[col] = {
                    'dtype': str(X[col].dtype),
                    'null_count': int(X[col].isnull().sum()),
                    'min': float(X[col].min()) if pd.api.types.is_numeric_dtype(X[col]) else None,
                    'max': float(X[col].max()) if pd.api.types.is_numeric_dtype(X[col]) else None,
                    'mean': float(X[col].mean()) if pd.api.types.is_numeric_dtype(X[col]) else None,
                    'std': float(X[col].std()) if pd.api.types.is_numeric_dtype(X[col]) else None
                }
        
        metadata = {
            'symbol': symbol,
            'interval': interval,
            'cache_key': cache_key,
            'created_at': datetime.now().isoformat(),
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'sample_count': len(X),
            'target_type': target_type,
            'target_horizon': target_horizon,
            'target_stats': {
                'min': float(np.min(y)),
                'max': float(np.max(y)),
                'mean': float(np.mean(y)),
                'std': float(np.std(y))
            },
            'feature_stats': feature_stats,
            'date_range': {
                'start': str(X.index.min()) if hasattr(X.index, 'min') else None,
                'end': str(X.index.max()) if hasattr(X.index, 'max') else None
            }
        }
        
        return metadata
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached features.
        
        Args:
            symbol: Specific symbol to clear (None clears all)
        """
        if symbol:
            pattern = os.path.join(self.cache_dir, f"{symbol}_*.parquet")
            files = list(Path(self.cache_dir).glob(f"{symbol}_*"))
        else:
            files = list(Path(self.cache_dir).glob("*.parquet")) + \
                   list(Path(self.cache_dir).glob("*.json"))
        
        removed_count = 0
        for file in files:
            try:
                os.remove(file)
                removed_count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {file}: {e}")
        
        logger.info(f"Cleared {removed_count} cached files")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached datasets."""
        cache_files = list(Path(self.cache_dir).glob("*_metadata.json"))
        
        cache_info = {
            'cache_dir': self.cache_dir,
            'total_cached_datasets': len(cache_files),
            'datasets': []
        }
        
        for metadata_file in cache_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Get corresponding parquet file size
                parquet_file = metadata_file.with_suffix('.parquet')
                file_size_mb = parquet_file.stat().st_size / (1024 * 1024) if parquet_file.exists() else 0
                
                cache_info['datasets'].append({
                    'symbol': metadata.get('symbol'),
                    'interval': metadata.get('interval'),
                    'samples': metadata.get('sample_count'),
                    'features': metadata.get('feature_count'),
                    'created_at': metadata.get('created_at'),
                    'file_size_mb': round(file_size_mb, 2)
                })
            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        # Calculate total cache size
        total_size_mb = sum(d['file_size_mb'] for d in cache_info['datasets'])
        cache_info['total_size_mb'] = round(total_size_mb, 2)
        
        return cache_info