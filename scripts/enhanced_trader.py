#!/usr/bin/env python3
"""
Enhanced Unified Trading Script with Robust Model Loading

This enhanced version supports:
- Loading models from multiple sources (local, imported, packaged)
- Robust fallback mechanisms for model loading
- Support for transferred models from other machines
- Enhanced error handling and logging
- Model validation and compatibility checking
"""

import os
import sys
import json
import pickle
import asyncio
import time
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Core imports
import pandas as pd
import numpy as np
import ccxt

# Project imports
from src.config.config_loader import ConfigLoader
from src.utils.logger import Logger
from src.data_pipeline.feature_engine import FeatureEngine
from src.data_pipeline.data_preprocessor import DataPreprocessor
from src.models.gru_trainer import GRUTrainer
from src.models.lgbm_trainer import LightGBMTrainer
from src.models.ppo_trainer import PPOTrainer
from src.trading.trading_metrics import TradingMetrics
from src.notifier.telegram_notifier import TelegramNotifier

# Enhanced model loading utilities
from src.utils.model_packaging import ModelPackager
from src.utils.model_transfer import ModelTransferManager

@dataclass
class ModelMetadata:
    """Enhanced model metadata for robust loading."""
    symbol: str
    model_type: str
    version: str
    created_at: str
    python_version: str
    dependencies: Dict[str, str]
    performance_metrics: Dict[str, float]
    file_path: str
    hash_md5: str
    source: str  # 'local', 'imported', 'packaged'
    validated: bool = False

class EnhancedUnifiedPaperTrader:
    """Enhanced paper trader with robust model loading capabilities."""
    
    def __init__(self, config_path: str = 'config.yaml', models_dir: str = 'models'):
        """Initialize the enhanced trader."""
        self.config = ConfigLoader(config_path).config
        self.models_dir = Path(models_dir)
        self.logger = Logger(name='enhanced_trader')
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.trading_metrics = TradingMetrics()
        self.telegram_notifier = TelegramNotifier(self.config)
        
        # Model management
        self.model_packager = ModelPackager()
        self.transfer_manager = ModelTransferManager()
        
        # Trading configuration
        self.symbols = self.config.get('symbols', ['BTCEUR', 'ETHEUR'])
        self.interval = self.config.get('interval', '30m')
        self.initial_balance = float(self.config.get('initial_balance', 10000))
        self.max_position_size = float(self.config.get('max_position_size', 0.1))
        
        # Model storage
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_metadata: Dict[str, Dict[str, ModelMetadata]] = {}
        self.preprocessors: Dict[str, Any] = {}
        self.symbol_feature_metadata: Dict[str, List[str]] = {}
        
        # Trading state
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.balance = self.initial_balance
        self.last_prices = {}
        
        # Performance tracking
        self.performance_history = []
        self.rejected_trades_count = 0
        
        # Data caching
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # seconds
        
        self.logger.logger.info(f"Enhanced trader initialized with ${self.initial_balance:,.2f}")
    
    def load_all_models(self):
        """Load all models with enhanced fallback mechanisms."""
        self.logger.logger.info("Loading models with enhanced fallback mechanisms...")
        
        for symbol in self.symbols:
            self.models[symbol] = {}
            self.model_metadata[symbol] = {}
            
            # Load each model type with multiple fallback sources
            for model_type in ['gru', 'lightgbm', 'ppo']:
                model_info = self._load_model_with_fallbacks(symbol, model_type)
                if model_info:
                    model, metadata = model_info
                    self.models[symbol][model_type] = model
                    self.model_metadata[symbol][model_type] = metadata
                    self.logger.logger.info(
                        f"Loaded {model_type} model for {symbol} from {metadata.source}: {metadata.file_path}"
                    )
                else:
                    self.logger.logger.warning(f"Failed to load {model_type} model for {symbol}")
            
            # Load preprocessor with fallbacks
            preprocessor = self._load_preprocessor_with_fallbacks(symbol)
            self.preprocessors[symbol] = preprocessor
            
            # Load feature metadata
            self._load_feature_metadata(symbol)
        
        total_models = sum(len(models) for models in self.models.values())
        self.logger.logger.info(f"Loaded {total_models} models across {len(self.symbols)} symbols")
    
    def _load_model_with_fallbacks(self, symbol: str, model_type: str) -> Optional[Tuple[Any, ModelMetadata]]:
        """Load model with multiple fallback sources."""
        # Define search strategies in order of preference
        strategies = [
            ('packaged', self._load_from_packaged_models),
            ('imported', self._load_from_imported_models),
            ('best_wf', self._load_from_best_walkforward),
            ('latest', self._load_from_latest_models),
            ('unified', self._load_from_unified_artifacts)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                result = strategy_func(symbol, model_type)
                if result:
                    model, file_path = result
                    # Create metadata
                    metadata = ModelMetadata(
                        symbol=symbol,
                        model_type=model_type,
                        version='unknown',
                        created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                        python_version=sys.version,
                        dependencies={},
                        performance_metrics={},
                        file_path=str(file_path),
                        hash_md5='',
                        source=strategy_name
                    )
                    
                    # Validate model
                    if self._validate_model(model, symbol, model_type):
                        metadata.validated = True
                        return model, metadata
                    else:
                        self.logger.logger.warning(
                            f"Model validation failed for {symbol} {model_type} from {strategy_name}"
                        )
                        
            except Exception as e:
                self.logger.logger.debug(
                    f"Strategy {strategy_name} failed for {symbol} {model_type}: {e}"
                )
                continue
        
        return None
    
    def _load_from_packaged_models(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from packaged models (highest priority)."""
        packages_dir = self.models_dir / 'packages'
        if not packages_dir.exists():
            return None
        
        # Look for packaged models for this symbol and type
        pattern = f"{symbol}_{model_type}_*.zip"
        package_files = list(packages_dir.glob(pattern))
        
        if not package_files:
            return None
        
        # Use the most recent package
        latest_package = max(package_files, key=lambda p: p.stat().st_mtime)
        
        try:
            # Extract and load the packaged model
            extracted_dir = self.model_packager.import_package(str(latest_package))
            
            # Find the model file in the extracted directory
            model_extensions = {
                'gru': ['.pth', '.pt'],
                'lightgbm': ['.pkl'],
                'ppo': ['.zip']
            }
            
            for ext in model_extensions.get(model_type, ['.pkl']):
                model_files = list(Path(extracted_dir).glob(f"*{ext}"))
                if model_files:
                    model_file = model_files[0]
                    model = self._load_model_file(model_file, model_type)
                    if model:
                        return model, model_file
        
        except Exception as e:
            self.logger.logger.debug(f"Failed to load packaged model: {e}")
        
        return None
    
    def _load_from_imported_models(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from imported models directory."""
        imported_dir = self.models_dir / 'imported'
        if not imported_dir.exists():
            return None
        
        # Look for imported models
        model_extensions = {
            'gru': ['.pth', '.pt'],
            'lightgbm': ['.pkl'],
            'ppo': ['.zip']
        }
        
        for ext in model_extensions.get(model_type, ['.pkl']):
            pattern = f"{symbol}*{model_type}*{ext}"
            model_files = list(imported_dir.rglob(pattern))
            
            if model_files:
                # Use the most recent file
                latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
                model = self._load_model_file(latest_file, model_type)
                if model:
                    return model, latest_file
        
        return None
    
    def _load_from_best_walkforward(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from best walk-forward results."""
        metadata_dir = self.models_dir / 'metadata'
        if not metadata_dir.exists():
            return None
        
        # Look for best walk-forward files
        if model_type == 'lightgbm':
            pattern = f"best_wf_lightgbm_{symbol}.pkl"
        elif model_type == 'gru':
            pattern = f"best_wf_gru_{symbol}.pt*"
        else:
            return None
        
        model_files = list(metadata_dir.glob(pattern))
        if model_files:
            latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
            model = self._load_model_file(latest_file, model_type)
            if model:
                return model, latest_file
        
        return None
    
    def _load_from_latest_models(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from latest model files."""
        if model_type == 'gru':
            pattern = f"gru_model_{symbol}_*.pth"
        elif model_type == 'lightgbm':
            pattern = f"lightgbm_model_{symbol}_*.pkl"
        elif model_type == 'ppo':
            pattern = f"ppo_model_{symbol}_*.zip"
        else:
            return None
        
        model_files = list(self.models_dir.glob(pattern))
        if model_files:
            latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
            model = self._load_model_file(latest_file, model_type)
            if model:
                return model, latest_file
        
        return None
    
    def _load_from_unified_artifacts(self, symbol: str, model_type: str) -> Optional[Tuple[Any, Path]]:
        """Load from unified trainer artifacts."""
        search_dir = self.models_dir / model_type / symbol
        if not search_dir.exists():
            return None
        
        if model_type == 'gru':
            filename = 'model.pth'
        elif model_type == 'lightgbm':
            filename = 'model.pkl'
        elif model_type == 'ppo':
            filename = 'model.zip'
        else:
            return None
        
        model_files = list(search_dir.rglob(filename))
        if model_files:
            latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
            model = self._load_model_file(latest_file, model_type)
            if model:
                return model, latest_file
        
        return None
    
    def _load_model_file(self, file_path: Path, model_type: str) -> Optional[Any]:
        """Load a model file based on its type."""
        try:
            if model_type == 'gru':
                return GRUTrainer.load_model(str(file_path), self.config)
            elif model_type == 'lightgbm':
                return LightGBMTrainer.load_model(str(file_path), self.config)
            elif model_type == 'ppo':
                return PPOTrainer.load_model(str(file_path), self.config)
            else:
                self.logger.logger.error(f"Unknown model type: {model_type}")
                return None
        except Exception as e:
            self.logger.logger.debug(f"Failed to load model from {file_path}: {e}")
            return None
    
    def _validate_model(self, model: Any, symbol: str, model_type: str) -> bool:
        """Validate that a loaded model is functional."""
        try:
            # Basic validation - check if model has required methods
            if model_type in ['gru', 'lightgbm']:
                if not hasattr(model, 'predict'):
                    return False
            elif model_type == 'ppo':
                if not hasattr(model, 'predict'):
                    return False
            
            # Additional validation could be added here
            # (e.g., test prediction with dummy data)
            
            return True
        except Exception as e:
            self.logger.logger.debug(f"Model validation failed: {e}")
            return False
    
    def _load_preprocessor_with_fallbacks(self, symbol: str) -> Any:
        """Load preprocessor with fallback mechanisms."""
        # Try multiple sources for preprocessor
        sources = [
            self.models_dir / 'imported' / f"preprocessor_{symbol}.pkl",
            self.models_dir / f"preprocessor_{symbol}_*.pkl",
            self.models_dir / 'gru' / symbol / '*' / 'preprocessor.pkl'
        ]
        
        for source in sources:
            try:
                if '*' in str(source):
                    # Handle glob patterns
                    files = list(Path(str(source).split('*')[0]).parent.glob(Path(str(source)).name))
                    if files:
                        source = max(files, key=lambda p: p.stat().st_mtime)
                    else:
                        continue
                
                if source.exists():
                    # Try pickle first
                    try:
                        with open(source, 'rb') as f:
                            preprocessor = pickle.load(f)
                        self.logger.logger.info(f"Loaded preprocessor for {symbol} from {source}")
                        return preprocessor
                    except Exception:
                        # Try joblib
                        try:
                            import joblib
                            preprocessor = joblib.load(source)
                            self.logger.logger.info(f"Loaded preprocessor (joblib) for {symbol} from {source}")
                            return preprocessor
                        except Exception:
                            continue
            except Exception:
                continue
        
        # Fallback to fresh preprocessor
        self.logger.logger.info(f"Using fresh preprocessor for {symbol}")
        return DataPreprocessor()
    
    def _load_feature_metadata(self, symbol: str):
        """Load feature metadata for the symbol."""
        # Try to load from various sources
        metadata_sources = [
            self.models_dir / 'imported' / f"features_{symbol}.json",
            self.models_dir / 'metadata' / f"features_{symbol}.json",
            self.models_dir / 'gru' / symbol / '*' / 'features.json'
        ]
        
        for source in metadata_sources:
            try:
                if '*' in str(source):
                    files = list(Path(str(source).split('*')[0]).parent.glob('*/features.json'))
                    if files:
                        source = max(files, key=lambda p: p.stat().st_mtime)
                    else:
                        continue
                
                if source.exists():
                    with open(source, 'r') as f:
                        features = json.load(f)
                    self.symbol_feature_metadata[symbol] = features
                    self.logger.logger.info(f"Loaded {len(features)} feature names for {symbol}")
                    return
            except Exception:
                continue
        
        # Fallback to default feature generation
        self.logger.logger.info(f"Using default feature generation for {symbol}")
        self.symbol_feature_metadata[symbol] = []
    
    async def get_market_data(self) -> dict:
        """Get latest market data for all symbols."""
        market_data = {}
        
        # Check cache first
        current_time = time.time()
        if self._has_cached_data(current_time):
            self.logger.logger.debug("Using cached market data")
            return self.data_cache.copy()
        
        # Fetch fresh data
        try:
            for symbol in self.symbols:
                try:
                    self.logger.logger.info(f"Fetching {symbol} data from Binance API...")
                    api_df = await self._fetch_data_from_binance_api(symbol, limit=300)
                    
                    if api_df is None or api_df.empty:
                        self.logger.logger.warning(f"No data fetched from API for {symbol}")
                        continue
                    
                    # Generate features
                    df_with_features = self.feature_engine.generate_all_features(api_df)
                    
                    # Align features with training metadata
                    feature_names = self.symbol_feature_metadata.get(symbol, [])
                    if feature_names:
                        df_aligned = df_with_features.reindex(columns=feature_names, fill_value=0).copy()
                        # Add back OHLCV columns
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in df_with_features.columns:
                                df_aligned[col] = df_with_features[col]
                        df_with_features = df_aligned
                    
                    # Clean features
                    df_with_features = self._clean_features_for_inference(df_with_features, symbol)
                    
                    if self._validate_market_data(df_with_features, symbol):
                        market_data[symbol] = df_with_features
                        self.last_prices[symbol] = df_with_features['close'].iloc[-1]
                        self.logger.logger.info(
                            f"Fetched {len(df_with_features)} records for {symbol} with {len(feature_names)} features"
                        )
                    else:
                        self.logger.logger.warning(f"Invalid market data for {symbol}")
                        
                except Exception as e:
                    self.logger.logger.error(f"Error processing data for {symbol}: {e}")
                    continue
        
        except Exception as e:
            self.logger.logger.error(f"Error in data pipeline: {e}")
            return {}
        
        # Update cache
        self.data_cache = market_data.copy()
        self.cache_expiry = {symbol: current_time + self.cache_duration for symbol in market_data}
        
        return market_data
    
    def _has_cached_data(self, current_time: float) -> bool:
        """Check if we have valid cached data."""
        if not self.data_cache:
            return False
        
        for symbol in self.symbols:
            if symbol not in self.cache_expiry or current_time > self.cache_expiry[symbol]:
                return False
        
        return True
    
    async def _fetch_data_from_binance_api(self, symbol: str, limit: int = 300) -> Optional[pd.DataFrame]:
        """Fetch data directly from Binance API."""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000
            })
            exchange.load_markets()
            
            formatted_symbol = self._convert_symbol_format(symbol)
            self.logger.logger.info(f"Fetching {limit} candles for {symbol} from Binance API")
            
            ohlcv = await self._fetch_with_retry(exchange, formatted_symbol, limit=limit)
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')
                
                # Add missing columns
                df['quote_volume'] = df['volume']
                df['trades'] = 0
                df['taker_buy_base'] = df['volume'] * 0.5
                df['taker_buy_quote'] = df['quote_volume'] * 0.5
                
                # Ensure numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                self.logger.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                return df
            else:
                self.logger.logger.warning(f"No OHLCV data returned for {symbol}")
                return None
                
        except Exception as e:
            self.logger.logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None
    
    async def _fetch_with_retry(self, exchange, symbol: str, max_retries: int = 3, limit: int = 100):
        """Fetch OHLCV data with retry logic."""
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, self.interval, limit=limit)
                return ohlcv
            except ccxt.RateLimitExceeded:
                self.logger.logger.warning(f"Rate limit exceeded for {symbol}, waiting...")
                await asyncio.sleep(10)
            except ccxt.NetworkError as e:
                self.logger.logger.warning(f"Network error for {symbol}: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.logger.error(f"Error fetching {symbol}: {e}")
                break
        
        return None
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format to ccxt standard."""
        if symbol.endswith('EUR'):
            base = symbol[:-3]
            return f"{base}/EUR"
        elif symbol.endswith('USD'):
            base = symbol[:-3]
            return f"{base}/USD"
        elif symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}/USDT"
        return symbol
    
    def _clean_features_for_inference(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean features for inference."""
        try:
            feature_names = self.symbol_feature_metadata.get(symbol, [])
            
            for col in feature_names:
                if col in df.columns:
                    if df[col].isna().any():
                        df[col] = df[col].ffill().bfill()
                        if df[col].isna().any():
                            mean_val = df[col].mean()
                            if pd.isna(mean_val):
                                mean_val = 0.0
                            df[col].fillna(mean_val, inplace=True)
            
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.logger.warning(f"Error cleaning features for {symbol}: {e}")
            return df
    
    def _validate_market_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validate market data."""
        if df.empty:
            return False
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return False
        
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            return False
        
        if len(df) < 50:
            return False
        
        return True
    
    def generate_signals(self, market_data: dict) -> dict:
        """Generate trading signals using loaded models."""
        signals = {}
        self.logger.logger.info(f"Generating signals for {len(market_data)} symbols")
        
        for symbol, df in market_data.items():
            try:
                signal = 0  # Default: hold
                
                if symbol not in self.models or not self.models[symbol]:
                    self.logger.logger.warning(f"No models available for {symbol}")
                    signals[symbol] = signal
                    continue
                
                # Get feature names for this symbol
                feature_names = self.symbol_feature_metadata.get(symbol, self.feature_engine.get_feature_names(df))
                features_for_supervised = df.reindex(columns=feature_names, fill_value=0).copy()
                features_for_supervised = features_for_supervised.ffill().bfill().fillna(0)
                
                if features_for_supervised.empty:
                    self.logger.logger.warning(f"No valid features for {symbol}")
                    signals[symbol] = signal
                    continue
                
                # Generate predictions from available models
                predictions = []
                
                # GRU prediction
                if 'gru' in self.models[symbol]:
                    gru_pred = self._get_gru_prediction(symbol, features_for_supervised)
                    if gru_pred is not None:
                        predictions.append(('gru', gru_pred))
                
                # LightGBM prediction
                if 'lightgbm' in self.models[symbol]:
                    lgbm_pred = self._get_lightgbm_prediction(symbol, features_for_supervised)
                    if lgbm_pred is not None:
                        predictions.append(('lightgbm', lgbm_pred))
                
                # PPO prediction
                if 'ppo' in self.models[symbol]:
                    ppo_pred = self._get_ppo_prediction(symbol, df)
                    if ppo_pred is not None:
                        predictions.append(('ppo', ppo_pred))
                
                # Combine predictions
                if predictions:
                    signal = self._combine_predictions(symbol, predictions, df)
                
                signals[symbol] = signal
                self.logger.logger.debug(f"Generated signal for {symbol}: {signal}")
                
            except Exception as e:
                self.logger.logger.error(f"Signal generation failed for {symbol}: {e}")
                signals[symbol] = 0
        
        return signals
    
    def _get_gru_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[float]:
        """Get GRU model prediction."""
        try:
            model = self.models[symbol]['gru']
            preprocessor = self.preprocessors[symbol]
            
            # Use the last row for prediction
            latest_features = features.iloc[-1:].values
            
            # Preprocess features
            if hasattr(preprocessor, 'transform'):
                latest_features = preprocessor.transform(latest_features)
            
            # Make prediction
            prediction = model.predict(latest_features)
            
            if isinstance(prediction, (list, np.ndarray)):
                prediction = float(prediction[0])
            
            return prediction
            
        except Exception as e:
            self.logger.logger.error(f"GRU prediction failed for {symbol}: {e}")
            return None
    
    def _get_lightgbm_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[float]:
        """Get LightGBM model prediction."""
        try:
            model = self.models[symbol]['lightgbm']
            
            # Use the last row for prediction
            latest_features = features.iloc[-1:]
            
            # Make prediction
            prediction = model.predict(latest_features)
            
            if isinstance(prediction, (list, np.ndarray)):
                prediction = float(prediction[0])
            
            return prediction
            
        except Exception as e:
            self.logger.logger.error(f"LightGBM prediction failed for {symbol}: {e}")
            return None
    
    def _get_ppo_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[float]:
        """Get PPO model prediction."""
        try:
            model = self.models[symbol]['ppo']
            
            # Use raw OHLCV data for PPO (last few rows)
            recent_data = df[['open', 'high', 'low', 'close', 'volume']].iloc[-10:].values
            
            # Make prediction
            action, _ = model.predict(recent_data, deterministic=True)
            
            # Convert action to signal
            if isinstance(action, (list, np.ndarray)):
                action = int(action[0])
            
            # Map PPO action to trading signal
            if action == 0:
                return -1  # Sell
            elif action == 1:
                return 0   # Hold
            elif action == 2:
                return 1   # Buy
            else:
                return 0   # Default hold
            
        except Exception as e:
            self.logger.logger.error(f"PPO prediction failed for {symbol}: {e}")
            return None
    
    def _combine_predictions(self, symbol: str, predictions: List[Tuple[str, float]], df: pd.DataFrame) -> int:
        """Combine predictions from multiple models."""
        try:
            if not predictions:
                return 0
            
            # Simple ensemble: average predictions
            total_pred = sum(pred for _, pred in predictions)
            avg_pred = total_pred / len(predictions)
            
            # Convert to trading signal
            if avg_pred > 0.1:
                return 1  # Buy
            elif avg_pred < -0.1:
                return -1  # Sell
            else:
                return 0  # Hold
            
        except Exception as e:
            self.logger.logger.error(f"Prediction combination failed for {symbol}: {e}")
            return 0
    
    async def execute_trades(self, signals: dict):
        """Execute trades based on signals (paper trading)."""
        for symbol, signal in signals.items():
            try:
                if signal == 0:  # Hold
                    continue
                
                current_price = self.last_prices.get(symbol)
                if not current_price:
                    continue
                
                current_position = self.positions[symbol]
                
                if signal == 1 and current_position <= 0:  # Buy signal
                    # Calculate position size
                    position_value = self.balance * self.max_position_size
                    shares = position_value / current_price
                    
                    # Execute buy
                    self.positions[symbol] += shares
                    self.balance -= position_value
                    
                    self.logger.logger.info(
                        f"BUY {symbol}: {shares:.6f} shares at ${current_price:.2f} (Total: ${position_value:.2f})"
                    )
                    
                elif signal == -1 and current_position > 0:  # Sell signal
                    # Sell all position
                    position_value = current_position * current_price
                    
                    # Execute sell
                    self.balance += position_value
                    self.positions[symbol] = 0
                    
                    self.logger.logger.info(
                        f"SELL {symbol}: {current_position:.6f} shares at ${current_price:.2f} (Total: ${position_value:.2f})"
                    )
                
            except Exception as e:
                self.logger.logger.error(f"Trade execution failed for {symbol}: {e}")
    
    def calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        total_value = self.balance
        
        for symbol, position in self.positions.items():
            if position > 0:
                current_price = self.last_prices.get(symbol, 0)
                total_value += position * current_price
        
        return total_value
    
    def log_performance(self):
        """Log current performance."""
        portfolio_value = self.calculate_portfolio_value()
        pnl = portfolio_value - self.initial_balance
        pnl_pct = (pnl / self.initial_balance) * 100
        
        self.logger.logger.info(
            f"Portfolio Value: ${portfolio_value:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)"
        )
        
        # Store performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'portfolio_value': portfolio_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
    
    async def run_trading_loop(self, iterations: int = None):
        """Run the main trading loop."""
        self.logger.logger.info("Starting enhanced trading loop...")
        
        # Load all models first
        self.load_all_models()
        
        iteration = 0
        while iterations is None or iteration < iterations:
            try:
                self.logger.logger.info(f"Trading iteration {iteration + 1}")
                
                # Get market data
                market_data = await self.get_market_data()
                
                if not market_data:
                    self.logger.logger.warning("No market data available, skipping iteration")
                    await asyncio.sleep(60)
                    continue
                
                # Generate signals
                signals = self.generate_signals(market_data)
                
                # Execute trades
                await self.execute_trades(signals)
                
                # Log performance
                self.log_performance()
                
                # Wait before next iteration
                await asyncio.sleep(300)  # 5 minutes
                iteration += 1
                
            except KeyboardInterrupt:
                self.logger.logger.info("Trading loop interrupted by user")
                break
            except Exception as e:
                self.logger.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)
                iteration += 1
        
        self.logger.logger.info("Trading loop completed")

def main():
    """Main function to run the enhanced trader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Unified Paper Trader')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--models-dir', default='models', help='Models directory path')
    parser.add_argument('--iterations', type=int, help='Number of trading iterations (default: infinite)')
    parser.add_argument('--validate-models', action='store_true', help='Validate all models before trading')
    
    args = parser.parse_args()
    
    # Create trader
    trader = EnhancedUnifiedPaperTrader(
        config_path=args.config,
        models_dir=args.models_dir
    )
    
    # Validate models if requested
    if args.validate_models:
        print("Validating models...")
        trader.load_all_models()
        print("Model validation completed")
        return
    
    # Run trading loop
    try:
        asyncio.run(trader.run_trading_loop(args.iterations))
    except KeyboardInterrupt:
        print("\nTrading stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()