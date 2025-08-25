"""
Unified Trading Script
=====================

Complete redesign combining the best features from trader.py and trader_v2.py.
Addresses all major issues: PPO predictions, feature mismatches, NaN handling, 
per-symbol models, and multi-symbol trading.
"""

import sys
import os
import argparse
import yaml
import asyncio
import time
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import ccxt
import sqlite3
import pickle
from typing import Optional, Dict, Any, Tuple, List
import logging
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from installed package
from src.utils.logger import setup_logging, TradingBotLogger
from src.utils.mlflow_init import initialize_mlflow_from_config
from src.data_pipeline.features import FeatureEngine
from src.data_pipeline.preprocess import DataPreprocessor
from src.models.gru_trainer import GRUTrainer
from src.models.lgbm_trainer import LightGBMTrainer
from src.models.ppo_trainer import PPOTrainer
from src.notifier.telegram import TelegramNotifier
from src.config.config_loader import ConfigLoader


class ModelMetadata:
    """Handles model metadata for feature consistency."""
    
    def __init__(self, metadata_dir: str = "./models/metadata"):
        self.metadata_dir = metadata_dir
        os.makedirs(metadata_dir, exist_ok=True)
    
    def save_metadata(self, model_name: str, symbol: str, metadata: Dict):
        """Save model metadata."""
        filename = f"{model_name}_{symbol}_metadata.json"
        filepath = os.path.join(self.metadata_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_metadata(self, model_name: str, symbol: str) -> Optional[Dict]:
        """Load model metadata."""
        filename = f"{model_name}_{symbol}_metadata.json"
        filepath = os.path.join(self.metadata_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None

    def discover_feature_names(self, symbol: str, desired_interval: Optional[str] = None, logger: Optional[Any] = None) -> Optional[List[str]]:
        """Load feature_names from models/metadata/{SYMBOL}_*_metadata.json.

        If multiple files exist, prefer ones matching desired_interval, else pick the most recent.
        Returns list of feature names or None.
        """
        try:
            import glob
            # Prefer interval-specific files if provided
            candidates: List[str] = []
            if desired_interval:
                pattern_specific = os.path.join(self.metadata_dir, f"{symbol}_{desired_interval}_*_metadata.json")
                candidates = glob.glob(pattern_specific)
            if not candidates:
                pattern_any = os.path.join(self.metadata_dir, f"{symbol}_*_metadata.json")
                candidates = glob.glob(pattern_any)
            if not candidates:
                return None

            # Choose most recent
            chosen = max(candidates, key=os.path.getmtime)
            with open(chosen, 'r') as f:
                data = json.load(f)
            features = data.get('feature_names')
            if isinstance(features, list) and features:
                if logger:
                    logger.info(f"Loaded feature metadata for {symbol} from {os.path.basename(chosen)} ({len(features)} features)")
                return features
            return None
        except Exception as e:
            if logger:
                logger.warning(f"Failed discovering feature metadata for {symbol}: {e}")
            return None


class UnifiedPaperTrader:
    """
    Unified paper trading bot addressing all major issues.
    """
    
    def __init__(self, config: dict, models_dir: str = "./models"):
        """Initialize the unified paper trader."""
        self.config = config
        self.models_dir = models_dir
        
        # Initialize logging
        self.logger = TradingBotLogger()
        self.logger.logger.info("Initializing Unified Paper Trader")
        
        # Initialize components
        self.feature_engine = FeatureEngine(config.get('features', {}))
        self.preprocessor = DataPreprocessor()
        # Trading parameters
        self.symbols = config.get('data', {}).get('symbols', ['BTCEUR'])
        # Resolve interval from config (trainer takes precedence)
        self.interval = (
            config.get('trainer', {}).get('interval')
            or config.get('data', {}).get('interval')
            or '15m'
        )
        # Loop cadence (can differ from data/model timeframe)
        self.loop_interval = (
            config.get('trading', {}).get('loop_interval')
            or self.interval
            or '15m'
        )
        # Load per-symbol feature metadata for robust alignment
        self.metadata_handler = ModelMetadata()
        self.symbol_feature_metadata = {}
        # Resolve interval from config to prefer matching metadata files
        desired_interval = (
            config.get('trainer', {}).get('interval')
            or config.get('data', {}).get('interval')
            or None
        )
        for symbol in self.symbols:
            metadata_path = os.path.join(self.metadata_handler.metadata_dir, f"features_{symbol}.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        meta = json.load(f)
                        self.symbol_feature_metadata[symbol] = meta.get("feature_names", [])
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load feature metadata for {symbol}: {e}")
                    self.symbol_feature_metadata[symbol] = self.feature_engine.get_feature_names(pd.DataFrame())
            else:
                # Try discovering from models/metadata/{SYMBOL}_*_metadata.json
                discovered = self.metadata_handler.discover_feature_names(symbol, desired_interval, self.logger.logger)
                if discovered:
                    self.symbol_feature_metadata[symbol] = discovered
                else:
                    self.logger.logger.warning(f"No feature metadata found for {symbol}, using engine default.")
                    self.symbol_feature_metadata[symbol] = self.feature_engine.get_feature_names(pd.DataFrame())
        
        # Initialize notification system
        try:
            self.notifier = TelegramNotifier.from_config(config)
            self.logger.logger.info("Telegram notifier initialized successfully")
        except Exception as e:
            self.logger.logger.warning(f"Failed to initialize notifications: {e}")
            self.notifier = None
        
        # Trading configuration
        trading_config = config.get('trading', {})
        self.initial_balance = trading_config.get('initial_balance', 10000.0)
        self.transaction_fee = trading_config.get('transaction_fee', 0.001)
        self.slippage = trading_config.get('slippage', 0.0005)
        self.max_position_size = trading_config.get('max_position_size', 0.1)
        # Optimization knobs - FIXED: Rebalanced weights to account for PPO's larger magnitude
        self.model_weights = trading_config.get('model_weights', {'gru': 0.45, 'lightgbm': 0.45, 'ppo': 0.1})
        self.ppo_scale = trading_config.get('ppo_scale', 0.002)
        self.min_trade_value = trading_config.get('min_trade_value', 5.0)
        # Prefer best PPO model (from EvalCallback) when available
        self.prefer_best_ppo = bool(trading_config.get('prefer_best_ppo', True))

        # Portfolio state
        self.balance = self.initial_balance
        self.positions = {}  # symbol -> {'amount': float, 'avg_price': float}
        self.trade_history = []
        self.last_prices = {}
        
        # Models - organized by symbol
        self.models = {}  # symbol -> {'gru': model, 'lightgbm': model, 'ppo': model}
        self.preprocessors = {}  # symbol -> preprocessor
        self.load_all_models()
        
        # Threshold configuration (tunable)
        thresholds_cfg = trading_config.get('thresholds', {})
        self.symbol_thresholds = thresholds_cfg.get('per_symbol', {
            'BTCEUR': 0.00008,
            'ETHEUR': 0.00008,
            'SOLEUR': 0.00012,
            'ADAEUR': 0.00012,
            'XRPEUR': 0.00012,
        })
        self.default_threshold = thresholds_cfg.get('default', 0.00010)
        # Cost floor and volatility scaling knobs
        self.use_cost_floor = thresholds_cfg.get('use_cost_floor', True)
        self.cost_floor_multiplier = float(thresholds_cfg.get('cost_floor_multiplier', 1.2))
        self.vol_reference = float(thresholds_cfg.get('vol_reference', 0.02))  # reference daily vol
        bounds = thresholds_cfg.get('vol_bounds', [0.5, 2.0])
        try:
            self.vol_bounds = (float(bounds[0]), float(bounds[1]))
        except Exception:
            self.vol_bounds = (0.5, 2.0)
        
        # Data caching
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # seconds
        
        # Performance tracking
        self.performance_history = []
        self.rejected_trades_count = 0
        
        self.logger.logger.info(f"Unified trader initialized with ${self.initial_balance:,.2f}")
    
    def load_all_models(self):
        """Load all per-symbol models."""
        self.logger.logger.info("Loading per-symbol models...")
        
        for symbol in self.symbols:
            self.models[symbol] = {}
            
            # Load GRU model
            # Prefer best walk-forward artifact if available
            gru_path = self._find_latest_best_wf('gru', symbol) or \
                       self._find_latest_model(f'gru_model_{symbol}_*.pth') or \
                       self._find_latest_unified_artifact('gru', symbol, 'model.pth')
            if gru_path:
                try:
                    self.models[symbol]['gru'] = GRUTrainer.load_model(gru_path, self.config)
                    self.logger.logger.info(f"Loaded GRU model for {symbol}: {gru_path}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load GRU model for {symbol}: {e}")
            
            # Load LightGBM model
            # Prefer best walk-forward artifact if available
            lgbm_path = self._find_latest_best_wf('lightgbm', symbol) or \
                        self._find_latest_model(f'lightgbm_model_{symbol}_*.pkl') or \
                        self._find_latest_unified_artifact('lightgbm', symbol, 'model.pkl')
            if lgbm_path:
                try:
                    self.models[symbol]['lightgbm'] = LightGBMTrainer.load_model(lgbm_path, self.config)
                    self.logger.logger.info(f"Loaded LightGBM model for {symbol}: {lgbm_path}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load LightGBM model for {symbol}: {e}")
            
            # Load PPO model (prefer best if available)
            ppo_path = None
            if self.prefer_best_ppo:
                ppo_path = self._find_latest_best_ppo(symbol)
                if ppo_path:
                    self.logger.logger.info(f"Using BEST PPO model for {symbol}: {ppo_path}")
            if not ppo_path:
                ppo_path = self._find_latest_model(f'ppo_model_{symbol}_*.zip') or \
                           self._find_latest_unified_artifact('ppo', symbol, 'model.zip')
            if ppo_path:
                try:
                    self.models[symbol]['ppo'] = PPOTrainer.load_model(ppo_path, self.config)
                    self.logger.logger.info(f"Loaded PPO model for {symbol}: {ppo_path}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load PPO model for {symbol}: {e}")
            
            # Load preprocessor for this symbol
            preprocessor_path = self._find_latest_model(f'preprocessor_{symbol}_*.pkl') or \
                                 self._find_latest_unified_artifact('gru', symbol, 'preprocessor.pkl')
            if preprocessor_path:
                try:
                    # First try standard pickle
                    with open(preprocessor_path, 'rb') as f:
                        self.preprocessors[symbol] = pickle.load(f)
                    self.logger.logger.info(f"Loaded preprocessor for {symbol} from {preprocessor_path}")
                except Exception as e_pickle:
                    # If pickle fails, try joblib (common for sklearn pipelines)
                    try:
                        import joblib  # type: ignore
                        self.preprocessors[symbol] = joblib.load(preprocessor_path)
                        self.logger.logger.info(f"Loaded preprocessor (joblib) for {symbol} from {preprocessor_path}")
                    except Exception as e_joblib:
                        # Fall back to fresh preprocessor if artifact is incompatible
                        self.logger.logger.info(
                            f"Preprocessor unavailable for {symbol}, using fresh scaler. (path={preprocessor_path}, pickle_error={e_pickle}, joblib_error={e_joblib})"
                        )
                        self.preprocessors[symbol] = DataPreprocessor()

                # Validate loaded preprocessor: must expose transform and fit/fit_transform
                pre = self.preprocessors.get(symbol)
                if not hasattr(pre, 'transform') or not (hasattr(pre, 'fit') or hasattr(pre, 'fit_transform')):
                    self.logger.logger.info(
                        f"Loaded preprocessor for {symbol} is not a transformer; using fresh scaler. (type={type(pre)})"
                    )
                    self.preprocessors[symbol] = DataPreprocessor()
            else:
                # Create new preprocessor for this symbol
                self.preprocessors[symbol] = DataPreprocessor()
        
        total_models = sum(len(models) for models in self.models.values())
        self.logger.logger.info(f"Loaded {total_models} models across {len(self.symbols)} symbols")
    
    def _find_latest_model(self, pattern: str) -> Optional[str]:
        """Find the latest model file matching pattern."""
        import glob
        model_files = glob.glob(os.path.join(self.models_dir, pattern))
        if model_files:
            return max(model_files, key=os.path.getmtime)
        return None

    def _find_latest_best_wf(self, model_type: str, symbol: str) -> Optional[str]:
        """Find the latest best walk-forward artifact saved by the harness.

        Looks under models/metadata for files like:
          - best_wf_lightgbm_{SYMBOL}.pkl
          - best_wf_gru_{SYMBOL}.pt or .pth
        Returns newest match by mtime or None.
        """
        import glob
        meta_dir = os.path.join(self.models_dir, 'metadata')
        if not os.path.isdir(meta_dir):
            # Also support default relative path if models_dir is not the repo root
            meta_dir = os.path.join('models', 'metadata')
        patterns: List[str] = []
        if model_type == 'lightgbm':
            patterns.append(os.path.join(meta_dir, f"best_wf_lightgbm_{symbol}.pkl"))
        elif model_type == 'gru':
            patterns.append(os.path.join(meta_dir, f"best_wf_gru_{symbol}.pt"))
            patterns.append(os.path.join(meta_dir, f"best_wf_gru_{symbol}.pth"))
        else:
            return None
        candidates: List[str] = []
        for pat in patterns:
            candidates.extend(glob.glob(pat))
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    def _find_latest_unified_artifact(self, model_type: str, symbol: str, filename: str) -> Optional[str]:
        """Find the latest artifact saved by the unified trainer in nested directories.

        Looks under: <models_dir>/<model_type>/<SYMBOL>/**/<filename>
        """
        import glob
        search_root = os.path.join(self.models_dir, model_type, symbol)
        pattern = os.path.join(search_root, '**', filename)
        files = glob.glob(pattern, recursive=True)
        if files:
            return max(files, key=os.path.getmtime)
        return None

    def _find_latest_best_ppo(self, symbol: str) -> Optional[str]:
        """Find the latest 'best' PPO model produced by EvalCallback.

        Search order (most specific first):
        1) <models_dir>/ppo/<SYMBOL>/**/best_model.zip
        2) <models_dir>/ppo_best_*/best_model.zip (global best from recent runs)
        Returns newest match by modification time, or None.
        """
        import glob
        candidates: List[str] = []
        # Symbol-scoped bests
        pattern_symbol = os.path.join(self.models_dir, 'ppo', symbol, '**', 'best_model.zip')
        candidates.extend(glob.glob(pattern_symbol, recursive=True))
        # Global best folders created by EvalCallback
        pattern_global = os.path.join(self.models_dir, 'ppo_best_*', 'best_model.zip')
        candidates.extend(glob.glob(pattern_global))
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)
    
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
    
    async def get_market_data(self) -> dict:
        """Get latest market data for all symbols using the same pipeline as training."""
        market_data = {}
        
        # Check cache first
        current_time = time.time()
        if self._has_cached_data(current_time):
            self.logger.logger.debug("Using cached market data")
            return self.data_cache.copy()
        
        # Initialize components for feature generation only (no database loading)
        try:
            # Get data directly from Binance API for each symbol
            for symbol in self.symbols:
                try:
                    # Fetch 300 30-minute candles directly from Binance API
                    self.logger.logger.info(f"Fetching {symbol} data directly from Binance API...")
                    api_df = await self._fetch_data_from_binance_api(symbol, limit=300)
                    
                    if api_df is None or api_df.empty:
                        self.logger.logger.warning(f"No data fetched from API for {symbol}")
                        continue
                    
                    # Generate features using the same FeatureEngine as training
                    df_with_features = self.feature_engine.generate_all_features(api_df)
                    
                    # Ensure feature alignment with training metadata
                    feature_names = self.symbol_feature_metadata.get(symbol, [])
                    if feature_names:
                        # Align features exactly as in training
                        df_aligned = df_with_features.reindex(columns=feature_names, fill_value=0).copy()
                        # Add back OHLCV columns
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in df_with_features.columns:
                                df_aligned[col] = df_with_features[col]
                        df_with_features = df_aligned
                    
                    # Robust NaN handling (same as training)
                    df_with_features = self._clean_features_for_inference(df_with_features, symbol)
                    
                    if self._validate_market_data(df_with_features, symbol):
                        market_data[symbol] = df_with_features
                        self.last_prices[symbol] = df_with_features['close'].iloc[-1]
                        self.logger.logger.info(f"Fetched {len(df_with_features)} records for {symbol} with {len(feature_names)} features via API")
                    else:
                        self.logger.logger.warning(f"Invalid market data for {symbol}")
                        
                except Exception as e:
                    self.logger.logger.error(f"Error processing data for {symbol}: {e}")
                    # Skip this symbol and continue with others
                    continue
        
        except Exception as e:
            self.logger.logger.error(f"Error in API data pipeline: {e}")
            # Return empty dict if the whole pipeline fails
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
    
    async def _fetch_live_data_supplement(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch recent live data from Binance to supplement historical data."""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000
            })
            exchange.load_markets()
            
            formatted_symbol = self._convert_symbol_format(symbol)
            # Fetch only recent data (last 50 candles) to supplement historical
            ohlcv = await self._fetch_with_retry(exchange, formatted_symbol, limit=50)
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')
                
                # Add missing columns to match training data format
                df['quote_volume'] = df['volume']
                df['trades'] = 0
                df['taker_buy_base'] = 0
                df['taker_buy_quote'] = 0
                
                return df
            
        except Exception as e:
            self.logger.logger.warning(f"Failed to fetch live data for {symbol}: {e}")
        
        return None
    
    def _merge_historical_and_live_data(self, historical_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
        """Merge historical and live data, avoiding duplicates."""
        try:
            # Find the cutoff point - where historical data ends
            if not historical_df.empty and not live_df.empty:
                historical_end = historical_df.index.max()
                
                # Only take live data that's newer than historical data
                new_live_data = live_df[live_df.index > historical_end]
                
                if not new_live_data.empty:
                    # Combine historical + new live data
                    combined_df = pd.concat([historical_df, new_live_data], axis=0)
                    combined_df = combined_df.sort_index()
                    # Remove any potential duplicates
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    return combined_df
            
            return historical_df
            
        except Exception as e:
            self.logger.logger.warning(f"Error merging historical and live data: {e}")
            return historical_df
    
    async def _fetch_data_from_binance_api(self, symbol: str, limit: int = 300) -> Optional[pd.DataFrame]:
        """Fetch data directly from Binance API for the specified number of candles."""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000
            })
            exchange.load_markets()
            
            formatted_symbol = self._convert_symbol_format(symbol)
            self.logger.logger.info(f"Fetching {limit} {self.interval} candles for {symbol} from Binance API")
            
            # Fetch the specified number of candles
            ohlcv = await self._fetch_with_retry(exchange, formatted_symbol, limit=limit)
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')
                
                # Add missing columns to match training data format
                df['quote_volume'] = df['volume']  # Approximate quote volume
                df['trades'] = 0  # Not available from basic OHLCV
                df['taker_buy_base'] = df['volume'] * 0.5  # Rough approximation
                df['taker_buy_quote'] = df['quote_volume'] * 0.5  # Rough approximation
                
                # Ensure numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base', 'taker_buy_quote']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove any NaN rows
                df = df.dropna()
                
                self.logger.logger.info(f"Successfully fetched {len(df)} records for {symbol} from API")
                return df
            else:
                self.logger.logger.warning(f"No OHLCV data returned from API for {symbol}")
                return None
                
        except Exception as e:
            self.logger.logger.error(f"Failed to fetch data from API for {symbol}: {e}")
            return None
    
    def _clean_features_for_inference(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean features for inference using the same method as training."""
        try:
            # Get feature names from metadata
            feature_names = self.symbol_feature_metadata.get(symbol, [])
            
            # Clean feature columns
            for col in feature_names:
                if col in df.columns:
                    # Fill NaN values using multiple strategies (same as training)
                    if df[col].isna().any():
                        # Strategy 1: Forward fill
                        df[col] = df[col].ffill()
                        # Strategy 2: Backward fill
                        df[col] = df[col].bfill()
                        # Strategy 3: Mean fill for any remaining NaN
                        if df[col].isna().any():
                            mean_val = df[col].mean()
                            if pd.isna(mean_val):
                                mean_val = 0.0
                            df[col].fillna(mean_val, inplace=True)
            
            # Replace infinite values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.logger.warning(f"Error cleaning features for {symbol}: {e}")
            return df
    
    async def _get_fallback_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fallback method using direct API fetch (original method)."""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
                'timeout': 30000
            })
            exchange.load_markets()
            
            formatted_symbol = self._convert_symbol_format(symbol)
            ohlcv = await self._fetch_with_retry(exchange, formatted_symbol)
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')
                
                # Add missing columns for compatibility
                df['quote_volume'] = df['volume']
                df['trades'] = 0
                
                # Generate features with robust NaN handling
                df = self._generate_features_with_validation(df, symbol)
                
                if self._validate_market_data(df, symbol):
                    return df
            
        except Exception as e:
            self.logger.logger.error(f"Fallback data fetch failed for {symbol}: {e}")
        
        return None
    
    async def _get_fallback_market_data_all(self) -> dict:
        """Fallback method for all symbols using original pipeline."""
        market_data = {}
        
        for symbol in self.symbols:
            fallback_data = await self._get_fallback_market_data(symbol)
            if fallback_data is not None:
                market_data[symbol] = fallback_data
                self.last_prices[symbol] = fallback_data['close'].iloc[-1]
        
        return market_data
    
    def _generate_features_with_validation(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate features with robust NaN handling."""
        try:
            # Generate all features
            df_with_features = self.feature_engine.generate_all_features(df)
            
            # Get feature names (exclude OHLCV columns)
            feature_names = self.feature_engine.get_feature_names(df_with_features)
            
            # Comprehensive NaN handling
            for col in feature_names:
                if col in df_with_features.columns:
                    # Fill NaN values using multiple strategies
                    if df_with_features[col].isna().any():
                        # Strategy 1: Forward fill
                        df_with_features[col] = df_with_features[col].ffill()
                        
                        # Strategy 2: Backward fill
                        df_with_features[col] = df_with_features[col].bfill()
                        
                        # Strategy 3: Mean fill for any remaining NaN
                        if df_with_features[col].isna().any():
                            mean_val = df_with_features[col].mean()
                            if pd.isna(mean_val):
                                mean_val = 0.0
                            df_with_features[col].fillna(mean_val, inplace=True)
            
            # Replace infinite values
            df_with_features.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_with_features.fillna(0, inplace=True)
            
            # Log feature statistics for debugging
            nan_counts = df_with_features[feature_names].isna().sum()
            if nan_counts.sum() > 0:
                self.logger.logger.warning(f"Remaining NaN values in {symbol}: {nan_counts[nan_counts > 0].to_dict()}")
            
            return df_with_features
            
        except Exception as e:
            self.logger.logger.error(f"Feature generation failed for {symbol}: {e}")
            return df
    
    def _validate_market_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Comprehensive market data validation."""
        if df.empty:
            self.logger.logger.warning(f"Empty DataFrame for {symbol}")
            return False
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.logger.error(f"Missing columns in {symbol}: {missing_cols}")
            return False
        
        # Validate price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                self.logger.logger.error(f"Non-positive prices in {col} for {symbol}")
                return False
        # Validate OHLC relationships
        if (df['high'] < df[['open', 'close']].max(axis=1)).any():
            self.logger.logger.error(f"Invalid high prices for {symbol}")
            return False
        
        if (df['low'] > df[['open', 'close']].min(axis=1)).any():
            self.logger.logger.error(f"Invalid low prices for {symbol}")
            return False
        # Check data freshness (within 2 hours)
        if isinstance(df.index, pd.DatetimeIndex):
            latest_time = df.index.max()
            if latest_time.tz is None:
                latest_time = latest_time.tz_localize('UTC')
            current_time = pd.Timestamp.now(tz='UTC')
            time_diff = current_time - latest_time
            if time_diff > pd.Timedelta(hours=2):
                self.logger.logger.warning(f"Data for {symbol} is stale by {time_diff}")
                return False
        
        # Ensure sufficient data points
        if len(df) < 50:
            self.logger.logger.warning(f"Insufficient data for {symbol}: {len(df)}")
            return False
        
        return True

    def generate_signals(self, market_data: dict) -> dict:
        """Generate trading signals using per-symbol models."""
        signals = {}
        self.logger.logger.info(f"Generating signals for {len(market_data)} symbols")
        for symbol, df in market_data.items():
            try:
                signal = 0  # Default: hold
                if symbol not in self.models or not self.models[symbol]:
                    self.logger.logger.warning(f"No models available for {symbol}")
                    signals[symbol] = signal
                    continue
                # Use feature names from metadata for this symbol (for GRU/LGBM)
                feature_names = self.symbol_feature_metadata.get(symbol, self.feature_engine.get_feature_names(df))
                # Align features for supervised models: add missing columns, order, fill missing with 0
                features_for_supervised = df.reindex(columns=feature_names, fill_value=0).copy()
                # Final NaN check and cleaning
                features_for_supervised = features_for_supervised.ffill().bfill().fillna(0)
                if features_for_supervised.empty:
                    self.logger.logger.warning(f"No valid features for {symbol}")
                    signals[symbol] = signal
                    continue
                # Generate predictions from each available model
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
                    # Use original market df (with raw OHLCV) for PPO to match training env
                    ppo_pred = self._get_ppo_prediction(symbol, df)
                    if ppo_pred is not None:
                        predictions.append(('ppo', ppo_pred))
                # Combine predictions into signal
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
            preprocessor = self.preprocessors.get(symbol, self.preprocessor)
            # Ensure preprocessor is valid transformer
            if not hasattr(preprocessor, 'transform'):
                preprocessor = self.preprocessor
            if not (hasattr(preprocessor, 'fit') or hasattr(preprocessor, 'fit_transform')):
                preprocessor = self.preprocessor
            
            sequence_length = self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
            
            if len(features) < sequence_length:
                self.logger.logger.debug(f"Insufficient data for GRU sequence: {len(features)} < {sequence_length}")
                return None
            
            # Scale features using symbol-specific preprocessor
            try:
                features_scaled = preprocessor.transform(features)
            except Exception:
                # Fallback: fit and transform if transform fails (on a fresh DataPreprocessor)
                if hasattr(preprocessor, 'fit_transform'):
                    features_scaled = preprocessor.fit_transform(features)
                else:
                    fresh = DataPreprocessor()
                    features_scaled = fresh.fit_transform(features)
            
            # Create sequence
            sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
            
            # Validate sequence
            if np.isnan(sequence).any() or np.isinf(sequence).any():
                self.logger.logger.warning(f"Invalid sequence for GRU prediction: {symbol}")
                return None
            
            # Get prediction
            pred = model.predict(sequence)[0]
            
            if np.isnan(pred) or np.isinf(pred):
                self.logger.logger.warning(f"Invalid GRU prediction for {symbol}: {pred}")
                return None
            
            self.logger.logger.debug(f"GRU prediction for {symbol}: {pred}")
            return float(pred)
            
        except Exception as e:
            self.logger.logger.error(f"GRU prediction failed for {symbol}: {e}")
            return None
    
    def _get_lightgbm_prediction(self, symbol: str, features: pd.DataFrame) -> Optional[float]:
        """Get LightGBM model prediction."""
        try:
            model = self.models[symbol]['lightgbm']
            
            # Use latest features
            latest_features = features.iloc[-1:].values
            # If model persisted selected_features indices, subset accordingly
            sel = getattr(model, 'selected_features', None)
            if sel is not None and isinstance(sel, (list, tuple, np.ndarray)) and len(sel) > 0:
                try:
                    latest_features = latest_features[:, list(sel)]
                except Exception:
                    pass
            
            # Validate features
            if np.isnan(latest_features).any() or np.isinf(latest_features).any():
                self.logger.logger.warning(f"Invalid features for LightGBM prediction: {symbol}")
                return None
            
            # Get prediction
            pred = model.predict(latest_features)[0]
            
            if np.isnan(pred) or np.isinf(pred):
                self.logger.logger.warning(f"Invalid LightGBM prediction for {symbol}: {pred}")
                return None
            
            self.logger.logger.debug(f"LightGBM prediction for {symbol}: {pred}")
            return float(pred)
            
        except Exception as e:
            self.logger.logger.error(f"LightGBM prediction failed for {symbol}: {e}")
            return None
    
    def _get_ppo_prediction(self, symbol: str, market_df: pd.DataFrame) -> Optional[float]:
        """Get PPO model prediction with observation shape aligned to training env (lookback, 10)."""
        try:
            model = self.models[symbol]['ppo']
            # Prefer PPO's own sequence length if provided; fallback to GRU's, then 20
            sequence_length = (
                self.config.get('models', {}).get('ppo', {}).get('sequence_length')
                or self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
            )
            
            if len(market_df) < sequence_length:
                self.logger.logger.debug(f"Insufficient data for PPO sequence: {len(market_df)} < {sequence_length}")
                return None
            # Ensure required raw columns exist
            base_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
            df = market_df.copy()
            for col in base_cols:
                if col not in df.columns:
                    # Fill missing optional columns with zeros
                    df[col] = 0.0
            df = df[base_cols]
            # Clean and clip similar to env preprocessing
            arr_np: np.ndarray = df.to_numpy(dtype=np.float32)
            arr_np = np.nan_to_num(arr_np, nan=0.0, posinf=1.0, neginf=-1.0)
            arr_np = np.clip(arr_np, -10.0, 10.0)
            market_sequence = arr_np[-sequence_length:, :]
            
            # Create portfolio state (normalized)
            current_balance = self.balance
            position_value = 0.0
            if symbol in self.positions:
                position_value = self.positions[symbol]['amount'] * self.last_prices.get(symbol, 0)
            
            total_value = current_balance + position_value
            balance_ratio = current_balance / max(total_value, 1.0)
            position_ratio = position_value / max(total_value, 1.0)
            # Unrealized PnL ratio approximation
            unrealized_pnl = 0.0
            if symbol in self.positions:
                pos = self.positions[symbol]
                unrealized_pnl = (self.last_prices.get(symbol, 0) - pos['avg_price']) * pos['amount']
            unrealized_pnl_ratio = unrealized_pnl / max(total_value, 1.0)
            
            # Create portfolio features (3 features to match env)
            portfolio_features = np.array([
                balance_ratio,
                position_ratio,
                unrealized_pnl_ratio
            ], dtype=np.float32)
            
            # Tile portfolio features to match sequence length
            portfolio_matrix = np.tile(portfolio_features, (sequence_length, 1))
            
            # Combine market and portfolio features
            observation = np.concatenate([market_sequence, portfolio_matrix], axis=1)
            
            # Validate observation shape - should match training
            expected_shape = (sequence_length, market_sequence.shape[1] + 3)
            if observation.shape != expected_shape:
                self.logger.logger.warning(f"PPO observation shape mismatch for {symbol}: {observation.shape} != {expected_shape}")
                return None
            
            # Validate observation values
            if np.isnan(observation).any() or np.isinf(observation).any():
                self.logger.logger.warning(f"Invalid observation for PPO prediction: {symbol}")
                return None
            
            # Get action from PPO model
            action, _ = model.predict(observation, deterministic=True)
            
            # Convert action to prediction value
            # For continuous action space in env (Box(1,)), interpret action as target position change
            # Convert to a directional score scaled by configurable ppo_scale
            action_value = float(action[0]) if hasattr(action, '__len__') else float(action)
            action_value = np.clip(action_value, -1.0, 1.0)
            pred = float(self.ppo_scale) * action_value
            
            self.logger.logger.debug(f"PPO prediction for {symbol}: action={action_value}, pred={pred}")
            return float(pred)
            
        except Exception as e:
            self.logger.logger.error(f"PPO prediction failed for {symbol}: {e}")
            return None
    
    def _combine_predictions(self, symbol: str, predictions: List[Tuple[str, float]], df: pd.DataFrame) -> int:
        """Combine model predictions into trading signal."""
        try:
            # Extract prediction values
            pred_values = [pred for _, pred in predictions]
            
            # Calculate weighted average (configurable)
            weights = self.model_weights
            weighted_sum = sum(weights.get(model, 1.0) * pred for model, pred in predictions)
            total_weight = sum(weights.get(model, 1.0) for model, _ in predictions)
            
            avg_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # Enhanced debug logging for model contributions
            if self.logger.logger.isEnabledFor(logging.DEBUG):
                contribution_details = []
                for model, pred in predictions:
                    weight = weights.get(model, 1.0)
                    contribution = weight * pred
                    contribution_pct = (contribution / avg_prediction * 100) if avg_prediction != 0 else 0
                    contribution_details.append(f"{model}={pred:.6f}(w:{weight:.2f},c:{contribution:.6f},{contribution_pct:.1f}%)")
                self.logger.logger.debug(f"Model contributions for {symbol}: {', '.join(contribution_details)}")
            
            # Calculate dynamic threshold based on recent volatility
            threshold = self._get_dynamic_threshold(symbol, df)
            
            # Convert to signal
            if avg_prediction > threshold:
                signal = 1  # Buy
            elif avg_prediction < -threshold:
                signal = -1  # Sell
            else:
                signal = 0  # Hold
            
            self.logger.logger.debug(f"Combined signal for {symbol}: pred={avg_prediction:.6f}, threshold={threshold:.6f}, signal={signal}")
            return signal
            
        except Exception as e:
            self.logger.logger.error(f"Signal combination failed for {symbol}: {e}")
            return 0
    
    def _get_dynamic_threshold(self, symbol: str, df: pd.DataFrame) -> float:
        """Calculate dynamic threshold based on recent volatility."""
        base_threshold = self.symbol_thresholds.get(symbol, self.default_threshold)

        try:
            if 'close' in df.columns and len(df) > 20:
                recent_returns = df['close'].pct_change().dropna().tail(20)
                volatility = recent_returns.std()
                # Optional cost floor to avoid trading under fees+slippage
                cost_floor = 0.0
                if self.use_cost_floor:
                    cost_floor = (self.transaction_fee + self.slippage) * self.cost_floor_multiplier
                # Volatility multiplier with configurable reference and bounds
                if volatility and volatility > 0:
                    lower, upper = self.vol_bounds
                    vol_mult = max(lower, min(upper, volatility / self.vol_reference))
                else:
                    vol_mult = 1.0
                dynamic_threshold = max(base_threshold, cost_floor) * vol_mult
                self.logger.logger.debug(
                    f"Dynamic threshold for {symbol}: {dynamic_threshold:.6f} (vol: {volatility:.6f}, cost:{cost_floor:.6f}, base:{base_threshold:.6f})"
                )
                return dynamic_threshold
        except Exception as e:
            self.logger.logger.warning(f"Failed to calculate dynamic threshold for {symbol}: {e}")

        return base_threshold
    
    async def execute_trades(self, signals: dict, market_data: dict):
        """Execute trades based on signals."""
        trades_executed = 0
        
        for symbol, signal in signals.items():
            try:
                if signal == 0:  # Hold
                    continue
                
                current_price = self.last_prices.get(symbol)
                if not current_price:
                    self.logger.logger.warning(f"No current price for {symbol}")
                    continue
                
                # Calculate position size
                position_size = self._calculate_position_size(symbol, current_price, signal)
                
                if abs(position_size) < 0.0001:  # Minimum position size
                    if signal != 0:  # Only log if there was actually a signal
                        current_pos = self.positions.get(symbol, {}).get('amount', 0.0)
                        self.logger.logger.info(f"Position size too small for {symbol}: {position_size:.6f} "
                                              f"(signal: {signal}, current_pos: {current_pos:.6f}, price: {current_price:.4f})")
                    continue
                
                # Execute trade
                if signal > 0:  # Buy
                    success = self._execute_buy(symbol, position_size, current_price)
                else:  # Sell
                    success = self._execute_sell(symbol, abs(position_size), current_price)
                
                if success:
                    trades_executed += 1
                
            except Exception as e:
                self.logger.logger.error(f"Trade execution failed for {symbol}: {e}")
        
        if trades_executed > 0:
            self._log_portfolio_status()
            
            # Send notification
            if self.notifier and self.notifier.enabled:
                await self._send_trading_notification(trades_executed)
    
    def _calculate_position_size(self, symbol: str, price: float, signal: int) -> float:
        """Calculate position size based on available balance and risk management."""
        try:
            # Maximum position value based on balance and max position size
            max_position_value = self.balance * self.max_position_size
            
            # Calculate position size in base currency units
            position_size = max_position_value / price
            
            # Apply signal direction
            if signal < 0:  # Sell signal
                # For sells, we can only sell what we own
                current_position = self.positions.get(symbol, {}).get('amount', 0.0)
                if current_position > 0:
                    position_size = min(position_size, current_position)
                else:
                    # No position to sell, return 0
                    position_size = 0.0
            
            # Ensure minimum viable trade size (configurable)
            min_position_size = float(self.min_trade_value) / price
            
            if position_size < min_position_size:
                return 0.0
            
            return position_size
            
        except Exception as e:
            self.logger.logger.error(f"Position size calculation failed for {symbol}: {e}")
            return 0.0
    
    def _execute_buy(self, symbol: str, amount: float, price: float) -> bool:
        """Execute a buy order."""
        try:
            # Calculate total cost including fees
            gross_cost = amount * price
            fee = gross_cost * self.transaction_fee
            slippage_cost = gross_cost * self.slippage
            total_cost = gross_cost + fee + slippage_cost
            
            # Check if we have enough balance
            if total_cost > self.balance:
                self.logger.logger.warning(f"Insufficient balance for {symbol}: need {total_cost:.2f}, have {self.balance:.2f}")
                self.rejected_trades_count += 1
                return False
            
            # Execute the buy
            self.balance -= total_cost
            
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {'amount': 0.0, 'avg_price': 0.0}
            
            current_amount = self.positions[symbol]['amount']
            current_avg_price = self.positions[symbol]['avg_price']
            
            # Calculate new average price
            total_amount = current_amount + amount
            new_avg_price = ((current_amount * current_avg_price) + (amount * price)) / total_amount
            
            self.positions[symbol] = {
                'amount': total_amount,
                'avg_price': new_avg_price
            }
            
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'buy',
                'amount': amount,
                'price': price,
                'fee': fee,
                'slippage': slippage_cost,
                'total_cost': total_cost,
                'balance_after': self.balance
            }
            self.trade_history.append(trade)
            
            self.logger.logger.info(f"BUY {symbol}: {amount:.6f} @ {price:.4f} (cost: {total_cost:.2f})")
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Buy execution failed for {symbol}: {e}")
            return False
    
    def _execute_sell(self, symbol: str, amount: float, price: float) -> bool:
        """Execute a sell order."""
        try:
            # Check if we have enough to sell
            current_position = self.positions.get(symbol, {}).get('amount', 0.0)
            if amount > current_position:
                self.logger.logger.warning(f"Insufficient position for {symbol}: trying to sell {amount:.6f}, have {current_position:.6f}")
                self.rejected_trades_count += 1
                return False
            
            # Calculate proceeds after fees and slippage
            gross_proceeds = amount * price
            fee = gross_proceeds * self.transaction_fee
            slippage_cost = gross_proceeds * self.slippage
            net_proceeds = gross_proceeds - fee - slippage_cost
            
            # Execute the sell
            self.balance += net_proceeds
            
            # Update position
            remaining_amount = current_position - amount
            if remaining_amount < 0.0001:  # Close position if very small remainder
                del self.positions[symbol]
            else:
                self.positions[symbol]['amount'] = remaining_amount
            
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': 'sell',
                'amount': amount,
                'price': price,
                'fee': fee,
                'slippage': slippage_cost,
                'net_proceeds': net_proceeds,
                'balance_after': self.balance
            }
            self.trade_history.append(trade)
            
            self.logger.logger.info(f"SELL {symbol}: {amount:.6f} @ {price:.4f} (proceeds: {net_proceeds:.2f})")
            return True
            
        except Exception as e:
            self.logger.logger.error(f"Sell execution failed for {symbol}: {e}")
            return False
    
    def _log_portfolio_status(self):
        """Log current portfolio status."""
        total_position_value = 0.0
        
        for symbol, position in self.positions.items():
            current_price = self.last_prices.get(symbol, position['avg_price'])
            position_value = position['amount'] * current_price
            total_position_value += position_value
            
            pnl = position_value - (position['amount'] * position['avg_price'])
            pnl_pct = (pnl / (position['amount'] * position['avg_price'])) * 100 if position['avg_price'] > 0 else 0
            
            self.logger.logger.info(f"Position {symbol}: {position['amount']:.6f} @ {position['avg_price']:.4f} "
                                  f"(current: {current_price:.4f}, PnL: {pnl:+.2f} ({pnl_pct:+.1f}%))")
        
        total_portfolio_value = self.balance + total_position_value
        total_pnl = total_portfolio_value - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100
        
        self.logger.logger.info(f"Portfolio: Balance={self.balance:.2f}, Positions={total_position_value:.2f}, "
                              f"Total={total_portfolio_value:.2f}, PnL={total_pnl:+.2f} ({total_pnl_pct:+.1f}%)")
    
    async def _send_trading_notification(self, trades_count: int):
        """Send trading notification via Telegram."""
        try:
            if not self.notifier:
                return
            
            total_value = self.balance + sum(
                pos['amount'] * self.last_prices.get(symbol, pos['avg_price'])
                for symbol, pos in self.positions.items()
            )
            
            pnl = total_value - self.initial_balance
            pnl_pct = (pnl / self.initial_balance) * 100
            
            message = (
                f"📊 Trading Update\n"
                f"Executed {trades_count} trades\n"
                f"Portfolio Value: €{total_value:,.2f}\n"
                f"P&L: €{pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
                f"Cash: €{self.balance:,.2f}"
            )
            
            if self.positions:
                message += "\n\nPositions:"
                for symbol, pos in self.positions.items():
                    current_price = self.last_prices.get(symbol, pos['avg_price'])
                    pos_value = pos['amount'] * current_price
                    message += f"\n{symbol}: €{pos_value:,.0f}"
            
            await self.notifier.send_message(message)
            
        except Exception as e:
            self.logger.logger.error(f"Failed to send notification: {e}")
    
    async def run_trading_loop(self, iterations: Optional[int] = None):
        """Main trading loop."""
        self.logger.logger.info("Starting unified trading loop...")
        
        if self.notifier and self.notifier.enabled:
            await self.notifier.send_message("🚀 Unified Paper Trader started!")
        
        iteration = 0
        
        try:
            while iterations is None or iteration < iterations:
                iteration += 1
                
                self.logger.logger.info(f"=== Trading Iteration {iteration} ===")
                
                # Get market data
                market_data = await self.get_market_data()
                
                if not market_data:
                    self.logger.logger.warning("No market data available, skipping iteration")
                    await asyncio.sleep(self._time_to_next_tick())
                    continue
                
                self.logger.logger.info(f"Retrieved data for {len(market_data)} symbols")
                
                # Generate signals
                signals = self.generate_signals(market_data)
                
                # Log signals
                active_signals = {k: v for k, v in signals.items() if v != 0}
                if active_signals:
                    self.logger.logger.info(f"Active signals: {active_signals}")
                else:
                    self.logger.logger.info("No active signals")
                
                # Execute trades
                await self.execute_trades(signals, market_data)
                
                # Log performance
                if iteration % 10 == 0:  # Every 10 iterations
                    self._log_portfolio_status()
                
                # Wait before next iteration (skip waiting if this was the final requested iteration)
                if iterations is not None and iteration >= iterations:
                    self.logger.logger.info("Completed requested iterations; exiting loop without additional wait.")
                    break
                sleep_sec = self._time_to_next_tick()
                next_run_ts = datetime.utcnow() + timedelta(seconds=sleep_sec)
                self.logger.logger.info(f"Waiting {int(sleep_sec)}s until next loop tick ({self.loop_interval}) (~{next_run_ts:%Y-%m-%d %H:%M:%S} UTC)")
                await asyncio.sleep(sleep_sec)
                
        except KeyboardInterrupt:
            self.logger.logger.info("Trading loop interrupted by user")
        except Exception as e:
            self.logger.logger.error(f"Trading loop error: {e}")
            if self.notifier and self.notifier.enabled:
                await self.notifier.send_message(f"❌ Trading error: {str(e)}")
            raise
        finally:
            self.logger.logger.info("Trading loop ended")
            self._log_portfolio_status()

    def _interval_to_seconds(self, interval: Optional[str] = None) -> int:
        """Convert timeframe string to seconds (e.g., '15m' -> 900). Defaults to self.interval."""
        tf = (interval or self.interval or '15m').strip().lower()
        try:
            unit = tf[-1]
            value = int(tf[:-1])
            if unit == 'm':
                return value * 60
            if unit == 'h':
                return value * 3600
            if unit == 'd':
                return value * 86400
            # Fallback assume minutes if unit missing
            return int(tf) * 60
        except Exception:
            return 900  # default 15 minutes

    def _time_to_next_candle(self) -> int:
        """Seconds until the next interval boundary for the configured timeframe."""
        interval_sec = max(1, self._interval_to_seconds())
        now = int(time.time())
        # Next boundary is the next multiple of interval_sec
        remainder = now % interval_sec
        wait = (interval_sec - remainder) if remainder != 0 else interval_sec
        # Add a tiny buffer (2s) to reduce race with exchange candle close
        wait = max(1, min(wait + 2, interval_sec))
        return int(wait)

    def _time_to_next_tick(self) -> int:
        """Seconds until the next loop tick boundary (trading.loop_interval)."""
        tick_sec = max(1, self._interval_to_seconds(self.loop_interval))
        now = int(time.time())
        remainder = now % tick_sec
        wait = (tick_sec - remainder) if remainder != 0 else tick_sec
        # A small buffer to avoid racing with exchange/clock drift
        wait = max(1, min(wait + 2, tick_sec))
        return int(wait)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration using ConfigLoader with auto-detection."""
    try:
        config_loader = ConfigLoader(config_path)
        return config_loader.config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Unified Crypto Trading Bot')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory containing trained models')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of trading iterations (default: infinite)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration. Exiting.")
        return

    # Initialize MLflow tracking with dynamic paths (fixes hardcoded path issues)
    try:
        if not initialize_mlflow_from_config(args.config):
            print("Warning: MLflow initialization failed, but continuing...")
    except Exception as e:
        print(f"Warning: MLflow initialization error: {e}")

    # Setup logging (ensures log file and handlers are created)
    setup_logging(config)

    # Initialize and run trader
    try:
        trader = UnifiedPaperTrader(config, args.models_dir)
        await trader.run_trading_loop(args.iterations)
    except Exception as e:
        print(f"Trading failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
