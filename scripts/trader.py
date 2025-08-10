"""
Paper Trading Script
===================

Main script for running the paper trading bot with trained models.
"""

import sys
import os
import argparse
import yaml
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import sqlite3
from typing import Optional, Dict, Any
import json

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from installed package
from src.utils.logger import setup_logging, TradingBotLogger
from src.data_pipeline.features import FeatureEngine
from src.data_pipeline.preprocess import DataPreprocessor
from src.models.gru_trainer import GRUTrainer
from src.models.lgbm_trainer import LightGBMTrainer
from src.models.ppo_trainer import PPOTrainer
from src.notifier.telegram import TelegramNotifier

class PaperTrader:
    """
    Paper trading bot that uses trained models to make trading decisions.
    """
    
    def __init__(self, config: dict, models_dir: str = "./models"):
        """
        Initialize paper trader.
        
        Args:
            config: Configuration dictionary
            models_dir: Directory containing trained models
        """
        self.config = config
        self.models_dir = models_dir
        
        # Initialize components
        self.logger = TradingBotLogger()
        # Use real-time data fetching instead of data folder
        self.feature_engine = FeatureEngine(config.get('features', {}))
        self.preprocessor = DataPreprocessor()
        # Remove data_dir dependency - trader uses real-time data only
        
        # Initialize notification system
        try:
            self.notifier = TelegramNotifier.from_config(config)
        except Exception as e:
            self.logger.logger.warning(f"Failed to initialize notifications: {e}")
            self.notifier = None
        
        # Trading state
        self.portfolio_value = config.get('trading', {}).get('initial_balance', 10000.0)
        self.positions = {}  # symbol -> quantity
        self.last_prices = {}  # symbol -> last_price
        self.daily_start_value = self.portfolio_value
        
        # Models
        self.models = {}
        self.load_models()
        
        # Trading parameters
        self.symbols = config.get('data', {}).get('symbols', ['BTCEUR'])
        self.max_position_size = config.get('trading', {}).get('max_position_size', 0.1)
        self.transaction_fee = config.get('trading', {}).get('transaction_fee', 0.001)
        
        # Data caching for performance
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # Cache data for 60 seconds
        
        self.logger.logger.info(f"Paper trader initialized with ${self.portfolio_value:,.2f}")
    
    def load_models(self):
        """Load trained models."""
        try:
            # Find latest model files with correct patterns
            model_files = {
                'gru': self._find_latest_model('gru_model_*.pth'),
                'lightgbm': self._find_latest_model('lightgbm_model_*.pkl'),
                'ppo': self._find_latest_model('ppo_model_*.zip')  # PPO models are .zip files
            }
            
            # Load GRU model
            if model_files['gru']:
                try:
                    self.models['gru'] = GRUTrainer.load_model(model_files['gru'], self.config)
                    self.logger.logger.info(f"Loaded GRU model: {model_files['gru']}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load GRU model: {e}")
            
            # Load LightGBM model
            if model_files['lightgbm']:
                try:
                    self.models['lightgbm'] = LightGBMTrainer.load_model(model_files['lightgbm'], self.config)
                    self.logger.logger.info(f"Loaded LightGBM model: {model_files['lightgbm']}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load LightGBM model: {e}")
            
            # Load PPO model
            if model_files['ppo']:
                try:
                    self.models['ppo'] = PPOTrainer.load_model(model_files['ppo'], self.config)
                    self.logger.logger.info(f"Loaded PPO model: {model_files['ppo']}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load PPO model: {e}")
            
            if not self.models:
                self.logger.logger.warning("No models loaded! Trading will use simple strategy.")
        
        except Exception as e:
            self.logger.logger.error(f"Error loading models: {e}")
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """
        Convert symbol format to ccxt standard format.
        
        Args:
            symbol: Symbol in format like BTCEUR, SOLEUR, etc.
            
        Returns:
            Symbol in ccxt format like BTC/EUR, SOL/EUR, etc.
        """
        # Handle different base currencies
        if symbol.endswith('EUR'):
            base = symbol[:-3]
            return f"{base}/EUR"
        elif symbol.endswith('USD'):
            base = symbol[:-3]
            return f"{base}/USD"
        elif symbol.endswith('USDT'):
            base = symbol[:-4]
            return f"{base}/USDT"
        elif symbol.endswith('BTC'):
            base = symbol[:-3]
            return f"{base}/BTC"
        elif symbol.endswith('ETH'):
            base = symbol[:-3]
            return f"{base}/ETH"
        else:
            # If already in correct format or unknown format, return as is
            return symbol

    def _find_latest_model(self, pattern: str) -> Optional[str]:
        """Find the latest model file matching pattern."""
        import glob
        
        model_files = glob.glob(os.path.join(self.models_dir, pattern))
        if model_files:
            # Sort by modification time and return the latest
            return max(model_files, key=os.path.getmtime)
        return None
    

    async def get_market_data(self) -> dict:
        """Get latest market data for all symbols using ccxt with error handling."""
        market_data = {}
        
        # Check cache first
        current_time = time.time()
        cached_data = self._get_cached_data(current_time)
        if cached_data:
            self.logger.logger.debug("Using cached market data")
            return cached_data
        
        # Initialize exchange connection with error handling
        exchange = None
        try:
            # Use synchronous ccxt for now - async ccxt requires different import
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                },
                'timeout': 30000  # 30 second timeout
            })
            # Load markets synchronously
            exchange.load_markets()
        except Exception as e:
            self.logger.logger.error(f"Error initializing exchange connection: {e}")
            # Synchronous ccxt doesn't require explicit connection closing
            # Connection will be closed automatically
            return market_data
        
        # Track API call statistics
        api_calls = 0
        api_errors = 0
        
        for symbol in self.symbols:
            try:
                # Fetch OHLCV data with retry mechanism
                max_retries = 3
                retry_count = 0
                ohlcv = None
                
                while retry_count < max_retries:
                    try:
                        # Convert symbol format to ccxt standard format
                        formatted_symbol = self._convert_symbol_format(symbol)
                        # Use synchronous version of ccxt
                        ohlcv = exchange.fetch_ohlcv(formatted_symbol, '15m', limit=100)
                        api_calls += 1
                        break
                    except ccxt.RateLimitExceeded as e:
                        self.logger.logger.warning(f"Rate limit exceeded for {symbol}, waiting...")
                        # Use asyncio.sleep for async method
                        await asyncio.sleep(10)  # Wait 10 seconds before retry
                        retry_count += 1
                    except ccxt.NetworkError as e:
                        self.logger.logger.warning(f"Network error for {symbol}: {e}")
                        await asyncio.sleep(5)  # Wait 5 seconds before retry
                        retry_count += 1
                    except Exception as e:
                        self.logger.logger.error(f"Error fetching data for {symbol}: {e}")
                        retry_count += 1
                
                # Check if we successfully fetched data
                if ohlcv is None:
                    api_errors += 1
                    self.logger.logger.warning(f"Failed to fetch data for {symbol} after {max_retries} attempts")
                    continue
                
                # Convert to DataFrame
                if not ohlcv:
                    self.logger.logger.warning(f"No data returned for {symbol}")
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('datetime')
                
                # Add missing columns with default values for compatibility
                df['quote_volume'] = df['volume']
                df['trades'] = 0
                df['taker_buy_base'] = 0
                df['taker_buy_quote'] = 0
                
                # Convert to numeric and handle errors
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Use enhanced validation method
                if not self._validate_market_data(df, symbol):
                    continue
                
                # Generate features
                try:
                    features_df = self.feature_engine.generate_all_features(df)
                    self.logger.logger.debug(f"Generated {len(features_df.columns)} features for {symbol}")
                    
                    # Validate features
                    feature_nan_count = features_df.isnull().sum().sum()
                    if feature_nan_count > 0:
                        self.logger.logger.warning(f"Found {feature_nan_count} NaN values in {symbol} features")
                        # Try to clean features
                        features_df = features_df.fillna(0)
                    
                    market_data[symbol] = features_df
                    
                    # Update last price
                    self.last_prices[symbol] = float(df['close'].iloc[-1])
                    self.logger.logger.debug(f"Updated price for {symbol}: {self.last_prices[symbol]}")
                    
                except Exception as e:
                    self.logger.logger.error(f"Error generating features for {symbol}: {e}")
                    continue
                
            except Exception as e:
                api_errors += 1
                self.logger.logger.error(f"Error getting data for {symbol}: {e}")
                # Continue with other symbols even if one fails
                continue
        
        # Synchronous ccxt doesn't require explicit connection closing
        # Connection will be closed automatically
        
        # Cache the data if we got any
        if market_data:
            self._cache_data(market_data, time.time())
        
        # Log API statistics
        self.logger.logger.info(f"API calls: {api_calls}, Errors: {api_errors}")
        
        if not market_data:
            self.logger.logger.warning("No market data available for any symbol")
        else:
            self.logger.logger.info(f"Successfully fetched data for {len(market_data)} symbols")
        
        return market_data
    
    def _get_cached_data(self, current_time: float) -> Optional[Dict[str, Any]]:
        """Get cached data if still valid."""
        if not self.data_cache:
            return None
        
        # Check if cache is still valid
        for symbol in self.symbols:
            if symbol not in self.cache_expiry or current_time > self.cache_expiry[symbol]:
                return None
        
        return self.data_cache.copy()
    
    def _cache_data(self, data: Dict[str, Any], current_time: float) -> None:
        """Cache market data with expiry time."""
        self.data_cache = data.copy()
        expiry_time = current_time + self.cache_duration
        
        for symbol in data.keys():
            self.cache_expiry[symbol] = expiry_time
    
    def _validate_market_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Enhanced data validation for market data.
        
        Args:
            df: DataFrame with market data
            symbol: Symbol being validated
            
        Returns:
            True if data is valid, False otherwise
        """
        if df.empty:
            self.logger.logger.warning(f"Empty DataFrame for {symbol}")
            return False
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            self.logger.logger.error(f"Missing columns {missing_cols} for {symbol}")
            return False
        
        # Check for NaN values in critical columns
        for col in required_columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                self.logger.logger.warning(f"Found {nan_count} NaN values in {col} for {symbol}")
                # Fill NaN values with forward fill, then backward fill
                df[col] = df[col].ffill().bfill()
        
        # Check for negative prices or volumes
        for col in ['open', 'high', 'low', 'close']:
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                self.logger.logger.error(f"Found {negative_count} non-positive prices in {col} for {symbol}")
                return False
        
        negative_volumes = (df['volume'] < 0).sum()
        if negative_volumes > 0:
            self.logger.logger.error(f"Found {negative_volumes} negative volumes for {symbol}")
            return False
        
        # Validate OHLC relationships
        invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
        invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
        if invalid_high > 0 or invalid_low > 0:
            self.logger.logger.error(f"Invalid OHLC relationships in {symbol}: {invalid_high} high, {invalid_low} low")
            return False
        
        # Check data freshness (should be within last 2 hours)
        if isinstance(df.index, pd.DatetimeIndex):
            latest_timestamp = df.index.max()
            # Ensure both timestamps have the same timezone
            if latest_timestamp.tz is None:
                latest_timestamp = latest_timestamp.tz_localize('UTC')
            current_time = pd.Timestamp.now(tz='UTC')
            time_diff = current_time - latest_timestamp
            if time_diff > pd.Timedelta(hours=2):
                self.logger.logger.warning(f"Stale data for {symbol}: {time_diff} old")
                return False
        
        # Check for sufficient data points
        if len(df) < 50:
            self.logger.logger.warning(f"Insufficient data points for {symbol}: {len(df)} (need at least 50)")
            return False
        
        return True
    
    def generate_signals(self, market_data: dict) -> dict:
        """
        Generate trading signals using loaded models.
        
        Args:
            market_data: Dictionary of symbol -> DataFrame
            
        Returns:
            Dictionary of symbol -> signal (-1, 0, 1)
        """
        signals = {}
        
        self.logger.logger.debug(f"Generating signals for {len(market_data)} symbols")
        
        for symbol, df in market_data.items():
            try:
                self.logger.logger.debug(f"Processing {symbol}")
                signal = 0  # Default: hold
                
                if df.empty:
                    self.logger.logger.debug(f"Empty dataframe for {symbol}")
                    signals[symbol] = signal
                    continue
                
                # Prepare features
                feature_names = self.feature_engine.get_feature_names(df)
                self.logger.logger.debug(f"Found {len(feature_names)} feature names for {symbol}")
                features = df[feature_names].dropna()
                self.logger.logger.debug(f"Features shape for {symbol}: {features.shape}")
                
                if features.empty:
                    self.logger.logger.debug(f"Empty features for {symbol}")
                    signals[symbol] = signal
                    continue
                
                # Validate feature count for model compatibility
                original_feature_count = features.shape[1]
                
                # Apply feature validation for each model type
                if 'lightgbm' in self.models:
                    features_lgbm = self.feature_engine.pad_features_for_model(features.copy(), 'lightgbm')
                    if features_lgbm.shape[1] != original_feature_count:
                        self.logger.logger.debug(f"Adjusted features for LightGBM: {original_feature_count} -> {features_lgbm.shape[1]}")
                
                # Get latest features
                latest_features = features.iloc[-1:].values
                self.logger.logger.debug(f"Latest features shape for {symbol}: {latest_features.shape}")
                
                # Ensemble prediction
                predictions = []
                
                # GRU prediction
                if 'gru' in self.models:
                    try:
                        # Prepare sequence for GRU
                        sequence_length = self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
                        if len(features) >= sequence_length:
                            # Scale features using preprocessor (fit on current data for consistency)
                            try:
                                features_scaled = self.preprocessor.fit_transform(features)
                            except Exception as e:
                                self.logger.logger.warning(f"Preprocessing failed for {symbol}: {e}")
                                # Try with just transform if fit fails
                                try:
                                    features_scaled = self.preprocessor.transform(features)
                                except Exception as e2:
                                    self.logger.logger.error(f"Transform also failed for {symbol}: {e2}")
                                    continue
                            
                            # Create sequence
                            sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
                            
                            # Validate sequence
                            if np.isnan(sequence).any() or np.isinf(sequence).any():
                                self.logger.logger.warning(f"Invalid values in sequence for {symbol}, skipping GRU prediction")
                                continue
                            
                            # Predict
                            try:
                                gru_pred = self.models['gru'].predict(sequence)[0]
                                # Validate prediction
                                if np.isnan(gru_pred) or np.isinf(gru_pred):
                                    self.logger.logger.warning(f"Invalid prediction from GRU for {symbol}, skipping")
                                    continue
                                predictions.append(('gru', gru_pred))
                            except Exception as e:
                                self.logger.logger.debug(f"GRU prediction failed for {symbol}: {e}")
                    except Exception as e:
                        self.logger.logger.debug(f"GRU prediction failed for {symbol}: {e}")
                
                # LightGBM prediction
                if 'lightgbm' in self.models:
                    try:
                        # Use properly padded features for LightGBM
                        features_lgbm = self.feature_engine.pad_features_for_model(features.copy(), 'lightgbm')
                        latest_features_lgbm = features_lgbm.iloc[-1:].values
                        
                        # Validate input features
                        if np.isnan(latest_features_lgbm).any() or np.isinf(latest_features_lgbm).any():
                            self.logger.logger.warning(f"Invalid values in features for {symbol}, skipping LightGBM prediction")
                            continue
                        
                        lgbm_pred = self.models['lightgbm'].predict(latest_features_lgbm)[0]
                        
                        # Validate prediction
                        if np.isnan(lgbm_pred) or np.isinf(lgbm_pred):
                            self.logger.logger.warning(f"Invalid prediction from LightGBM for {symbol}, skipping")
                            continue
                            
                        predictions.append(('lightgbm', lgbm_pred))
                    except Exception as e:
                        self.logger.logger.debug(f"LightGBM prediction failed for {symbol}: {e}")
                
                # PPO prediction (if available)
                if 'ppo' in self.models:
                    try:
                        # For PPO, we need to prepare the observation space correctly
                        sequence_length = self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
                        
                        if len(features) >= sequence_length:
                            # Ensure we have exactly 113 market features for PPO compatibility
                            features_ppo = self.feature_engine.pad_features_for_model(features.copy(), 'ppo')
                            
                            # Create sequence for PPO with proper preprocessing
                            try:
                                features_scaled = self.preprocessor.fit_transform(features_ppo)
                            except Exception as e:
                                self.logger.logger.warning(f"Preprocessing failed for {symbol}: {e}")
                                # Try with just transform if fit fails
                                try:
                                    features_scaled = self.preprocessor.transform(features_ppo)
                                except Exception as e2:
                                    self.logger.logger.error(f"Transform also failed for {symbol}: {e2}")
                                    continue
                            
                            # Create sequence with proper 2D shape for PPO: (sequence_length, features)
                            sequence_2d = features_scaled[-sequence_length:]  # Shape: (20, 113)
                            
                            # Add portfolio features to match TradingEnvironment format (20, 116)
                            portfolio_features = np.array([
                                1.0,  # balance_ratio (normalized)
                                0.0,  # position_ratio
                                0.0   # pnl_ratio
                            ])
                            
                            # Repeat portfolio features for each timestep
                            portfolio_matrix = np.tile(portfolio_features, (sequence_length, 1))  # Shape: (20, 3)
                            
                            # Combine market and portfolio features
                            observation = np.concatenate([sequence_2d, portfolio_matrix], axis=1)  # Shape: (20, 116)
                            
                            # Validate observation shape and values
                            if observation.shape != (sequence_length, 116):
                                self.logger.logger.warning(f"PPO observation shape mismatch: {observation.shape} != ({sequence_length}, 116)")
                                continue
                                
                            if np.isnan(observation).any() or np.isinf(observation).any():
                                self.logger.logger.warning(f"Invalid observation for {symbol}, skipping PPO")
                                continue
                            
                            # PPO expects 2D observation: (sequence_length, features)
                            ppo_action, _ = self.models['ppo'].predict(observation, deterministic=True)
                            
                            # Convert action to prediction value
                            # Action 0 = Hold, 1 = Buy, 2 = Sell
                            if ppo_action == 1:
                                ppo_pred = 0.002  # Strong buy signal
                            elif ppo_action == 2:
                                ppo_pred = -0.002  # Strong sell signal
                            else:
                                ppo_pred = 0.0  # Hold signal
                            
                            # Validate prediction
                            if np.isnan(ppo_pred) or np.isinf(ppo_pred):
                                self.logger.logger.warning(f"Invalid prediction from PPO for {symbol}, skipping")
                                continue
                                
                            predictions.append(('ppo', ppo_pred))
                            self.logger.logger.debug(f"PPO prediction for {symbol}: action={ppo_action}, pred={ppo_pred}")
                        else:
                            self.logger.logger.debug(f"Insufficient data for PPO sequence: {len(features)} < {sequence_length}")
                    except Exception as e:
                        self.logger.logger.debug(f"PPO prediction failed for {symbol}: {e}")
                
                # Convert predictions to signals
                if predictions:
                    # Simple ensemble: average predictions
                    try:
                        prediction_values = [pred for _, pred in predictions]
                        avg_prediction = np.mean(prediction_values)
                        
                        # Validate average prediction
                        if np.isnan(avg_prediction) or np.isinf(avg_prediction):
                            self.logger.logger.warning(f"Invalid average prediction for {symbol}: {avg_prediction}")
                            signal = 0  # Hold
                        else:
                            # Convert to signal
                            threshold = 0.001  # 0.1% threshold
                            if avg_prediction > threshold:
                                signal = 1  # Buy
                            elif avg_prediction < -threshold:
                                signal = -1  # Sell
                            else:
                                signal = 0  # Hold
                    except Exception as e:
                        self.logger.logger.warning(f"Error calculating average prediction for {symbol}: {e}")
                        signal = 0  # Hold on error
                else:
                    signal = 0  # Hold if no predictions
                
                signals[symbol] = signal
                
                if predictions and signal != 0:
                    avg_prediction = np.mean([pred for _, pred in predictions])
                    self.logger.logger.info(f"Signal for {symbol}: {signal} (prediction: {avg_prediction:.6f})")
                elif not predictions:
                    self.logger.logger.debug(f"No predictions available for {symbol}")
                else:
                    self.logger.logger.debug(f"Hold signal for {symbol}")
            
            except Exception as e:
                self.logger.logger.error(f"Error generating signal for {symbol}: {e}")
                signals[symbol] = 0
        
        self.logger.logger.debug(f"Generated signals for {len(signals)} symbols")
        return signals
    
    def execute_trades(self, signals: dict):
        """
        Execute trades based on signals.
        
        Args:
            signals: Dictionary of symbol -> signal
        """
        for symbol, signal in signals.items():
            if signal == 0:
                continue
            
            try:
                current_price = self.last_prices.get(symbol)
                if not current_price or current_price <= 0:
                    self.logger.logger.warning(f"Invalid price for {symbol}: {current_price}")
                    continue
                
                current_position = self.positions.get(symbol, 0.0)
                
                # Calculate target position based on signal strength
                # Signal values are typically between -1 and 1
                position_ratio = abs(signal) * self.max_position_size
                target_position_value = self.portfolio_value * position_ratio
                target_position = target_position_value / current_price if current_price > 0 else 0.0
                
                # Apply direction based on signal
                if signal < 0:  # Sell/Short
                    target_position = -target_position
                
                # Calculate trade size
                trade_size = target_position - current_position
                
                # Check minimum trade size
                min_trade_size = 0.0001  # Minimum trade size for most crypto exchanges
                if abs(trade_size) < min_trade_size:
                    continue
                
                # Execute trade
                trade_value = abs(trade_size) * current_price
                fee = trade_value * self.transaction_fee
                
                # Check if we have enough funds for the trade
                if trade_size > 0 and (trade_value + fee) > self.portfolio_value:
                    # Not enough funds, adjust trade size
                    available_funds = self.portfolio_value / (current_price * (1 + self.transaction_fee))
                    trade_size = min(trade_size, available_funds)
                    if abs(trade_size) < min_trade_size:
                        continue
                    trade_value = abs(trade_size) * current_price
                    fee = trade_value * self.transaction_fee
                
                # Update portfolio
                if trade_size > 0:  # Buying
                    self.portfolio_value -= (trade_value + fee)
                    side = "BUY"
                else:  # Selling
                    self.portfolio_value += (trade_value - fee)
                    side = "SELL"
                
                # Update position
                self.positions[symbol] = target_position
                
                # Log trade
                self.logger.log_trade(
                    symbol=symbol,
                    side=side,
                    quantity=abs(trade_size),
                    price=current_price,
                    portfolio_value=self.portfolio_value
                )
                
                # Send notification
                if self.notifier and self.notifier.enabled:
                    asyncio.create_task(self.notifier.send_trade_notification(
                        symbol=symbol,
                        side=side,
                        quantity=abs(trade_size),
                        price=current_price,
                        portfolio_value=self.portfolio_value
                    ))
            
            except Exception as e:
                self.logger.logger.error(f"Error executing trade for {symbol}: {e}")
    
    def calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value including positions."""
        total_value = self.portfolio_value
        
        # Validate that portfolio_value is not negative
        if total_value < 0:
            self.logger.logger.warning(f"Negative portfolio value: {total_value}")
            total_value = 0.0
        
        for symbol, quantity in self.positions.items():
            if abs(quantity) > 1e-6:  # Non-zero position
                current_price = self.last_prices.get(symbol, 0)
                # Validate price
                if current_price > 0:
                    position_value = quantity * current_price
                    # Validate position value
                    if not (np.isnan(position_value) or np.isinf(position_value)):
                        total_value += position_value
                    else:
                        self.logger.logger.warning(f"Invalid position value for {symbol}: {position_value}")
                else:
                    self.logger.logger.warning(f"Invalid price for {symbol}: {current_price}")
        
        # Ensure total value is not negative
        if total_value < 0:
            self.logger.logger.warning(f"Calculated negative portfolio value: {total_value}, setting to 0")
            total_value = 0.0
            
        return total_value
    
    async def send_daily_report(self):
        """Send daily portfolio report."""
        try:
            current_value = self.calculate_portfolio_value()
            daily_pnl = current_value - self.daily_start_value
            daily_return = daily_pnl / self.daily_start_value
            
            # Log portfolio update
            self.logger.log_portfolio_update(
                portfolio_value=current_value,
                daily_pnl=daily_pnl,
                positions=self.positions
            )
            
            # Send notification
            if self.notifier and self.notifier.enabled:
                await self.notifier.send_portfolio_update(
                    portfolio_value=current_value,
                    daily_pnl=daily_pnl,
                    daily_return=daily_return,
                    positions=self.positions
                )
            
            # Reset daily start value
            self.daily_start_value = current_value
        
        except Exception as e:
            self.logger.logger.error(f"Error sending daily report: {e}")
    
    async def run_trading_loop(self, interval_minutes: int = 15):
        """
        Main trading loop.
        
        Args:
            interval_minutes: Trading interval in minutes
        """
        self.logger.logger.info(f"Starting paper trading loop (interval: {interval_minutes} minutes)")
        
        if self.notifier and self.notifier.enabled:
            await self.notifier.send_system_status(
                status="RUNNING",
                active_models=list(self.models.keys())
            )
        
        last_daily_report = datetime.now().date()
        
        try:
            while True:
                loop_start = time.time()
                
                try:
                    # Get market data
                    market_data = await self.get_market_data()
                    
                    if market_data:
                        # Generate signals
                        signals = self.generate_signals(market_data)
                        
                        # Execute trades
                        self.execute_trades(signals)
                        
                        # Send daily report if needed
                        current_date = datetime.now().date()
                        if current_date > last_daily_report:
                            await self.send_daily_report()
                            last_daily_report = current_date
                    
                    else:
                        self.logger.logger.warning("No market data available")
                
                except Exception as e:
                    self.logger.logger.error(f"Error in trading loop: {e}")
                    
                    if self.notifier and self.notifier.enabled:
                        await self.notifier.send_error_notification(
                            error_type="TRADING_LOOP_ERROR",
                            error_message=str(e)
                        )
                
                # Wait for next interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, interval_minutes * 60 - loop_duration)
                
                self.logger.logger.debug(f"Loop completed in {loop_duration:.2f}s, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.logger.info("Trading loop interrupted by user")
            
            if self.notifier and self.notifier.enabled:
                await self.notifier.send_system_status(status="STOPPED")
        
        except Exception as e:
            self.logger.logger.error(f"Trading loop failed: {e}")
            
            if self.notifier and self.notifier.enabled:
                await self.notifier.send_error_notification(
                    error_type="TRADING_LOOP_FAILURE",
                    error_message=str(e)
                )
            
            raise

def load_config(config_path: str = "src/config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Crypto Trading Bot - Paper Trader')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory containing trained models')
    parser.add_argument('--interval', type=int, default=15,
                       help='Trading interval in minutes')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration. Exiting.")
        return
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("CRYPTO TRADING BOT - PAPER TRADER")
    logger.info("=" * 60)
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Trading interval: {args.interval} minutes")
    
    # Initialize and run paper trader
    try:
        trader = PaperTrader(config, args.models_dir)
        await trader.run_trading_loop(args.interval)
    
    except Exception as e:
        logger.error(f"Paper trader failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())