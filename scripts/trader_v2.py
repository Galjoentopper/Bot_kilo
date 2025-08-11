"""
Paper Trading Script V2
=======================

Enhanced paper trading bot with realistic position tracking,
ownership validation, and comprehensive Telegram notifications.
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
from typing import Optional, Dict, Any, Tuple
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
from src.trading.position_tracker import PositionTracker, OrderSide

class EnhancedPaperTrader:
    """
    Enhanced paper trading bot with realistic position tracking and validation.
    """
    
    def __init__(self, config: dict, models_dir: str = "./models"):
        """
        Initialize enhanced paper trader.
        
        Args:
            config: Configuration dictionary
            models_dir: Directory containing trained models
        """
        self.config = config
        self.models_dir = models_dir
        
        # Initialize components
        self.logger = TradingBotLogger()
        self.feature_engine = FeatureEngine(config.get('features', {}))
        self.preprocessor = DataPreprocessor()
        
        # Initialize notification system
        try:
            self.notifier = TelegramNotifier.from_config(config)
            self.logger.logger.info("Telegram notifier initialized successfully")
        except Exception as e:
            self.logger.logger.warning(f"Failed to initialize notifications: {e}")
            self.notifier = None
        
        # Initialize position tracker with realistic constraints
        trading_config = config.get('trading', {})
        self.position_tracker = PositionTracker(
            initial_capital=trading_config.get('initial_balance', 10000.0),
            transaction_fee=trading_config.get('transaction_fee', 0.001),
            slippage=trading_config.get('slippage', 0.0005),
            max_position_size=trading_config.get('max_position_size', 0.1)
        )
        
        # Trading state
        self.last_prices = {}  # symbol -> last_price
        self.daily_start_value = self.position_tracker.initial_capital
        
        # Models
        self.models = {}
        self.load_models()
        
        # Trading parameters
        self.symbols = config.get('data', {}).get('symbols', ['BTCEUR'])
        
        # Symbol-specific thresholds based on typical volatility
        self.symbol_thresholds = {
            'BTCEUR': 0.00005,   # BTC is less volatile in percentage terms
            'ETHEUR': 0.00005,   # ETH similar to BTC
            'SOLEUR': 0.00010,   # SOL is more volatile
            'ADAEUR': 0.00010,   # ADA is more volatile
            'XRPEUR': 0.00010,   # XRP is more volatile
        }
        self.default_threshold = 0.00008
        
        # Data caching for performance
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # Cache data for 60 seconds
        
        # Performance tracking
        self.performance_history = []
        self.rejected_trades_count = 0
        
        self.logger.logger.info(f"Enhanced paper trader initialized with ${self.position_tracker.initial_capital:,.2f}")
    
    def load_models(self):
        """Load trained models."""
        try:
            # Find latest model files with correct patterns
            model_files = {
                'gru': self._find_latest_model('gru_model_*.pth'),
                'lightgbm': self._find_latest_model('lightgbm_model_*.pkl'),
                'ppo': self._find_latest_model('ppo_model_*.zip')
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
    
    def _find_latest_model(self, pattern: str) -> Optional[str]:
        """Find the latest model file matching pattern."""
        import glob
        
        model_files = glob.glob(os.path.join(self.models_dir, pattern))
        if model_files:
            # Sort by modification time and return the latest
            return max(model_files, key=os.path.getmtime)
        return None
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format to ccxt standard format."""
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
            return symbol

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
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                },
                'timeout': 30000  # 30 second timeout
            })
            exchange.load_markets()
        except Exception as e:
            self.logger.logger.error(f"Error initializing exchange connection: {e}")
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
                        formatted_symbol = self._convert_symbol_format(symbol)
                        ohlcv = exchange.fetch_ohlcv(formatted_symbol, '15m', limit=100)
                        api_calls += 1
                        break
                    except ccxt.RateLimitExceeded as e:
                        self.logger.logger.warning(f"Rate limit exceeded for {symbol}, waiting...")
                        await asyncio.sleep(10)
                        retry_count += 1
                    except ccxt.NetworkError as e:
                        self.logger.logger.warning(f"Network error for {symbol}: {e}")
                        await asyncio.sleep(5)
                        retry_count += 1
                    except Exception as e:
                        self.logger.logger.error(f"Error fetching data for {symbol}: {e}")
                        retry_count += 1
                
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
                
                # Validate market data
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
                continue
        
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
        """Enhanced data validation for market data."""
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
        """Generate trading signals using loaded models."""
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
                
                # Ensemble prediction
                predictions = []
                
                # GRU prediction
                if 'gru' in self.models:
                    try:
                        sequence_length = self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
                        if len(features) >= sequence_length:
                            features_gru = self.feature_engine.pad_features_for_model(features.copy(), 'gru')
                            self.logger.logger.debug(f"GRU features shape after padding: {features_gru.shape}")
                            
                            try:
                                features_scaled = self.preprocessor.fit_transform(features_gru)
                            except Exception as e:
                                self.logger.logger.warning(f"Preprocessing failed for {symbol}: {e}")
                                try:
                                    features_scaled = self.preprocessor.transform(features_gru)
                                except Exception as e2:
                                    self.logger.logger.error(f"Transform also failed for {symbol}: {e2}")
                                    continue
                            
                            sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
                            
                            if np.isnan(sequence).any() or np.isinf(sequence).any():
                                self.logger.logger.warning(f"Invalid values in sequence for {symbol}, skipping GRU prediction")
                                continue
                            
                            try:
                                gru_pred = self.models['gru'].predict(sequence)[0]
                                if np.isnan(gru_pred) or np.isinf(gru_pred):
                                    self.logger.logger.warning(f"Invalid prediction from GRU for {symbol}, skipping")
                                    continue
                                predictions.append(('gru', gru_pred))
                                self.logger.logger.debug(f"GRU prediction for {symbol}: {gru_pred}")
                            except Exception as e:
                                self.logger.logger.debug(f"GRU prediction failed for {symbol}: {e}")
                    except Exception as e:
                        self.logger.logger.debug(f"GRU prediction failed for {symbol}: {e}")
                
                # LightGBM prediction
                if 'lightgbm' in self.models:
                    try:
                        features_lgbm = self.feature_engine.pad_features_for_model(features.copy(), 'lightgbm')
                        latest_features_lgbm = features_lgbm.iloc[-1:].values
                        
                        if np.isnan(latest_features_lgbm).any() or np.isinf(latest_features_lgbm).any():
                            self.logger.logger.warning(f"Invalid values in features for {symbol}, skipping LightGBM prediction")
                            continue
                        
                        lgbm_pred = self.models['lightgbm'].predict(latest_features_lgbm)[0]
                        
                        if np.isnan(lgbm_pred) or np.isinf(lgbm_pred):
                            self.logger.logger.warning(f"Invalid prediction from LightGBM for {symbol}, skipping")
                            continue
                            
                        predictions.append(('lightgbm', lgbm_pred))
                        self.logger.logger.debug(f"LightGBM prediction for {symbol}: {lgbm_pred}")
                    except Exception as e:
                        self.logger.logger.debug(f"LightGBM prediction failed for {symbol}: {e}")
                
                # PPO prediction
                if 'ppo' in self.models:
                    try:
                        sequence_length = self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
                        
                        if len(features) >= sequence_length:
                            features_ppo = self.feature_engine.pad_features_for_model(features.copy(), 'ppo')
                            ppo_feature_names = self.feature_engine.get_feature_names(features_ppo)
                            
                            try:
                                features_scaled = self.preprocessor.fit_transform(features_ppo[ppo_feature_names])
                            except Exception as e:
                                self.logger.logger.warning(f"Preprocessing failed for {symbol}: {e}")
                                try:
                                    features_scaled = self.preprocessor.transform(features_ppo[ppo_feature_names])
                                except Exception as e2:
                                    self.logger.logger.error(f"Transform also failed for {symbol}: {e2}")
                                    continue
                            
                            sequence_2d = features_scaled[-sequence_length:]
                            
                            portfolio_features = np.array([
                                1.0,  # balance_ratio (normalized)
                                0.0,  # position_ratio
                                0.0,  # pnl_ratio
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # additional features
                            ])
                            
                            portfolio_matrix = np.tile(portfolio_features, (sequence_length, 1))
                            observation = np.concatenate([sequence_2d, portfolio_matrix], axis=1)
                            
                            if observation.shape != (sequence_length, 125):
                                self.logger.logger.warning(f"PPO observation shape mismatch: {observation.shape} != ({sequence_length}, 125)")
                                continue
                                
                            if np.isnan(observation).any() or np.isinf(observation).any():
                                self.logger.logger.warning(f"Invalid observation for {symbol}, skipping PPO")
                                continue
                            
                            ppo_action, _ = self.models['ppo'].predict(observation, deterministic=True)
                            
                            if hasattr(ppo_action, '__len__'):
                                action_value = int(ppo_action[0])
                            else:
                                action_value = int(ppo_action)
                            
                            if action_value == 1:
                                ppo_pred = 0.0015  # Buy signal
                            elif action_value == 2:
                                ppo_pred = -0.0015  # Sell signal
                            else:
                                ppo_pred = 0.0  # Hold signal
                            
                            if np.isnan(ppo_pred) or np.isinf(ppo_pred):
                                self.logger.logger.warning(f"Invalid prediction from PPO for {symbol}, skipping")
                                continue
                                
                            predictions.append(('ppo', ppo_pred))
                            self.logger.logger.debug(f"PPO prediction for {symbol}: action={ppo_action}, pred={ppo_pred}")
                        else:
                            self.logger.logger.debug(f"Insufficient data for PPO sequence: {len(features)} < {sequence_length}")
                    except Exception as e:
                        self.logger.logger.debug(f"PPO prediction failed for {symbol}: {e}")
                
                # Calculate recent volatility for dynamic threshold adjustment
                volatility = 0.01  # Default volatility
                if 'close' in df.columns and len(df) > 20:
                    recent_returns = df['close'].pct_change().dropna().tail(20)
                    volatility = recent_returns.std()
                    self.logger.logger.debug(f"{symbol} recent volatility: {volatility:.6f}")
                
                # Convert predictions to signals
                if predictions:
                    try:
                        prediction_values = [pred for _, pred in predictions]
                        avg_prediction = np.mean(prediction_values)
                        
                        if np.isnan(avg_prediction) or np.isinf(avg_prediction):
                            self.logger.logger.warning(f"Invalid average prediction for {symbol}: {avg_prediction}")
                            signal = 0  # Hold
                        else:
                            # Get symbol-specific threshold
                            base_threshold = self.symbol_thresholds.get(symbol, self.default_threshold)
                            
                            # Adjust threshold based on volatility
                            threshold = base_threshold * (1 + volatility * 5)
                            
                            # Convert to signal
                            if avg_prediction > threshold:
                                signal = 1  # Buy
                            elif avg_prediction < -threshold:
                                signal = -1  # Sell
                            else:
                                signal = 0  # Hold
                            
                            self.logger.logger.info(f"Ensemble prediction for {symbol}: avg={avg_prediction:.8f}, threshold={threshold:.8f}, signal={signal}")
                            self.logger.logger.info(f"Individual predictions: {predictions}")
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
    
    async def execute_trades(self, signals: dict):
        """Execute trades based on signals with proper validation."""
        # Calculate available capital per symbol
        active_symbols = [s for s, sig in signals.items() if sig != 0]
        if not active_symbols:
            return
        
        # Allocate capital across active symbols
        portfolio_value = self.position_tracker.calculate_portfolio_value(self.last_prices)
        capital_per_symbol = portfolio_value * self.position_tracker.max_position_size / max(len(active_symbols), 1)
        
        for symbol, signal in signals.items():
            if signal == 0:
                continue
            
            try:
                current_price = self.last_prices.get(symbol)
                if not current_price or current_price <= 0:
                    self.logger.logger.warning(f"Invalid price for {symbol}: {current_price}")
                    continue
                
                # Get current position from position tracker
                current_position = self.position_tracker.positions.get(symbol)
                current_quantity = current_position.quantity if current_position else 0.0
                
                # Use allocated capital for this symbol
                target_position_value = capital_per_symbol
                
                if signal > 0:  # Buy signal
                    # Calculate how much to buy
                    target_quantity = target_position_value / current_price
                    buy_quantity = target_quantity - current_quantity
                    
                    # Check minimum trade size (adjusted for each symbol)
                    min_trade_sizes = {
                        'BTCEUR': 0.00001,   # Smaller for expensive assets
                        'ETHEUR': 0.0001,
                        'SOLEUR': 0.01,
                        'ADAEUR': 1.0,
                        'XRPEUR': 1.0
                    }
                    min_trade_size = min_trade_sizes.get(symbol, 0.001)
                    
                    if buy_quantity > min_trade_size:
                        # Execute buy order with validation
                        success, trade, error_msg = self.position_tracker.execute_buy(
                            symbol=symbol,
                            quantity=buy_quantity,
                            price=current_price
                        )
                        
                        if success and trade:
                            # Log successful trade
                            self.logger.log_trade(
                                symbol=symbol,
                                side="BUY",
                                quantity=buy_quantity,
                                price=current_price,
                                portfolio_value=portfolio_value
                            )
                            
                            # Send Telegram notification
                            if self.notifier and self.notifier.enabled:
                                await self._send_trade_notification(trade, portfolio_value)
                            
                            self.logger.logger.info(f"Executed SELL {sell_quantity:.6f} {symbol} @ {current_price:.2f}")
                            
                            self.logger.logger.info(f"Executed BUY {buy_quantity:.6f} {symbol} @ {current_price:.2f}")
                        else:
                            # Log rejected trade
                            self.rejected_trades_count += 1
                            self.logger.logger.warning(f"Buy order rejected for {symbol}: {error_msg}")
                            
                            # Send rejection notification
                            if self.notifier and self.notifier.enabled:
                                await self._send_rejection_notification(
                                    symbol=symbol,
                                    side="BUY",
                                    quantity=buy_quantity,
                                    price=current_price,
                                    reason=error_msg
                                )
                
                elif signal < 0:  # Sell signal
                    # Calculate how much to sell
                    if current_quantity > 0:
                        sell_quantity = min(current_quantity, target_position_value / current_price)
                        
                        if sell_quantity > min_trade_size:
                            # Execute sell order with validation
                            success, trade, error_msg = self.position_tracker.execute_sell(
                                symbol=symbol,
                                quantity=sell_quantity,
                                price=current_price
                            )
                            
                            if success and trade:
                                # Log successful trade
                                self.logger.log_trade(
                                    symbol=symbol,
                                    side="SELL",
                                    quantity=sell_quantity,
                                    price=current_price,
                                    portfolio_value=portfolio_value
                                )
                                
                                # Send Telegram notification
                                if self.notifier and self.notifier.enabled:
                                    await self._send_trade_notification(trade, portfolio_value)
                            else:
                                # Log rejected trade
                                self.rejected_trades_count += 1
                                self.logger.logger.warning(f"Sell order rejected for {symbol}: {error_msg}")
                                
                                # Send rejection notification
                                if self.notifier and self.notifier.enabled:
                                    await self._send_rejection_notification(
                                        symbol=symbol,
                                        side="SELL",
                                        quantity=sell_quantity,
                                        price=current_price,
                                        reason=error_msg
                                    )
                    else:
                        self.logger.logger.info(f"Sell signal for {symbol} ignored - no position to sell")
            
            except Exception as e:
                self.logger.logger.error(f"Error executing trade for {symbol}: {e}")
    
    async def _send_trade_notification(self, trade, portfolio_value: float):
        """Send enhanced trade notification via Telegram."""
        try:
            # Get position summary
            position = self.position_tracker.positions.get(trade.symbol)
            position_quantity = position.quantity if position else 0.0
            avg_price = position.average_entry_price if position else 0.0
            
            # Calculate P&L if selling
            realized_pnl = 0.0
            if trade.side == OrderSide.SELL and position:
                # Estimate realized P&L for this trade
                realized_pnl = (trade.price - avg_price) * trade.quantity - trade.commission - trade.slippage
            
            message = (
                f"🔔 **Trade Executed**\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"**Symbol:** {trade.symbol}\n"
                f"**Side:** {trade.side.value} {'📈' if trade.side == OrderSide.BUY else '📉'}\n"
                f"**Quantity:** {trade.quantity:.6f}\n"
                f"**Price:** ${trade.price:.2f}\n"
                f"**Value:** ${trade.quantity * trade.price:.2f}\n"
                f"**Fees:** ${trade.commission + trade.slippage:.2f}\n"
            )
            
            if trade.side == OrderSide.SELL and realized_pnl != 0:
                message += f"**Realized P&L:** ${realized_pnl:+.2f} {'🟢' if realized_pnl > 0 else '🔴'}\n"
            
            message += (
                f"\n**Position Update:**\n"
                f"• Current Position: {position_quantity:.6f}\n"
                f"• Average Price: ${avg_price:.2f}\n"
                f"• Position Value: ${position_quantity * trade.price:.2f}\n"
                f"\n**Portfolio:**\n"
                f"• Total Value: ${portfolio_value:.2f}\n"
                f"• Cash: ${self.position_tracker.cash:.2f}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
            
            await self.notifier.send_message(message)
            
        except Exception as e:
            self.logger.logger.error(f"Error sending trade notification: {e}")
    
    async def _send_rejection_notification(self, symbol: str, side: str, quantity: float, price: float, reason: str):
        """Send trade rejection notification via Telegram."""
        try:
            message = (
                f"⚠️ **Trade Rejected**\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"**Symbol:** {symbol}\n"
                f"**Side:** {side} {'📈' if side == 'BUY' else '📉'}\n"
                f"**Quantity:** {quantity:.6f}\n"
                f"**Price:** ${price:.2f}\n"
                f"**Reason:** {reason}\n"
                f"\n**Current Status:**\n"
                f"• Cash Available: ${self.position_tracker.cash:.2f}\n"
            )
            
            # Add position info if relevant
            position = self.position_tracker.positions.get(symbol)
            if position and position.quantity > 0:
                message += f"• Position in {symbol}: {position.quantity:.6f}\n"
            
            message += (
                f"• Total Rejected Trades: {self.rejected_trades_count}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
            
            await self.notifier.send_message(message)
            
        except Exception as e:
            self.logger.logger.error(f"Error sending rejection notification: {e}")
    
    async def send_daily_report(self):
        """Send comprehensive daily portfolio report with enhanced metrics."""
        try:
            # Get portfolio summary
            portfolio_summary = self.position_tracker.get_portfolio_summary(self.last_prices)
            
            # Calculate daily metrics
            current_value = portfolio_summary['portfolio_value']
            daily_pnl = current_value - self.daily_start_value
            daily_return = daily_pnl / self.daily_start_value
            
            # Log portfolio update
            self.logger.log_portfolio_update(
                portfolio_value=current_value,
                daily_pnl=daily_pnl,
                positions={p['symbol']: p['quantity'] for p in portfolio_summary['positions']}
            )
            
            # Build detailed report message
            message = (
                f"📊 **Daily Portfolio Report**\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"\n**Portfolio Performance:**\n"
                f"• Total Value: ${current_value:,.2f}\n"
                f"• Daily P&L: ${daily_pnl:+,.2f} ({daily_return:+.2%}) "
                f"{'🟢' if daily_pnl >= 0 else '🔴'}\n"
                f"• Total Return: ${portfolio_summary['total_pnl']:+,.2f} "
                f"({portfolio_summary['total_return_pct']:+.2f}%)\n"
                f"• Realized P&L: ${portfolio_summary['realized_pnl']:+,.2f}\n"
                f"• Unrealized P&L: ${portfolio_summary['unrealized_pnl']:+,.2f}\n"
                f"\n**Account Status:**\n"
                f"• Cash Balance: ${portfolio_summary['cash']:,.2f}\n"
                f"• Positions Value: ${current_value - portfolio_summary['cash']:,.2f}\n"
                f"• Total Trades: {portfolio_summary['total_trades']}\n"
                f"• Rejected Orders: {portfolio_summary['rejected_orders']}\n"
            )
            
            # Add position details
            if portfolio_summary['positions']:
                message += f"\n**Open Positions:**\n"
                for pos in portfolio_summary['positions']:
                    pnl_emoji = '🟢' if pos['unrealized_pnl'] >= 0 else '🔴'
                    message += (
                        f"\n• **{pos['symbol']}**\n"
                        f"  - Quantity: {pos['quantity']:.6f}\n"
                        f"  - Avg Price: ${pos['average_price']:.2f}\n"
                        f"  - Current: ${pos['current_price']:.2f}\n"
                        f"  - Value: ${pos['market_value']:.2f}\n"
                        f"  - Unrealized P&L: ${pos['unrealized_pnl']:+.2f} {pnl_emoji}\n"
                    )
            else:
                message += f"\n**No open positions**\n"
            
            # Add performance metrics
            if len(self.performance_history) > 0:
                returns = [p['daily_return'] for p in self.performance_history[-30:]]  # Last 30 days
                if returns:
                    avg_return = np.mean(returns)
                    volatility = np.std(returns) * np.sqrt(365)  # Annualized
                    sharpe = (avg_return * 365) / volatility if volatility > 0 else 0
                    
                    message += (
                        f"\n**Performance Metrics (30d):**\n"
                        f"• Avg Daily Return: {avg_return:.2%}\n"
                        f"• Volatility: {volatility:.2%}\n"
                        f"• Sharpe Ratio: {sharpe:.2f}\n"
                    )
            
            message += f"\n━━━━━━━━━━━━━━━━━━"
            
            # Send notification
            if self.notifier and self.notifier.enabled:
                await self.notifier.send_message(message)
            
            # Store performance data
            self.performance_history.append({
                'date': datetime.now(),
                'portfolio_value': current_value,
                'daily_pnl': daily_pnl,
                'daily_return': daily_return,
                'total_return': portfolio_summary['total_return'],
                'positions': len(portfolio_summary['positions'])
            })
            
            # Reset daily start value
            self.daily_start_value = current_value
        
        except Exception as e:
            self.logger.logger.error(f"Error sending daily report: {e}")
    
    async def run_trading_loop(self, interval_minutes: int = 15):
        """
        Main trading loop with enhanced error handling and notifications.
        
        Args:
            interval_minutes: Trading interval in minutes
        """
        self.logger.logger.info(f"Starting enhanced paper trading loop (interval: {interval_minutes} minutes)")
        
        # Send startup notification
        if self.notifier and self.notifier.enabled:
            startup_message = (
                f"🚀 **Trading Bot Started**\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"**Initial Capital:** ${self.position_tracker.initial_capital:,.2f}\n"
                f"**Active Models:** {', '.join(self.models.keys()) if self.models else 'None'}\n"
                f"**Symbols:** {', '.join(self.symbols)}\n"
                f"**Max Position Size:** {self.position_tracker.max_position_size * 100:.0f}%\n"
                f"**Transaction Fee:** {self.position_tracker.transaction_fee * 100:.1f}%\n"
                f"**Trading Interval:** {interval_minutes} minutes\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
            await self.notifier.send_message(startup_message)
        
        last_daily_report = datetime.now().date()
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        try:
            while True:
                loop_start = time.time()
                
                try:
                    # Get market data
                    market_data = await self.get_market_data()
                    
                    if market_data:
                        # Generate signals
                        signals = self.generate_signals(market_data)
                        
                        # Execute trades with validation
                        await self.execute_trades(signals)
                        
                        # Send hourly portfolio update
                        if datetime.now().minute == 0:  # On the hour
                            portfolio_summary = self.position_tracker.get_portfolio_summary(self.last_prices)
                            hourly_message = (
                                f"📈 **Hourly Update**\n"
                                f"Portfolio Value: ${portfolio_summary['portfolio_value']:,.2f}\n"
                                f"Active Positions: {len(portfolio_summary['positions'])}\n"
                                f"Cash Available: ${portfolio_summary['cash']:,.2f}"
                            )
                            if self.notifier and self.notifier.enabled:
                                await self.notifier.send_message(hourly_message)
                        
                        # Send daily report if needed
                        current_date = datetime.now().date()
                        if current_date > last_daily_report:
                            await self.send_daily_report()
                            last_daily_report = current_date
                        
                        # Reset error counter on successful iteration
                        consecutive_errors = 0
                    
                    else:
                        self.logger.logger.warning("No market data available")
                        consecutive_errors += 1
                
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.logger.error(f"Error in trading loop: {e}")
                    
                    # Send error notification if critical
                    if consecutive_errors >= max_consecutive_errors:
                        if self.notifier and self.notifier.enabled:
                            error_message = (
                                f"🚨 **Critical Error**\n"
                                f"━━━━━━━━━━━━━━━━━━\n"
                                f"**Error:** Trading loop has failed {consecutive_errors} times\n"
                                f"**Last Error:** {str(e)}\n"
                                f"**Action:** Bot will continue trying...\n"
                                f"━━━━━━━━━━━━━━━━━━"
                            )
                            await self.notifier.send_message(error_message)
                
                # Wait for next interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, interval_minutes * 60 - loop_duration)
                
                self.logger.logger.debug(f"Loop completed in {loop_duration:.2f}s, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.logger.info("Trading loop interrupted by user")
            
            # Send shutdown notification
            if self.notifier and self.notifier.enabled:
                portfolio_summary = self.position_tracker.get_portfolio_summary(self.last_prices)
                shutdown_message = (
                    f"🛑 **Trading Bot Stopped**\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"**Final Portfolio Value:** ${portfolio_summary['portfolio_value']:,.2f}\n"
                    f"**Total Return:** {portfolio_summary['total_return_pct']:+.2f}%\n"
                    f"**Total Trades:** {portfolio_summary['total_trades']}\n"
                    f"**Session Duration:** {datetime.now() - self.performance_history[0]['date'] if self.performance_history else 'N/A'}\n"
                    f"━━━━━━━━━━━━━━━━━━"
                )
                await self.notifier.send_message(shutdown_message)
        
        except Exception as e:
            self.logger.logger.error(f"Trading loop failed: {e}")
            
            if self.notifier and self.notifier.enabled:
                await self.notifier.send_message(
                    f"💥 **Fatal Error**\n"
                    f"Trading bot crashed: {str(e)}"
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
    parser = argparse.ArgumentParser(description='Enhanced Crypto Trading Bot - Paper Trader V2')
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
    logger.info("ENHANCED CRYPTO TRADING BOT - PAPER TRADER V2")
    logger.info("=" * 60)
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Trading interval: {args.interval} minutes")
    logger.info("Features: Realistic position tracking, ownership validation, FIFO P&L")
    
    # Initialize and run enhanced paper trader
    try:
        trader = EnhancedPaperTrader(config, args.models_dir)
        await trader.run_trading_loop(args.interval)
    
    except Exception as e:
        logger.error(f"Enhanced paper trader failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())