"""
Feature Engineering Module
==========================

Generates technical indicators and features for the crypto trading bot.
Optimized for GPU training with Paperspace Gradient.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Feature engineering class for generating technical indicators and market features.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FeatureEngine with configuration.
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config or self._get_default_config()
        logger.info("FeatureEngine initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default feature engineering configuration."""
        return {
            'sma_periods': [5, 10, 20, 50],
            'ema_periods': [5, 10, 20, 50],
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2,
            'atr_period': 14,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'cci_period': 20,
            'returns_periods': [1, 5, 15],
            'volatility_periods': [10, 20, 50],
            'include_hour': True,
            'include_day_of_week': True,
            'include_month': True
        }
    
    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the input DataFrame with robust validation.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with all engineered features
        """
        if df.empty:
            logger.warning("Empty DataFrame provided to feature engine")
            return df
        
        logger.info(f"Generating features for {len(df)} records")
        
        # Create a copy to avoid modifying original data
        features_df = df.copy()
        
        # CRITICAL: Validate and clean source data FIRST with absolute bounds
        features_df = self._validate_and_clean_source_data(features_df)
        
        # Apply ultra-aggressive initial clipping to prevent extreme calculations
        for col in ['open', 'high', 'low', 'close']:
            if col in features_df.columns:
                features_df[col] = np.clip(features_df[col], 0.01, 100000)  # $0.01 to $100k hard bounds
        if 'volume' in features_df.columns:
            features_df['volume'] = np.clip(features_df['volume'], 0, 1e9)  # Max 1B volume
        
        # Price-based features with intermediate validation
        features_df = self._add_price_features(features_df)
        features_df = self._apply_intermediate_validation(features_df, "after price features")
        
        # Technical indicators with intermediate validation
        features_df = self._add_technical_indicators(features_df)
        features_df = self._apply_intermediate_validation(features_df, "after technical indicators")
        
        # Volume features with intermediate validation
        features_df = self._add_volume_features(features_df)
        features_df = self._apply_intermediate_validation(features_df, "after volume features")
        
        # Volatility features with intermediate validation
        features_df = self._add_volatility_features(features_df)
        features_df = self._apply_intermediate_validation(features_df, "after volatility features")
        
        # Momentum features with intermediate validation
        features_df = self._add_momentum_features(features_df)
        features_df = self._apply_intermediate_validation(features_df, "after momentum features")
        
        # Time-based features (typically safe)
        features_df = self._add_time_features(features_df)
        
        # Custom features with intermediate validation
        features_df = self._add_custom_features(features_df)
        features_df = self._apply_intermediate_validation(features_df, "after custom features")
        
        # Advanced features for PPO compatibility (9 additional features)
        features_df = self._add_advanced_features(features_df)
        features_df = self._apply_intermediate_validation(features_df, "after advanced features")
        
        # CRITICAL: Apply final robust feature validation and cleaning
        features_df = self._apply_robust_feature_validation(features_df)
        
        # Log feature generation summary
        original_cols = len(df.columns)
        new_cols = len(features_df.columns)
        logger.info(f"Generated {new_cols - original_cols} new features")
        logger.info(f"Final data validation: NaN={features_df.isnull().sum().sum()}, Inf={np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()}")
        
        return features_df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Basic price features
        df['hl_avg'] = (df['high'] + df['low']) / 2
        df['hlc_avg'] = (df['high'] + df['low'] + df['close']) / 3
        df['ohlc_avg'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Price ranges
        df.loc[:, 'price_range'] = df['high'] - df['low']
        # Avoid division by zero
        df.loc[:, 'price_range_pct'] = df['price_range'] / (df['close'] + 1e-10)
        
        # Body and shadow features
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / (df['close'] + 1e-10)
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['upper_shadow_pct'] = df['upper_shadow'] / (df['close'] + 1e-10)
        df['lower_shadow_pct'] = df['lower_shadow'] / (df['close'] + 1e-10)
        
        # Returns for different periods - with error handling
        returns_periods = self.config.get('returns_periods', [1, 5, 15])
        if not returns_periods:
            logger.warning("No returns_periods configured, using default [1, 5, 15]")
            returns_periods = [1, 5, 15]
            
        for period in returns_periods:
            # Avoid division by zero in returns calculation
            df[f'return_{period}'] = df['close'].pct_change(period).fillna(0)
            # Log returns with safety check
            close_shifted = df['close'].shift(period)
            log_return = np.log((df['close'] + 1e-10) / (close_shifted + 1e-10))
            df[f'log_return_{period}'] = pd.Series(log_return, index=df.index).fillna(0)
            df[f'high_return_{period}'] = df['high'].pct_change(period).fillna(0)
            df[f'low_return_{period}'] = df['low'].pct_change(period).fillna(0)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        # Simple Moving Averages - with error handling
        sma_periods = self.config.get('sma_periods', [5, 10, 20, 50])
        for period in sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            # Fill NaN values in SMA
            df[f'sma_{period}'] = df[f'sma_{period}'].bfill().fillna(df['close'])
            # SMA ratio - avoid division by zero
            df[f'sma_{period}_ratio'] = df['close'] / (df[f'sma_{period}'] + 1e-10)
        
        # Exponential Moving Averages - with error handling
        ema_periods = self.config.get('ema_periods', [5, 10, 20, 50])
        for period in ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            # Fill NaN values in EMA
            df[f'ema_{period}'] = df[f'ema_{period}'].bfill().fillna(df['close'])
            # EMA ratio - avoid division by zero
            df[f'ema_{period}_ratio'] = df['close'] / (df[f'ema_{period}'] + 1e-10)
        
        # RSI - with error handling
        rsi_period = self.config.get('rsi_period', 14)
        df['rsi'] = self._calculate_rsi(df['close'], rsi_period)
        
        # MACD - with error handling
        macd_fast = self.config.get('macd_fast', 12)
        macd_slow = self.config.get('macd_slow', 26)
        macd_signal = self.config.get('macd_signal', 9)
        macd_line, macd_signal_line, macd_histogram = self._calculate_macd(
            df['close'], macd_fast, macd_slow, macd_signal
        )
        df['macd'] = macd_line
        df['macd_signal'] = macd_signal_line
        df['macd_histogram'] = macd_histogram
        
        # Bollinger Bands - with error handling
        bollinger_period = self.config.get('bollinger_period', 20)
        bollinger_std = self.config.get('bollinger_std', 2)
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            df['close'], bollinger_period, bollinger_std
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        # Avoid division by zero in Bollinger Band calculations
        df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        # ATR (Average True Range) - with error handling
        atr_period = self.config.get('atr_period', 14)
        df['atr'] = self._calculate_atr(df, atr_period)
        # Avoid division by zero in ATR percentage
        df['atr_pct'] = df['atr'] / (df['close'] + 1e-10)
        
        # Stochastic Oscillator - with error handling
        stoch_k_period = self.config.get('stoch_k_period', 14)
        stoch_d_period = self.config.get('stoch_d_period', 3)
        stoch_k, stoch_d = self._calculate_stochastic(
            df, stoch_k_period, stoch_d_period
        )
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # CCI (Commodity Channel Index) - with error handling
        cci_period = self.config.get('cci_period', 20)
        df['cci'] = self._calculate_cci(df, cci_period)
        
        # ADX (Average Directional Index) - additional feature for 114 total
        adx_period = self.config.get('adx_period', 14)
        df['adx'] = self._calculate_adx(df, adx_period)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving averages
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        # Fill NaN values in volume SMAs
        df['volume_sma_10'] = df['volume_sma_10'].bfill().fillna(0)
        df['volume_sma_20'] = df['volume_sma_20'].bfill().fillna(0)
        # Volume ratio - avoid division by zero
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)
        df['volume_ratio'] = df['volume_ratio'].bfill().fillna(1)
        
        # Volume-price features
        # Avoid division by zero in VWAP calculation
        volume_cumsum = df['volume'].cumsum()
        vwap_numerator = (df['volume'] * df['hlc_avg']).cumsum()
        df['vwap'] = vwap_numerator / (volume_cumsum + 1e-10)
        df['vwap'] = df['vwap'].bfill().fillna(df['hlc_avg'])
        df['volume_price_trend'] = df['volume'] * (df['close'] - df['open'])
        
        # On-Balance Volume (OBV)
        df['obv'] = self._calculate_obv(df)
        
        # Volume Rate of Change - avoid division by zero
        df['volume_roc_5'] = df['volume'].pct_change(5).fillna(0)
        df['volume_roc_10'] = df['volume'].pct_change(10).fillna(0)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        volatility_periods = self.config.get('volatility_periods', [10, 20, 50])
        for period in volatility_periods:
            # Rolling standard deviation
            df[f'volatility_{period}'] = df['return_1'].rolling(window=period).std()
            # Fill NaN values in volatility
            df[f'volatility_{period}'] = df[f'volatility_{period}'].fillna(0)
            
            # Historical volatility (annualized)
            df[f'hist_vol_{period}'] = df[f'volatility_{period}'] * np.sqrt(365 * 24 * 4)  # 15-min intervals
            
            # Parkinson volatility (high-low based)
            # Avoid division by zero in log calculation
            high_low_ratio = df['high'] / (df['low'] + 1e-10)
            log_hl_ratio = pd.Series(np.log(high_low_ratio), index=df.index)
            parkinson_vol = np.sqrt(
                (log_hl_ratio ** 2).rolling(window=period).mean() / (4 * np.log(2))
            )
            df[f'parkinson_vol_{period}'] = parkinson_vol.fillna(0)
        
        # Volatility ratios - handle division by zero
        df['vol_ratio_10_20'] = df['volatility_10'] / (df['volatility_20'] + 1e-10)
        df['vol_ratio_20_50'] = df['volatility_20'] / (df['volatility_50'] + 1e-10)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        # Rate of Change
        for period in [5, 10, 20]:
            # Avoid division by zero
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / (df['close'].shift(period) + 1e-10)) * 100
            # Fill NaN values
            df[f'roc_{period}'] = df[f'roc_{period}'].bfill().fillna(0)
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            # Fill NaN values
            df[f'momentum_{period}'] = df[f'momentum_{period}'].bfill().fillna(0)
        
        # Williams %R
        for period in [14, 21]:
            df[f'williams_r_{period}'] = self._calculate_williams_r(df, period)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if df.empty:
            logger.warning("Empty DataFrame provided for time features")
            return df
        include_hour = self.config.get('include_hour', True)
        include_day_of_week = self.config.get('include_day_of_week', True)
        include_month = self.config.get('include_month', True)
        
        # Check if index is a DatetimeIndex before accessing datetime attributes
        if include_hour and isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if include_day_of_week and isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        if include_month and isinstance(df.index, pd.DatetimeIndex):
            df['month'] = df.index.month
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Market session features
        if isinstance(df.index, pd.DatetimeIndex):
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            # Handle case where hour might be NaN
            hour_values = df.index.hour if hasattr(df.index, 'hour') else np.zeros(len(df))
            df['is_night'] = ((hour_values >= 22) | (hour_values <= 6)).astype(int)
        
        return df
    
    def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom engineered features."""
        # Price relative to moving averages
        if 'sma_20' in df.columns:
            df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        if 'ema_20' in df.columns:
            df['price_above_ema20'] = (df['close'] > df['ema_20']).astype(int)
        
        # Moving average crosses
        if 'sma_5' in df.columns and 'sma_20' in df.columns:
            df['sma_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
        if 'ema_5' in df.columns and 'ema_20' in df.columns:
            df['ema_cross'] = (df['ema_5'] > df['ema_20']).astype(int)
        
        # RSI signals
        if 'rsi' in df.columns:
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD signals
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Bollinger Band signals
        if 'bb_position' in df.columns:
            # Avoid division by zero in Bollinger Band squeeze calculation
            bb_width_rolling = df['bb_width'].rolling(20).mean()
            df['bb_squeeze'] = (df['bb_width'] < (bb_width_rolling + 1e-10)).astype(int)
            df['bb_breakout_upper'] = (df['close'] > df['bb_upper']).astype(int)
            df['bb_breakout_lower'] = (df['close'] < df['bb_lower']).astype(int)
        
        # Volume confirmation
        if 'volume_ratio' in df.columns:
            df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
            df['low_volume'] = (df['volume_ratio'] < 0.5).astype(int)
        
        return df
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 9 advanced technical indicators for PPO compatibility."""
        
        # 1. Ichimoku Cloud Components (3 features)
        df['ichimoku_tenkan'] = self._calculate_ichimoku_tenkan(df)
        df['ichimoku_kijun'] = self._calculate_ichimoku_kijun(df)
        df['ichimoku_senkou_a'] = self._calculate_ichimoku_senkou_a(df)
        
        # 2. Advanced Volume Indicators (2 features)
        df['vwap_deviation'] = self._calculate_vwap_deviation(df)
        df['accumulation_distribution'] = self._calculate_accumulation_distribution(df)
        
        # 3. Market Microstructure (2 features)
        df['spread_proxy'] = self._calculate_spread_proxy(df)
        df['price_impact'] = self._calculate_price_impact(df)
        
        # 4. Regime Detection (2 features)
        df['trend_strength_index'] = self._calculate_trend_strength_index(df)
        df['market_regime'] = self._calculate_market_regime(df)
        
        return df
    
    # Technical indicator calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with improved error handling."""
        delta = prices.diff()
        # Ensure delta is numeric before comparison
        delta = pd.to_numeric(delta, errors='coerce')
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)  # Add small epsilon to prevent division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Ensure RSI is within valid range [0, 100]
        rsi = pd.Series(np.clip(rsi, 0, 100), index=rsi.index)
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        # Fill initial NaN values
        macd_line = macd_line.bfill().fillna(0)
        macd_signal = macd_signal.bfill().fillna(0)
        macd_histogram = macd_histogram.bfill().fillna(0)
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        # Handle case where std is NaN or 0
        std = std.fillna(0)
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.Series(np.maximum(high_low, np.maximum(high_close, low_close)), index=df.index)
        atr = true_range.rolling(window=period).mean()
        # Fill initial NaN values
        atr = atr.bfill().fillna(0)
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        # Avoid division by zero
        denominator = highest_high - lowest_low
        k_percent = 100 * ((df['close'] - lowest_low) / (denominator + 1e-10))
        # Fill initial NaN values
        k_percent = k_percent.bfill().fillna(50)  # Fill with neutral value
        d_percent = k_percent.rolling(window=d_period).mean()
        # Fill initial NaN values for D line
        d_percent = d_percent.bfill().fillna(50)  # Fill with neutral value
        return k_percent, d_percent
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index with improved error handling."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        # Avoid division by zero
        cci = (typical_price - sma) / (0.015 * mean_deviation + 1e-10)
        
        # Clip extreme values to prevent numerical instability
        cci = pd.Series(np.clip(cci, -500, 500), index=cci.index)
        
        # Fill initial NaN values
        cci = cci.bfill().fillna(0)
        
        return cci
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        if len(df) > 0:
            obv.iloc[0] = df['volume'].iloc[0] if len(df) > 0 else 0
        
        for i in range(1, len(df)):
            if len(df) > i and len(df['close']) > i and len(df['volume']) > i:
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            else:
                obv.iloc[i] = obv.iloc[i-1] if i > 0 else 0
        
        # Fill any remaining NaN values
        obv = obv.bfill().fillna(0)
        return obv
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R with improved error handling."""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        williams_r = -100 * ((highest_high - df['close']) / (denominator + 1e-10))
        
        # Ensure Williams %R is within valid range [-100, 0]
        williams_r = pd.Series(np.clip(williams_r, -100, 0), index=williams_r.index)
        
        # Fill initial NaN values
        williams_r = williams_r.bfill().fillna(-50)  # Fill with neutral value
        
        return williams_r
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX) with improved error handling."""
        # Calculate True Range (already used in ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.Series(np.maximum(high_low, np.maximum(high_close, low_close)), index=df.index)
        
        # Calculate Directional Movement
        high_diff = df['high'] - df['high'].shift()
        low_diff = df['low'].shift() - df['low']
        
        # Positive and Negative Directional Movement
        plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=df.index)
        minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=df.index)
        
        # Smooth the values using Wilder's smoothing (similar to EMA with alpha = 1/period)
        alpha = 1.0 / period
        
        # Smoothed True Range
        atr_smooth = true_range.ewm(alpha=alpha, adjust=False).mean()
        
        # Smoothed Directional Movement
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
        
        # Directional Indicators
        plus_di = 100 * (plus_dm_smooth / (atr_smooth + 1e-10))
        minus_di = 100 * (minus_dm_smooth / (atr_smooth + 1e-10))
        
        # Directional Index
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        
        # Average Directional Index
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        
        # Ensure ADX is within valid range [0, 100] and handle NaN values
        adx = pd.Series(np.clip(adx, 0, 100), index=adx.index)
        adx = adx.bfill().fillna(25)  # Fill with neutral value (25 is typical neutral ADX)
        
        return adx
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of all feature column names."""
        # Original OHLCV columns
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 
                        'trades', 'taker_buy_base', 'taker_buy_quote']
        
        # Return only the engineered features
        feature_cols = [col for col in df.columns if col not in original_cols]
        return feature_cols
    
    def prepare_model_input(self, df: pd.DataFrame, sequence_length: int = 20) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data for model input with proper scaling and windowing.
        
        Args:
            df: DataFrame with features
            sequence_length: Length of sequences for GRU model
            
        Returns:
            Tuple of (feature_array, feature_names)
        """
        # Get feature columns (exclude original OHLCV)
        feature_cols = self.get_feature_names(df)
        
        # Select only numeric features and drop NaN
        feature_df = df[feature_cols].select_dtypes(include=[np.number])
        feature_df = feature_df.dropna()
        
        if feature_df.empty:
            logger.warning("No valid features after preprocessing")
            return np.array([]), []
        
        # Convert to numpy array
        feature_array = feature_df.values
        
        logger.info(f"Prepared {feature_array.shape[1]} features for {feature_array.shape[0]} samples")
        
        return feature_array, list(feature_df.columns)
    
    def validate_feature_consistency(self, df: pd.DataFrame, expected_count: int = 105) -> pd.DataFrame:
        """
        Ensure consistent feature count for model compatibility.
        
        Args:
            df: DataFrame with features
            expected_count: Expected number of features
            
        Returns:
            DataFrame with consistent feature count
        """
        feature_cols = self.get_feature_names(df)
        current_count = len(feature_cols)
        
        if current_count != expected_count:
            logger.warning(f"Feature count mismatch: {current_count} vs expected {expected_count}")
            
            if current_count < expected_count:
                # Add zero-filled dummy features
                for i in range(current_count, expected_count):
                    df[f'dummy_feature_{i}'] = 0.0
                logger.info(f"Added {expected_count - current_count} dummy features")
            elif current_count > expected_count:
                # Remove excess features (keep most important ones based on name patterns)
                excess_features = feature_cols[expected_count:]
                df = df.drop(columns=excess_features)
                logger.info(f"Removed {current_count - expected_count} excess features")
        
        return df
    
    def pad_features_for_model(self, df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """
        Pad or trim features for specific model compatibility.
        
        Args:
            df: DataFrame with features
            model_type: Type of model ('lightgbm', 'gru', 'ppo')
            
        Returns:
            DataFrame with model-compatible feature count
        """
        expected_counts = {
            'lightgbm': 114,  # Updated to match new standard
            'gru': 114,       # Updated to match GRU model expectations
            'ppo': 113        # PPO was trained with 113 market features (before ADX was added)
        }
        
        expected = expected_counts.get(model_type, len(self.get_feature_names(df)))
        
        # Special handling for PPO: exclude ADX feature to maintain compatibility
        if model_type == 'ppo':
            feature_names = self.get_feature_names(df)
            if 'adx' in feature_names and len(feature_names) == 114:
                # Remove ADX feature for PPO compatibility
                df_ppo = df.drop(columns=['adx'])
                return self.validate_feature_consistency(df_ppo, 113)
        
        return self.validate_feature_consistency(df, expected)
    
    # Advanced technical indicator calculation methods
    def _calculate_ichimoku_tenkan(self, df: pd.DataFrame, period: int = 9) -> pd.Series:
        """Calculate Ichimoku Tenkan-sen (Conversion Line)."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        tenkan = (high_max + low_min) / 2
        return tenkan.bfill().fillna(df['close'])
    
    def _calculate_ichimoku_kijun(self, df: pd.DataFrame, period: int = 26) -> pd.Series:
        """Calculate Ichimoku Kijun-sen (Base Line)."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        kijun = (high_max + low_min) / 2
        return kijun.bfill().fillna(df['close'])
    
    def _calculate_ichimoku_senkou_a(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Ichimoku Senkou Span A (Leading Span A)."""
        tenkan = self._calculate_ichimoku_tenkan(df)
        kijun = self._calculate_ichimoku_kijun(df)
        senkou_a = (tenkan + kijun) / 2
        return senkou_a.bfill().fillna(df['close'])
    
    def _calculate_vwap_deviation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP deviation as percentage."""
        if 'vwap' not in df.columns:
            # Calculate VWAP if not already present
            volume_cumsum = df['volume'].cumsum()
            vwap_numerator = (df['volume'] * df['hlc_avg']).cumsum()
            vwap = vwap_numerator / (volume_cumsum + 1e-10)
        else:
            vwap = df['vwap']
        
        # Calculate deviation as percentage
        vwap_dev = (df['close'] - vwap) / (vwap + 1e-10) * 100
        return vwap_dev.bfill().fillna(0)
    
    def _calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        # Money Flow Multiplier
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        # Money Flow Volume
        mfv = clv * df['volume']
        # Accumulation/Distribution Line (cumulative)
        adl = mfv.cumsum()
        return adl.bfill().fillna(0)
    
    def _calculate_spread_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate spread proxy as normalized high-low range."""
        mid_price = (df['high'] + df['low']) / 2
        spread_proxy = (df['high'] - df['low']) / (mid_price + 1e-10) * 100
        return spread_proxy.bfill().fillna(0)
    
    def _calculate_price_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price impact indicator."""
        price_change = df['close'].pct_change().abs()
        volume_normalized = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        price_impact = price_change * volume_normalized
        return price_impact.bfill().fillna(0)
    
    def _calculate_trend_strength_index(self, df: pd.DataFrame, period: int = 25) -> pd.Series:
        """Calculate Trend Strength Index."""
        # Calculate momentum
        momentum = df['close'] - df['close'].shift(period)
        # Calculate absolute momentum sum
        abs_momentum_sum = momentum.abs().rolling(window=period).sum()
        # Calculate directional momentum sum
        positive_momentum = momentum.where(momentum > 0, 0).rolling(window=period).sum()
        negative_momentum = momentum.where(momentum < 0, 0).abs().rolling(window=period).sum()
        
        # TSI calculation
        tsi = 100 * (positive_momentum - negative_momentum) / (abs_momentum_sum + 1e-10)
        return tsi.bfill().fillna(0)
    
    def _calculate_market_regime(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate market regime indicator (trending vs ranging)."""
        # Calculate price efficiency ratio
        price_change = abs(df['close'] - df['close'].shift(period))
        volatility_sum = df['close'].diff().abs().rolling(window=period).sum()
        efficiency_ratio = price_change / (volatility_sum + 1e-10)
        
        # Smooth the efficiency ratio
        regime = efficiency_ratio.ewm(span=10).mean()
        
        # Normalize to 0-100 scale (higher values = trending, lower = ranging)
        regime_normalized = regime * 100
        return regime_normalized.bfill().fillna(50)  # Fill with neutral value
    
    def _validate_and_clean_source_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean source OHLCV data before feature engineering.
        This prevents extreme values from propagating through the feature pipeline.
        """
        logger.info("Validating and cleaning source data...")
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        
        # Create a copy to avoid modifying original
        clean_df = df.copy()
        
        # Check for infinite and NaN values FIRST
        for col in ['open', 'high', 'low', 'close']:
            if col in clean_df.columns:
                # Replace infinite values with NaN
                clean_df[col] = clean_df[col].replace([np.inf, -np.inf], np.nan)
                
                # Count problematic values
                nan_count = clean_df[col].isnull().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN/Inf values in {col}, forward-filling")
                    # Forward fill, then backward fill, then use median
                    clean_df[col] = clean_df[col].ffill().bfill()
                    remaining_nan = clean_df[col].isnull().sum()
                    if remaining_nan > 0:
                        col_median = clean_df[col].median()
                        if pd.isna(col_median):
                            col_median = 50000.0  # Fallback reasonable crypto price
                        clean_df[col] = clean_df[col].fillna(col_median)
                        logger.warning(f"Filled {remaining_nan} remaining NaN values in {col} with median {col_median:.2f}")

        # Now check for extreme values AFTER handling NaN/Inf
        for col in ['open', 'high', 'low', 'close']:
            if col in clean_df.columns:
                col_data = clean_df[col].dropna()
                if len(col_data) == 0:
                    continue
                    
                col_max = col_data.max()
                col_min = col_data.min()
                
                # Use robust percentile-based bounds instead of median (which can be corrupted by outliers)
                try:
                    # Use 25th and 75th percentiles for robust statistics
                    q25 = col_data.quantile(0.25)
                    q75 = col_data.quantile(0.75)
                    iqr = q75 - q25
                    
                    # If IQR is too small or corrupted, fall back to absolute bounds
                    if iqr < 1 or pd.isna(iqr):
                        # Absolute bounds for financial data - very conservative
                        upper_bound = 200000  # $200k absolute max for crypto
                        lower_bound = 0.01    # $0.01 absolute min
                        logger.warning(f"Using absolute bounds for {col} due to corrupted statistics")
                    else:
                        # Robust outlier detection using IQR method
                        robust_upper = q75 + 3 * iqr  # 3x IQR above Q3
                        robust_lower = q25 - 3 * iqr  # 3x IQR below Q1
                        
                        # Cap at reasonable financial bounds
                        upper_bound = min(robust_upper, 200000)  # Max $200k
                        lower_bound = max(robust_lower, 0.01)    # Min $0.01
                        
                except Exception as e:
                    logger.warning(f"Percentile calculation failed for {col}: {e}, using absolute bounds")
                    upper_bound = 200000  # $200k absolute max
                    lower_bound = 0.01    # $0.01 absolute min
                
                # Check if cleaning is needed using absolute thresholds
                needs_cleaning = (col_max > upper_bound or col_min < lower_bound or
                                col_max > 500000 or col_min < 0)  # Emergency absolute bounds
                
                if needs_cleaning:
                    logger.error(f"CRITICAL: Extreme values in {col} - min: {col_min:.2f}, max: {col_max:.2f}")
                    
                    before_count = len(clean_df)
                    extreme_mask = (clean_df[col] > upper_bound) | (clean_df[col] < lower_bound)
                    extreme_count = extreme_mask.sum()
                    
                    if extreme_count > 0:
                        logger.warning(f"Aggressively clipping {extreme_count}/{before_count} extreme values in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")
                        clean_df[col] = np.clip(clean_df[col], lower_bound, upper_bound)
        
        # Check for volume anomalies
        if 'volume' in clean_df.columns:
            vol_median = clean_df['volume'].median()
            vol_max = clean_df['volume'].max()
            
            # Flag volume spikes > 100x median
            if vol_max > vol_median * 100:
                logger.warning(f"Large volume spike detected - max: {vol_max:.2f}, median: {vol_median:.2f}")
                # Cap volume at 50x median
                clean_df['volume'] = np.clip(clean_df['volume'], 0, vol_median * 50)
        
        # Check for NaN/Inf values in source data
        nan_count = clean_df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in source data, forward-filling")
            clean_df = clean_df.ffill().bfill()
        
        # Check for infinite values
        inf_count = np.isinf(clean_df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values in source data, replacing")
            clean_df = clean_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        # Validate OHLC relationships
        if all(col in clean_df.columns for col in ['open', 'high', 'low', 'close']):
            # Check if high >= max(open, close) and low <= min(open, close)
            high_valid = clean_df['high'] >= np.maximum(clean_df['open'], clean_df['close'])
            low_valid = clean_df['low'] <= np.minimum(clean_df['open'], clean_df['close'])
            
            invalid_high = (~high_valid).sum()
            invalid_low = (~low_valid).sum()
            
            if invalid_high > 0 or invalid_low > 0:
                logger.warning(f"Invalid OHLC relationships: {invalid_high} high violations, {invalid_low} low violations")
                
                # Fix invalid relationships
                clean_df['high'] = np.maximum(clean_df['high'], np.maximum(clean_df['open'], clean_df['close']))
                clean_df['low'] = np.minimum(clean_df['low'], np.minimum(clean_df['open'], clean_df['close']))
        
        logger.info("Source data validation and cleaning completed")
        return clean_df
    
    def _apply_robust_feature_validation(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply robust validation and cleaning to generated features.
        This prevents extreme feature values from causing gradient explosions.
        """
        logger.info("Applying robust feature validation...")
        
        # Replace infinite values with NaN first
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Count and log initial issues
        nan_count_before = features_df.isnull().sum().sum()
        if nan_count_before > 0:
            logger.info(f"Found {nan_count_before} NaN/Inf values in generated features")
        
        # Identify and handle features with extreme values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        extreme_features = []
        
        for col in numeric_cols:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue  # Skip original OHLCV columns
                
            col_data = features_df[col].dropna()
            if len(col_data) == 0:
                continue
                
            col_abs_max = np.abs(col_data).max()
            col_std = col_data.std()
            col_median = col_data.median()
            
            # Financial-data-aware feature validation for ML stability
            if col_abs_max > 500000 or (col_std > 0 and col_abs_max > abs(col_median) + 50 * col_std):
                extreme_features.append((col, col_abs_max))
                logger.warning(f"Extreme values in feature '{col}': max_abs={col_abs_max:.2f}, std={col_std:.6f}")
                
                # Financial-appropriate clipping for ML stability
                if col_abs_max > 1000000:
                    # Extremely large values - clip to large but reasonable range
                    features_df[col] = np.clip(features_df[col], -500000, 500000)
                    logger.warning(f"Applied extreme clipping [-500k, 500k] to feature '{col}' (was {col_abs_max:.2f})")
                elif col_abs_max > 200000:
                    # Large values - clip to moderate range
                    features_df[col] = np.clip(features_df[col], -150000, 150000)
                    logger.warning(f"Applied large clipping [-150k, 150k] to feature '{col}' (was {col_abs_max:.2f})")
                elif col_abs_max > 100000:
                    # Moderate values - clip conservatively (BTC prices are often >100k)
                    features_df[col] = np.clip(features_df[col], -120000, 120000)
                    logger.warning(f"Applied moderate clipping [-120k, 120k] to feature '{col}' (was {col_abs_max:.2f})")
        
        # Handle NaN values with improved strategy
        for col in numeric_cols:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue  # Skip original OHLCV columns
                
            nan_ratio = features_df[col].isnull().sum() / len(features_df)
            
            if nan_ratio > 0.8:
                # Feature is mostly NaN - fill with zeros
                features_df[col] = features_df[col].fillna(0)
                logger.debug(f"Feature '{col}' was {nan_ratio:.1%} NaN, filled with zeros")
            elif nan_ratio > 0.5:
                # Feature is significantly NaN - fill with median
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val if pd.notna(median_val) else 0)
                logger.debug(f"Feature '{col}' was {nan_ratio:.1%} NaN, filled with median")
        
        # Final NaN cleanup - forward fill, backward fill, then zero fill
        features_df = features_df.ffill().bfill().fillna(0)
        
        # Final validation - ensure no NaN or infinite values remain
        nan_count_after = features_df.isnull().sum().sum()
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        
        if nan_count_after > 0:
            logger.error(f"Still have {nan_count_after} NaN values after cleaning, force-filling with 0")
            features_df = features_df.fillna(0)
        
        if inf_count > 0:
            logger.error(f"Still have {inf_count} infinite values after cleaning, replacing with large finite values")
            features_df = features_df.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Final safety check - ensure ALL values are finite
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Log validation summary
        if extreme_features:
            logger.warning(f"Cleaned {len(extreme_features)} features with extreme values")
        
        final_max = features_df.select_dtypes(include=[np.number]).abs().max().max()
        logger.info(f"Feature validation completed - max absolute value: {final_max:.6f}")
        
        return features_df
    
    def _apply_intermediate_validation(self, features_df: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Apply intermediate validation during feature generation to prevent extreme value propagation.
        Uses financial-data-appropriate bounds instead of overly aggressive clipping.
        """
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        clipped_features = []
        
        for col in numeric_cols:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                continue  # Skip original OHLCV columns
                
            col_data = features_df[col].dropna()
            if len(col_data) == 0:
                continue
                
            col_abs_max = np.abs(col_data).max()
            
            # Financial-data-appropriate clipping bounds
            if col_abs_max > 1000000:  # 1M+ (extreme outliers)
                features_df[col] = np.clip(features_df[col], -500000, 500000)
                clipped_features.append((col, col_abs_max))
                logger.warning(f"Intermediate clipping [-500k, 500k] on '{col}' {stage} (was {col_abs_max:.2f})")
            elif col_abs_max > 200000:  # 200k+ (very high values)
                features_df[col] = np.clip(features_df[col], -150000, 150000)
                clipped_features.append((col, col_abs_max))
                logger.warning(f"Intermediate clipping [-150k, 150k] on '{col}' {stage} (was {col_abs_max:.2f})")
            elif col_abs_max > 50000:  # 50k+ (moderately high values - still reasonable for BTC)
                # Only clip if it's truly excessive (>200k) - 50k-200k is normal for BTC
                if col_abs_max > 200000:
                    features_df[col] = np.clip(features_df[col], -150000, 150000)
                    clipped_features.append((col, col_abs_max))
                    logger.debug(f"Intermediate clipping [-150k, 150k] on '{col}' {stage} (was {col_abs_max:.2f})")
        
        # Replace any remaining NaN/Inf values immediately
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if clipped_features:
            logger.info(f"Applied intermediate validation {stage}: clipped {len(clipped_features)} features")
        
        return features_df