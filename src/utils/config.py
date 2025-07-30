"""
Configuration Utilities
========================

Utilities for handling configuration structures and transformations.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def flatten_feature_config(nested_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested feature configuration for FeatureEngine compatibility.
    
    Args:
        nested_config: Nested configuration from config.yaml features section
        
    Returns:
        Flattened configuration dictionary
    """
    flattened = {}
    
    # Technical indicators
    tech_indicators = nested_config.get('technical_indicators', {})
    flattened.update({
        'sma_periods': tech_indicators.get('sma_periods', [5, 10, 20, 50]),
        'ema_periods': tech_indicators.get('ema_periods', [5, 10, 20, 50]),
        'rsi_period': tech_indicators.get('rsi_period', 14),
        'macd_fast': tech_indicators.get('macd_fast', 12),
        'macd_slow': tech_indicators.get('macd_slow', 26),
        'macd_signal': tech_indicators.get('macd_signal', 9),
        'bollinger_period': tech_indicators.get('bollinger_period', 20),
        'bollinger_std': tech_indicators.get('bollinger_std', 2),
        'atr_period': tech_indicators.get('atr_period', 14),
        'stoch_k_period': tech_indicators.get('stoch_k_period', 14),
        'stoch_d_period': tech_indicators.get('stoch_d_period', 3),
        'cci_period': tech_indicators.get('cci_period', 20),
    })
    
    # Price features
    price_features = nested_config.get('price_features', {})
    flattened.update({
        'returns_periods': price_features.get('returns_periods', [1, 5, 15]),
        'volatility_periods': price_features.get('volatility_periods', [10, 20, 50]),
    })
    
    # Time features
    time_features = nested_config.get('time_features', {})
    flattened.update({
        'include_hour': time_features.get('include_hour', True),
        'include_day_of_week': time_features.get('include_day_of_week', True),
        'include_month': time_features.get('include_month', True),
    })
    
    logger.debug(f"Flattened feature config: {list(flattened.keys())}")
    return flattened


def validate_feature_config(config: Dict[str, Any]) -> bool:
    """
    Validate that all required feature configuration keys are present.
    
    Args:
        config: Feature configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        'returns_periods', 'volatility_periods', 'sma_periods', 'ema_periods',
        'rsi_period', 'macd_fast', 'macd_slow', 'macd_signal',
        'bollinger_period', 'bollinger_std', 'atr_period',
        'stoch_k_period', 'stoch_d_period', 'cci_period',
        'include_hour', 'include_day_of_week', 'include_month'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        logger.error(f"Missing required feature config keys: {missing_keys}")
        return False
    
    return True


def prepare_feature_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare feature configuration for FeatureEngine, handling both flat and nested formats.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Properly formatted feature configuration
    """
    features_config = config.get('features', {})
    
    # Check if it's already flat (has returns_periods at top level)
    if 'returns_periods' in features_config:
        logger.debug("Feature config is already flat")
        return features_config
    
    # Check if it's nested (has technical_indicators, price_features, etc.)
    if any(key in features_config for key in ['technical_indicators', 'price_features', 'time_features']):
        logger.debug("Feature config is nested, flattening...")
        flattened = flatten_feature_config(features_config)
        
        if not validate_feature_config(flattened):
            logger.warning("Flattened config validation failed, using defaults")
            # Return default config if validation fails
            return get_default_feature_config()
        
        return flattened
    
    # If neither flat nor nested, return default config
    logger.warning("Feature config format not recognized, using defaults")
    return get_default_feature_config()


def get_default_feature_config() -> Dict[str, Any]:
    """
    Get default feature engineering configuration.
    
    Returns:
        Default feature configuration dictionary
    """
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