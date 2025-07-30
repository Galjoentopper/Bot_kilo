# Configuration Fix Plan for Trainer.py Error

## Problem Analysis

The trainer.py script fails with `KeyError: 'returns_periods'` because of a configuration structure mismatch:

- **FeatureEngine expects**: Flat configuration with keys like `returns_periods`, `sma_periods`, etc.
- **config.yaml provides**: Nested structure under `features` with subcategories
- **trainer.py passes**: `config.get('features', {})` which gives nested structure, not flat

## Root Cause

In `scripts/trainer.py:113`:
```python
feature_engine = FeatureEngine(config.get('features', {}))
```

This passes a nested config like:
```yaml
technical_indicators:
  sma_periods: [5, 10, 20, 50]
  # ...
price_features:
  returns_periods: [1, 5, 15]
  # ...
```

But `FeatureEngine` expects flat config like:
```python
{
  'sma_periods': [5, 10, 20, 50],
  'returns_periods': [1, 5, 15],
  # ...
}
```

## Solution Strategy

### 1. Create Configuration Utility Function

Create `src/utils/config.py` with:

```python
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
            from src.data_pipeline.features import FeatureEngine
            return FeatureEngine()._get_default_config()
        
        return flattened
    
    # If neither flat nor nested, return default config
    logger.warning("Feature config format not recognized, using defaults")
    from src.data_pipeline.features import FeatureEngine
    return FeatureEngine()._get_default_config()
```

### 2. Update src/utils/__init__.py

Add the new config utilities:

```python
"""
Utilities Module
================

Contains utility functions and helper classes.
"""

from .logger import setup_logging, TradingBotLogger
from .config import flatten_feature_config, validate_feature_config, prepare_feature_config

__all__ = [
    'setup_logging', 
    'TradingBotLogger',
    'flatten_feature_config',
    'validate_feature_config', 
    'prepare_feature_config'
]
```

### 3. Update scripts/trainer.py

Replace line 113:
```python
# OLD:
feature_engine = FeatureEngine(config.get('features', {}))

# NEW:
from src.utils.config import prepare_feature_config
feature_config = prepare_feature_config(config)
feature_engine = FeatureEngine(feature_config)
```

### 4. Enhance FeatureEngine Error Handling

Update `src/data_pipeline/features.py` to add better error handling:

```python
def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add price-based features."""
    # ... existing code ...
    
    # Returns for different periods - with error handling
    returns_periods = self.config.get('returns_periods', [1, 5, 15])
    if not returns_periods:
        logger.warning("No returns_periods configured, using default [1, 5, 15]")
        returns_periods = [1, 5, 15]
        
    for period in returns_periods:
        df[f'return_{period}'] = df['close'].pct_change(period)
        df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        df[f'high_return_{period}'] = df['high'].pct_change(period)
        df[f'low_return_{period}'] = df['low'].pct_change(period)
    
    return df
```

Apply similar error handling to all config access points in FeatureEngine.

## Implementation Steps

1. **Create config utility module** (`src/utils/config.py`)
2. **Update utils __init__.py** to export new functions
3. **Update trainer.py** to use the new config preparation function
4. **Enhance FeatureEngine** with better error handling for missing config keys
5. **Test the fix** by running `python scripts/trainer.py --model all`
6. **Add validation** to ensure all required config keys are present
7. **Document** the configuration requirements

## Expected Outcome

After implementing these changes:
- ✅ `trainer.py --model all` should run without KeyError
- ✅ FeatureEngine will work with both flat and nested config formats
- ✅ Better error messages for missing configuration
- ✅ Backward compatibility maintained
- ✅ Robust configuration handling for future changes

## Files to Modify

1. `src/utils/config.py` (new file)
2. `src/utils/__init__.py` (update imports)
3. `scripts/trainer.py` (fix config passing)
4. `src/data_pipeline/features.py` (add error handling)

## Testing Strategy

1. Run `python scripts/trainer.py --model all` to verify fix
2. Test with modified config.yaml to ensure flexibility
3. Test with missing config keys to verify error handling
4. Verify all model trainers still work correctly