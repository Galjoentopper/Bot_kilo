"""Compatibility wrapper for FeatureEngine.

The enhanced_trader script imports FeatureEngine from
`src.data_pipeline.feature_engine`, but the actual implementation
resides in `features.py`. This shim preserves backward compatibility
without modifying the large trading script.
"""
from .features import FeatureEngine  # re-export

__all__ = ["FeatureEngine"]
