"""
Data Pipeline Module
===================

Handles data loading, preprocessing, and feature engineering for the crypto trading bot.
"""

from .loader import DataLoader
from .features import FeatureEngine
from .preprocess import DataPreprocessor

__all__ = ['DataLoader', 'FeatureEngine', 'DataPreprocessor']