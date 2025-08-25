"""Compatibility wrapper for DataPreprocessor.

Enhanced trader expects `src.data_pipeline.data_preprocessor` but the
implementation lives in `preprocess.py`. This shim re-exports the class.
"""
from .preprocess import DataPreprocessor  # re-export

__all__ = ["DataPreprocessor"]
