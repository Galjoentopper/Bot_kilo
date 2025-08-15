"""
Time-Series Cross-Validation Module
===================================

Implements purged time-series cross-validation with embargo periods
to prevent data leakage in financial time series.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, List, Union, Dict, Any
from sklearn.model_selection import BaseCrossValidator
import logging

logger = logging.getLogger(__name__)


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Time series cross-validator with purging and embargo.
    
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus
    shuffling in cross validator is inappropriate.
    
    This cross-validator additionally purges training data to avoid
    leakage and adds embargo periods after test sets.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        embargo: int = 0
    ):
        """
        Initialize PurgedTimeSeriesSplit.
        
        Args:
            n_splits: Number of splits
            max_train_size: Maximum size for a single training set
            test_size: Fixed test set size (if None, grows with each split)
            gap: Gap between training and test set (purge period)
            embargo: Embargo period after test set
        """
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap
        self.embargo = embargo
        
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data
            y: Target variable (not used, for compatibility)
            groups: Group labels (not used, for compatibility)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        n_folds = self.n_splits + 1
        
        if self.test_size:
            # Fixed test size
            test_size = self.test_size
            if test_size >= n_samples:
                raise ValueError(
                    f"test_size={test_size} must be smaller than n_samples={n_samples}"
                )
        else:
            # Growing test size
            test_size = n_samples // n_folds
        
        indices = np.arange(n_samples)
        
        # Calculate margins
        margin = 0
        if self.gap > 0:
            margin += self.gap
        if self.embargo > 0:
            margin += self.embargo
            
        for i in range(self.n_splits):
            # Calculate test start position
            if self.test_size:
                # Fixed test size: evenly space the test sets
                test_start = n_samples - (self.n_splits - i) * (test_size + margin)
            else:
                # Growing test size
                test_start = (i + 1) * test_size
            
            test_end = test_start + test_size
            
            # Ensure we don't exceed bounds
            if test_end + self.embargo > n_samples:
                test_end = n_samples - self.embargo
                test_start = test_end - test_size
            
            if test_start < 0:
                continue
                
            # Apply purge (gap before test set)
            train_end = test_start - self.gap if self.gap > 0 else test_start
            
            # Get train indices
            train_indices = indices[:train_end]
            
            # Apply max_train_size if specified
            if self.max_train_size and len(train_indices) > self.max_train_size:
                train_indices = train_indices[-self.max_train_size:]
            
            # Get test indices
            test_indices = indices[test_start:test_end]
            
            # Skip if not enough training data
            if len(train_indices) == 0 or len(test_indices) == 0:
                continue
                
            yield train_indices, test_indices
            
            logger.debug(
                f"Fold {i+1}/{self.n_splits}: "
                f"Train [{train_indices[0]}, {train_indices[-1]}] "
                f"Test [{test_indices[0]}, {test_indices[-1]}]"
            )
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


class BlockingTimeSeriesSplit(BaseCrossValidator):
    """
    Time series cross-validator using blocking strategy.
    
    Instead of expanding window, uses fixed-size blocks for both
    training and testing, moving forward in time.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        embargo: int = 0
    ):
        """
        Initialize BlockingTimeSeriesSplit.
        
        Args:
            n_splits: Number of splits
            train_size: Fixed training set size
            test_size: Fixed test set size
            gap: Gap between training and test set
            embargo: Embargo period after test set
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.embargo = embargo
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Training data
            y: Target variable (not used)
            groups: Group labels (not used)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate sizes
        if self.train_size is None:
            train_size = n_samples // (self.n_splits + 1)
        else:
            train_size = self.train_size
            
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        # Calculate total size needed for one fold
        fold_size = train_size + self.gap + test_size + self.embargo
        
        # Calculate step size
        total_size = train_size + self.gap + test_size
        if self.n_splits > 1:
            step = (n_samples - total_size - self.embargo) // (self.n_splits - 1)
        else:
            step = 0
        
        for i in range(self.n_splits):
            # Calculate positions
            train_start = i * step
            train_end = train_start + train_size
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            # Check bounds
            if test_end + self.embargo > n_samples:
                break
                
            # Get indices
            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
            
            logger.debug(
                f"Block {i+1}/{self.n_splits}: "
                f"Train [{train_start}, {train_end}] "
                f"Test [{test_start}, {test_end}]"
            )
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


def create_time_series_splits(
    n_samples: int,
    n_splits: int = 5,
    test_ratio: float = 0.2,
    gap_ratio: float = 0.02,
    embargo_ratio: float = 0.02,
    method: str = "purged"
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series splits with automatic parameter calculation.
    
    Args:
        n_samples: Total number of samples
        n_splits: Number of splits
        test_ratio: Ratio of test set size to total samples
        gap_ratio: Ratio of gap size to total samples
        embargo_ratio: Ratio of embargo size to total samples
        method: Split method ('purged' or 'blocking')
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Calculate sizes
    test_size = int(n_samples * test_ratio)
    gap = int(n_samples * gap_ratio)
    embargo = int(n_samples * embargo_ratio)
    
    # Create splitter
    if method == "purged":
        splitter = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            embargo=embargo
        )
    elif method == "blocking":
        train_size = int(n_samples * (1 - test_ratio) / n_splits)
        splitter = BlockingTimeSeriesSplit(
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size,
            gap=gap,
            embargo=embargo
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Generate splits
    X_dummy = np.zeros((n_samples, 1))
    splits = list(splitter.split(X_dummy))
    
    logger.info(
        f"Created {len(splits)} time series splits using {method} method "
        f"(test_size={test_size}, gap={gap}, embargo={embargo})"
    )
    
    return splits


def plot_cv_indices(
    cv: BaseCrossValidator,
    X: Union[np.ndarray, pd.DataFrame],
    group: Optional[Union[np.ndarray, pd.Series]] = None,
    ax=None,
    n_splits: Optional[int] = None,
    lw: int = 10
):
    """
    Create a visualization of the cross-validation behavior.
    
    Args:
        cv: Cross-validator object
        X: Training data
        group: Group labels for the samples
        ax: Matplotlib axis (created if None)
        n_splits: Number of splits to plot
        lw: Line width for plotting
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Generate the training/test splits
    n_samples = len(X)
    splits = list(cv.split(X, groups=group))
    n_splits = n_splits or len(splits)
    
    # Create color maps
    cmap_cv = plt.cm.coolwarm
    cmap_data = plt.cm.Paired
    
    # Plot the splits
    for ii, (train_indices, test_indices) in enumerate(splits[:n_splits]):
        # Fill in indices
        indices = np.zeros(n_samples)
        indices[train_indices] = 1
        indices[test_indices] = 2
        
        # Visualize the results
        ax.scatter(
            range(n_samples),
            [ii + 0.5] * n_samples,
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=0,
            vmax=2,
        )
    
    # Plot formatting
    ax.set_ylim(0, n_splits + 1)
    ax.set_xlim(0, n_samples)
    ax.set_ylabel("CV iteration", fontsize=12)
    ax.set_xlabel("Sample index", fontsize=12)
    ax.set_title("Time Series Cross-Validation", fontsize=14)
    
    # Create legend
    train_patch = Patch(color=cmap_cv(0.2), label='Training set')
    test_patch = Patch(color=cmap_cv(0.8), label='Test set')
    ax.legend(handles=[train_patch, test_patch], loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    return ax


def validate_time_series_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    cv: BaseCrossValidator
) -> Dict[str, Any]:
    """
    Validate time series split for potential issues.
    
    Args:
        X: Feature matrix
        y: Target variable
        cv: Cross-validator object
        
    Returns:
        Dictionary with validation results
    """
    issues = []
    stats = {
        'n_splits': 0,
        'min_train_size': float('inf'),
        'max_train_size': 0,
        'min_test_size': float('inf'),
        'max_test_size': 0,
        'train_sizes': [],
        'test_sizes': []
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        stats['n_splits'] += 1
        train_size = len(train_idx)
        test_size = len(test_idx)
        
        stats['train_sizes'].append(train_size)
        stats['test_sizes'].append(test_size)
        stats['min_train_size'] = min(stats['min_train_size'], train_size)
        stats['max_train_size'] = max(stats['max_train_size'], train_size)
        stats['min_test_size'] = min(stats['min_test_size'], test_size)
        stats['max_test_size'] = max(stats['max_test_size'], test_size)
        
        # Check for overlap
        if len(set(train_idx) & set(test_idx)) > 0:
            issues.append(f"Fold {fold_idx}: Train/test overlap detected")
        
        # Check temporal order
        if len(train_idx) > 0 and len(test_idx) > 0:
            if max(train_idx) >= min(test_idx):
                # This might be intentional with gap/embargo
                if hasattr(cv, 'gap') and cv.gap == 0:
                    issues.append(f"Fold {fold_idx}: Training data after test data")
        
        # Check for sufficient data
        min_train_samples = 100
        min_test_samples = 50
        
        if train_size < min_train_samples:
            issues.append(f"Fold {fold_idx}: Insufficient training samples ({train_size})")
        
        if test_size < min_test_samples:
            issues.append(f"Fold {fold_idx}: Insufficient test samples ({test_size})")
    
    # Calculate statistics
    stats['avg_train_size'] = np.mean(stats['train_sizes'])
    stats['avg_test_size'] = np.mean(stats['test_sizes'])
    stats['train_size_std'] = np.std(stats['train_sizes'])
    stats['test_size_std'] = np.std(stats['test_sizes'])
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'stats': stats
    }