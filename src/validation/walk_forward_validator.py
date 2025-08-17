"""
Walk-Forward Validation for Financial ML
=======================================

Implements time-aware validation strategies specifically designed for financial ML models.
Prevents data leakage and provides realistic performance estimates by respecting the 
temporal nature of financial data.

Key Features:
- Multiple walk-forward validation strategies (expanding, rolling, purged)
- Time-aware train/validation/test splits with no future data leakage
- Financial market regime analysis and performance breakdown
- Out-of-sample performance tracking with statistical significance testing
- Integration with Bayesian optimization and hyperparameter tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationStrategy(Enum):
    """Walk-forward validation strategies."""
    EXPANDING_WINDOW = "expanding"      # Growing training window
    ROLLING_WINDOW = "rolling"          # Fixed-size rolling window  
    ANCHORED_WALK = "anchored"          # Fixed start, expanding end
    PURGED_CROSS_VAL = "purged"         # Time-series cross validation with purging

class PerformanceMetric(Enum):
    """Performance metrics for financial ML validation."""
    MSE = "mse"
    MAE = "mae"
    RMSE = "rmse"
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"

@dataclass
class ValidationSplit:
    """Single validation split with time-aware indices."""
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: Optional[int] = None
    test_end: Optional[int] = None
    split_date: Optional[pd.Timestamp] = None

@dataclass
class ValidationResult:
    """Results from a single validation fold."""
    split_info: ValidationSplit
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    actual_values: Optional[np.ndarray] = None
    model_params: Optional[Dict[str, Any]] = None
    training_time: float = 0.0
    
@dataclass 
class WalkForwardResults:
    """Complete walk-forward validation results."""
    validation_results: List[ValidationResult] = field(default_factory=list)
    summary_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stability_analysis: Dict[str, Any] = field(default_factory=dict)
    regime_analysis: Dict[str, Any] = field(default_factory=dict)
    statistical_significance: Dict[str, Any] = field(default_factory=dict)
    total_validation_time: float = 0.0

class WalkForwardValidator:
    """
    Walk-forward validator for financial ML models with time-aware splitting.
    
    Implements proper temporal validation to prevent data leakage and provide
    realistic out-of-sample performance estimates for financial models.
    """
    
    def __init__(
        self,
        strategy: ValidationStrategy = ValidationStrategy.EXPANDING_WINDOW,
        min_train_size: int = 1000,
        val_size: int = 200,
        test_size: int = 100,
        step_size: int = 50,
        purge_length: int = 0,
        embargo_length: int = 0,
        max_splits: Optional[int] = None
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            strategy: Validation strategy to use
            min_train_size: Minimum training set size
            val_size: Validation set size (fixed or minimum for expanding)
            test_size: Test set size (if using three-way splits)
            step_size: Step size between validation windows
            purge_length: Number of samples to purge before validation (prevent leakage)
            embargo_length: Number of samples to embargo after training (prevent leakage)
            max_splits: Maximum number of validation splits (None for all possible)
        """
        self.strategy = strategy
        self.min_train_size = min_train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size
        self.purge_length = purge_length
        self.embargo_length = embargo_length
        self.max_splits = max_splits
        
        logger.info(f"WalkForwardValidator initialized with {strategy.value} strategy")
        logger.info(f"Training size: {min_train_size}+, Validation: {val_size}, Test: {test_size}")
        
    def generate_splits(
        self,
        data_length: int,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> List[ValidationSplit]:
        """
        Generate time-aware validation splits.
        
        Args:
            data_length: Total length of the dataset
            timestamps: Optional datetime index for date tracking
            
        Returns:
            List of ValidationSplit objects
        """
        splits = []
        
        if self.strategy == ValidationStrategy.EXPANDING_WINDOW:
            splits = self._generate_expanding_splits(data_length, timestamps)
        elif self.strategy == ValidationStrategy.ROLLING_WINDOW:
            splits = self._generate_rolling_splits(data_length, timestamps)
        elif self.strategy == ValidationStrategy.ANCHORED_WALK:
            splits = self._generate_anchored_splits(data_length, timestamps)
        elif self.strategy == ValidationStrategy.PURGED_CROSS_VAL:
            splits = self._generate_purged_splits(data_length, timestamps)
        else:
            raise ValueError(f"Unknown validation strategy: {self.strategy}")
        
        # Limit splits if max_splits is specified
        if self.max_splits is not None:
            splits = splits[:self.max_splits]
        
        logger.info(f"Generated {len(splits)} validation splits using {self.strategy.value}")
        return splits
    
    def _generate_expanding_splits(
        self, 
        data_length: int, 
        timestamps: Optional[pd.DatetimeIndex]
    ) -> List[ValidationSplit]:
        """Generate expanding window validation splits."""
        splits = []
        
        # Start with minimum training size
        train_start = 0
        train_end = self.min_train_size
        
        while train_end + self.embargo_length + self.val_size <= data_length:
            # Apply embargo to prevent leakage
            val_start = train_end + self.embargo_length
            val_end = val_start + self.val_size
            
            # Optional test set
            test_start = None
            test_end = None
            if val_end + self.test_size <= data_length:
                test_start = val_end
                test_end = test_start + self.test_size
            
            # Get split date if timestamps available
            split_date = timestamps[val_start] if timestamps is not None else None
            
            split = ValidationSplit(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                split_date=split_date
            )
            splits.append(split)
            
            # Expand training window
            train_end += self.step_size
            
        return splits
    
    def _generate_rolling_splits(
        self, 
        data_length: int, 
        timestamps: Optional[pd.DatetimeIndex]
    ) -> List[ValidationSplit]:
        """Generate rolling window validation splits."""
        splits = []
        
        # Fixed training window size
        train_window_size = self.min_train_size
        
        train_start = 0
        while train_start + train_window_size + self.embargo_length + self.val_size <= data_length:
            train_end = train_start + train_window_size
            
            # Apply embargo
            val_start = train_end + self.embargo_length
            val_end = val_start + self.val_size
            
            # Optional test set
            test_start = None
            test_end = None
            if val_end + self.test_size <= data_length:
                test_start = val_end
                test_end = test_start + self.test_size
            
            split_date = timestamps[val_start] if timestamps is not None else None
            
            split = ValidationSplit(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                split_date=split_date
            )
            splits.append(split)
            
            # Roll the window
            train_start += self.step_size
            
        return splits
    
    def _generate_anchored_splits(
        self, 
        data_length: int, 
        timestamps: Optional[pd.DatetimeIndex]
    ) -> List[ValidationSplit]:
        """Generate anchored walk-forward splits (fixed start, expanding end)."""
        splits = []
        
        train_start = 0  # Fixed start
        train_end = self.min_train_size
        
        while train_end + self.embargo_length + self.val_size <= data_length:
            val_start = train_end + self.embargo_length
            val_end = val_start + self.val_size
            
            test_start = None
            test_end = None
            if val_end + self.test_size <= data_length:
                test_start = val_end
                test_end = test_start + self.test_size
            
            split_date = timestamps[val_start] if timestamps is not None else None
            
            split = ValidationSplit(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
                split_date=split_date
            )
            splits.append(split)
            
            # Expand only the end of training window
            train_end += self.step_size
            
        return splits
    
    def _generate_purged_splits(
        self, 
        data_length: int, 
        timestamps: Optional[pd.DatetimeIndex]
    ) -> List[ValidationSplit]:
        """Generate purged cross-validation splits for time series."""
        splits = []
        
        # Number of folds
        n_folds = max(3, min(10, data_length // (self.min_train_size + self.val_size)))
        fold_size = data_length // n_folds
        
        for fold in range(n_folds):
            # Validation set for this fold
            val_start = fold * fold_size
            val_end = min((fold + 1) * fold_size, data_length)
            
            if val_end - val_start < self.val_size:
                continue  # Skip too small validation sets
            
            # Training data: everything before validation with purging
            train_start = 0
            train_end = max(0, val_start - self.purge_length)
            
            if train_end - train_start < self.min_train_size:
                continue  # Skip insufficient training data
            
            split_date = timestamps[val_start] if timestamps is not None else None
            
            split = ValidationSplit(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                split_date=split_date
            )
            splits.append(split)
            
        return splits
    
    def validate_model(
        self,
        train_function: callable,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        prices: Optional[np.ndarray] = None,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> WalkForwardResults:
        """
        Run walk-forward validation on a model.
        
        Args:
            train_function: Function that trains model and returns predictions
            X: Feature matrix
            y: Target values  
            timestamps: Datetime index for time-aware analysis
            prices: Price data for trading simulation (optional)
            hyperparams: Fixed hyperparameters for training
            
        Returns:
            WalkForwardResults with comprehensive validation results
        """
        logger.info("Starting walk-forward validation")
        start_time = pd.Timestamp.now()
        
        # Generate validation splits
        splits = self.generate_splits(len(X), timestamps)
        
        validation_results = []
        
        for i, split in enumerate(splits):
            logger.info(f"Processing validation split {i+1}/{len(splits)}")
            logger.info(f"Train: [{split.train_start}:{split.train_end}], "
                       f"Val: [{split.val_start}:{split.val_end}]")
            
            try:
                # Extract data for this split
                X_train = X[split.train_start:split.train_end]
                y_train = y[split.train_start:split.train_end]
                X_val = X[split.val_start:split.val_end]
                y_val = y[split.val_start:split.val_end]
                
                # Optional test set
                X_test, y_test = None, None
                if split.test_start is not None and split.test_end is not None:
                    X_test = X[split.test_start:split.test_end]
                    y_test = y[split.test_start:split.test_end]
                
                # Train model and get predictions
                fold_start_time = pd.Timestamp.now()
                
                train_result = train_function(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    hyperparams=hyperparams
                )
                
                training_time = (pd.Timestamp.now() - fold_start_time).total_seconds()
                
                # Extract predictions and metrics
                val_predictions = train_result.get('val_predictions')
                test_predictions = train_result.get('test_predictions')
                
                # Calculate comprehensive metrics
                val_metrics = self._calculate_metrics(y_val, val_predictions, prices)
                test_metrics = {}
                if y_test is not None and test_predictions is not None:
                    test_metrics = self._calculate_metrics(y_test, test_predictions, prices)
                
                # Store validation result
                result = ValidationResult(
                    split_info=split,
                    val_metrics=val_metrics,
                    test_metrics=test_metrics,
                    predictions=val_predictions,
                    actual_values=y_val,
                    model_params=train_result.get('model_params'),
                    training_time=training_time
                )
                
                validation_results.append(result)
                
                logger.info(f"Fold {i+1} completed - Val RMSE: {val_metrics.get('rmse', 'N/A'):.6f}")
                
            except Exception as e:
                logger.error(f"Error in validation fold {i+1}: {e}")
                continue
        
        total_time = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Compile comprehensive results
        results = WalkForwardResults(
            validation_results=validation_results,
            total_validation_time=total_time
        )
        
        # Calculate summary statistics
        results.summary_metrics = self._calculate_summary_metrics(validation_results)
        
        # Perform stability analysis
        results.stability_analysis = self._analyze_performance_stability(validation_results)
        
        # Analyze performance by market regime (if timestamps available)
        if timestamps is not None:
            results.regime_analysis = self._analyze_regime_performance(validation_results, timestamps)
        
        # Statistical significance testing
        results.statistical_significance = self._test_statistical_significance(validation_results)
        
        logger.info(f"Walk-forward validation completed in {total_time:.2f} seconds")
        logger.info(f"Processed {len(validation_results)} successful folds")
        
        return results
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        prices: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if y_pred is None or len(y_pred) == 0:
            return {}
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred) 
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Directional accuracy
        y_true_direction = np.sign(y_true)
        y_pred_direction = np.sign(y_pred)
        metrics['directional_accuracy'] = np.mean(y_true_direction == y_pred_direction)
        
        # Financial metrics (if we have return predictions)
        try:
            # Assume predictions are returns
            returns = y_pred
            
            # Sharpe ratio (annualized)
            if np.std(returns) > 0:
                metrics['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = abs(np.min(drawdown))
            
            # Win rate and profit factor
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]
            
            metrics['win_rate'] = len(winning_returns) / max(len(returns[returns != 0]), 1)
            
            if len(losing_returns) > 0:
                avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
                avg_loss = np.mean(losing_returns)
                metrics['profit_factor'] = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
            else:
                metrics['profit_factor'] = float('inf') if len(winning_returns) > 0 else 1.0
                
        except Exception as e:
            logger.warning(f"Error calculating financial metrics: {e}")
        
        return metrics
    
    def _calculate_summary_metrics(self, results: List[ValidationResult]) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics across all validation folds."""
        if not results:
            return {}
        
        summary = {}
        
        # Collect all metrics across folds
        all_metrics = {}
        for result in results:
            for metric_name, value in result.val_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate summary statistics
        for metric_name, values in all_metrics.items():
            if len(values) > 0:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        return summary
    
    def _analyze_performance_stability(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze performance stability across validation folds."""
        if len(results) < 2:
            return {'error': 'Insufficient results for stability analysis'}
        
        # Extract key metrics across folds
        rmse_values = [r.val_metrics.get('rmse', np.nan) for r in results]
        sharpe_values = [r.val_metrics.get('sharpe_ratio', np.nan) for r in results]
        dir_acc_values = [r.val_metrics.get('directional_accuracy', np.nan) for r in results]
        
        # Remove NaN values
        rmse_values = [x for x in rmse_values if not np.isnan(x)]
        sharpe_values = [x for x in sharpe_values if not np.isnan(x)]
        dir_acc_values = [x for x in dir_acc_values if not np.isnan(x)]
        
        stability_metrics = {}
        
        if rmse_values:
            stability_metrics['rmse_stability'] = {
                'coefficient_of_variation': np.std(rmse_values) / np.mean(rmse_values) if np.mean(rmse_values) > 0 else float('inf'),
                'trend_slope': self._calculate_trend_slope(rmse_values),
                'volatility': np.std(rmse_values)
            }
        
        if sharpe_values:
            stability_metrics['sharpe_stability'] = {
                'coefficient_of_variation': np.std(sharpe_values) / abs(np.mean(sharpe_values)) if np.mean(sharpe_values) != 0 else float('inf'),
                'trend_slope': self._calculate_trend_slope(sharpe_values),
                'volatility': np.std(sharpe_values)
            }
        
        if dir_acc_values:
            stability_metrics['directional_accuracy_stability'] = {
                'coefficient_of_variation': np.std(dir_acc_values) / np.mean(dir_acc_values) if np.mean(dir_acc_values) > 0 else float('inf'),
                'trend_slope': self._calculate_trend_slope(dir_acc_values),
                'volatility': np.std(dir_acc_values)
            }
        
        # Overall stability score (lower is more stable)
        cv_scores = [s.get('coefficient_of_variation', float('inf')) for s in stability_metrics.values()]
        stability_metrics['overall_stability_score'] = np.mean([cv for cv in cv_scores if cv != float('inf')])
        
        return stability_metrics
    
    def _analyze_regime_performance(
        self, 
        results: List[ValidationResult], 
        timestamps: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Analyze performance across different market regimes."""
        regime_analysis = {}
        
        try:
            # Simple regime classification based on volatility
            for result in results:
                if result.split_info.split_date is not None:
                    # Get period around validation
                    split_date = result.split_info.split_date
                    
                    # Classify regime (simplified)
                    if hasattr(result, 'regime_volatility'):
                        if result.regime_volatility > 0.03:
                            regime = 'high_vol'
                        elif result.regime_volatility < 0.015:
                            regime = 'low_vol'  
                        else:
                            regime = 'normal'
                    else:
                        regime = 'unknown'
                    
                    if regime not in regime_analysis:
                        regime_analysis[regime] = {
                            'rmse_values': [],
                            'sharpe_values': [],
                            'directional_accuracy_values': [],
                            'count': 0
                        }
                    
                    regime_analysis[regime]['rmse_values'].append(
                        result.val_metrics.get('rmse', np.nan)
                    )
                    regime_analysis[regime]['sharpe_values'].append(
                        result.val_metrics.get('sharpe_ratio', np.nan)
                    )
                    regime_analysis[regime]['directional_accuracy_values'].append(
                        result.val_metrics.get('directional_accuracy', np.nan)
                    )
                    regime_analysis[regime]['count'] += 1
            
            # Calculate summary statistics for each regime
            for regime, data in regime_analysis.items():
                for metric_name in ['rmse_values', 'sharpe_values', 'directional_accuracy_values']:
                    values = [x for x in data[metric_name] if not np.isnan(x)]
                    if values:
                        data[f'{metric_name}_mean'] = np.mean(values)
                        data[f'{metric_name}_std'] = np.std(values)
                        
        except Exception as e:
            logger.warning(f"Error in regime analysis: {e}")
            regime_analysis['error'] = str(e)
        
        return regime_analysis
    
    def _test_statistical_significance(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Test statistical significance of model performance."""
        if len(results) < 5:
            return {'error': 'Insufficient samples for statistical testing'}
        
        significance_tests = {}
        
        try:
            # Test if performance is significantly different from random
            dir_acc_values = [r.val_metrics.get('directional_accuracy', 0.5) for r in results]
            dir_acc_values = [x for x in dir_acc_values if not np.isnan(x)]
            
            if dir_acc_values:
                # T-test against random (50% directional accuracy)
                t_stat, p_value = stats.ttest_1samp(dir_acc_values, 0.5)
                significance_tests['directional_accuracy_vs_random'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'is_significant': p_value < 0.05,
                    'mean_performance': np.mean(dir_acc_values)
                }
            
            # Test performance consistency (normality)
            rmse_values = [r.val_metrics.get('rmse', np.nan) for r in results]
            rmse_values = [x for x in rmse_values if not np.isnan(x)]
            
            if len(rmse_values) >= 8:  # Minimum for normality test
                shapiro_stat, shapiro_p = stats.shapiro(rmse_values)
                significance_tests['rmse_normality'] = {
                    'shapiro_statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
        
        except Exception as e:
            logger.warning(f"Error in statistical significance testing: {e}")
            significance_tests['error'] = str(e)
        
        return significance_tests
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return float(slope)
    
    def save_results(self, results: WalkForwardResults, filepath: str):
        """Save walk-forward validation results to file."""
        import pickle
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for complete preservation
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Also save summary as JSON for readability
        summary_data = {
            'summary_metrics': results.summary_metrics,
            'stability_analysis': results.stability_analysis,
            'regime_analysis': results.regime_analysis,
            'statistical_significance': results.statistical_significance,
            'total_folds': len(results.validation_results),
            'total_time': results.total_validation_time,
            'validation_strategy': self.strategy.value
        }
        
        import json
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def plot_performance_over_time(
        self, 
        results: WalkForwardResults, 
        save_path: Optional[str] = None
    ):
        """Plot performance metrics over time."""
        if not results.validation_results:
            logger.warning("No results to plot")
            return
        
        # Extract data for plotting
        dates = []
        rmse_values = []
        sharpe_values = []
        dir_acc_values = []
        
        for result in results.validation_results:
            if result.split_info.split_date:
                dates.append(result.split_info.split_date)
                rmse_values.append(result.val_metrics.get('rmse', np.nan))
                sharpe_values.append(result.val_metrics.get('sharpe_ratio', np.nan))
                dir_acc_values.append(result.val_metrics.get('directional_accuracy', np.nan))
        
        if not dates:
            logger.warning("No dates available for time-based plotting")
            return
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Walk-Forward Validation Performance Over Time')
        
        # RMSE over time
        axes[0].plot(dates, rmse_values, 'b-', marker='o', alpha=0.7)
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Root Mean Square Error')
        axes[0].grid(True, alpha=0.3)
        
        # Sharpe ratio over time
        axes[1].plot(dates, sharpe_values, 'g-', marker='o', alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].set_title('Sharpe Ratio')
        axes[1].grid(True, alpha=0.3)
        
        # Directional accuracy over time
        axes[2].plot(dates, dir_acc_values, 'r-', marker='o', alpha=0.7)
        axes[2].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random')
        axes[2].set_ylabel('Directional Accuracy')
        axes[2].set_title('Directional Accuracy')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        else:
            plt.show()

def create_walk_forward_validator(
    strategy: str = "expanding",
    min_train_size: int = 1000,
    val_size: int = 200,
    step_size: int = 50,
    embargo_length: int = 0
) -> WalkForwardValidator:
    """
    Convenience function to create a walk-forward validator.
    
    Args:
        strategy: "expanding", "rolling", "anchored", or "purged"
        min_train_size: Minimum training samples
        val_size: Validation samples  
        step_size: Step between validation windows
        embargo_length: Embargo length to prevent leakage
        
    Returns:
        Configured WalkForwardValidator
    """
    strategy_enum = ValidationStrategy(strategy)
    
    return WalkForwardValidator(
        strategy=strategy_enum,
        min_train_size=min_train_size,
        val_size=val_size,
        step_size=step_size,
        embargo_length=embargo_length
    )