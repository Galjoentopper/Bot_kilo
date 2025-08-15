"""
Cost-Aware Metrics Module
========================

Implements trading metrics that account for transaction costs,
slippage, and other real-world trading considerations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)


class TradingMetrics:
    """
    Calculate cost-aware trading metrics for strategy evaluation.
    """
    
    def __init__(
        self,
        fee_bps: float = 10.0,
        slippage_bps: float = 5.0,
        initial_capital: float = 10000.0,
        annualization_factor: int = 252 * 96  # 15-min bars in a year
    ):
        """
        Initialize TradingMetrics.
        
        Args:
            fee_bps: Trading fee in basis points (1 bps = 0.01%)
            slippage_bps: Slippage in basis points
            initial_capital: Initial capital for return calculations
            annualization_factor: Factor for annualizing metrics
        """
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.initial_capital = initial_capital
        self.annualization_factor = annualization_factor
        
        # Convert basis points to decimal
        self.fee_rate = fee_bps / 10000
        self.slippage_rate = slippage_bps / 10000
        self.total_cost_rate = self.fee_rate + self.slippage_rate
        
        logger.info(
            f"TradingMetrics initialized with fee={fee_bps}bps, "
            f"slippage={slippage_bps}bps, total_cost={fee_bps + slippage_bps}bps"
        )
    
    def calculate_returns(
        self,
        prices: np.ndarray,
        positions: np.ndarray,
        include_costs: bool = True
    ) -> Dict[str, float]:
        """
        Calculate returns from prices and positions.
        
        Args:
            prices: Array of prices
            positions: Array of positions (-1, 0, 1)
            include_costs: Whether to include transaction costs
            
        Returns:
            Dictionary of return metrics
        """
        # Calculate price returns
        price_returns = np.diff(prices) / prices[:-1]
        
        # Align positions with returns (positions are for next period)
        if len(positions) > len(price_returns):
            positions = positions[:-1]
        else:
            price_returns = price_returns[:len(positions)]
        
        # Calculate strategy returns
        strategy_returns = positions * price_returns
        
        # Detect position changes (always calculate for trade counting)
        position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        position_changes = position_changes[:len(strategy_returns)]
        
        # Calculate transaction costs if requested
        if include_costs:
            # Apply costs to returns
            transaction_costs = position_changes * self.total_cost_rate
            net_returns = strategy_returns - transaction_costs
        else:
            net_returns = strategy_returns
            transaction_costs = np.zeros_like(strategy_returns)
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + net_returns)
        total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
        
        # Calculate metrics
        metrics = {
            'total_return': total_return,
            'mean_return': np.mean(net_returns),
            'std_return': np.std(net_returns),
            'total_trades': np.sum(position_changes > 0),
            'total_cost': np.sum(transaction_costs),
            'gross_return': np.sum(strategy_returns),
            'net_return': np.sum(net_returns)
        }
        
        return metrics
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annualized)
            annualize: Whether to annualize the ratio
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / self.annualization_factor
        
        mean_excess_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_excess_return / std_return
        
        if annualize:
            sharpe *= np.sqrt(self.annualization_factor)
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        target_return: float = 0.0,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Args:
            returns: Array of returns
            target_return: Target return threshold
            annualize: Whether to annualize the ratio
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - target_return / self.annualization_factor
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return float('inf')
        
        mean_excess_return = np.mean(excess_returns)
        sortino = mean_excess_return / downside_deviation
        
        if annualize:
            sortino *= np.sqrt(self.annualization_factor)
        
        return sortino
    
    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and duration.
        
        Args:
            cumulative_returns: Array of cumulative returns
            
        Returns:
            Tuple of (max_drawdown, start_idx, end_idx)
        """
        if len(cumulative_returns) == 0:
            return 0.0, 0, 0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find start of drawdown
        start_idx = np.argmax(cumulative_returns[:max_dd_idx + 1])
        
        return abs(max_dd), start_idx, max_dd_idx
    
    def calculate_calmar_ratio(
        self,
        returns: np.ndarray,
        annualize: bool = True
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).
        
        Args:
            returns: Array of returns
            annualize: Whether to annualize returns
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)
        
        # Calculate annualized return
        total_periods = len(returns)
        total_return = cumulative_returns[-1] - 1
        
        if annualize and total_periods > 0:
            annualized_return = (1 + total_return) ** (self.annualization_factor / total_periods) - 1
        else:
            annualized_return = total_return
        
        # Calculate max drawdown
        max_dd, _, _ = self.calculate_max_drawdown(cumulative_returns)
        
        if max_dd == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_dd
    
    def calculate_win_rate(self, returns: np.ndarray) -> float:
        """
        Calculate win rate (percentage of positive returns).
        
        Args:
            returns: Array of returns
            
        Returns:
            Win rate (0 to 1)
        """
        if len(returns) == 0:
            return 0.0
        
        return np.sum(returns > 0) / len(returns)
    
    def calculate_profit_factor(self, returns: np.ndarray) -> float:
        """
        Calculate profit factor (gross profits / gross losses).
        
        Args:
            returns: Array of returns
            
        Returns:
            Profit factor
        """
        profits = returns[returns > 0]
        losses = abs(returns[returns < 0])
        
        total_profits = np.sum(profits)
        total_losses = np.sum(losses)
        
        if total_losses == 0:
            return float('inf') if total_profits > 0 else 0.0
        
        return total_profits / total_losses
    
    def calculate_all_metrics(
        self,
        prices: np.ndarray,
        positions: np.ndarray,
        include_costs: bool = True
    ) -> Dict[str, float]:
        """
        Calculate all trading metrics.
        
        Args:
            prices: Array of prices
            positions: Array of positions
            include_costs: Whether to include transaction costs
            
        Returns:
            Dictionary of all metrics
        """
        # Calculate returns
        return_metrics = self.calculate_returns(prices, positions, include_costs)
        
        # Get individual returns
        price_returns = np.diff(prices) / prices[:-1]
        if len(positions) > len(price_returns):
            positions = positions[:-1]
        else:
            price_returns = price_returns[:len(positions)]
        
        strategy_returns = positions * price_returns
        
        # Apply costs
        if include_costs:
            position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
            position_changes = position_changes[:len(strategy_returns)]
            transaction_costs = position_changes * self.total_cost_rate
            net_returns = strategy_returns - transaction_costs
        else:
            net_returns = strategy_returns
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + net_returns)
        
        # Calculate all metrics
        metrics = {
            **return_metrics,
            'sharpe_ratio': self.calculate_sharpe_ratio(net_returns),
            'sortino_ratio': self.calculate_sortino_ratio(net_returns),
            'calmar_ratio': self.calculate_calmar_ratio(net_returns),
            'win_rate': self.calculate_win_rate(net_returns),
            'profit_factor': self.calculate_profit_factor(net_returns),
            'max_drawdown': self.calculate_max_drawdown(cumulative_returns)[0],
            'avg_trade_return': return_metrics['net_return'] / max(return_metrics['total_trades'], 1),
            'cost_per_trade': return_metrics['total_cost'] / max(return_metrics['total_trades'], 1)
        }
        
        return metrics


def find_optimal_threshold(
    probabilities: np.ndarray,
    prices: np.ndarray,
    metrics_calculator: TradingMetrics,
    thresholds: Optional[np.ndarray] = None,
    metric: str = 'sharpe_ratio'
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal probability threshold for trading decisions.
    
    Args:
        probabilities: Predicted probabilities
        prices: Price series
        metrics_calculator: TradingMetrics instance
        thresholds: Thresholds to test (default: 0.3 to 0.7)
        metric: Metric to optimize ('sharpe_ratio', 'calmar_ratio', etc.)
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_optimal)
    """
    if thresholds is None:
        thresholds = np.linspace(0.3, 0.7, 21)
    
    best_threshold = 0.5
    best_metric_value = -float('inf')
    best_metrics = {}
    
    for threshold in thresholds:
        # Generate positions based on threshold
        positions = np.where(probabilities > threshold, 1, -1)
        
        # Calculate metrics
        metrics = metrics_calculator.calculate_all_metrics(prices, positions)
        
        # Get metric value
        metric_value = metrics.get(metric, 0)
        
        # Update best if improved
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
            best_metrics = metrics
    
    logger.info(
        f"Optimal threshold: {best_threshold:.3f} "
        f"({metric}={best_metric_value:.3f})"
    )
    
    return best_threshold, best_metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate classification metrics for trading signals.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        threshold: Threshold for probability conversion
        
    Returns:
        Dictionary of classification metrics
    """
    # Convert probabilities to predictions if needed
    if y_prob is not None:
        y_pred_binary = (y_prob > threshold).astype(int)
    else:
        y_pred_binary = y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1_score': f1_score(y_true, y_pred_binary, zero_division=0)
    }
    
    # Add directional accuracy for regression converted to classification
    if len(np.unique(y_true)) > 2:
        # Convert to direction
        y_true_dir = (y_true > 0).astype(int)
        y_pred_dir = (y_pred > 0).astype(int)
        metrics['directional_accuracy'] = accuracy_score(y_true_dir, y_pred_dir)
    
    return metrics


def calculate_portfolio_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        returns: Portfolio returns series
        benchmark_returns: Benchmark returns (optional)
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of portfolio metrics
    """
    metrics = {}
    
    # Basic statistics
    metrics['mean_return'] = returns.mean()
    metrics['std_return'] = returns.std()
    metrics['skewness'] = returns.skew()
    metrics['kurtosis'] = returns.kurtosis()
    
    # Risk metrics
    metrics['var_95'] = returns.quantile(0.05)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
    
    # Performance metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['sharpe_ratio'] = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    metrics['max_drawdown'] = drawdown.min()
    metrics['avg_drawdown'] = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
    
    # Recovery metrics
    drawdown_periods = (drawdown < 0).astype(int)
    drawdown_changes = drawdown_periods.diff()
    drawdown_starts = drawdown_changes == 1
    drawdown_ends = drawdown_changes == -1
    
    if drawdown_starts.sum() > 0:
        # Calculate average recovery time
        recovery_times = []
        start_dates = returns.index[drawdown_starts]
        end_dates = returns.index[drawdown_ends]
        
        for i in range(min(len(start_dates), len(end_dates))):
            recovery_time = (end_dates[i] - start_dates[i]).days
            recovery_times.append(recovery_time)
        
        metrics['avg_recovery_days'] = np.mean(recovery_times) if recovery_times else 0
    else:
        metrics['avg_recovery_days'] = 0
    
    # Benchmark comparison
    if benchmark_returns is not None and len(benchmark_returns) == len(returns):
        excess_returns = returns - benchmark_returns
        metrics['excess_return'] = excess_returns.mean()
        metrics['tracking_error'] = excess_returns.std()
        metrics['information_ratio'] = metrics['excess_return'] / metrics['tracking_error'] * np.sqrt(252)
        
        # Beta and alpha
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        metrics['beta'] = covariance / benchmark_variance if benchmark_variance > 0 else 0
        metrics['alpha'] = metrics['mean_return'] - metrics['beta'] * benchmark_returns.mean()
    
    return metrics