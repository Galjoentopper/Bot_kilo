"""
Cost-Aware Metrics Module
========================

Implements trading metrics that account for transaction costs,
slippage, and other real-world trading considerations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Callable
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

        return float(sharpe)
    
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
            cumulative_returns: Array of cumulative returns (cumulative equity curve)

        Returns:
            Tuple of (max_drawdown, start_idx, end_idx)
        """
        if cumulative_returns is None or len(cumulative_returns) == 0:
            return 0.0, 0, 0

        # Ensure numpy array
        cumulative_returns = np.asarray(cumulative_returns, dtype=float)

        # Running maximum and drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        # Avoid divide-by-zero
        running_max = np.where(running_max == 0, 1e-12, running_max)
        drawdown = (cumulative_returns - running_max) / running_max

        # Max drawdown and indices
        max_dd_idx = int(np.argmin(drawdown))
        max_dd = float(drawdown[max_dd_idx])

        # Start index of the drawdown period
        start_idx = int(np.argmax(cumulative_returns[: max_dd_idx + 1]))

        return float(abs(max_dd)), int(start_idx), int(max_dd_idx)
    
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

        return float(np.sum(returns > 0) / len(returns))
    
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


# New: Net-objective threshold optimizer with turnover penalty and asymmetric bands
def optimize_threshold(
    y_true: Optional[np.ndarray],
    y_proba: np.ndarray,
    prices: np.ndarray,
    metrics_calculator: TradingMetrics,
    thresholds: Optional[np.ndarray] = None,
    turnover_lambda: float = 0.0,
    asymmetric: bool = True,
    objective: str = "sharpe_ratio"
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Search thresholds to maximize a net trading objective with turnover penalty.

    Args:
        y_true: Optional true labels (for diagnostics; not required for objective)
        y_proba: Predicted positive-class probabilities
        prices: Price series aligned to y_proba
        metrics_calculator: TradingMetrics instance (cost-aware)
        thresholds: Grid of thresholds; if None, auto grid
        turnover_lambda: Penalty weight per unit turnover (0 disables)
        asymmetric: If True, use two-sided band (buy>=th, sell<=1-th). If False, single-side long-only
        objective: Metric key to maximize (e.g., 'sharpe_ratio', 'calmar_ratio')

    Returns:
        best_params: dict with selected thresholds and objective value
        metrics_by_th: map of candidate -> computed metrics
    """
    y_proba = np.asarray(y_proba).flatten()
    prices = np.asarray(prices).flatten()
    if thresholds is None:
        thresholds = np.linspace(0.4, 0.6, 41)

    best = {"threshold": 0.5, "objective": -1e18}
    metrics_by = {}

    for th in thresholds:
        if asymmetric:
            long_sig = (y_proba >= th).astype(int)
            short_sig = (y_proba <= (1.0 - th)).astype(int)
            positions = long_sig - short_sig
        else:
            positions = (y_proba >= th).astype(int)  # long-only

        # Compute cost-aware metrics
        m = metrics_calculator.calculate_all_metrics(prices, positions, include_costs=True)

        # Turnover
        turnover = np.abs(np.diff(np.r_[0, positions])).sum()
        m["turnover"] = float(turnover)

        # Net objective
        obj = m.get(objective, 0.0) - turnover_lambda * (turnover / max(len(positions), 1))
        m["objective"] = float(obj)
        metrics_by[f"th_{th:.3f}"] = m

        if obj > best["objective"]:
            best = {"threshold": float(th), "objective": float(obj)}

    logger.info(
        f"optimize_threshold -> best_th={best['threshold']:.3f}, objective={best['objective']:.3f}"
    )
    return best, metrics_by


# New: Map calibrated probabilities to position sizes (capped Kelly with optional vol targeting)
def prob_to_size(
    p: Union[np.ndarray, float],
    p0: float = 0.5,
    kelly_scale: float = 0.5,
    max_leverage: float = 1.0,
    vol: Optional[Union[np.ndarray, float]] = None,
    vol_target: Optional[float] = None,
) -> np.ndarray:
    """
    Convert probability of being correct to position size using a capped-Kelly mapping.

    Args:
        p: Calibrated probability (or array)
        p0: Indifference probability (0.5 for balanced classes)
        kelly_scale: Scale fraction of Kelly (0..1)
        max_leverage: Cap on absolute size
        vol: Realized/forecast volatility aligned to p (optional)
        vol_target: Target volatility to scale sizes toward (optional)

    Returns:
        Array of position sizes in [-max_leverage, max_leverage]
    """
    arr = np.asarray(p, dtype=float)
    edge = np.clip(arr - p0, -0.49, 0.49)
    frac = kelly_scale * (edge / (1 - np.abs(edge) + 1e-9))
    if vol is not None and vol_target is not None:
        vol = np.asarray(vol, dtype=float)
        vol_scale = np.clip(vol_target / (vol + 1e-9), 0.1, 3.0)
        frac = frac * vol_scale
    return np.clip(frac, -max_leverage, max_leverage)


# New: Simple regime gate to disable trading in hostile regimes
class RegimeGate:
    def __init__(self, p_thresh: float = 0.6):
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
        except Exception as e:
            raise ImportError("scikit-learn is required for RegimeGate") from e
        self.model = None
        self.p_thresh = p_thresh

    def fit(self, X: np.ndarray, y_regime: np.ndarray) -> "RegimeGate":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y_regime)
        self.model = clf
        return self

    def allow_trade(self, x_row: np.ndarray) -> bool:
        if self.model is None:
            return True
        pr = self.model.predict_proba(x_row.reshape(1, -1))[0, 1]
        return pr >= self.p_thresh


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
    metrics: Dict[str, float] = {}

    # Ensure numpy arrays
    if isinstance(returns, pd.Series):
        r_values = returns.astype(float).to_numpy()
    else:
        r_values = np.asarray(returns, dtype=float)

    n = len(r_values)
    if n == 0:
        return {
            'mean_return': 0.0,
            'std_return': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'avg_drawdown': 0.0,
            'avg_recovery_periods': 0.0,
        }

    mean_r = float(np.mean(r_values))
    std_r = float(np.std(r_values, ddof=0))
    metrics['mean_return'] = mean_r
    metrics['std_return'] = std_r

    # Skewness and kurtosis (population definitions)
    if std_r > 0:
        z = (r_values - mean_r) / (std_r + 1e-12)
        metrics['skewness'] = float(np.mean(z**3))
        metrics['kurtosis'] = float(np.mean(z**4))
    else:
        metrics['skewness'] = 0.0
        metrics['kurtosis'] = 0.0

    # Risk metrics
    var_95 = float(np.quantile(r_values, 0.05))
    metrics['var_95'] = var_95
    metrics['cvar_95'] = float(np.mean(r_values[r_values <= var_95])) if np.any(r_values <= var_95) else 0.0

    # Performance metrics
    metrics['total_return'] = float(np.prod(1.0 + r_values) - 1.0)
    metrics['sharpe_ratio'] = float(((mean_r - risk_free_rate) / (std_r + 1e-12)) * np.sqrt(252))

    # Drawdown analysis
    cumulative = np.cumprod(1.0 + r_values)
    running_max = np.maximum.accumulate(cumulative)
    running_max = np.where(running_max == 0, 1e-12, running_max)
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = float(np.min(drawdown))
    neg_drawdowns = drawdown[drawdown < 0]
    metrics['avg_drawdown'] = float(np.mean(neg_drawdowns)) if neg_drawdowns.size > 0 else 0.0

    # Recovery metrics (in periods)
    under_water = (drawdown < 0).astype(int)
    # Find contiguous segments where under_water == 1
    if np.any(under_water == 1):
        diffs = np.diff(np.r_[0, under_water, 0])
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        durations = ends - starts
        metrics['avg_recovery_periods'] = float(np.mean(durations)) if durations.size > 0 else 0.0
    else:
        metrics['avg_recovery_periods'] = 0.0

    # Benchmark comparison
    if benchmark_returns is not None:
        if isinstance(benchmark_returns, pd.Series):
            b_values = benchmark_returns.astype(float).to_numpy()
        else:
            b_values = np.asarray(benchmark_returns, dtype=float)
        if len(b_values) == n:
            excess = r_values - b_values
            metrics['excess_return'] = float(np.mean(excess))
            te = float(np.std(excess, ddof=0))
            metrics['tracking_error'] = te
            metrics['information_ratio'] = float((metrics['excess_return'] / (te + 1e-12)) * np.sqrt(252))

            cov = float(np.cov(np.vstack([r_values, b_values]))[0, 1])
            var_b = float(np.var(b_values))
            beta = float(cov / var_b) if var_b > 0 else 0.0
            metrics['beta'] = beta
            metrics['alpha'] = float(mean_r - beta * float(np.mean(b_values)))

    return metrics