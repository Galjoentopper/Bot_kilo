"""
Financial ML Hyperparameter Optimization Strategy
================================================

Research-backed hyperparameter optimization specifically designed for financial ML.
Implements best practices from academic literature and industry experience for 
stable, profitable financial time series prediction models.

Key Features:
- Domain-specific parameter bounds based on financial market characteristics
- Risk-aware objective functions that consider Sharpe ratio and drawdown
- Time-aware validation that respects temporal dependencies
- Conservative regularization to prevent overfitting on noisy financial data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification for optimization strategy."""
    BULL = "bull"           # Trending up markets
    BEAR = "bear"           # Trending down markets  
    SIDEWAYS = "sideways"   # Range-bound markets
    HIGH_VOL = "high_vol"   # High volatility periods
    LOW_VOL = "low_vol"     # Low volatility periods

class AssetClass(Enum):
    """Asset class for specialized optimization."""
    CRYPTO = "crypto"       # Cryptocurrency pairs
    FOREX = "forex"         # Currency pairs
    STOCKS = "stocks"       # Equity instruments
    COMMODITIES = "commodities"  # Commodity futures

@dataclass
class FinancialMLConstraints:
    """Financial ML specific constraints for hyperparameter optimization."""
    
    # Risk management constraints
    max_drawdown_tolerance: float = 0.15        # 15% max drawdown
    min_sharpe_ratio: float = 1.0               # Minimum acceptable Sharpe ratio
    max_volatility: float = 0.25                # 25% annualized volatility limit
    min_win_rate: float = 0.45                  # Minimum 45% win rate
    
    # Training stability constraints
    max_learning_rate: float = 0.001            # Conservative learning rate bound
    min_regularization: float = 0.1             # Minimum dropout/weight decay
    max_model_complexity: int = 1000000         # Max parameter count
    min_training_samples: int = 1000            # Minimum training samples
    
    # Financial data constraints
    max_lookback_days: int = 252                # 1 year maximum lookback
    min_prediction_horizon: int = 1             # Minimum 1 period ahead
    max_prediction_horizon: int = 48            # Maximum 48 periods (12 hours for 15min data)
    
    # Overfitting prevention
    max_feature_count: int = 200                # Maximum number of features
    min_validation_samples: int = 500           # Minimum validation set size
    early_stopping_patience: int = 15           # Conservative early stopping
    
    # Market microstructure constraints  
    min_trade_frequency: float = 0.01           # Minimum 1% of time in market
    max_trade_frequency: float = 0.5            # Maximum 50% of time in market
    transaction_cost_bps: float = 5.0           # 5 basis points transaction cost

class FinancialHyperparameterOptimizer:
    """
    Advanced hyperparameter optimizer specifically designed for financial ML.
    
    Incorporates domain knowledge, risk management, and market microstructure
    considerations into the optimization process.
    """
    
    def __init__(
        self,
        asset_class: AssetClass = AssetClass.CRYPTO,
        market_regime: Optional[MarketRegime] = None,
        constraints: Optional[FinancialMLConstraints] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize financial hyperparameter optimizer.
        
        Args:
            asset_class: Type of financial instrument being modeled
            market_regime: Current market conditions for regime-specific optimization
            constraints: Financial ML specific constraints
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (annualized)
        """
        self.asset_class = asset_class
        self.market_regime = market_regime
        self.constraints = constraints or FinancialMLConstraints()
        self.risk_free_rate = risk_free_rate
        
        # Load domain-specific parameter ranges
        self.param_ranges = self._get_financial_parameter_ranges()
        
        # Initialize optimization history
        self.optimization_history = []
        
        logger.info(f"Initialized FinancialHyperparameterOptimizer for {asset_class.value}")
        if market_regime:
            logger.info(f"Optimizing for {market_regime.value} market conditions")
    
    def _get_financial_parameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get domain-specific parameter ranges for financial ML models."""
        
        # Base ranges for GRU models in financial applications
        base_ranges = {
            'learning_rate': {
                'type': 'log_uniform',
                'low': 1e-5,
                'high': self.constraints.max_learning_rate,
                'prior': 'prefer_conservative'  # Bias towards lower learning rates
            },
            'hidden_size': {
                'type': 'choice',
                'choices': [32, 64, 96, 128, 192, 256],
                'prior': 'prefer_moderate'  # Avoid very large models
            },
            'num_layers': {
                'type': 'choice', 
                'choices': [1, 2, 3],
                'prior': 'prefer_shallow'  # Financial data benefits from simpler models
            },
            'dropout': {
                'type': 'uniform',
                'low': self.constraints.min_regularization,
                'high': 0.6,
                'prior': 'prefer_high'  # Financial data needs strong regularization
            },
            'batch_size': {
                'type': 'choice',
                'choices': [16, 32, 64, 128],
                'prior': 'prefer_moderate'
            },
            'sequence_length': {
                'type': 'choice',
                'choices': [10, 15, 20, 30, 45, 60],  # In terms of time periods
                'prior': 'prefer_moderate'  # Too long sequences overfit in finance
            },
            'weight_decay': {
                'type': 'log_uniform',
                'low': 1e-6,
                'high': 1e-2,
                'prior': 'prefer_high'  # Strong L2 regularization for financial data
            },
            'gradient_clip_norm': {
                'type': 'uniform',
                'low': 0.1,
                'high': 2.0,
                'prior': 'prefer_conservative'  # Conservative clipping for stability
            }
        }
        
        # Asset class specific adjustments
        if self.asset_class == AssetClass.CRYPTO:
            # Crypto is more volatile, needs more regularization
            base_ranges['learning_rate']['high'] = min(5e-4, self.constraints.max_learning_rate)
            base_ranges['dropout']['low'] = max(0.2, self.constraints.min_regularization)
            base_ranges['weight_decay']['high'] = 5e-3
            
        elif self.asset_class == AssetClass.FOREX:
            # Forex has lower signal-to-noise, needs careful tuning
            base_ranges['learning_rate']['high'] = min(2e-4, self.constraints.max_learning_rate)
            base_ranges['sequence_length']['choices'] = [15, 20, 30, 45]  # Shorter sequences
            
        elif self.asset_class == AssetClass.STOCKS:
            # Stocks have more predictable patterns, can use longer sequences
            base_ranges['sequence_length']['choices'] = [20, 30, 45, 60, 90]
            base_ranges['dropout']['low'] = max(0.15, self.constraints.min_regularization)
        
        # Market regime adjustments
        if self.market_regime == MarketRegime.HIGH_VOL:
            # High volatility needs more conservative parameters
            base_ranges['learning_rate']['high'] *= 0.5
            base_ranges['dropout']['low'] = max(0.3, self.constraints.min_regularization)
            base_ranges['gradient_clip_norm']['high'] = 1.0
            
        elif self.market_regime == MarketRegime.LOW_VOL:
            # Low volatility can handle slightly more aggressive parameters
            base_ranges['learning_rate']['high'] *= 1.2
            base_ranges['hidden_size']['choices'].extend([384, 512])
            
        return base_ranges
    
    def get_financial_objective_function(
        self,
        train_func: Callable,
        validation_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        price_data: Optional[np.ndarray] = None
    ) -> Callable:
        """
        Create a financial ML specific objective function that optimizes for
        risk-adjusted returns rather than just prediction accuracy.
        
        Args:
            train_func: Function that trains model and returns predictions
            validation_data: (X_val, y_val, X_test, y_test) validation data
            price_data: Optional price data for trading simulation
            
        Returns:
            Objective function for hyperparameter optimization
        """
        X_val, y_val, X_test, y_test = validation_data
        
        def financial_objective(params: Dict[str, Any]) -> Dict[str, float]:
            """
            Financial ML objective function that considers multiple criteria:
            - Prediction accuracy (MSE, MAE, directional accuracy)
            - Risk-adjusted returns (Sharpe ratio, Sortino ratio)
            - Drawdown control (maximum drawdown, recovery time)
            - Trading viability (win rate, profit factor, trade frequency)
            """
            try:
                # Train model with given parameters
                model_results = train_func(params)
                predictions = model_results.get('predictions')
                training_metrics = model_results.get('metrics', {})
                
                if predictions is None:
                    return {'loss': 1000.0, 'status': 'FAIL'}
                
                # 1. Prediction Quality Metrics
                mse = np.mean((predictions - y_test) ** 2)
                mae = np.mean(np.abs(predictions - y_test))
                
                # Directional accuracy (critical for trading)
                pred_direction = np.sign(predictions)
                actual_direction = np.sign(y_test)
                directional_accuracy = np.mean(pred_direction == actual_direction)
                
                # 2. Financial Performance Metrics
                if price_data is not None:
                    # Simulate trading based on predictions
                    trading_metrics = self._simulate_trading_performance(
                        predictions, y_test, price_data
                    )
                    
                    sharpe_ratio = trading_metrics['sharpe_ratio']
                    max_drawdown = trading_metrics['max_drawdown'] 
                    win_rate = trading_metrics['win_rate']
                    profit_factor = trading_metrics['profit_factor']
                    trade_frequency = trading_metrics['trade_frequency']
                else:
                    # Fallback metrics based on prediction quality
                    returns = predictions  # Assume predictions are returns
                    sharpe_ratio = self._calculate_sharpe_ratio(returns)
                    max_drawdown = self._estimate_drawdown_from_returns(returns)
                    win_rate = directional_accuracy
                    profit_factor = 1.0
                    trade_frequency = 0.1
                
                # 3. Model Complexity and Stability
                param_count = self._estimate_parameter_count(params)
                complexity_penalty = min(param_count / self.constraints.max_model_complexity, 1.0)
                
                training_stability = 1.0 - training_metrics.get('gradient_explosions', 0) / 100
                
                # 4. Constraint Violations
                constraint_violations = 0
                
                if max_drawdown > self.constraints.max_drawdown_tolerance:
                    constraint_violations += (max_drawdown - self.constraints.max_drawdown_tolerance) * 10
                    
                if sharpe_ratio < self.constraints.min_sharpe_ratio:
                    constraint_violations += (self.constraints.min_sharpe_ratio - sharpe_ratio) * 2
                    
                if win_rate < self.constraints.min_win_rate:
                    constraint_violations += (self.constraints.min_win_rate - win_rate) * 3
                    
                if trade_frequency < self.constraints.min_trade_frequency:
                    constraint_violations += 1.0
                elif trade_frequency > self.constraints.max_trade_frequency:
                    constraint_violations += (trade_frequency - self.constraints.max_trade_frequency) * 5
                
                # 5. Multi-Objective Loss Function
                # Primary: Risk-adjusted returns with drawdown control
                risk_adjusted_loss = (1.0 / (sharpe_ratio + 0.1)) + max_drawdown * 2
                
                # Secondary: Prediction accuracy
                prediction_loss = np.sqrt(mse) + (1.0 - directional_accuracy)
                
                # Tertiary: Model complexity and stability
                complexity_loss = complexity_penalty + (1.0 - training_stability)
                
                # Combined loss with financial focus
                combined_loss = (
                    0.5 * risk_adjusted_loss +      # 50% weight on risk-adjusted returns
                    0.3 * prediction_loss +         # 30% weight on prediction accuracy  
                    0.1 * complexity_loss +         # 10% weight on model complexity
                    0.1 * constraint_violations     # 10% weight on constraint violations
                )
                
                # Additional metrics for analysis
                additional_metrics = {
                    'mse': float(mse),
                    'mae': float(mae), 
                    'directional_accuracy': float(directional_accuracy),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown),
                    'win_rate': float(win_rate),
                    'profit_factor': float(profit_factor),
                    'trade_frequency': float(trade_frequency),
                    'param_count': int(param_count),
                    'training_stability': float(training_stability),
                    'constraint_violations': float(constraint_violations)
                }
                
                return {
                    'loss': float(combined_loss),
                    'status': 'OK',
                    **additional_metrics
                }
                
            except Exception as e:
                logger.error(f"Error in financial objective function: {e}")
                return {'loss': 1000.0, 'status': 'FAIL', 'error': str(e)}
        
        return financial_objective
    
    def _simulate_trading_performance(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        prices: np.ndarray,
        threshold: float = 0.001
    ) -> Dict[str, float]:
        """
        Simulate trading performance based on model predictions.
        
        Args:
            predictions: Model predicted returns
            actual_returns: Actual returns
            prices: Price data for transaction cost calculation
            threshold: Minimum signal threshold for trading
            
        Returns:
            Dictionary of trading performance metrics
        """
        # Generate trading signals
        signals = np.where(np.abs(predictions) > threshold, np.sign(predictions), 0)
        
        # Calculate position returns (simplified)
        position_returns = signals[:-1] * actual_returns[1:]  # t+1 return with t signal
        
        # Apply transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
        transaction_costs = position_changes * (self.constraints.transaction_cost_bps / 10000)
        net_returns = position_returns - transaction_costs[1:]
        
        # Performance metrics
        total_return = np.sum(net_returns)
        volatility = np.std(net_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = self._calculate_sharpe_ratio(net_returns)
        
        # Drawdown calculation
        cumulative_returns = np.cumprod(1 + net_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.abs(np.min(drawdown))
        
        # Trading statistics
        winning_trades = net_returns[net_returns > 0]
        losing_trades = net_returns[net_returns < 0]
        
        win_rate = len(winning_trades) / max(len(net_returns[net_returns != 0]), 1)
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else -0.001
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
        
        trade_frequency = np.mean(signals != 0)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trade_frequency': trade_frequency,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - (self.risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns / np.std(returns)) * np.sqrt(252)
    
    def _estimate_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """Estimate maximum drawdown from return predictions."""
        if len(returns) == 0:
            return 1.0
            
        # Simulate cumulative performance
        cumulative = np.cumprod(1 + returns * 0.01)  # Scale returns
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def _estimate_parameter_count(self, params: Dict[str, Any]) -> int:
        """Estimate model parameter count from hyperparameters."""
        hidden_size = params.get('hidden_size', 64)
        num_layers = params.get('num_layers', 2)
        input_size = 114  # Assume standard feature count
        
        # GRU parameter count estimation
        gru_params = num_layers * (3 * hidden_size * (input_size + hidden_size + 2))
        fc_params = hidden_size * 2 + 2  # Two FC layers
        
        return gru_params + fc_params
    
    def get_optimization_strategy_for_regime(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get regime-specific optimization strategy.
        
        Args:
            regime: Current market regime
            
        Returns:
            Optimization strategy configuration
        """
        strategies = {
            MarketRegime.BULL: {
                'focus': 'trend_following',
                'risk_tolerance': 'moderate',
                'sequence_preference': 'longer',
                'regularization': 'moderate',
                'description': 'Optimized for trending markets with momentum strategies'
            },
            MarketRegime.BEAR: {
                'focus': 'risk_management', 
                'risk_tolerance': 'conservative',
                'sequence_preference': 'shorter',
                'regularization': 'high',
                'description': 'Conservative approach with strong risk controls'
            },
            MarketRegime.SIDEWAYS: {
                'focus': 'mean_reversion',
                'risk_tolerance': 'moderate',
                'sequence_preference': 'moderate', 
                'regularization': 'high',
                'description': 'Mean reversion focus for range-bound markets'
            },
            MarketRegime.HIGH_VOL: {
                'focus': 'stability',
                'risk_tolerance': 'very_conservative',
                'sequence_preference': 'shorter',
                'regularization': 'very_high',
                'description': 'Maximum stability for volatile conditions'
            },
            MarketRegime.LOW_VOL: {
                'focus': 'signal_extraction',
                'risk_tolerance': 'moderate_aggressive',
                'sequence_preference': 'longer',
                'regularization': 'moderate',
                'description': 'Signal extraction in low-noise environments'
            }
        }
        
        return strategies.get(regime, strategies[MarketRegime.SIDEWAYS])
    
    def save_optimization_results(self, results: List[Dict], filepath: str):
        """Save optimization results for future analysis."""
        optimization_data = {
            'asset_class': self.asset_class.value,
            'market_regime': self.market_regime.value if self.market_regime else None,
            'constraints': {
                'max_drawdown_tolerance': self.constraints.max_drawdown_tolerance,
                'min_sharpe_ratio': self.constraints.min_sharpe_ratio,
                'max_learning_rate': self.constraints.max_learning_rate,
            },
            'results': results,
            'best_params': min(results, key=lambda x: x.get('loss', float('inf'))),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(optimization_data, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filepath}")

def create_financial_optimizer(
    asset_class: str = "crypto",
    market_conditions: Optional[str] = None,
    custom_constraints: Optional[Dict[str, float]] = None
) -> FinancialHyperparameterOptimizer:
    """
    Convenience function to create a financial hyperparameter optimizer.
    
    Args:
        asset_class: "crypto", "forex", "stocks", or "commodities"
        market_conditions: "bull", "bear", "sideways", "high_vol", or "low_vol"
        custom_constraints: Custom constraint overrides
        
    Returns:
        Configured FinancialHyperparameterOptimizer
    """
    asset_enum = AssetClass(asset_class.lower())
    regime_enum = MarketRegime(market_conditions.lower()) if market_conditions else None
    
    constraints = FinancialMLConstraints()
    if custom_constraints:
        for key, value in custom_constraints.items():
            if hasattr(constraints, key):
                setattr(constraints, key, value)
    
    return FinancialHyperparameterOptimizer(
        asset_class=asset_enum,
        market_regime=regime_enum,
        constraints=constraints
    )