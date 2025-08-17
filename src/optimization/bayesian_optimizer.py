"""
Bayesian Optimization for Financial ML
=====================================

Implements Bayesian optimization with proper bounds and financial-aware constraints
for hyperparameter tuning of financial ML models. Uses Gaussian Process regression
to efficiently explore the hyperparameter space while respecting financial domain constraints.

Key Features:
- Financial domain-aware parameter bounds and priors
- Risk-adjusted objective functions (Sharpe ratio, drawdown, etc.)
- Constraint handling for financial ML stability requirements
- Multi-objective optimization for risk vs. return trade-offs
- Time-series aware validation and early stopping
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
from dataclasses import dataclass, field
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
import json
from pathlib import Path
import time

from .financial_hyperopt import FinancialHyperparameterOptimizer, FinancialMLConstraints, AssetClass, MarketRegime
from ..config.hierarchical_config import ConfigurationManager, ValidationLevel

logger = logging.getLogger(__name__)

@dataclass
class BayesianOptimizationResult:
    """Results from Bayesian optimization run."""
    
    best_params: Dict[str, Any]
    best_score: float
    best_metrics: Dict[str, float]
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    total_evaluations: int = 0
    convergence_iteration: int = -1
    optimization_time: float = 0.0
    gp_model: Optional[Any] = None

class FinancialBayesianOptimizer:
    """
    Bayesian optimizer specifically designed for financial ML hyperparameter tuning.
    
    Incorporates domain knowledge, financial constraints, and risk-aware objective
    functions to efficiently find optimal hyperparameters for financial models.
    """
    
    def __init__(
        self,
        financial_optimizer: FinancialHyperparameterOptimizer,
        config_manager: ConfigurationManager,
        n_initial_points: int = 10,
        n_calls: int = 100,
        random_state: int = 42,
        acquisition_function: str = "EI",  # Expected Improvement
        xi: float = 0.01,  # Exploration vs exploitation trade-off
        kappa: float = 1.96  # For Upper Confidence Bound
    ):
        """
        Initialize Bayesian optimizer for financial ML.
        
        Args:
            financial_optimizer: Financial hyperparameter optimizer
            config_manager: Configuration manager for validation
            n_initial_points: Number of initial random evaluations
            n_calls: Total number of optimization calls
            random_state: Random seed for reproducibility
            acquisition_function: "EI", "PI", "UCB", or "LCB"
            xi: Exploration parameter for EI/PI
            kappa: Confidence parameter for UCB/LCB
        """
        self.financial_optimizer = financial_optimizer
        self.config_manager = config_manager
        self.n_initial_points = n_initial_points
        self.n_calls = n_calls
        self.random_state = random_state
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.kappa = kappa
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_state)
        
        # Optimization state
        self.X_observed = []  # Parameter vectors
        self.y_observed = []  # Objective values
        self.additional_metrics = []  # Additional metrics for analysis
        
        # Gaussian Process model
        self.gp_model = None
        self._initialize_gaussian_process()
        
        # Parameter space definition
        self.param_bounds = {}
        self.param_types = {}
        self._initialize_parameter_space()
        
        logger.info(f"FinancialBayesianOptimizer initialized with {n_calls} evaluations")
        logger.info(f"Using {acquisition_function} acquisition function")
    
    def _initialize_gaussian_process(self):
        """Initialize Gaussian Process regression model with financial-appropriate kernel."""
        # Use Matern kernel which is good for financial optimization landscapes
        # nu=2.5 provides a good balance between smoothness and flexibility
        kernel = C(1.0, (1e-3, 1e3)) * Matern(1.0, (1e-2, 1e2), nu=2.5)
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,  # Small noise term for numerical stability
            normalize_y=True,  # Normalize objective values
            n_restarts_optimizer=5,  # Multiple restarts for robust fitting
            random_state=self.random_state
        )
        
        logger.info("Gaussian Process model initialized with Matern kernel")
    
    def _initialize_parameter_space(self):
        """Initialize parameter space with financial ML appropriate bounds."""
        param_ranges = self.financial_optimizer.param_ranges
        
        for param_name, param_config in param_ranges.items():
            param_type = param_config['type']
            self.param_types[param_name] = param_type
            
            if param_type == 'uniform':
                self.param_bounds[param_name] = (param_config['low'], param_config['high'])
            elif param_type == 'log_uniform':
                # Convert to log space for optimization
                self.param_bounds[param_name] = (
                    np.log10(param_config['low']),
                    np.log10(param_config['high'])
                )
            elif param_type == 'choice':
                # Convert choices to continuous indices
                choices = param_config['choices']
                self.param_bounds[param_name] = (0, len(choices) - 1)
                # Store choices for later conversion
                param_config['_choice_list'] = choices
        
        logger.info(f"Initialized parameter space with {len(self.param_bounds)} parameters")
    
    def _encode_parameters(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode parameter dictionary to continuous vector for GP."""
        param_vector = []
        
        for param_name in sorted(self.param_bounds.keys()):
            value = params[param_name]
            param_config = self.financial_optimizer.param_ranges[param_name]
            param_type = param_config['type']
            
            if param_type == 'uniform':
                param_vector.append(float(value))
            elif param_type == 'log_uniform':
                param_vector.append(np.log10(float(value)))
            elif param_type == 'choice':
                choices = param_config['_choice_list']
                idx = choices.index(value)
                param_vector.append(float(idx))
        
        return np.array(param_vector)
    
    def _decode_parameters(self, param_vector: np.ndarray) -> Dict[str, Any]:
        """Decode continuous vector back to parameter dictionary."""
        params = {}
        
        for i, param_name in enumerate(sorted(self.param_bounds.keys())):
            value = param_vector[i]
            param_config = self.financial_optimizer.param_ranges[param_name]
            param_type = param_config['type']
            
            if param_type == 'uniform':
                params[param_name] = float(value)
            elif param_type == 'log_uniform':
                params[param_name] = float(10 ** value)
            elif param_type == 'choice':
                choices = param_config['_choice_list']
                idx = int(np.round(np.clip(value, 0, len(choices) - 1)))
                params[param_name] = choices[idx]
        
        return params
    
    def _sample_random_parameters(self) -> Dict[str, Any]:
        """Sample random parameters from the defined space."""
        param_vector = []
        
        for param_name in sorted(self.param_bounds.keys()):
            low, high = self.param_bounds[param_name]
            
            # Add prior-based sampling if specified
            param_config = self.financial_optimizer.param_ranges[param_name]
            prior = param_config.get('prior', 'uniform')
            
            if prior == 'prefer_conservative':
                # Bias towards lower values (more conservative)
                beta_sample = self.rng.beta(2, 5)  # Skewed towards 0
                value = low + beta_sample * (high - low)
            elif prior == 'prefer_high':
                # Bias towards higher values
                beta_sample = self.rng.beta(5, 2)  # Skewed towards 1
                value = low + beta_sample * (high - low)
            elif prior == 'prefer_moderate':
                # Bias towards middle values
                beta_sample = self.rng.beta(3, 3)  # Bell-shaped around 0.5
                value = low + beta_sample * (high - low)
            else:
                # Uniform sampling
                value = self.rng.uniform(low, high)
            
            param_vector.append(value)
        
        return self._decode_parameters(np.array(param_vector))
    
    def _acquisition_function(self, X: np.ndarray, gp: GaussianProcessRegressor) -> np.ndarray:
        """
        Calculate acquisition function values for candidate points.
        
        Args:
            X: Candidate parameter vectors (n_points, n_params)
            gp: Fitted Gaussian Process model
            
        Returns:
            Acquisition values for each candidate point
        """
        if len(self.y_observed) == 0:
            # No observations yet, return uniform acquisition
            return np.ones(X.shape[0])
        
        # Get GP predictions
        mu, sigma = gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1).flatten()
        
        # Handle numerical issues
        sigma = np.maximum(sigma, 1e-10)
        
        # Current best observed value
        f_best = np.min(self.y_observed)
        
        if self.acquisition_function == "EI":
            # Expected Improvement
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Z = (f_best - mu - self.xi) / sigma
                acquisition = (f_best - mu - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
                acquisition[sigma == 0.0] = 0.0
                
        elif self.acquisition_function == "PI":
            # Probability of Improvement
            Z = (f_best - mu - self.xi) / sigma
            acquisition = norm.cdf(Z)
            
        elif self.acquisition_function == "UCB":
            # Upper Confidence Bound (for minimization, we use Lower Confidence Bound)
            acquisition = -(mu - self.kappa * sigma)  # Negative because we minimize
            
        elif self.acquisition_function == "LCB":
            # Lower Confidence Bound
            acquisition = mu - self.kappa * sigma
            
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
        
        return acquisition
    
    def _suggest_next_point(self) -> Dict[str, Any]:
        """Suggest next parameter configuration to evaluate."""
        if len(self.X_observed) < self.n_initial_points:
            # Random sampling for initial points
            return self._sample_random_parameters()
        
        # Fit GP model on observed data
        X_array = np.array(self.X_observed)
        y_array = np.array(self.y_observed)
        
        self.gp_model.fit(X_array, y_array)
        
        # Optimize acquisition function
        bounds = [self.param_bounds[param] for param in sorted(self.param_bounds.keys())]
        
        # Multi-start optimization for robust acquisition optimization
        best_acquisition = -np.inf
        best_point = None
        
        n_restarts = 10
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.array([self.rng.uniform(low, high) for low, high in bounds])
            
            # Optimize acquisition function
            try:
                result = minimize(
                    lambda x: -self._acquisition_function(x.reshape(1, -1), self.gp_model)[0],
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                if result.success and -result.fun > best_acquisition:
                    best_acquisition = -result.fun
                    best_point = result.x
                    
            except Exception as e:
                logger.warning(f"Acquisition optimization failed: {e}")
                continue
        
        if best_point is None:
            # Fallback to random sampling
            logger.warning("Acquisition optimization failed, falling back to random sampling")
            return self._sample_random_parameters()
        
        return self._decode_parameters(best_point)
    
    def _apply_financial_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply financial ML constraints to ensure stable parameters."""
        constraints = self.financial_optimizer.constraints
        
        # Learning rate constraints
        if 'learning_rate' in params:
            params['learning_rate'] = np.clip(
                params['learning_rate'], 
                1e-6, 
                constraints.max_learning_rate
            )
        
        # Regularization constraints
        if 'dropout' in params:
            params['dropout'] = np.clip(
                params['dropout'],
                constraints.min_regularization,
                0.8
            )
        
        # Architecture constraints
        if 'hidden_size' in params and 'num_layers' in params:
            # Prevent overly complex models
            total_params = params['hidden_size'] * params['num_layers']
            if total_params > 1000:  # Arbitrary complexity limit
                # Reduce hidden size if too complex
                params['hidden_size'] = min(params['hidden_size'], 128)
        
        # Force financial ML critical settings
        params['mixed_precision'] = False
        params['deterministic'] = True
        
        return params
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], Dict[str, float]],
        save_progress: bool = True,
        progress_file: Optional[str] = None
    ) -> BayesianOptimizationResult:
        """
        Run Bayesian optimization for hyperparameter tuning.
        
        Args:
            objective_function: Function that takes parameters and returns metrics dict
            save_progress: Whether to save optimization progress
            progress_file: Path to save progress (auto-generated if None)
            
        Returns:
            BayesianOptimizationResult with optimization results
        """
        logger.info(f"Starting Bayesian optimization with {self.n_calls} evaluations")
        start_time = time.time()
        
        optimization_history = []
        best_score = float('inf')
        best_params = None
        best_metrics = {}
        convergence_iteration = -1
        
        for iteration in range(self.n_calls):
            # Suggest next parameter configuration
            suggested_params = self._suggest_next_point()
            
            # Apply financial constraints
            constrained_params = self._apply_financial_constraints(suggested_params)
            
            # Validate parameters
            try:
                # Create temporary config for validation
                self.config_manager.apply_manual_override({'gru': constrained_params})
                config = self.config_manager.get_merged_config()
                is_valid, messages = self.config_manager.validate_config(config)
                
                if not is_valid:
                    logger.warning(f"Invalid parameters at iteration {iteration}: {messages}")
                    # Skip this iteration and try again
                    continue
                    
            except Exception as e:
                logger.warning(f"Parameter validation failed at iteration {iteration}: {e}")
                continue
            
            # Evaluate objective function
            try:
                logger.info(f"Iteration {iteration + 1}/{self.n_calls}: Evaluating parameters...")
                logger.info(f"Parameters: {constrained_params}")
                
                result = objective_function(constrained_params)
                
                if result.get('status') == 'FAIL':
                    logger.warning(f"Objective evaluation failed at iteration {iteration}")
                    continue
                
                score = result['loss']
                
                # Store observation
                param_vector = self._encode_parameters(constrained_params)
                self.X_observed.append(param_vector)
                self.y_observed.append(score)
                self.additional_metrics.append(result)
                
                # Update best result
                if score < best_score:
                    best_score = score
                    best_params = constrained_params.copy()
                    best_metrics = result.copy()
                    convergence_iteration = iteration
                    logger.info(f"New best score: {best_score:.6f}")
                
                # Log progress
                iteration_result = {
                    'iteration': iteration,
                    'parameters': constrained_params,
                    'score': score,
                    'metrics': result,
                    'is_best': score == best_score
                }
                optimization_history.append(iteration_result)
                
                logger.info(f"Iteration {iteration + 1} complete - Score: {score:.6f}")
                
                # Save progress periodically
                if save_progress and (iteration + 1) % 5 == 0:
                    if progress_file is None:
                        progress_file = f"bayesian_opt_progress_{int(time.time())}.json"
                    
                    self._save_progress(optimization_history, progress_file)
                
            except Exception as e:
                logger.error(f"Error evaluating objective at iteration {iteration}: {e}")
                continue
        
        optimization_time = time.time() - start_time
        
        # Create final result
        result = BayesianOptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            best_metrics=best_metrics,
            optimization_history=optimization_history,
            total_evaluations=len(optimization_history),
            convergence_iteration=convergence_iteration,
            optimization_time=optimization_time,
            gp_model=self.gp_model
        )
        
        logger.info(f"Bayesian optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Best score: {best_score:.6f} (iteration {convergence_iteration + 1})")
        logger.info(f"Total successful evaluations: {len(optimization_history)}")
        
        return result
    
    def _save_progress(self, optimization_history: List[Dict], filepath: str):
        """Save optimization progress to file."""
        progress_data = {
            'optimization_history': optimization_history,
            'best_result': min(optimization_history, key=lambda x: x['score']) if optimization_history else None,
            'total_evaluations': len(optimization_history),
            'optimizer_config': {
                'n_calls': self.n_calls,
                'n_initial_points': self.n_initial_points,
                'acquisition_function': self.acquisition_function,
                'random_state': self.random_state
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)
        
        logger.debug(f"Progress saved to {filepath}")
    
    def get_convergence_analysis(self, result: BayesianOptimizationResult) -> Dict[str, Any]:
        """Analyze convergence characteristics of the optimization."""
        if not result.optimization_history:
            return {'error': 'No optimization history available'}
        
        scores = [entry['score'] for entry in result.optimization_history]
        
        # Calculate cumulative best scores
        cumulative_best = []
        current_best = float('inf')
        for score in scores:
            if score < current_best:
                current_best = score
            cumulative_best.append(current_best)
        
        # Find improvement points
        improvement_iterations = []
        for i, (score, cum_best) in enumerate(zip(scores, cumulative_best)):
            if i == 0 or cum_best < cumulative_best[i-1]:
                improvement_iterations.append(i)
        
        # Calculate convergence metrics
        initial_score = scores[0]
        final_score = result.best_score
        improvement = (initial_score - final_score) / abs(initial_score) if initial_score != 0 else 0
        
        # Estimate convergence point
        if len(improvement_iterations) > 1:
            convergence_point = improvement_iterations[-1]
            stagnation_length = len(scores) - convergence_point
        else:
            convergence_point = 0
            stagnation_length = len(scores)
        
        return {
            'total_evaluations': len(scores),
            'improvement_iterations': improvement_iterations,
            'convergence_iteration': convergence_point,
            'stagnation_length': stagnation_length,
            'relative_improvement': improvement,
            'optimization_efficiency': len(improvement_iterations) / len(scores),
            'final_score': final_score,
            'score_reduction': initial_score - final_score
        }

def create_financial_bayesian_optimizer(
    asset_class: str = "crypto",
    market_regime: Optional[str] = None,
    n_calls: int = 50,
    acquisition_function: str = "EI",
    random_state: int = 42
) -> FinancialBayesianOptimizer:
    """
    Create a Bayesian optimizer configured for financial ML.
    
    Args:
        asset_class: "crypto", "forex", "stocks", or "commodities"
        market_regime: Optional market regime for specialized optimization
        n_calls: Number of optimization evaluations
        acquisition_function: "EI", "PI", "UCB", or "LCB"
        random_state: Random seed for reproducibility
        
    Returns:
        Configured FinancialBayesianOptimizer
    """
    from .financial_hyperopt import create_financial_optimizer
    from ..config.hierarchical_config import create_financial_config_manager
    
    # Create components
    financial_opt = create_financial_optimizer(asset_class, market_regime)
    config_manager = create_financial_config_manager(validation_level="financial")
    
    # Create Bayesian optimizer
    bayesian_opt = FinancialBayesianOptimizer(
        financial_optimizer=financial_opt,
        config_manager=config_manager,
        n_calls=n_calls,
        acquisition_function=acquisition_function,
        random_state=random_state
    )
    
    return bayesian_opt