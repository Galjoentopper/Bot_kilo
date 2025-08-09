"""
PPO Trainer Module
==================

PPO (Proximal Policy Optimization) trainer for reinforcement learning trading agent.
Optimized for GPU training on Paperspace Gradient.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
from datetime import datetime
import torch
import torch.nn as nn

try:
    from stable_baselines3 import PPO as SB3_PPO  # type: ignore[import-untyped]
    from stable_baselines3.common.env_util import make_vec_env  # type: ignore[import-untyped]
    from stable_baselines3.common.callbacks import BaseCallback as SB3_BaseCallback, EvalCallback as SB3_EvalCallback  # type: ignore[import-untyped]
    from stable_baselines3.common.monitor import Monitor as SB3_Monitor  # type: ignore[import-untyped]
    from stable_baselines3.common.vec_env import DummyVecEnv as SB3_DummyVecEnv, SubprocVecEnv  # type: ignore[import-untyped]
    SB3_AVAILABLE = True
    
except ImportError:
    # Create dummy classes for type hints when SB3 is not available
    class SB3_BaseCallback:
        def __init__(self, verbose: int = 0):
            self.verbose = verbose
            self.locals: Optional[Dict[str, Any]] = None
            self.num_timesteps: int = 0
            
        def _on_step(self) -> bool:
            return True
    
    class SB3_PPO:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
            
        def learn(self, *args: Any, **kwargs: Any) -> 'SB3_PPO':
            return self
            
        def predict(self, observation: Any, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
            return np.array([0]), None
            
        def save(self, path: str) -> None:
            pass
            
        @classmethod
        def load(cls, path: str, **kwargs: Any) -> 'SB3_PPO':
            return cls()
    
    class SB3_EvalCallback:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
    
    class SB3_Monitor:
        def __init__(self, env: Any, *args: Any, **kwargs: Any) -> None:
            self.env = env
            
        def __getattr__(self, name: str) -> Any:
            return getattr(self.env, name)
    
    class SB3_DummyVecEnv:
        def __init__(self, env_fns: List[Any]) -> None:
            self.env_fns = env_fns
    
    class SubprocVecEnv:
        def __init__(self, env_fns: List[Any]) -> None:
            self.env_fns = env_fns
    
    SB3_AVAILABLE = False

# Create aliases for consistent usage
BaseCallback = SB3_BaseCallback
PPO = SB3_PPO
EvalCallback = SB3_EvalCallback
Monitor = SB3_Monitor
DummyVecEnv = SB3_DummyVecEnv
# SubprocVecEnv is already imported directly when SB3 is available
# When SB3 is not available, we use our dummy implementation

try:
    import mlflow  # type: ignore[import-untyped]
    import mlflow.pytorch  # type: ignore
    MLFLOW_AVAILABLE = True
except ImportError:
    # Create dummy mlflow module for type hints
    class _DummyMLflow:
        @staticmethod
        def start_run(*args: Any, **kwargs: Any) -> Any:
            from contextlib import nullcontext
            return nullcontext()
        
        @staticmethod
        def log_params(params: Dict[str, Any]) -> None:
            pass
            
        @staticmethod
        def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
            pass
            
        @staticmethod
        def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
            pass
    
    mlflow = _DummyMLflow()  # type: ignore
    MLFLOW_AVAILABLE = False

from ..rl_env.trading_env import TradingEnvironment

logger = logging.getLogger(__name__)

class TradingCallback(BaseCallback):
    """
    Custom callback for monitoring trading performance during training.
    Optimized to reduce synchronization overhead by limiting logging frequency.
    """
    
    def __init__(self, eval_freq: int = 1000, log_freq: int = 10000, verbose: int = 0):
        if SB3_AVAILABLE:
            super(TradingCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.log_freq = log_freq  # Frequency of logging operations to reduce overhead
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_log_step = 0  # Track last step when logging occurred
        
    def _on_step(self) -> bool:
        if not SB3_AVAILABLE:
            return True
            
        # Only process logging at specified frequency to reduce synchronization overhead
        current_step = getattr(self, 'num_timesteps', 0)
        if current_step - self.last_log_step < self.log_freq:
            return True  # Skip logging if not enough steps have passed
            
        # Update last log step
        self.last_log_step = current_step
            
        # Log episode rewards and lengths (less frequently now)
        try:
            if hasattr(self, 'locals') and self.locals and len(self.locals.get('infos', [])) > 0:
                for info in self.locals['infos']:
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
                        
                        if MLFLOW_AVAILABLE and mlflow is not None:
                            mlflow.log_metric("episode_reward", info['episode']['r'], step=current_step)
                            mlflow.log_metric("episode_length", info['episode']['l'], step=current_step)
        except Exception:
            pass  # Ignore logging errors
        
        return True

class PPOTrainer:
    """
    PPO trainer for trading agent with GPU optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PPO trainer.
        
        Args:
            config: Configuration dictionary
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PPO training")
        
        self.config = config
        self.model_config = config.get('models', {}).get('ppo', {})
        self.training_config = config.get('training', {})
        
        # PPO parameters
        self.learning_rate = self.model_config.get('learning_rate', 0.0003)
        self.n_steps = self.model_config.get('n_steps', 4096)
        self.batch_size = self.model_config.get('batch_size', 256)
        self.n_epochs = self.model_config.get('n_epochs', 20)
        self.gamma = self.model_config.get('gamma', 0.99)
        self.gae_lambda = self.model_config.get('gae_lambda', 0.95)
        self.clip_range = self.model_config.get('clip_range', 0.2)
        self.ent_coef = self.model_config.get('ent_coef', 0.01)
        self.vf_coef = self.model_config.get('vf_coef', 0.5)
        
        # Training settings
        self.total_timesteps = 100000
        self.eval_freq = 5000
        self.n_eval_episodes = 10
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model and environment
        self.model = None
        self.env = None
        self.eval_env = None
        
        logger.info(f"PPO Trainer initialized with device: {self.device}")
    
    def create_environment(
        self,
        train_data: pd.DataFrame,
        eval_data: Optional[pd.DataFrame] = None,
        env_kwargs: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Optional[Any]]:
        """
        Create training and evaluation environments with data validation.
        
        Args:
            train_data: Training data
            eval_data: Evaluation data (optional)
            env_kwargs: Additional environment arguments
            
        Returns:
            Tuple of (train_env, eval_env)
        """
        if env_kwargs is None:
            env_kwargs = {}
        
        # Validate training data
        self._validate_data(train_data, "training")
        if eval_data is not None:
            self._validate_data(eval_data, "evaluation")
        
        # Default environment parameters
        default_kwargs = {
            'initial_balance': 10000.0,
            'transaction_fee': 0.001,
            'slippage': 0.0005,
            'max_position_size': 0.1,
            'lookback_window': 20
        }
        if env_kwargs:
            default_kwargs.update(env_kwargs)
        
        # Create training environment
        def make_train_env(*args, **kwargs):
            env = TradingEnvironment(train_data, **default_kwargs)
            env = Monitor(env)
            return env
        
        # Use vectorized environment for better performance
        # Create multiple environments for parallelization
        n_envs = 8  # Number of parallel environments
        
        # For SubprocVecEnv, we need to create the environment creation function
        # in a way that's picklable
        def make_env_fn(data, kwargs):
            def _init(*args, **env_kwargs):
                env = TradingEnvironment(data, **kwargs)
                env = Monitor(env)
                return env
            return _init
        
        # Create environment functions that are picklable
        env_fns = [make_env_fn(train_data, default_kwargs) for _ in range(n_envs)]
        self.env = SubprocVecEnv(env_fns)
        
        # Create evaluation environment if eval data provided
        if eval_data is not None:
            def make_eval_env(*args, **kwargs):
                env = TradingEnvironment(eval_data, **default_kwargs)
                env = Monitor(env)
                return env
            
            # For evaluation, we typically use a single environment
            self.eval_env = DummyVecEnv([make_eval_env])
        else:
            self.eval_env = None
        
        logger.info("Training and evaluation environments created")
        
        return self.env, self.eval_env
    
    def _validate_data(self, data: pd.DataFrame, data_type: str) -> None:
        """
        Validate data for NaN and infinite values.
        
        Args:
            data: Data to validate
            data_type: Type of data (for logging)
        """
        # Check for NaN values
        nan_count = data.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"{data_type} data contains {nan_count} NaN values")
            # Fill NaN values with forward fill, then backward fill
            data.ffill(inplace=True)
            data.bfill(inplace=True)
            # If still NaN, fill with 0
            data.fillna(0, inplace=True)
        
        # Check for infinite values
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning(f"{data_type} data contains {inf_count} infinite values")
            # Replace infinite values with large but finite numbers
            data.replace([np.inf, -np.inf], [1e6, -1e6], inplace=True)
        
        # Ensure 'close' column exists and is valid
        if 'close' not in data.columns:
            raise ValueError(f"{data_type} data must contain 'close' column")
        
        if (data['close'] <= 0).any():
            logger.warning(f"{data_type} data contains non-positive close prices")
            # Replace non-positive prices with small positive value
            data.loc[data['close'] <= 0, 'close'] = 0.0001
        
        logger.info(f"{data_type} data validated - Shape: {data.shape}")
    
    def build_model(self, env: Any) -> Any:
        """
        Build PPO model.
        
        Args:
            env: Training environment
            
        Returns:
            PPO model
        """
        # Policy network configuration with gradient clipping
        policy_kwargs = {
            'net_arch': [dict(pi=[128, 128], vf=[128, 128])],  # Smaller network to reduce NaN risk
            'activation_fn': nn.Tanh,  # Use Tanh instead of ReLU for better gradient stability
            'ortho_init': False,  # Disable orthogonal initialization which can cause issues
        }
        
        # Create PPO model with conservative parameters
        if SB3_AVAILABLE:
            self.model = SB3_PPO(
                policy='MlpPolicy',
                env=env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=0.5,  # Add gradient clipping
                policy_kwargs=policy_kwargs,
                device=self.device,
                verbose=1,
                seed=42  # Set seed for reproducibility
            )
        else:
            self.model = PPO()  # Use dummy class
        
        logger.info("PPO model built")
        return self.model
    
    def train(
        self,
        train_data: pd.DataFrame,
        eval_data: Optional[pd.DataFrame] = None,
        total_timesteps: int = 100000,
        experiment_name: str = "ppo_trading"
    ) -> Dict[str, Any]:
        """
        Train the PPO agent.
        
        Args:
            train_data: Training data
            eval_data: Evaluation data (optional)
            total_timesteps: Total training timesteps
            experiment_name: MLflow experiment name
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting PPO agent training")
        
        # Create environments
        train_env, eval_env = self.create_environment(train_data, eval_data)
        
        # Build model
        self.build_model(train_env)
        
        # Start MLflow run (if available)
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow_context = mlflow.start_run(run_name=f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            from contextlib import nullcontext
            mlflow_context = nullcontext()
        
        with mlflow_context:
            # Log parameters (if MLflow available)
            if MLFLOW_AVAILABLE and mlflow is not None:
                mlflow.log_params({
                    "model_type": "PPO",
                    "learning_rate": self.learning_rate,
                    "n_steps": self.n_steps,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_range": self.clip_range,
                    "ent_coef": self.ent_coef,
                    "vf_coef": self.vf_coef,
                    "total_timesteps": total_timesteps,
                    "device": self.device
                })
            
            # Setup callbacks
            callbacks: List[Any] = [TradingCallback(eval_freq=1000, log_freq=10000)]
            
            if eval_env is not None and SB3_AVAILABLE:
                eval_callback = SB3_EvalCallback(
                    eval_env,
                    best_model_save_path=f"./models/ppo_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    log_path="./logs/",
                    eval_freq=self.eval_freq,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True,
                    render=False
                )
                callbacks.append(eval_callback)
            
            # Train the model
            if self.model is not None:
                self.model.learn(
                    total_timesteps=total_timesteps,
                    callback=callbacks if SB3_AVAILABLE else None,
                    progress_bar=False  # Disable progress bar to avoid tqdm/rich dependency issues
                )
            
            # Save model with signature and input example (if MLflow available)
            if MLFLOW_AVAILABLE and mlflow is not None and self.model is not None and SB3_AVAILABLE:
                # Save the model
                model_path = f"ppo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.model.save(model_path)
                
                # Log model with signature and input example
                # For Stable Baselines3 models, we'll log the artifact with additional metadata
                mlflow.log_artifact(f"{model_path}.zip", "model")
                
                # Additionally, try to log with model signature if possible
                try:
                    # Create a sample input for signature inference
                    # For PPO, we need to create an observation sample
                    sample_observation = np.random.randn(1, train_env.observation_space.shape[0]).astype(np.float32)
                    
                    # Log additional model metadata
                    mlflow.log_params({
                        "observation_shape": str(train_env.observation_space.shape),
                        "action_shape": str(train_env.action_space.shape) if hasattr(train_env.action_space, 'shape') else str(train_env.action_space.n)
                    })
                except Exception:
                    # If we can't create signature, just log the artifact as before
                    pass
        
        # Training results
        results = {
            "model": self.model,
            "total_timesteps": total_timesteps,
            "training_completed": True
        }
        
        logger.info("PPO training completed")
        return results
    
    def evaluate(
        self,
        eval_data: pd.DataFrame,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the trained agent.
        
        Args:
            eval_data: Evaluation data
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating PPO agent for {n_episodes} episodes")
        
        # Create evaluation environment
        eval_env = TradingEnvironment(eval_data)
        
        episode_rewards = []
        episode_lengths = []
        portfolio_stats = []
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                if self.model is not None:
                    action, _ = self.model.predict(obs, deterministic=deterministic)
                    # Ensure action is numpy array
                    if not isinstance(action, np.ndarray):
                        action = np.array([action])
                else:
                    action = np.array([0])  # Default action as numpy array
                
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                    
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Get portfolio statistics for this episode
            stats = eval_env.get_portfolio_stats()
            portfolio_stats.append(stats)
        
        # Calculate aggregate statistics
        results = {
            "mean_episode_reward": np.mean(episode_rewards),
            "std_episode_reward": np.std(episode_rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "portfolio_stats": portfolio_stats,
            "mean_total_return": np.mean([stats.get('total_return', 0) for stats in portfolio_stats]),
            "mean_sharpe_ratio": np.mean([stats.get('sharpe_ratio', 0) for stats in portfolio_stats]),
            "mean_max_drawdown": np.mean([stats.get('max_drawdown', 0) for stats in portfolio_stats]),
            "mean_win_rate": np.mean([stats.get('win_rate', 0) for stats in portfolio_stats])
        }
        
        logger.info(f"Evaluation completed - Mean reward: {results['mean_episode_reward']:.4f}")
        logger.info(f"Mean total return: {results['mean_total_return']:.4f}")
        logger.info(f"Mean Sharpe ratio: {results['mean_sharpe_ratio']:.4f}")
        
        return results
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make prediction with trained model.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action, log_prob)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        if self.model is not None:
            self.model.save(filepath)
        logger.info(f"PPO model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, config: Dict[str, Any]) -> 'PPOTrainer':
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            config: Configuration dictionary
            
        Returns:
            Loaded PPOTrainer instance
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for PPO training")
        
        # Check if filepath already has .zip extension
        if not filepath.endswith('.zip'):
            filepath_with_extension = f"{filepath}.zip"
        else:
            filepath_with_extension = filepath
        
        if not os.path.exists(filepath_with_extension):
            raise FileNotFoundError(f"Model file not found: {filepath_with_extension}")
        
        # Create trainer instance
        trainer = cls(config)
        
        # Load model
        if SB3_AVAILABLE:
            trainer.model = SB3_PPO.load(filepath)
        else:
            trainer.model = PPO()
        
        logger.info(f"PPO model loaded from {filepath_with_extension}")
        return trainer
    
    def hyperparameter_tuning(
        self,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame,
        param_grid: Optional[Dict[str, List]] = None,
        n_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna (if available).
        
        Args:
            train_data: Training data
            eval_data: Evaluation data
            param_grid: Parameter grid for tuning
            n_trials: Number of trials
            
        Returns:
            Best parameters and results
        """
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna not available. Skipping hyperparameter tuning.")
            return {}
        
        if param_grid is None:
            param_grid = {
                'learning_rate': [1e-5, 1e-3],
                'n_steps': [1024, 2048, 4096],
                'batch_size': [32, 64, 128, 256],
                'gamma': [0.95, 0.99, 0.999],
                'clip_range': [0.1, 0.2, 0.3]
            }
        
        def objective(trial):
            # Sample hyperparameters
            learning_rate = trial.suggest_float('learning_rate', *param_grid['learning_rate'])
            n_steps = trial.suggest_categorical('n_steps', param_grid['n_steps'])
            batch_size = trial.suggest_categorical('batch_size', param_grid['batch_size'])
            gamma = trial.suggest_categorical('gamma', param_grid['gamma'])
            clip_range = trial.suggest_categorical('clip_range', param_grid['clip_range'])
            
            # Update parameters
            self.learning_rate = learning_rate
            self.n_steps = n_steps
            self.batch_size = batch_size
            self.gamma = gamma
            self.clip_range = clip_range
            
            # Train model with current parameters
            try:
                self.train(train_data, eval_data, total_timesteps=50000)
                
                # Evaluate model
                eval_results = self.evaluate(eval_data, n_episodes=5)
                
                # Return metric to optimize (e.g., mean Sharpe ratio)
                return eval_results['mean_sharpe_ratio']
            
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return -np.inf
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Hyperparameter tuning completed - Best value: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }