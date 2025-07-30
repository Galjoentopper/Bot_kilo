"""
Trading Environment Module
==========================

Gym-compatible trading environment for reinforcement learning.
Simulates realistic trading conditions with transaction costs and slippage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import logging

# Type checking imports
if TYPE_CHECKING:
    try:
        import gymnasium as gym
        BaseEnvType = gym.Env
    except ImportError:
        try:
            import gym  # type: ignore
            BaseEnvType = gym.Env  # type: ignore
        except ImportError:
            BaseEnvType = object
else:
    BaseEnvType = object

# Runtime imports with fallback
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
    BaseEnv = gym.Env
except ImportError:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
        GYM_AVAILABLE = True
        BaseEnv = gym.Env  # type: ignore
    except ImportError:
        # Create dummy classes for type hints when gym is not available
        class _DummyBox:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
        
        class _DummySpaces:
            Box = _DummyBox
        
        class _DummyEnv:
            def __init__(self):
                pass
            def reset(self):
                return None
            def step(self, action):
                return None, 0, False, {}
            def render(self, mode='human'):
                pass
            def close(self):
                pass
        
        class _DummyGym:
            Env = _DummyEnv
        
        spaces = _DummySpaces()  # type: ignore
        gym = _DummyGym()  # type: ignore
        BaseEnv = _DummyEnv
        GYM_AVAILABLE = False

logger = logging.getLogger(__name__)

class TradingEnvironment(BaseEnvType):  # type: ignore
    """
    Gym-compatible trading environment for crypto trading.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.1,
        lookback_window: int = 20,
        reward_scaling: float = 1.0
    ):
        """
        Initialize trading environment.
        
        Args:
            data: Market data with features
            initial_balance: Starting balance
            transaction_fee: Transaction fee rate
            slippage: Slippage rate
            max_position_size: Maximum position size as fraction of balance
            lookback_window: Number of past observations to include in state
            reward_scaling: Scaling factor for rewards
        """
        # Initialize gym environment
        if GYM_AVAILABLE:
            super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Current position size (-1 to 1, where 1 is max long)
        self.total_trades = 0
        self.total_fees_paid = 0.0
        
        # Performance tracking
        self.portfolio_values = []
        self.trades_history = []
        self.rewards_history = []
        
        # Action space: [position_change] where position_change is in [-1, 1]
        # -1 = sell all, 0 = hold, 1 = buy max
        if GYM_AVAILABLE:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
            
            # Observation space: [market_features, portfolio_state]
            n_features = len(self.data.columns) - 1  # Exclude 'close' price
            portfolio_features = 3  # balance, position, unrealized_pnl
            
            # Use reasonable bounds instead of infinity to prevent NaN issues
            self.observation_space = spaces.Box(
                low=-100.0,
                high=100.0,
                shape=(self.lookback_window, n_features + portfolio_features),
                dtype=np.float32
            )
        else:
            # Dummy spaces when gym is not available
            self.action_space = None
            self.observation_space = None
        
        # Feature columns (exclude 'close' for features, but keep for price calculation)
        self.feature_columns = [col for col in self.data.columns if col != 'close']
        
        logger.info(f"Trading environment initialized with {len(self.data)} steps")
        if GYM_AVAILABLE and self.action_space is not None:
            logger.info(f"Action space: {self.action_space}")
            if self.observation_space is not None and hasattr(self.observation_space, 'shape'):
                logger.info(f"Observation space: {self.observation_space.shape}")
        else:
            logger.info("Gym not available - running in compatibility mode")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed (for gymnasium compatibility)
            options: Additional options (for gymnasium compatibility)
        
        Returns:
            Tuple of (initial observation, info dict)
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_trades = 0
        self.total_fees_paid = 0.0
        
        # Clear history
        self.portfolio_values = []
        self.trades_history = []
        self.rewards_history = []
        
        observation = self._get_observation()
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_step': self.current_step
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take [position_change]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, False, {}
        
        # Extract action
        position_change = np.clip(action[0], -1.0, 1.0)
        
        # Calculate new position
        new_position = np.clip(
            self.position + position_change, 
            -self.max_position_size, 
            self.max_position_size
        )
        
        # Execute trade if position changes
        reward = 0.0
        if abs(new_position - self.position) > 1e-6:
            reward = self._execute_trade(new_position)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value and unrealized PnL
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self._calculate_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)
        
        # Get next observation
        observation = self._get_observation()
        
        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = portfolio_value <= self.initial_balance * 0.1  # Stop if 90% loss
        
        # Additional info
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees_paid,
            'current_price': current_price
        }
        
        self.rewards_history.append(reward)
        
        return observation, reward * self.reward_scaling, terminated, truncated, info
    
    def _execute_trade(self, new_position: float) -> float:
        """
        Execute a trade and calculate reward.
        
        Args:
            new_position: New position size
            
        Returns:
            Reward from the trade
        """
        current_price = self.data.iloc[self.current_step]['close']
        position_change = new_position - self.position
        
        # Calculate trade size in currency units
        trade_size = abs(position_change) * self.balance
        
        # Apply transaction costs
        fee = trade_size * self.transaction_fee
        slippage_cost = trade_size * self.slippage
        total_cost = fee + slippage_cost
        
        # Update balance and position
        self.balance -= total_cost
        self.total_fees_paid += total_cost
        self.total_trades += 1
        
        # Record trade
        self.trades_history.append({
            'step': self.current_step,
            'price': current_price,
            'position_change': position_change,
            'new_position': new_position,
            'cost': total_cost
        })
        
        # Update position
        old_position = self.position
        self.position = new_position
        
        # Calculate immediate reward based on price movement prediction
        if self.current_step < len(self.data) - 1:
            next_price = self.data.iloc[self.current_step + 1]['close']
            price_change = (next_price - current_price) / current_price
            
            # Reward is based on position alignment with price movement
            position_reward = self.position * price_change * self.balance
            
            # Penalty for transaction costs
            cost_penalty = -total_cost
            
            # Small penalty for excessive trading
            trading_penalty = -abs(position_change) * 0.001 * self.balance
            
            total_reward = position_reward + cost_penalty + trading_penalty
        else:
            total_reward = -total_cost
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation with proper data validation and normalization.
        
        Returns:
            Current state observation
        """
        # Get market features for lookback window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        market_data = self.data.iloc[start_idx:end_idx][self.feature_columns].values
        
        # Pad if necessary
        if len(market_data) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(market_data), len(self.feature_columns)))
            market_data = np.vstack([padding, market_data])
        
        # Clean market data - replace NaN and inf values
        market_data = np.nan_to_num(market_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize market data to prevent extreme values
        # Use robust scaling to handle outliers
        market_data = np.clip(market_data, -10.0, 10.0)
        
        # Portfolio state features
        current_price = self.data.iloc[self.current_step]['close']
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        
        # Ensure portfolio features are valid
        balance_ratio = np.clip(self.balance / self.initial_balance, 0.01, 10.0)
        position_ratio = np.clip(self.position, -1.0, 1.0)
        pnl_ratio = np.clip(unrealized_pnl / self.initial_balance, -2.0, 2.0)
        
        portfolio_features = np.array([
            balance_ratio,  # Normalized balance (clipped)
            position_ratio,  # Current position (clipped)
            pnl_ratio  # Normalized unrealized PnL (clipped)
        ])
        
        # Ensure no NaN or inf values in portfolio features
        portfolio_features = np.nan_to_num(portfolio_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Repeat portfolio features for each timestep in lookback window
        portfolio_matrix = np.tile(portfolio_features, (self.lookback_window, 1))
        
        # Combine market and portfolio features
        observation = np.hstack([market_data, portfolio_matrix])
        
        # Final validation - ensure all values are finite and within bounds
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        observation = np.clip(observation, -100.0, 100.0)
        
        return observation.astype(np.float32)
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            current_price: Current asset price
            
        Returns:
            Total portfolio value
        """
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        return self.balance + unrealized_pnl
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized profit/loss with proper validation.
        
        Args:
            current_price: Current asset price
            
        Returns:
            Unrealized PnL
        """
        if self.position == 0 or current_price <= 0:
            return 0.0
        
        # For simplicity, assume average entry price is tracked
        # In a more sophisticated implementation, you'd track actual entry prices
        position_value = abs(self.position) * self.balance
        
        # Estimate PnL based on position and recent price movement
        if len(self.trades_history) > 0:
            last_trade_price = self.trades_history[-1]['price']
            if last_trade_price > 0:
                price_change = (current_price - last_trade_price) / last_trade_price
                # Clip price change to prevent extreme values
                price_change = np.clip(price_change, -0.5, 0.5)
                unrealized_pnl = self.position * price_change * position_value
            else:
                unrealized_pnl = 0.0
        else:
            unrealized_pnl = 0.0
        
        # Ensure the result is finite
        unrealized_pnl = np.nan_to_num(unrealized_pnl, nan=0.0, posinf=0.0, neginf=0.0)
        
        return float(unrealized_pnl)
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """
        Calculate portfolio performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        if len(self.portfolio_values) == 0:
            return {}
        
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - self.initial_balance) / self.initial_balance
        
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 4)  # Annualized for 15-min data
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            volatility = np.std(returns) * np.sqrt(252 * 24 * 4)  # Annualized
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            volatility = 0.0
        
        # Win rate
        if len(self.trades_history) > 0:
            profitable_trades = sum(1 for trade in self.trades_history if trade.get('pnl', 0) > 0)
            win_rate = profitable_trades / len(self.trades_history)
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees_paid,
            'final_balance': self.balance
        }
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_values: Array of portfolio values
            
        Returns:
            Maximum drawdown as a fraction
        """
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return float(abs(np.min(drawdown)))
    
    def render(self, mode: str = 'human'):
        """
        Render the environment (optional).
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            current_price = self.data.iloc[self.current_step]['close']
            portfolio_value = self._calculate_portfolio_value(current_price)
            
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.4f}")
            print(f"Position: {self.position:.4f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Portfolio Value: {portfolio_value:.2f}")
            print(f"Total Trades: {self.total_trades}")
            print("-" * 40)
    
    def close(self):
        """Close the environment."""
        pass

class MultiAssetTradingEnvironment(TradingEnvironment):
    """
    Extended trading environment for multiple assets.
    """
    
    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        **kwargs
    ):
        """
        Initialize multi-asset trading environment.
        
        Args:
            data_dict: Dictionary mapping asset names to DataFrames
            **kwargs: Additional arguments for base class
        """
        self.assets = list(data_dict.keys())
        self.data_dict = data_dict
        
        # Combine all asset data (simplified approach)
        # In practice, you might want more sophisticated handling
        combined_data = pd.concat([
            df.add_suffix(f'_{asset}') for asset, df in data_dict.items()
        ], axis=1)
        
        super().__init__(combined_data, **kwargs)
        
        # Override action space for multiple assets
        # Action: [position_change_asset1, position_change_asset2, ...]
        if GYM_AVAILABLE:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(len(self.assets),),
                dtype=np.float32
            )
        else:
            self.action_space = None
        
        # Track positions for each asset
        self.positions = {asset: 0.0 for asset in self.assets}
        
        logger.info(f"Multi-asset trading environment initialized with {len(self.assets)} assets")
    
    def _execute_trade_multi(self, action: np.ndarray) -> float:
        """
        Execute trades for multiple assets.
        
        Args:
            action: Array of position changes for each asset
            
        Returns:
            Total reward from all trades
        """
        total_reward = 0.0
        
        for i, asset in enumerate(self.assets):
            position_change = np.clip(action[i], -1.0, 1.0)
            new_position = np.clip(
                self.positions[asset] + position_change,
                -self.max_position_size,
                self.max_position_size
            )
            
            if abs(new_position - self.positions[asset]) > 1e-6:
                # Execute trade for this asset
                current_price = self.data_dict[asset].iloc[self.current_step]['close']
                trade_size = abs(position_change) * self.balance / len(self.assets)
                
                # Apply costs
                fee = trade_size * self.transaction_fee
                slippage_cost = trade_size * self.slippage
                total_cost = fee + slippage_cost
                
                self.balance -= total_cost
                self.total_fees_paid += total_cost
                self.total_trades += 1
                
                # Calculate reward
                if self.current_step < len(self.data_dict[asset]) - 1:
                    next_price = self.data_dict[asset].iloc[self.current_step + 1]['close']
                    price_change = (next_price - current_price) / current_price
                    position_reward = new_position * price_change * trade_size
                    total_reward += position_reward - total_cost
                
                self.positions[asset] = new_position
        
        return total_reward