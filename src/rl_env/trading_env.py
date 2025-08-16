"""
Trading Environment Module
==========================

Gym-compatible trading environment for reinforcement learning.
Simulates realistic trading conditions with transaction costs and slippage.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """Gym-compatible environment for crypto trading."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_fee: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.1,
        lookback_window: int = 20,
        reward_scaling: float = 1.0,
        reward_mode: str = "pnl_pct",
        episode_length: Optional[int] = None,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
    ) -> None:
        super().__init__()

        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        # Data slicing for domain randomization
        self.lookback_window = int(lookback_window)
        self.data = data.copy()
        if window_start is not None or window_end is not None:
            start = max(0, int(window_start or 0))
            end = int(window_end) if window_end is not None else len(self.data)
            end = max(start + self.lookback_window + 1, min(end, len(self.data)))
            self.data = self.data.iloc[start:end].reset_index(drop=True)
            logger.debug(f"Using data window slice: start={start}, end={end}, len={len(self.data)}")

        # Params
        self.initial_balance = float(initial_balance)
        self.transaction_fee = float(transaction_fee)
        self.slippage = float(slippage)
        self.max_position_size = float(max_position_size)
        self.reward_scaling = float(reward_scaling)
        self.reward_mode = str(reward_mode)
        self.episode_length = int(episode_length) if episode_length is not None else None

        # Preprocess features
        self._preprocess_data()

        # State
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.total_trades = 0
        self.total_fees_paid = 0.0
        self.portfolio_values = []
        self.trades_history = []
        self.rewards_history = []

        # Buffers
        n_features = len(self.feature_columns)
        portfolio_features = 3
        self._observation_buffer = np.zeros((self.lookback_window, n_features + portfolio_features), dtype=np.float32)
        self._market_data_buffer = np.zeros((self.lookback_window, n_features), dtype=np.float32)
        self._portfolio_features_buffer = np.zeros(portfolio_features, dtype=np.float32)

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(self.lookback_window, n_features + portfolio_features), dtype=np.float32
        )

        logger.info(f"Trading environment initialized with {len(self.data)} steps")
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space.shape}")

    def _preprocess_data(self) -> None:
        # Feature columns (exclude 'close')
        self.feature_columns = [c for c in self.data.columns if c != 'close']
        if self.feature_columns:
            market = self.data[self.feature_columns].to_numpy(dtype=np.float32)
            market = np.nan_to_num(market, nan=0.0, posinf=1.0, neginf=-1.0)
            market = np.clip(market, -10.0, 10.0)
        else:
            market = np.zeros((len(self.data), 0), dtype=np.float32)
        self._normalized_data = market

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
        self.current_step = self.lookback_window
        self.balance = float(self.initial_balance)
        self.position = 0.0
        self.total_trades = 0
        self.total_fees_paid = 0.0
        self.portfolio_values = []
        self.trades_history = []
        self.rewards_history = []
        obs = self._get_observation()
        info: Dict[str, float] = {
            'balance': float(self.balance),
            'position': float(self.position),
            'current_step': float(self.current_step),
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0.0, True, False, {}

        position_change = float(np.clip(float(action[0]), -1.0, 1.0))
        new_position = float(np.clip(self.position + position_change, -self.max_position_size, self.max_position_size))

        reward = 0.0
        if abs(new_position - self.position) > 1e-6:
            reward = self._execute_trade(new_position)

        self.current_step += 1
        current_price = float(self.data.iloc[self.current_step]['close'])
        portfolio_value = float(self._calculate_portfolio_value(current_price))
        self.portfolio_values.append(portfolio_value)

        obs = self._get_observation()
        terminated = self.current_step >= len(self.data) - 1
        if self.episode_length is not None:
            terminated = terminated or (self.current_step - self.lookback_window >= self.episode_length)
        truncated = portfolio_value <= self.initial_balance * 0.1

        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.balance,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees_paid,
            'current_price': current_price,
        }
        self.rewards_history.append(float(reward))
        return obs, float(reward) * float(self.reward_scaling), bool(terminated), bool(truncated), info

    def _execute_trade(self, new_position: float) -> float:
        current_price = float(self.data.iloc[self.current_step]['close'])
        position_change = float(new_position - self.position)
        trade_size = abs(position_change) * float(self.balance)

        fee = trade_size * float(self.transaction_fee)
        slippage_cost = trade_size * float(self.slippage)
        total_cost = float(fee + slippage_cost)

        self.balance -= total_cost
        self.total_fees_paid += total_cost
        self.total_trades += 1

        self.trades_history.append({
            'step': int(self.current_step),
            'price': current_price,
            'position_change': position_change,
            'new_position': new_position,
            'cost': total_cost,
        })

        self.position = float(new_position)

        if self.current_step < len(self.data) - 1:
            next_price = float(self.data.iloc[self.current_step + 1]['close'])
            price_change = (next_price - current_price) / max(current_price, 1e-8)
            if self.reward_mode == 'pnl_pct':
                position_reward = float(self.position) * float(price_change)
                cost_penalty = -(total_cost / max(self.balance, 1e-8))
                trading_penalty = -abs(position_change) * 0.001
                total_reward = float(position_reward + cost_penalty + trading_penalty)
            else:
                position_reward = float(self.position) * float(price_change) * float(self.balance)
                cost_penalty = -total_cost
                trading_penalty = -abs(position_change) * 0.001 * float(self.balance)
                total_reward = float(position_reward + cost_penalty + trading_penalty)
        else:
            total_reward = -(total_cost / max(self.balance, 1e-8)) if self.reward_mode == 'pnl_pct' else -total_cost
        return float(total_reward)

    def _get_observation(self) -> np.ndarray:
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        market_data = self._normalized_data[start_idx:end_idx]

        self._market_data_buffer.fill(0.0)
        buffer_start = self.lookback_window - len(market_data)
        if market_data.shape[0] > 0:
            self._market_data_buffer[buffer_start:self.lookback_window] = market_data

        current_price = float(self.data.iloc[self.current_step]['close'])
        unrealized_pnl = float(self._calculate_unrealized_pnl(current_price))
        self._portfolio_features_buffer[0] = np.clip(self.balance / self.initial_balance, 0.01, 10.0)
        self._portfolio_features_buffer[1] = np.clip(self.position, -1.0, 1.0)
        self._portfolio_features_buffer[2] = np.clip(unrealized_pnl / self.initial_balance, -2.0, 2.0)

        portfolio_matrix = np.tile(self._portfolio_features_buffer, (self.lookback_window, 1))
        self._observation_buffer[:, : self._market_data_buffer.shape[1]] = self._market_data_buffer
        self._observation_buffer[:, self._market_data_buffer.shape[1] :] = portfolio_matrix
        return self._observation_buffer.copy()

    def _calculate_portfolio_value(self, current_price: float) -> float:
        return float(self.balance + self._calculate_unrealized_pnl(current_price))

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        if self.position == 0 or current_price <= 0:
            return 0.0
        position_value = abs(self.position) * float(self.balance)
        if len(self.trades_history) > 0:
            last_trade_price = float(self.trades_history[-1]['price'])
            if last_trade_price > 0:
                price_change = np.clip((current_price - last_trade_price) / last_trade_price, -0.5, 0.5)
                unrealized_pnl = float(self.position) * float(price_change) * float(position_value)
            else:
                unrealized_pnl = 0.0
        else:
            unrealized_pnl = 0.0
        return float(np.nan_to_num(unrealized_pnl, nan=0.0, posinf=0.0, neginf=0.0))

    def get_portfolio_stats(self) -> Dict[str, float]:
        if len(self.portfolio_values) == 0:
            return {}
        pv = np.array(self.portfolio_values, dtype=np.float64)
        rets = np.diff(pv) / pv[:-1]
        total_return = float((pv[-1] - self.initial_balance) / self.initial_balance)
        if len(rets) > 1:
            sharpe = float(np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252 * 24 * 4))
            max_dd = self._calculate_max_drawdown(pv)
            vol = float(np.std(rets) * np.sqrt(252 * 24 * 4))
        else:
            sharpe = 0.0
            max_dd = 0.0
            vol = 0.0
        if len(self.trades_history) > 0:
            pnl_values = np.array([t.get('pnl', 0.0) for t in self.trades_history])
            win_rate = float(np.sum(pnl_values > 0) / len(self.trades_history))
        else:
            win_rate = 0.0
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'volatility': vol,
            'win_rate': win_rate,
            'total_trades': float(self.total_trades),
            'total_fees': float(self.total_fees_paid),
            'final_balance': float(self.balance),
        }

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return float(abs(np.min(drawdown)))