"""
Backtesting Framework
====================

Comprehensive backtesting system for evaluating trading strategies.
Includes walk-forward validation and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float
    slippage: float
    total_cost: float
    portfolio_value_before: float
    portfolio_value_after: float

@dataclass
class BacktestResults:
    """Contains backtesting results and metrics."""
    trades: List[Trade]
    portfolio_values: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]
    drawdowns: pd.Series
    positions: pd.DataFrame
    
class Backtester:
    """
    Comprehensive backtesting framework for trading strategies.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        transaction_fee: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.1,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            transaction_fee: Transaction fee rate
            slippage: Slippage rate
            max_position_size: Maximum position size as fraction of portfolio
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        
        # State variables
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.portfolio_values = []
        self.timestamps = []
        
        logger.info(f"Backtester initialized with ${initial_capital:,.2f} capital")
    
    def run_backtest(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        strategy_func: Callable,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        rebalance_freq: str = '15T'  # 15 minutes
    ) -> BacktestResults:
        """
        Run backtest with given strategy.
        
        Args:
            data: Market data (DataFrame or dict of DataFrames for multiple assets)
            strategy_func: Strategy function that returns trading signals
            start_date: Start date for backtest
            end_date: End date for backtest
            rebalance_freq: Rebalancing frequency
            
        Returns:
            BacktestResults object
        """
        logger.info("Starting backtest")
        
        # Handle single vs multiple assets
        if isinstance(data, pd.DataFrame):
            data_dict = {'ASSET': data}
            symbols = ['ASSET']
        else:
            data_dict = data
            symbols = list(data.keys())
        
        # Filter data by date range
        if start_date or end_date:
            for symbol in symbols:
                if start_date:
                    data_dict[symbol] = data_dict[symbol][data_dict[symbol].index >= start_date]
                if end_date:
                    data_dict[symbol] = data_dict[symbol][data_dict[symbol].index <= end_date]
        
        # Get common time index
        common_index = data_dict[symbols[0]].index
        for symbol in symbols[1:]:
            common_index = common_index.intersection(data_dict[symbol].index)
        
        # Initialize positions
        for symbol in symbols:
            self.positions[symbol] = 0.0
        
        # Run backtest
        for timestamp in common_index:
            # Get current market data
            current_data = {}
            for symbol in symbols:
                if timestamp in data_dict[symbol].index:
                    current_data[symbol] = data_dict[symbol].loc[timestamp]
            
            # Skip if no data available
            if not current_data:
                continue
            
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value(current_data)
            self.portfolio_values.append(portfolio_value)
            self.timestamps.append(timestamp)
            
            # Get strategy signals
            try:
                signals = strategy_func(current_data, self.positions, portfolio_value)
                
                # Execute trades based on signals
                if signals:
                    self._execute_trades(signals, current_data, timestamp, portfolio_value)
                    
            except Exception as e:
                logger.warning(f"Strategy function failed at {timestamp}: {e}")
                continue
        
        # Create results
        results = self._create_results()
        
        logger.info(f"Backtest completed with {len(self.trades)} trades")
        return results
    
    def walk_forward_analysis(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        strategy_func: Callable,
        train_period: int = 252,  # Trading days
        test_period: int = 30,   # Trading days
        step_size: int = 30      # Trading days
    ) -> Dict[str, Any]:
        """
        Perform walk-forward analysis.
        
        Args:
            data: Market data
            strategy_func: Strategy function
            train_period: Training period in days
            test_period: Test period in days
            step_size: Step size for rolling window
            
        Returns:
            Walk-forward analysis results
        """
        logger.info("Starting walk-forward analysis")
        
        # Handle single vs multiple assets
        if isinstance(data, pd.DataFrame):
            data_dict = {'ASSET': data}
            symbols = ['ASSET']
        else:
            data_dict = data
            symbols = list(data.keys())
        
        # Get common time index
        common_index = data_dict[symbols[0]].index
        for symbol in symbols[1:]:
            common_index = common_index.intersection(data_dict[symbol].index)
        
        # Convert periods to actual timestamps
        total_periods = len(common_index)
        train_size = min(train_period * 96, total_periods // 3)  # 96 = 15-min intervals per day
        test_size = min(test_period * 96, total_periods // 10)
        step_size_intervals = step_size * 96
        
        results = []
        current_start = 0
        
        while current_start + train_size + test_size <= total_periods:
            # Define train and test periods
            train_start = current_start
            train_end = current_start + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            train_index = common_index[train_start:train_end]
            test_index = common_index[test_start:test_end]
            
            # Prepare train and test data
            train_data = {}
            test_data = {}
            
            for symbol in symbols:
                train_data[symbol] = data_dict[symbol].loc[train_index]
                test_data[symbol] = data_dict[symbol].loc[test_index]
            
            # Reset backtester state
            self._reset_state()
            
            # Run backtest on test period
            try:
                backtest_results = self.run_backtest(
                    test_data,
                    strategy_func,
                    start_date=test_index[0].strftime('%Y-%m-%d'),
                    end_date=test_index[-1].strftime('%Y-%m-%d')
                )
                
                # Store results
                period_result = {
                    'train_start': train_index[0],
                    'train_end': train_index[-1],
                    'test_start': test_index[0],
                    'test_end': test_index[-1],
                    'metrics': backtest_results.metrics,
                    'total_return': backtest_results.metrics.get('total_return', 0),
                    'sharpe_ratio': backtest_results.metrics.get('sharpe_ratio', 0),
                    'max_drawdown': backtest_results.metrics.get('max_drawdown', 0),
                    'num_trades': len(backtest_results.trades)
                }
                
                results.append(period_result)
                
                logger.info(f"Walk-forward period {len(results)}: "
                          f"Return={period_result['total_return']:.2%}, "
                          f"Sharpe={period_result['sharpe_ratio']:.2f}")
                
            except Exception as e:
                logger.error(f"Walk-forward period failed: {e}")
                continue
            
            # Move to next period
            current_start += step_size_intervals
        
        # Aggregate results
        if results:
            aggregate_metrics = {
                'num_periods': len(results),
                'avg_return': np.mean([r['total_return'] for r in results]),
                'std_return': np.std([r['total_return'] for r in results]),
                'avg_sharpe': np.mean([r['sharpe_ratio'] for r in results]),
                'avg_max_drawdown': np.mean([r['max_drawdown'] for r in results]),
                'win_rate': np.mean([r['total_return'] > 0 for r in results]),
                'total_trades': sum([r['num_trades'] for r in results])
            }
        else:
            aggregate_metrics = {}
        
        logger.info(f"Walk-forward analysis completed: {len(results)} periods")
        
        return {
            'periods': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def _execute_trades(
        self,
        signals: Dict[str, float],
        current_data: Dict[str, pd.Series],
        timestamp: datetime,
        portfolio_value: float
    ):
        """
        Execute trades based on signals.
        
        Args:
            signals: Dictionary of symbol -> target_weight
            current_data: Current market data
            timestamp: Current timestamp
            portfolio_value: Current portfolio value
        """
        for symbol, target_weight in signals.items():
            if symbol not in current_data:
                continue
            
            current_price = current_data[symbol]['close']
            current_position = self.positions.get(symbol, 0.0)
            
            # Calculate target position value
            target_value = target_weight * portfolio_value
            current_value = current_position * current_price
            
            # Calculate trade size
            trade_value = target_value - current_value
            
            if abs(trade_value) < portfolio_value * 0.001:  # Minimum trade threshold
                continue
            
            # Calculate quantity to trade
            if trade_value > 0:  # Buy
                side = 'buy'
                # Account for fees and slippage in purchase
                effective_price = current_price * (1 + self.slippage + self.transaction_fee)
                quantity = trade_value / effective_price
            else:  # Sell
                side = 'sell'
                # Account for fees and slippage in sale
                effective_price = current_price * (1 - self.slippage - self.transaction_fee)
                quantity = abs(trade_value) / current_price
            
            # Calculate costs
            fee = abs(trade_value) * self.transaction_fee
            slippage_cost = abs(trade_value) * self.slippage
            total_cost = fee + slippage_cost
            
            # Update capital and position
            if side == 'buy':
                self.current_capital -= (trade_value + total_cost)
                self.positions[symbol] = current_position + quantity
            else:
                self.current_capital += (abs(trade_value) - total_cost)
                self.positions[symbol] = current_position - quantity
            
            # Record trade
            trade = Trade(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=current_price,
                fee=fee,
                slippage=slippage_cost,
                total_cost=total_cost,
                portfolio_value_before=portfolio_value,
                portfolio_value_after=self._calculate_portfolio_value(current_data)
            )
            
            self.trades.append(trade)
    
    def _calculate_portfolio_value(self, current_data: Dict[str, pd.Series]) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            current_data: Current market data
            
        Returns:
            Total portfolio value
        """
        total_value = self.current_capital
        
        for symbol, quantity in self.positions.items():
            if symbol in current_data and quantity != 0:
                current_price = current_data[symbol]['close']
                total_value += quantity * current_price
        
        return total_value
    
    def _create_results(self) -> BacktestResults:
        """
        Create BacktestResults object.
        
        Returns:
            BacktestResults object
        """
        # Create portfolio values series
        portfolio_series = pd.Series(
            self.portfolio_values,
            index=self.timestamps
        )
        
        # Calculate returns
        returns = portfolio_series.pct_change().dropna()
        
        # Calculate drawdowns
        rolling_max = portfolio_series.expanding().max()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        
        # Create positions DataFrame
        positions_data = []
        for trade in self.trades:
            positions_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.quantity * trade.price
            })
        
        positions_df = pd.DataFrame(positions_data) if positions_data else pd.DataFrame()
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_series, returns, drawdowns)
        
        return BacktestResults(
            trades=self.trades,
            portfolio_values=portfolio_series,
            returns=returns,
            metrics=metrics,
            drawdowns=drawdowns,
            positions=positions_df
        )
    
    def _calculate_metrics(
        self,
        portfolio_values: pd.Series,
        returns: pd.Series,
        drawdowns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            portfolio_values: Portfolio values over time
            returns: Portfolio returns
            drawdowns: Portfolio drawdowns
            
        Returns:
            Dictionary of metrics
        """
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized metrics (assuming 15-minute data)
        periods_per_year = 365 * 24 * 4  # 15-minute intervals per year
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1
        annualized_volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe_ratio = excess_returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods_per_year)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
            sortino_ratio = annualized_return / (downside_deviation + 1e-8)
        else:
            sortino_ratio = np.inf
        
        # Maximum drawdown
        max_drawdown = abs(drawdowns.min())
        
        # Calmar ratio
        calmar_ratio = annualized_return / (max_drawdown + 1e-8)
        
        # Win rate and profit factor
        if len(self.trades) > 0:
            profitable_trades = [t for t in self.trades if self._calculate_trade_pnl(t) > 0]
            win_rate = len(profitable_trades) / len(self.trades)
            
            total_profit = sum([self._calculate_trade_pnl(t) for t in profitable_trades])
            total_loss = abs(sum([self._calculate_trade_pnl(t) for t in self.trades if self._calculate_trade_pnl(t) < 0]))
            profit_factor = total_profit / (total_loss + 1e-8)
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        # Additional metrics
        num_trades = len(self.trades)
        avg_trade_return = total_return / (num_trades + 1e-8)
        
        # Recovery factor
        recovery_factor = total_return / (max_drawdown + 1e-8)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'recovery_factor': recovery_factor,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': num_trades,
            'avg_trade_return': avg_trade_return,
            'final_portfolio_value': portfolio_values.iloc[-1],
            'total_fees_paid': sum([t.fee for t in self.trades]),
            'total_slippage_cost': sum([t.slippage for t in self.trades])
        }
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """
        Calculate P&L for a trade (simplified).
        
        Args:
            trade: Trade object
            
        Returns:
            Trade P&L
        """
        # This is a simplified calculation
        # In practice, you'd need to track entry/exit prices properly
        return trade.portfolio_value_after - trade.portfolio_value_before
    
    def _reset_state(self):
        """Reset backtester state."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.timestamps = []
    
    def plot_results(self, results: BacktestResults, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot backtest results.
        
        Args:
            results: BacktestResults object
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Portfolio value over time
        axes[0, 0].plot(results.portfolio_values.index, results.portfolio_values.values)
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        axes[0, 1].fill_between(
            results.drawdowns.index,
            results.drawdowns.values * 100,
            0,
            alpha=0.7,
            color='red'
        )
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Returns distribution
        if len(results.returns) > 0:
            axes[1, 0].hist(results.returns * 100, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(x=results.returns.mean() * 100, color='r', linestyle='--')
            axes[1, 0].set_title('Returns Distribution')
            axes[1, 0].set_xlabel('Return (%)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Trade analysis
        if results.trades:
            trade_returns = [self._calculate_trade_pnl(t) for t in results.trades]
            axes[1, 1].scatter(range(len(trade_returns)), trade_returns, alpha=0.6)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
            axes[1, 1].set_title('Trade P&L')
            axes[1, 1].set_xlabel('Trade Number')
            axes[1, 1].set_ylabel('P&L ($)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results: BacktestResults) -> str:
        """
        Generate a comprehensive backtest report.
        
        Args:
            results: BacktestResults object
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("BACKTEST REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic information
        report.append("BASIC INFORMATION")
        report.append("-" * 20)
        report.append(f"Initial Capital: ${self.initial_capital:,.2f}")
        report.append(f"Final Portfolio Value: ${results.metrics.get('final_portfolio_value', 0):,.2f}")
        report.append(f"Total Return: {results.metrics.get('total_return', 0):.2%}")
        report.append(f"Number of Trades: {results.metrics.get('num_trades', 0)}")
        report.append("")
        
        # Performance metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 20)
        report.append(f"Annualized Return: {results.metrics.get('annualized_return', 0):.2%}")
        report.append(f"Annualized Volatility: {results.metrics.get('annualized_volatility', 0):.2%}")
        report.append(f"Sharpe Ratio: {results.metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Sortino Ratio: {results.metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"Calmar Ratio: {results.metrics.get('calmar_ratio', 0):.2f}")
        report.append(f"Maximum Drawdown: {results.metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Recovery Factor: {results.metrics.get('recovery_factor', 0):.2f}")
        report.append("")
        
        # Trading statistics
        report.append("TRADING STATISTICS")
        report.append("-" * 20)
        report.append(f"Win Rate: {results.metrics.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor: {results.metrics.get('profit_factor', 0):.2f}")
        report.append(f"Average Trade Return: {results.metrics.get('avg_trade_return', 0):.2%}")
        report.append(f"Total Fees Paid: ${results.metrics.get('total_fees_paid', 0):.2f}")
        report.append(f"Total Slippage Cost: ${results.metrics.get('total_slippage_cost', 0):.2f}")
        report.append("")
        
        return "\n".join(report)