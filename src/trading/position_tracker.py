"""
Position Tracking Module
========================

Implements realistic position tracking with ownership validation,
transaction history, and proper P&L calculations.
"""

import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Trade:
    """Represents a single executed trade."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost including fees."""
        if self.side == OrderSide.BUY:
            return (self.quantity * self.price) + self.commission + self.slippage
        else:  # SELL
            return (self.quantity * self.price) - self.commission - self.slippage
    
    @property
    def net_proceeds(self) -> float:
        """Calculate net proceeds from trade."""
        if self.side == OrderSide.SELL:
            return (self.quantity * self.price) - self.commission - self.slippage
        else:
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'slippage': self.slippage,
            'total_cost': self.total_cost
        }


@dataclass
class Position:
    """Represents a position in a single asset."""
    symbol: str
    trades: List[Trade] = field(default_factory=list)
    
    @property
    def quantity(self) -> float:
        """Calculate current position size."""
        total = 0.0
        for trade in self.trades:
            if trade.side == OrderSide.BUY:
                total += trade.quantity
            else:  # SELL
                total -= trade.quantity
        return total
    
    @property
    def average_entry_price(self) -> float:
        """Calculate weighted average entry price for current position."""
        buy_trades = [(t.quantity, t.price) for t in self.trades if t.side == OrderSide.BUY]
        if not buy_trades:
            return 0.0
        
        # For FIFO, we need to track which buys have been sold
        remaining_buys = self._get_remaining_buys_fifo()
        if not remaining_buys:
            return 0.0
        
        total_cost = sum(q * p for q, p in remaining_buys)
        total_quantity = sum(q for q, _ in remaining_buys)
        
        return total_cost / total_quantity if total_quantity > 0 else 0.0
    
    def _get_remaining_buys_fifo(self) -> List[Tuple[float, float]]:
        """Get remaining buy positions using FIFO accounting."""
        buy_queue = []
        
        for trade in sorted(self.trades, key=lambda t: t.timestamp):
            if trade.side == OrderSide.BUY:
                buy_queue.append([trade.quantity, trade.price])
            else:  # SELL
                remaining_sell = trade.quantity
                while remaining_sell > 0 and buy_queue:
                    if buy_queue[0][0] <= remaining_sell:
                        # Consume entire buy lot
                        remaining_sell -= buy_queue[0][0]
                        buy_queue.pop(0)
                    else:
                        # Partial consumption of buy lot
                        buy_queue[0][0] -= remaining_sell
                        remaining_sell = 0
        
        return [(q, p) for q, p in buy_queue]
    
    def calculate_realized_pnl(self) -> float:
        """Calculate realized P&L using FIFO method."""
        buy_queue = []
        realized_pnl = 0.0
        
        for trade in sorted(self.trades, key=lambda t: t.timestamp):
            if trade.side == OrderSide.BUY:
                buy_queue.append([trade.quantity, trade.price, trade.commission + trade.slippage])
            else:  # SELL
                remaining_sell = trade.quantity
                sell_price = trade.price
                sell_costs = trade.commission + trade.slippage
                
                while remaining_sell > 0 and buy_queue:
                    buy_quantity, buy_price, buy_costs = buy_queue[0]
                    
                    if buy_quantity <= remaining_sell:
                        # Consume entire buy lot
                        # P&L = (sell_price - buy_price) * quantity - proportional costs
                        pnl = (sell_price - buy_price) * buy_quantity
                        pnl -= buy_costs  # Subtract buy costs
                        pnl -= (buy_quantity / trade.quantity) * sell_costs  # Proportional sell costs
                        
                        realized_pnl += pnl
                        remaining_sell -= buy_quantity
                        buy_queue.pop(0)
                    else:
                        # Partial consumption of buy lot
                        # P&L = (sell_price - buy_price) * quantity - proportional costs
                        pnl = (sell_price - buy_price) * remaining_sell
                        pnl -= (remaining_sell / buy_quantity) * buy_costs  # Proportional buy costs
                        pnl -= (remaining_sell / trade.quantity) * sell_costs  # Proportional sell costs
                        
                        realized_pnl += pnl
                        buy_queue[0][0] -= remaining_sell
                        buy_queue[0][2] *= (buy_queue[0][0] / buy_quantity)  # Adjust remaining costs
                        remaining_sell = 0
        
        return realized_pnl
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for current position."""
        if self.quantity == 0:
            return 0.0
        
        current_value = self.quantity * current_price
        cost_basis = self.quantity * self.average_entry_price
        
        # Account for future transaction costs when closing position
        # This is an estimate - actual costs may vary
        estimated_closing_costs = abs(current_value) * 0.002  # 0.1% fee + 0.1% slippage
        
        return current_value - cost_basis - estimated_closing_costs


class PositionTracker:
    """
    Main position tracking system with ownership validation and P&L tracking.
    """
    
    def __init__(
        self,
        initial_capital: float,
        transaction_fee: float = 0.001,
        slippage: float = 0.001,
        max_position_size: float = 0.1
    ):
        """
        Initialize position tracker.
        
        Args:
            initial_capital: Starting capital
            transaction_fee: Transaction fee rate
            slippage: Slippage rate
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position_size = max_position_size
        
        self.positions: Dict[str, Position] = {}
        self.transaction_history: List[Trade] = []
        self.rejected_orders: List[Dict[str, Any]] = []
        
        logger.info(f"Position tracker initialized with ${initial_capital:,.2f}")
    
    def validate_buy_order(
        self, 
        symbol: str, 
        quantity: float, 
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate buy order against available capital.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if price <= 0:
            return False, "Price must be positive"
        
        # Calculate total cost including fees
        order_value = quantity * price
        commission = order_value * self.transaction_fee
        slippage_cost = order_value * self.slippage
        total_cost = order_value + commission + slippage_cost
        
        # Check available capital
        if total_cost > self.cash:
            return False, f"Insufficient funds: need ${total_cost:.2f}, have ${self.cash:.2f}"
        
        # Check position size limit
        portfolio_value = self.calculate_portfolio_value({symbol: price})
        position_value = order_value
        
        if symbol in self.positions:
            current_position_value = self.positions[symbol].quantity * price
            position_value += current_position_value
        
        if position_value > portfolio_value * self.max_position_size:
            return False, f"Position size exceeds limit: {self.max_position_size * 100}% of portfolio"
        
        return True, None
    
    def validate_sell_order(
        self, 
        symbol: str, 
        quantity: float, 
        price: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate sell order against current holdings.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if quantity <= 0:
            return False, "Quantity must be positive"
        
        if price <= 0:
            return False, "Price must be positive"
        
        # Check if we have a position
        if symbol not in self.positions:
            return False, f"No position in {symbol}"
        
        # Check if we have enough shares
        current_quantity = self.positions[symbol].quantity
        if current_quantity < quantity:
            return False, f"Insufficient shares: trying to sell {quantity}, but only own {current_quantity}"
        
        return True, None
    
    def execute_buy(
        self, 
        symbol: str, 
        quantity: float, 
        price: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, Optional[Trade], Optional[str]]:
        """
        Execute a buy order with validation.
        
        Returns:
            Tuple of (success, trade, error_message)
        """
        # Validate order
        is_valid, error_msg = self.validate_buy_order(symbol, quantity, price)
        if not is_valid:
            self.rejected_orders.append({
                'timestamp': timestamp or datetime.now(),
                'symbol': symbol,
                'side': 'BUY',
                'quantity': quantity,
                'price': price,
                'reason': error_msg
            })
            logger.warning(f"Buy order rejected: {error_msg}")
            return False, None, error_msg
        
        # Calculate costs
        order_value = quantity * price
        commission = order_value * self.transaction_fee
        slippage_cost = order_value * self.slippage
        total_cost = order_value + commission + slippage_cost
        
        # Create trade record
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            timestamp=timestamp or datetime.now(),
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage_cost
        )
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        self.positions[symbol].trades.append(trade)
        
        # Update cash
        self.cash -= total_cost
        
        # Record transaction
        self.transaction_history.append(trade)
        
        logger.info(f"Buy executed: {quantity} {symbol} @ ${price:.2f}, total cost: ${total_cost:.2f}")
        
        return True, trade, None
    
    def execute_sell(
        self, 
        symbol: str, 
        quantity: float, 
        price: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, Optional[Trade], Optional[str]]:
        """
        Execute a sell order with validation.
        
        Returns:
            Tuple of (success, trade, error_message)
        """
        # Validate order
        is_valid, error_msg = self.validate_sell_order(symbol, quantity, price)
        if not is_valid:
            self.rejected_orders.append({
                'timestamp': timestamp or datetime.now(),
                'symbol': symbol,
                'side': 'SELL',
                'quantity': quantity,
                'price': price,
                'reason': error_msg
            })
            logger.warning(f"Sell order rejected: {error_msg}")
            return False, None, error_msg
        
        # Calculate proceeds
        order_value = quantity * price
        commission = order_value * self.transaction_fee
        slippage_cost = order_value * self.slippage
        net_proceeds = order_value - commission - slippage_cost
        
        # Create trade record
        trade = Trade(
            trade_id=str(uuid.uuid4()),
            timestamp=timestamp or datetime.now(),
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage_cost
        )
        
        # Update position
        self.positions[symbol].trades.append(trade)
        
        # Update cash
        self.cash += net_proceeds
        
        # Record transaction
        self.transaction_history.append(trade)
        
        logger.info(f"Sell executed: {quantity} {symbol} @ ${price:.2f}, net proceeds: ${net_proceeds:.2f}")
        
        return True, trade, None
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including cash and positions.
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Total portfolio value
        """
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if position.quantity > 0 and symbol in current_prices:
                position_value = position.quantity * current_prices[symbol]
                total_value += position_value
        
        return total_value
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Portfolio summary dictionary
        """
        portfolio_value = self.calculate_portfolio_value(current_prices)
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        positions_summary = []
        total_unrealized_pnl = 0.0
        total_realized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                current_price = current_prices.get(symbol, 0)
                unrealized_pnl = position.calculate_unrealized_pnl(current_price)
                realized_pnl = position.calculate_realized_pnl()
                
                total_unrealized_pnl += unrealized_pnl
                total_realized_pnl += realized_pnl
                
                positions_summary.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'average_price': position.average_entry_price,
                    'current_price': current_price,
                    'market_value': position.quantity * current_price,
                    'unrealized_pnl': unrealized_pnl,
                    'realized_pnl': realized_pnl,
                    'total_trades': len(position.trades)
                })
        
        return {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'positions': positions_summary,
            'total_trades': len(self.transaction_history),
            'rejected_orders': len(self.rejected_orders)
        }
    
    def get_trade_history(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get trade history, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of trade dictionaries
        """
        trades = self.transaction_history
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        return [t.to_dict() for t in sorted(trades, key=lambda t: t.timestamp, reverse=True)]
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export transaction history to pandas DataFrame."""
        if not self.transaction_history:
            return pd.DataFrame()
        
        data = []
        for trade in self.transaction_history:
            data.append({
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'total_cost': trade.total_cost if trade.side == OrderSide.BUY else -trade.net_proceeds
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df