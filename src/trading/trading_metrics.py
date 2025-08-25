#!/usr/bin/env python3
"""
Trading Metrics Module
=====================

Provides trading performance metrics and analytics.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime


class TradingMetrics:
    """Trading metrics and performance tracking."""
    
    def __init__(self):
        """Initialize trading metrics."""
        self.logger = logging.getLogger(__name__)
        self.trades = []
        self.performance_data = {}
        self.start_time = datetime.now()
        
    def record_trade(self, symbol: str, action: str, quantity: float, price: float, timestamp: datetime = None):
        """Record a trade for metrics tracking."""
        if timestamp is None:
            timestamp = datetime.now()
            
        trade = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'value': quantity * price
        }
        
        self.trades.append(trade)
        self.logger.debug(f"Recorded trade: {trade}")
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.trades:
            return {'total_trades': 0, 'total_value': 0.0}
            
        total_trades = len(self.trades)
        total_value = sum(trade['value'] for trade in self.trades)
        
        return {
            'total_trades': total_trades,
            'total_value': total_value,
            'start_time': self.start_time,
            'duration': datetime.now() - self.start_time
        }
        
    def reset_metrics(self):
        """Reset all metrics."""
        self.trades.clear()
        self.performance_data.clear()
        self.start_time = datetime.now()
        self.logger.info("Trading metrics reset")