"""
Logging Utilities
=================

Centralized logging configuration for the trading bot.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import yaml

def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: str = "10MB",
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up centralized logging configuration.
    
    Args:
        config: Configuration dictionary
        log_level: Logging level
        log_file: Log file path
        max_file_size: Maximum log file size
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Parse configuration
    if config:
        logging_config = config.get('logging', {})
        log_level = logging_config.get('level', log_level)
        log_file = logging_config.get('file', log_file)
        max_file_size = logging_config.get('max_file_size', max_file_size)
        backup_count = logging_config.get('backup_count', backup_count)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Parse max file size
        size_bytes = _parse_size(max_file_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=size_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Create main application logger
    app_logger = logging.getLogger('crypto_trading_bot')
    app_logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    
    return app_logger

def _parse_size(size_str: str) -> int:
    """
    Parse size string to bytes.
    
    Args:
        size_str: Size string (e.g., '10MB', '1GB')
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(float(size_str[:-2]) * 1024)
    elif size_str.endswith('MB'):
        return int(float(size_str[:-2]) * 1024 * 1024)
    elif size_str.endswith('GB'):
        return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
    else:
        # Assume bytes
        return int(size_str)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class TradingBotLogger:
    """
    Specialized logger for trading bot with structured logging.
    """
    
    def __init__(self, name: str = "crypto_trading_bot"):
        """
        Initialize trading bot logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.name = name
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        level: str = "INFO"
    ):
        """
        Log trade execution.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            quantity: Trade quantity
            price: Trade price
            portfolio_value: Portfolio value
            level: Log level
        """
        message = (
            f"TRADE EXECUTED - Symbol: {symbol}, Side: {side}, "
            f"Quantity: {quantity:.6f}, Price: €{price:.4f}, "
            f"Value: €{quantity * price:.2f}, Portfolio: €{portfolio_value:.2f}"
        )
        
        getattr(self.logger, level.lower())(message)
    
    def log_model_update(
        self,
        model_type: str,
        action: str,
        metrics: Optional[Dict[str, float]] = None,
        level: str = "INFO"
    ):
        """
        Log model updates.
        
        Args:
            model_type: Type of model
            action: Action performed
            metrics: Performance metrics
            level: Log level
        """
        message = f"MODEL UPDATE - Type: {model_type}, Action: {action}"
        
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            message += f", Metrics: {metrics_str}"
        
        getattr(self.logger, level.lower())(message)
    
    def log_portfolio_update(
        self,
        portfolio_value: float,
        daily_pnl: float,
        positions: Dict[str, float],
        level: str = "INFO"
    ):
        """
        Log portfolio updates.
        
        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily P&L
            positions: Current positions
            level: Log level
        """
        active_positions = {k: v for k, v in positions.items() if abs(v) > 1e-6}
        
        message = (
            f"PORTFOLIO UPDATE - Value: €{portfolio_value:.2f}, "
            f"Daily P&L: €{daily_pnl:.2f}, "
            f"Positions: {len(active_positions)}"
        )
        
        getattr(self.logger, level.lower())(message)
        
        # Log individual positions at debug level
        for symbol, quantity in active_positions.items():
            self.logger.debug(f"Position - {symbol}: {quantity:.6f}")
    
    def log_system_event(
        self,
        event_type: str,
        message: str,
        level: str = "INFO",
        **kwargs
    ):
        """
        Log system events.
        
        Args:
            event_type: Type of event
            message: Event message
            level: Log level
            **kwargs: Additional context
        """
        log_message = f"SYSTEM EVENT - Type: {event_type}, Message: {message}"
        
        if kwargs:
            context_str = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
            log_message += f", Context: {context_str}"
        
        getattr(self.logger, level.lower())(log_message)
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None,
        **kwargs
    ):
        """
        Log errors with context.
        
        Args:
            error_type: Type of error
            error_message: Error message
            traceback: Error traceback
            **kwargs: Additional context
        """
        log_message = f"ERROR - Type: {error_type}, Message: {error_message}"
        
        if kwargs:
            context_str = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
            log_message += f", Context: {context_str}"
        
        self.logger.error(log_message)
        
        if traceback:
            self.logger.error(f"Traceback: {traceback}")
    
    def log_performance_metrics(
        self,
        metrics: Dict[str, float],
        period: str = "daily",
        level: str = "INFO"
    ):
        """
        Log performance metrics.
        
        Args:
            metrics: Performance metrics
            period: Time period
            level: Log level
        """
        message = f"PERFORMANCE METRICS ({period.upper()})"
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'ratio' in metric.lower() or 'return' in metric.lower():
                    message += f", {metric}: {value:.4f}"
                else:
                    message += f", {metric}: {value:.2f}"
            else:
                message += f", {metric}: {value}"
        
        getattr(self.logger, level.lower())(message)

def create_performance_logger(log_file: str = "logs/performance.log") -> logging.Logger:
    """
    Create a dedicated performance logger.
    
    Args:
        log_file: Performance log file path
        
    Returns:
        Performance logger
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in perf_logger.handlers[:]:
        perf_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    
    # Create formatter for performance logs
    formatter = logging.Formatter(
        '%(asctime)s,%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    perf_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    perf_logger.propagate = False
    
    return perf_logger

def log_trade_performance(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    portfolio_value: float,
    pnl: float,
    logger: Optional[logging.Logger] = None
):
    """
    Log trade performance data in CSV format.
    
    Args:
        symbol: Trading symbol
        side: Trade side
        quantity: Trade quantity
        price: Trade price
        portfolio_value: Portfolio value
        pnl: Profit/Loss
        logger: Performance logger (optional)
    """
    if logger is None:
        logger = create_performance_logger()
    
    # Log in CSV format for easy analysis
    log_entry = f"{symbol},{side},{quantity},{price},{portfolio_value},{pnl}"
    logger.info(log_entry)

class LoggingContext:
    """
    Context manager for temporary logging configuration.
    """
    
    def __init__(self, logger: logging.Logger, level: int):
        """
        Initialize logging context.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
    
    def __enter__(self):
        """Enter context - set new log level."""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore old log level."""
        self.logger.setLevel(self.old_level)

def with_debug_logging(logger: logging.Logger):
    """
    Context manager for temporary debug logging.
    
    Args:
        logger: Logger to modify
        
    Returns:
        LoggingContext for debug level
    """
    return LoggingContext(logger, logging.DEBUG)

def with_quiet_logging(logger: logging.Logger):
    """
    Context manager for temporary quiet logging.
    
    Args:
        logger: Logger to modify
        
    Returns:
        LoggingContext for warning level
    """
    return LoggingContext(logger, logging.WARNING)