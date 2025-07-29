"""
Telegram Notification Module
============================

Telegram bot for sending trading notifications and alerts.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """
    Telegram notification system for trading bot.
    """
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
            enabled: Whether notifications are enabled
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning("python-telegram-bot not available. Telegram notifications disabled.")
            self.enabled = False
            return
        
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled and bool(bot_token) and bool(chat_id)
        
        if self.enabled:
            self.bot = Bot(token=bot_token)
            logger.info("Telegram notifier initialized")
        else:
            self.bot = None
            logger.warning("Telegram notifier disabled - missing token or chat_id")
    
    async def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a message to Telegram.
        
        Args:
            message: Message to send
            parse_mode: Parse mode ('HTML' or 'Markdown')
            
        Returns:
            True if message sent successfully
        """
        if not self.enabled or not self.bot:
            logger.debug(f"Telegram disabled - would send: {message}")
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            logger.debug("Telegram message sent successfully")
            return True
            
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False
    
    def send_message_sync(self, message: str, parse_mode: str = 'HTML') -> bool:
        """
        Send a message synchronously.
        
        Args:
            message: Message to send
            parse_mode: Parse mode ('HTML' or 'Markdown')
            
        Returns:
            True if message sent successfully
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.send_message(message, parse_mode))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.send_message(message, parse_mode))
    
    async def send_trade_notification(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        pnl: Optional[float] = None
    ) -> bool:
        """
        Send trade notification.
        
        Args:
            symbol: Trading symbol
            side: Trade side ('BUY' or 'SELL')
            quantity: Trade quantity
            price: Trade price
            portfolio_value: Current portfolio value
            pnl: Profit/Loss (optional)
            
        Returns:
            True if notification sent successfully
        """
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        
        message = f"""
{emoji} <b>TRADE EXECUTED</b>

<b>Symbol:</b> {symbol}
<b>Side:</b> {side.upper()}
<b>Quantity:</b> {quantity:.6f}
<b>Price:</b> €{price:.4f}
<b>Value:</b> €{quantity * price:.2f}

<b>Portfolio Value:</b> €{portfolio_value:.2f}
"""
        
        if pnl is not None:
            pnl_emoji = "💰" if pnl > 0 else "💸"
            message += f"<b>P&L:</b> {pnl_emoji} €{pnl:.2f}\n"
        
        message += f"\n<i>Time:</i> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_message(message)
    
    async def send_portfolio_update(
        self,
        portfolio_value: float,
        daily_pnl: float,
        daily_return: float,
        positions: Dict[str, float]
    ) -> bool:
        """
        Send portfolio update notification.
        
        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily P&L
            daily_return: Daily return percentage
            positions: Current positions
            
        Returns:
            True if notification sent successfully
        """
        pnl_emoji = "📈" if daily_pnl > 0 else "📉"
        
        message = f"""
{pnl_emoji} <b>PORTFOLIO UPDATE</b>

<b>Portfolio Value:</b> €{portfolio_value:.2f}
<b>Daily P&L:</b> €{daily_pnl:.2f}
<b>Daily Return:</b> {daily_return:.2%}

<b>Current Positions:</b>
"""
        
        if positions:
            for symbol, quantity in positions.items():
                if abs(quantity) > 1e-6:  # Only show non-zero positions
                    message += f"• {symbol}: {quantity:.6f}\n"
        else:
            message += "• No open positions\n"
        
        message += f"\n<i>Time:</i> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_message(message)
    
    async def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "INFO"
    ) -> bool:
        """
        Send alert notification.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Severity level ('INFO', 'WARNING', 'ERROR')
            
        Returns:
            True if notification sent successfully
        """
        severity_emojis = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨"
        }
        
        emoji = severity_emojis.get(severity.upper(), "ℹ️")
        
        alert_message = f"""
{emoji} <b>{severity.upper()} ALERT</b>

<b>Type:</b> {alert_type}
<b>Message:</b> {message}

<i>Time:</i> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return await self.send_message(alert_message)
    
    async def send_system_status(
        self,
        status: str,
        uptime: Optional[str] = None,
        last_trade: Optional[str] = None,
        active_models: Optional[List[str]] = None
    ) -> bool:
        """
        Send system status notification.
        
        Args:
            status: System status ('RUNNING', 'STOPPED', 'ERROR')
            uptime: System uptime (optional)
            last_trade: Last trade time (optional)
            active_models: List of active models (optional)
            
        Returns:
            True if notification sent successfully
        """
        status_emojis = {
            "RUNNING": "🟢",
            "STOPPED": "🔴",
            "ERROR": "🚨"
        }
        
        emoji = status_emojis.get(status.upper(), "⚪")
        
        message = f"""
{emoji} <b>SYSTEM STATUS</b>

<b>Status:</b> {status.upper()}
"""
        
        if uptime:
            message += f"<b>Uptime:</b> {uptime}\n"
        
        if last_trade:
            message += f"<b>Last Trade:</b> {last_trade}\n"
        
        if active_models:
            message += f"<b>Active Models:</b> {', '.join(active_models)}\n"
        
        message += f"\n<i>Time:</i> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_message(message)
    
    async def send_backtest_results(
        self,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        num_trades: int,
        win_rate: float
    ) -> bool:
        """
        Send backtest results notification.
        
        Args:
            total_return: Total return percentage
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown percentage
            num_trades: Number of trades
            win_rate: Win rate percentage
            
        Returns:
            True if notification sent successfully
        """
        performance_emoji = "🎯" if total_return > 0 else "📉"
        
        message = f"""
{performance_emoji} <b>BACKTEST RESULTS</b>

<b>Total Return:</b> {total_return:.2%}
<b>Sharpe Ratio:</b> {sharpe_ratio:.2f}
<b>Max Drawdown:</b> {max_drawdown:.2%}
<b>Number of Trades:</b> {num_trades}
<b>Win Rate:</b> {win_rate:.2%}

<i>Time:</i> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return await self.send_message(message)
    
    async def send_model_update(
        self,
        model_type: str,
        action: str,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Send model update notification.
        
        Args:
            model_type: Type of model ('GRU', 'LightGBM', 'PPO')
            action: Action performed ('TRAINED', 'LOADED', 'UPDATED')
            performance_metrics: Performance metrics (optional)
            
        Returns:
            True if notification sent successfully
        """
        action_emojis = {
            "TRAINED": "🎓",
            "LOADED": "📥",
            "UPDATED": "🔄"
        }
        
        emoji = action_emojis.get(action.upper(), "🤖")
        
        message = f"""
{emoji} <b>MODEL UPDATE</b>

<b>Model:</b> {model_type}
<b>Action:</b> {action.upper()}
"""
        
        if performance_metrics:
            message += "\n<b>Performance Metrics:</b>\n"
            for metric, value in performance_metrics.items():
                if isinstance(value, float):
                    if 'ratio' in metric.lower() or 'return' in metric.lower():
                        message += f"• {metric.title()}: {value:.4f}\n"
                    else:
                        message += f"• {metric.title()}: {value:.2f}\n"
                else:
                    message += f"• {metric.title()}: {value}\n"
        
        message += f"\n<i>Time:</i> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_message(message)
    
    async def send_error_notification(
        self,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None
    ) -> bool:
        """
        Send error notification.
        
        Args:
            error_type: Type of error
            error_message: Error message
            traceback: Error traceback (optional)
            
        Returns:
            True if notification sent successfully
        """
        message = f"""
🚨 <b>ERROR ALERT</b>

<b>Type:</b> {error_type}
<b>Message:</b> {error_message}
"""
        
        if traceback:
            # Truncate traceback if too long
            if len(traceback) > 1000:
                traceback = traceback[:1000] + "..."
            message += f"\n<b>Traceback:</b>\n<code>{traceback}</code>"
        
        message += f"\n<i>Time:</i> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return await self.send_message(message)
    
    def test_connection(self) -> bool:
        """
        Test Telegram connection.
        
        Returns:
            True if connection successful
        """
        if not self.enabled:
            logger.info("Telegram notifications are disabled")
            return False
        
        test_message = f"""
🧪 <b>CONNECTION TEST</b>

Telegram notifications are working correctly!

<i>Time:</i> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return self.send_message_sync(test_message)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'TelegramNotifier':
        """
        Create TelegramNotifier from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            TelegramNotifier instance
        """
        telegram_config = config.get('notifications', {}).get('telegram', {})
        
        bot_token = telegram_config.get('bot_token') or os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = telegram_config.get('chat_id') or os.getenv('TELEGRAM_CHAT_ID')
        enabled = telegram_config.get('enabled', True)
        
        return cls(bot_token=bot_token, chat_id=chat_id, enabled=enabled)

class NotificationManager:
    """
    Manages multiple notification channels.
    """
    
    def __init__(self):
        """Initialize notification manager."""
        self.notifiers = {}
        logger.info("Notification manager initialized")
    
    def add_notifier(self, name: str, notifier: TelegramNotifier):
        """
        Add a notifier.
        
        Args:
            name: Notifier name
            notifier: Notifier instance
        """
        self.notifiers[name] = notifier
        logger.info(f"Added notifier: {name}")
    
    async def broadcast_message(self, message: str, channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Broadcast message to multiple channels.
        
        Args:
            message: Message to broadcast
            channels: List of channel names (None for all)
            
        Returns:
            Dictionary of channel -> success status
        """
        if channels is None:
            channels = list(self.notifiers.keys())
        
        results = {}
        for channel in channels:
            if channel in self.notifiers:
                results[channel] = await self.notifiers[channel].send_message(message)
            else:
                logger.warning(f"Unknown notification channel: {channel}")
                results[channel] = False
        
        return results
    
    async def send_trade_alert(self, **kwargs) -> Dict[str, bool]:
        """Send trade alert to all notifiers."""
        results = {}
        for name, notifier in self.notifiers.items():
            results[name] = await notifier.send_trade_notification(**kwargs)
        return results
    
    async def send_portfolio_update(self, **kwargs) -> Dict[str, bool]:
        """Send portfolio update to all notifiers."""
        results = {}
        for name, notifier in self.notifiers.items():
            results[name] = await notifier.send_portfolio_update(**kwargs)
        return results
    
    async def send_system_alert(self, **kwargs) -> Dict[str, bool]:
        """Send system alert to all notifiers."""
        results = {}
        for name, notifier in self.notifiers.items():
            results[name] = await notifier.send_alert(**kwargs)
        return results