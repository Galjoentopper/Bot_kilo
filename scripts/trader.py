"""
Paper Trading Script
===================

Main script for running the paper trading bot with trained models.
"""

import sys
import os
import argparse
import yaml
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Import from installed package
from src.utils.logger import setup_logging, TradingBotLogger
from src.data_pipeline.loader import DataLoader
from src.data_pipeline.features import FeatureEngine
from src.data_pipeline.preprocess import DataPreprocessor
from src.models.gru_trainer import GRUTrainer
from src.models.lgbm_trainer import LightGBMTrainer
from src.models.ppo_trainer import PPOTrainer
from src.notifier.telegram import TelegramNotifier

class PaperTrader:
    """
    Paper trading bot that uses trained models to make trading decisions.
    """
    
    def __init__(self, config: dict, models_dir: str = "./models"):
        """
        Initialize paper trader.
        
        Args:
            config: Configuration dictionary
            models_dir: Directory containing trained models
        """
        self.config = config
        self.models_dir = models_dir
        
        # Initialize components
        self.logger = TradingBotLogger()
        self.data_loader = DataLoader(config.get('data', {}).get('data_dir', './data'))
        self.feature_engine = FeatureEngine(config.get('features', {}))
        self.preprocessor = DataPreprocessor()
        
        # Initialize notification system
        try:
            self.notifier = TelegramNotifier.from_config(config)
        except Exception as e:
            self.logger.logger.warning(f"Failed to initialize notifications: {e}")
            self.notifier = None
        
        # Trading state
        self.portfolio_value = config.get('trading', {}).get('initial_balance', 10000.0)
        self.positions = {}  # symbol -> quantity
        self.last_prices = {}  # symbol -> last_price
        self.daily_start_value = self.portfolio_value
        
        # Models
        self.models = {}
        self.load_models()
        
        # Trading parameters
        self.symbols = config.get('data', {}).get('symbols', ['BTCEUR'])
        self.max_position_size = config.get('trading', {}).get('max_position_size', 0.1)
        self.transaction_fee = config.get('trading', {}).get('transaction_fee', 0.001)
        
        self.logger.logger.info(f"Paper trader initialized with ${self.portfolio_value:,.2f}")
    
    def load_models(self):
        """Load trained models."""
        try:
            # Find latest model files
            model_files = {
                'gru': self._find_latest_model('gru_model_*.pth'),
                'lightgbm': self._find_latest_model('lightgbm_model_*.pkl'),
                'ppo': self._find_latest_model('ppo_model_*')
            }
            
            # Load GRU model
            if model_files['gru']:
                try:
                    self.models['gru'] = GRUTrainer.load_model(model_files['gru'], self.config)
                    self.logger.logger.info(f"Loaded GRU model: {model_files['gru']}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load GRU model: {e}")
            
            # Load LightGBM model
            if model_files['lightgbm']:
                try:
                    self.models['lightgbm'] = LightGBMTrainer.load_model(model_files['lightgbm'], self.config)
                    self.logger.logger.info(f"Loaded LightGBM model: {model_files['lightgbm']}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load LightGBM model: {e}")
            
            # Load PPO model
            if model_files['ppo']:
                try:
                    self.models['ppo'] = PPOTrainer.load_model(model_files['ppo'], self.config)
                    self.logger.logger.info(f"Loaded PPO model: {model_files['ppo']}")
                except Exception as e:
                    self.logger.logger.warning(f"Failed to load PPO model: {e}")
            
            if not self.models:
                self.logger.logger.warning("No models loaded! Trading will use simple strategy.")
        
        except Exception as e:
            self.logger.logger.error(f"Error loading models: {e}")
    
    def _find_latest_model(self, pattern: str) -> str:
        """Find the latest model file matching pattern."""
        import glob
        
        model_files = glob.glob(os.path.join(self.models_dir, pattern))
        if model_files:
            # Sort by modification time and return the latest
            return max(model_files, key=os.path.getmtime)
        return None
    
    async def get_market_data(self) -> dict:
        """Get latest market data for all symbols."""
        market_data = {}
        
        for symbol in self.symbols:
            try:
                # Get recent data (last 100 periods for feature calculation)
                df = self.data_loader.get_latest_data(symbol, n_periods=100)
                
                if not df.empty:
                    # Generate features
                    features_df = self.feature_engine.generate_all_features(df)
                    market_data[symbol] = features_df
                    
                    # Update last price
                    self.last_prices[symbol] = df['close'].iloc[-1]
                else:
                    self.logger.logger.warning(f"No data available for {symbol}")
            
            except Exception as e:
                self.logger.logger.error(f"Error getting data for {symbol}: {e}")
        
        return market_data
    
    def generate_signals(self, market_data: dict) -> dict:
        """
        Generate trading signals using loaded models.
        
        Args:
            market_data: Dictionary of symbol -> DataFrame
            
        Returns:
            Dictionary of symbol -> signal (-1, 0, 1)
        """
        signals = {}
        
        for symbol, df in market_data.items():
            try:
                signal = 0  # Default: hold
                
                if df.empty:
                    signals[symbol] = signal
                    continue
                
                # Prepare features
                feature_names = self.feature_engine.get_feature_names(df)
                features = df[feature_names].dropna()
                
                if features.empty:
                    signals[symbol] = signal
                    continue
                
                # Get latest features
                latest_features = features.iloc[-1:].values
                
                # Ensemble prediction
                predictions = []
                
                # GRU prediction
                if 'gru' in self.models:
                    try:
                        # Prepare sequence for GRU
                        sequence_length = self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
                        if len(features) >= sequence_length:
                            # Scale features
                            features_scaled = self.preprocessor.fit_transform(features)
                            
                            # Create sequence
                            sequence = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
                            
                            # Predict
                            gru_pred = self.models['gru'].predict(sequence)[0]
                            predictions.append(('gru', gru_pred))
                    except Exception as e:
                        self.logger.logger.debug(f"GRU prediction failed for {symbol}: {e}")
                
                # LightGBM prediction
                if 'lightgbm' in self.models:
                    try:
                        lgbm_pred = self.models['lightgbm'].predict(latest_features)[0]
                        predictions.append(('lightgbm', lgbm_pred))
                    except Exception as e:
                        self.logger.logger.debug(f"LightGBM prediction failed for {symbol}: {e}")
                
                # PPO prediction (if available)
                if 'ppo' in self.models:
                    try:
                        # For PPO, we need to prepare the observation space
                        # Use the latest features as observation
                        observation = latest_features[-1:]  # Get last row as observation
                        if observation.shape[1] > 0:  # Ensure we have features
                            ppo_action, _ = self.models['ppo'].predict(observation, deterministic=True)
                            # Convert action to prediction value
                            # Action 0 = Hold, 1 = Buy, 2 = Sell
                            if ppo_action == 1:
                                ppo_pred = 0.002  # Strong buy signal
                            elif ppo_action == 2:
                                ppo_pred = -0.002  # Strong sell signal
                            else:
                                ppo_pred = 0.0  # Hold signal
                            predictions.append(('ppo', ppo_pred))
                    except Exception as e:
                        self.logger.logger.debug(f"PPO prediction failed for {symbol}: {e}")
                
                # Convert predictions to signals
                if predictions:
                    # Simple ensemble: average predictions
                    avg_prediction = np.mean([pred for _, pred in predictions])
                    
                    # Convert to signal
                    threshold = 0.001  # 0.1% threshold
                    if avg_prediction > threshold:
                        signal = 1  # Buy
                    elif avg_prediction < -threshold:
                        signal = -1  # Sell
                    else:
                        signal = 0  # Hold
                
                signals[symbol] = signal
                
                if predictions and signal != 0:
                    avg_prediction = np.mean([pred for _, pred in predictions])
                    self.logger.logger.info(f"Signal for {symbol}: {signal} (prediction: {avg_prediction:.6f})")
                elif not predictions:
                    self.logger.logger.debug(f"No predictions available for {symbol}")
            
            except Exception as e:
                self.logger.logger.error(f"Error generating signal for {symbol}: {e}")
                signals[symbol] = 0
        
        return signals
    
    def execute_trades(self, signals: dict):
        """
        Execute trades based on signals.
        
        Args:
            signals: Dictionary of symbol -> signal
        """
        for symbol, signal in signals.items():
            if signal == 0:
                continue
            
            try:
                current_price = self.last_prices.get(symbol)
                if not current_price:
                    continue
                
                current_position = self.positions.get(symbol, 0.0)
                
                # Calculate target position
                if signal == 1:  # Buy
                    target_position = self.max_position_size * self.portfolio_value / current_price
                elif signal == -1:  # Sell
                    target_position = -self.max_position_size * self.portfolio_value / current_price
                else:
                    target_position = 0.0
                
                # Calculate trade size
                trade_size = target_position - current_position
                
                if abs(trade_size) < 0.001:  # Minimum trade size
                    continue
                
                # Execute trade
                trade_value = abs(trade_size) * current_price
                fee = trade_value * self.transaction_fee
                
                # Update portfolio
                if trade_size > 0:  # Buying
                    self.portfolio_value -= (trade_value + fee)
                    side = "BUY"
                else:  # Selling
                    self.portfolio_value += (trade_value - fee)
                    side = "SELL"
                
                # Update position
                self.positions[symbol] = target_position
                
                # Log trade
                self.logger.log_trade(
                    symbol=symbol,
                    side=side,
                    quantity=abs(trade_size),
                    price=current_price,
                    portfolio_value=self.portfolio_value
                )
                
                # Send notification
                if self.notifier and self.notifier.enabled:
                    asyncio.create_task(self.notifier.send_trade_notification(
                        symbol=symbol,
                        side=side,
                        quantity=abs(trade_size),
                        price=current_price,
                        portfolio_value=self.portfolio_value
                    ))
            
            except Exception as e:
                self.logger.logger.error(f"Error executing trade for {symbol}: {e}")
    
    def calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value including positions."""
        total_value = self.portfolio_value
        
        for symbol, quantity in self.positions.items():
            if abs(quantity) > 1e-6:  # Non-zero position
                current_price = self.last_prices.get(symbol, 0)
                total_value += quantity * current_price
        
        return total_value
    
    async def send_daily_report(self):
        """Send daily portfolio report."""
        try:
            current_value = self.calculate_portfolio_value()
            daily_pnl = current_value - self.daily_start_value
            daily_return = daily_pnl / self.daily_start_value
            
            # Log portfolio update
            self.logger.log_portfolio_update(
                portfolio_value=current_value,
                daily_pnl=daily_pnl,
                positions=self.positions
            )
            
            # Send notification
            if self.notifier and self.notifier.enabled:
                await self.notifier.send_portfolio_update(
                    portfolio_value=current_value,
                    daily_pnl=daily_pnl,
                    daily_return=daily_return,
                    positions=self.positions
                )
            
            # Reset daily start value
            self.daily_start_value = current_value
        
        except Exception as e:
            self.logger.logger.error(f"Error sending daily report: {e}")
    
    async def run_trading_loop(self, interval_minutes: int = 15):
        """
        Main trading loop.
        
        Args:
            interval_minutes: Trading interval in minutes
        """
        self.logger.logger.info(f"Starting paper trading loop (interval: {interval_minutes} minutes)")
        
        if self.notifier and self.notifier.enabled:
            await self.notifier.send_system_status(
                status="RUNNING",
                active_models=list(self.models.keys())
            )
        
        last_daily_report = datetime.now().date()
        
        try:
            while True:
                loop_start = time.time()
                
                try:
                    # Get market data
                    market_data = await self.get_market_data()
                    
                    if market_data:
                        # Generate signals
                        signals = self.generate_signals(market_data)
                        
                        # Execute trades
                        self.execute_trades(signals)
                        
                        # Send daily report if needed
                        current_date = datetime.now().date()
                        if current_date > last_daily_report:
                            await self.send_daily_report()
                            last_daily_report = current_date
                    
                    else:
                        self.logger.logger.warning("No market data available")
                
                except Exception as e:
                    self.logger.logger.error(f"Error in trading loop: {e}")
                    
                    if self.notifier and self.notifier.enabled:
                        await self.notifier.send_error_notification(
                            error_type="TRADING_LOOP_ERROR",
                            error_message=str(e)
                        )
                
                # Wait for next interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, interval_minutes * 60 - loop_duration)
                
                self.logger.logger.debug(f"Loop completed in {loop_duration:.2f}s, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.logger.logger.info("Trading loop interrupted by user")
            
            if self.notifier and self.notifier.enabled:
                await self.notifier.send_system_status(status="STOPPED")
        
        except Exception as e:
            self.logger.logger.error(f"Trading loop failed: {e}")
            
            if self.notifier and self.notifier.enabled:
                await self.notifier.send_error_notification(
                    error_type="TRADING_LOOP_FAILURE",
                    error_message=str(e)
                )
            
            raise

def load_config(config_path: str = "src/config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Crypto Trading Bot - Paper Trader')
    parser.add_argument('--config', type=str, default='src/config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory containing trained models')
    parser.add_argument('--interval', type=int, default=15,
                       help='Trading interval in minutes')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("Failed to load configuration. Exiting.")
        return
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("CRYPTO TRADING BOT - PAPER TRADER")
    logger.info("=" * 60)
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Trading interval: {args.interval} minutes")
    
    # Initialize and run paper trader
    try:
        trader = PaperTrader(config, args.models_dir)
        await trader.run_trading_loop(args.interval)
    
    except Exception as e:
        logger.error(f"Paper trader failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())