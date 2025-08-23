#!/usr/bin/env python3
"""
Configuration Template Generator

This module provides utilities for generating customized configuration templates
based on user preferences, system capabilities, and trading requirements.
"""

import os
import yaml
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import psutil

@dataclass
class TradingPreferences:
    """User trading preferences."""
    symbols: List[str]
    risk_level: str  # 'conservative', 'moderate', 'aggressive'
    trading_style: str  # 'scalping', 'day_trading', 'swing_trading'
    max_position_size: float
    use_stop_loss: bool
    use_take_profit: bool
    notification_level: str  # 'minimal', 'normal', 'verbose'

@dataclass
class SystemCapabilities:
    """System capability information."""
    memory_gb: float
    cpu_cores: int
    gpu_available: bool
    storage_gb: float
    network_speed: str  # 'slow', 'medium', 'fast'
    is_dedicated: bool  # True if dedicated trading machine

class ConfigGenerator:
    """Generates customized configuration templates."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the configuration generator."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / 'config'
        self.templates_dir = self.config_dir / 'templates'
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Default symbol lists
        self.symbol_presets = {
            'crypto_major': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT'],
            'crypto_altcoins': ['LINKUSDT', 'LTCUSDT', 'XRPUSDT', 'SOLUSDT', 'AVAXUSDT'],
            'crypto_defi': ['UNIUSDT', 'AAVEUSDT', 'SUSHIUSDT', 'COMPUSDT', 'MKRUSDT'],
            'crypto_small': ['BTCUSDT', 'ETHUSDT'],
            'crypto_medium': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT'],
            'crypto_large': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'XRPUSDT', 'SOLUSDT', 'AVAXUSDT']
        }
    
    def detect_system_capabilities(self) -> SystemCapabilities:
        """Detect current system capabilities."""
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_cores = psutil.cpu_count()
        
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                pass
        
        # Estimate storage (simplified)
        try:
            storage_gb = psutil.disk_usage('.').total / (1024**3)
        except:
            storage_gb = 100  # Default assumption
        
        # Determine if this is likely a dedicated machine
        is_dedicated = memory_gb >= 8 and cpu_cores >= 4
        
        # Estimate network speed (simplified heuristic)
        network_speed = 'medium'  # Default assumption
        
        return SystemCapabilities(
            memory_gb=memory_gb,
            cpu_cores=cpu_cores,
            gpu_available=gpu_available,
            storage_gb=storage_gb,
            network_speed=network_speed,
            is_dedicated=is_dedicated
        )
    
    def generate_training_config(self, 
                               preferences: TradingPreferences,
                               capabilities: SystemCapabilities,
                               advanced_features: bool = True) -> Dict[str, Any]:
        """Generate training configuration based on preferences and capabilities."""
        
        # Base configuration
        config = {
            'data': {
                'symbols': preferences.symbols,
                'interval': '1h',
                'lookback_days': 365 if capabilities.storage_gb > 50 else 180,
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'cache_duration': '1h'
            },
            'training': {
                'cross_validation': {
                    'enabled': True,
                    'n_splits': 5 if capabilities.memory_gb >= 8 else 3,
                    'embargo_days': 7
                },
                'parallel': {
                    'n_workers': min(capabilities.cpu_cores, len(preferences.symbols)),
                    'batch_size': 64 if capabilities.memory_gb >= 8 else 32
                },
                'epochs': {
                    'gru': 100 if capabilities.gpu_available else 50,
                    'ppo': 50000 if capabilities.gpu_available else 25000
                },
                'optuna': {
                    'enabled': advanced_features,
                    'n_trials': 100 if capabilities.is_dedicated else 50,
                    'timeout_hours': 24 if capabilities.is_dedicated else 12
                }
            },
            'models': self._get_model_config(capabilities, advanced_features),
            'features': self._get_feature_config(advanced_features, capabilities),
            'backtesting': self._get_backtesting_config(preferences),
            'output': {
                'base_dir': 'models',
                'save_models': True,
                'save_metadata': True,
                'save_features': True,
                'save_preprocessors': True,
                'package_models': True,
                'create_transfer_bundle': True,
                'versioning': {
                    'enabled': True,
                    'keep_versions': 5
                }
            },
            'logging': self._get_logging_config(preferences.notification_level),
            'mlflow': {
                'enabled': advanced_features,
                'experiment_name': 'trading_bot_training',
                'tracking_uri': './mlruns'
            },
            'environment': {
                'type': 'training',
                'machine_id': platform.node(),
                'python_version': f"{platform.python_version()}",
                'required_memory': '4GB',
                'recommended_memory': '8GB',
                'gpu_recommended': True,
                'description': 'High-performance training configuration'
            }
        }
        
        return config
    
    def generate_trading_config(self,
                              preferences: TradingPreferences,
                              capabilities: SystemCapabilities) -> Dict[str, Any]:
        """Generate trading configuration based on preferences and capabilities."""
        
        # Determine update intervals based on trading style
        if preferences.trading_style == 'scalping':
            signal_interval = 60  # 1 minute
            execution_interval = 30  # 30 seconds
        elif preferences.trading_style == 'day_trading':
            signal_interval = 300  # 5 minutes
            execution_interval = 60  # 1 minute
        else:  # swing_trading
            signal_interval = 900  # 15 minutes
            execution_interval = 300  # 5 minutes
        
        config = {
            'data': {
                'symbols': preferences.symbols,
                'interval': '1h',
                'lookback_days': 30,  # Minimal for trading
                'cache_duration': '5m'
            },
            'trading': {
                'initial_balance': 10000,
                'position_size': preferences.max_position_size,
                'fees': {
                    'maker': 0.001,
                    'taker': 0.001
                },
                'slippage': 0.001,
                'risk_management': self._get_risk_management_config(preferences),
                'intervals': {
                    'signal_generation': signal_interval,
                    'order_execution': execution_interval,
                    'portfolio_update': 300
                }
            },
            'model_loading': {
                'ensemble_weights': {
                    'gru': 0.4,
                    'lightgbm': 0.4,
                    'ppo': 0.2
                },
                'loading_strategy': 'best_available',
                'validation_enabled': True,
                'search_paths': [
                    'models/packaged',
                    'models/imports',
                    'models/best_wf',
                    'models/latest',
                    'models/unified'
                ]
            },
            'features': self._get_lightweight_feature_config(),
            'data_sources': {
                'primary': 'binance',
                'fallback': 'ccxt',
                'timeout': 10,
                'retry_attempts': 3
            },
            'performance': {
                'max_workers': min(2, capabilities.cpu_cores),
                'batch_size': 16,
                'memory_limit_mb': int(capabilities.memory_gb * 512),  # Use half available memory
                'cache_size': 1000,
                'gc_interval': 300
            },
            'logging': self._get_logging_config(preferences.notification_level, trading=True),
            'monitoring': {
                'enabled': True,
                'update_interval': 300,
                'metrics': ['portfolio_value', 'positions', 'pnl', 'signals']
            },
            'notifications': {
                'telegram': {
                    'enabled': False,  # User needs to configure
                    'bot_token': '',
                    'chat_id': '',
                    'notification_level': preferences.notification_level
                }
            },
            'environment': {
                'type': 'trading',
                'machine_id': platform.node(),
                'python_version': f"{platform.python_version()}",
                'required_memory': '1GB',
                'recommended_memory': '2GB',
                'gpu_required': False,
                'description': 'Lightweight trading configuration'
            }
        }
        
        return config
    
    def _get_model_config(self, capabilities: SystemCapabilities, advanced: bool) -> Dict[str, Any]:
        """Get model configuration based on capabilities."""
        return {
            'lightgbm': {
                'n_estimators': 200 if advanced else 100,
                'max_depth': 8 if capabilities.memory_gb >= 8 else 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': capabilities.cpu_cores
            },
            'gru': {
                'hidden_size': 128 if capabilities.memory_gb >= 8 else 64,
                'num_layers': 3 if advanced else 2,
                'dropout': 0.2,
                'sequence_length': 60,
                'batch_size': 64 if capabilities.memory_gb >= 8 else 32,
                'learning_rate': 0.001,
                'device': 'cuda' if capabilities.gpu_available else 'cpu'
            },
            'ppo': {
                'policy': 'MlpPolicy',
                'learning_rate': 0.0003,
                'n_steps': 2048 if capabilities.memory_gb >= 8 else 1024,
                'batch_size': 64 if capabilities.memory_gb >= 8 else 32,
                'n_epochs': 10,
                'gamma': 0.99,
                'device': 'cuda' if capabilities.gpu_available else 'cpu'
            }
        }
    
    def _get_feature_config(self, advanced: bool, capabilities: SystemCapabilities) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        config = {
            'technical_indicators': {
                'sma_periods': [5, 10, 20, 50],
                'ema_periods': [12, 26],
                'rsi_period': 14,
                'macd': True,
                'bollinger_bands': True,
                'stochastic': True
            },
            'price_features': {
                'returns': [1, 5, 10],
                'log_returns': True,
                'price_ratios': True,
                'volatility': [10, 20]
            },
            'volume_features': {
                'volume_sma': [10, 20],
                'volume_ratio': True,
                'vwap': True
            }
        }
        
        if advanced and capabilities.memory_gb >= 4:
            config.update({
                'advanced_features': {
                    'fourier_transform': True,
                    'wavelet_transform': True,
                    'fractal_dimension': True,
                    'entropy_features': True
                },
                'statistical_features': {
                    'rolling_stats': [10, 20, 50],
                    'correlation_features': True,
                    'regime_detection': True
                }
            })
        
        return config
    
    def _get_lightweight_feature_config(self) -> Dict[str, Any]:
        """Get lightweight feature configuration for trading."""
        return {
            'technical_indicators': {
                'sma_periods': [10, 20],
                'ema_periods': [12, 26],
                'rsi_period': 14,
                'macd': True,
                'bollinger_bands': True
            },
            'price_features': {
                'returns': [1, 5],
                'log_returns': True,
                'volatility': [10]
            },
            'volume_features': {
                'volume_sma': [10],
                'volume_ratio': True
            },
            'advanced_features': {
                'enabled': False
            },
            'statistical_features': {
                'enabled': False
            }
        }
    
    def _get_backtesting_config(self, preferences: TradingPreferences) -> Dict[str, Any]:
        """Get backtesting configuration based on preferences."""
        return {
            'initial_balance': 10000,
            'position_size': preferences.max_position_size,
            'fees': {
                'maker': 0.001,
                'taker': 0.001
            },
            'slippage': 0.001,
            'risk_management': self._get_risk_management_config(preferences)
        }
    
    def _get_risk_management_config(self, preferences: TradingPreferences) -> Dict[str, Any]:
        """Get risk management configuration based on preferences."""
        if preferences.risk_level == 'conservative':
            return {
                'max_position_size': min(0.1, preferences.max_position_size),
                'stop_loss': 0.02 if preferences.use_stop_loss else None,
                'take_profit': 0.04 if preferences.use_take_profit else None,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.1
            }
        elif preferences.risk_level == 'moderate':
            return {
                'max_position_size': min(0.2, preferences.max_position_size),
                'stop_loss': 0.03 if preferences.use_stop_loss else None,
                'take_profit': 0.06 if preferences.use_take_profit else None,
                'max_daily_loss': 0.1,
                'max_drawdown': 0.15
            }
        else:  # aggressive
            return {
                'max_position_size': preferences.max_position_size,
                'stop_loss': 0.05 if preferences.use_stop_loss else None,
                'take_profit': 0.1 if preferences.use_take_profit else None,
                'max_daily_loss': 0.15,
                'max_drawdown': 0.2
            }
    
    def _get_logging_config(self, level: str, trading: bool = False) -> Dict[str, Any]:
        """Get logging configuration based on notification level."""
        if level == 'minimal':
            log_level = 'WARNING'
            console_level = 'ERROR'
        elif level == 'verbose':
            log_level = 'DEBUG'
            console_level = 'INFO'
        else:  # normal
            log_level = 'INFO'
            console_level = 'INFO'
        
        config = {
            'level': log_level,
            'console_level': console_level,
            'file_logging': True,
            'log_dir': 'logs',
            'max_file_size': '10MB',
            'backup_count': 5
        }
        
        if trading:
            config.update({
                'trade_logging': True,
                'performance_logging': True,
                'error_notifications': True
            })
        
        return config
    
    def interactive_config_generation(self) -> Dict[str, Any]:
        """Interactive configuration generation with user prompts."""
        print("\n=== Trading Bot Configuration Generator ===")
        print("This wizard will help you create a customized configuration.\n")
        
        # Environment type
        print("1. Environment Type:")
        print("   1) Training Computer (High Performance)")
        print("   2) Trading Computer (Lightweight)")
        env_choice = input("Select environment type (1-2): ").strip()
        is_training = env_choice == '1'
        
        # Symbol selection
        print("\n2. Trading Symbols:")
        print("   1) Crypto Major (5 symbols)")
        print("   2) Crypto Medium (7 symbols)")
        print("   3) Crypto Large (10 symbols)")
        print("   4) Custom symbols")
        symbol_choice = input("Select symbol set (1-4): ").strip()
        
        if symbol_choice == '4':
            symbols_input = input("Enter symbols (comma-separated, e.g., BTCUSDT,ETHUSDT): ")
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
        else:
            symbol_map = {
                '1': 'crypto_major',
                '2': 'crypto_medium',
                '3': 'crypto_large'
            }
            symbols = self.symbol_presets.get(symbol_map.get(symbol_choice, 'crypto_major'), self.symbol_presets['crypto_major'])
        
        # Risk level
        print("\n3. Risk Level:")
        print("   1) Conservative (Low risk, small positions)")
        print("   2) Moderate (Balanced risk/reward)")
        print("   3) Aggressive (High risk, large positions)")
        risk_choice = input("Select risk level (1-3): ").strip()
        risk_map = {'1': 'conservative', '2': 'moderate', '3': 'aggressive'}
        risk_level = risk_map.get(risk_choice, 'moderate')
        
        # Trading style
        print("\n4. Trading Style:")
        print("   1) Scalping (Very short-term, frequent trades)")
        print("   2) Day Trading (Intraday positions)")
        print("   3) Swing Trading (Multi-day positions)")
        style_choice = input("Select trading style (1-3): ").strip()
        style_map = {'1': 'scalping', '2': 'day_trading', '3': 'swing_trading'}
        trading_style = style_map.get(style_choice, 'day_trading')
        
        # Position size
        max_position = float(input("\n5. Maximum position size (0.01-1.0): ") or "0.1")
        max_position = max(0.01, min(1.0, max_position))
        
        # Risk management
        use_stop_loss = input("\n6. Use stop loss? (Y/n): ").strip().lower() != 'n'
        use_take_profit = input("7. Use take profit? (Y/n): ").strip().lower() != 'n'
        
        # Notification level
        print("\n8. Notification Level:")
        print("   1) Minimal (Errors only)")
        print("   2) Normal (Important events)")
        print("   3) Verbose (Detailed logging)")
        notif_choice = input("Select notification level (1-3): ").strip()
        notif_map = {'1': 'minimal', '2': 'normal', '3': 'verbose'}
        notification_level = notif_map.get(notif_choice, 'normal')
        
        # Advanced features (for training)
        advanced_features = True
        if is_training:
            advanced_features = input("\n9. Enable advanced features (Fourier, Wavelet, etc.)? (Y/n): ").strip().lower() != 'n'
        
        # Create preferences and capabilities
        preferences = TradingPreferences(
            symbols=symbols,
            risk_level=risk_level,
            trading_style=trading_style,
            max_position_size=max_position,
            use_stop_loss=use_stop_loss,
            use_take_profit=use_take_profit,
            notification_level=notification_level
        )
        
        capabilities = self.detect_system_capabilities()
        
        # Generate configuration
        if is_training:
            config = self.generate_training_config(preferences, capabilities, advanced_features)
        else:
            config = self.generate_trading_config(preferences, capabilities)
        
        return config
    
    def save_config(self, config: Dict[str, Any], filename: str) -> Path:
        """Save configuration to file."""
        config_path = self.project_root / filename
        
        # Backup existing config
        if config_path.exists():
            backup_path = config_path.with_suffix('.yaml.backup')
            config_path.rename(backup_path)
            print(f"Backed up existing config to {backup_path}")
        
        # Save new config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"Configuration saved to {config_path}")
        return config_path

def main():
    """Main function for interactive configuration generation."""
    generator = ConfigGenerator()
    
    try:
        config = generator.interactive_config_generation()
        
        print("\n=== Configuration Generated ===")
        print(f"Environment: {config['environment']['type']}")
        print(f"Symbols: {', '.join(config['data']['symbols'])}")
        print(f"Risk Level: {config.get('trading', {}).get('risk_management', {}).get('max_position_size', 'N/A')}")
        
        # Save configuration
        save_choice = input("\nSave this configuration? (Y/n): ").strip().lower()
        if save_choice != 'n':
            filename = input("Enter filename (default: config.yaml): ").strip() or 'config.yaml'
            generator.save_config(config, filename)
            
            print("\n=== Next Steps ===")
            if config['environment']['type'] == 'training':
                print("1. Review the generated config.yaml")
                print("2. Run setup_environment.bat to install dependencies")
                print("3. Use train_models.bat to start training")
            else:
                print("1. Review the generated config.yaml")
                print("2. Import models from your training computer")
                print("3. Run setup_environment.bat to install dependencies")
                print("4. Use deploy_trading.bat to start trading")
        
    except KeyboardInterrupt:
        print("\n\nConfiguration generation cancelled.")
    except Exception as e:
        print(f"\nError generating configuration: {e}")

if __name__ == '__main__':
    main()