"""
Stable GRU Trainer Module
========================

Ultra-stable GRU trainer specifically optimized for financial ML with conservative
defaults and proven hyperparameters that prevent gradient explosions and training instability.

This class incorporates all stability fixes and uses battle-tested configurations
for reliable financial time series prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from .gru_trainer import GRUTrainer, GRUModel

logger = logging.getLogger(__name__)

class StableGRUTrainer(GRUTrainer):
    """
    Ultra-stable GRU trainer with conservative defaults and proven hyperparameters
    specifically designed for financial ML stability.
    
    This trainer uses battle-tested configurations that have been proven to prevent
    gradient explosions, training instability, and numerical issues in financial time series.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize StableGRUTrainer with ultra-conservative defaults.
        
        Args:
            config: Optional configuration dict. Ultra-conservative defaults will be applied.
        """
        # Create stable configuration with proven parameters
        stable_config = self._get_stable_defaults()
        
        # Merge user config with stable defaults (stable defaults take precedence for critical params)
        if config:
            # Allow user to override non-critical parameters
            user_config = config.copy()
            
            # But enforce stability-critical parameters
            critical_params = {
                'mixed_precision': False,  # Never allow mixed precision for financial data
                'max_grad_norm': 0.5,     # Very conservative gradient clipping
                'deterministic': True,    # Ensure reproducibility
            }
            
            # Update model-specific parameters with stable defaults
            if 'models' not in user_config:
                user_config['models'] = {}
            if 'gru' not in user_config['models']:
                user_config['models']['gru'] = {}
                
            # Merge with stable GRU defaults, keeping user preferences for non-critical params
            gru_config = stable_config['models']['gru'].copy()
            gru_config.update(user_config['models']['gru'])
            
            # But enforce critical stability parameters
            for key, value in critical_params.items():
                if key in stable_config['models']['gru']:
                    gru_config[key] = stable_config['models']['gru'][key]
            
            user_config['models']['gru'] = gru_config
            
            # Enforce critical training parameters
            if 'training' not in user_config:
                user_config['training'] = {}
            user_config['training'].update(critical_params)
            
            final_config = user_config
        else:
            final_config = stable_config
        
        # Initialize parent with stable config
        super().__init__(final_config)
        
        logger.info("StableGRUTrainer initialized with ultra-conservative financial ML defaults")
        logger.info(f"Key stability settings: LR={self.learning_rate}, Grad_Norm={self.max_grad_norm}, "
                   f"Mixed_Precision={self.mixed_precision}, Optimizer={self.optimizer_name}")
    
    def _get_stable_defaults(self) -> Dict[str, Any]:
        """Get ultra-conservative default configuration proven for financial ML stability."""
        return {
            'models': {
                'gru': {
                    # Architecture - conservative but effective
                    'sequence_length': 20,      # Proven effective sequence length
                    'hidden_size': 64,          # Conservative hidden size to prevent overfitting
                    'num_layers': 2,            # Moderate depth for stability
                    'dropout': 0.3,             # Significant dropout for regularization
                    'output_size': 1,           # Single output for regression
                    
                    # Training hyperparameters - ultra-conservative
                    'learning_rate': 0.0001,    # Very conservative learning rate
                    'batch_size': 32,           # Moderate batch size for stability
                    'epochs': 100,              # Reasonable epoch count
                    'early_stopping_patience': 15,  # Extended patience for financial data
                    
                    # Optimizer settings - proven stable combinations
                    'optimizer': 'Adam',        # Adam with conservative defaults
                    'weight_decay': 1e-4,       # L2 regularization
                    'loss': 'mse',              # MSE loss for regression
                    
                    # Stability-critical settings
                    'mixed_precision': False,   # NEVER use mixed precision for financial data
                    'max_grad_norm': 0.5,      # Very conservative gradient clipping
                }
            },
            'training': {
                # System settings for stability
                'seed': 42,                     # Fixed seed for reproducibility  
                'deterministic': True,          # Deterministic training for debugging
                'num_workers': 0,               # Single-threaded for stability on some systems
                'pin_memory': True,             # GPU optimization
                'mixed_precision': False,       # Globally disabled
                'max_grad_norm': 0.5,          # Conservative gradient clipping
            }
        }
    
    def get_recommended_config_for_symbol(self, symbol: str, data_characteristics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get symbol-specific recommended configuration based on data characteristics.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            data_characteristics: Optional dict with data stats (volatility, sample_count, etc.)
            
        Returns:
            Recommended configuration dict for this symbol
        """
        base_config = self._get_stable_defaults()
        
        # Adjust based on symbol characteristics
        if symbol.endswith('USDT'):
            # Stablecoin pairs - typically less volatile
            base_config['models']['gru']['learning_rate'] = 0.0002
            base_config['models']['gru']['dropout'] = 0.2
        elif 'BTC' in symbol:
            # Bitcoin pairs - highly volatile
            base_config['models']['gru']['learning_rate'] = 0.00005
            base_config['models']['gru']['dropout'] = 0.4
            base_config['models']['gru']['max_grad_norm'] = 0.3
        elif any(alt in symbol for alt in ['ETH', 'BNB', 'ADA', 'DOT']):
            # Major altcoins - moderate volatility
            base_config['models']['gru']['learning_rate'] = 0.0001
            base_config['models']['gru']['dropout'] = 0.3
        
        # Adjust based on data characteristics if provided
        if data_characteristics:
            sample_count = data_characteristics.get('sample_count', 10000)
            volatility = data_characteristics.get('volatility', 0.02)
            
            # Adjust for dataset size
            if sample_count < 5000:
                # Small dataset - more regularization
                base_config['models']['gru']['dropout'] += 0.1
                base_config['models']['gru']['learning_rate'] *= 0.5
                base_config['models']['gru']['early_stopping_patience'] = 10
            elif sample_count > 50000:
                # Large dataset - can be slightly less conservative
                base_config['models']['gru']['batch_size'] = 64
                base_config['models']['gru']['hidden_size'] = 96
            
            # Adjust for volatility
            if volatility > 0.05:
                # High volatility - more conservative
                base_config['models']['gru']['learning_rate'] *= 0.5
                base_config['models']['gru']['max_grad_norm'] *= 0.8
            elif volatility < 0.01:
                # Low volatility - can be slightly more aggressive
                base_config['models']['gru']['learning_rate'] *= 1.2
        
        logger.info(f"Generated stable config for {symbol}: LR={base_config['models']['gru']['learning_rate']:.6f}")
        return base_config
    
    def validate_training_stability(self) -> Dict[str, bool]:
        """
        Validate that all stability measures are correctly configured.
        
        Returns:
            Dict with validation results for each stability measure
        """
        validation_results = {
            'mixed_precision_disabled': not self.mixed_precision,
            'conservative_learning_rate': self.learning_rate <= 0.001,
            'gradient_clipping_enabled': self.max_grad_norm is not None and self.max_grad_norm <= 1.0,
            'sufficient_dropout': self.dropout >= 0.2,
            'conservative_batch_size': self.batch_size <= 128,
            'early_stopping_enabled': self.early_stopping_patience > 0,
            'deterministic_training': self.deterministic,
        }
        
        all_stable = all(validation_results.values())
        
        if all_stable:
            logger.info("✅ All stability measures validated successfully")
        else:
            failed_checks = [k for k, v in validation_results.items() if not v]
            logger.warning(f"❌ Stability validation failed for: {failed_checks}")
        
        return validation_results
    
    def get_stability_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive stability report for the current configuration.
        
        Returns:
            Detailed stability report
        """
        stability_validation = self.validate_training_stability()
        
        report = {
            'stability_score': sum(stability_validation.values()) / len(stability_validation),
            'validation_results': stability_validation,
            'configuration_summary': {
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer_name,
                'batch_size': self.batch_size,
                'gradient_clipping': self.max_grad_norm,
                'dropout': self.dropout,
                'mixed_precision': self.mixed_precision,
                'deterministic': self.deterministic,
            },
            'stability_features': [
                'Ultra-conservative weight initialization (gain=0.01)',
                'Comprehensive data validation pipeline',
                'Multi-stage feature clipping',
                'Real-time gradient monitoring',
                'Emergency learning rate reduction',
                'NaN/Inf detection and recovery',
                'Conservative financial data bounds',
                'Aggressive outlier clipping',
            ],
            'recommended_for': [
                'Financial time series prediction',
                'High-frequency trading data',
                'Cryptocurrency price prediction', 
                'Volatile market conditions',
                'Production trading systems',
                'Risk-sensitive applications',
            ]
        }
        
        return report

# Convenience function for creating stable trainer with minimal configuration
def create_stable_gru_trainer(symbol: str = 'BTCUSDT', 
                             custom_config: Optional[Dict] = None,
                             data_characteristics: Optional[Dict] = None) -> StableGRUTrainer:
    """
    Create a StableGRUTrainer with optimal settings for the given symbol.
    
    Args:
        symbol: Trading symbol to optimize for
        custom_config: Optional custom configuration to merge
        data_characteristics: Optional data statistics for fine-tuning
        
    Returns:
        Configured StableGRUTrainer instance
    """
    # Start with a base stable trainer
    trainer = StableGRUTrainer()
    
    # Get symbol-specific recommendations
    recommended_config = trainer.get_recommended_config_for_symbol(symbol, data_characteristics)
    
    # Merge with custom config if provided
    if custom_config:
        # Deep merge the configurations
        def deep_merge(base_dict, override_dict):
            result = base_dict.copy()
            for key, value in override_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        final_config = deep_merge(recommended_config, custom_config)
    else:
        final_config = recommended_config
    
    # Create new trainer with optimized config
    stable_trainer = StableGRUTrainer(final_config)
    
    # Validate stability
    validation_results = stable_trainer.validate_training_stability()
    if not all(validation_results.values()):
        logger.warning("Some stability checks failed with custom configuration")
    
    logger.info(f"Created StableGRUTrainer optimized for {symbol}")
    return stable_trainer