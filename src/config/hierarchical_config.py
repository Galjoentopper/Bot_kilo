"""
Hierarchical Configuration System
================================

Implements a robust configuration hierarchy for financial ML models:
1. Base Config: Default values and sensible defaults
2. Validation: Parameter validation and constraint checking  
3. Optuna Override: Hyperparameter optimization integration
4. Manual Override: User-specified parameter overrides

This system ensures configuration consistency, prevents invalid parameters,
and provides flexibility for both automated optimization and manual tuning.
"""

import yaml
import json
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from copy import deepcopy
import numpy as np

logger = logging.getLogger(__name__)

class ConfigPriority(Enum):
    """Configuration priority levels (higher numbers take precedence)."""
    BASE = 1           # Base defaults
    FILE = 2           # Configuration file
    ENVIRONMENT = 3    # Environment variables
    OPTUNA = 4         # Hyperparameter optimization
    MANUAL = 5         # Manual overrides (highest priority)

class ValidationLevel(Enum):
    """Configuration validation strictness levels."""
    PERMISSIVE = "permissive"   # Warn but allow invalid values
    STRICT = "strict"           # Reject invalid configurations
    FINANCIAL = "financial"     # Financial ML specific validation

@dataclass
class ConfigValidationRule:
    """Individual configuration validation rule."""
    parameter: str
    rule_type: str  # "range", "choice", "type", "custom"
    constraint: Any
    error_message: str
    level: ValidationLevel = ValidationLevel.STRICT

@dataclass 
class BaseGRUConfig:
    """Base GRU model configuration with financial ML defaults."""
    
    # Model Architecture
    sequence_length: int = 20
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    output_size: int = 1
    
    # Training Parameters
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    
    # Optimizer Settings
    optimizer: str = "Adam"
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam
    
    # Regularization
    gradient_clip_norm: float = 0.5
    loss_function: str = "mse"
    
    # Stability Settings (Financial ML Critical)
    mixed_precision: bool = False
    deterministic: bool = True
    seed: int = 42
    
    # Data Processing
    feature_count: int = 114
    target_type: str = "return"
    target_horizon: int = 1
    
    # Advanced Settings
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-7

@dataclass
class TrainingConfig:
    """Training environment configuration."""
    
    # System Settings
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Logging and Monitoring
    log_level: str = "INFO"
    mlflow_tracking: bool = True
    save_best_model: bool = True
    
    # Validation Settings
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Data Settings
    data_dir: str = "./data"
    cache_dir: str = "./models/metadata"
    model_save_dir: str = "./models/saved"

@dataclass
class HierarchicalConfig:
    """Complete hierarchical configuration combining all components."""
    
    gru: BaseGRUConfig = field(default_factory=BaseGRUConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Metadata
    config_version: str = "1.0"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    source_priority: Dict[str, str] = field(default_factory=dict)

class ConfigurationManager:
    """
    Manages hierarchical configuration with validation, overrides, and optimization integration.
    """
    
    def __init__(self, 
                 base_config_path: Optional[str] = None,
                 validation_level: ValidationLevel = ValidationLevel.FINANCIAL):
        """
        Initialize configuration manager.
        
        Args:
            base_config_path: Path to base configuration file
            validation_level: Validation strictness level
        """
        self.validation_level = validation_level
        self.validation_rules = self._initialize_validation_rules()
        
        # Configuration stack (priority order)
        self.config_stack: Dict[ConfigPriority, Dict[str, Any]] = {}
        
        # Initialize base configuration
        self.base_config = HierarchicalConfig()
        self.config_stack[ConfigPriority.BASE] = asdict(self.base_config)
        
        # Load file-based configuration if provided
        if base_config_path and Path(base_config_path).exists():
            self.load_config_file(base_config_path)
        
        logger.info(f"ConfigurationManager initialized with {validation_level.value} validation")
    
    def _initialize_validation_rules(self) -> List[ConfigValidationRule]:
        """Initialize comprehensive validation rules for financial ML."""
        return [
            # Model Architecture Validation
            ConfigValidationRule(
                parameter="gru.sequence_length",
                rule_type="range",
                constraint=(5, 100),
                error_message="Sequence length must be between 5 and 100 for financial stability",
                level=ValidationLevel.FINANCIAL
            ),
            ConfigValidationRule(
                parameter="gru.hidden_size", 
                rule_type="choice",
                constraint=[16, 32, 48, 64, 96, 128, 192, 256, 384, 512],
                error_message="Hidden size must be a power-of-2-friendly value for efficiency"
            ),
            ConfigValidationRule(
                parameter="gru.num_layers",
                rule_type="range", 
                constraint=(1, 4),
                error_message="Number of layers should be 1-4 for financial data"
            ),
            ConfigValidationRule(
                parameter="gru.dropout",
                rule_type="range",
                constraint=(0.0, 0.8),
                error_message="Dropout must be between 0.0 and 0.8"
            ),
            
            # Learning Parameters Validation
            ConfigValidationRule(
                parameter="gru.learning_rate",
                rule_type="range", 
                constraint=(1e-6, 0.01),
                error_message="Learning rate must be between 1e-6 and 0.01 for financial stability",
                level=ValidationLevel.FINANCIAL
            ),
            ConfigValidationRule(
                parameter="gru.batch_size",
                rule_type="choice",
                constraint=[8, 16, 32, 64, 128, 256],
                error_message="Batch size should be a power of 2"
            ),
            ConfigValidationRule(
                parameter="gru.gradient_clip_norm",
                rule_type="range",
                constraint=(0.01, 5.0), 
                error_message="Gradient clipping norm should be between 0.01 and 5.0"
            ),
            
            # Financial ML Critical Validations
            ConfigValidationRule(
                parameter="gru.mixed_precision", 
                rule_type="custom",
                constraint=lambda x: x is False,
                error_message="Mixed precision must be disabled for financial ML stability",
                level=ValidationLevel.FINANCIAL
            ),
            ConfigValidationRule(
                parameter="gru.deterministic",
                rule_type="custom", 
                constraint=lambda x: x is True,
                error_message="Deterministic training is required for reproducible financial ML",
                level=ValidationLevel.FINANCIAL
            ),
            
            # Optimizer Validation
            ConfigValidationRule(
                parameter="gru.optimizer",
                rule_type="choice",
                constraint=["Adam", "AdamW", "RMSprop", "SGD"],
                error_message="Optimizer must be one of: Adam, AdamW, RMSprop, SGD"
            ),
            ConfigValidationRule(
                parameter="gru.weight_decay",
                rule_type="range",
                constraint=(0.0, 0.1),
                error_message="Weight decay should be between 0.0 and 0.1"
            ),
            
            # Training Configuration Validation  
            ConfigValidationRule(
                parameter="training.validation_split",
                rule_type="range",
                constraint=(0.1, 0.4),
                error_message="Validation split should be between 10% and 40%"
            ),
            ConfigValidationRule(
                parameter="training.test_split", 
                rule_type="range",
                constraint=(0.05, 0.3),
                error_message="Test split should be between 5% and 30%"
            )
        ]
    
    def load_config_file(self, config_path: str, priority: ConfigPriority = ConfigPriority.FILE):
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_path: Path to configuration file
            priority: Configuration priority level
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            self.config_stack[priority] = file_config
            logger.info(f"Loaded configuration from {config_path} with priority {priority.name}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def apply_optuna_override(self, trial_params: Dict[str, Any]):
        """
        Apply Optuna hyperparameter optimization overrides.
        
        Args:
            trial_params: Parameters suggested by Optuna trial
        """
        # Convert flat Optuna parameters to hierarchical structure
        hierarchical_params = self._flatten_to_hierarchical(trial_params)
        
        self.config_stack[ConfigPriority.OPTUNA] = hierarchical_params
        logger.info(f"Applied Optuna overrides: {list(trial_params.keys())}")
    
    def apply_manual_override(self, overrides: Dict[str, Any]):
        """
        Apply manual parameter overrides (highest priority).
        
        Args:
            overrides: Manual parameter overrides
        """
        self.config_stack[ConfigPriority.MANUAL] = overrides
        logger.info(f"Applied manual overrides: {list(overrides.keys())}")
    
    def get_merged_config(self) -> HierarchicalConfig:
        """
        Get final merged configuration applying all overrides in priority order.
        
        Returns:
            Final merged configuration
        """
        # Start with empty config
        merged = {}
        
        # Apply configurations in priority order
        for priority in sorted(self.config_stack.keys(), key=lambda x: x.value):
            config = self.config_stack[priority]
            merged = self._deep_merge(merged, config)
        
        # Convert back to structured config
        try:
            # Create configuration objects from merged dict
            gru_config = BaseGRUConfig(**merged.get('gru', {}))
            training_config = TrainingConfig(**merged.get('training', {}))
            
            final_config = HierarchicalConfig(
                gru=gru_config,
                training=training_config,
                config_version=merged.get('config_version', '1.0'),
                created_at=merged.get('created_at'),
                updated_at=merged.get('updated_at')
            )
            
            return final_config
            
        except Exception as e:
            logger.error(f"Failed to create structured config: {e}")
            # Fallback to base config
            return self.base_config
    
    def validate_config(self, config: Optional[HierarchicalConfig] = None) -> Tuple[bool, List[str]]:
        """
        Validate configuration against all rules.
        
        Args:
            config: Configuration to validate (uses merged config if None)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if config is None:
            config = self.get_merged_config()
        
        config_dict = asdict(config)
        errors = []
        warnings = []
        
        for rule in self.validation_rules:
            # Skip rules not applicable to current validation level
            if (self.validation_level != ValidationLevel.FINANCIAL and 
                rule.level == ValidationLevel.FINANCIAL):
                continue
                
            try:
                value = self._get_nested_value(config_dict, rule.parameter)
                is_valid = self._validate_parameter(value, rule)
                
                if not is_valid:
                    if rule.level == ValidationLevel.PERMISSIVE:
                        warnings.append(f"WARNING: {rule.error_message} (value: {value})")
                    else:
                        errors.append(f"ERROR: {rule.error_message} (value: {value})")
                        
            except KeyError:
                if rule.level != ValidationLevel.PERMISSIVE:
                    errors.append(f"ERROR: Required parameter '{rule.parameter}' not found")
            except Exception as e:
                errors.append(f"ERROR: Validation failed for '{rule.parameter}': {e}")
        
        # Financial ML specific validations
        if self.validation_level == ValidationLevel.FINANCIAL:
            financial_errors = self._validate_financial_constraints(config)
            errors.extend(financial_errors)
        
        # Log validation results
        if errors:
            logger.error(f"Configuration validation failed with {len(errors)} errors")
            for error in errors:
                logger.error(error)
        
        if warnings:
            logger.warning(f"Configuration validation has {len(warnings)} warnings")
            for warning in warnings:
                logger.warning(warning)
        
        return len(errors) == 0, errors + warnings
    
    def _validate_financial_constraints(self, config: HierarchicalConfig) -> List[str]:
        """Apply financial ML specific constraint validation."""
        errors = []
        
        # Risk management constraints
        if config.gru.learning_rate > 0.001:
            errors.append("Learning rate too high for financial ML stability")
        
        if config.gru.dropout < 0.2:
            errors.append("Insufficient regularization (dropout < 0.2) for financial data")
        
        if config.gru.gradient_clip_norm > 2.0:
            errors.append("Gradient clipping norm too high, may cause training instability")
        
        # Architecture constraints for financial data
        if config.gru.hidden_size > 256 and config.gru.num_layers > 2:
            errors.append("Model too complex for financial data, risk of overfitting")
        
        if config.gru.sequence_length > 60:
            errors.append("Sequence length too long, financial patterns are typically short-term")
        
        # Training constraints
        total_split = config.training.validation_split + config.training.test_split
        if total_split > 0.5:
            errors.append("Combined validation + test split too large, insufficient training data")
        
        return errors
    
    def _validate_parameter(self, value: Any, rule: ConfigValidationRule) -> bool:
        """Validate individual parameter against rule."""
        if rule.rule_type == "range":
            min_val, max_val = rule.constraint
            return min_val <= value <= max_val
            
        elif rule.rule_type == "choice":
            return value in rule.constraint
            
        elif rule.rule_type == "type":
            return isinstance(value, rule.constraint)
            
        elif rule.rule_type == "custom":
            if callable(rule.constraint):
                return rule.constraint(value)
            else:
                return bool(rule.constraint)
        
        return True
    
    def _get_nested_value(self, config_dict: Dict, parameter_path: str) -> Any:
        """Get nested parameter value using dot notation."""
        keys = parameter_path.split('.')
        value = config_dict
        
        for key in keys:
            value = value[key]
        
        return value
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries with override taking precedence."""
        result = base.copy()
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _flatten_to_hierarchical(self, flat_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat parameter dict to hierarchical structure."""
        hierarchical = {}
        
        for param_name, value in flat_params.items():
            if '.' in param_name:
                # Split nested parameter name
                parts = param_name.split('.')
                current = hierarchical
                
                # Navigate/create nested structure
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set final value
                current[parts[-1]] = value
            else:
                hierarchical[param_name] = value
        
        return hierarchical
    
    def save_config(self, config: HierarchicalConfig, filepath: str, format: str = "yaml"):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            filepath: Output file path
            format: File format ("yaml" or "json")
        """
        config_dict = asdict(config)
        
        # Add metadata
        config_dict['saved_at'] = pd.Timestamp.now().isoformat()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w') as f:
                if format.lower() == "yaml":
                    yaml.dump(config_dict, f, indent=2, default_flow_style=False)
                elif format.lower() == "json": 
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration hierarchy."""
        merged = self.get_merged_config()
        is_valid, messages = self.validate_config(merged)
        
        return {
            'validation_status': 'VALID' if is_valid else 'INVALID',
            'validation_level': self.validation_level.value,
            'active_priorities': [p.name for p in self.config_stack.keys()],
            'validation_messages': messages,
            'parameter_count': {
                'gru_params': len(asdict(merged.gru)),
                'training_params': len(asdict(merged.training))
            },
            'key_settings': {
                'learning_rate': merged.gru.learning_rate,
                'hidden_size': merged.gru.hidden_size,
                'dropout': merged.gru.dropout,
                'mixed_precision': merged.gru.mixed_precision,
                'deterministic': merged.gru.deterministic
            }
        }

def create_financial_config_manager(
    config_file: Optional[str] = None,
    validation_level: str = "financial"
) -> ConfigurationManager:
    """
    Convenience function to create a financial ML configuration manager.
    
    Args:
        config_file: Optional configuration file path
        validation_level: "permissive", "strict", or "financial"
        
    Returns:
        Configured ConfigurationManager instance
    """
    validation_enum = ValidationLevel(validation_level.lower())
    
    manager = ConfigurationManager(
        base_config_path=config_file,
        validation_level=validation_enum
    )
    
    return manager

# Example usage patterns
if __name__ == "__main__":
    # Example 1: Basic usage
    config_manager = create_financial_config_manager()
    
    # Example 2: Manual override
    config_manager.apply_manual_override({
        'gru': {
            'learning_rate': 0.0001,
            'hidden_size': 128
        }
    })
    
    # Example 3: Get final configuration
    final_config = config_manager.get_merged_config()
    is_valid, messages = config_manager.validate_config(final_config)
    
    if is_valid:
        print("Configuration is valid!")
    else:
        print("Configuration errors:", messages)
    
    # Example 4: Configuration summary
    summary = config_manager.get_config_summary()
    print("Config Summary:", summary)