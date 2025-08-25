#!/usr/bin/env python3
"""
Environment Configuration Manager

This module provides utilities for managing environment-specific configurations
for the distributed trading bot system. It handles switching between training
and trading configurations, validates environment compatibility, and manages
configuration templates.
"""

import os
import sys
import yaml
import shutil
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import psutil

@dataclass
class EnvironmentInfo:
    """Information about the current environment."""
    type: str  # 'training' or 'trading'
    machine_id: str
    python_version: str
    total_memory_gb: float
    cpu_count: int
    gpu_available: bool
    platform: str
    architecture: str

class EnvironmentManager:
    """Manages environment-specific configurations."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the environment manager."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / 'config'
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration file paths
        self.training_config = self.project_root / 'src' / 'config' / 'config_training.yaml'
        self.trading_config = self.project_root / 'src' / 'config' / 'config_trading.yaml'
        self.active_config = self.project_root / 'config.yaml'
        self.env_info_file = self.config_dir / 'environment_info.yaml'
        
        # Current environment info
        self.current_env = self._detect_environment()
    
    def _detect_environment(self) -> EnvironmentInfo:
        """Detect current environment characteristics."""
        # Get system information
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Check for GPU availability
        gpu_available = self._check_gpu_availability()
        
        # Determine environment type based on resources
        env_type = self._determine_environment_type(memory_gb, cpu_count, gpu_available)
        
        return EnvironmentInfo(
            type=env_type,
            machine_id=platform.node(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            total_memory_gb=memory_gb,
            cpu_count=cpu_count,
            gpu_available=gpu_available,
            platform=platform.system(),
            architecture=platform.architecture()[0]
        )
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            pass
        
        return False
    
    def _determine_environment_type(self, memory_gb: float, cpu_count: int, gpu_available: bool) -> str:
        """Determine environment type based on system resources."""
        # Heuristics for determining environment type
        if memory_gb >= 8 and cpu_count >= 4:
            return 'training'
        else:
            return 'trading'
    
    def get_environment_info(self) -> EnvironmentInfo:
        """Get current environment information."""
        return self.current_env
    
    def save_environment_info(self):
        """Save current environment information to file."""
        env_data = {
            'type': self.current_env.type,
            'machine_id': self.current_env.machine_id,
            'python_version': self.current_env.python_version,
            'total_memory_gb': self.current_env.total_memory_gb,
            'cpu_count': self.current_env.cpu_count,
            'gpu_available': self.current_env.gpu_available,
            'platform': self.current_env.platform,
            'architecture': self.current_env.architecture,
            'detected_at': self._get_timestamp()
        }
        
        with open(self.env_info_file, 'w') as f:
            yaml.dump(env_data, f, default_flow_style=False)
    
    def load_environment_info(self) -> Optional[Dict[str, Any]]:
        """Load saved environment information."""
        if self.env_info_file.exists():
            with open(self.env_info_file, 'r') as f:
                return yaml.safe_load(f)
        return None
    
    def setup_environment(self, env_type: str, force: bool = False) -> bool:
        """Setup environment configuration."""
        if env_type not in ['training', 'trading']:
            raise ValueError(f"Invalid environment type: {env_type}. Must be 'training' or 'trading'")
        
        # Check if environment is compatible
        if not force and not self._is_environment_compatible(env_type):
            print(f"Warning: Current system may not be optimal for {env_type} environment")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return False
        
        # Copy appropriate configuration
        if env_type == 'training':
            source_config = self.training_config
        else:
            source_config = self.trading_config
        
        if not source_config.exists():
            print(f"Error: Configuration template {source_config} not found")
            return False
        
        # Backup existing config if it exists
        if self.active_config.exists():
            backup_path = self.active_config.with_suffix('.yaml.backup')
            shutil.copy2(self.active_config, backup_path)
            print(f"Backed up existing config to {backup_path}")
        
        # Copy new configuration
        shutil.copy2(source_config, self.active_config)
        
        # Update environment info
        self.current_env.type = env_type
        self.save_environment_info()
        
        print(f"Environment configured for {env_type}")
        return True
    
    def _is_environment_compatible(self, env_type: str) -> bool:
        """Check if current environment is compatible with the requested type."""
        if env_type == 'training':
            # Training requires more resources
            return (
                self.current_env.total_memory_gb >= 4 and
                self.current_env.cpu_count >= 2
            )
        else:
            # Trading can run on minimal resources
            return (
                self.current_env.total_memory_gb >= 1 and
                self.current_env.cpu_count >= 1
            )
    
    def get_recommended_environment(self) -> str:
        """Get recommended environment type based on system resources."""
        return self.current_env.type
    
    def validate_configuration(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Validate configuration against current environment."""
        if config_path is None:
            config_path = self.active_config
        
        if not config_path.exists():
            return {'valid': False, 'errors': ['Configuration file not found']}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            return {'valid': False, 'errors': [f'Failed to load configuration: {e}']}
        
        errors = []
        warnings = []
        
        # Check environment compatibility
        env_config = config.get('environment', {})
        required_memory = self._parse_memory_string(env_config.get('required_memory', '1GB'))
        recommended_memory = self._parse_memory_string(env_config.get('recommended_memory', '2GB'))
        
        if self.current_env.total_memory_gb < required_memory:
            errors.append(f"Insufficient memory: {self.current_env.total_memory_gb:.1f}GB < {required_memory}GB required")
        elif self.current_env.total_memory_gb < recommended_memory:
            warnings.append(f"Below recommended memory: {self.current_env.total_memory_gb:.1f}GB < {recommended_memory}GB recommended")
        
        # Check Python version compatibility
        required_python = env_config.get('python_version', '3.8+')
        if not self._check_python_version(required_python):
            errors.append(f"Python version {self.current_env.python_version} does not meet requirement {required_python}")
        
        # Check GPU requirements
        if env_config.get('gpu_required', False) and not self.current_env.gpu_available:
            errors.append("GPU required but not available")
        elif env_config.get('gpu_recommended', False) and not self.current_env.gpu_available:
            warnings.append("GPU recommended but not available")
        
        # Check model directories
        models_dir = Path(config.get('model_management', {}).get('models_dir', 'models'))
        if not models_dir.exists():
            warnings.append(f"Models directory {models_dir} does not exist")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'environment_type': env_config.get('type', 'unknown'),
            'compatible': len(errors) == 0
        }
    
    def _parse_memory_string(self, memory_str: str) -> float:
        """Parse memory string (e.g., '4GB', '512MB') to GB."""
        memory_str = memory_str.upper().strip()
        
        if memory_str.endswith('GB'):
            return float(memory_str[:-2])
        elif memory_str.endswith('MB'):
            return float(memory_str[:-2]) / 1024
        elif memory_str.endswith('KB'):
            return float(memory_str[:-2]) / (1024 * 1024)
        else:
            # Assume GB if no unit
            return float(memory_str)
    
    def _check_python_version(self, required_version: str) -> bool:
        """Check if current Python version meets requirements."""
        if '+' in required_version:
            min_version = required_version.replace('+', '')
            min_parts = [int(x) for x in min_version.split('.')]
            current_parts = [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]
            
            for i, (current, minimum) in enumerate(zip(current_parts, min_parts)):
                if current > minimum:
                    return True
                elif current < minimum:
                    return False
            return True
        else:
            # Exact version match
            return self.current_env.python_version == required_version
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def create_environment_report(self) -> Dict[str, Any]:
        """Create a comprehensive environment report."""
        validation = self.validate_configuration()
        
        return {
            'environment_info': {
                'type': self.current_env.type,
                'machine_id': self.current_env.machine_id,
                'python_version': self.current_env.python_version,
                'total_memory_gb': self.current_env.total_memory_gb,
                'cpu_count': self.current_env.cpu_count,
                'gpu_available': self.current_env.gpu_available,
                'platform': self.current_env.platform,
                'architecture': self.current_env.architecture
            },
            'configuration_validation': validation,
            'recommended_environment': self.get_recommended_environment(),
            'available_configurations': {
                'training': self.training_config.exists(),
                'trading': self.trading_config.exists(),
                'active': self.active_config.exists()
            },
            'system_resources': {
                'memory_usage_percent': psutil.virtual_memory().percent,
                'cpu_usage_percent': psutil.cpu_percent(interval=1),
                'disk_usage_percent': psutil.disk_usage('.').percent
            },
            'generated_at': self._get_timestamp()
        }
    
    def print_environment_status(self):
        """Print current environment status."""
        report = self.create_environment_report()
        
        print("\n=== Environment Status ===")
        print(f"Environment Type: {report['environment_info']['type']}")
        print(f"Machine ID: {report['environment_info']['machine_id']}")
        print(f"Python Version: {report['environment_info']['python_version']}")
        print(f"Memory: {report['environment_info']['total_memory_gb']:.1f} GB")
        print(f"CPU Cores: {report['environment_info']['cpu_count']}")
        print(f"GPU Available: {report['environment_info']['gpu_available']}")
        print(f"Platform: {report['environment_info']['platform']} ({report['environment_info']['architecture']})")
        
        print("\n=== Configuration Status ===")
        validation = report['configuration_validation']
        print(f"Configuration Valid: {validation['valid']}")
        
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        print(f"\nRecommended Environment: {report['recommended_environment']}")
        
        print("\n=== System Resources ===")
        resources = report['system_resources']
        print(f"Memory Usage: {resources['memory_usage_percent']:.1f}%")
        print(f"CPU Usage: {resources['cpu_usage_percent']:.1f}%")
        print(f"Disk Usage: {resources['disk_usage_percent']:.1f}%")

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Environment Configuration Manager')
    parser.add_argument('--setup', choices=['training', 'trading'], help='Setup environment configuration')
    parser.add_argument('--status', action='store_true', help='Show environment status')
    parser.add_argument('--validate', action='store_true', help='Validate current configuration')
    parser.add_argument('--force', action='store_true', help='Force setup even if not optimal')
    parser.add_argument('--report', help='Generate environment report to file')
    
    args = parser.parse_args()
    
    manager = EnvironmentManager()
    
    if args.setup:
        success = manager.setup_environment(args.setup, force=args.force)
        if success:
            print(f"Environment successfully configured for {args.setup}")
        else:
            print(f"Failed to configure environment for {args.setup}")
            sys.exit(1)
    
    elif args.status:
        manager.print_environment_status()
    
    elif args.validate:
        validation = manager.validate_configuration()
        print(f"Configuration valid: {validation['valid']}")
        
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        if validation['warnings']:
            print("Warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
    
    elif args.report:
        report = manager.create_environment_report()
        with open(args.report, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        print(f"Environment report saved to {args.report}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()