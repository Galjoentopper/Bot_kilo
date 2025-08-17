#!/usr/bin/env python3
"""
MLflow Initialization Utility
============================
Automatically creates MLflow experiments and meta.yaml files with dynamic paths.
This solves the issue where MLflow expects meta.yaml files that don't exist automatically.
"""

import os
import yaml
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MLflowInitializer:
    """Handles MLflow initialization with automatic meta.yaml creation."""
    
    def __init__(self, tracking_uri: str = "./mlruns"):
        """
        Initialize MLflow setup utility.
        
        Args:
            tracking_uri: Path to MLflow tracking directory (default: "./mlruns")
        """
        self.tracking_uri = Path(tracking_uri).resolve()
        self.experiments_created = []
        
    def get_dynamic_artifact_location(self, experiment_id: str) -> str:
        """Generate dynamic artifact location URI for current environment."""
        # Convert to absolute path and handle Windows/Unix differences
        abs_path = self.tracking_uri / experiment_id
        
        if os.name == 'nt':  # Windows
            # Convert to file URI format for Windows
            return f"file:///{abs_path.as_posix()}"
        else:  # Unix-like systems
            return f"file://{abs_path.as_posix()}"
    
    def create_experiment_meta(self, experiment_id: str, name: str, 
                             creation_time: Optional[int] = None,
                             tags: Optional[Dict[str, str]] = None) -> bool:
        """
        Create meta.yaml file for an MLflow experiment.
        
        Args:
            experiment_id: Experiment ID (e.g., "0", "1")
            name: Experiment name (e.g., "Default", "crypto_trading_bot")
            creation_time: Unix timestamp (uses current time if None)
            tags: Additional tags for the experiment
            
        Returns:
            True if created successfully, False otherwise
        """
        try:
            # Create experiment directory
            exp_dir = self.tracking_uri / experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate meta.yaml content
            current_time = int(time.time() * 1000)  # MLflow uses milliseconds
            if creation_time is None:
                creation_time = current_time
                
            meta_content = {
                'artifact_location': self.get_dynamic_artifact_location(experiment_id),
                'creation_time': creation_time,
                'experiment_id': experiment_id,
                'last_update_time': current_time,
                'lifecycle_stage': 'active',
                'name': name,
                'tags': tags or {}
            }
            
            # Write meta.yaml file
            meta_file = exp_dir / "meta.yaml"
            with open(meta_file, 'w') as f:
                yaml.dump(meta_content, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Created MLflow experiment meta.yaml: {meta_file}")
            self.experiments_created.append({
                'id': experiment_id,
                'name': name,
                'path': str(meta_file)
            })
            return True
            
        except Exception as e:
            logger.error(f"Failed to create experiment meta.yaml for {experiment_id}: {e}")
            return False
    
    def ensure_default_experiment(self) -> bool:
        """Ensure the default experiment (ID: 0) exists."""
        meta_file = self.tracking_uri / "0" / "meta.yaml"
        
        if meta_file.exists():
            logger.debug("Default experiment meta.yaml already exists")
            return True
        
        logger.info("Creating default experiment meta.yaml")
        return self.create_experiment_meta("0", "Default")
    
    def ensure_experiment(self, experiment_name: str, 
                         experiment_id: Optional[str] = None) -> Optional[str]:
        """
        Ensure an experiment exists, creating it if necessary.
        
        Args:
            experiment_name: Name of the experiment
            experiment_id: Specific ID to use (auto-assigned if None)
            
        Returns:
            Experiment ID if successful, None otherwise
        """
        # Find next available ID if not specified
        if experiment_id is None:
            existing_ids = []
            if self.tracking_uri.exists():
                for item in self.tracking_uri.iterdir():
                    if item.is_dir() and item.name.isdigit():
                        existing_ids.append(int(item.name))
            
            # Start from 1 (0 is reserved for default)
            next_id = max(existing_ids) + 1 if existing_ids else 1
            experiment_id = str(next_id)
        
        meta_file = self.tracking_uri / experiment_id / "meta.yaml"
        
        if meta_file.exists():
            logger.debug(f"Experiment '{experiment_name}' meta.yaml already exists")
            return experiment_id
        
        logger.info(f"Creating experiment '{experiment_name}' with ID {experiment_id}")
        if self.create_experiment_meta(experiment_id, experiment_name):
            return experiment_id
        return None
    
    def initialize_mlflow_tracking(self, experiments: Optional[Dict[str, str]] = None) -> bool:
        """
        Initialize complete MLflow tracking setup.
        
        Args:
            experiments: Dict of experiment_name -> experiment_id mappings
                        If None, uses default experiments
                        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Ensure tracking directory exists
            self.tracking_uri.mkdir(parents=True, exist_ok=True)
            
            # Always ensure default experiment exists
            if not self.ensure_default_experiment():
                logger.error("Failed to create default experiment")
                return False
            
            # Create specified experiments
            if experiments is None:
                experiments = {"crypto_trading_bot": "1"}
            
            for exp_name, exp_id in experiments.items():
                if not self.ensure_experiment(exp_name, exp_id):
                    logger.error(f"Failed to create experiment '{exp_name}'")
                    return False
            
            logger.info(f"MLflow initialization complete. Created {len(self.experiments_created)} experiments")
            return True
            
        except Exception as e:
            logger.error(f"MLflow initialization failed: {e}")
            return False
    
    def setup_mlflow_for_config(self, config: Dict[str, Any]) -> bool:
        """
        Setup MLflow based on configuration file.
        
        Args:
            config: Configuration dictionary containing MLflow settings
            
        Returns:
            True if setup successful, False otherwise
        """
        try:
            mlflow_config = config.get('mlflow', {})
            tracking_uri = mlflow_config.get('tracking_uri', './mlruns')
            experiment_name = mlflow_config.get('experiment_name', 'crypto_trading_bot')
            
            # Update tracking URI if different
            if str(tracking_uri) != str(self.tracking_uri):
                self.tracking_uri = Path(tracking_uri).resolve()
            
            # Initialize with the configured experiment
            experiments = {experiment_name: "1"}  # Use ID 1 for the main experiment
            
            return self.initialize_mlflow_tracking(experiments)
            
        except Exception as e:
            logger.error(f"Failed to setup MLflow from config: {e}")
            return False


def initialize_mlflow_from_config(config_path: str = "src/config/config.yaml") -> bool:
    """
    Convenience function to initialize MLflow from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        initializer = MLflowInitializer()
        return initializer.setup_mlflow_for_config(config)
        
    except Exception as e:
        logger.error(f"Failed to initialize MLflow from config: {e}")
        return False


if __name__ == "__main__":
    # Test initialization
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MLflow initialization...")
    success = initialize_mlflow_from_config()
    
    if success:
        print("[SUCCESS] MLflow initialization successful!")
    else:
        print("[ERROR] MLflow initialization failed!")