#!/usr/bin/env python3
"""
Training Checkpoint System
=========================

Provides checkpoint and resume functionality for long-running model training sessions.
Handles 6-hour runtime limits by saving training state and allowing seamless continuation.

Features:
- Save/load training progress and state
- Resume from last checkpoint
- Graceful shutdown handling
- Progress tracking and validation
- Automatic cleanup
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TrainingProgress:
    """Tracks current training progress"""
    current_symbol_index: int = 0
    current_model_index: int = 0
    current_fold_index: int = 0
    completed_models: List[Tuple[str, str]] = None  # (model_type, symbol)
    total_symbols: int = 0
    total_models: int = 0
    total_folds: int = 0
    start_time: str = None
    last_checkpoint_time: str = None
    estimated_completion: str = None
    
    def __post_init__(self):
        if self.completed_models is None:
            self.completed_models = []
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint validation"""
    checkpoint_version: str = "1.0"
    created_at: str = None
    config_hash: str = None
    symbols: List[str] = None
    models: List[str] = None
    experiment_name: str = None
    output_dir: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class TrainingCheckpoint:
    """Manages training checkpoints for resume capability"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint files
        self.progress_file = self.checkpoint_dir / "training_progress.json"
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.config_file = self.checkpoint_dir / "training_config.pkl"
        self.results_file = self.checkpoint_dir / "partial_results.pkl"
        
        self.progress = TrainingProgress()
        self.metadata = CheckpointMetadata()
        self.partial_results = {}
        
    def save_checkpoint(
        self,
        progress: TrainingProgress,
        config: Dict[str, Any],
        partial_results: Dict[str, Any] = None,
        metadata: CheckpointMetadata = None
    ) -> bool:
        """Save training checkpoint
        
        Args:
            progress: Current training progress
            config: Training configuration
            partial_results: Partial training results
            metadata: Checkpoint metadata
            
        Returns:
            True if checkpoint saved successfully
        """
        try:
            # Update progress timestamp
            progress.last_checkpoint_time = datetime.now().isoformat()
            
            # Calculate estimated completion
            if progress.start_time:
                start_dt = datetime.fromisoformat(progress.start_time)
                elapsed = datetime.now() - start_dt
                
                total_work = progress.total_symbols * progress.total_models * progress.total_folds
                completed_work = (
                    progress.current_symbol_index * progress.total_models * progress.total_folds +
                    progress.current_model_index * progress.total_folds +
                    progress.current_fold_index
                )
                
                if completed_work > 0:
                    estimated_total_time = elapsed * (total_work / completed_work)
                    estimated_completion = start_dt + estimated_total_time
                    progress.estimated_completion = estimated_completion.isoformat()
            
            # Save progress
            with open(self.progress_file, 'w') as f:
                json.dump(asdict(progress), f, indent=2)
            
            # Save metadata
            if metadata is None:
                metadata = self.metadata
            metadata.created_at = datetime.now().isoformat()
            metadata.config_hash = self._calculate_config_hash(config)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Save config
            with open(self.config_file, 'wb') as f:
                pickle.dump(config, f)
            
            # Save partial results
            if partial_results:
                with open(self.results_file, 'wb') as f:
                    pickle.dump(partial_results, f)
            
            logger.info(f"Checkpoint saved at {progress.last_checkpoint_time}")
            logger.info(f"Progress: Symbol {progress.current_symbol_index + 1}/{progress.total_symbols}, "
                       f"Model {progress.current_model_index + 1}/{progress.total_models}, "
                       f"Fold {progress.current_fold_index + 1}/{progress.total_folds}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self) -> Tuple[Optional[TrainingProgress], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Load training checkpoint
        
        Returns:
            Tuple of (progress, config, partial_results) or (None, None, None) if no checkpoint
        """
        try:
            if not self.has_checkpoint():
                return None, None, None
            
            # Load progress
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
                progress = TrainingProgress(**progress_data)
            
            # Load config
            config = None
            if self.config_file.exists():
                with open(self.config_file, 'rb') as f:
                    config = pickle.load(f)
            
            # Load partial results
            partial_results = {}
            if self.results_file.exists():
                with open(self.results_file, 'rb') as f:
                    partial_results = pickle.load(f)
            
            logger.info(f"Checkpoint loaded from {progress.last_checkpoint_time}")
            logger.info(f"Resuming: Symbol {progress.current_symbol_index + 1}/{progress.total_symbols}, "
                       f"Model {progress.current_model_index + 1}/{progress.total_models}, "
                       f"Fold {progress.current_fold_index + 1}/{progress.total_folds}")
            
            return progress, config, partial_results
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None, None, None
    
    def has_checkpoint(self) -> bool:
        """Check if valid checkpoint exists"""
        return (
            self.progress_file.exists() and
            self.metadata_file.exists() and
            self.config_file.exists()
        )
    
    def validate_checkpoint(self, config: Dict[str, Any]) -> bool:
        """Validate checkpoint compatibility with current config
        
        Args:
            config: Current training configuration
            
        Returns:
            True if checkpoint is compatible
        """
        try:
            if not self.has_checkpoint():
                return False
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                metadata_data = json.load(f)
                metadata = CheckpointMetadata(**metadata_data)
            
            # Check config hash
            current_hash = self._calculate_config_hash(config)
            if metadata.config_hash != current_hash:
                logger.warning("Configuration has changed since checkpoint was created")
                return False
            
            # Check checkpoint age (warn if older than 24 hours)
            checkpoint_time = datetime.fromisoformat(metadata.created_at)
            age = datetime.now() - checkpoint_time
            if age > timedelta(hours=24):
                logger.warning(f"Checkpoint is {age.total_seconds() / 3600:.1f} hours old")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate checkpoint: {e}")
            return False
    
    def cleanup_checkpoint(self) -> bool:
        """Remove checkpoint files
        
        Returns:
            True if cleanup successful
        """
        try:
            files_to_remove = [
                self.progress_file,
                self.metadata_file,
                self.config_file,
                self.results_file
            ]
            
            for file_path in files_to_remove:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed {file_path}")
            
            # Remove checkpoint directory if empty
            if self.checkpoint_dir.exists() and not any(self.checkpoint_dir.iterdir()):
                self.checkpoint_dir.rmdir()
                logger.debug(f"Removed empty checkpoint directory {self.checkpoint_dir}")
            
            logger.info("Checkpoint cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoint: {e}")
            return False
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get human-readable progress summary
        
        Returns:
            Progress summary dictionary
        """
        if not self.has_checkpoint():
            return {"status": "No checkpoint found"}
        
        try:
            progress, _, _ = self.load_checkpoint()
            if not progress:
                return {"status": "Failed to load checkpoint"}
            
            total_work = progress.total_symbols * progress.total_models * progress.total_folds
            completed_work = len(progress.completed_models)
            current_work = (
                progress.current_symbol_index * progress.total_models * progress.total_folds +
                progress.current_model_index * progress.total_folds +
                progress.current_fold_index
            )
            
            completion_percentage = (current_work / total_work * 100) if total_work > 0 else 0
            
            summary = {
                "status": "Checkpoint available",
                "progress": {
                    "current_symbol": f"{progress.current_symbol_index + 1}/{progress.total_symbols}",
                    "current_model": f"{progress.current_model_index + 1}/{progress.total_models}",
                    "current_fold": f"{progress.current_fold_index + 1}/{progress.total_folds}",
                    "completed_models": len(progress.completed_models),
                    "completion_percentage": f"{completion_percentage:.1f}%"
                },
                "timing": {
                    "started": progress.start_time,
                    "last_checkpoint": progress.last_checkpoint_time,
                    "estimated_completion": progress.estimated_completion
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get progress summary: {e}")
            return {"status": "Error reading checkpoint"}
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """Calculate hash of configuration for validation
        
        Args:
            config: Configuration dictionary
            
        Returns:
            SHA256 hash of config
        """
        # Create a stable string representation of config
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def create_checkpoint_manager(checkpoint_dir: str = "checkpoints") -> TrainingCheckpoint:
    """Factory function to create checkpoint manager
    
    Args:
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        TrainingCheckpoint instance
    """
    return TrainingCheckpoint(checkpoint_dir)