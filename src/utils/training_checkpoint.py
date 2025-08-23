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
    failed_models: List[Dict[str, Any]] = None  # Track failures for resume strategy
    total_symbols: int = 0
    total_models: int = 0
    total_folds: int = 0
    start_time: str = None
    last_checkpoint_time: str = None
    estimated_completion: str = None
    last_successful_save_path: str = None  # For resume post-save
    
    def __post_init__(self):
        if self.completed_models is None:
            self.completed_models = []
        if self.failed_models is None:
            self.failed_models = []
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
        self._ensure_checkpoint_directory()
        
        # Checkpoint files
        self.progress_file = self.checkpoint_dir / "training_progress.json"
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.config_file = self.checkpoint_dir / "training_config.pkl"
        self.results_file = self.checkpoint_dir / "partial_results.pkl"
        
        self.progress = TrainingProgress()
        self.metadata = CheckpointMetadata()
        self.partial_results = {}
    
    def _atomic_write_json(self, file_path: Path, data: Dict[str, Any]) -> bool:
        """Atomically write JSON data to file
        
        Args:
            file_path: Target file path
            data: Data to write
            
        Returns:
            True if write successful
        """
        temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            # Atomic replace
            temp_path.replace(file_path)
            return True
        except Exception as e:
            logger.error(f"Failed atomic JSON write to {file_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def _atomic_write_pickle(self, file_path: Path, data: Any) -> bool:
        """Atomically write pickle data to file
        
        Args:
            file_path: Target file path
            data: Data to write
            
        Returns:
            True if write successful
        """
        temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            # Atomic replace
            temp_path.replace(file_path)
            return True
        except Exception as e:
            logger.error(f"Failed atomic pickle write to {file_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def _ensure_checkpoint_directory(self) -> None:
        """Ensure checkpoint directory exists with robust error handling
        
        Raises:
            OSError: If directory cannot be created or accessed
        """
        try:
            # Create directory with parents if needed
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Verify directory was created and is accessible
            if not self.checkpoint_dir.exists():
                raise OSError(f"Failed to create checkpoint directory: {self.checkpoint_dir}")
            
            if not self.checkpoint_dir.is_dir():
                raise OSError(f"Checkpoint path exists but is not a directory: {self.checkpoint_dir}")
            
            # Test write permissions by creating a temporary file
            test_file = self.checkpoint_dir / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                raise OSError(f"No write permission in checkpoint directory {self.checkpoint_dir}: {e}")
            
            logger.debug(f"Checkpoint directory verified: {self.checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"Failed to ensure checkpoint directory {self.checkpoint_dir}: {e}")
            raise OSError(f"Cannot create or access checkpoint directory {self.checkpoint_dir}: {e}")
        
    def save_checkpoint(
        self,
        progress: TrainingProgress,
        config: Dict[str, Any],
        partial_results: Dict[str, Any] = None,
        metadata: CheckpointMetadata = None
    ) -> bool:
        """Save training checkpoint with atomic writes
        
        Args:
            progress: Current training progress
            config: Training configuration
            partial_results: Partial training results
            metadata: Checkpoint metadata
            
        Returns:
            True if checkpoint saved successfully
        """
        try:
            # Ensure checkpoint directory exists before saving
            self._ensure_checkpoint_directory()
            
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
            
            # Atomic save operations - all must succeed
            save_success = True
            
            # Save progress atomically
            if not self._atomic_write_json(self.progress_file, asdict(progress)):
                save_success = False
            
            # Save metadata atomically
            if metadata is None:
                metadata = self.metadata
            metadata.created_at = datetime.now().isoformat()
            metadata.config_hash = self._calculate_config_hash(config)
            
            if not self._atomic_write_json(self.metadata_file, asdict(metadata)):
                save_success = False
            
            # Save config atomically
            if not self._atomic_write_pickle(self.config_file, config):
                save_success = False
            
            # Save partial results atomically
            if partial_results:
                if not self._atomic_write_pickle(self.results_file, partial_results):
                    save_success = False
            
            if save_success:
                logger.info(f"Checkpoint saved atomically at {progress.last_checkpoint_time}")
                logger.info(f"Progress: Symbol {progress.current_symbol_index + 1}/{progress.total_symbols}, "
                           f"Model {progress.current_model_index + 1}/{progress.total_models}, "
                           f"Fold {progress.current_fold_index + 1}/{progress.total_folds}")
                return True
            else:
                logger.error("Failed to save checkpoint - some atomic writes failed")
                return False
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def record_model_failure(self, symbol: str, model_type: str, error: str, 
                           progress: TrainingProgress) -> None:
        """Record a model training failure for resume strategy
        
        Args:
            symbol: Symbol that failed
            model_type: Model type that failed
            error: Error message
            progress: Current progress to update
        """
        failure_record = {
            'symbol': symbol,
            'model_type': model_type,
            'error': str(error),
            'timestamp': datetime.now().isoformat(),
            'symbol_index': progress.current_symbol_index,
            'model_index': progress.current_model_index
        }
        
        progress.failed_models.append(failure_record)
        logger.warning(f"Recorded failure: {symbol}_{model_type} - {error}")
    
    def record_successful_save(self, saved_path: str, progress: TrainingProgress) -> None:
        """Record successful model save for resume capability
        
        Args:
            saved_path: Path where model was saved
            progress: Current progress to update
        """
        progress.last_successful_save_path = saved_path
        logger.debug(f"Recorded successful save: {saved_path}")
    
    def should_skip_model(self, symbol: str, model_type: str, 
                         progress: TrainingProgress, max_retries: int = 2) -> bool:
        """Check if model should be skipped due to repeated failures
        
        Args:
            symbol: Symbol to check
            model_type: Model type to check
            progress: Current progress
            max_retries: Maximum retry attempts
            
        Returns:
            True if model should be skipped
        """
        failure_count = sum(
            1 for failure in progress.failed_models
            if failure['symbol'] == symbol and failure['model_type'] == model_type
        )
        
        if failure_count >= max_retries:
            logger.info(f"Skipping {symbol}_{model_type} - exceeded {max_retries} retry attempts")
            return True
        
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