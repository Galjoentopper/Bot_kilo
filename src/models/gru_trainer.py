"""
GRU Trainer Module
==================

PyTorch-based GRU model for short-term price prediction.
Optimized for GPU training on Paperspace Gradient.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import random

logger = logging.getLogger(__name__)

try:
    import mlflow  # type: ignore[import-untyped]
    import mlflow.pytorch  # type: ignore
    MLFLOW_AVAILABLE = True
except ImportError:
    # Create dummy mlflow module for type hints
    class _DummyMLflow:
        @staticmethod
        def start_run(*args: Any, **kwargs: Any) -> Any:
            from contextlib import nullcontext
            return nullcontext()
        
        @staticmethod
        def log_params(params: Dict[str, Any]) -> None:
            pass
            
        @staticmethod
        def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
            pass
            
        @staticmethod
        def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
            pass
        
        class pytorch:
            @staticmethod
            def log_model(model: Any, artifact_path: str) -> None:
                pass
    
    mlflow = _DummyMLflow()  # type: ignore
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Experiment tracking will be disabled.")

class GRUModel(nn.Module):
    """
    GRU-based neural network for time series prediction.
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize GRU model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            output_size: Output size (1 for regression)
        """
        super(GRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take the last output
        last_output = gru_out[:, -1, :]
        
        # Apply dropout
        dropped = self.dropout_layer(last_output)
        
        # Fully connected layers
        fc1_out = self.relu(self.fc1(dropped))
        output = self.fc2(fc1_out)
        
        return output

class GRUTrainer:
    """
    Trainer class for GRU model with GPU optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GRU trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config.get('models', {}).get('gru', {})
        self.training_config = config.get('training', {})
        
        # Device configuration (GPU optimization for Paperspace)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Reproducibility and perf tuning
        self.seed = int(self.training_config.get('seed', 42))
        self.deterministic = bool(self.training_config.get('deterministic', False))
        self._set_seed(self.seed, self.deterministic)
        
        # Model parameters
        self.sequence_length = self.model_config.get('sequence_length', 20)
        self.hidden_size = self.model_config.get('hidden_size', 128)
        self.num_layers = self.model_config.get('num_layers', 2)
        self.dropout = self.model_config.get('dropout', 0.2)
        self.learning_rate = self.model_config.get('learning_rate', 0.001)
        # Optimizer selection (default Adam, allow RMSprop/AdamW/SGD via config)
        self.optimizer_name = str(self.model_config.get('optimizer', 'Adam')).lower()
        self.batch_size = self.model_config.get('batch_size', 64)
        self.epochs = self.model_config.get('epochs', 100)
        self.early_stopping_patience = self.model_config.get('early_stopping_patience', 10)

        # Training optimization settings
        self.mixed_precision = self.training_config.get('mixed_precision', True)
        self.num_workers = self.training_config.get('num_workers', 4)
        self.pin_memory = self.training_config.get('pin_memory', True)
        self.max_grad_norm = float(self.training_config.get('max_grad_norm', 1.0))
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        # Configurable loss: mse (default) or huber (smoothl1)
        loss_name = str(self.model_config.get('loss', 'mse')).lower()
        if loss_name in ('huber', 'smoothl1', 'smooth_l1'):
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and torch.cuda.is_available() else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Feature tracking for persistence
        self.feature_names = []
        self.selected_features = None  # Indices of selected features
        self.feature_count = None
        self.input_size = None
        # Track instability during training
        self._consecutive_bad_batches = 0
        
        logger.info("GRU Trainer initialized with GPU optimization")

    def _set_seed(self, seed: int, deterministic: bool = False) -> None:
        """Set random seeds and cudnn flags for reproducibility/perf."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True
    
    def build_model(self, input_size: int) -> nn.Module:
        """
        Build and initialize the GRU model.
        
        Args:
            input_size: Number of input features
            
        Returns:
            Initialized GRU model
        """
        self.model = GRUModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=1
        ).to(self.device)
        
        # Initialize optimizer based on config
        opt_name = self.optimizer_name
        params = self.model.parameters()
        if opt_name == 'rmsprop':
            self.optimizer = optim.RMSprop(params, lr=self.learning_rate, alpha=0.99, eps=1e-8, weight_decay=1e-5)
        elif opt_name == 'adamw':
            self.optimizer = optim.AdamW(params, lr=self.learning_rate, weight_decay=1e-4)
        elif opt_name == 'sgd':
            self.optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=1e-5)
        else:
            # default Adam
            self.optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=1e-5)
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model built with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        return self.model
    
    def prepare_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders for training.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Validate input data for NaN/Inf values
        if not np.isfinite(X_train).all():
            logger.warning("Non-finite values detected in X_train, replacing with zeros")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if not np.isfinite(y_train).all():
            logger.warning("Non-finite values detected in y_train, replacing with zeros")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if not np.isfinite(X_val).all():
            logger.warning("Non-finite values detected in X_val, replacing with zeros")
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if not np.isfinite(y_val).all():
            logger.warning("Non-finite values detected in y_val, replacing with zeros")
            y_val = np.nan_to_num(y_val, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders with GPU optimization
        # Safer persistent_workers on Windows
        _persistent = True if self.num_workers and self.num_workers > 0 else False
        if os.name == 'nt':
            _persistent = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.pin_memory and self.device.type == 'cuda'),
            persistent_workers=_persistent,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.pin_memory and self.device.type == 'cuda'),
            persistent_workers=_persistent
        )
        
        logger.info(f"Data loaders prepared - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        if self.model is None:
            raise ValueError("Model must be built before training")
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            # Per-batch sanitization to avoid propagating NaN/Inf
            data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if self.optimizer is None:
                raise ValueError("Optimizer must be initialized before training")
            self.optimizer.zero_grad()
            
            if self.mixed_precision and self.scaler:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    if self.model is None:
                        raise ValueError("Model must be built before training")
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping batch")
                    self._consecutive_bad_batches += 1
                    # Auto-fallback: if too many bad batches, reduce LR and disable AMP
                    if self._consecutive_bad_batches >= 3:
                        if self.optimizer is not None:
                            for g in self.optimizer.param_groups:
                                g['lr'] = max(g['lr'] * 0.5, 1e-6)
                        if self.scaler is not None:
                            # drop AMP for stability
                            self.mixed_precision = False
                            self.scaler = None
                            logger.warning("Disabling mixed precision and reducing LR due to unstable loss")
                    self.optimizer.zero_grad()
                    continue
                
                self.scaler.scale(loss).backward()
                if self.scaler is None or self.optimizer is None:
                    raise ValueError("Scaler and optimizer must be initialized")
                # Gradient clipping (requires unscale first)
                if self.max_grad_norm and self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                # Check gradient finiteness
                _bad_grad = False
                for p in self.model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            _bad_grad = True
                            break
                if _bad_grad:
                    logger.warning(f"Non-finite gradients at batch {batch_idx}, skipping optimizer step")
                    self.optimizer.zero_grad(set_to_none=True)
                    self._consecutive_bad_batches += 1
                    if self._consecutive_bad_batches >= 3:
                        for g in self.optimizer.param_groups:
                            g['lr'] = max(g['lr'] * 0.5, 1e-6)
                        self.mixed_precision = False
                        self.scaler = None
                        logger.warning("Disabling mixed precision due to gradient instability")
                    continue
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                if self.model is None:
                    raise ValueError("Model must be built before training")
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping batch")
                    self._consecutive_bad_batches += 1
                    # Reduce LR if persistent
                    if self._consecutive_bad_batches >= 3 and self.optimizer is not None:
                        for g in self.optimizer.param_groups:
                            g['lr'] = max(g['lr'] * 0.5, 1e-6)
                    self.optimizer.zero_grad()
                    continue
                
                loss.backward()
                if self.optimizer is None:
                    raise ValueError("Optimizer must be initialized before training")
                if self.max_grad_norm and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            loss_value = loss.item()
            # Additional safety check for loss value
            if not torch.isfinite(torch.tensor(loss_value)):
                logger.warning(f"Non-finite loss value {loss_value} at batch {batch_idx}, skipping")
                self._consecutive_bad_batches += 1
                continue
            else:
                # Reset counter on good batch
                self._consecutive_bad_batches = 0
                
            total_loss += loss_value
            num_batches += 1
            
            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Return NaN if no valid batches processed
        if num_batches == 0:
            logger.warning("No valid batches in training, returning NaN")
            return float('nan')
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        if self.model is None:
            raise ValueError("Model must be built before validation")
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
                target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
                
                if self.mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        if self.model is None:
                            raise ValueError("Model must be built before validation")
                        output = self.model(data)
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            logger.warning("Non-finite model output in validation, skipping batch")
                            continue
                        loss = self.criterion(output, target)
                else:
                    if self.model is None:
                        raise ValueError("Model must be built before validation")
                    output = self.model(data)
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        logger.warning("Non-finite model output in validation, skipping batch")
                        continue
                    loss = self.criterion(output, target)
                
                # Check for NaN/Inf in validation loss
                loss_value = loss.item()
                if not torch.isfinite(torch.tensor(loss_value)):
                    logger.warning(f"Non-finite validation loss detected, skipping batch")
                    continue
                
                total_loss += loss_value
                num_batches += 1
        
        # Return NaN if no valid batches processed
        if num_batches == 0:
            logger.warning("No valid batches in validation, returning NaN")
            return float('nan')
        
        return total_loss / num_batches
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        experiment_name: str = "gru_training",
        feature_names: Optional[List[str]] = None,
        selected_features: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Train the GRU model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            experiment_name: MLflow experiment name
            feature_names: Names of features used
            selected_features: Indices of selected features
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting GRU model training")
        
        # Store feature information for persistence
        self.input_size = X_train.shape[2]  # Features dimension
        self.feature_count = self.input_size
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.input_size)]
        self.selected_features = selected_features
        
        # Build model
        self.build_model(self.input_size)
        
        # Prepare data loaders
        train_loader, val_loader = self.prepare_data(X_train, y_train, X_val, y_val)
        
        # Start MLflow run (if available)
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow_context = mlflow.start_run(run_name=f"gru_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            from contextlib import nullcontext
            mlflow_context = nullcontext()
        
        with mlflow_context:
            # Log parameters (if MLflow available)
            if MLFLOW_AVAILABLE and mlflow is not None:
                mlflow.log_params({
                    "model_type": "GRU",
                    "sequence_length": self.sequence_length,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "device": str(self.device),
                    "mixed_precision": self.mixed_precision
                })
            
            # Training loop
            patience_counter = 0
            epoch = 0
            
            for epoch in range(self.epochs):
                # Train
                train_loss = self.train_epoch(train_loader)
                
                # Validate
                val_loss = self.validate_epoch(val_loader)
                
                # Check for NaN/Inf validation loss
                if not torch.isfinite(torch.tensor(val_loss)):
                    logger.error(f"Non-finite validation loss {val_loss} at epoch {epoch+1}, stopping training")
                    break
                
                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                
                # Store losses
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                # Log metrics (if MLflow available)
                if MLFLOW_AVAILABLE and mlflow is not None:
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                    }, step=epoch)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    if self.model is not None:
                        self.best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - New best validation loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Load best model
            if self.best_model_state:
                if self.model is not None and self.best_model_state is not None:
                    self.model.load_state_dict(self.best_model_state)
            
            # Log final metrics (if MLflow available)
            if MLFLOW_AVAILABLE and mlflow is not None:
                mlflow.log_metrics({
                    "best_val_loss": self.best_val_loss,
                    "total_epochs": epoch + 1
                })
                
                # Save model with signature and input example
                if self.model is not None and mlflow is not None:
                    # Create a sample input for signature inference
                    sample_input = torch.randn(1, self.sequence_length, self.input_size).to(self.device)
                    
                    # Log model with signature and input example
                    try:
                        mlflow.pytorch.log_model(  # type: ignore[attr-defined]
                            self.model,
                            artifact_path="model"
                        )
                    except Exception as e:
                        logger.warning(f"MLflow log_model failed: {e}")
        
        # Training results
        results = {
            "best_val_loss": self.best_val_loss,
            "total_epochs": epoch + 1,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "model_state": self.best_model_state
        }
        
        logger.info(f"Training completed - Best validation loss: {self.best_val_loss:.6f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        predictions: List[np.ndarray] = []
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            # Process in batches to handle memory efficiently
            batch_size = self.batch_size
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                
                if self.mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        batch_pred = self.model(batch)
                else:
                    batch_pred = self.model(batch)
                
                predictions.append(batch_pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0).flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics dictionary
        """
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # Directional accuracy
        y_test_direction = np.sign(y_test)
        pred_direction = np.sign(predictions)
        directional_accuracy = np.mean(y_test_direction == pred_direction)
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "directional_accuracy": directional_accuracy
        }
        
        logger.info(f"Model evaluation - RMSE: {rmse:.6f}, R²: {r2:.4f}, Dir. Acc: {directional_accuracy:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str, symbol: Optional[str] = None):
        """
        Save the trained model with feature index persistence.
        
        Args:
            filepath: Path to save the model
            symbol: Optional symbol for symbol-specific models
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Enhanced save with feature information
        save_data = {
            'model_state_dict': self.best_model_state or self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'output_size': 1
            },
            'training_config': self.model_config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            # Feature persistence information
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'feature_count': self.feature_count,
            'input_size_actual': self.input_size,
            'symbol': symbol,
            'created_at': datetime.now().isoformat(),
            'model_type': 'gru'
        }
        
        torch.save(save_data, filepath)
        
        logger.info(f"GRU model saved to {filepath} with {len(self.feature_names)} features")
        if self.selected_features:
            logger.info(f"Selected feature indices: {len(self.selected_features)} features")
    
    @classmethod
    def load_model(cls, filepath: str, config: Dict[str, Any]) -> 'GRUTrainer':
        """
        Load a trained model with feature index restoration and architecture compatibility.
        
        Args:
            filepath: Path to the saved model
            config: Configuration dictionary
            
        Returns:
            Loaded GRUTrainer instance with restored feature information
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create trainer instance
        trainer = cls(config)
        
        # Use saved model configuration instead of current config for architecture compatibility
        model_config = checkpoint['model_config']
        trainer.hidden_size = model_config['hidden_size']
        trainer.num_layers = model_config['num_layers']
        trainer.dropout = model_config['dropout']
        
        # Restore feature information
        trainer.feature_names = checkpoint.get('feature_names', [f"feature_{i}" for i in range(model_config['input_size'])])
        trainer.selected_features = checkpoint.get('selected_features', None)
        trainer.feature_count = checkpoint.get('feature_count', model_config['input_size'])
        trainer.input_size = checkpoint.get('input_size_actual', model_config['input_size'])
        
        # Build model with saved configuration
        trainer.build_model(model_config['input_size'])
        
        # Load model state
        if trainer.model is not None:
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.model.eval()
        
        # Restore training history
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.best_model_state = checkpoint.get('model_state_dict', None)
        
        # Log feature information
        logger.info(f"GRU model loaded from {filepath} with architecture: {model_config}")
        logger.info(f"Restored {len(trainer.feature_names)} feature names")
        if trainer.selected_features:
            logger.info(f"Restored {len(trainer.selected_features)} selected feature indices")
        
        # Validate feature consistency
        if trainer.feature_count != model_config['input_size']:
            logger.warning(f"Feature count mismatch: stored={trainer.feature_count}, model={model_config['input_size']}")
        
        return trainer