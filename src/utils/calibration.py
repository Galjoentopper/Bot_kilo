"""
Probability Calibration Module
==============================

Implements probability calibration methods for classification models
to improve prediction reliability and threshold optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import logging
import joblib
import json

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """
    Calibrate predicted probabilities using various methods.
    """
    
    def __init__(self, method: str = "isotonic", n_bins: int = 10):
        """
        Initialize probability calibrator.
        
        Args:
            method: Calibration method ('isotonic', 'platt', 'beta')
            n_bins: Number of bins for calibration curve
        """
        self.method = method
        self.n_bins = n_bins
        self.calibrator = None
        self.is_fitted = False
        self.calibration_stats = {}
        
        logger.info(f"Initialized {method} calibrator")
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on validation data.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Self for method chaining
        """
        # Validate inputs
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        
        if len(y_true) != len(y_prob):
            raise ValueError("y_true and y_prob must have same length")
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_prob))
        y_true = y_true[mask]
        y_prob = y_prob[mask]
        
        # Clip probabilities to avoid numerical issues
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        
        # Fit calibrator based on method
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
            
        elif self.method == "platt":
            # Platt scaling uses sigmoid calibration
            self.calibrator = LogisticRegression()
            # Reshape for sklearn
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
            
        elif self.method == "beta":
            # Beta calibration (custom implementation)
            self.calibrator = self._fit_beta_calibration(y_true, y_prob)
            
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        # Calculate calibration statistics
        self.calibration_stats = self._calculate_calibration_stats(y_true, y_prob)
        self.is_fitted = True
        
        logger.info(f"Calibrator fitted on {len(y_true)} samples")
        
        return self
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Transform probabilities using fitted calibrator.
        
        Args:
            y_prob: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        # Handle input shape
        y_prob = np.asarray(y_prob)
        original_shape = y_prob.shape
        y_prob_flat = y_prob.flatten()
        
        # Clip probabilities
        y_prob_flat = np.clip(y_prob_flat, 1e-7, 1 - 1e-7)
        
        # Apply calibration
        if self.method == "isotonic":
            calibrated = self.calibrator.transform(y_prob_flat)
            
        elif self.method == "platt":
            # Get probability of positive class
            calibrated = self.calibrator.predict_proba(y_prob_flat.reshape(-1, 1))[:, 1]
            
        elif self.method == "beta":
            calibrated = self._apply_beta_calibration(y_prob_flat)
            
        else:
            calibrated = y_prob_flat
        
        # Ensure valid probability range
        calibrated = np.clip(calibrated, 0, 1)
        
        # Restore original shape
        return calibrated.reshape(original_shape)
    
    def fit_transform(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """
        Fit calibrator and transform probabilities.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        self.fit(y_true, y_prob)
        return self.transform(y_prob)
    
    def _fit_beta_calibration(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Fit beta calibration parameters.
        
        Beta calibration maps probabilities using:
        p_calibrated = p^a / (p^a + (1-p)^b)
        """
        from scipy.optimize import minimize
        
        def negative_log_likelihood(params):
            a, b = params
            # Avoid numerical issues
            a = max(a, 1e-3)
            b = max(b, 1e-3)
            
            # Beta calibration formula
            p_cal = y_prob**a / (y_prob**a + (1 - y_prob)**b)
            p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)
            
            # Negative log likelihood
            nll = -np.mean(y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal))
            return nll
        
        # Optimize parameters
        result = minimize(
            negative_log_likelihood,
            x0=[1.0, 1.0],
            bounds=[(0.01, 10), (0.01, 10)],
            method='L-BFGS-B'
        )
        
        return {'a': result.x[0], 'b': result.x[1]}
    
    def _apply_beta_calibration(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply beta calibration to probabilities."""
        a = self.calibrator['a']
        b = self.calibrator['b']
        
        return y_prob**a / (y_prob**a + (1 - y_prob)**b)
    
    def _calculate_calibration_stats(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """Calculate calibration statistics."""
        # Get calibration curve
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=self.n_bins, strategy='uniform'
        )
        
        # Calculate expected calibration error (ECE)
        # Note: fraction_pos and mean_pred may have fewer elements than n_bins
        # if some bins are empty
        n_actual_bins = len(fraction_pos)
        bin_counts = np.histogram(y_prob, bins=self.n_bins)[0]
        
        # Only use bins that have predictions
        non_empty_bins = bin_counts > 0
        bin_weights = bin_counts[non_empty_bins] / len(y_prob)
        
        # Ensure arrays have same length
        if len(bin_weights) == len(fraction_pos):
            ece = np.sum(bin_weights * np.abs(fraction_pos - mean_pred))
        else:
            # Fallback: simple average if bins don't match
            ece = np.mean(np.abs(fraction_pos - mean_pred))
        
        # Calculate maximum calibration error (MCE)
        mce = np.max(np.abs(fraction_pos - mean_pred))
        
        # Brier score
        brier_score = np.mean((y_prob - y_true) ** 2)
        
        return {
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(brier_score),
            'fraction_positive': fraction_pos.tolist(),
            'mean_predicted': mean_pred.tolist(),
            'n_samples': len(y_true)
        }
    
    def get_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get calibration curve data for plotting.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary with calibration curve data
        """
        # Original probabilities
        fraction_pos_orig, mean_pred_orig = calibration_curve(
            y_true, y_prob, n_bins=self.n_bins, strategy='uniform'
        )
        
        # Calibrated probabilities
        if self.is_fitted:
            y_prob_cal = self.transform(y_prob)
            fraction_pos_cal, mean_pred_cal = calibration_curve(
                y_true, y_prob_cal, n_bins=self.n_bins, strategy='uniform'
            )
        else:
            fraction_pos_cal = fraction_pos_orig
            mean_pred_cal = mean_pred_orig
        
        return {
            'fraction_positive_original': fraction_pos_orig,
            'mean_predicted_original': mean_pred_orig,
            'fraction_positive_calibrated': fraction_pos_cal,
            'mean_predicted_calibrated': mean_pred_cal,
            'perfect_calibration': np.linspace(0, 1, 100)
        }
    
    def save(self, filepath: str) -> None:
        """
        Save calibrator to file.
        
        Args:
            filepath: Path to save calibrator
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")
        
        # Save calibrator data
        data = {
            'method': self.method,
            'n_bins': self.n_bins,
            'calibration_stats': self.calibration_stats,
            'is_fitted': self.is_fitted
        }
        
        # Save method-specific data
        if self.method in ['isotonic', 'platt']:
            # Save sklearn model
            data['calibrator_path'] = filepath + '_model.pkl'
            joblib.dump(self.calibrator, data['calibrator_path'])
        elif self.method == 'beta':
            data['calibrator_params'] = self.calibrator
        
        # Save metadata
        with open(filepath + '_metadata.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Calibrator saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ProbabilityCalibrator':
        """
        Load calibrator from file.
        
        Args:
            filepath: Path to load calibrator from
            
        Returns:
            Loaded calibrator instance
        """
        # Load metadata
        with open(filepath + '_metadata.json', 'r') as f:
            data = json.load(f)
        
        # Create calibrator instance
        calibrator = cls(method=data['method'], n_bins=data['n_bins'])
        calibrator.calibration_stats = data['calibration_stats']
        calibrator.is_fitted = data['is_fitted']
        
        # Load method-specific data
        if data['method'] in ['isotonic', 'platt']:
            calibrator.calibrator = joblib.load(data['calibrator_path'])
        elif data['method'] == 'beta':
            calibrator.calibrator = data['calibrator_params']
        
        logger.info(f"Calibrator loaded from {filepath}")
        return calibrator


def calibrate_probabilities_cv(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cv_folds: List[Tuple[np.ndarray, np.ndarray]],
    method: str = "isotonic"
) -> Tuple[np.ndarray, List[ProbabilityCalibrator]]:
    """
    Calibrate probabilities using cross-validation.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        cv_folds: List of (train_idx, val_idx) tuples
        method: Calibration method
        
    Returns:
        Tuple of (calibrated_probabilities, calibrators)
    """
    calibrated_probs = np.zeros_like(y_prob)
    calibrators = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
        # Fit calibrator on validation fold
        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(y_true[val_idx], y_prob[val_idx])
        calibrators.append(calibrator)
        
        # Apply to validation fold
        calibrated_probs[val_idx] = calibrator.transform(y_prob[val_idx])
        
        logger.info(f"Calibrated fold {fold_idx + 1}/{len(cv_folds)}")
    
    return calibrated_probs, calibrators


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    calibrator: Optional[ProbabilityCalibrator] = None,
    title: str = "Calibration Plot",
    ax=None
):
    """
    Plot calibration curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        calibrator: Fitted calibrator (optional)
        title: Plot title
        ax: Matplotlib axis (created if None)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
        return None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get calibration data
    if calibrator and calibrator.is_fitted:
        curve_data = calibrator.get_calibration_curve(y_true, y_prob)
        
        # Plot original
        ax.plot(
            curve_data['mean_predicted_original'],
            curve_data['fraction_positive_original'],
            marker='o',
            label='Original',
            alpha=0.7
        )
        
        # Plot calibrated
        ax.plot(
            curve_data['mean_predicted_calibrated'],
            curve_data['fraction_positive_calibrated'],
            marker='s',
            label='Calibrated',
            alpha=0.7
        )
    else:
        # Just plot original
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred, fraction_pos, marker='o', label='Original')
    
    # Plot perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Formatting
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return ax