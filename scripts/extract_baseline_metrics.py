#!/usr/bin/env python3
"""
Baseline Metrics Extraction Script
==================================

Extract performance metrics from previous MLflow runs for comparison analysis.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

class BaselineMetricsExtractor:
    """Extract and analyze baseline metrics from MLflow runs."""
    
    def __init__(self, mlruns_path: str = "./mlruns"):
        """Initialize the extractor."""
        self.mlruns_path = Path(mlruns_path)
        self.baseline_metrics = {}
        
    def extract_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """Extract metrics from a specific MLflow run."""
        run_path = self.mlruns_path / "0" / run_id
        
        if not run_path.exists():
            logger.warning(f"Run path does not exist: {run_path}")
            return {}
        
        metrics = {}
        params = {}
        
        # Extract metrics
        metrics_path = run_path / "metrics"
        if metrics_path.exists():
            for metric_file in metrics_path.glob("*"):
                try:
                    with open(metric_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            # Get the last (most recent) metric value
                            last_line = lines[-1].strip()
                            parts = last_line.split()
                            if len(parts) >= 2:
                                metrics[metric_file.name] = float(parts[1])
                except Exception as e:
                    logger.debug(f"Error reading metric {metric_file.name}: {e}")
        
        # Extract parameters
        params_path = run_path / "params"
        if params_path.exists():
            for param_file in params_path.glob("*"):
                try:
                    with open(param_file, 'r') as f:
                        params[param_file.name] = f.read().strip()
                except Exception as e:
                    logger.debug(f"Error reading param {param_file.name}: {e}")
        
        # Extract tags for model type identification
        tags = {}
        tags_path = run_path / "tags"
        if tags_path.exists():
            for tag_file in tags_path.glob("*"):
                try:
                    with open(tag_file, 'r') as f:
                        tags[tag_file.name] = f.read().strip()
                except Exception as e:
                    logger.debug(f"Error reading tag {tag_file.name}: {e}")
        
        return {
            'run_id': run_id,
            'metrics': metrics,
            'params': params,
            'tags': tags,
            'timestamp': run_path.stat().st_mtime if run_path.exists() else 0
        }
    
    def extract_all_baselines(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract baseline metrics from all available runs."""
        experiment_path = self.mlruns_path / "0"
        
        if not experiment_path.exists():
            logger.error(f"Experiment path does not exist: {experiment_path}")
            return {}
        
        runs_by_model = {
            'gru': [],
            'lightgbm': [],
            'ppo': []
        }
        
        # Process all run directories
        for run_dir in experiment_path.iterdir():
            if run_dir.is_dir() and run_dir.name not in ['meta.yaml', 'models']:
                run_data = self.extract_run_metrics(run_dir.name)
                
                if run_data and run_data['metrics']:
                    # Determine model type from parameters or tags
                    model_type = self._identify_model_type(run_data)
                    if model_type:
                        runs_by_model[model_type].append(run_data)
        
        # Sort runs by timestamp (newest first)
        for model_type in runs_by_model:
            runs_by_model[model_type].sort(key=lambda x: x['timestamp'], reverse=True)
        
        logger.info(f"Extracted baseline metrics:")
        for model_type, runs in runs_by_model.items():
            logger.info(f"  {model_type}: {len(runs)} runs")
        
        return runs_by_model
    
    def _identify_model_type(self, run_data: Dict[str, Any]) -> Optional[str]:
        """Identify model type from run data."""
        params = run_data.get('params', {})
        tags = run_data.get('tags', {})
        
        # Check model_type parameter
        if 'model_type' in params:
            model_type = params['model_type'].lower()
            if 'gru' in model_type:
                return 'gru'
            elif 'lightgbm' in model_type or 'lgbm' in model_type:
                return 'lightgbm'
            elif 'ppo' in model_type:
                return 'ppo'
        
        # Check for model-specific parameters
        if 'hidden_size' in params or 'sequence_length' in params:
            return 'gru'
        elif 'num_leaves' in params or 'boosting_type' in params:
            return 'lightgbm'
        elif 'n_steps' in params or 'clip_range' in params:
            return 'ppo'
        
        # Check run name in tags
        run_name = tags.get('mlflow.runName', '').lower()
        if 'gru' in run_name:
            return 'gru'
        elif 'lgbm' in run_name or 'lightgbm' in run_name:
            return 'lightgbm'
        elif 'ppo' in run_name:
            return 'ppo'
        
        return None
    
    def get_latest_baseline(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get the latest baseline metrics for a specific model type."""
        baselines = self.extract_all_baselines()
        
        if model_type in baselines and baselines[model_type]:
            return baselines[model_type][0]  # Most recent run
        
        return None
    
    def compare_metrics(self, old_metrics: Dict[str, float], new_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Compare old and new metrics."""
        comparison = {}
        
        # Find common metrics
        common_metrics = set(old_metrics.keys()) & set(new_metrics.keys())
        
        for metric in common_metrics:
            old_val = old_metrics[metric]
            new_val = new_metrics[metric]
            
            # Calculate improvement
            if old_val != 0:
                improvement_pct = ((new_val - old_val) / abs(old_val)) * 100
            else:
                improvement_pct = float('inf') if new_val > 0 else float('-inf')
            
            comparison[metric] = {
                'old': old_val,
                'new': new_val,
                'improvement': new_val - old_val,
                'improvement_pct': improvement_pct
            }
        
        return comparison
    
    def generate_baseline_report(self) -> str:
        """Generate a comprehensive baseline report."""
        baselines = self.extract_all_baselines()
        
        report = []
        report.append("=" * 60)
        report.append("BASELINE METRICS EXTRACTION REPORT")
        report.append("=" * 60)
        report.append("")
        
        for model_type, runs in baselines.items():
            report.append(f"{model_type.upper()} MODEL BASELINES")
            report.append("-" * 30)
            
            if not runs:
                report.append("No baseline runs found")
                report.append("")
                continue
            
            # Show latest run metrics
            latest_run = runs[0]
            report.append(f"Latest Run ID: {latest_run['run_id']}")
            report.append(f"Number of Historical Runs: {len(runs)}")
            report.append("")
            
            # Display metrics
            metrics = latest_run['metrics']
            if metrics:
                report.append("Key Metrics:")
                for metric, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        if 'accuracy' in metric or 'ratio' in metric:
                            report.append(f"  {metric}: {value:.4f}")
                        elif 'loss' in metric or 'error' in metric:
                            report.append(f"  {metric}: {value:.6f}")
                        else:
                            report.append(f"  {metric}: {value:.4f}")
                    else:
                        report.append(f"  {metric}: {value}")
            else:
                report.append("No metrics available")
            
            report.append("")
            
            # Show parameters
            params = latest_run['params']
            if params:
                report.append("Model Parameters:")
                for param, value in sorted(params.items()):
                    report.append(f"  {param}: {value}")
            
            report.append("")
            report.append("")
        
        return "\n".join(report)
    
    def save_baseline_data(self, output_path: str = "baseline_metrics.json"):
        """Save baseline data to JSON file."""
        baselines = self.extract_all_baselines()
        
        # Convert to serializable format
        serializable_data = {}
        for model_type, runs in baselines.items():
            serializable_data[model_type] = []
            for run in runs:
                serializable_data[model_type].append({
                    'run_id': run['run_id'],
                    'metrics': run['metrics'],
                    'params': run['params'],
                    'timestamp': run['timestamp']
                })
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Baseline data saved to {output_path}")

def main():
    """Main function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Extract baseline metrics
    extractor = BaselineMetricsExtractor()
    
    # Generate and display report
    report = extractor.generate_baseline_report()
    print(report)
    
    # Save baseline data
    extractor.save_baseline_data()
    
    # Show comparison example
    print("\n" + "=" * 60)
    print("RETRAINED MODEL METRICS (from current MLflow runs)")
    print("=" * 60)
    
    # Get current metrics from the retrained models
    current_baselines = extractor.extract_all_baselines()
    
    for model_type in ['gru', 'lightgbm', 'ppo']:
        if model_type in current_baselines and current_baselines[model_type]:
            latest = current_baselines[model_type][0]
            print(f"\n{model_type.upper()} - Latest Retrained Model:")
            print(f"Run ID: {latest['run_id']}")
            
            metrics = latest['metrics']
            if metrics:
                for metric, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        if 'accuracy' in metric or 'ratio' in metric:
                            print(f"  {metric}: {value:.4f}")
                        elif 'loss' in metric or 'error' in metric:
                            print(f"  {metric}: {value:.6f}")
                        else:
                            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()