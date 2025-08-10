#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
====================================

Comprehensive evaluation of retrained models with performance benchmarks,
edge case testing, and comparative analysis.
"""

import os
import sys
import time
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.logger import setup_logging
from src.data_pipeline.loader import DataLoader
from src.data_pipeline.features import FeatureEngine
from src.data_pipeline.preprocess import DataPreprocessor
from src.models.gru_trainer import GRUTrainer
from src.models.lgbm_trainer import LightGBMTrainer
from src.models.ppo_trainer import PPOTrainer
from src.backtesting.backtest import Backtester
from scripts.extract_baseline_metrics import BaselineMetricsExtractor

logger = logging.getLogger(__name__)

class ComprehensiveModelEvaluator:
    """Comprehensive model evaluation with benchmarks and analysis."""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """Initialize the evaluator."""
        import yaml
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = "./models"
        self.data_loader = DataLoader(self.config.get('data', {}).get('data_dir', './data'))
        self.feature_engine = FeatureEngine(self.config.get('features', {}))
        self.preprocessor = DataPreprocessor()
        self.baseline_extractor = BaselineMetricsExtractor()
        
        # Load baseline metrics
        self.baseline_metrics = self.baseline_extractor.extract_all_baselines()
        
        # Results storage
        self.evaluation_results = {}
        self.performance_benchmarks = {}
        self.edge_case_results = {}
        
        logger.info("Comprehensive Model Evaluator initialized")
    
    def load_retrained_models(self) -> Dict[str, Any]:
        """Load all retrained models."""
        models = {}
        
        try:
            # Load GRU model
            gru_files = list(Path(self.models_dir).glob("gru_model_*.pth"))
            if gru_files:
                latest_gru = max(gru_files, key=os.path.getmtime)
                models['gru'] = GRUTrainer.load_model(str(latest_gru), self.config)
                logger.info(f"Loaded GRU model: {latest_gru}")
            
            # Load LightGBM model
            lgbm_files = list(Path(self.models_dir).glob("lightgbm_model_*.pkl"))
            if lgbm_files:
                latest_lgbm = max(lgbm_files, key=os.path.getmtime)
                models['lightgbm'] = LightGBMTrainer.load_model(str(latest_lgbm), self.config)
                logger.info(f"Loaded LightGBM model: {latest_lgbm}")
            
            # Load PPO model
            ppo_files = list(Path(self.models_dir).glob("ppo_model_*.zip"))
            if ppo_files:
                latest_ppo = max(ppo_files, key=os.path.getmtime)
                models['ppo'] = PPOTrainer.load_model(str(latest_ppo), self.config)
                logger.info(f"Loaded PPO model: {latest_ppo}")
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
        
        return models
    
    def prepare_test_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare test datasets for evaluation."""
        test_datasets = {}
        
        # Load recent market data
        symbols = self.config.get('data', {}).get('symbols', ['BTCEUR'])
        
        for symbol in symbols:
            try:
                # Get recent data (last 30 days)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                data = self.data_loader.load_symbol_data(
                    symbol, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty:
                    # Generate features
                    data_with_features = self.feature_engine.generate_all_features(data)
                    test_datasets[symbol] = data_with_features
                    logger.info(f"Prepared test data for {symbol}: {len(data_with_features)} samples")
            
            except Exception as e:
                logger.warning(f"Could not load data for {symbol}: {e}")
        
        return test_datasets
    
    def evaluate_model_accuracy(self, models: Dict[str, Any], test_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Evaluate model accuracy metrics."""
        accuracy_results = {}
        
        for model_name, model in models.items():
            accuracy_results[model_name] = {}
            
            for symbol, data in test_data.items():
                try:
                    # Prepare features and targets
                    feature_names = self.feature_engine.get_feature_names(data)
                    features = data[feature_names].dropna()
                    
                    if features.empty:
                        continue
                    
                    # Create targets (next period returns)
                    targets = data['close'].pct_change().shift(-1).dropna()
                    
                    # Align features and targets
                    min_len = min(len(features), len(targets))
                    features = features.iloc[:min_len]
                    targets = targets.iloc[:min_len]
                    
                    if model_name == 'gru':
                        # Prepare sequences for GRU
                        sequence_length = self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
                        if len(features) >= sequence_length:
                            # Scale features
                            features_scaled = self.preprocessor.fit_transform(features)
                            
                            # Create sequences
                            X_sequences = []
                            y_sequences = []
                            
                            for i in range(sequence_length, len(features_scaled)):
                                X_sequences.append(features_scaled[i-sequence_length:i])
                                y_sequences.append(targets.iloc[i])
                            
                            if X_sequences:
                                X_test = np.array(X_sequences)
                                y_test = np.array(y_sequences)
                                
                                # Evaluate
                                metrics = model.evaluate(X_test, y_test)
                                accuracy_results[model_name][symbol] = metrics
                    
                    elif model_name == 'lightgbm':
                        # Pad features for LightGBM compatibility
                        features_padded = self.feature_engine.pad_features_for_model(features, 'lightgbm')
                        
                        # Align again after padding
                        min_len = min(len(features_padded), len(targets))
                        features_padded = features_padded.iloc[:min_len]
                        targets_aligned = targets.iloc[:min_len]
                        
                        # Evaluate
                        metrics = model.evaluate(features_padded, targets_aligned.values)
                        accuracy_results[model_name][symbol] = metrics
                    
                    elif model_name == 'ppo':
                        # PPO evaluation requires environment setup
                        # For now, we'll skip detailed PPO accuracy evaluation
                        # and focus on trading performance
                        accuracy_results[model_name][symbol] = {'note': 'PPO evaluated via trading performance'}
                
                except Exception as e:
                    logger.warning(f"Error evaluating {model_name} on {symbol}: {e}")
        
        return accuracy_results
    
    def benchmark_processing_speed(self, models: Dict[str, Any], test_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Benchmark model processing speed and latency."""
        speed_results = {}
        
        for model_name, model in models.items():
            speed_results[model_name] = {}
            
            for symbol, data in test_data.items():
                try:
                    # Prepare test features
                    feature_names = self.feature_engine.get_feature_names(data)
                    features = data[feature_names].dropna()
                    
                    if features.empty:
                        continue
                    
                    # Benchmark parameters
                    n_predictions = 100
                    latencies = []
                    
                    if model_name == 'gru':
                        sequence_length = self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
                        if len(features) >= sequence_length:
                            features_scaled = self.preprocessor.fit_transform(features)
                            
                            for i in range(n_predictions):
                                # Create single sequence
                                start_idx = np.random.randint(sequence_length, len(features_scaled))
                                sequence = features_scaled[start_idx-sequence_length:start_idx].reshape(1, sequence_length, -1)
                                
                                # Measure prediction time
                                start_time = time.time()
                                _ = model.predict(sequence)
                                end_time = time.time()
                                
                                latencies.append((end_time - start_time) * 1000)  # Convert to ms
                    
                    elif model_name == 'lightgbm':
                        features_padded = self.feature_engine.pad_features_for_model(features, 'lightgbm')
                        
                        for i in range(n_predictions):
                            # Get random sample
                            sample_idx = np.random.randint(0, len(features_padded))
                            sample = features_padded.iloc[sample_idx:sample_idx+1]
                            
                            # Measure prediction time
                            start_time = time.time()
                            _ = model.predict(sample)
                            end_time = time.time()
                            
                            latencies.append((end_time - start_time) * 1000)  # Convert to ms
                    
                    elif model_name == 'ppo':
                        # PPO speed benchmark with correct observation space
                        for i in range(n_predictions):
                            # Create proper 2D observation: (20, 116)
                            sequence_length = 20
                            feature_count = 116
                            observation = np.random.randn(sequence_length, feature_count).astype(np.float32)
                            
                            # Measure prediction time
                            start_time = time.time()
                            _ = model.predict(observation, deterministic=True)
                            end_time = time.time()
                            
                            latencies.append((end_time - start_time) * 1000)  # Convert to ms
                    
                    if latencies:
                        speed_results[model_name][symbol] = {
                            'mean_latency_ms': np.mean(latencies),
                            'median_latency_ms': np.median(latencies),
                            'p95_latency_ms': np.percentile(latencies, 95),
                            'p99_latency_ms': np.percentile(latencies, 99),
                            'throughput_per_sec': 1000 / np.mean(latencies),
                            'std_latency_ms': np.std(latencies)
                        }
                
                except Exception as e:
                    logger.warning(f"Error benchmarking {model_name} on {symbol}: {e}")
        
        return speed_results
    
    def test_edge_cases(self, models: Dict[str, Any], test_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Test models under edge case scenarios."""
        edge_case_results = {}
        
        for model_name, model in models.items():
            edge_case_results[model_name] = {}
            
            for symbol, data in test_data.items():
                try:
                    feature_names = self.feature_engine.get_feature_names(data)
                    features = data[feature_names].dropna()
                    
                    if features.empty:
                        continue
                    
                    edge_cases = {}
                    
                    # Test 1: High volatility periods
                    volatility = data['close'].pct_change().rolling(20).std()
                    high_vol_mask = volatility > volatility.quantile(0.9)
                    high_vol_features = features[high_vol_mask]
                    
                    if not high_vol_features.empty:
                        edge_cases['high_volatility'] = self._test_model_stability(
                            model, model_name, high_vol_features, 'high_volatility'
                        )
                    
                    # Test 2: Low volatility periods
                    low_vol_mask = volatility < volatility.quantile(0.1)
                    low_vol_features = features[low_vol_mask]
                    
                    if not low_vol_features.empty:
                        edge_cases['low_volatility'] = self._test_model_stability(
                            model, model_name, low_vol_features, 'low_volatility'
                        )
                    
                    # Test 3: Extreme values
                    extreme_features = features.copy()
                    # Add some extreme values (3 standard deviations)
                    for col in extreme_features.columns:
                        if extreme_features[col].std() > 0:
                            extreme_val = extreme_features[col].mean() + 3 * extreme_features[col].std()
                            extreme_features.loc[extreme_features.index[0], col] = extreme_val
                    
                    edge_cases['extreme_values'] = self._test_model_stability(
                        model, model_name, extreme_features, 'extreme_values'
                    )
                    
                    # Test 4: Missing data handling
                    missing_features = features.copy()
                    # Introduce some NaN values
                    mask = np.random.random(missing_features.shape) < 0.1
                    missing_features = missing_features.mask(mask)
                    missing_features = missing_features.fillna(0)  # Fill with zeros
                    
                    edge_cases['missing_data'] = self._test_model_stability(
                        model, model_name, missing_features, 'missing_data'
                    )
                    
                    edge_case_results[model_name][symbol] = edge_cases
                
                except Exception as e:
                    logger.warning(f"Error testing edge cases for {model_name} on {symbol}: {e}")
        
        return edge_case_results
    
    def _test_model_stability(self, model: Any, model_name: str, features: pd.DataFrame, test_name: str) -> Dict[str, Any]:
        """Test model stability under specific conditions."""
        try:
            predictions = []
            errors = 0
            
            if model_name == 'gru':
                sequence_length = self.config.get('models', {}).get('gru', {}).get('sequence_length', 20)
                if len(features) >= sequence_length:
                    features_scaled = self.preprocessor.fit_transform(features)
                    
                    for i in range(min(10, len(features_scaled) - sequence_length)):
                        try:
                            sequence = features_scaled[i:i+sequence_length].reshape(1, sequence_length, -1)
                            pred = model.predict(sequence)[0]
                            predictions.append(pred)
                        except Exception:
                            errors += 1
            
            elif model_name == 'lightgbm':
                features_padded = self.feature_engine.pad_features_for_model(features, 'lightgbm')
                
                for i in range(min(10, len(features_padded))):
                    try:
                        sample = features_padded.iloc[i:i+1]
                        pred = model.predict(sample)[0]
                        predictions.append(pred)
                    except Exception:
                        errors += 1
            
            elif model_name == 'ppo':
                for i in range(min(10, len(features))):
                    try:
                        # Create proper 2D observation: (20, 116)
                        sequence_length = 20
                        feature_count = 116
                        observation = np.random.randn(sequence_length, feature_count).astype(np.float32)
                        pred, _ = model.predict(observation, deterministic=True)
                        predictions.append(pred)
                    except Exception:
                        errors += 1
            
            # Analyze predictions
            if predictions:
                predictions = np.array(predictions)
                return {
                    'num_predictions': len(predictions),
                    'num_errors': errors,
                    'error_rate': errors / (len(predictions) + errors) if (len(predictions) + errors) > 0 else 0,
                    'prediction_mean': float(np.mean(predictions)),
                    'prediction_std': float(np.std(predictions)),
                    'has_nan': bool(np.isnan(predictions).any()),
                    'has_inf': bool(np.isinf(predictions).any()),
                    'stability_score': 1.0 - (errors / max(1, len(predictions) + errors))
                }
            else:
                return {
                    'num_predictions': 0,
                    'num_errors': errors,
                    'error_rate': 1.0,
                    'stability_score': 0.0
                }
        
        except Exception as e:
            return {
                'error': str(e),
                'stability_score': 0.0
            }
    
    def compare_with_baseline(self, current_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Compare current results with baseline metrics."""
        comparisons = {}
        
        for model_type in ['gru', 'lightgbm', 'ppo']:
            if model_type in self.baseline_metrics and self.baseline_metrics[model_type]:
                baseline = self.baseline_metrics[model_type][0]  # Latest baseline
                baseline_metrics = baseline.get('metrics', {})
                
                if model_type in current_results:
                    current_metrics = current_results[model_type]
                    
                    # Compare common metrics
                    comparison = self.baseline_extractor.compare_metrics(baseline_metrics, current_metrics)
                    comparisons[model_type] = {
                        'baseline_run_id': baseline.get('run_id'),
                        'baseline_timestamp': baseline.get('timestamp'),
                        'metric_comparisons': comparison,
                        'overall_improvement': self._calculate_overall_improvement(comparison)
                    }
        
        return comparisons
    
    def _calculate_overall_improvement(self, comparison: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall improvement score."""
        if not comparison:
            return 0.0
        
        improvements = []
        for metric, data in comparison.items():
            if 'improvement_pct' in data and not np.isinf(data['improvement_pct']):
                # Weight different metrics
                weight = 1.0
                if 'accuracy' in metric.lower():
                    weight = 2.0  # Higher weight for accuracy
                elif 'loss' in metric.lower() or 'error' in metric.lower():
                    weight = 2.0
                    # For loss/error metrics, improvement is negative change
                    improvements.append(-data['improvement_pct'] * weight)
                else:
                    improvements.append(data['improvement_pct'] * weight)
        
        return np.mean(improvements) if improvements else 0.0
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL RE-EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        
        if hasattr(self, 'models') and self.models:
            report.append(f"Models Evaluated: {', '.join(self.models.keys())}")
        
        if hasattr(self, 'test_datasets') and self.test_datasets:
            report.append(f"Symbols Tested: {', '.join(self.test_datasets.keys())}")
        
        report.append("")
        
        # Accuracy Results
        if hasattr(self, 'accuracy_results'):
            report.append("MODEL ACCURACY EVALUATION")
            report.append("-" * 40)
            
            for model_name, results in self.accuracy_results.items():
                report.append(f"\n{model_name.upper()} Model:")
                for symbol, metrics in results.items():
                    report.append(f"  {symbol}:")
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            if 'accuracy' in metric or 'r2' in metric:
                                report.append(f"    {metric}: {value:.4f}")
                            else:
                                report.append(f"    {metric}: {value:.6f}")
                        else:
                            report.append(f"    {metric}: {value}")
        
        # Speed Benchmarks
        if hasattr(self, 'speed_results'):
            report.append("\n\nPROCESSING SPEED BENCHMARKS")
            report.append("-" * 40)
            
            for model_name, results in self.speed_results.items():
                report.append(f"\n{model_name.upper()} Model:")
                for symbol, metrics in results.items():
                    report.append(f"  {symbol}:")
                    for metric, value in metrics.items():
                        report.append(f"    {metric}: {value:.2f}")
        
        # Edge Case Results
        if hasattr(self, 'edge_case_results'):
            report.append("\n\nEDGE CASE TESTING RESULTS")
            report.append("-" * 40)
            
            for model_name, results in self.edge_case_results.items():
                report.append(f"\n{model_name.upper()} Model:")
                for symbol, edge_cases in results.items():
                    report.append(f"  {symbol}:")
                    for case_name, case_results in edge_cases.items():
                        stability_score = case_results.get('stability_score', 0)
                        report.append(f"    {case_name}: Stability Score {stability_score:.3f}")
        
        # Baseline Comparison
        if hasattr(self, 'baseline_comparison'):
            report.append("\n\nBASELINE COMPARISON")
            report.append("-" * 40)
            
            for model_name, comparison in self.baseline_comparison.items():
                report.append(f"\n{model_name.upper()} Model:")
                overall_improvement = comparison.get('overall_improvement', 0)
                report.append(f"  Overall Improvement: {overall_improvement:.2f}%")
                
                metric_comparisons = comparison.get('metric_comparisons', {})
                for metric, data in metric_comparisons.items():
                    improvement = data.get('improvement_pct', 0)
                    if not np.isinf(improvement):
                        report.append(f"  {metric}: {improvement:+.2f}%")
        
        report.append("\n\n" + "=" * 80)
        
        return "\n".join(report)
    
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation suite."""
        logger.info("Starting comprehensive model evaluation")
        
        # Load models
        self.models = self.load_retrained_models()
        if not self.models:
            logger.error("No models loaded for evaluation")
            return {}
        
        # Prepare test data
        self.test_datasets = self.prepare_test_data()
        if not self.test_datasets:
            logger.error("No test data available")
            return {}
        
        # Run evaluations
        logger.info("Evaluating model accuracy...")
        self.accuracy_results = self.evaluate_model_accuracy(self.models, self.test_datasets)
        
        logger.info("Benchmarking processing speed...")
        self.speed_results = self.benchmark_processing_speed(self.models, self.test_datasets)
        
        logger.info("Testing edge cases...")
        self.edge_case_results = self.test_edge_cases(self.models, self.test_datasets)
        
        logger.info("Comparing with baseline...")
        # Extract current metrics for comparison
        current_metrics = {}
        for model_name in self.models.keys():
            if model_name in self.accuracy_results:
                # Aggregate metrics across symbols
                all_metrics = {}
                for symbol, metrics in self.accuracy_results[model_name].items():
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if metric not in all_metrics:
                                all_metrics[metric] = []
                            all_metrics[metric].append(value)
                
                # Average metrics
                current_metrics[model_name] = {
                    metric: np.mean(values) for metric, values in all_metrics.items()
                }
        
        self.baseline_comparison = self.compare_with_baseline(current_metrics)
        
        # Generate report
        report = self.generate_comprehensive_report()
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(self.models.keys()),
            'symbols_tested': list(self.test_datasets.keys()),
            'accuracy_results': self.accuracy_results,
            'speed_results': self.speed_results,
            'edge_case_results': self.edge_case_results,
            'baseline_comparison': self.baseline_comparison,
            'report': report
        }
        
        # Save to file
        with open('comprehensive_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Comprehensive evaluation completed")
        return results

async def main():
    """Main function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation
    evaluator = ComprehensiveModelEvaluator()
    results = await evaluator.run_comprehensive_evaluation()
    
    # Display report
    if 'report' in results:
        print(results['report'])
    
    print(f"\nDetailed results saved to: comprehensive_evaluation_results.json")

if __name__ == "__main__":
    asyncio.run(main())