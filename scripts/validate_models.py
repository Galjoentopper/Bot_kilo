#!/usr/bin/env python3
"""
Model Validation Script
======================

This script validates imported models to ensure they are compatible
with the current trading environment and can be loaded successfully.
"""

import os
import sys
import json
import logging
import pickle
import joblib
import platform
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"Error: Required dependencies not installed: {e}")
    print("Please run: pip install numpy pandas")
    sys.exit(1)

from src.utils.logger import setup_logging
from src.utils.model_packaging import ModelPackager
from src.models.adapters import create_model_adapter


class ModelValidator:
    """
    Validates imported models for compatibility and functionality.
    """
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.packager = ModelPackager()
        self.logger = logging.getLogger(__name__)
        
        # Track validation results
        self.validation_results = {
            'total_models': 0,
            'valid_models': 0,
            'invalid_models': 0,
            'warnings': 0,
            'errors': [],
            'details': []
        }
        self.last_validation_time = None
    
    def validate_all_models(self, skip_loading: bool = False) -> Dict[str, Any]:
        """
        Validate all models in the models directory.
        
        Args:
            skip_loading: Skip model loading tests for faster validation
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Starting comprehensive model validation...")
        
        if not self.models_dir.exists():
            error_msg = f"Models directory not found: {self.models_dir}"
            self.logger.error(error_msg)
            self.validation_results['errors'].append(error_msg)
            return self.validation_results
        
        # Find all model directories
        model_paths = self._find_model_paths()
        self.validation_results['total_models'] = len(model_paths)
        
        if not model_paths:
            warning_msg = "No models found to validate"
            self.logger.warning(warning_msg)
            self.validation_results['warnings'] += 1
            return self.validation_results
        
        # Validate each model
        for model_path, model_type, symbol in model_paths:
            try:
                result = self._validate_single_model(model_path, model_type, symbol, skip_loading=skip_loading)
                self.validation_results['details'].append(result)
                
                if result['status'] == 'valid':
                    self.validation_results['valid_models'] += 1
                elif result['status'] == 'invalid':
                    self.validation_results['invalid_models'] += 1
                    
                if result.get('warnings'):
                    self.validation_results['warnings'] += len(result['warnings'])
                    
            except Exception as e:
                error_msg = f"Validation failed for {model_type}/{symbol}: {str(e)}"
                self.logger.error(error_msg)
                self.validation_results['errors'].append(error_msg)
                self.validation_results['invalid_models'] += 1
        
        self._log_summary()
        self.last_validation_time = datetime.now()
        return self.validation_results
    
    def validate_filtered_models(self, model_type_filter: Optional[str] = None, 
                                symbol_filter: Optional[str] = None,
                                skip_loading: bool = False) -> Dict[str, Any]:
        """
        Validate models with filters applied.
        
        Args:
            model_type_filter: Only validate this model type
            symbol_filter: Only validate models for this symbol
            skip_loading: Skip model loading tests
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Scanning models directory with filters: {self.models_dir}")
        
        if not self.models_dir.exists():
            error_msg = f"Models directory not found: {self.models_dir}"
            self.logger.error(error_msg)
            self.validation_results['errors'].append(error_msg)
            return self.validation_results
        
        # Find all model paths
        all_model_paths = self._find_model_paths()
        
        # Apply filters
        filtered_paths = []
        for model_path, model_type, symbol in all_model_paths:
            if model_type_filter and model_type != model_type_filter:
                continue
            if symbol_filter and symbol.upper() != symbol_filter.upper():
                continue
            filtered_paths.append((model_path, model_type, symbol))
        
        self.validation_results['total_models'] = len(filtered_paths)
        
        if not filtered_paths:
            warning_msg = "No models found matching the specified filters"
            self.logger.warning(warning_msg)
            self.validation_results['warnings'] += 1
            return self.validation_results
        
        self.logger.info(f"Found {len(filtered_paths)} models matching filters")
        
        # Validate each filtered model
        for model_path, model_type, symbol in filtered_paths:
            try:
                result = self._validate_single_model(model_path, model_type, symbol, skip_loading=skip_loading)
                self.validation_results['details'].append(result)
                
                if result['status'] == 'valid':
                    self.validation_results['valid_models'] += 1
                elif result['status'] == 'invalid':
                    self.validation_results['invalid_models'] += 1
                    
                if result.get('warnings'):
                    self.validation_results['warnings'] += len(result['warnings'])
                    
            except Exception as e:
                error_msg = f"Validation failed for {model_type}/{symbol}: {str(e)}"
                self.logger.error(error_msg)
                self.validation_results['errors'].append(error_msg)
                self.validation_results['invalid_models'] += 1
        
        self._log_summary()
        self.last_validation_time = datetime.now()
        return self.validation_results
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate an HTML report of validation results.
        
        Args:
            output_file: Path to save the HTML report
            
        Returns:
            Path to the generated report file
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'logs/validation_report_{timestamp}.html'
        
        # Ensure logs directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML content
        html_content = self._generate_html_report()
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Validation report generated: {output_file}")
        return output_file
    
    def _generate_html_report(self) -> str:
        """
        Generate HTML content for the validation report.
        
        Returns:
            HTML content as string
        """
        timestamp = self.last_validation_time or datetime.now()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Model Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .model {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .valid {{ border-left: 5px solid #4CAF50; }}
        .invalid {{ border-left: 5px solid #f44336; }}
        .warning {{ color: #ff9800; }}
        .error {{ color: #f44336; }}
        .metadata {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Validation Report</h1>
        <p>Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Models Directory: {self.models_dir}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Models</td><td>{self.validation_results['total_models']}</td></tr>
            <tr><td>Valid Models</td><td>{self.validation_results['valid_models']}</td></tr>
            <tr><td>Invalid Models</td><td>{self.validation_results['invalid_models']}</td></tr>
            <tr><td>Warnings</td><td>{self.validation_results['warnings']}</td></tr>
            <tr><td>Errors</td><td>{len(self.validation_results['errors'])}</td></tr>
        </table>
    </div>
"""
        
        # Add global errors if any
        if self.validation_results['errors']:
            html += "\n    <div class='summary'>\n        <h2>Global Errors</h2>\n        <ul>\n"
            for error in self.validation_results['errors']:
                html += f"            <li class='error'>{error}</li>\n"
            html += "        </ul>\n    </div>\n"
        
        # Add model details
        html += "\n    <div class='models'>\n        <h2>Model Details</h2>\n"
        
        for detail in self.validation_results['details']:
            status_class = 'valid' if detail['status'] == 'valid' else 'invalid'
            html += f"\n        <div class='model {status_class}'>\n"
            html += f"            <h3>{detail['model_type'].upper()} - {detail['symbol']}</h3>\n"
            html += f"            <p><strong>Path:</strong> {detail['path']}</p>\n"
            html += f"            <p><strong>Status:</strong> {detail['status']}</p>\n"
            
            # Add warnings
            if detail.get('warnings'):
                html += "            <div class='warnings'>\n                <h4>Warnings:</h4>\n                <ul>\n"
                for warning in detail['warnings']:
                    html += f"                    <li class='warning'>{warning}</li>\n"
                html += "                </ul>\n            </div>\n"
            
            # Add errors
            if detail.get('errors'):
                html += "            <div class='errors'>\n                <h4>Errors:</h4>\n                <ul>\n"
                for error in detail['errors']:
                    html += f"                    <li class='error'>{error}</li>\n"
                html += "                </ul>\n            </div>\n"
            
            # Add metadata
            if detail.get('metadata'):
                html += "            <div class='metadata'>\n                <h4>Metadata:</h4>\n"
                for key, value in detail['metadata'].items():
                    html += f"                <p><strong>{key}:</strong> {value}</p>\n"
                html += "            </div>\n"
            
            html += "        </div>\n"
        
        html += "    </div>\n</body>\n</html>"
        
        return html
    
    def _find_model_paths(self) -> List[Tuple[Path, str, str]]:
        """
        Find all model paths in the models directory.
        
        Returns:
            List of tuples (model_path, model_type, symbol)
        """
        model_paths = []
        
        # Look for model type directories (lightgbm, gru, ppo)
        for model_type_dir in self.models_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
                
            model_type = model_type_dir.name
            if model_type in ['exports', 'metadata']:
                continue
            
            # Look for symbol directories
            for symbol_dir in model_type_dir.iterdir():
                if not symbol_dir.is_dir():
                    continue
                    
                symbol = symbol_dir.name
                
                # Find the actual model directory (latest or timestamped)
                model_path = self._find_latest_model_path(symbol_dir)
                if model_path:
                    model_paths.append((model_path, model_type, symbol))
        
        return model_paths
    
    def _find_latest_model_path(self, symbol_dir: Path) -> Optional[Path]:
        """
        Find the latest model path in a symbol directory.
        
        Args:
            symbol_dir: Path to symbol directory
            
        Returns:
            Path to latest model or None if not found
        """
        # Check for 'latest' symlink or directory
        latest_path = symbol_dir / 'latest'
        if latest_path.exists():
            if latest_path.is_symlink():
                return latest_path.resolve()
            elif latest_path.is_dir():
                return latest_path
        
        # Check for latest_pointer.txt
        pointer_file = symbol_dir / 'latest_pointer.txt'
        if pointer_file.exists():
            try:
                with open(pointer_file, 'r') as f:
                    pointed_path = Path(f.read().strip())
                    if pointed_path.exists():
                        return pointed_path
            except Exception:
                pass
        
        # Find the most recent timestamped directory
        timestamped_dirs = []
        for item in symbol_dir.iterdir():
            if item.is_dir() and item.name != 'latest':
                try:
                    # Try to parse as timestamp
                    datetime.strptime(item.name, '%Y%m%d_%H%M%S')
                    timestamped_dirs.append(item)
                except ValueError:
                    continue
        
        if timestamped_dirs:
            # Return the most recent one
            return max(timestamped_dirs, key=lambda x: x.name)
        
        return None
    
    def _validate_single_model(self, model_path: Path, model_type: str, symbol: str, skip_loading: bool = False) -> Dict[str, Any]:
        """
        Validate a single model.
        
        Args:
            model_path: Path to model directory
            model_type: Type of model (lightgbm, gru, ppo)
            symbol: Trading symbol
            skip_loading: Skip model loading tests
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'model_type': model_type,
            'symbol': symbol,
            'path': str(model_path),
            'status': 'unknown',
            'warnings': [],
            'errors': [],
            'metadata': {}
        }
        
        self.logger.info(f"Validating {model_type} model for {symbol}...")
        
        try:
            # Check if model directory exists and has required files
            if not model_path.exists():
                result['status'] = 'invalid'
                result['errors'].append(f"Model path does not exist: {model_path}")
                return result
            
            # Check for required files
            required_files = self._get_required_files(model_type)
            missing_files = []
            
            for file_name in required_files:
                file_path = model_path / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                result['status'] = 'invalid'
                result['errors'].append(f"Missing required files: {missing_files}")
                return result
            
            # Load and validate metadata if available
            metadata_file = model_path / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    result['metadata'] = metadata
                    
                    # Validate metadata structure
                    self._validate_metadata(metadata, result)
                    
                except Exception as e:
                    result['warnings'].append(f"Failed to load metadata: {str(e)}")
            else:
                result['warnings'].append("No metadata file found")
            
            # Try to load the model using the adapter (unless skipped)
            if not skip_loading:
                try:
                    self._test_model_loading(model_path, model_type, result)
                except Exception as e:
                    result['status'] = 'invalid'
                    result['errors'].append(f"Model loading failed: {str(e)}")
                    return result
            else:
                result['metadata']['loading_test'] = 'skipped'
            
            # Check features file
            features_file = model_path / 'features.json'
            if features_file.exists():
                try:
                    with open(features_file, 'r') as f:
                        features_data = json.load(f)
                    
                    if 'feature_names' not in features_data:
                        result['warnings'].append("Features file missing feature_names")
                    elif not features_data['feature_names']:
                        result['warnings'].append("Empty feature_names list")
                    else:
                        result['metadata']['actual_features'] = len(features_data['feature_names'])
                        
                        # Validate feature count consistency
                        if 'expected_features' in result['metadata']:
                            expected = result['metadata']['expected_features']
                            actual = result['metadata']['actual_features']
                            if expected != actual:
                                result['warnings'].append(
                                    f"Feature count mismatch: metadata says {expected}, features file has {actual}"
                                )
                        
                except Exception as e:
                    result['warnings'].append(f"Failed to load features file: {str(e)}")
            else:
                result['warnings'].append("No features file found")
            
            # Validate preprocessor files
            self._validate_preprocessors(model_path, result)
            
            # Validate model file integrity
            self._validate_model_file_integrity(model_path, model_type, result)
            
            # If we get here, the model is valid
            if result['status'] == 'unknown':
                result['status'] = 'valid'
            
        except Exception as e:
            result['status'] = 'invalid'
            result['errors'].append(f"Validation error: {str(e)}")
        
        return result
    
    def _get_required_files(self, model_type: str) -> List[str]:
        """
        Get list of required files for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            List of required file names
        """
        base_files = ['features.json']
        
        if model_type == 'lightgbm':
            return base_files + ['model.pkl']
        elif model_type == 'gru':
            return base_files + ['model.pt']
        elif model_type == 'ppo':
            return base_files + ['model.zip']
        else:
            return base_files
    
    def _validate_metadata(self, metadata: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Validate metadata structure and content with comprehensive compatibility checking.
        
        Args:
            metadata: Metadata dictionary
            result: Result dictionary to update
        """
        required_fields = ['model_type', 'symbol', 'created_at', 'python_version']
        recommended_fields = ['training_config', 'feature_count', 'model_version', 'dependencies']
        
        # Check required fields
        for field in required_fields:
            if field not in metadata:
                result['warnings'].append(f"Missing required metadata field: {field}")
        
        # Check recommended fields
        for field in recommended_fields:
            if field not in metadata:
                result['warnings'].append(f"Missing recommended metadata field: {field}")
        
        # Validate Python version compatibility
        self._validate_python_version(metadata, result)
        
        # Validate model age
        self._validate_model_age(metadata, result)
        
        # Validate dependencies
        self._validate_dependencies(metadata, result)
        
        # Validate training environment
        self._validate_training_environment(metadata, result)
        
        # Validate feature compatibility
        self._validate_feature_compatibility(metadata, result)
    
    def _validate_python_version(self, metadata: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate Python version compatibility."""
        if 'python_version' not in metadata:
            return
            
        current_version = sys.version_info[:2]
        try:
            model_version_str = metadata['python_version']
            model_version = tuple(map(int, model_version_str.split('.')[:2]))
            
            if model_version != current_version:
                # Check if versions are compatible
                major_diff = abs(model_version[0] - current_version[0])
                minor_diff = abs(model_version[1] - current_version[1])
                
                if major_diff > 0:
                    result['errors'].append(
                        f"Major Python version incompatibility: model trained on {model_version_str}, "
                        f"current is {'.'.join(map(str, current_version))}"
                    )
                elif minor_diff > 1:
                    result['warnings'].append(
                        f"Python minor version difference may cause issues: model trained on {model_version_str}, "
                        f"current is {'.'.join(map(str, current_version))}"
                    )
                else:
                    result['warnings'].append(
                        f"Python version mismatch (likely compatible): model trained on {model_version_str}, "
                        f"current is {'.'.join(map(str, current_version))}"
                    )
        except Exception as e:
            result['warnings'].append(f"Invalid python_version format in metadata: {str(e)}")
    
    def _validate_model_age(self, metadata: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate model age and freshness."""
        if 'created_at' not in metadata:
            return
            
        try:
            created_at = datetime.fromisoformat(metadata['created_at'].replace('Z', '+00:00'))
            age = datetime.now() - created_at.replace(tzinfo=None)
            
            if age > timedelta(days=90):
                result['warnings'].append(f"Model is {age.days} days old - consider retraining")
            elif age > timedelta(days=30):
                result['warnings'].append(f"Model is {age.days} days old - monitor performance")
                
        except Exception as e:
            result['warnings'].append(f"Invalid created_at format in metadata: {str(e)}")
    
    def _validate_dependencies(self, metadata: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate required dependencies."""
        if 'dependencies' not in metadata:
            return
            
        dependencies = metadata['dependencies']
        if not isinstance(dependencies, dict):
            result['warnings'].append("Dependencies should be a dictionary")
            return
            
        missing_deps = []
        version_mismatches = []
        
        for dep_name, required_version in dependencies.items():
            try:
                spec = importlib.util.find_spec(dep_name)
                if spec is None:
                    missing_deps.append(dep_name)
                    continue
                    
                # Try to get version (this is best effort)
                try:
                    module = importlib.import_module(dep_name)
                    if hasattr(module, '__version__'):
                        current_version = module.__version__
                        if current_version != required_version:
                            version_mismatches.append(f"{dep_name}: required {required_version}, found {current_version}")
                except Exception:
                    pass  # Version checking is best effort
                    
            except Exception:
                missing_deps.append(dep_name)
        
        if missing_deps:
            result['errors'].append(f"Missing dependencies: {', '.join(missing_deps)}")
        
        if version_mismatches:
            result['warnings'].append(f"Version mismatches: {'; '.join(version_mismatches)}")
    
    def _validate_training_environment(self, metadata: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate training environment compatibility."""
        if 'training_environment' not in metadata:
            return
            
        training_env = metadata['training_environment']
        current_platform = platform.system().lower()
        
        if 'platform' in training_env:
            training_platform = training_env['platform'].lower()
            if training_platform != current_platform:
                result['warnings'].append(
                    f"Platform difference: trained on {training_platform}, running on {current_platform}"
                )
    
    def _validate_feature_compatibility(self, metadata: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate feature compatibility."""
        if 'feature_count' in metadata:
            expected_features = metadata['feature_count']
            if not isinstance(expected_features, int) or expected_features <= 0:
                result['warnings'].append("Invalid feature_count in metadata")
            else:
                result['metadata']['expected_features'] = expected_features
    
    def _validate_preprocessors(self, model_path: Path, result: Dict[str, Any]) -> None:
        """Validate preprocessor files."""
        preprocessor_files = [
            'preprocessor.pkl',
            'preprocessor.joblib',
            'scaler.pkl',
            'scaler.joblib'
        ]
        
        found_preprocessors = []
        for file_name in preprocessor_files:
            file_path = model_path / file_name
            if file_path.exists():
                found_preprocessors.append(file_name)
                
                # Try to load the preprocessor
                try:
                    if file_name.endswith('.pkl'):
                        with open(file_path, 'rb') as f:
                            preprocessor = pickle.load(f)
                    else:  # .joblib
                        preprocessor = joblib.load(file_path)
                    
                    # Check if it has transform method
                    if not hasattr(preprocessor, 'transform'):
                        result['warnings'].append(f"Preprocessor {file_name} missing transform method")
                    else:
                        result['metadata'][f'{file_name}_valid'] = True
                        
                except Exception as e:
                    result['warnings'].append(f"Failed to load preprocessor {file_name}: {str(e)}")
        
        if not found_preprocessors:
            result['warnings'].append("No preprocessor files found")
        else:
            result['metadata']['preprocessors_found'] = found_preprocessors
    
    def _validate_model_file_integrity(self, model_path: Path, model_type: str, result: Dict[str, Any]) -> None:
        """Validate model file integrity and size."""
        model_files = {
            'lightgbm': ['model.pkl'],
            'gru': ['model.pt'],
            'ppo': ['model.zip']
        }
        
        if model_type not in model_files:
            return
            
        for file_name in model_files[model_type]:
            file_path = model_path / file_name
            if file_path.exists():
                try:
                    file_size = file_path.stat().st_size
                    result['metadata'][f'{file_name}_size'] = file_size
                    
                    # Check for suspiciously small files
                    if file_size < 1024:  # Less than 1KB
                        result['warnings'].append(f"Model file {file_name} is suspiciously small ({file_size} bytes)")
                    
                    # Check for very large files
                    if file_size > 500 * 1024 * 1024:  # More than 500MB
                        result['warnings'].append(f"Model file {file_name} is very large ({file_size / (1024*1024):.1f} MB)")
                        
                except Exception as e:
                    result['warnings'].append(f"Failed to check file size for {file_name}: {str(e)}")
    
    def _test_model_loading(self, model_path: Path, model_type: str, result: Dict[str, Any]) -> None:
        """
        Test if the model can be loaded successfully with enhanced error reporting.
        
        Args:
            model_path: Path to model directory
            model_type: Type of model
            result: Result dictionary to update
        """
        try:
            # Create a dummy config for testing
            dummy_config = {
                'models': {
                    model_type: {
                        'enabled': True
                    }
                }
            }
            
            # Try to create model adapter
            adapter = create_model_adapter(model_type, dummy_config, 'regression')
            
            # Try to load the model
            if hasattr(adapter, 'load'):
                adapter.load(str(model_path))
                result['metadata']['loading_test'] = 'passed'
                
                # Additional validation for specific model types
                if model_type == 'lightgbm' and hasattr(adapter, 'model'):
                    if hasattr(adapter.model, 'num_feature'):
                        result['metadata']['model_features'] = adapter.model.num_feature()
                elif model_type == 'gru' and hasattr(adapter, 'model'):
                    if hasattr(adapter.model, 'parameters'):
                        param_count = sum(p.numel() for p in adapter.model.parameters())
                        result['metadata']['model_parameters'] = param_count
                        
            else:
                result['warnings'].append(f"Model adapter for {model_type} does not support loading")
                
        except ImportError as e:
            result['warnings'].append(f"Missing dependencies for {model_type}: {str(e)}")
        except Exception as e:
            raise Exception(f"Model loading test failed: {str(e)}")
    
    def _log_summary(self) -> None:
        """
        Log validation summary.
        """
        results = self.validation_results
        
        self.logger.info("\n" + "="*50)
        self.logger.info("MODEL VALIDATION SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total models checked: {results['total_models']}")
        self.logger.info(f"Valid models: {results['valid_models']}")
        self.logger.info(f"Invalid models: {results['invalid_models']}")
        self.logger.info(f"Warnings: {results['warnings']}")
        
        if results['errors']:
            self.logger.error("\nErrors encountered:")
            for error in results['errors']:
                self.logger.error(f"  - {error}")
        
        # Log details for each model
        self.logger.info("\nModel Details:")
        for detail in results['details']:
            status_icon = "✅" if detail['status'] == 'valid' else "❌"
            self.logger.info(f"  {status_icon} {detail['model_type']}/{detail['symbol']} - {detail['status']}")
            
            if detail['warnings']:
                for warning in detail['warnings']:
                    self.logger.warning(f"    ⚠️  {warning}")
            
            if detail['errors']:
                for error in detail['errors']:
                    self.logger.error(f"    ❌ {error}")
        
        self.logger.info("="*50)


def main():
    """
    Main function to run model validation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate trading models')
    parser.add_argument('--type', help='Filter by model type (lightgbm, gru, ppo)')
    parser.add_argument('--symbol', help='Filter by trading symbol')
    parser.add_argument('--quick', action='store_true', help='Skip model loading tests (default behavior)')
    parser.add_argument('--full-test', action='store_true', help='Enable model loading tests (may be slow/hang)')
    parser.add_argument('--report', help='Generate HTML report to specified file')
    parser.add_argument('--models-dir', default='models', help='Models directory path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/model_validation.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Run validation
    validator = ModelValidator(args.models_dir)
    
    try:
        # Default to quick mode (skip model loading) to prevent hanging
        # Only do full testing if explicitly requested with --full-test
        skip_loading = not getattr(args, 'full_test', False)
        
        if args.type or args.symbol:
            results = validator.validate_filtered_models(
                model_type_filter=args.type,
                symbol_filter=args.symbol,
                skip_loading=skip_loading
            )
        else:
            results = validator.validate_all_models(skip_loading=skip_loading)
        
        # Generate report if requested
        if args.report:
            report_path = validator.generate_report(args.report)
            print(f"\nHTML report generated: {report_path}")
        
        # Print summary
        print(f"\nValidation Summary:")
        print(f"Total Models: {results['total_models']}")
        print(f"Valid Models: {results['valid_models']}")
        print(f"Invalid Models: {results['invalid_models']}")
        print(f"Warnings: {results['warnings']}")
        print(f"Errors: {len(results['errors'])}")
        
        # Exit with appropriate code
        if results['invalid_models'] > 0 or results['errors']:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logging.error(f"Validation failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()