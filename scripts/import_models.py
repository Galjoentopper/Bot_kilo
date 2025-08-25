#!/usr/bin/env python3
"""
Model Import Script
==================

This script imports models from a transfer bundle created on another computer.
It handles unpacking, validation, and integration of transferred models.
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import zipfile
except ImportError:
    print("Error: zipfile module not available")
    sys.exit(1)

from src.utils.logger import setup_logging
from src.utils.model_transfer import ModelTransferManager
from src.utils.model_packaging import ModelPackager


class ModelImporter:
    """
    Handles importing models from transfer bundles.
    """
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.transfer_manager = ModelTransferManager()
        self.logger = logging.getLogger(__name__)
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def import_from_bundle(self, bundle_path: str, backup_existing: bool = True, 
                          validate_after_import: bool = True) -> Dict[str, Any]:
        """
        Import models from a transfer bundle.
        
        Args:
            bundle_path: Path to the transfer bundle ZIP file
            backup_existing: Whether to backup existing models before import
            validate_after_import: Whether to validate models after import
            
        Returns:
            Dictionary with import results
        """
        bundle_path = Path(bundle_path)
        
        if not bundle_path.exists():
            raise FileNotFoundError(f"Transfer bundle not found: {bundle_path}")
        
        if not bundle_path.suffix.lower() == '.zip':
            raise ValueError("Transfer bundle must be a ZIP file")
        
        self.logger.info(f"Starting import from bundle: {bundle_path}")
        
        # Create temporary extraction directory
        temp_dir = Path(f"temp_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract bundle
            self.logger.info("Extracting transfer bundle...")
            with zipfile.ZipFile(bundle_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Try transfer_manifest.json (directory-style bundle) first; fallback to bundle_info.json (package-style)
            manifest_path = temp_dir / 'transfer_manifest.json'
            bundle_info_path = temp_dir / 'bundle_info.json'
            
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                self.logger.info(f"Found {len(manifest['models'])} models in bundle (manifest)")
                
                # Backup existing models if requested
                if backup_existing:
                    self._backup_existing_models()
                
                # Import each model (directory-style)
                import_results = {
                    'total_models': len(manifest['models']),
                    'imported_models': 0,
                    'failed_models': 0,
                    'warnings': [],
                    'errors': [],
                    'details': []
                }
                
                for model_info in manifest['models']:
                    try:
                        result = self._import_single_model(temp_dir, model_info)
                        import_results['details'].append(result)
                        
                        if result['status'] == 'success':
                            import_results['imported_models'] += 1
                        else:
                            import_results['failed_models'] += 1
                            
                        if result.get('warnings'):
                            import_results['warnings'].extend(result['warnings'])
                            
                    except Exception as e:
                        error_msg = f"Failed to import {model_info.get('model_type', 'unknown')}/{model_info.get('symbol', 'unknown')}: {str(e)}"
                        self.logger.error(error_msg)
                        import_results['errors'].append(error_msg)
                        import_results['failed_models'] += 1
                
                # Validate imported models if requested
                if validate_after_import and import_results['imported_models'] > 0:
                    self.logger.info("Validating imported models...")
                    validation_results = self._validate_imported_models()
                    import_results['validation'] = validation_results
                
                self._log_import_summary(import_results)
                return import_results
            
            elif bundle_info_path.exists():
                # Package-style bundle created by ModelPackager.create_transfer_bundle
                with open(bundle_info_path, 'r') as f:
                    bundle_info = json.load(f)
                
                packages = bundle_info.get('packages', [])
                self.logger.info(f"Found {len(packages)} packaged models in bundle (bundle_info)")
                
                # Backup existing models if requested
                if backup_existing:
                    self._backup_existing_models()
                
                import_results = {
                    'total_models': len(packages),
                    'imported_models': 0,
                    'failed_models': 0,
                    'warnings': [],
                    'errors': [],
                    'details': []
                }
                
                packager = ModelPackager()
                for pkg in packages:
                    model_type = pkg.get('type') or pkg.get('model_type') or 'unknown'
                    symbol = pkg.get('symbol', 'unknown')
                    file_name = pkg.get('file') or pkg.get('path')
                    detail = {
                        'model_type': model_type,
                        'symbol': symbol,
                        'status': 'unknown',
                        'warnings': [],
                        'errors': []
                    }
                    
                    if not file_name:
                        detail['status'] = 'failed'
                        detail['errors'].append('Package file missing in bundle_info')
                        import_results['details'].append(detail)
                        import_results['failed_models'] += 1
                        continue
                    
                    package_file = temp_dir / file_name
                    try:
                        # Create proper target directory: models/model_type/symbol/
                        target_dir = self.models_dir / model_type / symbol
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        self.logger.info(f"Importing {model_type}/{symbol} to {target_dir}")
                        res = packager.import_model(str(package_file), target_dir=str(target_dir))
                        # res is typically a dict mapping package name to destination path
                        if isinstance(res, dict) and res:
                            dest = list(res.values())[0]
                            detail['destination'] = dest
                        detail['status'] = 'success'
                        import_results['imported_models'] += 1
                    except Exception as e:
                        detail['status'] = 'failed'
                        detail['errors'].append(f"Import error: {e}")
                        import_results['failed_models'] += 1
                    
                    import_results['details'].append(detail)
                
                # Validate imported models if requested
                if validate_after_import and import_results['imported_models'] > 0:
                    self.logger.info("Validating imported models...")
                    validation_results = self._validate_imported_models()
                    import_results['validation'] = validation_results
                
                self._log_import_summary(import_results)
                return import_results
            
            else:
                raise FileNotFoundError("Transfer manifest not found in bundle and no bundle_info.json present")
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            self.logger.info(f"Found {len(manifest['models'])} models in bundle")
            
            # Backup existing models if requested
            if backup_existing:
                self._backup_existing_models()
            
            # Import each model
            import_results = {
                'total_models': len(manifest['models']),
                'imported_models': 0,
                'failed_models': 0,
                'warnings': [],
                'errors': [],
                'details': []
            }
            
            for model_info in manifest['models']:
                try:
                    result = self._import_single_model(temp_dir, model_info)
                    import_results['details'].append(result)
                    
                    if result['status'] == 'success':
                        import_results['imported_models'] += 1
                    else:
                        import_results['failed_models'] += 1
                        
                    if result.get('warnings'):
                        import_results['warnings'].extend(result['warnings'])
                        
                except Exception as e:
                    error_msg = f"Failed to import {model_info.get('model_type', 'unknown')}/{model_info.get('symbol', 'unknown')}: {str(e)}"
                    self.logger.error(error_msg)
                    import_results['errors'].append(error_msg)
                    import_results['failed_models'] += 1
            
            # Validate imported models if requested
            if validate_after_import and import_results['imported_models'] > 0:
                self.logger.info("Validating imported models...")
                validation_results = self._validate_imported_models()
                import_results['validation'] = validation_results
            
            self._log_import_summary(import_results)
            return import_results
            
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _backup_existing_models(self) -> None:
        """
        Create a backup of existing models.
        """
        if not self.models_dir.exists() or not any(self.models_dir.iterdir()):
            self.logger.info("No existing models to backup")
            return
        
        backup_dir = Path(f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.logger.info(f"Creating backup of existing models: {backup_dir}")
        shutil.copytree(self.models_dir, backup_dir)
        
        self.logger.info(f"Backup created successfully: {backup_dir}")
    
    def _import_single_model(self, temp_dir: Path, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import a single model from the extracted bundle.
        
        Args:
            temp_dir: Temporary extraction directory
            model_info: Model information from manifest
            
        Returns:
            Dictionary with import results
        """
        result = {
            'model_type': model_info.get('model_type', 'unknown'),
            'symbol': model_info.get('symbol', 'unknown'),
            'status': 'unknown',
            'warnings': [],
            'errors': []
        }
        
        model_type = model_info['model_type']
        symbol = model_info['symbol']
        
        self.logger.info(f"Importing {model_type} model for {symbol}...")
        
        try:
            # Find source model directory in temp
            source_path = temp_dir / 'models' / model_type / symbol
            if not source_path.exists():
                result['status'] = 'failed'
                result['errors'].append(f"Source model directory not found: {source_path}")
                return result
            
            # Create destination directory
            dest_dir = self.models_dir / model_type / symbol
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for this import
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            timestamped_dest = dest_dir / timestamp
            
            # Copy model files
            self.logger.debug(f"Copying {source_path} to {timestamped_dest}")
            shutil.copytree(source_path, timestamped_dest)
            
            # Update 'latest' pointer
            latest_path = dest_dir / 'latest'
            if latest_path.exists():
                if latest_path.is_symlink():
                    latest_path.unlink()
                elif latest_path.is_dir():
                    shutil.rmtree(latest_path)
            
            # Create symlink or pointer file
            try:
                latest_path.symlink_to(timestamped_dest, target_is_directory=True)
            except (OSError, NotImplementedError):
                # Fallback to pointer file on systems that don't support symlinks
                with open(dest_dir / 'latest_pointer.txt', 'w') as f:
                    f.write(str(timestamped_dest))
            
            # Verify import
            if not timestamped_dest.exists():
                result['status'] = 'failed'
                result['errors'].append("Model files were not copied successfully")
                return result
            
            # Check for required files
            required_files = self._get_required_files(model_type)
            missing_files = []
            
            for file_name in required_files:
                if not (timestamped_dest / file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                result['warnings'].append(f"Missing files after import: {missing_files}")
            
            result['status'] = 'success'
            result['destination'] = str(timestamped_dest)
            
            self.logger.info(f"Successfully imported {model_type}/{symbol} to {timestamped_dest}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(f"Import error: {str(e)}")
        
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
    
    def _validate_imported_models(self) -> Dict[str, Any]:
        """
        Validate imported models using the validation script.
        
        Returns:
            Dictionary with validation results
        """
        try:
            # Import and run validator
            from scripts.validate_models import ModelValidator
            
            validator = ModelValidator(str(self.models_dir))
            return validator.validate_all_models()
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return {
                'error': str(e),
                'total_models': 0,
                'valid_models': 0,
                'invalid_models': 0
            }
    
    def _log_import_summary(self, results: Dict[str, Any]) -> None:
        """
        Log import summary.
        
        Args:
            results: Import results dictionary
        """
        self.logger.info("\n" + "="*50)
        self.logger.info("MODEL IMPORT SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total models in bundle: {results['total_models']}")
        self.logger.info(f"Successfully imported: {results['imported_models']}")
        self.logger.info(f"Failed imports: {results['failed_models']}")
        
        if results['warnings']:
            self.logger.warning(f"Warnings: {len(results['warnings'])}")
            for warning in results['warnings']:
                self.logger.warning(f"  - {warning}")
        
        if results['errors']:
            self.logger.error("Errors encountered:")
            for error in results['errors']:
                self.logger.error(f"  - {error}")
        
        # Log details for each model
        self.logger.info("\nImport Details:")
        for detail in results['details']:
            status_icon = "✅" if detail['status'] == 'success' else "❌"
            self.logger.info(f"  {status_icon} {detail['model_type']}/{detail['symbol']} - {detail['status']}")
            
            if detail.get('warnings'):
                for warning in detail['warnings']:
                    self.logger.warning(f"    ⚠️  {warning}")
            
            if detail.get('errors'):
                for error in detail['errors']:
                    self.logger.error(f"    ❌ {error}")
        
        # Log validation results if available
        if 'validation' in results:
            validation = results['validation']
            self.logger.info("\nValidation Results:")
            self.logger.info(f"  Valid models: {validation.get('valid_models', 0)}")
            self.logger.info(f"  Invalid models: {validation.get('invalid_models', 0)}")
            
            if validation.get('errors'):
                self.logger.error("  Validation errors:")
                for error in validation['errors']:
                    self.logger.error(f"    - {error}")
        
        self.logger.info("="*50)


def main():
    """
    Main function to run model import.
    """
    parser = argparse.ArgumentParser(description='Import models from transfer bundle')
    parser.add_argument('bundle_path', help='Path to the transfer bundle ZIP file')
    parser.add_argument('--models-dir', default='./models', help='Models directory (default: ./models)')
    parser.add_argument('--no-backup', action='store_true', help='Skip backing up existing models')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation after import')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Ensure logs directory exists before attaching FileHandler
    os.makedirs('logs', exist_ok=True)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/model_import.log', mode='a')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting model import from: {args.bundle_path}")
    
    try:
        # Run import
        importer = ModelImporter(args.models_dir)
        results = importer.import_from_bundle(
            args.bundle_path,
            backup_existing=not args.no_backup,
            validate_after_import=not args.no_validate
        )
        
        # Exit with appropriate code
        if results['failed_models'] > 0 or results['errors']:
            logger.error("Import completed with errors")
            sys.exit(1)
        elif results['warnings']:
            logger.warning("Import completed with warnings")
            sys.exit(0)
        else:
            logger.info("All models imported successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Import failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()