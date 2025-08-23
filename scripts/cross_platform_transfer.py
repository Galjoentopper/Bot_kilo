#!/usr/bin/env python3
"""
Cross-Platform Model Transfer Script
===================================

This script handles model transfer between Linux training and Windows trading environments.
It creates platform-independent transfer packages and handles the import process.

Usage:
    # Create transfer package (typically on Linux training machine)
    python cross_platform_transfer.py create --source ./models --output ./transfer_package.zip
    
    # Import transfer package (typically on Windows trading machine)
    python cross_platform_transfer.py import --package ./transfer_package.zip --destination ./models
    
    # Validate transfer package
    python cross_platform_transfer.py validate --package ./transfer_package.zip
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
import platform
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.utils.logger import setup_logging
except ImportError:
    # Fallback logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


class CrossPlatformTransfer:
    """
    Handles cross-platform model transfer between Linux and Windows.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_platform = platform.system().lower()
        
    def create_transfer_package(self, source_dir: str, output_path: str, 
                              include_metadata: bool = True) -> Dict[str, Any]:
        """
        Create a cross-platform transfer package.
        
        Args:
            source_dir: Source models directory
            output_path: Output ZIP file path
            include_metadata: Whether to include metadata files
            
        Returns:
            Dictionary with creation results
        """
        source_path = Path(source_dir)
        output_path = Path(output_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_path}")
        
        self.logger.info(f"Creating transfer package from {source_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect models and metadata
        models_info = self._scan_models_directory(source_path)
        
        if not models_info['models']:
            raise ValueError("No models found in source directory")
        
        # Create transfer manifest
        manifest = {
            'created_at': datetime.now().isoformat(),
            'source_platform': self.current_platform,
            'source_path': str(source_path.absolute()),
            'total_models': len(models_info['models']),
            'models': models_info['models'],
            'metadata': models_info['metadata'],
            'transfer_format_version': '1.0'
        }
        
        # Create ZIP package
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add manifest
            zipf.writestr('transfer_manifest.json', json.dumps(manifest, indent=2))
            
            # Add models
            for model_info in models_info['models']:
                model_path = Path(model_info['full_path'])
                if model_path.exists():
                    # Add all files in model directory
                    if model_path.is_dir():
                        for file_path in model_path.rglob('*'):
                            if file_path.is_file():
                                # Create archive path relative to models root
                                archive_path = 'models' / file_path.relative_to(source_path)
                                zipf.write(file_path, archive_path)
                    else:
                        # Single file
                        archive_path = 'models' / model_path.relative_to(source_path)
                        zipf.write(model_path, archive_path)
            
            # Add metadata files if requested
            if include_metadata and models_info['metadata']:
                for metadata_file in models_info['metadata']:
                    metadata_path = Path(metadata_file['full_path'])
                    if metadata_path.exists():
                        archive_path = 'metadata' / metadata_path.name
                        zipf.write(metadata_path, archive_path)
        
        result = {
            'package_path': str(output_path.absolute()),
            'package_size_mb': output_path.stat().st_size / (1024 * 1024),
            'total_models': len(models_info['models']),
            'created_at': manifest['created_at'],
            'source_platform': self.current_platform
        }
        
        self.logger.info(f"Transfer package created: {output_path}")
        self.logger.info(f"Package size: {result['package_size_mb']:.2f} MB")
        self.logger.info(f"Total models: {result['total_models']}")
        
        return result
    
    def import_transfer_package(self, package_path: str, destination_dir: str,
                              backup_existing: bool = True) -> Dict[str, Any]:
        """
        Import a transfer package to the destination directory.
        
        Args:
            package_path: Path to transfer package ZIP file
            destination_dir: Destination models directory
            backup_existing: Whether to backup existing models
            
        Returns:
            Dictionary with import results
        """
        package_path = Path(package_path)
        dest_path = Path(destination_dir)
        
        if not package_path.exists():
            raise FileNotFoundError(f"Transfer package not found: {package_path}")
        
        self.logger.info(f"Importing transfer package: {package_path}")
        
        # Create temporary extraction directory
        temp_dir = Path(f"temp_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract package
            with zipfile.ZipFile(package_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Load manifest
            manifest_path = temp_dir / 'transfer_manifest.json'
            if not manifest_path.exists():
                raise FileNotFoundError("Transfer manifest not found in package")
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            self.logger.info(f"Package created on {manifest['source_platform']} at {manifest['created_at']}")
            self.logger.info(f"Found {manifest['total_models']} models")
            
            # Backup existing models if requested
            if backup_existing and dest_path.exists():
                self._backup_existing_models(dest_path)
            
            # Ensure destination exists
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Import models
            models_dir = temp_dir / 'models'
            if models_dir.exists():
                # Copy models with platform-specific path handling
                self._copy_models_cross_platform(models_dir, dest_path)
            
            # Import metadata
            metadata_dir = temp_dir / 'metadata'
            if metadata_dir.exists():
                metadata_dest = dest_path / 'metadata'
                metadata_dest.mkdir(exist_ok=True)
                for metadata_file in metadata_dir.iterdir():
                    if metadata_file.is_file():
                        shutil.copy2(metadata_file, metadata_dest / metadata_file.name)
            
            result = {
                'imported_models': manifest['total_models'],
                'source_platform': manifest['source_platform'],
                'destination_platform': self.current_platform,
                'imported_at': datetime.now().isoformat(),
                'destination_path': str(dest_path.absolute())
            }
            
            self.logger.info(f"Successfully imported {result['imported_models']} models")
            return result
            
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def validate_transfer_package(self, package_path: str) -> Dict[str, Any]:
        """
        Validate a transfer package without importing it.
        
        Args:
            package_path: Path to transfer package ZIP file
            
        Returns:
            Dictionary with validation results
        """
        package_path = Path(package_path)
        
        if not package_path.exists():
            raise FileNotFoundError(f"Transfer package not found: {package_path}")
        
        self.logger.info(f"Validating transfer package: {package_path}")
        
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'package_info': {}
        }
        
        try:
            with zipfile.ZipFile(package_path, 'r') as zipf:
                # Check if manifest exists
                if 'transfer_manifest.json' not in zipf.namelist():
                    validation_result['errors'].append("Transfer manifest not found")
                    return validation_result
                
                # Read and validate manifest
                manifest_data = zipf.read('transfer_manifest.json')
                manifest = json.loads(manifest_data)
                
                validation_result['package_info'] = {
                    'created_at': manifest.get('created_at'),
                    'source_platform': manifest.get('source_platform'),
                    'total_models': manifest.get('total_models', 0),
                    'format_version': manifest.get('transfer_format_version')
                }
                
                # Check format version
                if manifest.get('transfer_format_version') != '1.0':
                    validation_result['warnings'].append(
                        f"Unknown format version: {manifest.get('transfer_format_version')}"
                    )
                
                # Check if models directory exists in archive
                models_files = [f for f in zipf.namelist() if f.startswith('models/')]
                if not models_files:
                    validation_result['errors'].append("No models found in package")
                    return validation_result
                
                # Platform compatibility check
                source_platform = manifest.get('source_platform', 'unknown')
                if source_platform != self.current_platform:
                    validation_result['warnings'].append(
                        f"Package created on {source_platform}, importing to {self.current_platform}"
                    )
                
                validation_result['valid'] = len(validation_result['errors']) == 0
                
        except zipfile.BadZipFile:
            validation_result['errors'].append("Invalid ZIP file")
        except json.JSONDecodeError:
            validation_result['errors'].append("Invalid manifest JSON")
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        if validation_result['valid']:
            self.logger.info("Package validation passed")
        else:
            self.logger.error(f"Package validation failed: {validation_result['errors']}")
        
        return validation_result
    
    def _scan_models_directory(self, models_dir: Path) -> Dict[str, Any]:
        """
        Scan models directory and collect information.
        """
        models_info = {
            'models': [],
            'metadata': []
        }
        
        if not models_dir.exists():
            return models_info
        
        # Scan for model directories
        for model_type_dir in models_dir.iterdir():
            if model_type_dir.is_dir() and model_type_dir.name != 'metadata':
                model_type = model_type_dir.name
                
                for symbol_dir in model_type_dir.iterdir():
                    if symbol_dir.is_dir():
                        symbol = symbol_dir.name
                        
                        # Find latest model
                        latest_model = self._find_latest_model(symbol_dir)
                        if latest_model:
                            models_info['models'].append({
                                'model_type': model_type,
                                'symbol': symbol,
                                'path': str(latest_model.relative_to(models_dir)),
                                'full_path': str(latest_model),
                                'size_bytes': self._get_directory_size(latest_model)
                            })
        
        # Scan for metadata files
        metadata_dir = models_dir / 'metadata'
        if metadata_dir.exists():
            for metadata_file in metadata_dir.iterdir():
                if metadata_file.is_file():
                    models_info['metadata'].append({
                        'name': metadata_file.name,
                        'path': str(metadata_file.relative_to(models_dir)),
                        'full_path': str(metadata_file),
                        'size_bytes': metadata_file.stat().st_size
                    })
        
        return models_info
    
    def _find_latest_model(self, symbol_dir: Path) -> Optional[Path]:
        """
        Find the latest model in a symbol directory.
        """
        # Look for 'latest' symlink or directory
        latest_path = symbol_dir / 'latest'
        if latest_path.exists():
            return latest_path
        
        # Look for timestamped directories
        timestamped_dirs = [d for d in symbol_dir.iterdir() 
                           if d.is_dir() and d.name.replace('_', '').isdigit()]
        
        if timestamped_dirs:
            return max(timestamped_dirs, key=lambda x: x.name)
        
        # Return the symbol directory itself if it contains model files
        model_files = list(symbol_dir.glob('*.pkl')) + list(symbol_dir.glob('*.pt')) + list(symbol_dir.glob('*.joblib'))
        if model_files:
            return symbol_dir
        
        return None
    
    def _get_directory_size(self, directory: Path) -> int:
        """
        Get total size of directory in bytes.
        """
        total_size = 0
        if directory.is_file():
            return directory.stat().st_size
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _backup_existing_models(self, models_dir: Path) -> None:
        """
        Create backup of existing models.
        """
        if not models_dir.exists() or not any(models_dir.iterdir()):
            return
        
        backup_dir = models_dir.parent / f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"Creating backup: {backup_dir}")
        shutil.copytree(models_dir, backup_dir)
    
    def _copy_models_cross_platform(self, source_dir: Path, dest_dir: Path) -> None:
        """
        Copy models with cross-platform path handling.
        """
        for source_file in source_dir.rglob('*'):
            if source_file.is_file():
                # Calculate relative path and convert to destination platform format
                rel_path = source_file.relative_to(source_dir)
                dest_file = dest_dir / rel_path
                
                # Ensure destination directory exists
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_file, dest_file)


def main():
    parser = argparse.ArgumentParser(description='Cross-platform model transfer')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create transfer package')
    create_parser.add_argument('--source', required=True, help='Source models directory')
    create_parser.add_argument('--output', required=True, help='Output package path')
    create_parser.add_argument('--no-metadata', action='store_true', help='Exclude metadata files')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import transfer package')
    import_parser.add_argument('--package', required=True, help='Transfer package path')
    import_parser.add_argument('--destination', required=True, help='Destination directory')
    import_parser.add_argument('--no-backup', action='store_true', help='Skip backup of existing models')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate transfer package')
    validate_parser.add_argument('--package', required=True, help='Transfer package path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    transfer = CrossPlatformTransfer()
    
    try:
        if args.command == 'create':
            result = transfer.create_transfer_package(
                args.source, 
                args.output, 
                include_metadata=not args.no_metadata
            )
            print(f"\nTransfer package created successfully!")
            print(f"Package: {result['package_path']}")
            print(f"Size: {result['package_size_mb']:.2f} MB")
            print(f"Models: {result['total_models']}")
            
        elif args.command == 'import':
            result = transfer.import_transfer_package(
                args.package,
                args.destination,
                backup_existing=not args.no_backup
            )
            print(f"\nModels imported successfully!")
            print(f"Imported: {result['imported_models']} models")
            print(f"From: {result['source_platform']} to {result['destination_platform']}")
            print(f"Destination: {result['destination_path']}")
            
        elif args.command == 'validate':
            result = transfer.validate_transfer_package(args.package)
            print(f"\nValidation Results:")
            print(f"Valid: {result['valid']}")
            
            if result['package_info']:
                info = result['package_info']
                print(f"Created: {info.get('created_at', 'Unknown')}")
                print(f"Source Platform: {info.get('source_platform', 'Unknown')}")
                print(f"Models: {info.get('total_models', 0)}")
            
            if result['warnings']:
                print("\nWarnings:")
                for warning in result['warnings']:
                    print(f"  - {warning}")
            
            if result['errors']:
                print("\nErrors:")
                for error in result['errors']:
                    print(f"  - {error}")
                return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())