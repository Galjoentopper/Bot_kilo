#!/usr/bin/env python3
"""
Model Transfer Utilities for Distributed Trading Bot

This module provides utilities for transferring models between machines,
including network transfer, USB/external drive transfer, and cloud storage integration.
"""

import os
import json
import shutil
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import hashlib
import tempfile

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .model_packaging import ModelPackager


class ModelTransferManager:
    """Manages model transfers between different machines and storage locations"""
    
    def __init__(self, base_dir: str = "models", transfer_dir: str = "model_transfers"):
        self.base_dir = Path(base_dir)
        self.transfer_dir = Path(transfer_dir)
        self.transfer_dir.mkdir(exist_ok=True)
        
        # Create transfer subdirectories
        (self.transfer_dir / "outgoing").mkdir(exist_ok=True)
        (self.transfer_dir / "incoming").mkdir(exist_ok=True)
        (self.transfer_dir / "staging").mkdir(exist_ok=True)
        
        self.packager = ModelPackager(base_dir=str(self.base_dir))
        self.logger = logging.getLogger(__name__)
    
    def prepare_for_transfer(self, 
                           model_types: List[str], 
                           symbols: List[str],
                           transfer_method: str = "usb",
                           include_backups: bool = False) -> Dict[str, Any]:
        """Prepare models for transfer to another machine"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transfer_name = f"model_transfer_{timestamp}"
        
        # Create transfer directory
        transfer_path = self.transfer_dir / "outgoing" / transfer_name
        transfer_path.mkdir(parents=True, exist_ok=True)
        
        # Package models
        packaged_models = []
        for model_type in model_types:
            for symbol in symbols:
                try:
                    # Find latest model for this type/symbol
                    model_files = self._find_model_files(model_type, symbol)
                    if model_files:
                        package_path = self._package_model_for_transfer(
                            model_files, model_type, symbol, str(transfer_path)
                        )
                        if package_path:
                            packaged_models.append({
                                "type": model_type,
                                "symbol": symbol,
                                "package": package_path,
                                "size_mb": round(Path(package_path).stat().st_size / (1024*1024), 2)
                            })
                except Exception as e:
                    self.logger.warning(f"Could not package {model_type}/{symbol}: {e}")
        
        if not packaged_models:
            raise ValueError("No models found to transfer")
        
        # Create transfer manifest
        manifest = {
            "transfer_id": transfer_name,
            "created_at": datetime.now().isoformat(),
            "transfer_method": transfer_method,
            "source_machine": os.environ.get("COMPUTERNAME", "unknown"),
            "models": packaged_models,
            "total_size_mb": sum(m["size_mb"] for m in packaged_models),
            "instructions": self._get_transfer_instructions(transfer_method)
        }
        
        # Save manifest
        manifest_path = transfer_path / "transfer_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create transfer scripts
        self._create_transfer_scripts(transfer_path, transfer_method)
        
        # Create README
        readme_path = transfer_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(self._generate_transfer_readme(manifest))
        
        self.logger.info(f"Transfer package prepared: {transfer_path}")
        return {
            "transfer_path": str(transfer_path),
            "manifest": manifest,
            "ready_for_transfer": True
        }
    
    def _find_model_files(self, model_type: str, symbol: str) -> Optional[Dict[str, str]]:
        """Find the latest model files for a given type and symbol"""
        model_files = {}
        
        # Look in various locations
        search_paths = [
            self.base_dir / "metadata",
            self.base_dir / model_type / symbol,
            Path("models") / "metadata",
            Path("models") / model_type / symbol
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            # Look for model files
            if model_type == "gru":
                model_pattern = f"*{symbol}*.pt"
            elif model_type == "lightgbm":
                model_pattern = f"*{symbol}*.pkl"
            elif model_type == "ppo":
                model_pattern = f"*{symbol}*.zip"  # PPO models are often zipped
            else:
                model_pattern = f"*{symbol}*"
            
            model_files_found = list(search_path.glob(model_pattern))
            if model_files_found:
                # Get the most recent file
                latest_model = max(model_files_found, key=lambda x: x.stat().st_mtime)
                model_files["model"] = str(latest_model)
                
                # Look for preprocessor
                preprocessor_files = list(search_path.glob(f"*{symbol}*preprocessor*.pkl"))
                if preprocessor_files:
                    latest_preprocessor = max(preprocessor_files, key=lambda x: x.stat().st_mtime)
                    model_files["preprocessor"] = str(latest_preprocessor)
                
                # Look for metadata
                metadata_files = list(search_path.glob(f"*{symbol}*metadata*.json"))
                if metadata_files:
                    latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
                    model_files["metadata"] = str(latest_metadata)
                
                break
        
        return model_files if model_files else None
    
    def _package_model_for_transfer(self, model_files: Dict[str, str], 
                                  model_type: str, symbol: str, 
                                  output_dir: str) -> Optional[str]:
        """Package a model with its files for transfer"""
        try:
            # Load existing metadata if available
            training_config = {}
            performance_metrics = {}
            feature_columns = []
            
            if "metadata" in model_files:
                with open(model_files["metadata"], 'r') as f:
                    existing_metadata = json.load(f)
                    training_config = existing_metadata.get("training_config", {})
                    performance_metrics = existing_metadata.get("performance_metrics", {})
                    feature_columns = existing_metadata.get("feature_columns", [])
            
            # Package the model
            package_path = self.packager.package_model(
                model_path=model_files["model"],
                model_type=model_type,
                symbol=symbol,
                preprocessor_path=model_files.get("preprocessor"),
                feature_columns=feature_columns,
                performance_metrics=performance_metrics,
                training_config=training_config,
                notes=f"Packaged for transfer on {datetime.now().isoformat()}"
            )
            
            # Move to transfer directory
            package_name = Path(package_path).name
            transfer_package_path = Path(output_dir) / package_name
            shutil.move(package_path, transfer_package_path)
            
            return str(transfer_package_path)
            
        except Exception as e:
            self.logger.error(f"Error packaging model {model_type}/{symbol}: {e}")
            return None
    
    def _get_transfer_instructions(self, method: str) -> Dict[str, str]:
        """Get transfer instructions for different methods"""
        instructions = {
            "usb": {
                "step1": "Copy the entire transfer folder to a USB drive or external storage",
                "step2": "On the target machine, copy the folder to the trading bot directory",
                "step3": "Run the import_models.py script in the transfer folder",
                "step4": "Verify models are loaded correctly using the validation script"
            },
            "network": {
                "step1": "Use the network_transfer.py script to send models over network",
                "step2": "On target machine, run the receive_models.py script",
                "step3": "Models will be automatically imported and validated",
                "step4": "Check the import log for any issues"
            },
            "cloud": {
                "step1": "Upload the transfer folder to your cloud storage (Google Drive, Dropbox, etc.)",
                "step2": "On target machine, download the folder",
                "step3": "Run the import_models.py script",
                "step4": "Clean up temporary files after successful import"
            }
        }
        return instructions.get(method, instructions["usb"])
    
    def _create_transfer_scripts(self, transfer_path: Path, method: str):
        """Create helper scripts for the transfer"""
        
        # Import script
        import_script = f'''#!/usr/bin/env python3
"""
Model Import Script for Transfer
Run this script on the target machine to import transferred models
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.model_transfer import ModelTransferManager
except ImportError:
    print("Error: Could not import ModelTransferManager")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

def main():
    print("Starting model import...")
    
    # Load manifest
    with open("transfer_manifest.json", "r") as f:
        manifest = json.load(f)
    
    transfer_manager = ModelTransferManager()
    
    print(f"Importing {{len(manifest['models'])}} models...")
    
    success_count = 0
    for model_info in manifest["models"]:
        print(f"Importing {{model_info['type']}}/{{model_info['symbol']}}...")
        try:
            result = transfer_manager.import_transferred_model(model_info["package"])
            if result["success"]:
                print(f"  ✓ Success: {{result['message']}}")
                success_count += 1
            else:
                print(f"  ✗ Failed: {{result['message']}}")
        except Exception as e:
            print(f"  ✗ Error: {{e}}")
    
    print(f"\nImport complete: {{success_count}}/{{len(manifest['models'])}} models imported successfully")
    
    if success_count == len(manifest["models"]):
        print("All models imported successfully! You can now run the trading bot.")
    else:
        print("Some models failed to import. Check the error messages above.")

if __name__ == "__main__":
    main()
'''
        
        with open(transfer_path / "import_models.py", 'w') as f:
            f.write(import_script)
        
        # Validation script
        validation_script = '''#!/usr/bin/env python3
"""
Model Validation Script
Validate that imported models are working correctly
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.model_transfer import ModelTransferManager
except ImportError:
    print("Error: Could not import ModelTransferManager")
    sys.exit(1)

def main():
    print("Validating imported models...")
    
    transfer_manager = ModelTransferManager()
    validation_results = transfer_manager.validate_imported_models()
    
    print(f"\nValidation Results:")
    for result in validation_results:
        status = "✓" if result["valid"] else "✗"
        print(f"{status} {result['model_type']}/{result['symbol']}: {result['message']}")
    
    valid_count = sum(1 for r in validation_results if r["valid"])
    total_count = len(validation_results)
    
    print(f"\nSummary: {valid_count}/{total_count} models are valid")
    
    if valid_count == total_count:
        print("All models are ready for trading!")
    else:
        print("Some models have issues. Please check the validation messages.")

if __name__ == "__main__":
    main()
'''
        
        with open(transfer_path / "validate_models.py", 'w') as f:
            f.write(validation_script)
    
    def _generate_transfer_readme(self, manifest: Dict[str, Any]) -> str:
        """Generate README for the transfer package"""
        readme = f"""# Model Transfer Package

**Transfer ID**: {manifest['transfer_id']}  
**Created**: {manifest['created_at']}  
**Source Machine**: {manifest['source_machine']}  
**Transfer Method**: {manifest['transfer_method']}  
**Total Size**: {manifest['total_size_mb']:.2f} MB

## Models Included

| Type | Symbol | Size (MB) |
|------|--------|----------|
"""
        
        for model in manifest['models']:
            readme += f"| {model['type']} | {model['symbol']} | {model['size_mb']:.2f} |\n"
        
        readme += f"""

## Transfer Instructions

### Method: {manifest['transfer_method'].upper()}

"""
        
        for step, instruction in manifest['instructions'].items():
            readme += f"**{step.upper()}**: {instruction}\n\n"
        
        readme += """
## Quick Start

1. Copy this entire folder to your target machine
2. Navigate to this folder in terminal/command prompt
3. Run: `python import_models.py`
4. Run: `python validate_models.py` to verify everything works
5. Start your trading bot!

## Files in this Package

- `transfer_manifest.json` - Contains metadata about this transfer
- `import_models.py` - Script to import all models
- `validate_models.py` - Script to validate imported models
- `*.zip` - Individual model packages
- `README.md` - This file

## Troubleshooting

If you encounter issues:

1. Make sure you're in the correct directory
2. Check that Python can find the trading bot modules
3. Verify all dependencies are installed
4. Check the error messages in the import script

For more help, check the main project documentation.
"""
        
        return readme
    
    def import_transferred_model(self, package_path: str) -> Dict[str, Any]:
        """Import a single transferred model package"""
        try:
            # Validate package first
            is_valid, issues = self.packager.validate_package(package_path)
            if not is_valid:
                return {
                    "success": False,
                    "message": f"Package validation failed: {'; '.join(issues)}"
                }
            
            # Import the model
            result = self.packager.import_model(package_path)
            
            return {
                "success": True,
                "message": f"Model imported successfully",
                "files": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Import failed: {str(e)}"
            }
    
    def validate_imported_models(self) -> List[Dict[str, Any]]:
        """Validate all imported models"""
        results = []
        
        # Check common model locations
        model_dirs = [
            self.base_dir / "gru",
            self.base_dir / "lightgbm", 
            self.base_dir / "ppo",
            self.base_dir / "metadata"
        ]
        
        for model_dir in model_dirs:
            if not model_dir.exists():
                continue
                
            for symbol_dir in model_dir.iterdir():
                if symbol_dir.is_dir():
                    # This is a symbol directory
                    model_type = model_dir.name
                    symbol = symbol_dir.name
                    
                    result = self._validate_single_model(model_type, symbol, symbol_dir)
                    results.append(result)
        
        return results
    
    def _validate_single_model(self, model_type: str, symbol: str, model_dir: Path) -> Dict[str, Any]:
        """Validate a single model"""
        try:
            # Check for required files
            model_files = list(model_dir.glob("*"))
            if not model_files:
                return {
                    "model_type": model_type,
                    "symbol": symbol,
                    "valid": False,
                    "message": "No model files found"
                }
            
            # Check for model file
            model_extensions = {
                "gru": [".pt", ".pth"],
                "lightgbm": [".pkl"],
                "ppo": [".zip", ".pkl"]
            }
            
            expected_extensions = model_extensions.get(model_type, [".pkl", ".pt"])
            model_file_found = any(
                any(f.suffix == ext for ext in expected_extensions) 
                for f in model_files
            )
            
            if not model_file_found:
                return {
                    "model_type": model_type,
                    "symbol": symbol,
                    "valid": False,
                    "message": f"No model file with expected extension {expected_extensions}"
                }
            
            # Check metadata
            metadata_files = [f for f in model_files if "metadata" in f.name.lower()]
            if not metadata_files:
                return {
                    "model_type": model_type,
                    "symbol": symbol,
                    "valid": True,
                    "message": "Model file found (no metadata)"
                }
            
            return {
                "model_type": model_type,
                "symbol": symbol,
                "valid": True,
                "message": "Model and metadata found"
            }
            
        except Exception as e:
            return {
                "model_type": model_type,
                "symbol": symbol,
                "valid": False,
                "message": f"Validation error: {str(e)}"
            }
    
    def cleanup_transfer(self, transfer_id: str, keep_packages: bool = True):
        """Clean up transfer files after successful import"""
        transfer_path = self.transfer_dir / "outgoing" / transfer_id
        
        if transfer_path.exists():
            if keep_packages:
                # Move to archive
                archive_dir = self.transfer_dir / "archive"
                archive_dir.mkdir(exist_ok=True)
                shutil.move(str(transfer_path), str(archive_dir / transfer_id))
                self.logger.info(f"Transfer archived: {transfer_id}")
            else:
                # Delete completely
                shutil.rmtree(transfer_path)
                self.logger.info(f"Transfer cleaned up: {transfer_id}")
    
    def list_transfers(self) -> List[Dict[str, Any]]:
        """List available transfers"""
        transfers = []
        
        for transfer_dir in (self.transfer_dir / "outgoing").iterdir():
            if transfer_dir.is_dir():
                manifest_path = transfer_dir / "transfer_manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    
                    transfers.append({
                        "id": manifest["transfer_id"],
                        "created_at": manifest["created_at"],
                        "models_count": len(manifest["models"]),
                        "total_size_mb": manifest["total_size_mb"],
                        "path": str(transfer_dir)
                    })
        
        return sorted(transfers, key=lambda x: x["created_at"], reverse=True)