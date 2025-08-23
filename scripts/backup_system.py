#!/usr/bin/env python3
"""
Automated Backup and Versioning System for Trading Bot

This script provides comprehensive backup functionality for:
- Trained models and metadata
- Configuration files
- Trading logs and performance data
- Model packages and transfers

Features:
- Automated daily/weekly backups
- Version control with timestamps
- Compression and space optimization
- Backup verification and integrity checks
- Easy restore functionality
- Cloud storage integration (optional)
"""

import os
import sys
import shutil
import tarfile
import zipfile
import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class BackupSystem:
    """
    Comprehensive backup and versioning system for the trading bot.
    """
    
    def __init__(self, backup_dir: str = 'backups'):
        """
        Initialize the backup system.
        
        Args:
            backup_dir: Directory to store backups
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Backup configuration
        self.backup_config = {
            'models': {
                'source': 'models',
                'include_patterns': ['*.pkl', '*.pt', '*.zip', '*.json'],
                'exclude_patterns': ['*.tmp', '*.log'],
                'compress': True
            },
            'config': {
                'source': 'config',
                'include_patterns': ['*.yaml', '*.yml', '*.json'],
                'exclude_patterns': [],
                'compress': False
            },
            'logs': {
                'source': 'logs',
                'include_patterns': ['*.log'],
                'exclude_patterns': ['*.tmp'],
                'compress': True,
                'max_age_days': 30
            },
            'exports': {
                'source': 'exports',
                'include_patterns': ['*.tar.gz', '*.zip'],
                'exclude_patterns': [],
                'compress': False
            },
            'scripts': {
                'source': 'scripts',
                'include_patterns': ['*.py', '*.bat'],
                'exclude_patterns': ['__pycache__', '*.pyc'],
                'compress': True
            }
        }
        
        # Retention policy
        self.retention_policy = {
            'daily': 7,    # Keep 7 daily backups
            'weekly': 4,   # Keep 4 weekly backups
            'monthly': 12  # Keep 12 monthly backups
        }
    
    def create_backup(self, backup_type: str = 'full', 
                     custom_name: Optional[str] = None) -> str:
        """
        Create a backup of the specified type.
        
        Args:
            backup_type: Type of backup ('full', 'models', 'config', 'logs')
            custom_name: Custom name for the backup
            
        Returns:
            Path to the created backup
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if custom_name:
            backup_name = f"{custom_name}_{timestamp}"
        else:
            backup_name = f"{backup_type}_backup_{timestamp}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating {backup_type} backup: {backup_path}")
        
        # Create backup manifest
        manifest = {
            'backup_type': backup_type,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat(),
            'files': [],
            'checksums': {},
            'total_size': 0
        }
        
        try:
            if backup_type == 'full':
                # Backup all components
                for component in self.backup_config.keys():
                    self._backup_component(component, backup_path, manifest)
            elif backup_type in self.backup_config:
                # Backup specific component
                self._backup_component(backup_type, backup_path, manifest)
            else:
                raise ValueError(f"Unknown backup type: {backup_type}")
            
            # Save manifest
            manifest_path = backup_path / 'backup_manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create compressed archive if requested
            if backup_type == 'full' or self.backup_config.get(backup_type, {}).get('compress', False):
                archive_path = self._create_archive(backup_path)
                shutil.rmtree(backup_path)  # Remove uncompressed version
                return str(archive_path)
            
            self.logger.info(f"Backup created successfully: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    def _backup_component(self, component: str, backup_path: Path, 
                         manifest: Dict) -> None:
        """
        Backup a specific component.
        
        Args:
            component: Component name
            backup_path: Backup destination path
            manifest: Backup manifest to update
        """
        config = self.backup_config[component]
        source_path = Path(config['source'])
        
        if not source_path.exists():
            self.logger.warning(f"Source path does not exist: {source_path}")
            return
        
        component_backup_path = backup_path / component
        component_backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files based on patterns
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                # Check include patterns
                if config['include_patterns']:
                    if not any(file_path.match(pattern) for pattern in config['include_patterns']):
                        continue
                
                # Check exclude patterns
                if any(file_path.match(pattern) for pattern in config['exclude_patterns']):
                    continue
                
                # Check age limit for logs
                if component == 'logs' and 'max_age_days' in config:
                    file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.days > config['max_age_days']:
                        continue
                
                # Copy file
                relative_path = file_path.relative_to(source_path)
                dest_path = component_backup_path / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(file_path, dest_path)
                
                # Update manifest
                file_size = file_path.stat().st_size
                checksum = self._calculate_checksum(file_path)
                
                manifest['files'].append({
                    'component': component,
                    'source': str(file_path),
                    'destination': str(dest_path),
                    'size': file_size,
                    'checksum': checksum
                })
                manifest['checksums'][str(dest_path)] = checksum
                manifest['total_size'] += file_size
        
        self.logger.info(f"Backed up component: {component}")
    
    def _create_archive(self, backup_path: Path) -> Path:
        """
        Create a compressed archive of the backup.
        
        Args:
            backup_path: Path to backup directory
            
        Returns:
            Path to created archive
        """
        archive_path = backup_path.with_suffix('.tar.gz')
        
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(backup_path, arcname=backup_path.name)
        
        self.logger.info(f"Created compressed archive: {archive_path}")
        return archive_path
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 checksum
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def list_backups(self) -> List[Dict]:
        """
        List all available backups.
        
        Returns:
            List of backup information
        """
        backups = []
        
        for item in self.backup_dir.iterdir():
            if item.is_dir():
                manifest_path = item / 'backup_manifest.json'
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    
                    backup_info = {
                        'name': item.name,
                        'path': str(item),
                        'type': manifest.get('backup_type', 'unknown'),
                        'created_at': manifest.get('created_at'),
                        'total_size': manifest.get('total_size', 0),
                        'file_count': len(manifest.get('files', [])),
                        'compressed': False
                    }
                    backups.append(backup_info)
            
            elif item.suffix == '.gz' and item.stem.endswith('.tar'):
                # Compressed backup
                backup_info = {
                    'name': item.stem.replace('.tar', ''),
                    'path': str(item),
                    'type': 'compressed',
                    'created_at': datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                    'total_size': item.stat().st_size,
                    'file_count': 'unknown',
                    'compressed': True
                }
                backups.append(backup_info)
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        return backups
    
    def restore_backup(self, backup_name: str, 
                      restore_path: Optional[str] = None,
                      components: Optional[List[str]] = None) -> bool:
        """
        Restore a backup.
        
        Args:
            backup_name: Name of backup to restore
            restore_path: Path to restore to (default: current directory)
            components: Specific components to restore
            
        Returns:
            True if successful
        """
        backup_path = self.backup_dir / backup_name
        
        # Handle compressed backups
        if not backup_path.exists():
            compressed_path = self.backup_dir / f"{backup_name}.tar.gz"
            if compressed_path.exists():
                self.logger.info(f"Extracting compressed backup: {compressed_path}")
                with tarfile.open(compressed_path, 'r:gz') as tar:
                    tar.extractall(self.backup_dir)
                backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            self.logger.error(f"Backup not found: {backup_name}")
            return False
        
        # Load manifest
        manifest_path = backup_path / 'backup_manifest.json'
        if not manifest_path.exists():
            self.logger.error(f"Backup manifest not found: {manifest_path}")
            return False
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        restore_root = Path(restore_path) if restore_path else Path.cwd()
        
        try:
            self.logger.info(f"Restoring backup: {backup_name}")
            
            for file_info in manifest['files']:
                component = file_info['component']
                
                # Skip if component not requested
                if components and component not in components:
                    continue
                
                source_path = Path(file_info['destination'])
                
                # Calculate restore destination
                relative_path = source_path.relative_to(backup_path / component)
                dest_path = restore_root / component / relative_path
                
                # Create destination directory
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, dest_path)
                
                # Verify checksum
                if self._calculate_checksum(dest_path) != file_info['checksum']:
                    self.logger.warning(f"Checksum mismatch for: {dest_path}")
            
            self.logger.info(f"Backup restored successfully to: {restore_root}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {str(e)}")
            return False
    
    def verify_backup(self, backup_name: str) -> bool:
        """
        Verify the integrity of a backup.
        
        Args:
            backup_name: Name of backup to verify
            
        Returns:
            True if backup is valid
        """
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            self.logger.error(f"Backup not found: {backup_name}")
            return False
        
        manifest_path = backup_path / 'backup_manifest.json'
        if not manifest_path.exists():
            self.logger.error(f"Backup manifest not found")
            return False
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.logger.info(f"Verifying backup: {backup_name}")
        
        errors = 0
        for file_info in manifest['files']:
            file_path = Path(file_info['destination'])
            
            if not file_path.exists():
                self.logger.error(f"Missing file: {file_path}")
                errors += 1
                continue
            
            # Verify checksum
            actual_checksum = self._calculate_checksum(file_path)
            expected_checksum = file_info['checksum']
            
            if actual_checksum != expected_checksum:
                self.logger.error(f"Checksum mismatch: {file_path}")
                errors += 1
        
        if errors == 0:
            self.logger.info(f"Backup verification successful: {backup_name}")
            return True
        else:
            self.logger.error(f"Backup verification failed: {errors} errors found")
            return False
    
    def cleanup_old_backups(self) -> None:
        """
        Clean up old backups according to retention policy.
        """
        backups = self.list_backups()
        
        # Group backups by type and age
        now = datetime.now()
        daily_backups = []
        weekly_backups = []
        monthly_backups = []
        
        for backup in backups:
            created_at = datetime.fromisoformat(backup['created_at'].replace('Z', '+00:00'))
            age_days = (now - created_at).days
            
            if age_days <= 7:
                daily_backups.append(backup)
            elif age_days <= 30:
                weekly_backups.append(backup)
            else:
                monthly_backups.append(backup)
        
        # Apply retention policy
        to_delete = []
        
        if len(daily_backups) > self.retention_policy['daily']:
            to_delete.extend(daily_backups[self.retention_policy['daily']:])
        
        if len(weekly_backups) > self.retention_policy['weekly']:
            to_delete.extend(weekly_backups[self.retention_policy['weekly']:])
        
        if len(monthly_backups) > self.retention_policy['monthly']:
            to_delete.extend(monthly_backups[self.retention_policy['monthly']:])
        
        # Delete old backups
        for backup in to_delete:
            backup_path = Path(backup['path'])
            try:
                if backup_path.is_dir():
                    shutil.rmtree(backup_path)
                else:
                    backup_path.unlink()
                self.logger.info(f"Deleted old backup: {backup['name']}")
            except Exception as e:
                self.logger.error(f"Failed to delete backup {backup['name']}: {str(e)}")
    
    def schedule_backup(self, backup_type: str = 'full', 
                       schedule: str = 'daily') -> None:
        """
        Schedule automatic backups.
        
        Args:
            backup_type: Type of backup to create
            schedule: Schedule frequency ('daily', 'weekly', 'monthly')
        """
        # This would integrate with system scheduler (cron, Task Scheduler, etc.)
        # For now, just create the backup
        self.logger.info(f"Creating scheduled {schedule} backup")
        self.create_backup(backup_type, f"{schedule}_scheduled")
        
        # Clean up old backups
        self.cleanup_old_backups()

def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description='Trading Bot Backup System')
    parser.add_argument('action', choices=['create', 'list', 'restore', 'verify', 'cleanup'],
                       help='Action to perform')
    parser.add_argument('--type', default='full', 
                       choices=['full', 'models', 'config', 'logs', 'exports', 'scripts'],
                       help='Backup type')
    parser.add_argument('--name', help='Backup name (for create/restore/verify)')
    parser.add_argument('--restore-path', help='Path to restore backup to')
    parser.add_argument('--components', nargs='+', help='Specific components to restore')
    parser.add_argument('--backup-dir', default='backups', help='Backup directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/backup_system.log')
        ]
    )
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Initialize backup system
    backup_system = BackupSystem(args.backup_dir)
    
    try:
        if args.action == 'create':
            backup_path = backup_system.create_backup(args.type, args.name)
            print(f"Backup created: {backup_path}")
        
        elif args.action == 'list':
            backups = backup_system.list_backups()
            if backups:
                print("\nAvailable Backups:")
                print("-" * 80)
                for backup in backups:
                    size_mb = backup['total_size'] / (1024 * 1024)
                    compressed = " (compressed)" if backup['compressed'] else ""
                    print(f"{backup['name']:<30} {backup['type']:<10} {size_mb:>8.1f} MB{compressed}")
                    print(f"  Created: {backup['created_at']}")
                    print()
            else:
                print("No backups found.")
        
        elif args.action == 'restore':
            if not args.name:
                print("Error: --name required for restore action")
                sys.exit(1)
            
            success = backup_system.restore_backup(args.name, args.restore_path, args.components)
            if success:
                print(f"Backup restored successfully: {args.name}")
            else:
                print(f"Failed to restore backup: {args.name}")
                sys.exit(1)
        
        elif args.action == 'verify':
            if not args.name:
                print("Error: --name required for verify action")
                sys.exit(1)
            
            success = backup_system.verify_backup(args.name)
            if success:
                print(f"Backup verification successful: {args.name}")
            else:
                print(f"Backup verification failed: {args.name}")
                sys.exit(1)
        
        elif args.action == 'cleanup':
            backup_system.cleanup_old_backups()
            print("Backup cleanup completed")
    
    except Exception as e:
        logging.error(f"Operation failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()