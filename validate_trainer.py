#!/usr/bin/env python3
"""
Comprehensive validation script for trainer.py fixes
Run this after installing dependencies to validate trainer functionality
"""

import sys
import os
import subprocess
import yaml
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("=== Checking Dependencies ===")
    
    required_packages = [
        'numpy', 'pandas', 'torch', 'lightgbm', 'scikit-learn', 
        'yaml', 'mlflow', 'optuna'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("✓ All dependencies satisfied")
        return True

def test_trainer_help():
    """Test trainer.py help command"""
    print("\n=== Testing Trainer Help ===")
    
    try:
        result = subprocess.run([
            sys.executable, 'scripts/trainer.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Trainer help command works")
            print("Available options:")
            # Extract key options from help
            lines = result.stdout.split('\n')
            for line in lines:
                if '--' in line and any(opt in line for opt in ['models', 'symbols', 'interval', 'target-type']):
                    print(f"  {line.strip()}")
        else:
            print(f"✗ Trainer help failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Trainer help command timed out")
        return False
    except Exception as e:
        print(f"✗ Error running trainer help: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading"""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        # Test basic config loading
        sys.path.insert(0, '.')
        from scripts.trainer import load_config
        
        config = load_config('src/config/config.yaml')
        if not config:
            print("✗ Failed to load configuration")
            return False
        
        print("✓ Configuration loaded successfully")
        print(f"  Symbols: {config.get('data', {}).get('symbols', [])}")
        print(f"  Interval: {config.get('data', {}).get('interval', 'unknown')}")
        print(f"  Models: {config.get('trainer', {}).get('default_models', [])}")
        
        # Test data file existence
        data_dir = config.get('data', {}).get('data_dir', './data')
        interval = config.get('data', {}).get('interval', '15m')
        symbols = config.get('data', {}).get('symbols', [])
        
        print(f"  Data files in {data_dir}:")
        for symbol in symbols:
            db_file = f"{symbol.lower()}_{interval}.db"
            file_path = os.path.join(data_dir, db_file)
            exists = os.path.exists(file_path)
            size = os.path.getsize(file_path) if exists else 0
            print(f"    {db_file}: {'✓' if exists else '✗'} ({size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_trainer_dry_run():
    """Test trainer with minimal dry run"""
    print("\n=== Testing Trainer Dry Run ===")
    
    try:
        # Test with minimal parameters for quick validation
        cmd = [
            sys.executable, 'scripts/trainer.py',
            '--models', 'lightgbm',
            '--symbols', 'BTCEUR',
            '--n-splits', '2',
            '--verbose'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("(This may take a few minutes...)")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✓ Trainer dry run successful")
            # Look for key indicators in output
            output = result.stdout + result.stderr
            if 'Training BTCEUR' in output:
                print("  ✓ Symbol training initiated")
            if 'lightgbm' in output:
                print("  ✓ LightGBM model processed")
            if 'Saved' in output and 'artifacts' in output:
                print("  ✓ Model artifacts saved")
        else:
            print(f"✗ Trainer dry run failed (exit code: {result.returncode})")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Trainer dry run timed out (may indicate infinite loop)")
        return False
    except Exception as e:
        print(f"✗ Error running trainer dry run: {e}")
        return False
    
    return True

def validate_outputs():
    """Validate that trainer produces expected outputs"""
    print("\n=== Validating Outputs ===")
    
    models_dir = './models'
    if not os.path.exists(models_dir):
        print("✗ Models directory not created")
        return False
    
    # Check for model subdirectories
    expected_structure = ['lightgbm', 'gru', 'ppo']
    found_models = []
    
    for model_type in expected_structure:
        model_path = os.path.join(models_dir, model_type)
        if os.path.exists(model_path):
            found_models.append(model_type)
            print(f"  ✓ {model_type} directory exists")
            
            # Check for symbol directories
            symbol_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            if symbol_dirs:
                print(f"    Symbol directories: {symbol_dirs}")
        else:
            print(f"  - {model_type} directory not found (may not have been trained)")
    
    if found_models:
        print(f"✓ Found model outputs for: {found_models}")
        return True
    else:
        print("⚠ No model outputs found (may need to run trainer first)")
        return False

def main():
    """Main validation function"""
    print("Bot_kilo Trainer Validation Script")
    print("=" * 50)
    
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..', '..')
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Trainer Help", test_trainer_help),
        ("Configuration", test_config_loading),
        ("Trainer Dry Run", test_trainer_dry_run),
        ("Output Validation", validate_outputs)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print(f"\n⚠ Test '{test_name}' interrupted by user")
            break
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Trainer is ready for use.")
    elif passed >= total * 0.8:
        print("⚠ Most tests passed. Some issues may need attention.")
    else:
        print("❌ Multiple failures detected. Check error messages above.")
    
    print("\nNext steps:")
    print("1. Run: python scripts/trainer.py --help")
    print("2. Train models: python scripts/trainer.py --models lightgbm gru --symbols BTCEUR")
    print("3. Use models: python scripts/trader.py")

if __name__ == "__main__":
    main()