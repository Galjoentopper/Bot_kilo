# Distributed Trading Bot System

This system allows you to train models on a powerful computer and run paper trading on a slower computer with easy model transfer capabilities.

## Overview

The distributed system consists of:
- **Training Computer**: Powerful machine for model training
- **Trading Computer**: Lighter machine for running paper trading
- **Model Transfer System**: Easy copying of trained models between machines

## Quick Start Guide

### 1. Setup Training Computer

1. **Configure Environment**:
   ```bash
   # Run the configuration script
   configure_environment.bat
   # Select option 1: "Set up for Training Computer"
   ```

2. **Train Models**:
   ```bash
   # Train all models
   train_models.bat
   
   # Or train specific models
   python scripts/enhanced_trainer.py --symbols BTCUSDT,ETHUSDT --models lightgbm,gru
   ```

3. **Package Models for Transfer**:
   ```bash
   # Package all trained models
   package_models.bat
   
   # This creates a transfer package in exports/
   ```

### 2. Setup Trading Computer

1. **Configure Environment**:
   ```bash
   # Run the configuration script
   configure_environment.bat
   # Select option 2: "Set up for Trading Computer"
   ```

2. **Import Models**:
   ```bash
   # Copy the package from training computer to imports/ folder
   # Then import the models
   import_models.bat
   ```

3. **Validate Models**:
   ```bash
   # Validate all imported models
   validate_models.bat
   ```

4. **Start Paper Trading**:
   ```bash
   # Start the enhanced paper trader
   start_trading.bat
   ```

## Detailed Instructions

### Training Computer Setup

#### Environment Configuration
The training computer should have:
- High-performance CPU/GPU
- Sufficient RAM (8GB+ recommended)
- Python 3.8+ with required packages

#### Training Process
1. **Configure Training Settings**:
   - Edit `config/config_training.yaml` for your preferences
   - Set symbols, timeframes, and model parameters

2. **Run Training**:
   ```bash
   # Full training pipeline
   python scripts/enhanced_trainer.py
   
   # With specific parameters
   python scripts/enhanced_trainer.py --symbols BTCUSDT,ETHUSDT --lookback 1000 --models all
   ```

3. **Monitor Training**:
   - Check logs in `logs/` directory
   - Monitor MLflow UI if configured
   - Review training metrics and performance

#### Model Packaging
```bash
# Package specific models
python scripts/package_models.py --symbols BTCUSDT --models lightgbm

# Package all models with compression
python scripts/package_models.py --compress --include-metadata
```

### Trading Computer Setup

#### Environment Configuration
The trading computer should have:
- Stable internet connection
- Python 3.8+ (same version as training computer)
- Minimal resource requirements

#### Model Import Process
1. **Transfer Package**:
   - Copy the `.tar.gz` package from training computer
   - Place in `imports/` directory

2. **Import Models**:
   ```bash
   # Import specific package
   python scripts/import_models.py --package imports/models_20240115_143022.tar.gz
   
   # Import latest package
   python scripts/import_models.py --latest
   ```

3. **Validate Compatibility**:
   ```bash
   # Full validation
   python scripts/validate_models.py
   
   # Quick validation (skip loading tests)
   python scripts/validate_models.py --quick
   
   # Generate HTML report
   python scripts/validate_models.py --report validation_report.html
   ```

#### Paper Trading
```bash
# Start enhanced paper trader
python scripts/enhanced_trader.py

# With specific configuration
python scripts/enhanced_trader.py --config config/config_trading.yaml
```

## Configuration Management

### Training Configuration (`config/config_training.yaml`)
- Optimized for high-performance training
- Includes advanced features and large datasets
- Supports multiple workers and GPU acceleration

### Trading Configuration (`config/config_trading.yaml`)
- Optimized for lightweight trading
- Reduced resource usage
- Fast model loading and inference

### Environment-Specific Settings
```bash
# Generate custom configuration
python src/config/config_generator.py

# Validate current environment
python src/config/environment_manager.py --validate
```

## Model Transfer Workflow

### 1. Training Computer
```bash
# Train models
python scripts/enhanced_trainer.py

# Package for transfer
python scripts/package_models.py --compress

# Verify package
ls -la exports/
```

### 2. File Transfer
- Copy package file to trading computer
- Use USB drive, network share, or cloud storage
- Ensure file integrity (check file size)

### 3. Trading Computer
```bash
# Import models
python scripts/import_models.py --package imports/your_package.tar.gz

# Validate models
python scripts/validate_models.py --report

# Start trading
python scripts/enhanced_trader.py
```

## Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check model compatibility
python scripts/validate_models.py --verbose

# Verify Python versions match
python --version
```

#### Package Import Failures
```bash
# Check package integrity
python scripts/import_models.py --verify-only

# Force reimport
python scripts/import_models.py --force
```

#### Trading Connection Issues
```bash
# Test API connectivity
python scripts/test_connection.py

# Check configuration
python src/config/environment_manager.py --status
```

### Performance Optimization

#### Training Computer
- Use GPU acceleration when available
- Increase worker processes for parallel training
- Enable advanced features for better models

#### Trading Computer
- Disable unnecessary features
- Use model caching for faster loading
- Optimize memory usage settings

## File Structure

```
Bot_kilo/
├── config/
│   ├── config_training.yaml    # Training computer config
│   ├── config_trading.yaml     # Trading computer config
│   └── config.yaml            # Active configuration
├── scripts/
│   ├── enhanced_trainer.py    # Enhanced training script
│   ├── enhanced_trader.py     # Enhanced trading script
│   ├── package_models.py      # Model packaging utility
│   ├── import_models.py       # Model import utility
│   └── validate_models.py     # Model validation script
├── src/
│   ├── config/
│   │   ├── environment_manager.py
│   │   └── config_generator.py
│   └── utils/
│       ├── model_packager.py
│       └── model_transfer.py
├── models/                    # Trained models
├── imports/                   # Import packages
├── exports/                   # Export packages
├── logs/                      # Log files
└── *.bat                     # Batch scripts
```

## Best Practices

### Model Training
1. Use consistent Python versions across machines
2. Train with sufficient historical data
3. Validate models before packaging
4. Include comprehensive metadata
5. Use version control for model tracking

### Model Transfer
1. Verify package integrity after transfer
2. Test models on trading computer before live use
3. Keep backup copies of working models
4. Document model performance and settings

### Paper Trading
1. Start with small position sizes
2. Monitor performance closely
3. Keep detailed logs
4. Regular model validation
5. Update models periodically

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files in `logs/` directory
3. Run validation scripts for diagnostics
4. Ensure environment configurations are correct

## Version Compatibility

- Python: 3.8+ (same version on both machines)
- Dependencies: Use `requirements.txt` for consistency
- Models: Validate compatibility before deployment
- Configuration: Use environment-specific configs