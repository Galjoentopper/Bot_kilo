# Distributed Trading Bot Setup Guide

This guide explains how to train models on a powerful computer and run paper trading on a slower computer using the enhanced distributed training system.

## Overview

The system consists of:
- **Training Computer**: Powerful machine for model training
- **Trading Computer**: Local machine for paper trading
- **Model Transfer System**: Easy transfer of trained models between computers

## Quick Start

### On Training Computer

1. **Setup Environment**
   ```bash
   # Run the setup script
   setup_environment.bat
   
   # Or manually:
   pip install -r requirements.txt
   ```

2. **Train Models**
   ```bash
   # Use the user-friendly training script
   train_models.bat
   
   # Or run directly:
   python scripts/enhanced_trainer.py --package-models --create-transfer-bundle
   ```

3. **Transfer Models**
   - Find the generated transfer bundle in `models/exports/transfer_bundle_YYYYMMDD_HHMMSS.zip`
   - Copy this file to your trading computer

### On Trading Computer

1. **Setup Environment**
   ```bash
   # Run the setup script
   setup_environment.bat
   ```

2. **Import Models**
   ```bash
   # Import the transfer bundle
   python scripts/import_models.py path/to/transfer_bundle.zip
   
   # Or validate existing models
   python scripts/validate_models.py
   ```

3. **Run Paper Trading**
   ```bash
   # Use the deployment script
   deploy_trading.bat
   
   # Or run directly:
   python scripts/trader.py
   ```

## Detailed Setup Instructions

### Training Computer Setup

#### 1. Environment Preparation

```bash
# Clone the repository
git clone <repository-url>
cd Bot_kilo

# Run setup script
setup_environment.bat
```

The setup script will:
- Check Python installation
- Install required dependencies
- Create necessary directories
- Guide you through configuration

#### 2. Configuration

Edit `config.yaml` to match your training preferences:

```yaml
trainer:
  symbols: ["BTCEUR", "ETHEUR", "ADAEUR"]  # Symbols to train
  interval: "30m"                           # Time interval
  target_type: "regression"                 # or "classification"
  target_horizon: 24                        # Prediction horizon
  cv_splits: 5                             # Cross-validation splits
  max_workers: 4                           # Parallel workers
  
models:
  lightgbm:
    enabled: true
  gru:
    enabled: true
  ppo:
    enabled: true
```

#### 3. Training Models

**Option A: Use the batch script (Recommended)**
```bash
train_models.bat
```

**Option B: Manual training**
```bash
# Basic training
python scripts/enhanced_trainer.py

# Training with packaging and transfer bundle
python scripts/enhanced_trainer.py --package-models --create-transfer-bundle

# Custom parameters
python scripts/enhanced_trainer.py --symbols BTCEUR ETHEUR --models lightgbm gru --package-models
```

#### 4. Transfer Bundle Creation

After training, you'll find:
- Individual model packages in `models/exports/packages/`
- Transfer bundle in `models/exports/transfer_bundle_YYYYMMDD_HHMMSS.zip`
- Import instructions in `models/exports/IMPORT_INSTRUCTIONS.txt`

### Trading Computer Setup

#### 1. Environment Preparation

```bash
# Ensure you have the same codebase
git clone <repository-url>
cd Bot_kilo

# Run setup script
setup_environment.bat
```

#### 2. Model Import

**Option A: Use import script**
```bash
# Import from transfer bundle
python scripts/import_models.py path/to/transfer_bundle.zip

# Import with options
python scripts/import_models.py transfer_bundle.zip --no-backup --verbose
```

**Option B: Manual import**
1. Extract the transfer bundle to a temporary location
2. Copy model directories to `models/`
3. Run validation: `python scripts/validate_models.py`

#### 3. Model Validation

```bash
# Validate all models
python scripts/validate_models.py

# Check specific model types
python scripts/validate_models.py --model-types lightgbm gru
```

#### 4. Configuration for Trading

Update `config.yaml` for your trading environment:

```yaml
trading:
  mode: "paper"                    # Paper trading mode
  symbols: ["BTCEUR", "ETHEUR"]   # Symbols to trade
  interval: "30m"                  # Must match training interval
  
models:
  lightgbm:
    enabled: true
    weight: 0.4
  gru:
    enabled: true
    weight: 0.4
  ppo:
    enabled: true
    weight: 0.2
```

#### 5. Run Paper Trading

**Option A: Use deployment script (Recommended)**
```bash
deploy_trading.bat
```

**Option B: Manual execution**
```bash
# Run trader
python scripts/trader.py

# Run with specific config
python scripts/trader.py --config config.yaml

# Run with validation
python scripts/validate_models.py && python scripts/trader.py
```

## File Structure

```
Bot_kilo/
├── scripts/
│   ├── enhanced_trainer.py      # Enhanced training script
│   ├── trader.py               # Paper trading bot
│   ├── import_models.py        # Model import utility
│   ├── validate_models.py      # Model validation utility
│   ├── train_models.bat        # Training batch script
│   ├── deploy_trading.bat      # Trading deployment script
│   └── setup_environment.bat   # Environment setup script
├── src/
│   └── utils/
│       ├── model_packaging.py  # Model packaging utilities
│       └── model_transfer.py   # Model transfer utilities
├── models/
│   ├── exports/               # Exported model packages
│   ├── lightgbm/             # LightGBM models
│   ├── gru/                   # GRU models
│   └── ppo/                   # PPO models
├── config.yaml               # Main configuration
└── requirements.txt          # Python dependencies
```

## Advanced Usage

### Custom Training Parameters

```bash
# Train specific symbols and models
python scripts/enhanced_trainer.py \
  --symbols BTCEUR ETHEUR \
  --models lightgbm gru \
  --target-type regression \
  --cv-splits 3 \
  --package-models

# Use custom configuration
python scripts/enhanced_trainer.py \
  --config custom_config.yaml \
  --output-dir custom_models \
  --create-transfer-bundle
```

### Model Management

```bash
# List available model packages
python -c "from src.utils.model_packaging import ModelPackager; ModelPackager().list_packages()"

# Validate specific models
python scripts/validate_models.py --model-types lightgbm --symbols BTCEUR

# Clean old models (keep last 5)
python -c "from src.utils.model_transfer import ModelTransferManager; ModelTransferManager().cleanup_old_transfers(keep=5)"
```

### Troubleshooting

#### Common Issues

1. **Missing Dependencies**
   ```bash
   # Install missing packages
   pip install lightgbm torch stable-baselines3
   ```

2. **Model Loading Errors**
   ```bash
   # Validate models
   python scripts/validate_models.py
   
   # Check logs
   tail -f logs/model_validation.log
   ```

3. **Transfer Bundle Issues**
   ```bash
   # Re-create transfer bundle
   python scripts/enhanced_trainer.py --create-transfer-bundle --no-train
   ```

4. **Configuration Errors**
   ```bash
   # Validate configuration
   python -c "import yaml; print(yaml.safe_load(open('config.yaml')))"
   ```

#### Log Files

- Training logs: `logs/training.log`
- Trading logs: `logs/trading.log`
- Import logs: `logs/model_import.log`
- Validation logs: `logs/model_validation.log`

## Best Practices

### Training Computer

1. **Regular Training Schedule**
   - Set up automated training (e.g., weekly)
   - Monitor training performance
   - Keep training logs for analysis

2. **Model Management**
   - Archive old models periodically
   - Document training parameters
   - Test models before transfer

3. **Resource Management**
   - Monitor GPU/CPU usage during training
   - Use appropriate batch sizes
   - Consider parallel training for multiple symbols

### Trading Computer

1. **Model Updates**
   - Regularly import new models
   - Validate models before deployment
   - Keep backup of working models

2. **Monitoring**
   - Monitor trading performance
   - Check logs regularly
   - Set up alerts for errors

3. **Maintenance**
   - Clean old logs periodically
   - Update dependencies regularly
   - Backup configuration files

## Security Considerations

1. **API Keys**
   - Store API keys securely
   - Use environment variables
   - Never commit keys to version control

2. **Model Transfer**
   - Verify model integrity after transfer
   - Use secure transfer methods
   - Validate models before use

3. **Network Security**
   - Use VPN for remote access
   - Secure file transfer protocols
   - Regular security updates

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review log files
3. Validate your configuration
4. Ensure all dependencies are installed

## Version Compatibility

- Python: 3.8+
- PyTorch: 1.9+
- LightGBM: 3.0+
- Stable Baselines3: 1.5+

Ensure both computers have compatible versions for successful model transfer.