# Cross-Platform Trading Bot Setup Guide

This guide explains how to set up the distributed trading bot system where models are trained on a Linux PC and paper trading runs on a Windows PC.

## Overview

The system consists of two main components:
1. **Linux Training Computer**: Trains models and creates transfer packages
2. **Windows Trading Computer**: Imports models and runs paper trading

## Prerequisites

### Linux Training Computer
- Python 3.8+
- pip3
- Git (optional, for cloning repository)
- Sufficient computational resources for model training
- CUDA-compatible GPU (recommended for faster training)

### Windows Trading Computer
- Python 3.8+
- pip
- Internet connection for market data
- Sufficient storage for models and logs

## Setup Instructions

### Part 1: Linux Training Computer Setup

#### 1. Clone or Copy the Repository
```bash
# If using git
git clone <repository-url> Bot_kilo
cd Bot_kilo

# Or copy the project files to your Linux machine
```

#### 2. Set Up Training Environment
```bash
# Make the setup script executable
chmod +x setup_training_environment.sh

# Run the setup script
./setup_training_environment.sh
```

This script will:
- Check Python installation
- Create a virtual environment
- Install required dependencies
- Create necessary directories
- Set up configuration files

#### 3. Configure Training Settings
Edit the configuration files as needed:
```bash
# Edit training configuration
nano config/training_config.yaml

# Edit main configuration
nano config.yaml
```

#### 4. Prepare Training Data
Ensure your training data is in the `data/` directory:
```bash
# Your data structure should look like:
data/
├── BTCUSDT_1h.csv
├── ETHUSDT_1h.csv
└── ... (other symbol data)
```

#### 5. Train Models and Create Transfer Package
```bash
# Make the training script executable
chmod +x train_models.sh

# Run training (this may take several hours)
./train_models.sh
```

This script will:
- Train models for all configured symbols
- Validate the trained models
- Create a transfer package (ZIP file)
- Display instructions for transferring to Windows

### Part 2: Transfer Models to Windows

#### 1. Locate the Transfer Package
After training completes, you'll find:
```bash
models/exports/transfer_package_YYYYMMDD_HHMMSS.zip
```

#### 2. Transfer to Windows Computer
Use any method to transfer the ZIP file:
- USB drive
- Network share
- Cloud storage (Google Drive, Dropbox, etc.)
- SCP/SFTP
- Email (if file size permits)

**Example using SCP:**
```bash
# From Linux to Windows (if Windows has SSH server)
scp models/exports/transfer_package_*.zip user@windows-pc:/path/to/Bot_kilo/

# Or use WinSCP, FileZilla, or similar tools
```

### Part 3: Windows Trading Computer Setup

#### 1. Prepare the Project Directory
Ensure you have the Bot_kilo project on your Windows machine with the same structure.

#### 2. Copy Transfer Package
Place the transfer package ZIP file in the root directory of your Bot_kilo project:
```
Bot_kilo/
├── transfer_package_YYYYMMDD_HHMMSS.zip  ← Place here
├── scripts/
├── models/
└── ...
```

#### 3. Import Models
Double-click `import_models.bat` or run from command prompt:
```cmd
import_models.bat
```

This script will:
- Validate the transfer package
- Import models to the `models/` directory
- Verify the import was successful
- Move the processed package to `processed_packages/`

#### 4. Validate Models (Optional but Recommended)
Run the validation script:
```cmd
validate_models.bat
```

#### 5. Configure Trading Settings
Edit the trading configuration:
```cmd
notepad config\trading_config.yaml
```

Key settings to review:
- API keys (if using real exchanges for data)
- Trading pairs
- Risk management parameters
- Telegram notifications (optional)

#### 6. Start Paper Trading
Double-click `deploy_trading.bat` or run from command prompt:
```cmd
deploy_trading.bat
```

This script will:
- Check all dependencies
- Validate models one more time
- Start the paper trading bot
- Display real-time trading activity

## File Structure After Setup

### Linux Training Computer
```
Bot_kilo/
├── data/                          # Training data
├── models/
│   ├── lightgbm/                 # Trained LightGBM models
│   ├── gru/                      # Trained GRU models
│   ├── ppo/                      # Trained PPO models
│   └── exports/                  # Transfer packages
├── mlruns/                       # MLflow tracking
├── logs/                         # Training logs
├── scripts/
│   ├── enhanced_trainer.py       # Enhanced training script
│   ├── cross_platform_transfer.py # Transfer utility
│   └── ...
├── config/
│   └── training_config.yaml      # Training configuration
├── train_models.sh               # Training script
└── setup_training_environment.sh # Setup script
```

### Windows Trading Computer
```
Bot_kilo/
├── models/                       # Imported models
│   ├── lightgbm/
│   ├── gru/
│   └── ppo/
├── logs/                         # Trading logs
├── data/                         # Market data cache
├── config/
│   └── trading_config.yaml       # Trading configuration
├── processed_packages/           # Processed transfer packages
├── scripts/
│   ├── enhanced_trader.py        # Enhanced trading script
│   ├── cross_platform_transfer.py # Transfer utility
│   └── ...
├── import_models.bat             # Model import script
├── validate_models.bat           # Model validation script
└── deploy_trading.bat            # Trading deployment script
```

## Workflow Summary

1. **Linux**: Set up training environment → Train models → Create transfer package
2. **Transfer**: Copy ZIP file from Linux to Windows
3. **Windows**: Import models → Validate → Configure → Start trading

## Updating Models

To update models with new training:

1. **Linux**: Run `./train_models.sh` again
2. **Transfer**: Copy the new transfer package to Windows
3. **Windows**: Run `import_models.bat` (old models will be backed up automatically)
4. **Windows**: Restart trading with `deploy_trading.bat`

## Troubleshooting

### Common Issues

#### Linux Training Issues
- **Out of memory**: Reduce batch size in training config
- **CUDA errors**: Check GPU drivers and CUDA installation
- **Missing data**: Ensure data files are in correct format and location

#### Transfer Issues
- **Large file size**: Use compression or split large packages
- **Corruption**: Verify file integrity after transfer
- **Permission errors**: Check file permissions on both systems

#### Windows Trading Issues
- **Import fails**: Check if ZIP file is corrupted
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Model validation fails**: Re-import models or check compatibility
- **Trading errors**: Check network connection and API credentials

### Getting Help

1. Check the logs directory for detailed error messages
2. Run validation scripts to identify specific issues
3. Ensure all dependencies are installed correctly
4. Verify configuration files are properly formatted

## Advanced Usage

### Manual Transfer Operations

#### Create Transfer Package Manually
```bash
# Linux
python scripts/cross_platform_transfer.py create --source ./models --output my_models.zip
```

#### Import Package Manually
```cmd
REM Windows
python scripts/cross_platform_transfer.py import --package my_models.zip --destination models
```

#### Validate Package
```bash
# Linux or Windows
python scripts/cross_platform_transfer.py validate --package my_models.zip
```

### Automation

You can automate the workflow using:
- Cron jobs on Linux for scheduled training
- Task Scheduler on Windows for automated trading restarts
- Network shares for automatic file transfer
- Cloud storage APIs for seamless synchronization

## Security Considerations

1. **API Keys**: Never include real API keys in transfer packages
2. **Network Transfer**: Use secure methods (SFTP, encrypted cloud storage)
3. **File Permissions**: Ensure proper file permissions on both systems
4. **Backup**: Always backup existing models before importing new ones

## Performance Tips

1. **Linux Training**:
   - Use SSD storage for faster I/O
   - Enable GPU acceleration when available
   - Monitor system resources during training

2. **Windows Trading**:
   - Ensure stable internet connection
   - Monitor system resources during trading
   - Regular log cleanup to prevent disk space issues

3. **Transfer Optimization**:
   - Compress transfer packages when possible
   - Use incremental updates for frequent model updates
   - Consider using rsync or similar tools for large transfers