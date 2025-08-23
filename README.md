# Cross-Platform Trading Bot System

## 🚀 Overview

This is a distributed trading bot system designed for optimal performance across different environments:
- **Training Computer**: Linux-based system for model training (high computational power)
- **Trading Computer**: Windows-based system for live paper trading (24/7 operation)

The system automatically packages trained models on Linux and seamlessly deploys them on Windows for trading.

## 📋 Prerequisites

### Linux Training Computer
- Python 3.8 or higher
- Git
- At least 8GB RAM (16GB+ recommended)
- CUDA-compatible GPU (optional but recommended)
- Internet connection for package downloads

### Windows Trading Computer
- Python 3.8 or higher
- PowerShell (built into Windows)
- At least 4GB RAM
- Stable internet connection
- Windows 10/11

## 🔧 Quick Start Guide

### Part 1: Linux Training Setup

#### Step 1: Clone and Setup Training Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd Bot_kilo

# Make scripts executable
chmod +x setup_training_environment.sh
chmod +x train_models.sh

# Run the setup script
./setup_training_environment.sh
```

#### Step 2: Collect Training Data (Automated)
```bash
# Make data collection script executable
chmod +x fetch_training_data.sh

# Run automated data collection
./fetch_training_data.sh

# This will:
# - Read symbols from config/config_training.yaml
# - Create data/ directory structure
# - Download bulk historical data (faster)
# - Fill gaps with API calls
# - Create SQLite databases for training
```

**Manual Data Preparation (Alternative)**
```bash
# If you prefer manual data setup:
mkdir -p data
# Place your CSV files: data/BTCUSDT.csv, data/ETHUSDT.csv
```

#### Step 3: Configure Training Settings
```bash
# Edit training configuration (optional)
nano config/config_training.yaml
```

#### Step 4: Train Models and Create Transfer Package
```bash
# Start training process
./train_models.sh

# This will:
# - Train GRU, LightGBM, and PPO models
# - Create a transfer package (models_transfer_YYYYMMDD_HHMMSS.zip)
# - Display package location when complete
```

### Part 2: Windows Trading Setup

#### Step 1: Prepare Windows Environment
```powershell
# Open PowerShell as Administrator and run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Clone or copy the repository to your Windows machine
git clone <your-repo-url>
cd Bot_kilo
```

#### Step 2: Transfer Model Package
```powershell
# Copy the .zip file from Linux to Windows
# Place it in the Bot_kilo root directory
# Example: models_transfer_20241220_143022.zip
```

#### Step 3: Import Models
```powershell
# Run the import script
.\import_models.bat

# This will:
# - Detect the transfer package
# - Validate and import models
# - Create necessary directories
```

#### Step 4: Validate Models (Optional but Recommended)
```powershell
# Validate imported models
.\validate_models.bat
```

#### Step 5: Start Paper Trading
```powershell
# Deploy and start trading
.\deploy_trading.bat

# The bot will start in paper trading mode
# Press Ctrl+C to stop when needed
```

## 📁 Directory Structure

```
Bot_kilo/
├── README.md                          # This file
├── config/
│   ├── config_training.yaml           # Training configuration
│   └── config_trading.yaml            # Trading configuration
├── data/                              # Trading data (CSV files)
├── src/                               # Source code modules
├── scripts/                           # Python utility scripts
├── logs/                              # Application logs
├── models/                            # Trained models
│   ├── exports/                       # Exported model packages
│   └── backups/                       # Model backups
├── Linux Scripts (root folder):
│   ├── setup_training_environment.sh  # Training environment setup
│   ├── train_models.sh                # Model training script
│   └── fetch_training_data.sh         # Data collection script
├── Windows Scripts (root folder):
│   ├── import_models.bat              # Model import utility
│   ├── validate_models.bat            # Model validation
│   └── deploy_trading.bat             # Trading deployment
├── Python Scripts (scripts/ folder):
│   ├── enhanced_trainer.py            # Model training
│   ├── enhanced_trader.py             # Trading bot
│   ├── cross_platform_transfer.py     # Model transfer
│   └── validate_models.py             # Model validation
└── processed_packages/                # Processed transfer packages
```

## 🔄 Complete Workflow

### Training Phase (Linux)
1. **Setup**: Run `setup_training_environment.sh`
2. **Data**: Run `fetch_training_data.sh` (reads symbols from config)
3. **Train**: Execute `train_models.sh`
4. **Package**: Transfer package is automatically created
5. **Transfer**: Copy `.zip` file to Windows machine

### Trading Phase (Windows)
1. **Import**: Run `import_models.bat`
2. **Validate**: Run `validate_models.bat` (optional)
3. **Deploy**: Execute `deploy_trading.bat`
4. **Monitor**: Check logs in `logs/` directory
5. **Stop**: Press Ctrl+C in trading terminal

## 🛠️ Configuration

### Training Configuration (`config/config_training.yaml`)
```yaml
training:
  symbols: ["BTCUSDT", "ETHUSDT"]  # Trading pairs
  lookback_period: 60               # Historical data points
  train_split: 0.8                  # Training data ratio
  
models:
  gru:
    epochs: 100
    batch_size: 32
  lightgbm:
    num_boost_round: 1000
  ppo:
    total_timesteps: 100000
```

### Trading Configuration (`config/config_trading.yaml`)
```yaml
trading:
  mode: "paper"                     # paper or live
  symbols: ["BTCUSDT", "ETHUSDT"]
  initial_balance: 10000
  
risk_management:
  max_position_size: 0.1            # 10% of portfolio
  stop_loss: 0.02                   # 2% stop loss
```

## 🚨 Troubleshooting

### Common Issues

#### "Python not found" Error
**Solution**: Install Python 3.8+ and ensure it's in your PATH
```bash
# Linux
sudo apt update && sudo apt install python3 python3-pip

# Windows
# Download from python.org and check "Add to PATH" during installation
```

#### "No transfer package found" Error
**Solution**: Ensure the `.zip` file is in the Bot_kilo root directory
```powershell
# Check for .zip files
dir *.zip

# If missing, copy from Linux machine
```

#### "Models failed to load" Error
**Solution**: Validate models and check compatibility
```powershell
.\scripts\validate_models.bat
```

#### Trading Bot Stops Unexpectedly
**Solution**: Check logs for errors
```powershell
# View recent logs
type logs\trader.log | Select-Object -Last 50
```

### Getting Help

1. **Check Logs**: Always check `logs/` directory first
2. **Validate Models**: Run validation scripts
3. **Review Configuration**: Ensure YAML files are correct
4. **Check README**: This file contains all necessary documentation

## 📊 Monitoring and Logs

### Log Files
- `logs/trainer.log` - Training process logs
- `logs/trader.log` - Trading bot logs
- `logs/transfer.log` - Model transfer logs
- `logs/validation.log` - Model validation logs

### Performance Monitoring
```powershell
# View trading performance
type logs\trader.log | findstr "Performance"

# Check model accuracy
type logs\validation.log | findstr "Accuracy"
```

## 🔐 Security Notes

- **Paper Trading**: System defaults to paper trading mode
- **API Keys**: Store in environment variables, never in code
- **Backups**: Models are automatically backed up before updates
- **Validation**: Always validate models before trading

## 🚀 Advanced Usage

### Custom Model Training
```bash
# Train specific symbols only
python scripts/enhanced_trainer.py --symbols BTCUSDT ETHUSDT

# Create transfer package without training
python scripts/enhanced_trainer.py --package-only
```

### Manual Model Management
```powershell
# Import specific package
python scripts/cross_platform_transfer.py --import package_name.zip

# Validate specific models
python scripts/validate_models.py --models gru lightgbm
```

## 📈 Next Steps

1. **Monitor Performance**: Check trading logs regularly
2. **Retrain Models**: Update models weekly/monthly
3. **Scale Up**: Add more trading pairs
4. **Go Live**: Switch to live trading (with caution)

---

## 📞 Support

For issues or questions:
1. Review log files in `logs/` directory
2. Validate your setup with provided scripts
3. Check the troubleshooting section above

**Happy Trading! 🎯**