# Trading Bot - Complete Setup Guide

This guide walks you through setting up the trading bot from GitHub cloning to running live trades.

## Overview

**Training Computer (Linux):** Powerful machine for model training  
**Trading Computer (Windows):** Your regular computer for running the bot

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/Bot_kilo.git
cd Bot_kilo
```

## Step 2: Linux Training Setup

### 2.1 Environment Setup
```bash
./setup_training_environment.sh
```
This installs Python dependencies, creates directories, and sets up the training environment.

### 2.2 Fetch Training Data
```bash
./fetch_training_data.sh
```
Collects historical market data for training.

**Options:**
- `./fetch_training_data.sh --symbol BTCEUR` - Fetch specific symbol
- `./fetch_training_data.sh --update` - Update existing data

### 2.3 Train Models
```bash
./train_models.sh
```

**Training Options:**
- `./train_models.sh --symbols BTCEUR` - Train specific symbol
- `./train_models.sh --resume` - Resume from checkpoint
- `./train_models.sh --checkpoint-dir /path/to/checkpoints` - Custom checkpoint location

**Important:** Training creates a transfer package in `models/exports/` for Windows.

## Step 3: Windows Trading Setup

### 3.1 Environment Setup
```cmd
setup_environment.bat
```
Installs Python dependencies and creates necessary directories.

### 3.2 Import Models
1. Copy the transfer package (*.zip) from Linux to Windows Bot_kilo directory
2. Run:
```cmd
import_models.bat
```

### 3.3 Deploy Trading Bot
```cmd
deploy_trading.bat
```
Starts paper trading with imported models.

## Quick Commands

**Linux Training:**
```bash
# Full training workflow
./setup_training_environment.sh
./fetch_training_data.sh
./train_models.sh

# Resume interrupted training
./train_models.sh --resume
```

**Windows Trading:**
```cmd
# Setup and deploy
setup_environment.bat
import_models.bat
deploy_trading.bat
```

## Troubleshooting

**Training stops unexpectedly:**
1. Check terminal output for errors
2. Resume: `./train_models.sh --resume`
3. Check `checkpoints/` directory exists

**Resume fails:**
1. Verify checkpoint files in `checkpoints/`
2. If corrupted, delete checkpoints and restart
3. Check configuration hasn't changed

**No data found:**
1. Run data collection script first
2. Verify `data/` directory has .csv files
3. Check configuration files

**Windows import fails:**
1. Ensure transfer package (*.zip) is in Bot_kilo directory
2. Run `validate_models.bat` after import
3. Check `logs/` directory for errors

## File Structure
```
Bot_kilo/
├── data/                     # Market data (.csv files)
├── models/exports/           # Packaged models for transfer
├── checkpoints/              # Auto-saved training progress
├── logs/                     # Application logs
├── scripts/                  # Core Python scripts
├── config/                   # Configuration files
├── train_models.sh           # Linux training script
├── fetch_training_data.sh    # Linux data collection
├── setup_training_environment.sh # Linux environment setup
├── setup_environment.bat     # Windows environment setup
├── import_models.bat         # Windows model import
└── deploy_trading.bat        # Windows trading deployment
```

## Important Notes

- **Never train on Windows computer** - Use Linux for training only
- **Checkpoint system** prevents loss from 6-hour runtime limits
- **Transfer packages** ensure easy model deployment
- **Paper trading first** - Test before live trading
- Training automatically saves progress every model completion
- Use `--resume` to continue interrupted training sessions