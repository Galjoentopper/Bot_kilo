# Quick Setup Guide - Distributed Trading Bot

## 🚀 Get Started in 5 Minutes

This guide helps you quickly set up distributed model training and paper trading.

## Prerequisites

- **Training Computer**: Powerful machine with Python 3.8+
- **Trading Computer**: Any machine with Python 3.8+ and internet
- **Same Python Version**: Both machines must use the same Python version

## Step 1: Training Computer Setup

### 1.1 Configure Environment
```bash
# Run this command and select option 1
configure_environment.bat
```

### 1.2 Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 1.3 Train Models
```bash
# Quick training with default settings
train_models.bat

# Or customize your training
python scripts/enhanced_trainer.py --symbols BTCUSDT,ETHUSDT --lookback 500
```

### 1.4 Package Models
```bash
# Package all trained models for transfer
package_models.bat
```

**✅ Result**: You'll find a `.tar.gz` file in the `exports/` folder

## Step 2: Transfer Models

### 2.1 Copy Package
- Copy the `.tar.gz` file from `exports/` folder on training computer
- Transfer to trading computer (USB, network, cloud, etc.)
- Place in `imports/` folder on trading computer

## Step 3: Trading Computer Setup

### 3.1 Configure Environment
```bash
# Run this command and select option 2
configure_environment.bat
```

### 3.2 Install Dependencies
```bash
# Install required packages (lighter version)
pip install -r requirements.txt
```

### 3.3 Import Models
```bash
# Import the transferred models
import_models.bat
```

### 3.4 Validate Models
```bash
# Check if models work correctly
validate_models.bat
```

### 3.5 Start Paper Trading
```bash
# Start the paper trading bot
start_trading.bat
```

**✅ Result**: Your paper trading bot is now running with models trained on the powerful computer!

## 📋 Checklist

### Training Computer ✓
- [ ] Environment configured for training
- [ ] Dependencies installed
- [ ] Models trained successfully
- [ ] Models packaged for transfer
- [ ] Package file copied to trading computer

### Trading Computer ✓
- [ ] Environment configured for trading
- [ ] Dependencies installed
- [ ] Models imported successfully
- [ ] Models validated (no errors)
- [ ] Paper trading started

## 🔧 Quick Commands Reference

### Training Computer
```bash
# Setup
configure_environment.bat          # Configure for training
train_models.bat                   # Train all models
package_models.bat                 # Package for transfer

# Advanced
python scripts/enhanced_trainer.py --help    # See training options
python scripts/package_models.py --help      # See packaging options
```

### Trading Computer
```bash
# Setup
configure_environment.bat          # Configure for trading
import_models.bat                  # Import models
validate_models.bat                # Validate models
start_trading.bat                  # Start trading

# Monitoring
python scripts/validate_models.py --report   # Generate validation report
tail -f logs/paper_trading.log               # Monitor trading logs
```

## 🚨 Troubleshooting

### Problem: "No models found"
**Solution**: Make sure you've run the import process and models are in the `models/` folder

### Problem: "Model loading failed"
**Solution**: 
1. Check Python versions match on both computers
2. Run `validate_models.bat` to see specific errors
3. Ensure all dependencies are installed

### Problem: "API connection failed"
**Solution**: 
1. Check internet connection
2. Verify API keys in configuration
3. Test with `python scripts/test_connection.py`

### Problem: "Package import failed"
**Solution**:
1. Verify package file is not corrupted
2. Check file permissions
3. Try `python scripts/import_models.py --force`

## 📊 Monitoring Your System

### Check Status
```bash
# Environment status
python src/config/environment_manager.py --status

# Model validation
python scripts/validate_models.py --quick

# Trading performance
tail -f logs/paper_trading.log
```

### Generate Reports
```bash
# Validation report
python scripts/validate_models.py --report validation.html

# Training report (on training computer)
python scripts/enhanced_trainer.py --report
```

## 🔄 Regular Workflow

### Weekly Model Update
1. **Training Computer**: Train new models with latest data
2. **Package**: Create new transfer package
3. **Transfer**: Copy package to trading computer
4. **Trading Computer**: Import and validate new models
5. **Deploy**: Restart trading with updated models

### Daily Monitoring
1. Check trading logs for any errors
2. Monitor portfolio performance
3. Verify model predictions are reasonable
4. Check system resource usage

## 📁 Important Files

```
config/
├── config_training.yaml    # Training settings
├── config_trading.yaml     # Trading settings
└── config.yaml            # Active config

logs/
├── training.log           # Training logs
├── paper_trading.log      # Trading logs
└── model_validation.log   # Validation logs

models/                    # Your trained models
imports/                   # Import packages here
exports/                   # Export packages created here
```

## 🎯 Next Steps

Once you have the basic system running:

1. **Customize Training**: Edit `config/config_training.yaml` for your preferences
2. **Optimize Trading**: Adjust `config/config_trading.yaml` for your risk tolerance
3. **Add Symbols**: Include more trading pairs in your configuration
4. **Monitor Performance**: Set up alerts and notifications
5. **Scale Up**: Add more sophisticated models and features

## 📚 Need More Help?

- **Detailed Guide**: See `README_DISTRIBUTED_TRAINING.md`
- **Configuration**: Check `src/config/` utilities
- **Logs**: Review files in `logs/` directory
- **Validation**: Run `validate_models.bat` for diagnostics

---

**🎉 Congratulations!** You now have a distributed trading bot system running. The heavy training happens on your powerful computer, while lightweight trading runs on any machine!