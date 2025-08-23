# Quick Start Guide - Cross-Platform Trading Bot

## 🚀 Fast Setup for Linux Training + Windows Trading

### Linux Training Computer (One-time setup)
```bash
# 1. Setup environment
chmod +x setup_training_environment.sh
./setup_training_environment.sh

# 2. Add your data to data/ folder
# 3. Configure settings in config/training_config.yaml

# 4. Train models and create package
chmod +x train_models.sh
./train_models.sh

# 5. Find your package in models/exports/
# 6. Transfer ZIP file to Windows computer
```

### Windows Trading Computer (Every model update)
```cmd
# 1. Copy transfer package to Bot_kilo folder
# 2. Import models
import_models.bat

# 3. Validate models (optional)
validate_models.bat

# 4. Start paper trading
deploy_trading.bat
```

## 📁 Required Files on Each System

### Linux (Training)
- `setup_training_environment.sh` ✓
- `train_models.sh` ✓
- `scripts/enhanced_trainer.py` ✓
- `scripts/cross_platform_transfer.py` ✓
- `config/training_config.yaml` ✓
- Your training data in `data/` folder

### Windows (Trading)
- `import_models.bat` ✓
- `validate_models.bat` ✓
- `deploy_trading.bat` ✓
- `scripts/enhanced_trader.py` ✓
- `scripts/cross_platform_transfer.py` ✓
- `config/trading_config.yaml` ✓

## 🔄 Regular Workflow

1. **Train on Linux** → `./train_models.sh`
2. **Transfer ZIP** → Copy to Windows
3. **Import on Windows** → `import_models.bat`
4. **Start Trading** → `deploy_trading.bat`

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Install Python 3.8+ |
| "No transfer package found" | Copy ZIP file to project root |
| "Models validation failed" | Re-run import or check package |
| "Trading bot crashes" | Check logs/ folder for errors |

## 📞 Need Help?

1. Check `CROSS_PLATFORM_SETUP.md` for detailed instructions
2. Check `TROUBLESHOOTING.md` for common issues
3. Look in `logs/` folder for error details

---
*For detailed setup instructions, see `CROSS_PLATFORM_SETUP.md`*