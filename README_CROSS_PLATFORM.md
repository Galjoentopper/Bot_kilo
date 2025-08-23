# Cross-Platform Trading Bot Setup

## 🎯 Overview

This trading bot system is designed for **distributed training and trading**:
- **Linux PC**: Powerful machine for model training
- **Windows PC**: Stable machine for paper trading

The system automatically packages trained models on Linux and seamlessly imports them on Windows.

## 🚀 Quick Start

### For Linux Training Computer
```bash
# One-time setup
./setup_training_environment.sh

# Train models (repeat as needed)
./train_models.sh
```

### For Windows Trading Computer
```cmd
REM Import models from Linux
import_models.bat

REM Start paper trading
deploy_trading.bat
```

## 📚 Documentation

| Document | Purpose |
|----------|----------|
| `QUICK_START.md` | Fast setup reference |
| `CROSS_PLATFORM_SETUP.md` | Detailed setup instructions |
| `CROSS_PLATFORM_WORKFLOW.md` | Complete workflow explanation |
| `TROUBLESHOOTING.md` | Common issues and solutions |

## 🛠️ Available Scripts

### Linux Scripts
- `setup_training_environment.sh` - Set up training environment
- `train_models.sh` - Train models and create transfer package

### Windows Scripts
- `import_models.bat` - Import models from transfer package
- `validate_models.bat` - Validate imported models
- `deploy_trading.bat` - Start paper trading bot

### Cross-Platform Utilities
- `scripts/cross_platform_transfer.py` - Handle model packaging/import
- `scripts/enhanced_trainer.py` - Enhanced training with auto-packaging
- `scripts/enhanced_trader.py` - Enhanced trading bot

## 🔄 Typical Workflow

1. **Linux**: Train models → Create transfer package
2. **Transfer**: Copy ZIP file to Windows
3. **Windows**: Import models → Start trading

## ✅ System Requirements

### Linux Training Computer
- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- GPU recommended for faster training
- Sufficient storage for training data and models

### Windows Trading Computer
- Python 3.8+
- 4GB+ RAM
- Stable internet connection
- Sufficient storage for models and logs

## 🎯 Key Features

- **Automatic Model Packaging**: Models are automatically packaged for transfer
- **Cross-Platform Compatibility**: Seamless transfer between Linux and Windows
- **Model Validation**: Automatic validation of imported models
- **Backup System**: Old models are automatically backed up
- **Error Handling**: Comprehensive error checking and user guidance
- **Progress Tracking**: Clear feedback on all operations

## 🔧 Configuration

### Linux Training
- Edit `config/training_config.yaml` for training parameters
- Place market data in `data/` directory

### Windows Trading
- Edit `config/trading_config.yaml` for trading parameters
- Configure API keys and trading pairs

## 📁 Directory Structure

```
Bot_kilo/
├── Linux Training Files
│   ├── setup_training_environment.sh
│   ├── train_models.sh
│   └── config/training_config.yaml
├── Windows Trading Files
│   ├── import_models.bat
│   ├── validate_models.bat
│   ├── deploy_trading.bat
│   └── config/trading_config.yaml
├── Cross-Platform Scripts
│   └── scripts/
│       ├── cross_platform_transfer.py
│       ├── enhanced_trainer.py
│       └── enhanced_trader.py
└── Documentation
    ├── QUICK_START.md
    ├── CROSS_PLATFORM_SETUP.md
    └── CROSS_PLATFORM_WORKFLOW.md
```

## 🆘 Need Help?

1. **Quick Reference**: Check `QUICK_START.md`
2. **Detailed Setup**: Read `CROSS_PLATFORM_SETUP.md`
3. **Workflow Understanding**: See `CROSS_PLATFORM_WORKFLOW.md`
4. **Issues**: Check `TROUBLESHOOTING.md`
5. **Logs**: Look in the `logs/` directory for detailed error information

## 🔒 Security Notes

- Never include real API keys in transfer packages
- Use secure methods for file transfer (SFTP, encrypted cloud storage)
- Always validate models before using them for trading
- Keep backups of working model versions

## 🎉 Success Indicators

### Linux Training Success
- Training completes without errors
- Transfer package is created in `models/exports/`
- Package validation passes

### Windows Import Success
- Models import without errors
- Validation script passes
- Trading bot starts successfully

---

**Ready to start?** Begin with `QUICK_START.md` for immediate setup or `CROSS_PLATFORM_SETUP.md` for detailed instructions.