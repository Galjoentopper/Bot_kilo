# Trading Bot Deployment Guide

This comprehensive guide will help you set up and deploy the distributed trading bot system across multiple computers.

## 🚀 Quick Start

### For Training Computer (Powerful Machine)
```bash
# 1. Configure environment
configure_environment.bat
# Select option 1: "Set up for Training Computer"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models
train_models.bat

# 4. Package models
package_models.bat
```

### For Trading Computer (This Machine)
```bash
# 1. Configure environment
configure_environment.bat
# Select option 2: "Set up for Trading Computer"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Import models (copy from training computer)
import_models.bat

# 4. Validate models
validate_models.bat

# 5. Start paper trading
paper_trading.bat
```

## 📋 Detailed Setup Instructions

### Prerequisites

- **Python 3.8+** installed on both computers
- **Git** for version control
- **Network access** for data downloads
- **USB drive or network share** for model transfer

### Training Computer Setup

#### 1. Environment Configuration
```bash
# Run the environment configurator
configure_environment.bat

# Select "Training Computer" when prompted
# This will:
# - Detect system capabilities (RAM, CPU, GPU)
# - Generate optimized training configuration
# - Set up directory structure
# - Configure logging and monitoring
```

#### 2. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# For GPU training (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Configure Trading Parameters
```bash
# Generate custom configuration
python src/config/config_generator.py

# Follow the interactive prompts to set:
# - Trading symbols (e.g., BTCUSDT, ETHUSDT)
# - Risk level (Conservative/Moderate/Aggressive)
# - Trading style (Scalping/Day Trading/Swing)
# - Position sizing
# - Risk management rules
```

#### 4. Train Models
```bash
# Start training with the batch script
train_models.bat

# Or run directly with Python
python scripts/train_models.py --config config/config_training.yaml

# Monitor training progress:
# - Check logs/training.log for detailed logs
# - Use MLflow UI for experiment tracking
# - Monitor system resources
```

#### 5. Package Models
```bash
# Package trained models for transfer
package_models.bat

# This creates:
# - exports/model_package_YYYYMMDD_HHMMSS.tar.gz
# - Complete model bundle with metadata
# - Transfer instructions
```

### Trading Computer Setup

#### 1. Environment Configuration
```bash
# Run the environment configurator
configure_environment.bat

# Select "Trading Computer" when prompted
# This will:
# - Detect system limitations
# - Generate lightweight trading configuration
# - Set up minimal resource usage
# - Configure real-time monitoring
```

#### 2. Install Dependencies
```bash
# Install minimal required packages
pip install -r requirements.txt

# Skip heavy ML libraries if not needed for inference
```

#### 3. Transfer Models

**Option A: USB Drive**
```bash
# 1. Copy model package from training computer to USB
# 2. Insert USB into trading computer
# 3. Run import script
import_models.bat
# Select USB drive location when prompted
```

**Option B: Network Share**
```bash
# 1. Set up shared folder on training computer
# 2. Map network drive on trading computer
# 3. Run import script
import_models.bat
# Select network location when prompted
```

**Option C: Cloud Storage**
```bash
# 1. Upload model package to cloud (Google Drive, Dropbox, etc.)
# 2. Download on trading computer
# 3. Run import script
import_models.bat
# Select download location when prompted
```

#### 4. Validate Models
```bash
# Comprehensive model validation
validate_models.bat

# This checks:
# - Model file integrity
# - Python version compatibility
# - Feature consistency
# - Dependency availability
# - Performance benchmarks
```

#### 5. Configure Trading
```bash
# Set up trading parameters
python src/config/config_generator.py

# Configure for trading computer:
# - Lighter resource usage
# - Real-time data feeds
# - Risk management
# - Notification settings
```

#### 6. Start Paper Trading
```bash
# Start paper trading bot
paper_trading.bat

# Monitor trading:
# - Check logs/trading.log
# - Monitor performance metrics
# - Review trade decisions
```

## 🔄 Regular Workflow

### Weekly Model Updates

**On Training Computer:**
```bash
# 1. Update data
python scripts/update_data.py

# 2. Retrain models
train_models.bat

# 3. Package new models
package_models.bat

# 4. Create backup
backup_manager.bat
# Select "Create Full Backup"
```

**On Trading Computer:**
```bash
# 1. Stop current trading
# Stop the paper_trading.bat process

# 2. Backup current setup
backup_manager.bat
# Select "Create Full Backup"

# 3. Import new models
import_models.bat

# 4. Validate new models
validate_models.bat

# 5. Restart trading
paper_trading.bat
```

### Daily Monitoring

```bash
# Check system status
monitor_system.bat

# Review trading performance
python scripts/generate_report.py --daily

# Verify model performance
validate_models.bat --quick
```

## 📊 Monitoring and Maintenance

### Performance Monitoring

- **Training Computer:**
  - GPU/CPU utilization
  - Memory usage
  - Training progress
  - Model accuracy metrics

- **Trading Computer:**
  - Latency monitoring
  - Memory usage
  - Trade execution speed
  - P&L tracking

### Log Management

```bash
# View recent logs
tail -f logs/trading.log
tail -f logs/training.log

# Archive old logs
backup_manager.bat
# Select "Create Logs Backup"
```

### Backup Strategy

```bash
# Daily backups (automated)
backup_manager.bat
# Select "Create Full Backup"

# Weekly model archives
package_models.bat

# Monthly system snapshots
backup_manager.bat
# Select "Create Full Backup" with custom name
```

## 🛠️ Troubleshooting

### Common Issues

#### Model Import Failures
```bash
# Check model integrity
validate_models.bat --verbose

# Verify file permissions
# Ensure model files are not corrupted
# Check Python version compatibility
```

#### Training Performance Issues
```bash
# Check system resources
monitor_system.bat

# Adjust batch size in config
# Reduce model complexity
# Enable GPU acceleration
```

#### Trading Bot Errors
```bash
# Check logs
type logs\trading.log | findstr ERROR

# Verify API connections
# Check model loading
# Validate configuration
```

### Recovery Procedures

#### Restore from Backup
```bash
# List available backups
backup_manager.bat
# Select "List All Backups"

# Restore specific backup
backup_manager.bat
# Select "Restore Backup"
```

#### Rollback Models
```bash
# Import previous model version
import_models.bat
# Select previous model package

# Validate rollback
validate_models.bat
```

## 📁 File Structure

```
Bot_kilo/
├── config/                     # Configuration files
│   ├── config_training.yaml     # Training computer config
│   ├── config_trading.yaml      # Trading computer config
│   └── environment_info.json    # Environment detection
├── src/                        # Source code
│   ├── config/                 # Configuration management
│   ├── models/                 # Model definitions
│   ├── data/                   # Data processing
│   └── trading/                # Trading logic
├── scripts/                    # Utility scripts
│   ├── train_models.py         # Model training
│   ├── package_models.py       # Model packaging
│   ├── import_models.py        # Model importing
│   ├── validate_models.py      # Model validation
│   └── backup_system.py        # Backup management
├── models/                     # Trained models
├── exports/                    # Model packages
├── backups/                    # System backups
├── logs/                       # Log files
└── *.bat                       # User-friendly scripts
```

## 🔐 Security Considerations

### API Keys
- Store API keys in environment variables
- Never commit keys to version control
- Use separate keys for training and trading
- Regularly rotate API keys

### Model Security
- Encrypt model packages during transfer
- Verify model checksums
- Use secure transfer methods
- Maintain model version history

### Network Security
- Use VPN for remote access
- Secure network shares
- Monitor network traffic
- Regular security updates

## 📈 Performance Optimization

### Training Computer
- Use GPU acceleration when available
- Optimize batch sizes for memory
- Parallel data processing
- Efficient feature engineering

### Trading Computer
- Minimize memory usage
- Optimize model loading
- Efficient data structures
- Real-time performance monitoring

## 🆘 Support and Resources

### Documentation
- `README_DISTRIBUTED_TRAINING.md` - Detailed technical documentation
- `QUICK_SETUP_GUIDE.md` - Step-by-step setup instructions
- `API_DOCUMENTATION.md` - API reference

### Scripts and Tools
- `configure_environment.bat` - Environment setup
- `train_models.bat` - Model training
- `package_models.bat` - Model packaging
- `import_models.bat` - Model importing
- `validate_models.bat` - Model validation
- `paper_trading.bat` - Trading bot
- `backup_manager.bat` - Backup management

### Monitoring Tools
- MLflow for experiment tracking
- Custom performance dashboards
- Log analysis tools
- System resource monitors

## 🎯 Best Practices

1. **Always backup before major changes**
2. **Validate models before deployment**
3. **Monitor system resources continuously**
4. **Keep detailed logs of all operations**
5. **Test in paper trading before live trading**
6. **Regular model retraining schedule**
7. **Maintain multiple model versions**
8. **Document configuration changes**
9. **Use version control for code changes**
10. **Regular security audits**

---

**Happy Trading! 🚀**

For additional support or questions, refer to the documentation or check the logs for detailed error information.