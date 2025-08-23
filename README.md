# Bot Kilo - Crypto Trading Bot Training System

A comprehensive cryptocurrency trading bot with machine learning models and automated training capabilities.

## Quick Start

### 1. Setup Environment
```bash
# On Linux/Mac
./setup_training_environment.sh

# On Windows
.\setup_training_environment.bat
```

### 2. Get Market Data
```bash
# On Linux/Mac
./fetch_training_data.sh

# On Windows PowerShell
.\fetch_training_data.ps1
```

### 3. Start Training
```bash
# Basic training (all models, all symbols)
./train_models.sh

# Resume from checkpoint (handles 6-hour limits)
./train_models.sh --resume

# Custom training
./train_models.sh --symbols BTCEUR,ETHEUR --models ppo,lightgbm
```

## Memory Management & Crash Prevention

**Important**: The system now includes automatic memory cleanup to prevent PC crashes during PPO training:

- **Automatic Environment Cleanup**: PPO environments are properly closed after each model
- **CUDA Memory Management**: GPU memory is cleared between training sessions
- **Garbage Collection**: Forced cleanup after each model completion
- **Checkpoint System**: Training state is saved every model completion

## Checkpoint System

The training system automatically saves progress and can resume from interruptions:

- **Auto-save**: Progress saved after each model completion
- **Resume**: Use `--resume` flag to continue from last checkpoint
- **6-hour Limit Handling**: Automatically resumes on remote systems with time limits
- **Graceful Shutdown**: Ctrl+C saves checkpoint before exit

## Training Options

```bash
# Available flags
--symbols BTCEUR,ETHEUR,LTCEUR    # Specific symbols
--models ppo,lightgbm,gru         # Specific models  
--resume                          # Resume from checkpoint
--package-models                  # Create transfer bundle
--experiment-name my_experiment   # Custom experiment name
```

## Model Transfer

After training, use the generated transfer bundle:

1. Training creates `models/exports/transfer_bundle_YYYYMMDD_HHMMSS.zip`
2. Copy to target machine
3. Run the included `import_models.py` script

## Troubleshooting

**PC Crashes During Training**: 
- The new memory cleanup system prevents this
- If crashes still occur, reduce batch sizes in `src/config/config.yaml`

**Training Interrupted**:
- Use `./train_models.sh --resume` to continue
- Check `checkpoints/` directory for saved progress

**Out of Memory**:
- Reduce `n_steps` and `batch_size` for PPO models
- Close other applications during training

## File Structure

```
Bot_kilo/
├── data/              # Market data databases
├── models/            # Trained models
├── checkpoints/       # Training progress saves
├── logs/              # Training logs
├── scripts/           # Training scripts
└── src/               # Source code
```

## Remote Training

For training on remote machines with time limits:

1. Start training: `./train_models.sh`
2. When session ends, restart with: `./train_models.sh --resume`
3. Repeat until all models complete
4. Download the transfer bundle from `models/exports/`

The checkpoint system ensures no progress is lost between sessions.