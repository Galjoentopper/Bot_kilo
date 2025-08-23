# Trading Bot - Model Training with Checkpoint System

## Quick Start

### For Linux Training Computer
```bash
# Make script executable
chmod +x train_models.sh

# Start training (with automatic checkpointing)
./train_models.sh

# Resume interrupted training
./train_models.sh --resume
```

### For Windows Training Computer
```cmd
# Start training (with automatic checkpointing)
train_models.bat

# Resume interrupted training
train_models.bat --resume
```

## Checkpoint System Features

### Automatic Checkpointing
- Training progress is automatically saved after each model completion
- Handles 6-hour runtime limits gracefully
- No training progress is lost during interruptions

### Resume Training
- Use `--resume` flag to continue from last checkpoint
- Automatically detects and loads the most recent checkpoint
- Skips already completed models and symbols

### Custom Options
```bash
# Linux examples
./train_models.sh --resume --checkpoint-dir custom_checkpoints
./train_models.sh --symbols BTCEUR --models lightgbm
./train_models.sh --auto-checkpoint --package-models

# Windows examples
train_models.bat --resume --checkpoint-dir custom_checkpoints
train_models.bat --symbols BTCEUR --models lightgbm
train_models.bat --auto-checkpoint --package-models
```

## Troubleshooting

### If Training Stops
1. Check the terminal output for any error messages
2. Resume with: `./train_models.sh --resume` (Linux) or `train_models.bat --resume` (Windows)
3. Checkpoint files are saved in the `checkpoints/` directory

### If Resume Fails
1. Check if checkpoint files exist in `checkpoints/` directory
2. Verify the checkpoint file is not corrupted
3. Start fresh training if needed (checkpoints will be recreated)

## File Structure
```
Bot_kilo/
├── checkpoints/          # Training checkpoint files
├── models/exports/       # Packaged models for transfer
├── train_models.sh       # Linux training script
├── train_models.bat      # Windows training script
└── scripts/enhanced_trainer.py  # Main training script
```

A robust checkpoint and resume system for handling 6-hour runtime limits on Linux systems.

## Features

- **Auto-save**: Automatically saves progress after each model completion
- **Resume capability**: Continue training from where it left off after interruption
- **Graceful shutdown**: Handles system signals to save state before shutdown
- **Progress tracking**: Tracks symbol index, model index, and completed models
- **Cleanup**: Automatically removes checkpoints after successful completion

## Usage

### Starting Fresh Training

```bash
python scripts/enhanced_trainer.py --config config/training_config.yaml
```

### Resuming from Checkpoint

```bash
python scripts/enhanced_trainer.py --config config/training_config.yaml --resume
```

### Custom Checkpoint Directory

```bash
python scripts/enhanced_trainer.py --config config/training_config.yaml --resume --checkpoint-dir /path/to/checkpoints
```

## How It Works

1. **Checkpoint Creation**: After each model completes training, the system saves:
   - Current progress (symbol/model indices)
   - List of completed models
   - Training configuration
   - Partial results and metadata

2. **Resume Detection**: On startup, the script checks for existing checkpoints
   - If found with `--resume` flag, continues from last saved state
   - If not found or no `--resume` flag, starts fresh training

3. **Graceful Shutdown**: Signal handlers (SIGTERM, SIGINT) ensure:
   - Current progress is saved before exit
   - No data loss during interruption

4. **Automatic Cleanup**: After successful completion:
   - All checkpoint files are removed
   - Only final trained models remain

## Files Modified

- `scripts/enhanced_trainer.py`: Main training script with checkpoint integration
- `src/utils/training_checkpoint.py`: Checkpoint utility classes

## Checkpoint Storage

- Default location: `./checkpoints/`
- Files: `checkpoint_latest.pkl` (binary format)
- Contains: Progress state, config hash, completed models list

## Error Handling

- Invalid checkpoints are automatically detected and ignored
- Configuration mismatches prevent resume (starts fresh)
- Corrupted checkpoint files trigger fresh start with warning

The system is designed to be robust and handle various failure scenarios while ensuring training can always continue from the last successful state.