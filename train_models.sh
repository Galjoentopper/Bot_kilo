#!/bin/bash
# ============================================================================
# Enhanced Model Training Script for Trading Bot (Linux)
# ============================================================================
# This script trains models and packages them for easy transfer to Windows trading computer
# Run this on your powerful Linux training computer
#
# Usage:
#   ./train_models.sh                    - Train all models with default settings
#   ./train_models.sh --symbols BTCEUR   - Train only BTCEUR models
#   ./train_models.sh --models lightgbm  - Train only LightGBM models
# ============================================================================

set -e  # Exit on any error

echo
echo "========================================"
echo "   Trading Bot Model Training (Linux)"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed or not in PATH"
    echo "Please install Python3 and try again"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "src/config/config.yaml" ]; then
    echo "ERROR: config.yaml not found. Are you in the Bot_kilo directory?"
    echo "Current directory: $(pwd)"
    echo "Please navigate to the Bot_kilo directory and run this script again"
    exit 1
fi

# Check if required directories exist
if [ ! -d "data" ]; then
    echo "ERROR: data directory not found"
    echo "Please ensure you have market data in the 'data' directory"
    exit 1
fi

# Create necessary directories
mkdir -p models
mkdir -p models/exports
mkdir -p mlruns

echo "Checking Python dependencies..."
if ! python3 -c "import numpy, pandas, yaml, lightgbm, torch" &> /dev/null; then
    echo "WARNING: Some Python dependencies may be missing"
    echo "Installing required packages..."
    pip3 install numpy pandas pyyaml lightgbm torch scikit-learn
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

echo
echo "Starting enhanced model training..."
echo "Training will include automatic model packaging for transfer to Windows"
echo

# Run the enhanced trainer with packaging enabled
python3 scripts/enhanced_trainer.py --package-models --create-transfer-bundle "$@"

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Training failed!"
    echo "Check the logs above for details"
    exit 1
fi

echo
echo "========================================"
echo "   Training Completed Successfully!"
echo "========================================"
echo
echo "Your trained models have been packaged and are ready for transfer."
echo "Look for the export directory in models/exports/"
echo
echo "Next steps:"
echo "1. Copy the entire export folder to your Windows trading computer"
echo "2. Run the import_models.py script on your Windows trading computer"
echo "3. Start paper trading with deploy_trading.bat"
echo
echo "Opening exports folder..."
if command -v xdg-open &> /dev/null; then
    xdg-open models/exports
elif command -v open &> /dev/null; then
    open models/exports
else
    echo "Export folder location: $(pwd)/models/exports"
fi

echo
echo "Training session complete!"
echo "Press Enter to continue..."
read