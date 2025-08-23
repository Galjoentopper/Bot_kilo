#!/bin/bash
# ============================================================================
# Linux Training Environment Setup Script
# ============================================================================
# This script sets up the training environment on a Linux machine
# Run this once before training models
# ============================================================================

set -e  # Exit on any error

echo
echo "========================================"
echo "   Trading Bot Training Setup (Linux)"
echo "========================================"
echo

# Check if we're running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    echo "WARNING: Running as root is not recommended"
    echo "Consider running as a regular user"
    echo "Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled"
        exit 0
    fi
fi

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo "Cannot detect Linux distribution"
    OS="Unknown"
fi

echo "Detected OS: $OS"
echo

# Check if Python 3.8+ is available
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Found Python $PYTHON_VERSION"
    
    # Check if version is 3.8 or higher
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        echo "Python version is compatible"
    else
        echo "ERROR: Python 3.8 or higher is required"
        echo "Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    echo "ERROR: Python3 is not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is not installed"
    echo "Please install pip3 and try again"
    exit 1
fi

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "WARNING: git is not installed"
    echo "Git is recommended for version control"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Installing essential Python dependencies..."
    pip install numpy pandas pyyaml scikit-learn lightgbm torch torchvision torchaudio
    pip install matplotlib seaborn plotly
    pip install mlflow optuna
    pip install python-telegram-bot
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data
mkdir -p models
mkdir -p models/exports
mkdir -p mlruns
mkdir -p logs

# Set up configuration
if [ ! -f "src/config/config.yaml" ] && [ -f "src/config/config.yaml.example" ]; then
    echo "Setting up configuration from example..."
    cp src/config/config.yaml.example src/config/config.yaml
    echo "Please edit src/config/config.yaml with your settings"
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x train_models.sh
chmod +x setup_training_environment.sh

# Check GPU availability
echo
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    
    # Check if PyTorch can use CUDA
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        echo "PyTorch CUDA support: Available"
    else
        echo "PyTorch CUDA support: Not available"
        echo "Consider installing PyTorch with CUDA support for faster training"
    fi
else
    echo "No NVIDIA GPU detected. Training will use CPU."
fi

echo
echo "========================================"
echo "   Setup Complete!"
echo "========================================"
echo
echo "Your Linux training environment is ready!"
echo
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Configure your settings in src/config/config.yaml"
echo "3. Add your market data to the data/ directory"
echo "4. Run training: ./train_models.sh"
echo
echo "To activate the environment in future sessions:"
echo "  cd $(pwd)"
echo "  source venv/bin/activate"
echo