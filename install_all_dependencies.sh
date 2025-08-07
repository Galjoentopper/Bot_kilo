#!/bin/bash

# Comprehensive Installation Script for Bot_kilo Trading Bot
# This script installs all project dependencies including LightGBM with environment-specific optimizations

set -e  # Exit immediately if a command exits with a non-zero status

# Default values for command-line options
SKIP_SYSTEM=false
SKIP_PYTHON=false
SKIP_VERIFICATION=false
INSTALL_EXTRAS=false
INSTALL_EDITABLE=false
HELP=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress bar variables
TOTAL_STEPS=0
CURRENT_STEP=0

# Logging function
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

log_error() {
    echo -e "${RED}$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}" >&2
}

log_info() {
    echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

log_debug() {
    echo -e "${CYAN}$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

log_header() {
    echo -e "${PURPLE}$(date '+%Y-%m-%d %H:%M:%S') - $1${NC}"
}

# Progress bar function
show_progress() {
    local message="$1"
    CURRENT_STEP=$((CURRENT_STEP + 1))
    local percentage=$((CURRENT_STEP * 100 / TOTAL_STEPS))
    local bar_length=40
    local filled_length=$((percentage * bar_length / 100))
    local empty_length=$((bar_length - filled_length))
    
    local bar=""
    for ((i=0; i<filled_length; i++)); do
        bar="${bar}█"
    done
    for ((i=0; i<empty_length; i++)); do
        bar="${bar}░"
    done
    
    printf "\r[${bar}] ${percentage}%% - ${message}" 
}

# Update progress total
update_total_steps() {
    TOTAL_STEPS=$1
    CURRENT_STEP=0
}

# Function to display help
show_help() {
    echo "Bot_kilo Trading Bot Installation Script"
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message and exit"
    echo "  --skip-system           Skip system dependencies installation"
    echo "  --skip-python           Skip Python dependencies installation"
    echo "  --skip-verification     Skip verification steps"
    echo "  --install-extras        Install extra dependencies (dev, jupyter, gpu)"
    echo "  --editable              Install local package in editable mode"
    echo ""
    echo "Examples:"
    echo "  $0                      # Full installation"
    echo "  $0 --skip-system        # Skip system dependencies"
    echo "  $0 --install-extras     # Install everything including extras"
    echo "  $0 --help               # Show this help"
}

# Function to parse command-line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                HELP=true
                shift
                ;;
            --skip-system)
                SKIP_SYSTEM=true
                shift
                ;;
            --skip-python)
                SKIP_PYTHON=true
                shift
                ;;
            --skip-verification)
                SKIP_VERIFICATION=true
                shift
                ;;
            --install-extras)
                INSTALL_EXTRAS=true
                shift
                ;;
            --editable)
                INSTALL_EDITABLE=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Function to check if we're running on Paperspace Gradient
is_paperspace_gradient() {
    # Check for Paperspace-specific environment variables or file system markers
    if [[ -n "$PAPERSPACE_METRIC_WORKER_ID" ]] || [[ -d "/notebooks" ]] || [[ -f "/etc/paperspace-release" ]]; then
        return 0
    else
        return 1
    fi
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_dependencies() {
    if [[ "$SKIP_SYSTEM" == true ]]; then
        log_info "Skipping system dependencies installation as requested"
        return 0
    fi
    
    show_progress "Installing system dependencies..."
    log_info "Installing system dependencies..."
    
    local os_type=$(detect_os)
    
    if [[ "$os_type" == "linux" ]]; then
        # Check if we're on Ubuntu/Debian or CentOS/RHEL
        if command -v apt-get &> /dev/null; then
            log_info "Detected Debian/Ubuntu system"
            sudo apt-get update
            
            # Install build tools and libraries
            sudo apt-get install -y \
                build-essential \
                cmake \
                git \
                wget \
                curl \
                libboost-dev \
                libboost-system-dev \
                libboost-filesystem-dev \
                libomp-dev \
                pkg-config \
                python3-dev \
                python3-pip \
                python3-venv
                
        elif command -v yum &> /dev/null; then
            log_info "Detected CentOS/RHEL system"
            sudo yum update -y
            
            # Install build tools and libraries
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                cmake \
                git \
                wget \
                curl \
                boost-devel \
                libomp-devel \
                python3-devel \
                python3-pip
        else
            log_warning "Unknown Linux distribution. Attempting generic installation."
            log_warning "You may need to manually install build tools and dependencies."
        fi
        
    elif [[ "$os_type" == "macos" ]]; then
        log_info "Detected macOS system"
        if command -v brew &> /dev/null; then
            brew install cmake boost libomp
        else
            log_error "Homebrew not found. Please install Homebrew first: https://brew.sh/"
            exit 1
        fi
        
    else
        log_warning "Unsupported OS. You may need to manually install system dependencies."
    fi
    
    log_success "System dependencies installed successfully"
}

# Special function to install LightGBM with multiple fallback methods
install_lightgbm_with_fallbacks() {
    show_progress "Installing LightGBM with fallback methods..."
    log_info "Installing LightGBM with fallback methods..."
    
    # Method 1: Direct pip install
    log_info "Method 1: Direct pip install"
    if python -m pip install --timeout 300 --retries 3 lightgbm>=4.0.0; then
        log_success "LightGBM installed successfully via direct pip install"
        return 0
    else
        log_warning "Failed to install LightGBM via direct pip install"
    fi
    
    # Method 2: Install with alternative index
    log_info "Method 2: Install with alternative index"
    if python -m pip install --index-url https://pypi.org/simple/ --timeout 300 --retries 3 lightgbm>=4.0.0; then
        log_success "LightGBM installed successfully via alternative index"
        return 0
    else
        log_warning "Failed to install LightGBM via alternative index"
    fi
    
    # Method 3: Install with conda (if available)
    log_info "Method 3: Install with conda (if available)"
    if command -v conda &> /dev/null; then
        if conda install -c conda-forge lightgbm -y; then
            log_success "LightGBM installed successfully via conda"
            return 0
        else
            log_warning "Failed to install LightGBM via conda"
        fi
    else
        log_warning "Conda not found, skipping conda installation method"
    fi
    
    # Method 4: Build from source (Paperspace-specific optimizations)
    log_info "Method 4: Building LightGBM from source"
    if build_lightgbm_from_source; then
        log_success "LightGBM built from source successfully"
        return 0
    else
        log_error "Failed to build LightGBM from source"
        return 1
    fi
}

# Function to build LightGBM from source
build_lightgbm_from_source() {
    show_progress "Building LightGBM from source..."
    log_info "Building LightGBM from source..."
    
    # Create temporary directory for building
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    # Clone LightGBM repository
    if ! git clone --recursive https://github.com/microsoft/LightGBM; then
        log_error "Failed to clone LightGBM repository"
        cd - > /dev/null
        rm -rf "$temp_dir"
        return 1
    fi
    
    cd LightGBM
    
    # Create build directory
    mkdir build
    cd build
    
    # Configure build with optimizations
    if ! cmake .. -DUSE_OPENMP=ON -DUSE_MPI=OFF; then
        log_error "Failed to configure LightGBM build"
        cd - > /dev/null
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Build LightGBM
    if ! make -j4; then
        log_error "Failed to build LightGBM"
        cd - > /dev/null
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Install Python package
    cd ../python-package
    if ! python setup.py install; then
        log_error "Failed to install LightGBM Python package"
        cd - > /dev/null
        rm -rf "$temp_dir"
        return 1
    fi
    
    # Clean up
    cd - > /dev/null
    rm -rf "$temp_dir"
    
    return 0
}

# Function to create and activate virtual environment
setup_virtual_environment() {
    show_progress "Setting up virtual environment..."
    log_info "Setting up virtual environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    log_success "Virtual environment activated"
}

# Function to install Python dependencies with multiple fallback methods
install_python_dependencies() {
    if [[ "$SKIP_PYTHON" == true ]]; then
        log_info "Skipping Python dependencies installation as requested"
        return 0
    fi
    
    show_progress "Installing Python dependencies..."
    log_info "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip first
    python -m pip install --upgrade pip
    
    # Install editable package if requested
    if [[ "$INSTALL_EDITABLE" == true ]]; then
        log_info "Installing local package in editable mode..."
        python -m pip install --timeout 300 --retries 3 -e .
    fi
    
    # Method 1: Install from requirements.txt with timeout and retries
    log_info "Method 1: Installing from requirements.txt"
    if python -m pip install --timeout 300 --retries 3 -r requirements.txt; then
        log_success "Successfully installed dependencies from requirements.txt"
        return 0
    else
        log_warning "Failed to install from requirements.txt, trying alternative methods..."
    fi
    
    # Method 2: Install core dependencies individually with fallback sources
    log_info "Method 2: Installing core dependencies individually"
    
    # Core dependencies
    show_progress "Installing core dependencies..."
    python -m pip install --timeout 300 --retries 3 python-dotenv>=1.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 python-dotenv>=1.0.0
    
    python -m pip install --timeout 300 --retries 3 pandas>=2.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 pandas>=2.0.0
    
    python -m pip install --timeout 300 --retries 3 "numpy>=1.24.0,<2.0.0" || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 "numpy>=1.24.0,<2.0.0"
    
    python -m pip install --timeout 300 --retries 3 PyYAML>=6.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 PyYAML>=6.0
    
    # Machine Learning dependencies
    show_progress "Installing machine learning dependencies..."
    python -m pip install --timeout 300 --retries 3 torch>=2.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 torch>=2.0.0
    
    python -m pip install --timeout 300 --retries 3 torchvision>=0.15.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 torchvision>=0.15.0
    
    python -m pip install --timeout 300 --retries 3 torchaudio>=2.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 torchaudio>=2.0.0
    
    # Special handling for LightGBM with multiple fallbacks
    install_lightgbm_with_fallbacks
    
    python -m pip install --timeout 300 --retries 3 scikit-learn>=1.3.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 scikit-learn>=1.3.0
    
    python -m pip install --timeout 300 --retries 3 scipy>=1.10.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 scipy>=1.10.0
    
    # Reinforcement Learning dependencies
    show_progress "Installing reinforcement learning dependencies..."
    python -m pip install --timeout 300 --retries 3 "stable-baselines3[extra]>=2.0.0" || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 "stable-baselines3[extra]>=2.0.0"
    
    python -m pip install --timeout 300 --retries 3 gymnasium>=0.29.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 gymnasium>=0.29.0
    
    # Data processing and APIs
    show_progress "Installing data processing and API dependencies..."
    python -m pip install --timeout 300 --retries 3 aiohttp>=3.8.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 aiohttp>=3.8.0
    
    python -m pip install --timeout 300 --retries 3 aiosqlite>=0.19.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 aiosqlite>=0.19.0
    
    python -m pip install --timeout 300 --retries 3 psutil>=5.9.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 psutil>=5.9.0
    
    python -m pip install --timeout 300 --retries 3 requests>=2.31.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 requests>=2.31.0
    
    python -m pip install --timeout 300 --retries 3 httpx>=0.24.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 httpx>=0.24.0
    
    # Trading and APIs
    show_progress "Installing trading and API dependencies..."
    python -m pip install --timeout 300 --retries 3 python-binance>=1.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 python-binance>=1.0.0
    
    python -m pip install --timeout 300 --retries 3 ccxt>=4.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 ccxt>=4.0.0
    
    python -m pip install --timeout 300 --retries 3 yfinance>=0.2.18 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 yfinance>=0.2.18
    
    # Notifications
    show_progress "Installing notification dependencies..."
    python -m pip install --timeout 300 --retries 3 python-telegram-bot>=20.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 python-telegram-bot>=20.0
    
    # Experiment tracking
    show_progress "Installing experiment tracking dependencies..."
    python -m pip install --timeout 300 --retries 3 mlflow>=2.5.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 mlflow>=2.5.0
    
    python -m pip install --timeout 300 --retries 3 wandb>=0.15.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 wandb>=0.15.0
    
    # Visualization
    show_progress "Installing visualization dependencies..."
    python -m pip install --timeout 300 --retries 3 matplotlib>=3.7.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 matplotlib>=3.7.0
    
    python -m pip install --timeout 300 --retries 3 seaborn>=0.12.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 seaborn>=0.12.0
    
    python -m pip install --timeout 300 --retries 3 plotly>=5.15.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 plotly>=5.15.0
    
    # Progress bars and CLI
    show_progress "Installing progress bar and CLI dependencies..."
    python -m pip install --timeout 300 --retries 3 tqdm>=4.65.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 tqdm>=4.65.0
    
    python -m pip install --timeout 300 --retries 3 rich>=13.4.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 rich>=13.4.0
    
    python -m pip install --timeout 300 --retries 3 click>=8.1.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 click>=8.1.0
    
    # Utilities
    show_progress "Installing utility dependencies..."
    python -m pip install --timeout 300 --retries 3 joblib>=1.3.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 joblib>=1.3.0
    
    python -m pip install --timeout 300 --retries 3 schedule>=1.2.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 schedule>=1.2.0
    
    python -m pip install --timeout 300 --retries 3 python-dateutil>=2.8.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 python-dateutil>=2.8.0
    
    python -m pip install --timeout 300 --retries 3 pytz>=2023.3 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 pytz>=2023.3
    
    python -m pip install --timeout 300 --retries 3 arrow>=1.2.3 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 arrow>=1.2.3
    
    # Configuration management
    show_progress "Installing configuration management dependencies..."
    python -m pip install --timeout 300 --retries 3 pydantic>=2.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 pydantic>=2.0.0
    
    python -m pip install --timeout 300 --retries 3 environs>=9.5.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 environs>=9.5.0
    
    # Time series and financial analysis
    show_progress "Installing time series and financial analysis dependencies..."
    python -m pip install --timeout 300 --retries 3 ta>=0.10.2 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 ta>=0.10.2
    
    python -m pip install --timeout 300 --retries 3 pandas-ta>=0.3.14b0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 pandas-ta>=0.3.14b0
    
    python -m pip install --timeout 300 --retries 3 statsmodels>=0.14.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 statsmodels>=0.14.0
    
    # Database support
    show_progress "Installing database dependencies..."
    python -m pip install --timeout 300 --retries 3 sqlalchemy>=2.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 sqlalchemy>=2.0.0
    
    # Better JSON handling
    show_progress "Installing JSON handling dependencies..."
    python -m pip install --timeout 300 --retries 3 orjson>=3.9.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 orjson>=3.9.0
    
    # WebSocket support
    show_progress "Installing WebSocket dependencies..."
    python -m pip install --timeout 300 --retries 3 websockets>=11.0.3 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 websockets>=11.0.3
    
    # Better error handling
    show_progress "Installing error handling dependencies..."
    python -m pip install --timeout 300 --retries 3 tenacity>=8.2.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 tenacity>=8.2.0
    
    # Hyperparameter optimization
    show_progress "Installing hyperparameter optimization dependencies..."
    python -m pip install --timeout 300 --retries 3 optuna>=3.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 optuna>=3.0.0
    
    # Development and testing
    if [[ "$INSTALL_EXTRAS" == true ]]; then
        show_progress "Installing development and testing dependencies..."
        python -m pip install --timeout 300 --retries 3 pytest>=7.4.0 || \
            python -m pip install --index-url https://pypi.org/simple/ --timeout 300 pytest>=7.4.0
        
        python -m pip install --timeout 300 --retries 3 pytest-asyncio>=0.21.0 || \
            python -m pip install --index-url https://pypi.org/simple/ --timeout 300 pytest-asyncio>=0.21.0
        
        python -m pip install --timeout 300 --retries 3 pytest-cov>=4.1.0 || \
            python -m pip install --index-url https://pypi.org/simple/ --timeout 300 pytest-cov>=4.1.0
        
        python -m pip install --timeout 300 --retries 3 black>=23.0.0 || \
            python -m pip install --index-url https://pypi.org/simple/ --timeout 300 black>=23.0.0
        
        python -m pip install --timeout 300 --retries 3 flake8>=6.0.0 || \
            python -m pip install --index-url https://pypi.org/simple/ --timeout 300 flake8>=6.0.0
    fi
    
    # Jupyter notebook support
    if [[ "$INSTALL_EXTRAS" == true ]]; then
        show_progress "Installing Jupyter notebook dependencies..."
        python -m pip install --timeout 300 --retries 3 jupyter>=1.0.0 || \
            python -m pip install --index-url https://pypi.org/simple/ --timeout 300 jupyter>=1.0.0
        
        python -m pip install --timeout 300 --retries 3 ipykernel>=6.0.0 || \
            python -m pip install --index-url https://pypi.org/simple/ --timeout 300 ipykernel>=6.0.0
    fi
    
    # Docker support
    show_progress "Installing Docker dependencies..."
    python -m pip install --timeout 300 --retries 3 docker>=6.0.0 || \
        python -m pip install --index-url https://pypi.org/simple/ --timeout 300 docker>=6.0.0
    
    # GPU support
    if [[ "$INSTALL_EXTRAS" == true ]]; then
        show_progress "Installing GPU dependencies..."
        python -m pip install --timeout 300 --retries 3 torchaudio>=2.0.0 || \
            python -m pip install --index-url https://pypi.org/simple/ --timeout 300 torchaudio>=2.0.0
    fi
    
    log_success "Python dependencies installed successfully"
}

# Function to verify installation
verify_installation() {
    if [[ "$SKIP_VERIFICATION" == true ]]; then
        log_info "Skipping verification as requested"
        return 0
    fi
    
    show_progress "Verifying installation..."
    log_info "Verifying installation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test importing key modules
    # Note: python-binance package is imported as "binance"
    local modules=("pandas" "numpy" "lightgbm" "torch" "sklearn" "ccxt" "binance")
    
    for module in "${modules[@]}"; do
        if python -c "import $module; print('$module imported successfully')" &> /dev/null; then
            log_success "$module verified successfully"
        else
            log_error "Failed to import $module"
            return 1
        fi
    done
    
    # Special verification for LightGBM
    if python -c "import lightgbm as lgb; print('LightGBM version:', lgb.__version__)" &> /dev/null; then
        log_success "LightGBM verified successfully"
    else
        log_error "Failed to verify LightGBM"
        return 1
    fi
    
    # Run a simple test with LightGBM
    if python -c "
import lightgbm as lgb
import numpy as np
X = np.random.rand(100, 10)
y = np.random.rand(100)
train_data = lgb.Dataset(X, label=y)
params = {'objective': 'regression', 'verbose': -1}
model = lgb.train(params, train_data, num_boost_round=10)
print('LightGBM basic functionality test passed')
" &> /dev/null; then
        log_success "LightGBM basic functionality test passed"
    else
        log_error "LightGBM basic functionality test failed"
        return 1
    fi
    
    log_success "All verifications passed"
    return 0
}

# Function to clean up temporary files
cleanup() {
    show_progress "Cleaning up temporary files..."
    log_info "Cleaning up temporary files..."
    # Add any cleanup steps here if needed
    log_success "Cleanup completed"
}

# Main installation function
main() {
    # Parse command-line arguments
    parse_arguments "$@"
    
    # Show help if requested
    if [[ "$HELP" == true ]]; then
        show_help
        exit 0
    fi
    
    log_header "Starting comprehensive installation for Bot_kilo trading bot..."
    
    # Determine total steps for progress bar
    local steps=5
    if [[ "$SKIP_SYSTEM" == false ]]; then
        steps=$((steps + 1))
    fi
    if [[ "$SKIP_PYTHON" == false ]]; then
        steps=$((steps + 1))
    fi
    if [[ "$SKIP_VERIFICATION" == false ]]; then
        steps=$((steps + 1))
    fi
    update_total_steps $steps
    
    # Check if we're on Paperspace Gradient
    if is_paperspace_gradient; then
        log_info "Detected Paperspace Gradient environment"
        log_info "Applying Paperspace-specific optimizations..."
    else
        log_info "Standard environment detected"
    fi
    
    # Install system dependencies
    if ! install_system_dependencies; then
        log_error "Failed to install system dependencies"
        exit 1
    fi
    
    # Setup virtual environment
    setup_virtual_environment
    
    # Install Python dependencies
    if ! install_python_dependencies; then
        log_error "Failed to install Python dependencies"
        exit 1
    fi
    
    # Verify installation
    if ! verify_installation; then
        log_error "Installation verification failed"
        exit 1
    fi
    
    # Cleanup
    cleanup
    
    log_success "Comprehensive installation completed successfully!"
    log_info "To activate the environment, run: source venv/bin/activate"
}

# Run main function with all arguments
main "$@"