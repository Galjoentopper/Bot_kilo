#!/bin/bash

# Streamlined Fetch Training Data Script for Linux Training Computer
# ==================================================================
# This script uses the config_based_collector.py to fetch training data
# based on configuration from config/config_training.yaml

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/config_training.yaml"
DATA_DIR="$SCRIPT_DIR/data"
COLLECTOR_SCRIPT="$DATA_DIR/config_based_collector.py"
LOG_FILE="$DATA_DIR/fetch_data.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Print usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -s, --symbol SYMBOL Collect data for specific symbol only"
    echo "  -c, --config FILE   Use specific config file (default: config/config_training.yaml)"
    echo "  -u, --update        Update mode: only fetch recent data"
    echo "  -f, --full          Full collection mode (default)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Collect all symbols from config"
    echo "  $0 -s BTCEUR        # Collect only BTCEUR data"
    echo "  $0 -c custom.yaml   # Use custom config file"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check for Python 3
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check for pip3
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip3 install --user pandas requests pyyaml || {
        log_error "Failed to install Python dependencies"
        exit 1
    }
    
    log_success "Dependencies checked/installed"
}

# Validate config file
validate_config() {
    log_info "Validating configuration..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    if [ ! -r "$CONFIG_FILE" ]; then
        log_error "Config file is not readable: $CONFIG_FILE"
        exit 1
    fi
    
    log_success "Configuration file validated: $CONFIG_FILE"
}

# Validate collector script
validate_collector() {
    log_info "Validating collector script..."
    
    if [ ! -f "$COLLECTOR_SCRIPT" ]; then
        log_error "Collector script not found: $COLLECTOR_SCRIPT"
        exit 1
    fi
    
    if [ ! -r "$COLLECTOR_SCRIPT" ]; then
        log_error "Collector script is not readable: $COLLECTOR_SCRIPT"
        exit 1
    fi
    
    # Test if the script can be executed
    if ! python3 "$COLLECTOR_SCRIPT" --help &>/dev/null; then
        log_error "Collector script cannot be executed or has errors"
        exit 1
    fi
    
    log_success "Collector script validated: $COLLECTOR_SCRIPT"
}

# Run data collection
run_data_collection() {
    local mode="${1:-full}"
    local specific_symbol="$2"
    
    log_info "Starting data collection (mode: $mode)..."
    
    local cmd="python3 \"$COLLECTOR_SCRIPT\" --config \"$CONFIG_FILE\""
    
    # Add mode-specific parameters
    case "$mode" in
        "update")
            cmd="$cmd --update-only"
            ;;
        "symbol")
            if [ -n "$specific_symbol" ]; then
                cmd="$cmd --symbol \"$specific_symbol\""
            else
                log_error "Symbol mode requires a specific symbol"
                return 1
            fi
            ;;
        "full")
            # Default mode, no additional parameters
            ;;
        *)
            log_error "Unknown mode: $mode"
            return 1
            ;;
    esac
    
    log_info "Executing: $cmd"
    
    # Execute the collector
    if eval "$cmd"; then
        log_success "Data collection completed successfully"
        return 0
    else
        log_error "Data collection failed"
        return 1
    fi
}

# Show collection summary
show_summary() {
    log_info "Showing collection summary..."
    
    # Let the Python collector show its own summary
    if python3 "$COLLECTOR_SCRIPT" --config "$CONFIG_FILE" --summary 2>/dev/null; then
        log_success "Summary displayed"
    else
        log_warning "Could not display summary (collection may not have run yet)"
    fi
}

# Main execution
main() {
    local mode="full"
    local specific_symbol=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --mode)
                mode="$2"
                shift 2
                ;;
            --symbol)
                specific_symbol="$2"
                mode="symbol"
                shift 2
                ;;
            --summary)
                show_summary
                exit 0
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    echo -e "${BLUE}"
    echo "==========================================="
    echo "  Fetch Training Data Script"
    echo "  Streamlined Data Collector"
    echo "==========================================="
    echo -e "${NC}"
    
    log_info "Starting data fetch process..."
    
    # Validate configuration and collector
    validate_config
    validate_collector
    
    # Check dependencies
    check_dependencies
    
    # Run data collection
    if run_data_collection "$mode" "$specific_symbol"; then
        log_success "Data fetch process completed successfully!"
        echo -e "${GREEN}"
        echo "==========================================="
        echo "  Data Fetch Complete!"
        echo "  Check data/databases/ for SQLite files"
        echo "  Ready for training!"
        echo "==========================================="
        echo -e "${NC}"
    else
        log_error "Data fetch process failed!"
        exit 1
    fi
}

# Handle script interruption
trap 'log_warning "Script interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"