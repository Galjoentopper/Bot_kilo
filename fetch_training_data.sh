#!/bin/bash

# Fetch Training Data Script for Linux Training Computer
# ====================================================
# This script reads symbols from config/config_training.yaml and fetches
# historical data using bulk API calls and normal API calls for missing parts.
# Creates SQLite databases compatible with the training system.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config/config_training.yaml"
DATA_DIR="$SCRIPT_DIR/data"
STATE_FILE="$DATA_DIR/fetch_state.json"
LOG_FILE="$DATA_DIR/fetch_data.log"
BINANCE_API="https://api.binance.com/api/v3"
BULK_API="https://data.binance.vision/data/spot/monthly/klines"

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

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required commands
    for cmd in python3 pip3 curl jq sqlite3 yq; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Installing missing dependencies..."
        
        # Install missing dependencies
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            for dep in "${missing_deps[@]}"; do
                case $dep in
                    "yq")
                        sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
                        sudo chmod +x /usr/local/bin/yq
                        ;;
                    "jq")
                        sudo apt-get install -y jq
                        ;;
                    "sqlite3")
                        sudo apt-get install -y sqlite3
                        ;;
                    *)
                        sudo apt-get install -y "$dep"
                        ;;
                esac
            done
        elif command -v yum &> /dev/null; then
            for dep in "${missing_deps[@]}"; do
                case $dep in
                    "yq")
                        sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
                        sudo chmod +x /usr/local/bin/yq
                        ;;
                    *)
                        sudo yum install -y "$dep"
                        ;;
                esac
            done
        else
            log_error "Package manager not found. Please install: ${missing_deps[*]}"
            exit 1
        fi
    fi
    
    # Install Python dependencies
    log_info "Installing Python dependencies..."
    pip3 install --user pandas requests pyyaml sqlite3 || {
        log_error "Failed to install Python dependencies"
        exit 1
    }
    
    log_success "All dependencies checked/installed"
}

# Read symbols from config file
read_symbols() {
    log_info "Reading symbols from config file..."
    
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Extract symbols using yq
    SYMBOLS=$(yq eval '.data.symbols[]' "$CONFIG_FILE" | tr -d "'\"" | tr '\n' ' ')
    INTERVAL=$(yq eval '.data.interval' "$CONFIG_FILE" | tr -d "'\"")
    LOOKBACK_DAYS=$(yq eval '.data.lookback_days' "$CONFIG_FILE")
    
    if [ -z "$SYMBOLS" ]; then
        log_error "No symbols found in config file"
        exit 1
    fi
    
    log_success "Found symbols: $SYMBOLS"
    log_info "Interval: $INTERVAL, Lookback days: $LOOKBACK_DAYS"
}

# Create directory structure
setup_directories() {
    log_info "Setting up directory structure..."
    
    mkdir -p "$DATA_DIR"
    mkdir -p "$DATA_DIR/databases"
    mkdir -p "$DATA_DIR/temp"
    mkdir -p "$DATA_DIR/logs"
    
    # Initialize state file if it doesn't exist
    if [ ! -f "$STATE_FILE" ]; then
        echo '{}' > "$STATE_FILE"
    fi
    
    log_success "Directory structure created"
}

# Load state from file
load_state() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        echo '{}'
    fi
}

# Save state to file
save_state() {
    local state="$1"
    echo "$state" > "$STATE_FILE"
}

# Get symbol progress from state
get_symbol_progress() {
    local symbol="$1"
    local state=$(load_state)
    echo "$state" | jq -r ".\"${symbol}\" // {\"last_bulk_month\": null, \"last_api_timestamp\": null, \"total_records\": 0}"
}

# Update symbol progress in state
update_symbol_progress() {
    local symbol="$1"
    local progress="$2"
    local state=$(load_state)
    local updated_state=$(echo "$state" | jq ".\"${symbol}\" = $progress")
    save_state "$updated_state"
}

# Create database for symbol
create_database() {
    local symbol="$1"
    local db_path="$DATA_DIR/databases/${symbol,,}_${INTERVAL}.db"
    
    log_info "Creating database for $symbol..."
    
    sqlite3 "$db_path" <<EOF
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL,
    datetime TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    quote_volume REAL NOT NULL,
    trades INTEGER NOT NULL,
    taker_buy_base REAL NOT NULL,
    taker_buy_quote REAL NOT NULL,
    UNIQUE(timestamp)
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_datetime ON market_data(datetime);
EOF
    
    log_success "Database created for $symbol: $db_path"
}

# Download bulk data for a specific month
download_bulk_data() {
    local symbol="$1"
    local year="$2"
    local month="$3"
    local temp_file="$DATA_DIR/temp/${symbol}_${year}_${month}.zip"
    
    local filename="${symbol}-${INTERVAL}-${year}-$(printf "%02d" $month).zip"
    local url="$BULK_API/$symbol/$INTERVAL/$filename"
    
    log_info "Downloading bulk data: $filename"
    
    # Download with retry logic
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if curl -f -L -o "$temp_file" "$url" 2>/dev/null; then
            log_success "Downloaded: $filename"
            echo "$temp_file"
            return 0
        else
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                log_warning "Download failed, retrying ($retry/$max_retries)..."
                sleep 2
            fi
        fi
    done
    
    log_warning "Bulk data not available: $filename"
    return 1
}

# Process bulk data and insert into database
process_bulk_data() {
    local symbol="$1"
    local zip_file="$2"
    local db_path="$DATA_DIR/databases/${symbol,,}_${INTERVAL}.db"
    
    log_info "Processing bulk data for $symbol..."
    
    # Extract CSV from ZIP
    local csv_file="$DATA_DIR/temp/${symbol}_temp.csv"
    unzip -p "$zip_file" "*.csv" > "$csv_file" 2>/dev/null || {
        log_error "Failed to extract CSV from $zip_file"
        return 1
    }
    
    # Process CSV with Python
    python3 << EOF
import pandas as pd
import sqlite3
from datetime import datetime

try:
    # Read CSV
    df = pd.read_csv('$csv_file', header=None, names=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Process data
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                   'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Select final columns
    df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close',
            'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
    
    # Remove invalid data
    df = df.dropna()
    
    # Connect to database and insert
    conn = sqlite3.connect('$db_path')
    
    # Insert data, ignoring duplicates
    df.to_sql('market_data', conn, if_exists='append', index=False, method='ignore')
    
    conn.commit()
    conn.close()
    
    print(f"Inserted {len(df)} records")
    
except Exception as e:
    print(f"Error processing data: {e}")
    exit(1)
EOF
    
    # Clean up temp files
    rm -f "$csv_file" "$zip_file"
    
    log_success "Processed bulk data for $symbol"
}

# Fetch data using API
fetch_api_data() {
    local symbol="$1"
    local start_time="$2"
    local limit="${3:-1000}"
    local db_path="$DATA_DIR/databases/${symbol,,}_${INTERVAL}.db"
    
    log_info "Fetching API data for $symbol from $(date -d @$((start_time/1000)))..."
    
    # Fetch data with rate limiting
    local response=$(curl -s "$BINANCE_API/klines?symbol=$symbol&interval=$INTERVAL&startTime=$start_time&limit=$limit")
    
    if [ "$response" = "null" ] || [ -z "$response" ]; then
        log_warning "No API data received for $symbol"
        return 1
    fi
    
    # Process API response with Python
    python3 << EOF
import json
import pandas as pd
import sqlite3
from datetime import datetime

try:
    # Parse JSON response
    data = json.loads('$response')
    
    if not data:
        print("No data received")
        exit(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Process data
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                   'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Select final columns
    df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close',
            'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
    
    # Remove invalid data
    df = df.dropna()
    
    # Connect to database and insert
    conn = sqlite3.connect('$db_path')
    
    # Insert data, ignoring duplicates
    df.to_sql('market_data', conn, if_exists='append', index=False, method='ignore')
    
    conn.commit()
    conn.close()
    
    print(f"Inserted {len(df)} records")
    
except Exception as e:
    print(f"Error processing API data: {e}")
    exit(1)
EOF
    
    # Rate limiting - wait 100ms between API calls
    sleep 0.1
    
    log_success "Fetched API data for $symbol"
}

# Fetch data for a single symbol
fetch_symbol_data() {
    local symbol="$1"
    
    log_info "Starting data fetch for $symbol..."
    
    # Create database
    create_database "$symbol"
    
    # Get current progress
    local progress=$(get_symbol_progress "$symbol")
    local last_bulk_month=$(echo "$progress" | jq -r '.last_bulk_month // "null"')
    local last_api_timestamp=$(echo "$progress" | jq -r '.last_api_timestamp // "null"')
    
    # Calculate date range
    local end_date=$(date +%s)
    local start_date=$((end_date - (LOOKBACK_DAYS * 24 * 3600)))
    local start_year=$(date -d @$start_date +%Y)
    local start_month=$(date -d @$start_date +%m)
    local end_year=$(date -d @$end_date +%Y)
    local end_month=$(date -d @$end_date +%m)
    
    log_info "Fetching data from $(date -d @$start_date) to $(date -d @$end_date)"
    
    # Phase 1: Bulk data download
    log_info "Phase 1: Downloading bulk data..."
    local current_year=$start_year
    local current_month=$((10#$start_month))  # Remove leading zero
    local bulk_end_timestamp=0
    
    while [ $current_year -le $end_year ]; do
        if [ $current_year -eq $end_year ] && [ $current_month -gt $((10#$end_month)) ]; then
            break
        fi
        
        # Skip if already processed
        local month_key="${current_year}-$(printf "%02d" $current_month)"
        if [ "$last_bulk_month" != "null" ] && [ "$month_key" "<" "$last_bulk_month" ]; then
            log_info "Skipping already processed month: $month_key"
        else
            if download_bulk_data "$symbol" $current_year $current_month; then
                local zip_file="$DATA_DIR/temp/${symbol}_${current_year}_${current_month}.zip"
                if process_bulk_data "$symbol" "$zip_file"; then
                    # Update progress
                    local updated_progress=$(echo "$progress" | jq ".last_bulk_month = \"$month_key\"")
                    update_symbol_progress "$symbol" "$updated_progress"
                    progress="$updated_progress"
                    
                    # Calculate end timestamp for this month
                    local month_end_date=$(date -d "$current_year-$(printf "%02d" $current_month)-01 +1 month -1 day" +%s)
                    bulk_end_timestamp=$((month_end_date * 1000))
                fi
            fi
        fi
        
        # Move to next month
        current_month=$((current_month + 1))
        if [ $current_month -gt 12 ]; then
            current_month=1
            current_year=$((current_year + 1))
        fi
    done
    
    # Phase 2: API data for missing/recent data
    log_info "Phase 2: Fetching missing data via API..."
    
    # Determine start timestamp for API calls
    local api_start_timestamp
    if [ $bulk_end_timestamp -gt 0 ]; then
        api_start_timestamp=$bulk_end_timestamp
    elif [ "$last_api_timestamp" != "null" ]; then
        api_start_timestamp=$last_api_timestamp
    else
        api_start_timestamp=$((start_date * 1000))
    fi
    
    # Fetch recent data via API
    local current_timestamp=$api_start_timestamp
    local end_timestamp=$((end_date * 1000))
    
    while [ $current_timestamp -lt $end_timestamp ]; do
        if fetch_api_data "$symbol" $current_timestamp 1000; then
            # Update progress
            local updated_progress=$(echo "$progress" | jq ".last_api_timestamp = $current_timestamp")
            update_symbol_progress "$symbol" "$updated_progress"
            progress="$updated_progress"
            
            # Move forward by ~1000 intervals (adjust based on interval)
            case $INTERVAL in
                "1m") current_timestamp=$((current_timestamp + 60000 * 1000)) ;;
                "5m") current_timestamp=$((current_timestamp + 300000 * 1000)) ;;
                "15m") current_timestamp=$((current_timestamp + 900000 * 1000)) ;;
                "30m") current_timestamp=$((current_timestamp + 1800000 * 1000)) ;;
                "1h") current_timestamp=$((current_timestamp + 3600000 * 1000)) ;;
                "1d") current_timestamp=$((current_timestamp + 86400000 * 1000)) ;;
                *) current_timestamp=$((current_timestamp + 900000 * 1000)) ;;  # Default to 15m
            esac
        else
            log_warning "API fetch failed, moving to next batch..."
            current_timestamp=$((current_timestamp + 900000 * 1000))  # Skip forward
        fi
    done
    
    # Get final record count
    local db_path="$DATA_DIR/databases/${symbol,,}_${INTERVAL}.db"
    local record_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM market_data;")
    
    # Update final progress
    local final_progress=$(echo "$progress" | jq ".total_records = $record_count")
    update_symbol_progress "$symbol" "$final_progress"
    
    log_success "Completed data fetch for $symbol: $record_count records"
}

# Validate databases
validate_databases() {
    log_info "Validating databases..."
    
    local total_records=0
    local valid_databases=0
    
    for symbol in $SYMBOLS; do
        local db_path="$DATA_DIR/databases/${symbol,,}_${INTERVAL}.db"
        
        if [ -f "$db_path" ]; then
            local record_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM market_data;" 2>/dev/null || echo "0")
            local date_range=$(sqlite3 "$db_path" "SELECT MIN(datetime), MAX(datetime) FROM market_data;" 2>/dev/null || echo "N/A,N/A")
            
            if [ "$record_count" -gt 0 ]; then
                log_success "$symbol: $record_count records ($date_range)"
                total_records=$((total_records + record_count))
                valid_databases=$((valid_databases + 1))
            else
                log_warning "$symbol: No data found"
            fi
        else
            log_error "$symbol: Database not found"
        fi
    done
    
    log_success "Validation complete: $valid_databases databases, $total_records total records"
}

# Main execution
main() {
    echo -e "${BLUE}"
    echo "==========================================="
    echo "  Fetch Training Data Script"
    echo "  Linux Training Computer Data Collector"
    echo "==========================================="
    echo -e "${NC}"
    
    log_info "Starting data fetch process..."
    
    # Check dependencies
    check_dependencies
    
    # Read configuration
    read_symbols
    
    # Setup directories
    setup_directories
    
    # Fetch data for each symbol
    local symbol_count=0
    local total_symbols=$(echo $SYMBOLS | wc -w)
    
    for symbol in $SYMBOLS; do
        symbol_count=$((symbol_count + 1))
        log_info "Processing symbol $symbol_count/$total_symbols: $symbol"
        
        if fetch_symbol_data "$symbol"; then
            log_success "Completed $symbol ($symbol_count/$total_symbols)"
        else
            log_error "Failed to fetch data for $symbol"
        fi
        
        # Progress indicator
        local progress=$((symbol_count * 100 / total_symbols))
        echo -e "${BLUE}Progress: $progress% ($symbol_count/$total_symbols symbols completed)${NC}"
    done
    
    # Validate all databases
    validate_databases
    
    log_success "Data fetch process completed!"
    echo -e "${GREEN}"
    echo "==========================================="
    echo "  Data Fetch Complete!"
    echo "  Databases created in: $DATA_DIR/databases"
    echo "  Ready for training!"
    echo "==========================================="
    echo -e "${NC}"
}

# Handle script interruption
trap 'log_warning "Script interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"