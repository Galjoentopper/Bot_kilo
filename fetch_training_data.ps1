#!/usr/bin/env pwsh
# PowerShell version of fetch_training_data.sh for Windows systems
# Streamlined data collection orchestrator for Binance training data

# Configuration
$CONFIG_FILE = "src/config/config_training.yaml"
$COLLECTOR_SCRIPT = "data/config_based_collector.py"
$DATA_DIR = "data"

# Colors for output
$RED = "`e[31m"
$GREEN = "`e[32m"
$YELLOW = "`e[33m"
$BLUE = "`e[34m"
$NC = "`e[0m"  # No Color

# Logging functions
function Write-LogInfo {
    param($Message)
    Write-Host "${BLUE}[INFO]${NC} $Message"
}

function Write-LogSuccess {
    param($Message)
    Write-Host "${GREEN}[SUCCESS]${NC} $Message"
}

function Write-LogWarning {
    param($Message)
    Write-Host "${YELLOW}[WARNING]${NC} $Message"
}

function Write-LogError {
    param($Message)
    Write-Host "${RED}[ERROR]${NC} $Message"
}

# Show usage information
function Show-Usage {
    Write-Host ""
    Write-Host "${BLUE}Fetch Training Data - PowerShell Version${NC}"
    Write-Host "Streamlined data collection orchestrator for Binance training data"
    Write-Host ""
    Write-Host "Usage: .\fetch_training_data.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  --config FILE     Use specific config file (default: config/config_training.yaml)"
    Write-Host "  --mode MODE       Collection mode: full, update, symbol (default: full)"
    Write-Host "  --symbol SYMBOL   Collect data for specific symbol only (requires --mode symbol)"
    Write-Host "  --summary         Show collection summary only"
    Write-Host "  --help            Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\fetch_training_data.ps1                    # Full collection using default config"
    Write-Host "  .\fetch_training_data.ps1 --mode update      # Update existing data"
    Write-Host "  .\fetch_training_data.ps1 --symbol BTCUSDT   # Collect specific symbol"
    Write-Host "  .\fetch_training_data.ps1 --summary          # Show summary only"
    Write-Host ""
}

# Validate configuration file
function Test-Config {
    param($ConfigPath)
    
    if (-not (Test-Path $ConfigPath)) {
        Write-LogError "Configuration file not found: $ConfigPath"
        Write-LogError "Please ensure the config file exists and is readable."
        return $false
    }
    
    Write-LogSuccess "Configuration file found: $ConfigPath"
    return $true
}

# Validate collector script
function Test-Collector {
    param($CollectorPath)
    
    if (-not (Test-Path $CollectorPath)) {
        Write-LogError "Collector script not found: $CollectorPath"
        Write-LogError "Please ensure the Python collector script exists."
        return $false
    }
    
    Write-LogSuccess "Collector script found: $CollectorPath"
    return $true
}

# Check dependencies
function Test-Dependencies {
    Write-LogInfo "Checking dependencies..."
    
    # Check Python
    try {
        $pythonVersion = python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "Python found: $pythonVersion"
        } else {
            throw "Python not found"
        }
    } catch {
        Write-LogError "Python is not installed or not in PATH"
        Write-LogError "Please install Python 3.7+ and ensure it's in your PATH"
        return $false
    }
    
    # Check pip
    try {
        $pipVersion = pip --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "pip found: $pipVersion"
        } else {
            throw "pip not found"
        }
    } catch {
        Write-LogError "pip is not installed or not in PATH"
        return $false
    }
    
    # Install required Python packages
    Write-LogInfo "Installing required Python packages..."
    $packages = @("pandas", "requests", "pyyaml")
    
    foreach ($package in $packages) {
        try {
            pip install $package --quiet
            if ($LASTEXITCODE -eq 0) {
                Write-LogSuccess "Package installed: $package"
            } else {
                throw "Failed to install $package"
            }
        } catch {
            Write-LogWarning "Failed to install $package, it might already be installed"
        }
    }
    
    return $true
}

# Run data collection
function Start-DataCollection {
    param(
        [string]$ConfigPath,
        [string]$Mode = "full",
        [string]$Symbol = ""
    )
    
    Write-LogInfo "Starting data collection..."
    Write-LogInfo "Mode: $Mode"
    if ($Symbol) {
        Write-LogInfo "Symbol: $Symbol"
    }
    
    # Build command arguments
    $args = @()
    $args += $COLLECTOR_SCRIPT
    $args += "--config"
    $args += $ConfigPath
    
    if ($Symbol) {
        $args += "--symbol"
        $args += $Symbol
    }
    
    # Execute Python collector
    try {
        Write-LogInfo "Executing: python $($args -join ' ')"
        & python @args
        
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "Data collection completed successfully!"
            return $true
        } else {
            Write-LogError "Data collection failed with exit code: $LASTEXITCODE"
            return $false
        }
    } catch {
        Write-LogError "Failed to execute data collection: $($_.Exception.Message)"
        return $false
    }
}

# Show collection summary
function Show-Summary {
    param([string]$ConfigPath)
    
    Write-LogInfo "Showing collection summary..."
    
    try {
        & python $COLLECTOR_SCRIPT --config $ConfigPath
        if ($LASTEXITCODE -eq 0) {
            return $true
        } else {
            Write-LogError "Failed to show summary"
            return $false
        }
    } catch {
        Write-LogError "Failed to show summary: $($_.Exception.Message)"
        return $false
    }
}

# Main function
function Main {
    param([string[]]$Arguments)
    
    # Parse command line arguments
    $ConfigPath = $CONFIG_FILE
    $Mode = "full"
    $Symbol = ""
    $ShowSummary = $false
    $ShowHelp = $false
    
    for ($i = 0; $i -lt $Arguments.Length; $i++) {
        switch ($Arguments[$i]) {
            "--config" {
                if ($i + 1 -lt $Arguments.Length) {
                    $ConfigPath = $Arguments[$i + 1]
                    $i++
                } else {
                    Write-LogError "--config requires a value"
                    return 1
                }
            }
            "--mode" {
                if ($i + 1 -lt $Arguments.Length) {
                    $Mode = $Arguments[$i + 1]
                    $i++
                } else {
                    Write-LogError "--mode requires a value"
                    return 1
                }
            }
            "--symbol" {
                if ($i + 1 -lt $Arguments.Length) {
                    $Symbol = $Arguments[$i + 1]
                    $Mode = "symbol"
                    $i++
                } else {
                    Write-LogError "--symbol requires a value"
                    return 1
                }
            }
            "--summary" {
                $ShowSummary = $true
            }
            "--help" {
                $ShowHelp = $true
            }
            default {
                Write-LogError "Unknown option: $($Arguments[$i])"
                Show-Usage
                return 1
            }
        }
    }
    
    # Show help if requested
    if ($ShowHelp) {
        Show-Usage
        return 0
    }
    
    # Header
    Write-Host ""
    Write-Host "${BLUE}===========================================${NC}"
    Write-Host "${BLUE}  Fetch Training Data Script (PowerShell)${NC}"
    Write-Host "${BLUE}  Windows Training Computer Data Collector${NC}"
    Write-Host "${BLUE}===========================================${NC}"
    Write-Host ""
    
    # Validate configuration and collector
    if (-not (Test-Config $ConfigPath)) {
        return 1
    }
    
    if (-not (Test-Collector $COLLECTOR_SCRIPT)) {
        return 1
    }
    
    # Check dependencies
    if (-not (Test-Dependencies)) {
        return 1
    }
    
    # Execute based on mode
    if ($ShowSummary) {
        if (-not (Show-Summary $ConfigPath)) {
            return 1
        }
    } else {
        if (-not (Start-DataCollection $ConfigPath $Mode $Symbol)) {
            return 1
        }
        
        # Show final summary
        Write-Host ""
        Write-LogInfo "Showing final summary..."
        Show-Summary $ConfigPath | Out-Null
    }
    
    # Success message
    Write-Host ""
    Write-Host "${GREEN}===========================================${NC}"
    Write-Host "${GREEN}  Data Collection Complete!${NC}"
    Write-Host "${GREEN}  Databases created in: $DATA_DIR${NC}"
    Write-Host "${GREEN}  Ready for training!${NC}"
    Write-Host "${GREEN}===========================================${NC}"
    Write-Host ""
    
    return 0
}

# Handle script interruption
trap {
    Write-LogWarning "Script interrupted by user"
    exit 1
}

# Run main function with all arguments
$exitCode = Main $args
exit $exitCode