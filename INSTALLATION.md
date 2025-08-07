# Comprehensive Installation Guide for Bot_kilo Trading Bot

This guide explains how to install all dependencies for the Bot_kilo trading bot using our new comprehensive installation script.

## Overview

The `install_all_dependencies.sh` script provides a unified installation workflow that handles all project dependencies, including LightGBM, with environment-specific optimizations. This replaces the previous standalone LightGBM installation approach.

## Prerequisites

- Linux, macOS, or Windows with WSL
- Python 3.8 or higher
- Git
- sudo access (on Linux systems)

## Installation Steps

1. **Make the script executable:**
   ```bash
   chmod +x install_all_dependencies.sh
   ```

2. **Run the installation script:**
   ```bash
   ./install_all_dependencies.sh
   ```

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

## What the Script Does

The installation script performs the following steps:

1. **Detects the environment** - Automatically identifies if running on Paperspace Gradient or a standard system
2. **Installs system dependencies** - Handles installation of build tools and libraries
3. **Sets up a virtual environment** - Creates and activates a Python virtual environment
4. **Installs all Python dependencies** - Installs everything from requirements.txt with multiple fallback methods
5. **Verifies the installation** - Runs tests to ensure all dependencies are properly installed
6. **Cleans up temporary files** - Removes any temporary files created during installation

## Environment-Specific Optimizations

### Paperspace Gradient
When the script detects a Paperspace Gradient environment, it applies specific optimizations:
- Uses Paperspace-specific installation paths
- Applies network timeout adjustments for better reliability
- Uses build optimizations suitable for Paperspace infrastructure

### Standard Environments
For standard Linux and macOS systems:
- Detects package managers (apt, yum, brew)
- Installs appropriate system dependencies
- Applies general optimization techniques

## Fallback Mechanisms

The script implements multiple fallback methods for dependency installation:

1. **Direct pip install** - Standard installation method
2. **Alternative package index** - Uses different PyPI mirrors if the default fails
3. **Conda installation** - Uses conda if available as an alternative
4. **Source compilation** - Builds from source if all else fails

## Verification Process

After installation, the script automatically verifies that key dependencies are properly installed:
- Tests importing core modules (pandas, numpy, torch, etc.)
- Specifically verifies LightGBM installation and functionality
- Runs a basic functionality test with LightGBM

## Troubleshooting

### Common Issues

1. **Permission errors:**
   ```bash
   sudo ./install_all_dependencies.sh
   ```

2. **Network timeouts:**
   The script automatically increases timeout values, but you can manually adjust them if needed.

3. **Missing system dependencies:**
   The script attempts to install system dependencies automatically, but you may need to install them manually on some systems.

### Manual Installation

If the script fails, you can manually install dependencies:

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Removing Previous Installation Files

This new approach replaces the following files which are no longer needed:
- `install_lightgbm_paperspace.sh`
- `LIGHTGBM_INSTALLATION_PAPERSPACE.md`

These files have been removed from the project as they are no longer necessary with the comprehensive installation script.

## Updating Dependencies

To update dependencies after the initial installation:

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Update dependencies:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## Contributing

If you encounter issues with the installation process, please open an issue with:
- Your operating system and version
- The error message you received
- Steps to reproduce the issue