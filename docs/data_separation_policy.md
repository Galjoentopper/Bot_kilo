# Data Separation Policy

## Overview

This document outlines the policy for separating live trading data from training dataset storage in the crypto trading bot system.

## Data Separation Principles

### Live Trading Data (Real-Time)
- Uses ccxt library to fetch live market data directly from exchanges
- Data is processed in-memory and not stored locally
- Features are generated on-the-fly for model predictions
- No persistent storage of live trading data in local databases

### Training Dataset Storage (Historical)
- Uses local SQLite databases in the `./data` directory
- Data is collected separately by dedicated data collection scripts
- Historical data is used exclusively for model training and backtesting
- Data collection is performed by `data/binance_data_collection.py`

## Implementation Details

### Live Trading Data Flow
1. Trading bot initializes ccxt exchange connection
2. Fetches latest OHLCV data for all configured symbols
3. Processes data in-memory for feature generation
4. Makes trading decisions based on model predictions
5. Executes trades without storing data locally

### Training Data Flow
1. Data collection scripts run separately to gather historical data
2. Historical data is stored in local SQLite databases
3. Training pipeline loads data from these databases
4. Models are trained on historical data
5. Trained models are saved for live trading use

## Configuration

The system is configured to maintain this separation through:

1. **Configuration File** (`src/config/config.yaml`):
   ```yaml
   data:
     symbols: ["BTCEUR", "SOLEUR", "ADAEUR", "XRPEUR", "ETHEUR"]
     interval: "15m"
     data_dir: "./data"  # For training only
     live_trading: true  # Enable real-time data fetching
   ```

2. **Code Structure**:
   - Live trading code in `scripts/trader.py` uses ccxt for data fetching
   - Training code in `src/data_pipeline/` uses local databases
   - Data collection scripts in `data/` populate local databases

## Benefits

1. **Clean Separation**: Live trading and training operations don't interfere with each other
2. **Data Integrity**: Training data remains unchanged during live trading
3. **Performance**: Live trading doesn't compete with training data access
4. **Scalability**: Each system can be optimized independently
5. **Reliability**: Issues in one system don't affect the other

## Best Practices

1. **Never mix data sources**: Live trading should never write to training databases
2. **Regular data collection**: Ensure historical data is regularly updated for training
3. **Monitor data quality**: Track both live and historical data quality metrics
4. **Backup training data**: Regularly backup local databases to prevent data loss
5. **Version control models**: Keep track of which models were trained on which data

## Monitoring and Maintenance

1. **Live Trading Monitoring**:
   - Track API call success rates
   - Monitor data freshness
   - Log data quality issues

2. **Training Data Maintenance**:
   - Regular data collection runs
   - Database integrity checks
   - Data quality validation

This policy ensures that live trading operations and model training/backtesting can coexist without interference while maintaining data integrity and system performance.