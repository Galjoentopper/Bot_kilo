#!/usr/bin/env python3
"""
Config-Based Binance Data Collector
===================================

A streamlined data collector that:
1. Reads configuration directly from config/config_training.yaml
2. Automatically collects data for specified symbols, interval, and timeframe
3. Creates properly named SQLite databases (symbol_interval.db)
4. Uses bulk data download when available, API for recent data
5. Maintains compatibility with existing model training pipeline
6. Includes progress tracking, error handling, and resume capability
"""

import os
import sys
import sqlite3
import requests
import zipfile
import io
import time
import json
import yaml
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import logging
from pathlib import Path

from src.config import ConfigLoader

# Configure logging with Windows-compatible encoding
import sys
if sys.platform == 'win32':
    # Set UTF-8 encoding for Windows console
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ConfigBasedCollector:
    """Config-based Binance data collector for streamlined model training"""
    
    def __init__(self, config_path: str = None):
        """Initialize collector with config file"""
        # Determine paths
        self.script_dir = Path(__file__).parent
        self.project_root = self.script_dir.parent
        
        # Load configuration using ConfigLoader
        config_loader = ConfigLoader()
        self.config = config_loader.load_config(config_path)
        self.data_config = self.config.get('data', {})
        
        # Extract data parameters
        self.symbols = self.data_config.get('symbols', [])
        self.interval = self.data_config.get('interval', '30m')
        self.lookback_days = self.data_config.get('lookback_days', 365)
        
        # API endpoints
        self.base_url = "https://api.binance.com/api/v3"
        self.bulk_url = "https://data.binance.vision/data/spot/monthly/klines"
        
        # Data directory (same as script location)
        self.data_dir = self.script_dir
        
        # State file for resume capability
        self.state_file = self.data_dir / 'config_collector_state.json'
        self.state = self.load_state()
        
        # Calculate date range
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=self.lookback_days)
        
        self.print_config()
    

    
    def print_config(self):
        """Print configuration summary"""
        print(f"\n[INFO] Config-Based Binance Data Collector")
        print(f"{'='*50}")
        print(f"[INFO] Data directory: {self.data_dir}")
        print(f"[INFO] Symbols: {', '.join(self.symbols)}")
        print(f"[INFO] Interval: {self.interval}")
        print(f"[INFO] Lookback: {self.lookback_days} days")
        print(f"[INFO] Date range: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"{'='*50}\n")
    
    def load_state(self) -> Dict:
        """Load collector state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        return {}
    
    def save_state(self):
        """Save collector state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def get_db_path(self, symbol: str) -> Path:
        """Get database path for symbol with proper naming"""
        return self.data_dir / f"{symbol.lower()}_{self.interval}.db"
    
    def create_database(self, symbol: str):
        """Create database with proper schema"""
        db_path = self.get_db_path(symbol)
        conn = sqlite3.connect(db_path)
        
        conn.execute("""
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
            )
        """)
        
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data(timestamp)")
        conn.commit()
        conn.close()
        
        logger.info(f"[SUCCESS] Database created/verified for {symbol}")
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate that symbol exists on Binance"""
        try:
            response = requests.get(f"{self.base_url}/exchangeInfo", timeout=10)
            if response.status_code == 200:
                data = response.json()
                symbols = [s['symbol'] for s in data.get('symbols', [])]
                return symbol in symbols
            return False
        except Exception as e:
            logger.warning(f"Could not validate {symbol}: {e}")
            return True  # Assume valid if validation fails
    
    def download_bulk_data(self, symbol: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """Download monthly bulk data from Binance"""
        filename = f"{symbol}-{self.interval}-{year:04d}-{month:02d}.zip"
        url = f"{self.bulk_url}/{symbol}/{self.interval}/{filename}"
        
        try:
            logger.info(f"[INFO] Downloading bulk data: {filename}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 404:
                logger.info(f"[INFO] Bulk data not available: {filename}")
                return None
            
            response.raise_for_status()
            
            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                csv_filename = filename.replace('.zip', '.csv')
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file, header=None, names=[
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
            
            # Filter by date range
            start_ts = int(self.start_date.timestamp() * 1000)
            end_ts = int(self.end_date.timestamp() * 1000)
            df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
            
            logger.info(f"[SUCCESS] Downloaded {len(df)} records for {symbol} {year}-{month:02d}")
            return df
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to download {filename}: {e}")
            return None
    
    def get_api_data(self, symbol: str, start_time: int, limit: int = 1000) -> Optional[List]:
        """Get kline data using Binance API"""
        try:
            params = {
                'symbol': symbol,
                'interval': self.interval,
                'startTime': start_time,
                'limit': limit
            }
            
            logger.info(f"[INFO] API request for {symbol} from {datetime.fromtimestamp(start_time/1000)}")
            response = requests.get(f"{self.base_url}/klines", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"[SUCCESS] Retrieved {len(data)} records via API")
            return data
            
        except Exception as e:
            logger.error(f"[ERROR] API request failed: {e}")
            return None
    
    def save_data_to_db(self, df: pd.DataFrame, symbol: str) -> int:
        """Save data to SQLite database with deduplication"""
        if df.empty:
            return 0
        
        db_path = self.get_db_path(symbol)
        conn = sqlite3.connect(db_path)
        
        # Get existing timestamps to avoid duplicates
        existing_query = """
            SELECT timestamp FROM market_data 
            WHERE timestamp BETWEEN ? AND ?
        """
        min_ts = int(df['timestamp'].min())
        max_ts = int(df['timestamp'].max())
        
        cursor = conn.execute(existing_query, (min_ts, max_ts))
        existing_timestamps = {row[0] for row in cursor.fetchall()}
        
        # Filter out existing data
        new_data = df[~df['timestamp'].isin(existing_timestamps)]
        
        if new_data.empty:
            conn.close()
            return 0
        
        # Insert new data
        records = []
        for _, row in new_data.iterrows():
            records.append((
                int(row['timestamp']),
                str(row['datetime']),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                float(row.get('quote_volume', 0)),
                int(row.get('trades', 0)),
                float(row.get('taker_buy_base', 0)),
                float(row.get('taker_buy_quote', 0))
            ))
        
        conn.executemany("""
            INSERT OR IGNORE INTO market_data 
            (timestamp, datetime, open, high, low, close, volume, 
             quote_volume, trades, taker_buy_base, taker_buy_quote)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        conn.commit()
        inserted = conn.total_changes
        conn.close()
        
        if inserted > 0:
            logger.info(f"[INFO] Inserted {inserted} new records for {symbol}")
        
        return inserted
    
    def collect_symbol_data(self, symbol: str):
        """Collect data for a specific symbol within the configured timeframe"""
        logger.info(f"\n🎯 Collecting data for {symbol}")
        
        # Validate symbol
        if not self.validate_symbol(symbol):
            logger.error(f"[ERROR] Invalid symbol: {symbol}")
            return
        
        # Create database
        self.create_database(symbol)
        
        # Get current progress
        symbol_state = self.state.get(symbol, {
            'last_bulk_month': None,
            'last_api_timestamp': None,
            'total_records': 0
        })
        
        total_inserted = 0
        
        # Phase 1: Bulk data collection
        logger.info(f"📦 Phase 1: Bulk data collection for {symbol}")
        
        # Calculate bulk data cutoff (use API for last 7 days)
        bulk_cutoff = self.end_date - timedelta(days=7)
        
        # Start from configured start date or resume point
        if symbol_state['last_bulk_month']:
            year, month = map(int, symbol_state['last_bulk_month'].split('-'))
            current_date = datetime(year, month, 1)
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        else:
            current_date = datetime(self.start_date.year, self.start_date.month, 1)
        
        while current_date <= bulk_cutoff:
            year = current_date.year
            month = current_date.month
            
            df = self.download_bulk_data(symbol, year, month)
            if df is not None:
                inserted = self.save_data_to_db(df, symbol)
                total_inserted += inserted
                
                # Update state
                symbol_state['last_bulk_month'] = f"{year}-{month:02d}"
                symbol_state['total_records'] += inserted
                self.state[symbol] = symbol_state
                self.save_state()
            
            # Rate limiting
            time.sleep(0.5)
            
            # Next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        # Phase 2: Recent data via API
        logger.info(f"[INFO] Phase 2: Recent data via API for {symbol}")
        
        # Start from bulk cutoff or resume point
        if symbol_state['last_api_timestamp']:
            start_timestamp = max(
                symbol_state['last_api_timestamp'],
                int(bulk_cutoff.timestamp() * 1000)
            )
        else:
            start_timestamp = int(bulk_cutoff.timestamp() * 1000)
        
        end_timestamp = int(self.end_date.timestamp() * 1000)
        
        while start_timestamp < end_timestamp:
            data = self.get_api_data(symbol, start_timestamp)
            
            if not data:
                break
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Process data
            numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce')
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Select columns
            df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close',
                    'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
            
            # Filter by date range
            start_ts = int(self.start_date.timestamp() * 1000)
            end_ts = int(self.end_date.timestamp() * 1000)
            df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)]
            
            # Save data
            inserted = self.save_data_to_db(df, symbol)
            total_inserted += inserted
            
            if not df.empty:
                # Get interval in milliseconds
                interval_ms = self.get_interval_ms()
                start_timestamp = int(df['timestamp'].max()) + interval_ms
                
                # Update state
                symbol_state['last_api_timestamp'] = start_timestamp
                symbol_state['total_records'] += inserted
                self.state[symbol] = symbol_state
                self.save_state()
            else:
                break
            
            # Rate limiting
            time.sleep(1.0)
        
        logger.info(f"[SUCCESS] Completed {symbol}: {total_inserted} total new records")
    
    def get_interval_ms(self) -> int:
        """Convert interval string to milliseconds"""
        interval_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000
        }
        return interval_map.get(self.interval, 30 * 60 * 1000)  # Default to 30m
    
    def collect_all_data(self):
        """Collect data for all configured symbols"""
        if not self.symbols:
            logger.error("[ERROR] No symbols configured in config file")
            return
        
        logger.info(f"\n[INFO] Starting data collection for {len(self.symbols)} symbols")
        
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"\n[INFO] Processing symbol {i}/{len(self.symbols)}: {symbol}")
            try:
                self.collect_symbol_data(symbol)
            except Exception as e:
                logger.error(f"[ERROR] Failed to collect {symbol}: {e}")
                continue
        
        logger.info(f"\n[SUCCESS] Data collection completed!")
        self.print_summary()
    
    def print_summary(self):
        """Print summary of collected data"""
        logger.info(f"\n📈 Data Collection Summary:")
        logger.info(f"{'='*50}")
        
        for symbol in self.symbols:
            db_path = self.get_db_path(symbol)
            
            if db_path.exists():
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(datetime) as start_date,
                        MAX(datetime) as end_date
                    FROM market_data
                """)
                row = cursor.fetchone()
                conn.close()
                
                if row and row[0] > 0:
                    logger.info(f"{symbol}: {row[0]:,} records ({row[1]} to {row[2]})")
                    logger.info(f"  [INFO] Database: {db_path.name}")
                else:
                    logger.info(f"{symbol}: No data")
            else:
                logger.info(f"{symbol}: No database file")
        
        logger.info(f"\n[INFO] Databases saved in: {self.data_dir}")
        logger.info(f"[CONFIG] Configuration: {self.interval} interval, {self.lookback_days} days lookback")

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Config-based Binance data collector for model training'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to config file (default: config/config_training.yaml)'
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        help='Collect data for specific symbol only'
    )
    
    args = parser.parse_args()
    
    try:
        collector = ConfigBasedCollector(args.config)
        
        if args.symbol:
            if args.symbol.upper() in collector.symbols:
                collector.collect_symbol_data(args.symbol.upper())
                collector.print_summary()
            else:
                logger.error(f"[ERROR] Symbol {args.symbol} not found in config")
                logger.info(f"Available symbols: {', '.join(collector.symbols)}")
        else:
            collector.collect_all_data()
            
    except KeyboardInterrupt:
        logger.info("\n[INFO] Collection interrupted by user")
    except Exception as e:
        logger.error(f"[ERROR] Collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()