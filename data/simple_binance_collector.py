#!/usr/bin/env python3
"""
Simple Binance Data Collector for Paper Trading
===============================================

A simplified, focused data collector that:
1. Downloads 15-minute kline data for specified symbols
2. Uses bulk data download when available, API when not
3. Saves SQLite databases in the same folder as the script
4. Maintains compatibility with existing model training pipeline

Symbols: BTCEUR, ETHEUR, ADAEUR, SOLEUR, XRPEUR
"""

import os
import sqlite3
import requests
import zipfile
import io
import time
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleBinanceCollector:
    """Simple, focused Binance data collector for paper trading"""
    
    def __init__(self, symbols: List[str] = None):
        """Initialize collector with symbols"""
        self.symbols = symbols or ['BTCEUR', 'ETHEUR', 'ADAEUR', 'SOLEUR', 'XRPEUR']
        self.interval = '15m'  # Fixed to 15-minute intervals
        self.base_url = "https://api.binance.com/api/v3"
        self.bulk_url = "https://data.binance.vision/data/spot/monthly/klines"
        
        # Create data directory in same folder as script
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        
        # State file for resume capability
        self.state_file = os.path.join(self.data_dir, 'collector_state.json')
        self.state = self.load_state()
        
        print(f"🚀 Simple Binance Data Collector")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"💰 Symbols: {', '.join(self.symbols)}")
        print(f"⏰ Interval: {self.interval}")
    
    def load_state(self) -> Dict:
        """Load collector state from file"""
        if os.path.exists(self.state_file):
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
    
    def get_db_path(self, symbol: str) -> str:
        """Get database path for symbol"""
        return os.path.join(self.data_dir, f"{symbol.lower()}_15m.db")
    
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
        
        logger.info(f"✅ Database created/verified for {symbol}")
    
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
        filename = f"{symbol}-15m-{year:04d}-{month:02d}.zip"
        url = f"{self.bulk_url}/{symbol}/15m/{filename}"
        
        try:
            logger.info(f"📥 Downloading bulk data: {filename}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 404:
                logger.info(f"❌ Bulk data not available: {filename}")
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
            
            logger.info(f"✅ Downloaded {len(df)} records for {symbol} {year}-{month:02d}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
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
            
            logger.info(f"📡 API request for {symbol}")
            response = requests.get(f"{self.base_url}/klines", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"✅ Retrieved {len(data)} records via API")
            return data
            
        except Exception as e:
            logger.error(f"❌ API request failed: {e}")
            return None
    
    def save_data_to_db(self, df: pd.DataFrame, symbol: str) -> int:
        """Save data to SQLite database"""
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
            logger.info(f"💾 Inserted {inserted} new records for {symbol}")
        
        return inserted
    
    def collect_symbol_data(self, symbol: str):
        """Collect all available data for a symbol"""
        logger.info(f"\n🎯 Collecting data for {symbol}")
        
        # Validate symbol
        if not self.validate_symbol(symbol):
            logger.error(f"❌ Invalid symbol: {symbol}")
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
        
        # Phase 1: Bulk data collection (2020-01 to recent)
        logger.info(f"📦 Phase 1: Bulk data collection for {symbol}")
        
        start_date = datetime(2020, 1, 1)
        cutoff_date = datetime.now() - timedelta(days=7)  # Use API for last 7 days
        
        # Resume from last completed month
        if symbol_state['last_bulk_month']:
            year, month = map(int, symbol_state['last_bulk_month'].split('-'))
            current_date = datetime(year, month, 1)
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        else:
            current_date = start_date
        
        while current_date <= cutoff_date:
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
        logger.info(f"📡 Phase 2: Recent data via API for {symbol}")
        
        # Start from where bulk data ended or from cutoff date
        if symbol_state['last_api_timestamp']:
            start_timestamp = symbol_state['last_api_timestamp']
        else:
            start_timestamp = int(cutoff_date.timestamp() * 1000)
        
        current_timestamp = int(datetime.now().timestamp() * 1000)
        
        while start_timestamp < current_timestamp:
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
            
            # Save data
            inserted = self.save_data_to_db(df, symbol)
            total_inserted += inserted
            
            if not df.empty:
                start_timestamp = int(df['timestamp'].max()) + (15 * 60 * 1000)  # Next 15m interval
                
                # Update state
                symbol_state['last_api_timestamp'] = start_timestamp
                symbol_state['total_records'] += inserted
                self.state[symbol] = symbol_state
                self.save_state()
            else:
                break
            
            # Rate limiting
            time.sleep(1.0)
        
        logger.info(f"🎉 Completed {symbol}: {total_inserted} total new records")
    
    def collect_all_data(self):
        """Collect data for all symbols"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Starting data collection for all symbols")
        logger.info(f"{'='*60}")
        
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"\n📊 Processing symbol {i}/{len(self.symbols)}: {symbol}")
            try:
                self.collect_symbol_data(symbol)
            except Exception as e:
                logger.error(f"❌ Failed to collect {symbol}: {e}")
                continue
        
        logger.info(f"\n🎉 Data collection completed!")
        self.print_summary()
    
    def print_summary(self):
        """Print summary of collected data"""
        logger.info(f"\n📈 Data Collection Summary:")
        logger.info(f"{'='*40}")
        
        for symbol in self.symbols:
            db_path = self.get_db_path(symbol)
            
            if os.path.exists(db_path):
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
                else:
                    logger.info(f"{symbol}: No data")
            else:
                logger.info(f"{symbol}: No database file")
        
        logger.info(f"\n📁 Databases saved in: {self.data_dir}")

def main():
    """Main function"""
    collector = SimpleBinanceCollector()
    collector.collect_all_data()

if __name__ == "__main__":
    main()