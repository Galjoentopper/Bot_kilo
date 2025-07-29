#!/usr/bin/env python3
"""
Improved Binance Data Collector with Async Support
Features: Connection pooling, async processing, batch operations, 
memory optimization, configuration file support, symbol validation,
data integrity checks, and resume capability.
"""

import os
import sqlite3
import aiosqlite
import aiohttp
import asyncio
import json
import zipfile
import io
import random
import threading
import psutil
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, AsyncGenerator
import logging
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from dataclasses import asdict
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CollectorState:
    """Manages collector state for resume capability"""
    
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load state from file"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return {}
    
    def save_state(self):
        """Save current state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state: {e}")
    
    def get_symbol_progress(self, symbol: str, interval: str) -> Dict:
        """Get progress for a specific symbol and interval"""
        key = f"{symbol}_{interval}"
        return self.state.get(key, {
            'last_bulk_month': None,
            'last_api_timestamp': None,
            'completed_phases': [],
            'total_records': 0
        })
    
    def update_symbol_progress(self, symbol: str, interval: str, **kwargs):
        """Update progress for a specific symbol and interval"""
        key = f"{symbol}_{interval}"
        if key not in self.state:
            self.state[key] = {
                'last_bulk_month': None,
                'last_api_timestamp': None,
                'completed_phases': [],
                'total_records': 0
            }
        self.state[key].update(kwargs)
        self.save_state()

class DataValidator:
    """Validates data integrity and quality"""
    
    @staticmethod
    def validate_ohlc(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate OHLC relationships and return cleaned data with errors"""
        errors = []
        original_len = len(df)
        
        if df.empty:
            return df, errors
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return df, errors
        
        # Convert to numeric and handle errors
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        df_clean = df.dropna(subset=required_cols)
        if len(df_clean) < original_len:
            errors.append(f"Removed {original_len - len(df_clean)} rows with NaN values")
        
        # Validate OHLC relationships
        invalid_high = df_clean['high'] < df_clean[['open', 'close']].max(axis=1)
        invalid_low = df_clean['low'] > df_clean[['open', 'close']].min(axis=1)
        invalid_volume = df_clean['volume'] < 0
        
        invalid_mask = invalid_high | invalid_low | invalid_volume
        
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            errors.append(f"Found {invalid_count} rows with invalid OHLC relationships")
            df_clean = df_clean[~invalid_mask]
        
        # Check for extreme price movements (>50% in one candle)
        if len(df_clean) > 0:
            price_change = abs(df_clean['close'] - df_clean['open']) / df_clean['open']
            extreme_moves = price_change > 0.5
            if extreme_moves.any():
                extreme_count = extreme_moves.sum()
                errors.append(f"Warning: {extreme_count} candles with >50% price movement")
        
        return df_clean, errors
    
    @staticmethod
    def validate_timestamps(df: pd.DataFrame, interval: str) -> Tuple[pd.DataFrame, List[str]]:
        """Validate timestamp consistency"""
        errors = []
        
        if df.empty or 'timestamp' not in df.columns:
            return df, errors
        
        # Convert interval to milliseconds
        interval_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }.get(interval, 15 * 60 * 1000)
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Check for duplicates
        duplicates = df_sorted['timestamp'].duplicated()
        if duplicates.any():
            dup_count = duplicates.sum()
            errors.append(f"Removed {dup_count} duplicate timestamps")
            df_sorted = df_sorted[~duplicates]
        
        # Check timestamp intervals
        if len(df_sorted) > 1:
            time_diffs = df_sorted['timestamp'].diff().dropna()
            expected_diff = interval_ms
            
            # Allow some tolerance (±10%)
            tolerance = expected_diff * 0.1
            irregular = abs(time_diffs - expected_diff) > tolerance
            
            if irregular.any():
                irregular_count = irregular.sum()
                errors.append(f"Warning: {irregular_count} irregular timestamp intervals")
        
        return df_sorted, errors

class DatabasePool:
    """Async database connection pool"""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool = asyncio.Queue(maxsize=pool_size)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the connection pool"""
        if self._initialized:
            return
        
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.db_path)
            await self._pool.put(conn)
        
        self._initialized = True
        logger.debug(f"Initialized database pool with {self.pool_size} connections")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        if not self._initialized:
            await self.initialize()
        
        conn = await self._pool.get()
        try:
            yield conn
        finally:
            await self._pool.put(conn)
    
    async def close_all(self):
        """Close all connections in the pool"""
        while not self._pool.empty():
            conn = await self._pool.get()
            await conn.close()
        logger.debug("Closed all database connections")

class RateLimitManager:
    """Async token bucket rate limiter"""
    
    def __init__(self, max_requests_per_minute: int = 50, burst_capacity: int = 10):
        self.max_requests_per_minute = max_requests_per_minute
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_refill = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()
        self.refill_rate = max_requests_per_minute / 60.0
        
        logger.info(f"RateLimitManager initialized: {max_requests_per_minute} req/min, burst: {burst_capacity}")
    
    async def acquire(self, tokens_needed: int = 1) -> bool:
        """Acquire tokens for making requests"""
        async with self.lock:
            current_time = asyncio.get_event_loop().time()
            time_elapsed = current_time - self.last_refill
            tokens_to_add = time_elapsed * self.refill_rate
            
            self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
            self.last_refill = current_time
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False
    
    async def wait_for_token(self, tokens_needed: int = 1):
        """Wait until tokens are available"""
        while not await self.acquire(tokens_needed):
            async with self.lock:
                tokens_deficit = tokens_needed - self.tokens
                wait_time = tokens_deficit / self.refill_rate
            
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s for {tokens_needed} tokens")
            await asyncio.sleep(min(wait_time + 0.1, 2.0))

class AsyncBinanceDataCollector:
    """Improved async Binance data collector"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.bulk_data_url = "https://data.binance.vision/data/spot"
        self.config = config
        
        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            raise ValueError(f"Configuration errors: {', '.join(config_errors)}")
        
        # Use the script's directory for storing databases
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = script_dir
        
        # Initialize components
        self.rate_limiter = RateLimitManager(
            self.config.max_requests_per_minute,
            self.config.burst_capacity
        )
        # Use state file in the same directory as the script
        state_file_path = os.path.join(self.data_dir, 'collector_state.json')
        self.state_manager = CollectorState(state_file_path)
        self.validator = DataValidator()
        
        # Database pools for each symbol
        self.db_pools: Dict[str, DatabasePool] = {}
        
        # Circuit breaker state
        self.api_failure_count = 0
        self.api_circuit_open = False
        self.circuit_reset_time = None
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        
        # Close all database pools
        for pool in self.db_pools.values():
            await pool.close_all()
    
    def _get_db_path(self, symbol: str, interval: str) -> str:
        """Get database path for symbol and interval"""
        return os.path.join(self.data_dir, f"{symbol.lower()}_{interval}.db")
    
    async def _get_db_pool(self, symbol: str, interval: str) -> DatabasePool:
        """Get or create database pool for symbol"""
        key = f"{symbol}_{interval}"
        if key not in self.db_pools:
            db_path = self._get_db_path(symbol, interval)
            self.db_pools[key] = DatabasePool(db_path, self.config.db_pool_size)
        return self.db_pools[key]
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate that symbol exists on Binance"""
        try:
            await self.rate_limiter.wait_for_token()
            
            async with self.session.get(f"{self.base_url}/exchangeInfo") as response:
                if response.status == 200:
                    data = await response.json()
                    symbols = [s['symbol'] for s in data.get('symbols', [])]
                    is_valid = symbol in symbols
                    
                    if is_valid:
                        logger.info(f"✅ Symbol {symbol} validated")
                    else:
                        logger.error(f"❌ Symbol {symbol} not found on Binance")
                    
                    return is_valid
                else:
                    logger.warning(f"Could not validate symbol {symbol}: HTTP {response.status}")
                    return True  # Assume valid if we can't check
        
        except Exception as e:
            logger.warning(f"Symbol validation failed for {symbol}: {e}")
            return True  # Assume valid if validation fails
    
    async def create_database(self, symbol: str, interval: str):
        """Create database with proper schema"""
        db_pool = await self._get_db_pool(symbol, interval)
        
        async with db_pool.get_connection() as conn:
            await conn.execute("""
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
            
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data(timestamp)"
            )
            
            await conn.commit()
        
        logger.info(f"Database created/verified for {symbol} {interval}")
    
    async def save_data_batch(self, df: pd.DataFrame, symbol: str, interval: str) -> int:
        """Save data in batches with validation"""
        if df.empty:
            return 0
        
        # Validate data
        df_clean, ohlc_errors = self.validator.validate_ohlc(df)
        df_clean, ts_errors = self.validator.validate_timestamps(df_clean, interval)
        
        # Log validation errors
        for error in ohlc_errors + ts_errors:
            logger.warning(f"{symbol} {interval}: {error}")
        
        if df_clean.empty:
            logger.warning(f"{symbol} {interval}: No valid data after validation")
            return 0
        
        db_pool = await self._get_db_pool(symbol, interval)
        total_inserted = 0
        
        # Process in chunks to manage memory
        chunk_size = self.config.batch_insert_size
        
        for i in range(0, len(df_clean), chunk_size):
            chunk = df_clean.iloc[i:i + chunk_size]
            
            async with db_pool.get_connection() as conn:
                # Get existing timestamps to avoid duplicates
                existing_query = """
                    SELECT timestamp FROM market_data 
                    WHERE timestamp BETWEEN ? AND ?
                """
                
                min_ts = int(chunk['timestamp'].min())
                max_ts = int(chunk['timestamp'].max())
                
                cursor = await conn.execute(existing_query, (min_ts, max_ts))
                existing_timestamps = {row[0] for row in await cursor.fetchall()}
                
                # Filter out existing data
                new_data = chunk[~chunk['timestamp'].isin(existing_timestamps)]
                
                if not new_data.empty:
                    # Prepare data for batch insert
                    records = []
                    for _, row in new_data.iterrows():
                        records.append((
                            int(row['timestamp']),
                            str(row['datetime']),  # Convert Timestamp to string
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
                    
                    # Batch insert
                    await conn.executemany("""
                        INSERT OR IGNORE INTO market_data 
                        (timestamp, datetime, open, high, low, close, volume, 
                         quote_volume, trades, taker_buy_base, taker_buy_quote)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, records)
                    
                    await conn.commit()
                    total_inserted += len(records)
            
            # Check memory usage
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > self.config.max_memory_mb:
                logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit, forcing garbage collection")
                import gc
                gc.collect()
        
        if total_inserted > 0:
            logger.info(f"{symbol} {interval}: Inserted {total_inserted} new records")
        
        return total_inserted
    
    async def download_bulk_file(self, symbol: str, interval: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """Download bulk historical data file"""
        filename = f"{symbol}-{interval}-{year:04d}-{month:02d}.zip"
        url = f"{self.bulk_data_url}/monthly/klines/{symbol}/{interval}/{filename}"
        
        for attempt in range(self.config.bulk_max_retries):
            try:
                logger.info(f"Downloading bulk file: {filename} (attempt {attempt + 1})")
                
                async with self.session.get(url) as response:
                    if response.status == 404:
                        logger.info(f"Bulk file not available: {symbol} {year}-{month:02d} (404)")
                        return None
                    
                    response.raise_for_status()
                    content = await response.read()
                
                # Extract CSV from ZIP
                with zipfile.ZipFile(io.BytesIO(content)) as zip_file:
                    csv_filename = filename.replace('.zip', '.csv')
                    with zip_file.open(csv_filename) as csv_file:
                        df = pd.read_csv(csv_file, header=None, names=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                            'taker_buy_quote', 'ignore'
                        ])
                
                # Process data
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                if df['timestamp'].abs().max() > 1e14:
                    df['timestamp'] = (df['timestamp'] // 1000).astype('int64')
                
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Select relevant columns
                df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close',
                        'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
                
                logger.info(f"Bulk file downloaded: {len(df)} candles for {symbol} {year}-{month:02d}")
                return df
                
            except Exception as e:
                if attempt < self.config.bulk_max_retries - 1:
                    backoff_delay = (2 ** attempt) * 2.0
                    logger.warning(f"Bulk download failed (attempt {attempt + 1}): {e}, retrying in {backoff_delay}s")
                    await asyncio.sleep(backoff_delay)
                else:
                    logger.error(f"Bulk download failed after {self.config.bulk_max_retries} attempts: {e}")
                    return None
            
            await asyncio.sleep(self.config.bulk_request_delay)
        
        return None
    
    async def get_klines_api(self, symbol: str, interval: str, start_time: int, end_time: int = None, limit: int = None) -> Optional[List]:
        """Get historical klines using API"""
        if limit is None:
            limit = self.config.api_batch_size
        
        try:
            await self.rate_limiter.wait_for_token()
            
            current_time = int(datetime.now().timestamp() * 1000)
            if start_time >= current_time:
                return []
            
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "limit": limit
            }
            
            if end_time:
                params["endTime"] = min(end_time, current_time)
            
            async with self.session.get(f"{self.base_url}/klines", params=params) as response:
                response.raise_for_status()
                data = await response.json()
                
                logger.info(f"API fetch successful: {len(data)} candles for {symbol} {interval}")
                return data
        
        except Exception as e:
            logger.error(f"API request failed for {symbol} {interval}: {e}")
            return None
    
    async def collect_symbol_data(self, symbol: str, interval: str) -> bool:
        """Collect data for a specific symbol and interval"""
        logger.info(f"Starting collection for {symbol} {interval}")
        
        # Validate symbol
        if not await self.validate_symbol(symbol):
            return False
        
        # Create database
        await self.create_database(symbol, interval)
        
        # Get progress state
        progress = self.state_manager.get_symbol_progress(symbol, interval)
        
        # Phase 1: Bulk data collection
        if 'bulk_complete' not in progress.get('completed_phases', []):
            await self._collect_bulk_data(symbol, interval, progress)
            progress['completed_phases'].append('bulk_complete')
            self.state_manager.update_symbol_progress(symbol, interval, **progress)
        
        # Phase 2: Recent data via API
        if 'api_complete' not in progress.get('completed_phases', []):
            await self._collect_recent_data(symbol, interval, progress)
            progress['completed_phases'].append('api_complete')
            self.state_manager.update_symbol_progress(symbol, interval, **progress)
        
        logger.info(f"✅ Collection completed for {symbol} {interval}")
        return True
    
    async def _collect_bulk_data(self, symbol: str, interval: str, progress: Dict):
        """Collect historical data using bulk downloads"""
        logger.info(f"Phase 1: Bulk data collection for {symbol} {interval}")
        
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d")
        current_dt = datetime.now()
        api_cutoff_dt = current_dt - timedelta(days=self.config.api_only_days)
        
        # Resume from last completed month if available
        if progress.get('last_bulk_month'):
            year, month = map(int, progress['last_bulk_month'].split('-'))
            current_month_dt = datetime(year, month, 1)
            if current_month_dt.month == 12:
                current_month_dt = current_month_dt.replace(year=current_month_dt.year + 1, month=1)
            else:
                current_month_dt = current_month_dt.replace(month=current_month_dt.month + 1)
        else:
            current_month_dt = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        total_records = 0
        
        while current_month_dt <= api_cutoff_dt:
            year = current_month_dt.year
            month = current_month_dt.month
            
            df = await self.download_bulk_file(symbol, interval, year, month)
            
            if df is not None and not df.empty:
                new_records = await self.save_data_batch(df, symbol, interval)
                total_records += new_records
                
                # Update progress
                progress['last_bulk_month'] = f"{year}-{month:02d}"
                progress['total_records'] = progress.get('total_records', 0) + new_records
                self.state_manager.update_symbol_progress(symbol, interval, **progress)
            
            await asyncio.sleep(self.config.bulk_request_delay)
            
            # Move to next month
            if current_month_dt.month == 12:
                current_month_dt = current_month_dt.replace(year=current_month_dt.year + 1, month=1)
            else:
                current_month_dt = current_month_dt.replace(month=current_month_dt.month + 1)
        
        logger.info(f"Bulk collection completed for {symbol} {interval}: {total_records} records")
    
    async def _collect_recent_data(self, symbol: str, interval: str, progress: Dict):
        """Collect recent data using API"""
        logger.info(f"Phase 2: Recent data collection for {symbol} {interval}")
        
        current_dt = datetime.now()
        api_cutoff_dt = current_dt - timedelta(days=self.config.api_only_days)
        api_cutoff_timestamp = int(api_cutoff_dt.timestamp() * 1000)
        current_timestamp = int(current_dt.timestamp() * 1000)
        
        # Resume from last API timestamp if available
        start_timestamp = progress.get('last_api_timestamp')
        if start_timestamp is None:
            start_timestamp = api_cutoff_timestamp
        
        interval_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }.get(interval, 15 * 60 * 1000)
        
        current_ts = start_timestamp
        total_records = 0
        
        while current_ts < current_timestamp and not self.api_circuit_open:
            klines_data = await self.get_klines_api(symbol, interval, current_ts, current_timestamp)
            
            if klines_data is None:
                logger.error(f"API request failed for {symbol} {interval} at timestamp {current_ts}")
                current_ts += (24 * 60 * 60 * 1000)  # Skip 1 day
                continue
            
            if not klines_data:
                break
            
            # Process data
            df = pd.DataFrame(klines_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                             'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce')
            
            if df['timestamp'].abs().max() > 1e14:
                df['timestamp'] = (df['timestamp'] // 1000).astype('int64')
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Select relevant columns
            df = df[['timestamp', 'datetime', 'open', 'high', 'low', 'close',
                    'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']]
            
            # Save data
            new_records = await self.save_data_batch(df, symbol, interval)
            total_records += new_records
            
            if not df.empty:
                current_ts = int(df['timestamp'].max()) + interval_ms
                
                # Update progress
                progress['last_api_timestamp'] = current_ts
                progress['total_records'] = progress.get('total_records', 0) + new_records
                self.state_manager.update_symbol_progress(symbol, interval, **progress)
            else:
                current_ts += (self.config.api_batch_size * interval_ms)
            
            await asyncio.sleep(self.config.api_request_delay)
        
        logger.info(f"Recent data collection completed for {symbol} {interval}: {total_records} records")
    
    async def collect_all_data(self) -> Dict[str, Dict[str, bool]]:
        """Collect data for all configured symbols and intervals"""
        logger.info(f"Starting data collection for {len(self.config.symbols)} symbols, {len(self.config.intervals)} intervals")
        
        results = {}
        
        # Create tasks for concurrent processing
        tasks = []
        for symbol in self.config.symbols:
            for interval in self.config.intervals:
                task = asyncio.create_task(
                    self.collect_symbol_data(symbol, interval),
                    name=f"{symbol}_{interval}"
                )
                tasks.append((symbol, interval, task))
        
        # Process tasks with limited concurrency
        semaphore = asyncio.Semaphore(3)  # Limit concurrent symbol processing
        
        async def process_with_semaphore(symbol, interval, task):
            async with semaphore:
                return symbol, interval, await task
        
        # Wait for all tasks to complete
        completed_tasks = await asyncio.gather(*[
            process_with_semaphore(symbol, interval, task)
            for symbol, interval, task in tasks
        ], return_exceptions=True)
        
        # Process results
        for result in completed_tasks:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            
            symbol, interval, success = result
            if symbol not in results:
                results[symbol] = {}
            results[symbol][interval] = success
        
        return results
    
    async def get_data_summary(self) -> Dict[str, Dict[str, Dict]]:
        """Get summary statistics for collected data"""
        summary = {}
        
        for symbol in self.config.symbols:
            summary[symbol] = {}
            for interval in self.config.intervals:
                db_path = self._get_db_path(symbol, interval)
                
                if os.path.exists(db_path):
                    db_pool = await self._get_db_pool(symbol, interval)
                    
                    async with db_pool.get_connection() as conn:
                        cursor = await conn.execute("""
                            SELECT 
                                COUNT(*) as total_records,
                                MIN(datetime) as start_date,
                                MAX(datetime) as end_date,
                                MIN(close) as min_price,
                                MAX(close) as max_price,
                                AVG(volume) as avg_volume
                            FROM market_data
                        """)
                        
                        row = await cursor.fetchone()
                        if row:
                            summary[symbol][interval] = {
                                'total_records': row[0],
                                'start_date': row[1],
                                'end_date': row[2],
                                'min_price': row[3],
                                'max_price': row[4],
                                'avg_volume': row[5]
                            }
                        else:
                            summary[symbol][interval] = {'status': 'No data found'}
                else:
                    summary[symbol][interval] = {'status': 'No database file found'}
        
        return summary

async def main():
    """Main async function"""
    print("🚀 Improved Binance Data Collector")
    print("===================================")
    
    try:
        async with AsyncBinanceDataCollector() as collector:
            # Collect all data
            results = await collector.collect_all_data()
            
            # Print results
            print("\n📊 Collection Results:")
            print("=====================")
            for symbol, intervals in results.items():
                for interval, success in intervals.items():
                    status = "✅ Success" if success else "❌ Failed"
                    print(f"{symbol} {interval}: {status}")
            
            # Print summary
            print("\n📈 Data Summary:")
            print("===============")
            summary = await collector.get_data_summary()
            
            for symbol, intervals in summary.items():
                print(f"\n{symbol}:")
                for interval, stats in intervals.items():
                    print(f"  {interval}:")
                    if 'status' in stats:
                        print(f"    {stats['status']}")
                    else:
                        print(f"    Records: {stats['total_records']:,}")
                        print(f"    Date range: {stats['start_date']} to {stats['end_date']}")
                        
                        # Handle None values for price range
                        if stats['min_price'] is not None and stats['max_price'] is not None:
                            print(f"    Price range: €{stats['min_price']:.4f} - €{stats['max_price']:.4f}")
                        else:
                            print(f"    Price range: No price data available")
                        
                        # Handle None value for average volume
                        if stats['avg_volume'] is not None:
                            print(f"    Avg volume: {stats['avg_volume']:,.2f}")
                        else:
                            print(f"    Avg volume: No volume data available")
    
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise
    
    print("\n🎉 Data collection completed!")
    print(f"📁 Data stored in: {config.data_dir}/")
    print("💡 Each symbol/interval has its own SQLite database file")

if __name__ == "__main__":
    asyncio.run(main())