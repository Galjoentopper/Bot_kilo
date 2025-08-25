#!/usr/bin/env python3
"""
Configuration management for Binance Data Collector
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class CollectorConfig:
    """Configuration class for data collector"""
    
    # Data collection settings
    symbols: List[str]
    intervals: List[str]
    start_date: str
    data_dir: str
    
    # Rate limiting settings
    max_requests_per_minute: int
    burst_capacity: int
    api_batch_size: int
    api_only_days: int
    
    # Retry and backoff settings
    api_max_retries: int
    bulk_max_retries: int
    base_backoff_delay: float
    max_backoff_delay: float
    circuit_reset_delay: int
    max_api_failures: int
    
    # Performance settings
    bulk_request_delay: float
    api_request_delay: float
    chunk_size: int
    max_memory_mb: int
    
    # Database settings
    db_pool_size: int
    db_timeout: int
    batch_insert_size: int
    
    # Resume settings
    state_file: str
    auto_resume: bool
    
    @classmethod
    def from_env(cls) -> 'CollectorConfig':
        """Create configuration from environment variables"""
        
        # Parse symbols from comma-separated string
        symbols_str = os.getenv('SYMBOLS', 'BTCEUR,ETHEUR,ADAEUR,SOLEUR,XRPEUR')
        symbols = [s.strip().upper() for s in symbols_str.split(',')]
        
        # Parse intervals from comma-separated string
        intervals_str = os.getenv('INTERVALS', '30m')
        intervals = [i.strip() for i in intervals_str.split(',')]
        
        return cls(
            # Data collection settings
            symbols=symbols,
            intervals=intervals,
            start_date=os.getenv('START_DATE', '2020-01-01'),
            data_dir=os.getenv('DATA_DIR', '.'),
            
            # Rate limiting settings
            max_requests_per_minute=int(os.getenv('MAX_REQUESTS_PER_MINUTE', '6')),
            burst_capacity=int(os.getenv('BURST_CAPACITY', '2')),
            api_batch_size=int(os.getenv('API_BATCH_SIZE', '50')),
            api_only_days=int(os.getenv('API_ONLY_DAYS', '3')),
            
            # Retry and backoff settings
            api_max_retries=int(os.getenv('API_MAX_RETRIES', '3')),
            bulk_max_retries=int(os.getenv('BULK_MAX_RETRIES', '5')),
            base_backoff_delay=float(os.getenv('BASE_BACKOFF_DELAY', '10.0')),
            max_backoff_delay=float(os.getenv('MAX_BACKOFF_DELAY', '600.0')),
            circuit_reset_delay=int(os.getenv('CIRCUIT_RESET_DELAY', '3600')),
            max_api_failures=int(os.getenv('MAX_API_FAILURES', '3')),
            
            # Performance settings
            bulk_request_delay=float(os.getenv('BULK_REQUEST_DELAY', '0.5')),
            api_request_delay=float(os.getenv('API_REQUEST_DELAY', '0.5')),
            chunk_size=int(os.getenv('CHUNK_SIZE', '10000')),
            max_memory_mb=int(os.getenv('MAX_MEMORY_MB', '500')),
            
            # Database settings
            db_pool_size=int(os.getenv('DB_POOL_SIZE', '5')),
            db_timeout=int(os.getenv('DB_TIMEOUT', '30')),
            batch_insert_size=int(os.getenv('BATCH_INSERT_SIZE', '1000')),
            
            # Resume settings
            state_file=os.getenv('STATE_FILE', os.path.join(os.getenv('DATA_DIR', '.'), 'collector_state.json')),
            auto_resume=os.getenv('AUTO_RESUME', 'true').lower() == 'true'
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.symbols:
            errors.append("At least one symbol must be specified")
        
        if not self.intervals:
            errors.append("At least one interval must be specified")
        
        if self.max_requests_per_minute <= 0:
            errors.append("max_requests_per_minute must be positive")
        
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.batch_insert_size <= 0:
            errors.append("batch_insert_size must be positive")
        
        return errors

# Global configuration instance
config = CollectorConfig.from_env()