"""
Data Loader Module
==================

Loads data from SQLite databases and prepares it for model training.
Integrates with existing data collection system.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader for crypto market data from SQLite databases.
    Compatible with existing data collection system.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing SQLite database files
        """
        self.data_dir = data_dir
        self.symbols = ["BTCEUR", "ETHEUR", "ADAEUR", "SOLEUR", "XRPEUR"]
        self.interval = "15m"
        
        # Validate data directory
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        
        logger.info(f"DataLoader initialized with data_dir: {data_dir}")
    
    def get_db_path(self, symbol: str) -> str:
        """Get database path for symbol."""
        return os.path.join(self.data_dir, f"{symbol.lower()}_{self.interval}.db")
    
    def load_symbol_data(
        self, 
        symbol: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load data for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCEUR')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            limit: Maximum number of records to load
            
        Returns:
            DataFrame with OHLCV data
        """
        db_path = self.get_db_path(symbol)
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found for {symbol}: {db_path}")
        
        # Build query
        query = "SELECT * FROM market_data"
        params = []
        conditions = []
        
        if start_date:
            conditions.append("datetime >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("datetime <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        if limit:
            query += f" LIMIT {limit}"
        
        # Load data
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                logger.warning(f"No data found for {symbol} with given criteria")
                return df
            
            # Convert datetime column
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN values
            df = df.dropna()
            
            logger.info(f"Loaded {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            raise
    
    def load_multiple_symbols(
        self, 
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of symbols to load (default: all available)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            limit: Maximum number of records per symbol
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.symbols
        
        data = {}
        for symbol in symbols:
            try:
                df = self.load_symbol_data(symbol, start_date, end_date, limit)
                if not df.empty:
                    data[symbol] = df
                else:
                    logger.warning(f"No data loaded for {symbol}")
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                continue
        
        logger.info(f"Loaded data for {len(data)} symbols")
        return data
    
    def get_data_summary(self, symbol: str) -> Dict:
        """
        Get summary statistics for a symbol's data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with summary statistics
        """
        db_path = self.get_db_path(symbol)
        
        if not os.path.exists(db_path):
            return {"error": f"Database not found: {db_path}"}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(datetime) as start_date,
                    MAX(datetime) as end_date,
                    MIN(close) as min_price,
                    MAX(close) as max_price,
                    AVG(close) as avg_price,
                    AVG(volume) as avg_volume,
                    SUM(volume) as total_volume
                FROM market_data
            """)
            
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0] > 0:
                return {
                    "symbol": symbol,
                    "total_records": row[0],
                    "start_date": row[1],
                    "end_date": row[2],
                    "min_price": row[3],
                    "max_price": row[4],
                    "avg_price": row[5],
                    "avg_volume": row[6],
                    "total_volume": row[7],
                    "price_range": row[4] - row[3] if row[3] and row[4] else None
                }
            else:
                return {"error": "No data found"}
                
        except Exception as e:
            logger.error(f"Error getting summary for {symbol}: {e}")
            return {"error": str(e)}
    
    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get summary statistics for all symbols."""
        summaries = {}
        for symbol in self.symbols:
            summaries[symbol] = self.get_data_summary(symbol)
        return summaries
    
    def create_train_test_split(
        self, 
        df: pd.DataFrame, 
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/validation/test split.
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion for training
            validation_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_latest_data(self, symbol: str, n_periods: int = 100) -> pd.DataFrame:
        """
        Get the most recent n periods of data for a symbol.
        
        Args:
            symbol: Trading symbol
            n_periods: Number of recent periods to retrieve
            
        Returns:
            DataFrame with recent data
        """
        return self.load_symbol_data(symbol, limit=n_periods)
    
    def check_data_availability(self) -> Dict[str, bool]:
        """
        Check which symbols have data available.
        
        Returns:
            Dictionary mapping symbol to availability status
        """
        availability = {}
        for symbol in self.symbols:
            db_path = self.get_db_path(symbol)
            availability[symbol] = os.path.exists(db_path)
        
        return availability
    
    def validate_data_integrity(self, symbol: str) -> Dict[str, Union[bool, str, int, float]]:
        """
        Validate data integrity for a symbol.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            df = self.load_symbol_data(symbol)
            
            if df.empty:
                return {"valid": False, "error": "No data found"}
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {"valid": False, "error": f"Missing columns: {missing_cols}"}
            
            # Check OHLC relationships
            invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            negative_volume = (df['volume'] < 0).sum()
            
            # Check for gaps in data
            df_sorted = df.sort_index()
            time_diffs = df_sorted.index.to_series().diff()
            expected_diff = pd.Timedelta(minutes=15)  # 15-minute intervals
            irregular_intervals = (abs(time_diffs - expected_diff) > pd.Timedelta(minutes=1)).sum()
            
            return {
                "valid": True,
                "total_records": len(df),
                "invalid_high": int(invalid_high),
                "invalid_low": int(invalid_low),
                "negative_volume": int(negative_volume),
                "irregular_intervals": int(irregular_intervals),
                "data_quality_score": float(1.0 - (invalid_high + invalid_low + negative_volume) / len(df))
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}