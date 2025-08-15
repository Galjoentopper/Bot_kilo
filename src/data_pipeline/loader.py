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
        # Base interval of stored DBs (observed in repo)
        self.interval = "15m"
        
        # Validate data directory
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        
        logger.info(f"DataLoader initialized with data_dir: {data_dir}")
    
    def get_db_path(self, symbol: str, interval: Optional[str] = None) -> str:
        """Get database path for symbol and interval."""
        iv = (interval or self.interval).lower()
        return os.path.join(self.data_dir, f"{symbol.lower()}_{iv}.db")
    
    def load_symbol_data(
        self, 
        symbol: str,
        interval: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load data for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCEUR')
            interval: Time interval (e.g., '15m','30m','1h'); falls back to base interval with resampling if needed
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            limit: Maximum number of records to load
            
        Returns:
            DataFrame with OHLCV data
        """
        request_interval = (interval or self.interval).lower()
        db_path = self.get_db_path(symbol, request_interval)
        
        # Fallback to base interval DB if requested DB is missing
        used_base_interval = False
        if not os.path.exists(db_path):
            base_db_path = self.get_db_path(symbol, self.interval)
            if not os.path.exists(base_db_path):
                raise FileNotFoundError(f"Database not found for {symbol}: {db_path}")
            db_path = base_db_path
            used_base_interval = True
        
        # Build query
        query = "SELECT * FROM market_data"
        params: List[Union[str, int, float]] = []
        conditions: List[str] = []
        
        if start_date:
            conditions.append("datetime >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("datetime <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        # Load data
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn, params=tuple(params))
            conn.close()
            
            if df.empty:
                logger.warning(f"No data found for {symbol} with given criteria")
                return df
            
            # Convert datetime column
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            elif 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.set_index('datetime')
            else:
                raise ValueError("Expected 'datetime' or 'timestamp' column in DB")
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any NaN values
            df = df.dropna()
            
            # Resample if we used base interval and requested a different interval
            if used_base_interval and request_interval != self.interval:
                rule = self._interval_to_rule(request_interval)
                if rule:
                    series_map: Dict[str, pd.Series] = {}
                    if 'open' in df.columns:
                        series_map['open'] = df['open'].resample(rule).first()
                    if 'high' in df.columns:
                        series_map['high'] = df['high'].resample(rule).max()
                    if 'low' in df.columns:
                        series_map['low'] = df['low'].resample(rule).min()
                    if 'close' in df.columns:
                        series_map['close'] = df['close'].resample(rule).last()
                    if 'volume' in df.columns:
                        series_map['volume'] = df['volume'].resample(rule).sum()
                    if 'quote_volume' in df.columns:
                        series_map['quote_volume'] = df['quote_volume'].resample(rule).sum()
                    if 'taker_buy_base' in df.columns:
                        series_map['taker_buy_base'] = df['taker_buy_base'].resample(rule).sum()
                    if 'taker_buy_quote' in df.columns:
                        series_map['taker_buy_quote'] = df['taker_buy_quote'].resample(rule).sum()
                    if series_map:
                        df = pd.concat(series_map, axis=1).dropna(how='any')
            
            logger.info(f"Loaded {len(df)} records for {symbol} at interval {request_interval}")
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
        
        data: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = self.load_symbol_data(symbol)
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
        summaries: Dict[str, Dict] = {}
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
        availability: Dict[str, bool] = {}
        for symbol in self.symbols:
            db_path = self.get_db_path(symbol)
            availability[symbol] = os.path.exists(db_path)
        
        return availability
    
    def validate_data_integrity(self, symbol: str, interval: Optional[str] = None) -> Dict[str, Union[bool, str, int, float]]:
        """
        Validate data integrity for a symbol.
        
        Args:
            symbol: Trading symbol to validate
            interval: Time interval for data
            
        Returns:
            Dictionary with validation results
        """
        try:
            df = self.load_symbol_data(symbol, interval=interval)
            
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
            expected_diff = self._interval_to_timedelta(interval)
            time_diffs_seconds = time_diffs.dt.total_seconds().fillna(0)
            expected_sec = expected_diff.total_seconds()
            irregular_intervals = int((np.abs(time_diffs_seconds - expected_sec) > 60).sum())
            
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
    
    def _interval_to_timedelta(self, interval: Optional[str] = None) -> pd.Timedelta:
        """Convert interval string like '15m' or '1h' to pandas Timedelta."""
        iv = (interval or self.interval or '15m').strip().lower()
        try:
            if iv.endswith('m'):
                return pd.Timedelta(minutes=int(iv[:-1]))
            if iv.endswith('h'):
                return pd.Timedelta(hours=int(iv[:-1]))
            if iv.endswith('d'):
                return pd.Timedelta(days=int(iv[:-1]))
        except Exception:
            pass
        return pd.Timedelta(minutes=15)
    
    def _interval_to_rule(self, interval: Optional[str] = None) -> Optional[str]:
        """Convert interval like '15m'/'30m'/'1h' to pandas resample rule ('15T','30T','1H')."""
        iv = (interval or self.interval or '15m').strip().lower()
        try:
            if iv.endswith('m'):
                return f"{int(iv[:-1])}T"
            if iv.endswith('h'):
                return f"{int(iv[:-1])}H"
            if iv.endswith('d'):
                return f"{int(iv[:-1])}D"
        except Exception:
            return None
        return None