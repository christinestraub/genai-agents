# import necessary python libraries
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Optional, Union, Dict, Any, List, Tuple
import hashlib
import pickle
import os
import warnings
from enum import Enum
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# For machine learning models
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Some forecasting features will be limited.")

from agno.agent import Agent
from agno.models.xai import xAI
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.playground import Playground, serve_playground_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    A reusable class for fetching real-time and historical stock market data.
    Supports caching and error handling to provide reliable market data access.
    """
    
    def __init__(self, cache_dir: str = "market_data_cache", cache_duration_hours: int = 1):
        """
        Initialize the MarketDataFetcher.
        
        Args:
            cache_dir: Directory to store cached data
            cache_duration_hours: How long to cache data (in hours)
        """
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
    def _generate_cache_key(self, ticker: str, days: int, interval: str) -> str:
        """Generate a unique cache key for the request."""
        key_string = f"{ticker}_{days}_{interval}"
        return hashlib.md5(key_string.encode()).hexdigest()
        
    def _get_cache_filepath(self, cache_key: str) -> str:
        """Get the full filepath for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
    def _is_cache_valid(self, filepath: str) -> bool:
        """Check if cached data is still valid."""
        if not os.path.exists(filepath):
            return False
            
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        return datetime.now() - file_time < self.cache_duration
        
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid."""
        filepath = self._get_cache_filepath(cache_key)
        
        if self._is_cache_valid(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Loaded data from cache for key: {cache_key}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache file {filepath}: {e}")
                
        return None
        
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save data to cache."""
        filepath = self._get_cache_filepath(cache_key)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved data to cache for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache file {filepath}: {e}")
            
    def fetch_stock_data(
        self, 
        ticker: str, 
        days: int = 30, 
        interval: str = "1d",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a given ticker and time range.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            days: Number of days of historical data to fetch
            interval: Data interval ('1d', '1h', '5m', etc.)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLC data and timestamps, or None if error
        """
        # Validate inputs
        if not ticker or not isinstance(ticker, str):
            logger.error("Invalid ticker provided")
            return None
            
        if days <= 0:
            logger.error("Days must be positive")
            return None
            
        ticker = ticker.upper().strip()
        cache_key = self._generate_cache_key(ticker, days, interval)
        
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
                
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            logger.info(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}")
            
            # Create yfinance ticker object
            stock = yf.Ticker(ticker)
            
            # Fetch historical data
            data = stock.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )
            
            if data.empty:
                logger.error(f"No data found for ticker: {ticker}")
                return None
                
            # Clean and format the data
            data = self._clean_data(data, ticker)
            
            # Save to cache
            if use_cache:
                self._save_to_cache(data, cache_key)
                
            logger.info(f"Successfully fetched {len(data)} rows of data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
            
    def _clean_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and format the fetched data."""
        # Ensure we have the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Keep only the columns we need
        data = data[expected_columns].copy()
        
        # Add ticker symbol as a column
        data['Ticker'] = ticker
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Round numeric columns to 2 decimal places
        numeric_columns = ['Open', 'High', 'Low', 'Close']
        data[numeric_columns] = data[numeric_columns].round(2)
        
        return data
        
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get the current/latest price for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current price as float, or None if error
        """
        try:
            stock = yf.Ticker(ticker.upper().strip())
            info = stock.info
            
            # Try different price fields
            current_price = (
                info.get('currentPrice') or 
                info.get('regularMarketPrice') or 
                info.get('previousClose')
            )
            
            if current_price is None:
                # Fallback: get latest from recent data
                recent_data = stock.history(period="1d", interval="1m")
                if not recent_data.empty:
                    current_price = recent_data['Close'].iloc[-1]
                    
            return float(current_price) if current_price is not None else None
            
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {str(e)}")
            return None
            
    def get_multiple_tickers(
        self, 
        tickers: list, 
        days: int = 30, 
        frequency: str = "daily",
        align_timestamps: bool = True,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers with aligned timestamps.
        
        Args:
            tickers: List of ticker symbols
            days: Number of days of historical data
            frequency: Data frequency ('daily', 'hourly', 'weekly', 'monthly', or yfinance interval)
            align_timestamps: Whether to align timestamps across all tickers
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping ticker to DataFrame with aligned timestamps
        """
        # Convert frequency to yfinance interval format
        interval = self._convert_frequency_to_interval(frequency)
        
        results = {}
        all_dataframes = []
        
        # Fetch data for each ticker
        for ticker in tickers:
            data = self.fetch_stock_data(ticker, days, interval, use_cache)
            if data is not None:
                results[ticker] = data
                all_dataframes.append(data)
            else:
                logger.warning(f"Failed to fetch data for {ticker}")
        
        # Align timestamps if requested and we have multiple DataFrames
        if align_timestamps and len(all_dataframes) > 1:
            results = self._align_timestamps(results)
                
        return results
    
    def _convert_frequency_to_interval(self, frequency: str) -> str:
        """
        Convert user-friendly frequency names to yfinance interval format.
        
        Args:
            frequency: User-friendly frequency name
            
        Returns:
            yfinance compatible interval string
        """
        frequency_mapping = {
            'daily': '1d',
            'hourly': '1h',
            'weekly': '1wk',
            'monthly': '1mo',
            'minute': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '90min': '90m'
        }
        
        # Return mapped frequency or original if it's already in correct format
        return frequency_mapping.get(frequency.lower(), frequency)
    
    def _align_timestamps(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align timestamps across all ticker DataFrames to ensure consistency.
        
        Args:
            ticker_data: Dictionary mapping ticker to DataFrame
            
        Returns:
            Dictionary with aligned DataFrames
        """
        if len(ticker_data) <= 1:
            return ticker_data
            
        # Get all unique timestamps from all DataFrames
        all_timestamps = set()
        for df in ticker_data.values():
            all_timestamps.update(df.index)
        
        # Create a common datetime index
        common_index = pd.DatetimeIndex(sorted(all_timestamps))
        
        # Find the intersection of all timestamps (only keep dates present in ALL stocks)
        intersection_index = None
        for df in ticker_data.values():
            if intersection_index is None:
                intersection_index = df.index
            else:
                intersection_index = intersection_index.intersection(df.index)
        
        aligned_data = {}
        
        for ticker, df in ticker_data.items():
            try:
                # Reindex to common timestamps, forward-fill missing values
                aligned_df = df.reindex(common_index)
                
                # Forward fill missing values for price data
                price_columns = ['Open', 'High', 'Low', 'Close']
                for col in price_columns:
                    if col in aligned_df.columns:
                        aligned_df[col] = aligned_df[col].ffill()
                
                # For volume, fill with 0 instead of forward fill
                if 'Volume' in aligned_df.columns:
                    aligned_df['Volume'] = aligned_df['Volume'].fillna(0)
                
                # Drop rows where we still have NaN in price columns (start of series)
                aligned_df = aligned_df.dropna(subset=price_columns, how='any')
                
                aligned_data[ticker] = aligned_df
                
                logger.info(f"Aligned {ticker}: {len(df)} -> {len(aligned_df)} rows")
                
            except Exception as e:
                logger.warning(f"Failed to align timestamps for {ticker}: {e}")
                # Keep original data if alignment fails
                aligned_data[ticker] = df
        
        return aligned_data
    
    def get_combined_data(
        self,
        tickers: list,
        days: int = 30,
        frequency: str = "daily",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch multiple tickers and combine into a single DataFrame with aligned timestamps.
        
        Args:
            tickers: List of ticker symbols
            days: Number of days of historical data
            frequency: Data frequency ('daily', 'hourly', 'weekly', 'monthly')
            use_cache: Whether to use cached data if available
            
        Returns:
            Combined DataFrame with MultiIndex columns (ticker, OHLC) or None if error
        """
        try:
            # Get aligned data for all tickers
            ticker_data = self.get_multiple_tickers(
                tickers=tickers,
                days=days,
                frequency=frequency,
                align_timestamps=True,
                use_cache=use_cache
            )
            
            if not ticker_data:
                logger.error("No data retrieved for any ticker")
                return None
            
            # Combine all DataFrames with MultiIndex columns
            combined_dfs = []
            
            for ticker, df in ticker_data.items():
                # Create MultiIndex columns (ticker, price_type)
                df_copy = df.copy()
                df_copy.columns = pd.MultiIndex.from_product([[ticker], df.columns])
                combined_dfs.append(df_copy)
            
            # Concatenate all DataFrames
            combined_df = pd.concat(combined_dfs, axis=1)
            
            # Sort columns by ticker for better readability
            combined_df = combined_df.sort_index(axis=1)
            
            logger.info(f"Combined data shape: {combined_df.shape}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining ticker data: {str(e)}")
            return None
        
    def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.cache_dir, filename)
                    os.remove(filepath)
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")


class MarketDataPreprocessor:
    """
    Comprehensive preprocessing class for market data.
    Cleans data, computes technical indicators, and formats for forecasting models.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def preprocess_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        indicators_config: Optional[Dict[str, Any]] = None,
        clean_missing: bool = True,
        add_features: bool = True
    ) -> Dict[str, Any]:
        """
        Main preprocessing function that handles both single and multi-ticker data.
        
        Args:
            data: Raw DataFrame or dict of DataFrames from MarketDataFetcher
            indicators_config: Configuration for technical indicators
            clean_missing: Whether to clean missing values
            add_features: Whether to add derived features
            
        Returns:
            Dictionary containing processed data and metadata
        """
        if indicators_config is None:
            indicators_config = self._get_default_indicators_config()
            
        try:
            # Handle single DataFrame vs multiple DataFrames
            if isinstance(data, pd.DataFrame):
                # Single ticker data
                result = self._process_single_ticker(
                    data, indicators_config, clean_missing, add_features
                )
                return {
                    'type': 'single_ticker',
                    'data': result['data'],
                    'metadata': result['metadata'],
                    'indicators': result['indicators'],
                    'summary': result['summary']
                }
            
            elif isinstance(data, dict):
                # Multiple ticker data
                results = {}
                combined_summary = {}
                
                for ticker, df in data.items():
                    result = self._process_single_ticker(
                        df, indicators_config, clean_missing, add_features
                    )
                    results[ticker] = result
                    combined_summary[ticker] = result['summary']
                
                return {
                    'type': 'multi_ticker',
                    'data': results,
                    'combined_summary': combined_summary,
                    'tickers': list(data.keys())
                }
            
            else:
                raise ValueError("Data must be DataFrame or Dict[str, DataFrame]")
                
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return {'error': str(e)}
    
    def _process_single_ticker(
        self,
        df: pd.DataFrame,
        indicators_config: Dict[str, Any],
        clean_missing: bool,
        add_features: bool
    ) -> Dict[str, Any]:
        """Process a single ticker's data."""
        try:
            # Step 1: Clean and validate data
            cleaned_df = self._clean_data(df, clean_missing)
            
            # Step 2: Compute technical indicators
            indicators_df = self._compute_technical_indicators(cleaned_df, indicators_config)
            
            # Step 3: Add derived features
            if add_features:
                features_df = self._add_derived_features(indicators_df)
            else:
                features_df = indicators_df
            
            # Step 4: Create final processed dataset
            processed_df = self._finalize_dataset(features_df)
            
            # Step 5: Generate metadata and summary
            metadata = self._generate_metadata(df, processed_df)
            summary = self._generate_summary(processed_df)
            
            # Step 6: Prepare for forecasting models
            model_ready_data = self._prepare_for_forecasting(processed_df)
            
            return {
                'data': processed_df,
                'model_ready': model_ready_data,
                'metadata': metadata,
                'indicators': list(indicators_config.keys()),
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Error processing single ticker: {str(e)}")
            raise
    
    def _clean_data(self, df: pd.DataFrame, clean_missing: bool) -> pd.DataFrame:
        """Clean the raw market data."""
        df_clean = df.copy()
        
        # Ensure datetime index
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            try:
                df_clean.index = pd.to_datetime(df_clean.index)
            except Exception as e:
                self.logger.error(f"Failed to convert index to datetime: {e}")
                raise
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        # Remove duplicate timestamps
        df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
        
        if clean_missing:
            # Handle missing values
            price_columns = ['Open', 'High', 'Low', 'Close']
            
            # Forward fill price data
            for col in price_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].ffill()
            
            # Fill volume with 0
            if 'Volume' in df_clean.columns:
                df_clean['Volume'] = df_clean['Volume'].fillna(0)
            
            # Drop rows with remaining NaN values in critical columns
            df_clean = df_clean.dropna(subset=['Close'])
        
        # Validate data integrity
        df_clean = self._validate_ohlc(df_clean)
        
        return df_clean
    
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC data integrity."""
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        
        if all(col in df.columns for col in ohlc_cols):
            # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
            df.loc[df['High'] < df[['Open', 'Close']].max(axis=1), 'High'] = df[['Open', 'Close']].max(axis=1)
            df.loc[df['Low'] > df[['Open', 'Close']].min(axis=1), 'Low'] = df[['Open', 'Close']].min(axis=1)
        
        return df
    
    def _compute_technical_indicators(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Compute technical indicators based on configuration."""
        df_indicators = df.copy()
        
        # Simple Moving Averages
        if 'sma' in config:
            for period in config['sma']['periods']:
                df_indicators[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        if 'ema' in config:
            for period in config['ema']['periods']:
                df_indicators[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI (Relative Strength Index)
        if 'rsi' in config:
            period = config['rsi']['period']
            df_indicators[f'RSI_{period}'] = self._calculate_rsi(df['Close'], period)
        
        # MACD (Moving Average Convergence Divergence)
        if 'macd' in config:
            macd_config = config['macd']
            macd_data = self._calculate_macd(
                df['Close'], 
                macd_config['fast'], 
                macd_config['slow'], 
                macd_config['signal']
            )
            df_indicators = pd.concat([df_indicators, macd_data], axis=1)
        
        # Bollinger Bands
        if 'bollinger' in config:
            bb_config = config['bollinger']
            bb_data = self._calculate_bollinger_bands(
                df['Close'], 
                bb_config['period'], 
                bb_config['std_dev']
            )
            df_indicators = pd.concat([df_indicators, bb_data], axis=1)
        
        # Volume indicators
        if 'volume_sma' in config and 'Volume' in df.columns:
            for period in config['volume_sma']['periods']:
                df_indicators[f'Volume_SMA_{period}'] = df['Volume'].rolling(window=period).mean()
        
        # Price momentum
        if 'momentum' in config:
            for period in config['momentum']['periods']:
                df_indicators[f'Momentum_{period}'] = df['Close'].pct_change(periods=period)
        
        return df_indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicators."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        })
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return pd.DataFrame({
            'BB_Upper': sma + (std * std_dev),
            'BB_Middle': sma,
            'BB_Lower': sma - (std * std_dev),
            'BB_Width': (std * std_dev * 2) / sma,
            'BB_Position': (prices - sma) / (std * std_dev)
        })
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for model training."""
        df_features = df.copy()
        
        # Price-based features
        if 'Close' in df.columns:
            # Returns
            df_features['Daily_Return'] = df['Close'].pct_change()
            df_features['Log_Return'] = np.log(df['Close']).diff()
            
            # Volatility (rolling standard deviation)
            df_features['Volatility_10'] = df['Close'].pct_change().rolling(10).std()
            df_features['Volatility_30'] = df['Close'].pct_change().rolling(30).std()
        
        # OHLC-based features
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            # True Range
            df_features['True_Range'] = self._calculate_true_range(df)
            
            # Average True Range
            df_features['ATR_14'] = df_features['True_Range'].rolling(14).mean()
            
            # Price position within the day's range
            df_features['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # Body size (Open to Close)
            df_features['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open']
            
            # Upper and lower shadows
            df_features['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
            df_features['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']
        
        # Volume-based features (if volume exists)
        if 'Volume' in df.columns:
            # Volume ratio to moving average
            df_features['Volume_Ratio_10'] = df['Volume'] / df['Volume'].rolling(10).mean()
            
            # Price-Volume trend
            if 'Close' in df.columns:
                df_features['PVT'] = ((df['Close'].pct_change() * df['Volume']).cumsum())
        
        # Time-based features
        df_features['Day_of_Week'] = df_features.index.dayofweek
        df_features['Month'] = df_features.index.month
        df_features['Quarter'] = df_features.index.quarter
        
        return df_features
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df['High'] - df['Low']
        high_close_prev = abs(df['High'] - df['Close'].shift(1))
        low_close_prev = abs(df['Low'] - df['Close'].shift(1))
        
        return pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    def _finalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize the dataset for model consumption."""
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with too many NaN values (>50%)
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)
        
        # Forward fill remaining NaN values
        df = df.ffill()
        
        # Drop any remaining rows with NaN values
        df = df.dropna()
        
        return df
    
    def _generate_metadata(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate metadata about the preprocessing."""
        return {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'date_range': {
                'start': processed_df.index[0].isoformat(),
                'end': processed_df.index[-1].isoformat(),
                'days': len(processed_df)
            },
            'columns': {
                'original': list(original_df.columns),
                'processed': list(processed_df.columns),
                'added': list(set(processed_df.columns) - set(original_df.columns))
            },
            'data_quality': {
                'missing_values': processed_df.isnull().sum().sum(),
                'infinite_values': np.isinf(processed_df.select_dtypes(include=[np.number])).sum().sum()
            }
        }
    
    def _generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for LLM consumption."""
        summary = {}
        
        if 'Close' in df.columns:
            close_prices = df['Close']
            summary['price_summary'] = {
                'current_price': float(close_prices.iloc[-1]),
                'min_price': float(close_prices.min()),
                'max_price': float(close_prices.max()),
                'avg_price': float(close_prices.mean()),
                'total_return': float((close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100),
                'volatility': float(close_prices.pct_change().std() * np.sqrt(252) * 100)
            }
        
        # Technical indicator summaries
        if 'RSI_14' in df.columns:
            rsi = df['RSI_14'].iloc[-1]
            summary['rsi_signal'] = {
                'value': float(rsi),
                'signal': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'
            }
        
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            macd_signal = df['MACD_Signal'].iloc[-1]
            summary['macd_signal'] = {
                'macd': float(macd),
                'signal': float(macd_signal),
                'trend': 'Bullish' if macd > macd_signal else 'Bearish'
            }
        
        # Moving average signals
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma_20 = df['SMA_20'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            summary['moving_average_signals'] = {
                'price_vs_sma20': 'Above' if current_price > sma_20 else 'Below',
                'price_vs_sma50': 'Above' if current_price > sma_50 else 'Below',
                'sma_crossover': 'Golden Cross' if sma_20 > sma_50 else 'Death Cross'
            }
        
        return summary
    
    def _prepare_for_forecasting(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data specifically for forecasting models."""
        # Select features for forecasting
        feature_columns = [col for col in df.columns if col not in ['Ticker']]
        
        # Create feature matrix
        X = df[feature_columns].values
        
        # Create target variable (next day's close price)
        if 'Close' in df.columns:
            y = df['Close'].shift(-1).dropna().values
            X = X[:-1]  # Remove last row to match y length
        else:
            y = None
        
        return {
            'features': X,
            'target': y,
            'feature_names': feature_columns,
            'timestamps': df.index[:-1].tolist() if y is not None else df.index.tolist(),
            'last_values': df.iloc[-1][feature_columns].to_dict(),
            'ready_for_prediction': True
        }
    
    def _get_default_indicators_config(self) -> Dict[str, Any]:
        """Get default configuration for technical indicators."""
        return {
            'sma': {'periods': [10, 20, 50, 200]},
            'ema': {'periods': [12, 26]},
            'rsi': {'period': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger': {'period': 20, 'std_dev': 2.0},
            'volume_sma': {'periods': [10, 20]},
            'momentum': {'periods': [1, 5, 10]}
        }
    
    def get_llm_explanation(self, processed_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation for LLM consumption."""
        if 'error' in processed_result:
            return f"Error in preprocessing: {processed_result['error']}"
        
        if processed_result['type'] == 'single_ticker':
            return self._explain_single_ticker(processed_result)
        else:
            return self._explain_multi_ticker(processed_result)
    
    def _explain_single_ticker(self, result: Dict[str, Any]) -> str:
        """Generate explanation for single ticker analysis."""
        summary = result['summary']
        metadata = result['metadata']
        
        explanation = f"""
## Market Data Analysis Summary

**Data Overview:**
- Analysis period: {metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}
- Total trading days: {metadata['date_range']['days']}
- Technical indicators computed: {', '.join(result['indicators'])}

**Price Performance:**"""
        
        if 'price_summary' in summary:
            ps = summary['price_summary']
            explanation += f"""
- Current price: ${ps['current_price']:.2f}
- Trading range: ${ps['min_price']:.2f} - ${ps['max_price']:.2f}
- Total return: {ps['total_return']:.2f}%
- Annualized volatility: {ps['volatility']:.2f}%"""
        
        if 'rsi_signal' in summary:
            rsi = summary['rsi_signal']
            explanation += f"""

**Technical Indicators:**
- RSI (14): {rsi['value']:.2f} - {rsi['signal']}"""
        
        if 'macd_signal' in summary:
            macd = summary['macd_signal']
            explanation += f"""
- MACD: {macd['trend']} trend (MACD: {macd['macd']:.4f}, Signal: {macd['signal']:.4f})"""
        
        if 'moving_average_signals' in summary:
            ma = summary['moving_average_signals']
            explanation += f"""
- Price vs SMA(20): {ma['price_vs_sma20']}
- Price vs SMA(50): {ma['price_vs_sma50']}
- SMA Crossover: {ma['sma_crossover']}"""
        
        explanation += f"""

**Data Quality:**
- Processed features: {metadata['processed_shape'][1]}
- Missing values: {metadata['data_quality']['missing_values']}
- Data ready for forecasting models: Yes"""
        
        return explanation
    
    def _explain_multi_ticker(self, result: Dict[str, Any]) -> str:
        """Generate explanation for multi-ticker analysis."""
        explanation = "## Multi-Ticker Market Analysis\n\n"
        
        for ticker in result['tickers']:
            if ticker in result['combined_summary']:
                summary = result['combined_summary'][ticker]
                explanation += f"### {ticker}\n"
                
                if 'price_summary' in summary:
                    ps = summary['price_summary']
                    explanation += f"- Price: ${ps['current_price']:.2f}, Return: {ps['total_return']:.2f}%\n"
                
                if 'rsi_signal' in summary:
                    rsi = summary['rsi_signal']
                    explanation += f"- RSI: {rsi['value']:.2f} ({rsi['signal']})\n"
                
                explanation += "\n"
        
        return explanation


class DataSource(Enum):
    """Enum for data source types."""
    REAL_TIME = "real_time"
    CSV = "csv"


class ForecastingAgent:
    """
    Advanced forecasting agent that can work with both real-time data and CSV files.
    Provides predictions, trend analysis, and structured forecasting results.
    """
    
    def __init__(
        self,
        fetcher: Optional[MarketDataFetcher] = None,
        preprocessor: Optional[MarketDataPreprocessor] = None,
        default_model: str = "random_forest"
    ):
        """
        Initialize the ForecastingAgent.
        
        Args:
            fetcher: MarketDataFetcher instance for real-time data
            preprocessor: MarketDataPreprocessor instance for data preparation
            default_model: Default ML model to use ('random_forest', 'gradient_boost', 'linear')
        """
        self.fetcher = fetcher or MarketDataFetcher()
        self.preprocessor = preprocessor or MarketDataPreprocessor()
        self.default_model = default_model
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize models
        self.models = self._initialize_models()
        self.trained_models = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize available ML models."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available. Using simple prediction methods.")
            return {}
            
        return {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'linear': LinearRegression()
        }
    
    def forecast(
        self,
        ticker: Optional[str] = None,
        time_window: int = 60,
        forecast_days: int = 5,
        csv_path: Optional[str] = None,
        data_source: DataSource = DataSource.REAL_TIME,
        model_type: Optional[str] = None,
        frequency: str = "daily"
    ) -> Dict[str, Any]:
        """
        Main forecasting method that handles both real-time and CSV data sources.
        
        Args:
            ticker: Stock ticker symbol (required for real-time data)
            time_window: Number of historical days to use for training
            forecast_days: Number of days to forecast ahead
            csv_path: Path to CSV file (for CSV mode)
            data_source: DataSource.REAL_TIME or DataSource.CSV
            model_type: ML model to use (overrides default)
            frequency: Data frequency for real-time data
            
        Returns:
            Structured forecasting results
        """
        try:
            # Validate inputs
            if data_source == DataSource.REAL_TIME and not ticker:
                raise ValueError("Ticker symbol required for real-time data source")
            
            if data_source == DataSource.CSV and not csv_path:
                raise ValueError("CSV path required for CSV data source")
            
            # Get and preprocess data
            if data_source == DataSource.REAL_TIME:
                processed_data = self._get_real_time_data(ticker, time_window, frequency)
            else:
                processed_data = self._get_csv_data(csv_path)
            
            if 'error' in processed_data:
                return processed_data
            
            # Run predictions
            prediction_results = self._run_predictions(
                processed_data, 
                forecast_days, 
                model_type or self.default_model
            )
            
            # Generate structured results
            structured_results = self._generate_structured_results(
                processed_data,
                prediction_results,
                ticker,
                forecast_days,
                data_source
            )
            
            return structured_results
            
        except Exception as e:
            self.logger.error(f"Error in forecasting: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def _get_real_time_data(self, ticker: str, time_window: int, frequency: str) -> Dict[str, Any]:
        """Fetch and preprocess real-time data."""
        try:
            # Fetch raw data
            raw_data = self.fetcher.fetch_stock_data(
                ticker=ticker,
                days=time_window,
                interval=self.fetcher._convert_frequency_to_interval(frequency)
            )
            
            if raw_data is None:
                return {'error': f'Failed to fetch data for {ticker}'}
            
            # Preprocess data
            processed_result = self.preprocessor.preprocess_data(raw_data)
            
            if 'error' in processed_result:
                return processed_result
            
            # Add source information
            processed_result['source'] = {
                'type': 'real_time',
                'ticker': ticker,
                'time_window': time_window,
                'frequency': frequency
            }
            
            return processed_result
            
        except Exception as e:
            return {'error': f'Error fetching real-time data: {str(e)}'}
    
    def _get_csv_data(self, csv_path: str) -> Dict[str, Any]:
        """Load and preprocess CSV data."""
        try:
            # Validate file exists
            if not Path(csv_path).exists():
                return {'error': f'CSV file not found: {csv_path}'}
            
            # Load CSV data
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = ['Date', 'Close']
            if not all(col in df.columns for col in required_columns):
                return {'error': f'CSV must contain columns: {required_columns}'}
            
            # Set date index
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Ensure we have OHLC columns or create them
            if 'Open' not in df.columns:
                df['Open'] = df['Close'].shift(1).fillna(df['Close'])
            if 'High' not in df.columns:
                df['High'] = df['Close']
            if 'Low' not in df.columns:
                df['Low'] = df['Close']
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            
            # Preprocess data
            processed_result = self.preprocessor.preprocess_data(df)
            
            if 'error' in processed_result:
                return processed_result
            
            # Add source information
            processed_result['source'] = {
                'type': 'csv',
                'file_path': csv_path,
                'original_shape': df.shape
            }
            
            return processed_result
            
        except Exception as e:
            return {'error': f'Error loading CSV data: {str(e)}'}
    
    def _run_predictions(
        self,
        processed_data: Dict[str, Any],
        forecast_days: int,
        model_type: str
    ) -> Dict[str, Any]:
        """Run ML predictions on processed data."""
        try:
            # Get model-ready data
            if processed_data['type'] == 'single_ticker':
                model_data = processed_data['model_ready']
            else:
                # For multi-ticker, use the first ticker
                first_ticker = list(processed_data['data'].keys())[0]
                model_data = processed_data['data'][first_ticker]['model_ready']
            
            X = model_data['features']
            y = model_data['target']
            feature_names = model_data['feature_names']
            
            if len(X) < 20:  # Need minimum data for training
                return self._simple_prediction_fallback(processed_data, forecast_days)
            
            # Split data for training and validation
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Train model
            if SKLEARN_AVAILABLE and model_type in self.models:
                model = self.models[model_type]
                model.fit(X_train, y_train)
                
                # Validate model
                val_predictions = model.predict(X_val)
                metrics = self._calculate_metrics(y_val, val_predictions)
                
                # Generate forecasts
                forecasts = self._generate_forecasts(model, X, forecast_days, feature_names)
                
                return {
                    'model_type': model_type,
                    'validation_metrics': metrics,
                    'forecasts': forecasts,
                    'model_trained': True,
                    'training_size': len(X_train)
                }
            else:
                return self._simple_prediction_fallback(processed_data, forecast_days)
                
        except Exception as e:
            self.logger.error(f"Error in predictions: {str(e)}")
            return self._simple_prediction_fallback(processed_data, forecast_days)
    
    def _simple_prediction_fallback(
        self, 
        processed_data: Dict[str, Any], 
        forecast_days: int
    ) -> Dict[str, Any]:
        """Simple prediction fallback when ML models aren't available."""
        try:
            # Get close prices
            if processed_data['type'] == 'single_ticker':
                df = processed_data['data']
            else:
                first_ticker = list(processed_data['data'].keys())[0]
                df = processed_data['data'][first_ticker]['data']
            
            close_prices = df['Close'].values
            
            # Simple trend-based prediction
            if len(close_prices) >= 10:
                # Use average of last 10 days' returns
                recent_returns = np.diff(close_prices[-10:]) / close_prices[-10:-1]
                avg_return = np.mean(recent_returns)
                
                # Generate simple forecasts
                forecasts = []
                last_price = close_prices[-1]
                
                for i in range(forecast_days):
                    next_price = last_price * (1 + avg_return)
                    forecasts.append({
                        'day': i + 1,
                        'predicted_price': float(next_price),
                        'confidence': max(0.3, 0.8 - i * 0.1)  # Decreasing confidence
                    })
                    last_price = next_price
                
                return {
                    'model_type': 'simple_trend',
                    'forecasts': forecasts,
                    'model_trained': False,
                    'avg_return': float(avg_return)
                }
            else:
                # Not enough data - use last price
                last_price = close_prices[-1] if len(close_prices) > 0 else 100.0
                forecasts = []
                
                for i in range(forecast_days):
                    forecasts.append({
                        'day': i + 1,
                        'predicted_price': float(last_price),
                        'confidence': 0.2
                    })
                
                return {
                    'model_type': 'last_value',
                    'forecasts': forecasts,
                    'model_trained': False
                }
                
        except Exception as e:
            self.logger.error(f"Error in simple prediction: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics."""
        try:
            return {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2': float(r2_score(y_true, y_pred)),
                'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            }
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def _generate_forecasts(
        self, 
        model: Any, 
        X: np.ndarray, 
        forecast_days: int,
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate future forecasts using trained model."""
        try:
            forecasts = []
            
            # Use last known values as starting point
            last_features = X[-1].copy()
            
            for day in range(forecast_days):
                # Predict next value
                prediction = model.predict([last_features])[0]
                
                # Calculate confidence (simple heuristic)
                confidence = max(0.3, 0.9 - day * 0.1)
                
                forecasts.append({
                    'day': day + 1,
                    'predicted_price': float(prediction),
                    'confidence': float(confidence)
                })
                
                # Update features for next prediction (simple approach)
                # In a more sophisticated system, you'd update all relevant features
                if 'Close' in feature_names:
                    close_idx = feature_names.index('Close')
                    last_features[close_idx] = prediction
                
            return forecasts
            
        except Exception as e:
            self.logger.error(f"Error generating forecasts: {str(e)}")
            return []
    
    def _generate_structured_results(
        self,
        processed_data: Dict[str, Any],
        prediction_results: Dict[str, Any],
        ticker: Optional[str],
        forecast_days: int,
        data_source: DataSource
    ) -> Dict[str, Any]:
        """Generate comprehensive structured results."""
        try:
            # Get current analysis
            if processed_data['type'] == 'single_ticker':
                summary = processed_data['summary']
                metadata = processed_data['metadata']
            else:
                first_ticker = list(processed_data['data'].keys())[0]
                summary = processed_data['combined_summary'][first_ticker]
                metadata = processed_data['data'][first_ticker]['metadata']
            
            # Determine trend
            forecasts = prediction_results.get('forecasts', [])
            trend = self._analyze_trend(forecasts)
            
            # Build structured result
            result = {
                'success': True,
                'ticker': ticker,
                'data_source': data_source.value,
                'analysis_date': datetime.now().isoformat(),
                'forecast_horizon': forecast_days,
                
                # Current market analysis
                'current_analysis': {
                    'summary': summary,
                    'data_period': {
                        'start': metadata['date_range']['start'],
                        'end': metadata['date_range']['end'],
                        'days': metadata['date_range']['days']
                    }
                },
                
                # Prediction results
                'predictions': {
                    'model_info': {
                        'type': prediction_results.get('model_type', 'unknown'),
                        'trained': prediction_results.get('model_trained', False),
                        'training_size': prediction_results.get('training_size', 0)
                    },
                    'validation_metrics': prediction_results.get('validation_metrics', {}),
                    'forecasts': forecasts,
                    'trend_analysis': trend
                },
                
                # LLM-ready explanation
                'explanation': self._generate_forecast_explanation(
                    summary, prediction_results, trend, ticker
                )
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating structured results: {str(e)}")
            return {'error': str(e), 'success': False}
    
    def _analyze_trend(self, forecasts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trend from forecasts."""
        if not forecasts or len(forecasts) < 2:
            return {'direction': 'unknown', 'strength': 0.0, 'confidence': 0.0}
        
        # Calculate price changes
        prices = [f['predicted_price'] for f in forecasts]
        changes = np.diff(prices)
        
        # Determine overall direction
        total_change = prices[-1] - prices[0]
        avg_confidence = np.mean([f['confidence'] for f in forecasts])
        
        if total_change > 0:
            direction = 'bullish'
        elif total_change < 0:
            direction = 'bearish'
        else:
            direction = 'sideways'
        
        # Calculate strength (normalized by first price)
        strength = abs(total_change) / prices[0] * 100 if prices[0] != 0 else 0
        
        return {
            'direction': direction,
            'strength': float(strength),
            'confidence': float(avg_confidence),
            'total_change_percent': float(total_change / prices[0] * 100) if prices[0] != 0 else 0.0
        }
    
    def _generate_forecast_explanation(
        self,
        summary: Dict[str, Any],
        prediction_results: Dict[str, Any],
        trend: Dict[str, Any],
        ticker: Optional[str]
    ) -> str:
        """Generate human-readable forecast explanation."""
        try:
            explanation = f"## Stock Forecast Analysis"
            
            if ticker:
                explanation += f" for {ticker}"
            
            explanation += "\n\n"
            
            # Current analysis
            if 'price_summary' in summary:
                ps = summary['price_summary']
                explanation += f"**Current Market Status:**\n"
                explanation += f"- Current Price: ${ps['current_price']:.2f}\n"
                explanation += f"- Total Return: {ps['total_return']:.2f}%\n"
                explanation += f"- Volatility: {ps['volatility']:.2f}%\n\n"
            
            # Technical indicators
            if 'rsi_signal' in summary:
                rsi = summary['rsi_signal']
                explanation += f"**Technical Indicators:**\n"
                explanation += f"- RSI: {rsi['value']:.2f} ({rsi['signal']})\n"
            
            if 'macd_signal' in summary:
                macd = summary['macd_signal']
                explanation += f"- MACD: {macd['trend']} trend\n"
            
            # Forecast results
            explanation += f"\n**Forecast Analysis:**\n"
            explanation += f"- Model: {prediction_results.get('model_type', 'Unknown')}\n"
            explanation += f"- Predicted Trend: {trend['direction'].upper()}\n"
            explanation += f"- Trend Strength: {trend['strength']:.2f}%\n"
            explanation += f"- Confidence: {trend['confidence']:.2f}\n"
            
            if trend['direction'] != 'sideways':
                explanation += f"- Expected Change: {trend['total_change_percent']:.2f}%\n"
            
            # Model performance
            metrics = prediction_results.get('validation_metrics', {})
            if metrics:
                explanation += f"\n**Model Performance:**\n"
                if 'r2' in metrics:
                    explanation += f"- R² Score: {metrics['r2']:.3f}\n"
                if 'mape' in metrics:
                    explanation += f"- Accuracy (MAPE): {100 - metrics['mape']:.1f}%\n"
            
            return explanation
            
        except Exception as e:
            return f"Error generating explanation: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            'available_models': list(self.models.keys()) if SKLEARN_AVAILABLE else [],
            'default_model': self.default_model,
            'sklearn_available': SKLEARN_AVAILABLE,
            'trained_models': list(self.trained_models.keys())
        }


class ExplanationAgent:
    """
    Advanced explanation agent that generates human-readable summaries of stock analysis.
    Integrates with MarketDataFetcher and ForecastingAgent to provide natural language insights.
    """
    
    def __init__(self, llm_model: Optional[Any] = None):
        """
        Initialize the ExplanationAgent.
        
        Args:
            llm_model: Pre-configured LLM model (uses xAI if available)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize LLM model
        try:
            from agno.models.xai import xAI
            self.llm = llm_model or xAI(id="grok-beta")
            self.llm_available = True
        except ImportError:
            self.logger.warning("xAI model not available. Using template-based explanations.")
            self.llm = None
            self.llm_available = False
        
        # Prompt templates
        self.prompt_templates = self._initialize_prompt_templates()
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize prompt templates for different explanation types."""
        return {
            'stock_analysis_summary': """
You are a professional financial analyst providing clear, concise stock analysis summaries.

Given the following stock data, generate a natural language summary that explains:
1. Recent price performance and trends
2. Technical indicator insights
3. Forecast predictions with confidence
4. Key support and resistance levels
5. Overall investment outlook

Stock Information:
- Ticker: {ticker}
- Current Price: ${current_price:.2f}
- Time Period: {time_period}
- Price Range: ${min_price:.2f} - ${max_price:.2f}
- Total Return: {total_return:.2f}%
- Volatility: {volatility:.2f}%

Technical Indicators:
{technical_indicators}

Forecast Analysis:
- Model Type: {model_type}
- Predicted Trend: {predicted_trend}
- Forecast Horizon: {forecast_days} days
- Confidence Level: {confidence:.2f}
- Expected Change: {expected_change:.2f}%

Detailed Forecasts:
{forecast_details}

Please provide a professional, clear summary in 2-3 paragraphs that a general investor could understand. Focus on actionable insights and avoid overly technical jargon.
""",
            
            'trend_explanation': """
Analyze the following stock trend data and provide a clear explanation:

Stock: {ticker}
Recent Performance: {recent_performance}
Technical Signals: {technical_signals}
Forecast: {forecast_summary}

Generate a single paragraph explaining the trend in simple terms.
""",
            
            'forecast_confidence': """
Explain the confidence level and reliability of this stock forecast:

Model: {model_type}
Validation Metrics: {validation_metrics}
Confidence Score: {confidence}
Data Quality: {data_quality}

Provide a brief explanation of what investors should consider about this forecast's reliability.
""",
            
            'risk_assessment': """
Provide a risk assessment for this stock based on the analysis:

Volatility: {volatility}%
Technical Indicators: {indicators}
Market Conditions: {market_conditions}
Forecast Uncertainty: {forecast_uncertainty}

Generate a risk assessment paragraph suitable for investment decision-making.
"""
        }
    
    def generate_stock_summary(
        self,
        forecast_result: Dict[str, Any],
        analysis_type: str = "comprehensive",
        include_technical_details: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive stock analysis summary.
        
        Args:
            forecast_result: Output from ForecastingAgent
            analysis_type: Type of analysis ('comprehensive', 'brief', 'technical')
            include_technical_details: Whether to include technical indicator details
            
        Returns:
            Dictionary containing generated explanations
        """
        try:
            if not forecast_result.get('success'):
                return {
                    'error': 'Invalid forecast result provided',
                    'success': False
                }
            
            # Extract key information
            ticker = forecast_result.get('ticker', 'Unknown')
            current_analysis = forecast_result.get('current_analysis', {})
            predictions = forecast_result.get('predictions', {})
            
            # Generate main summary
            main_summary = self._generate_main_summary(
                forecast_result, include_technical_details
            )
            
            # Generate additional analyses based on type
            result = {
                'success': True,
                'ticker': ticker,
                'analysis_type': analysis_type,
                'main_summary': main_summary,
                'generated_at': datetime.now().isoformat()
            }
            
            if analysis_type in ['comprehensive', 'technical']:
                result['trend_explanation'] = self._generate_trend_explanation(forecast_result)
                result['forecast_confidence'] = self._generate_confidence_explanation(predictions)
                result['risk_assessment'] = self._generate_risk_assessment(forecast_result)
            
            if analysis_type == 'comprehensive':
                result['technical_breakdown'] = self._generate_technical_breakdown(current_analysis)
                result['investment_recommendations'] = self._generate_investment_recommendations(forecast_result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating stock summary: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
    
    def _generate_main_summary(
        self,
        forecast_result: Dict[str, Any],
        include_technical: bool = True
    ) -> str:
        """Generate the main stock analysis summary."""
        try:
            # Extract data
            ticker = forecast_result.get('ticker', 'Unknown')
            current_analysis = forecast_result.get('current_analysis', {})
            predictions = forecast_result.get('predictions', {})
            
            summary_data = current_analysis.get('summary', {})
            price_summary = summary_data.get('price_summary', {})
            data_period = current_analysis.get('data_period', {})
            
            # Prepare technical indicators text
            technical_text = self._format_technical_indicators(summary_data) if include_technical else "Technical analysis included."
            
            # Prepare forecast details
            forecasts = predictions.get('forecasts', [])
            forecast_text = self._format_forecast_details(forecasts)
            
            # Prepare template variables
            template_vars = {
                'ticker': ticker,
                'current_price': price_summary.get('current_price', 0),
                'time_period': f"{data_period.get('days', 0)} days",
                'min_price': price_summary.get('min_price', 0),
                'max_price': price_summary.get('max_price', 0),
                'total_return': price_summary.get('total_return', 0),
                'volatility': price_summary.get('volatility', 0),
                'technical_indicators': technical_text,
                'model_type': predictions.get('model_info', {}).get('type', 'Unknown'),
                'predicted_trend': predictions.get('trend_analysis', {}).get('direction', 'unclear').title(),
                'forecast_days': forecast_result.get('forecast_horizon', 0),
                'confidence': predictions.get('trend_analysis', {}).get('confidence', 0),
                'expected_change': predictions.get('trend_analysis', {}).get('total_change_percent', 0),
                'forecast_details': forecast_text
            }
            
            # Generate explanation using LLM or template
            if self.llm_available:
                return self._call_llm_for_explanation(
                    self.prompt_templates['stock_analysis_summary'],
                    template_vars
                )
            else:
                return self._generate_template_summary(template_vars)
                
        except Exception as e:
            self.logger.error(f"Error in main summary generation: {str(e)}")
            return f"Error generating summary for {ticker}: {str(e)}"
    
    def _format_technical_indicators(self, summary_data: Dict[str, Any]) -> str:
        """Format technical indicators for natural language description."""
        indicators = []
        
        # RSI
        if 'rsi_signal' in summary_data:
            rsi = summary_data['rsi_signal']
            indicators.append(f"RSI: {rsi['value']:.1f} ({rsi['signal']})")
        
        # MACD
        if 'macd_signal' in summary_data:
            macd = summary_data['macd_signal']
            indicators.append(f"MACD: {macd['trend']} trend")
        
        # Moving Averages
        if 'moving_average_signals' in summary_data:
            ma = summary_data['moving_average_signals']
            indicators.append(f"Price vs SMA(20): {ma['price_vs_sma20']}")
            indicators.append(f"SMA Crossover: {ma['sma_crossover']}")
        
        return "; ".join(indicators) if indicators else "Technical indicators analyzed"
    
    def _format_forecast_details(self, forecasts: List[Dict[str, Any]]) -> str:
        """Format forecast details for natural language description."""
        if not forecasts:
            return "No specific forecasts available"
        
        details = []
        for i, forecast in enumerate(forecasts[:5]):  # First 5 days
            day = forecast.get('day', i + 1)
            price = forecast.get('predicted_price', 0)
            confidence = forecast.get('confidence', 0)
            details.append(f"Day {day}: ${price:.2f} (confidence: {confidence:.2f})")
        
        return "; ".join(details)
    
    def _call_llm_for_explanation(self, template: str, variables: Dict[str, Any]) -> str:
        """Call LLM to generate explanation using the template."""
        try:
            # Format the prompt
            formatted_prompt = template.format(**variables)
            
            # Call LLM (simulated - actual implementation would depend on agno framework)
            # This is a placeholder for the actual LLM call
            response = self._simulate_llm_response(variables)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling LLM: {str(e)}")
            return self._generate_template_summary(variables)
    
    def _simulate_llm_response(self, variables: Dict[str, Any]) -> str:
        """Simulate LLM response for demonstration purposes."""
        ticker = variables.get('ticker', 'the stock')
        days = variables.get('time_period', '30 days')
        return_pct = variables.get('total_return', 0)
        trend = variables.get('predicted_trend', 'uncertain')
        confidence = variables.get('confidence', 0)
        expected_change = variables.get('expected_change', 0)
        
        # Generate a response similar to the requested format
        if return_pct > 0:
            performance_desc = f"shown steady upward movement with an average gain of {abs(return_pct):.1f}%"
        elif return_pct < 0:
            performance_desc = f"experienced downward pressure with a decline of {abs(return_pct):.1f}%"
        else:
            performance_desc = "remained relatively stable"
        
        trend_desc = "continued growth" if trend.lower() == 'bullish' else "potential decline" if trend.lower() == 'bearish' else "sideways movement"
        
        confidence_desc = "high confidence" if confidence > 0.7 else "moderate confidence" if confidence > 0.4 else "lower confidence"
        
        summary = f"Over the past {days}, ${ticker} has {performance_desc}. "
        summary += f"The model forecasts {trend_desc} with {confidence_desc} (confidence: {confidence:.2f}), "
        summary += f"expecting a {abs(expected_change):.1f}% {'increase' if expected_change > 0 else 'decrease' if expected_change < 0 else 'stable price'} "
        summary += f"in the coming forecast period. "
        
        # Add technical context
        technical_indicators = variables.get('technical_indicators', '')
        if 'RSI' in technical_indicators and 'MACD' in technical_indicators:
            summary += "Technical indicators including RSI and MACD support this outlook, "
            summary += "with moving averages providing additional confirmation of the trend direction."
        
        return summary
    
    def _generate_template_summary(self, variables: Dict[str, Any]) -> str:
        """Generate summary using template without LLM."""
        ticker = variables.get('ticker', 'Unknown')
        total_return = variables.get('total_return', 0)
        trend = variables.get('predicted_trend', 'uncertain')
        confidence = variables.get('confidence', 0)
        days = variables.get('time_period', '30 days')
        
        summary = f"Analysis of ${ticker} over {days}: "
        
        if total_return > 2:
            summary += f"Strong positive performance with {total_return:.1f}% gain. "
        elif total_return > 0:
            summary += f"Modest positive performance with {total_return:.1f}% gain. "
        elif total_return < -2:
            summary += f"Negative performance with {abs(total_return):.1f}% decline. "
        else:
            summary += "Stable performance with minimal price movement. "
        
        summary += f"Forecast indicates {trend.lower()} trend with {confidence:.2f} confidence. "
        
        technical = variables.get('technical_indicators', '')
        if technical:
            summary += f"Technical analysis shows: {technical}."
        
        return summary
    
    def _generate_trend_explanation(self, forecast_result: Dict[str, Any]) -> str:
        """Generate trend-specific explanation."""
        try:
            predictions = forecast_result.get('predictions', {})
            trend_analysis = predictions.get('trend_analysis', {})
            
            direction = trend_analysis.get('direction', 'unclear')
            strength = trend_analysis.get('strength', 0)
            confidence = trend_analysis.get('confidence', 0)
            
            if direction == 'bullish':
                explanation = f"The stock shows a bullish trend with {strength:.1f}% strength. "
                explanation += f"This upward momentum is supported by technical indicators and has {confidence:.2f} confidence."
            elif direction == 'bearish':
                explanation = f"The stock exhibits a bearish trend with {strength:.1f}% downward pressure. "
                explanation += f"Technical signals suggest continued weakness with {confidence:.2f} confidence."
            else:
                explanation = f"The stock is in a sideways trend with low volatility. "
                explanation += f"Price action suggests consolidation with {confidence:.2f} confidence in the neutral outlook."
            
            return explanation
            
        except Exception as e:
            return f"Unable to generate trend explanation: {str(e)}"
    
    def _generate_confidence_explanation(self, predictions: Dict[str, Any]) -> str:
        """Generate explanation of forecast confidence."""
        try:
            model_info = predictions.get('model_info', {})
            validation_metrics = predictions.get('validation_metrics', {})
            trend_analysis = predictions.get('trend_analysis', {})
            
            model_type = model_info.get('type', 'unknown')
            r2_score = validation_metrics.get('r2', 0)
            confidence = trend_analysis.get('confidence', 0)
            
            explanation = f"This forecast uses a {model_type} model "
            
            if r2_score > 0.8:
                explanation += "with excellent predictive accuracy (R² > 0.8). "
            elif r2_score > 0.6:
                explanation += "with good predictive accuracy (R² > 0.6). "
            else:
                explanation += "with moderate predictive accuracy. "
            
            if confidence > 0.7:
                explanation += "The high confidence level suggests reliable predictions."
            elif confidence > 0.5:
                explanation += "The moderate confidence level indicates reasonable reliability with some uncertainty."
            else:
                explanation += "The lower confidence level suggests higher uncertainty and should be interpreted cautiously."
            
            return explanation
            
        except Exception as e:
            return f"Unable to generate confidence explanation: {str(e)}"
    
    def _generate_risk_assessment(self, forecast_result: Dict[str, Any]) -> str:
        """Generate risk assessment explanation."""
        try:
            current_analysis = forecast_result.get('current_analysis', {})
            summary_data = current_analysis.get('summary', {})
            price_summary = summary_data.get('price_summary', {})
            
            volatility = price_summary.get('volatility', 0)
            
            if volatility > 30:
                risk_level = "high"
                risk_desc = "significant price swings and higher uncertainty"
            elif volatility > 15:
                risk_level = "moderate"
                risk_desc = "reasonable price stability with some fluctuation"
            else:
                risk_level = "low"
                risk_desc = "relatively stable price action"
            
            assessment = f"Risk Assessment: {risk_level.title()} risk level with {volatility:.1f}% volatility indicating {risk_desc}. "
            
            # Add technical risk factors
            if 'rsi_signal' in summary_data:
                rsi = summary_data['rsi_signal']
                if rsi['signal'] in ['Overbought', 'Oversold']:
                    assessment += f"RSI signals {rsi['signal'].lower()} conditions, suggesting potential reversal risk. "
            
            assessment += "Consider position sizing and risk management strategies appropriate for this volatility level."
            
            return assessment
            
        except Exception as e:
            return f"Unable to generate risk assessment: {str(e)}"
    
    def _generate_technical_breakdown(self, current_analysis: Dict[str, Any]) -> str:
        """Generate detailed technical analysis breakdown."""
        try:
            summary_data = current_analysis.get('summary', {})
            breakdown = "Technical Analysis Breakdown: "
            
            # Price action
            price_summary = summary_data.get('price_summary', {})
            if price_summary:
                breakdown += f"Price trading at ${price_summary.get('current_price', 0):.2f}, "
                breakdown += f"within a range of ${price_summary.get('min_price', 0):.2f} - ${price_summary.get('max_price', 0):.2f}. "
            
            # Technical indicators
            indicators = []
            
            if 'rsi_signal' in summary_data:
                rsi = summary_data['rsi_signal']
                indicators.append(f"RSI({rsi['value']:.1f}) suggests {rsi['signal'].lower()} conditions")
            
            if 'macd_signal' in summary_data:
                macd = summary_data['macd_signal']
                indicators.append(f"MACD shows {macd['trend'].lower()} momentum")
            
            if 'moving_average_signals' in summary_data:
                ma = summary_data['moving_average_signals']
                indicators.append(f"price {ma['price_vs_sma20'].lower()} key moving averages")
            
            if indicators:
                breakdown += "Key indicators: " + ", ".join(indicators) + "."
            
            return breakdown
            
        except Exception as e:
            return f"Unable to generate technical breakdown: {str(e)}"
    
    def _generate_investment_recommendations(self, forecast_result: Dict[str, Any]) -> str:
        """Generate investment recommendations based on analysis."""
        try:
            predictions = forecast_result.get('predictions', {})
            trend_analysis = predictions.get('trend_analysis', {})
            current_analysis = forecast_result.get('current_analysis', {})
            
            direction = trend_analysis.get('direction', 'uncertain')
            confidence = trend_analysis.get('confidence', 0)
            strength = trend_analysis.get('strength', 0)
            
            recommendations = "Investment Considerations: "
            
            if direction == 'bullish' and confidence > 0.7:
                recommendations += "Strong bullish signals suggest potential buying opportunities. "
                recommendations += "Consider gradual position building on pullbacks."
            elif direction == 'bearish' and confidence > 0.7:
                recommendations += "Bearish outlook suggests caution. "
                recommendations += "Consider profit-taking or defensive positioning."
            else:
                recommendations += "Mixed or uncertain signals suggest a wait-and-see approach. "
                recommendations += "Monitor for clearer directional confirmation."
            
            # Add risk management
            summary_data = current_analysis.get('summary', {})
            volatility = summary_data.get('price_summary', {}).get('volatility', 0)
            
            if volatility > 20:
                recommendations += " High volatility warrants smaller position sizes and tight risk management."
            
            recommendations += " Always consider your investment timeline and risk tolerance."
            
            return recommendations
            
        except Exception as e:
            return f"Unable to generate investment recommendations: {str(e)}"
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """Get available prompt templates."""
        return self.prompt_templates.copy()
    
    def update_prompt_template(self, template_name: str, new_template: str) -> bool:
        """Update a prompt template."""
        try:
            self.prompt_templates[template_name] = new_template
            return True
        except Exception as e:
            self.logger.error(f"Error updating template {template_name}: {str(e)}")
            return False


# create the AI finance agent
agent = Agent(
    name="xAI Finance Agent",
    model = xAI(id="grok-beta"),
    tools=[DuckDuckGoTools(), YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions = ["Always use tables to display financial/numerical data. For text data use bullet points and small paragrpahs."],
    show_tool_calls = True,
    markdown = True,
    )

# UI for finance agent
app = Playground(agents=[agent]).get_app()

# Example usage of complete pipeline including ForecastingAgent
def example_usage():
    """Complete usage examples with data fetching, preprocessing, and forecasting."""
    print("=== Complete Finance AI Pipeline Example ===\n")
    
    # Create instances
    fetcher = MarketDataFetcher(cache_duration_hours=2)
    preprocessor = MarketDataPreprocessor()
    forecasting_agent = ForecastingAgent(fetcher=fetcher, preprocessor=preprocessor)
    
    # Example 1: Real-time forecasting
    print("1. Real-Time Stock Forecasting:")
    try:
        forecast_result = forecasting_agent.forecast(
            ticker="AAPL",
            time_window=60,
            forecast_days=5,
            data_source=DataSource.REAL_TIME,
            model_type="random_forest"
        )
        
        if forecast_result.get('success'):
            print(f"   ✓ Forecast generated for {forecast_result['ticker']}")
            print(f"   ✓ Model: {forecast_result['predictions']['model_info']['type']}")
            print(f"   ✓ Trend: {forecast_result['predictions']['trend_analysis']['direction']}")
            print(f"   ✓ Confidence: {forecast_result['predictions']['trend_analysis']['confidence']:.2f}")
            
            # Show first few predictions
            forecasts = forecast_result['predictions']['forecasts'][:3]
            print("   ✓ Predictions (first 3 days):")
            for f in forecasts:
                print(f"      Day {f['day']}: ${f['predicted_price']:.2f} (confidence: {f['confidence']:.2f})")
                
        else:
            print(f"   ✗ Forecast failed: {forecast_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ✗ Error in real-time forecasting: {str(e)}")
    print()
    
    # Example 2: CSV-based forecasting (backward compatibility)
    print("2. CSV-Based Forecasting (Backward Compatibility):")
    try:
        # Create a sample CSV for demonstration (normally you'd have an existing file)
        sample_data = fetcher.fetch_stock_data("GOOGL", days=30)
        if sample_data is not None:
            csv_path = "sample_stock_data.csv"
            sample_data.to_csv(csv_path)
            
            # Test CSV forecasting
            csv_forecast = forecasting_agent.forecast(
                csv_path=csv_path,
                forecast_days=3,
                data_source=DataSource.CSV,
                model_type="linear"
            )
            
            if csv_forecast.get('success'):
                print(f"   ✓ CSV forecast successful")
                print(f"   ✓ Data source: {csv_forecast['data_source']}")
                print(f"   ✓ Model: {csv_forecast['predictions']['model_info']['type']}")
                print(f"   ✓ Trend: {csv_forecast['predictions']['trend_analysis']['direction']}")
            else:
                print(f"   ✗ CSV forecast failed: {csv_forecast.get('error', 'Unknown error')}")
                
            # Clean up sample file
            import os
            if os.path.exists(csv_path):
                os.remove(csv_path)
                
    except Exception as e:
        print(f"   ✗ Error in CSV forecasting: {str(e)}")
    print()
    
    # Example 3: Model comparison
    print("3. Model Comparison:")
    model_info = forecasting_agent.get_model_info()
    print(f"   Available models: {model_info['available_models']}")
    print(f"   Default model: {model_info['default_model']}")
    print(f"   Sklearn available: {model_info['sklearn_available']}")
    
    if model_info['available_models']:
        print("   Testing different models:")
        
        for model_type in model_info['available_models'][:2]:  # Test first 2 models
            try:
                test_result = forecasting_agent.forecast(
                    ticker="MSFT",
                    time_window=45,
                    forecast_days=3,
                    model_type=model_type
                )
                
                if test_result.get('success'):
                    trend = test_result['predictions']['trend_analysis']
                    metrics = test_result['predictions'].get('validation_metrics', {})
                    
                    print(f"      {model_type}: {trend['direction']} trend")
                    if 'r2' in metrics:
                        print(f"         R² Score: {metrics['r2']:.3f}")
                        
            except Exception as e:
                print(f"      {model_type}: Error - {str(e)}")
    print()
    
    # Example 4: Complete analysis with explanation
    print("4. Complete Analysis with LLM Explanation:")
    try:
        detailed_forecast = forecasting_agent.forecast(
            ticker="TSLA",
            time_window=90,
            forecast_days=7,
            data_source=DataSource.REAL_TIME
        )
        
        if detailed_forecast.get('success'):
            print("   ✓ Analysis completed successfully")
            
            # Show structured results
            current_analysis = detailed_forecast['current_analysis']
            predictions = detailed_forecast['predictions']
            
            print(f"   ✓ Analysis period: {current_analysis['data_period']['days']} days")
            print(f"   ✓ Model performance (R²): {predictions.get('validation_metrics', {}).get('r2', 'N/A')}")
            
            # Show LLM explanation
            explanation = detailed_forecast.get('explanation', '')
            if explanation:
                print("   ✓ LLM Explanation (excerpt):")
                lines = explanation.split('\n')[:8]  # First 8 lines
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
                        
    except Exception as e:
        print(f"   ✗ Error in detailed analysis: {str(e)}")
    print()
    
    # Example 5: ExplanationAgent with real-time data
    print("5. ExplanationAgent with Real-Time Analysis:")
    try:
        explanation_agent = ExplanationAgent()
        
        # Get forecast result for explanation
        forecast_for_explanation = forecasting_agent.forecast(
            ticker="AAPL",
            time_window=60,
            forecast_days=5,
            data_source=DataSource.REAL_TIME
        )
        
        if forecast_for_explanation.get('success'):
            # Generate comprehensive explanation
            explanation_result = explanation_agent.generate_stock_summary(
                forecast_for_explanation,
                analysis_type="comprehensive",
                include_technical_details=True
            )
            
            if explanation_result.get('success'):
                print(f"   ✓ Explanation generated for {explanation_result['ticker']}")
                print(f"   ✓ Analysis type: {explanation_result['analysis_type']}")
                
                # Show main summary (like the requested format)
                main_summary = explanation_result.get('main_summary', '')
                if main_summary:
                    print("   ✓ Main Summary:")
                    print(f"      {main_summary}")
                
                # Show additional analyses
                if 'trend_explanation' in explanation_result:
                    print("   ✓ Trend Analysis:")
                    print(f"      {explanation_result['trend_explanation']}")
                
                if 'risk_assessment' in explanation_result:
                    print("   ✓ Risk Assessment:")
                    print(f"      {explanation_result['risk_assessment']}")
                    
            else:
                print(f"   ✗ Explanation failed: {explanation_result.get('error', 'Unknown error')}")
        else:
            print("   ✗ Could not generate forecast for explanation")
            
    except Exception as e:
        print(f"   ✗ Error in explanation generation: {str(e)}")
    print()
    
    # Example 6: Prompt Templates and Customization
    print("6. Prompt Templates and Customization:")
    try:
        explanation_agent = ExplanationAgent()
        templates = explanation_agent.get_prompt_templates()
        
        print(f"   Available templates: {list(templates.keys())}")
        
        # Show a sample template
        trend_template = templates.get('trend_explanation', '')
        if trend_template:
            print("   Sample trend explanation template:")
            lines = trend_template.strip().split('\n')[:5]
            for line in lines:
                if line.strip():
                    print(f"      {line}")
            print("      ...")
        
        # Demonstrate template customization
        custom_template = """
Provide a brief stock analysis for {ticker}:
Recent performance: {recent_performance}
Technical outlook: {technical_signals}
Forecast summary: {forecast_summary}

Generate a single, clear investment insight.
"""
        
        success = explanation_agent.update_prompt_template('custom_brief', custom_template)
        print(f"   Custom template added: {success}")
        
    except Exception as e:
        print(f"   Error in template demonstration: {str(e)}")
    print()
    
    # Example 7: Complete integrated pipeline
    print("7. Complete Integrated Pipeline (Fetcher -> Preprocessor -> Forecaster -> Explainer):")
    try:
        # Step 1: Fetch data
        print("   Step 1: Fetching real-time data...")
        raw_data = fetcher.fetch_stock_data("MSFT", days=45)
        
        if raw_data is not None:
            # Step 2: Preprocess
            print("   Step 2: Preprocessing with technical indicators...")
            processed = preprocessor.preprocess_data(raw_data)
            
            if 'error' not in processed:
                # Step 3: Generate forecast
                print("   Step 3: Generating ML forecast...")
                forecast = forecasting_agent.forecast(
                    ticker="MSFT",
                    time_window=45,
                    forecast_days=3,
                    data_source=DataSource.REAL_TIME,
                    model_type="random_forest"
                )
                
                if forecast.get('success'):
                    # Step 4: Generate explanation
                    print("   Step 4: Generating natural language explanation...")
                    explanation = ExplanationAgent().generate_stock_summary(
                        forecast,
                        analysis_type="brief"
                    )
                    
                    if explanation.get('success'):
                        print("   ✓ Complete pipeline successful!")
                        print(f"   ✓ Final summary: {explanation['main_summary'][:150]}...")
                    else:
                        print(f"   ✗ Explanation step failed: {explanation.get('error')}")
                else:
                    print(f"   ✗ Forecast step failed: {forecast.get('error')}")
            else:
                print(f"   ✗ Preprocessing step failed: {processed.get('error')}")
        else:
            print("   ✗ Data fetching step failed")
            
    except Exception as e:
        print(f"   ✗ Error in integrated pipeline: {str(e)}")
    
    print("\n=== Complete Finance AI Pipeline Examples Completed ===")  


if __name__ == "__main__":
    # Uncomment the line below to run the example
    # example_usage()
    
    # Run the main agent application
    serve_playground_app("xai_finance_agent:app", reload=True)