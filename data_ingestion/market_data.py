
# data_ingestion/market_data.py
"""
Market Data API Loaders
Handles real-time and historical financial data from various sources
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from abc import ABC, abstractmethod
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataLoader(ABC):
    """Abstract base class for market data loaders"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.rate_limit_delay = 0.1  # Default delay between requests
    
    @abstractmethod
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical stock data"""
        pass
    
    @abstractmethod
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time stock quote"""
        pass
    
    def _handle_rate_limit(self):
        """Handle API rate limiting"""
        time.sleep(self.rate_limit_delay)


class YahooFinanceLoader(MarketDataLoader):
    """Yahoo Finance data loader using yfinance library"""
    
    def __init__(self):
        super().__init__()
        self.rate_limit_delay = 0.5
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical stock data from Yahoo Finance
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            data['symbol'] = symbol
            data['timestamp'] = data.index
            
            logger.info(f"Successfully loaded {len(data)} records for {symbol}")
            self._handle_rate_limit()
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            raise
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote = {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'change': info.get('regularMarketChange'),
                'change_percent': info.get('regularMarketChangePercent'),
                'volume': info.get('regularMarketVolume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'timestamp': datetime.now().isoformat()
            }
            
            self._handle_rate_limit()
            return quote
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            raise
    
    def get_company_info(self, symbol: str) -> Dict:
        """Get company information and fundamentals"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_data = {
                'symbol': symbol,
                'company_name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'description': info.get('longBusinessSummary'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'revenue': info.get('totalRevenue'),
                'profit_margins': info.get('profitMargins'),
                'employees': info.get('fullTimeEmployees')
            }
            
            self._handle_rate_limit()
            return company_data
            
        except Exception as e:
            logger.error(f"Error getting company info for {symbol}: {str(e)}")
            raise


class AlphaVantageLoader(MarketDataLoader):
    """Alpha Vantage API loader for comprehensive market data"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Free tier: 5 calls per minute
    
    def get_stock_data(self, symbol: str, period: str = "daily") -> pd.DataFrame:
        """
        Get historical stock data from Alpha Vantage
        
        Args:
            symbol: Stock ticker symbol
            period: 'intraday', 'daily', 'weekly', 'monthly'
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            function_map = {
                'daily': 'TIME_SERIES_DAILY_ADJUSTED',
                'weekly': 'TIME_SERIES_WEEKLY_ADJUSTED',
                'monthly': 'TIME_SERIES_MONTHLY_ADJUSTED',
                'intraday': 'TIME_SERIES_INTRADAY'
            }
            
            params = {
                'function': function_map.get(period, 'TIME_SERIES_DAILY_ADJUSTED'),
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }
            
            if period == 'intraday':
                params['interval'] = '5min'
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract time series data
            time_series_key = [key for key in data.keys() if 'Time Series' in key][0]
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high', 
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adj_close',
                '6. volume': 'volume',
                '5. volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            df = df.astype(float)
            df['symbol'] = symbol
            df['timestamp'] = df.index
            
            logger.info(f"Successfully loaded {len(df)} records for {symbol}")
            self._handle_rate_limit()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Alpha Vantage data for {symbol}: {str(e)}")
            raise
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Alpha Vantage"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()['Global Quote']
            
            quote = {
                'symbol': data['01. symbol'],
                'price': float(data['05. price']),
                'change': float(data['09. change']),
                'change_percent': data['10. change percent'].rstrip('%'),
                'volume': int(data['06. volume']),
                'timestamp': datetime.now().isoformat()
            }
            
            self._handle_rate_limit()
            return quote
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage quote for {symbol}: {str(e)}")
            raise
    
    def get_company_overview(self, symbol: str) -> Dict:
        """Get company overview and fundamentals"""
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            self._handle_rate_limit()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting company overview for {symbol}: {str(e)}")
            raise

