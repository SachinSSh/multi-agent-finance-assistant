# data_ingestion/__init__.py
"""
Financial Data Ingestion Module
Handles market data, SEC filings, and vector embeddings
"""

from .market_data import MarketDataLoader, YahooFinanceLoader, AlphaVantageLoader
from .document_loaders import SECFilingLoader, DocumentProcessor
from .embeddings import EmbeddingManager, VectorStore

__all__ = [
    'MarketDataLoader',
    'YahooFinanceLoader', 
    'AlphaVantageLoader',
    'SECFilingLoader',
    'DocumentProcessor',
    'EmbeddingManager',
    'VectorStore'
]

__version__ = "1.0.0"
