# finance-assistant/agents/analysis_agent.py
"""
Analysis Agent - Quantitative Analysis
Handles financial calculations, technical analysis, and statistical modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AnalysisResult:
    metric_name: str
    value: float
    interpretation: str
    confidence_level: float
    supporting_data: Dict

@dataclass
class PortfolioMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    beta: float
    alpha: float

class AnalysisAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        try:
            result_df = df.copy()
            
            # Price-based indicators
            result_df['SMA_10'] = df['Close'].rolling(window=10).mean()
            result_df['SMA_20'] = df['Close'].rolling(window=20).mean()
            result_df['SMA_50'] = df['Close'].rolling(window=50).mean()
            result_df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            result_df['EMA_12'] = df['Close'].ewm(span=12).mean()
            result_df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # Bollinger Bands
            sma_20 = result_df['SMA_20']
            std_20 = df['Close'].rolling(window=20).std()
            result_df['BB_Upper'] = sma_20 + (std_20 * 2)
            result_df['BB_Lower'] = sma_20 - (std_20 * 2)
            result_df['BB_Width'] = result_df['BB_Upper'] - result_df['BB_Lower']
            result_df['BB_Position'] = (df['Close'] - result_df['BB_Lower']) / result_df['BB_Width']
            
          
