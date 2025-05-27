# agents/api_agent.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import asyncio
import redis
import json
import logging
import time
from datetime import datetime, timedelta
import os
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Agent - Market Data Service",
    description="Real-time and historical market data fetching agent",
    version="1.0.0"
)

# Configuration
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "demo")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Initialize services
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
    fd = FundamentalData(key=ALPHA_VANTAGE_KEY)
except Exception as e:
    logger.error(f"API Agent initialization error: {e}")
    raise HTTPException(status_code=500, detail="API Agent initialization failed")

# Request/Response models
class MarketDataRequest(BaseModel):
    query: str
    type: str
    portfolio_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = {}

class MarketDataResponse(BaseModel):
    data: Dict[str, Any]
    confidence: float
    sources: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    timestamp: datetime

# API agent logic
class ApiAgent:
    def __init__(self):
        self.client = requests.Session()
        self.confidence_threshold = 0.7
        
    async def route_query(self, request: MarketDataRequest) -> MarketDataResponse:
        """Main orchestration logic for processing queries"""
        start_time = time.time()
        
        try:
            # Step 1: Process query
            query = request.query
            
            # Step 2: Fetch market data
            data = await self.fetch_market_data(query, request.type)
            
            # Step 3: Compile final response
            total_latency = time.time() - start_time
            
            return MarketDataResponse(
                data=data,
                confidence=1.0,
                sources=[],
                metrics={
                    "total_latency": total_latency,
                    "agent_latencies": {},
                    "agents_used": ["api_agent"]
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Orchestration error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def fetch_market_data(self, query: str, data_type: str) -> Dict[str, Any]:
        """Fetch market data from various sources"""
        # Fetch data from Yahoo Finance
        yf_data = await self.fetch_yahoo_finance(query)
        
        if yf_data:
            return {"yahoo_finance": yf_data}
        
        # Fetch data from Alpha Vantage
        av_data = await self.fetch_alpha_vantage(query)
        
        if av_data:
            return {"alpha_vantage": av_data}
        
        # Fetch data from Google Finance
        gf_data = await self.fetch_google_finance(query)
        
        if gf_data:
            return {"google_finance": gf_data}
        
        # Fetch data from IEX Cloud
        iex_data = await self.fetch_iex_cloud(query)
        
        if iex_data:
            return {"iex_cloud": iex_data}
        
        # Fetch data from Quandl
        quandl_data = await self.fetch_quandl(query)
        
        if quandl_data:
            return {"quandl": quandl_data}
        
        raise HTTPException(status_code=500, detail="Unable to fetch market data")
    
    async def fetch_yahoo_finance(self, query: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Yahoo Finance"""
        try:
            # Fetch data from Yahoo Finance
            yf_data = yf.Ticker(query)
            
            # Extract relevant data
            data = {
                "name": yf_data.info["longName"],
                "price": yf_data.info["regularMarketPrice"],
                "change": yf_data.info["regularMarketChange"],
                "volume": yf_data.info["regularMarketVolume"],
                "open": yf_data.info["regularMarketOpen"],
                "high": yf_data.info["regularMarketDayHigh"],
                "low": yf_data.info["regularMarketDayLow"],
                "market_cap": yf_data.info["marketCap"],
                "dividend_yield": yf_data.info["dividendYield"],
                "pe_ratio": yf_data.info["trailingPE"],
                "eps": yf_data.info["trailingEps"],
                "beta": yf_data.info["trailingBeta"],
                "short_ratio": yf_data.info["shortRatio"],
                "shares_outstanding": yf_data.info["sharesOutstanding"],
                "avg_volume": yf_data.info["averageDailyVolume10Day"],
                "avg_volume_2y": yf_data.info["averageDailyVolume2Week"],
                "avg_volume_4y": yf_data.info["averageDailyVolume4Week"],
                "avg_volume_6y": yf_data.info["averageDailyVolume6Month"],
                "prev_close": yf_data.info["previousClose"],            
                "52_week_high": yf_data.info["fiftyTwoWeekHigh"],
                "52_week_low": yf_data.info["fiftyTwoWeekLow"],
                "50day_moving_avg": yf_data.info["fiftyDayAverage"],
                "200day_moving_avg": yf_data.info["twoHundredDayAverage"],
                "short_interest": yf_data.info["shortInterest"],
                "short_ratio": yf_data.info["shortRatio"],
                "trailing_pe": yf_data.info["trailingPE"],
                "trailing_eps": yf_data.info["trailingEps"],
                "trailing_beta": yf_data.info["trailingBeta"],
                "dividend_yield": yf_data.info["dividendYield"],
                "dividend_per_share": yf_data.info["dividendPerShare"],
                "ex_dividend_yield": yf_data.info["trailingAnnualDividendYield"],
                "next_dividend_date": yf_data.info["nextFiscalQuarterDate"],    
                "next_earnings_date": yf_data.info["nextEarningsDate"],
                "earnings_date": yf_data.info["earningsDate"],
                "market_cap": yf_data.info["marketCap"],
                "book_value": yf_data.info["bookValue"],
                "enterprise_value": yf_data.info["enterpriseValue"],
                "payout_ratio": yf_data.info["payoutRatio"],
                "forward_pe": yf_data.info["forwardPE"],
                "forward_eps": yf_data.info["forwardEps"],
                "forward_ppe": yf_data.info["forwardPPE"],
                "forward_pps": yf_data.info["forwardPPS"],
                "forward_dividend_yield": yf_data.info["forwardDividendYield"],
                "forward_dividend_payout": yf_data.info["forwardDividendPayout"],
                "implied_volatility": yf_data.info["impliedVolatility"],
                "implied_volatility_12m": yf_data.info["impliedVolatility12Month"],
                "implied_volatility_1y": yf_data.info["impliedVolatility1Year"],
                "implied_volatility_2y": yf_data.info["impliedVolatility2Year"],
                "implied_volatility_3y": yf_data.info["impliedVolatility3Year"],
                "implied_volatility_4y": yf_data.info["impliedVolatility4Year"],
                "last_fiscal_quarter": yf_data.info["lastFiscalQuarter"],
                "last_fiscal_quarter_end": yf_data.info["lastFiscalQuarterEnd"],
                "fiscal_year_end": yf_data.info["nextFiscalYearEnd"],
                "trailing_pv_ratio": yf_data.info["trailingPVRatio"],
                "trailing_pv_ratio_12m": yf_data.info["trailingPVRatio12Month"],
                "trailing_pv_ratio_1y": yf_data.info["trailingPVRatio1Year"],
                "trailing_pv_ratio_2y": yf_data.info["trailingPVRatio2Year"],
                "trailing_pv_ratio_3y": yf_data.info["trailingPVRatio3Year"],
                "trailing_pv_ratio_4y": yf_data.info["trailingPVRatio4Year"],
                "trailing_pv_ratio_5y": yf_data.info["trailingPVRatio5Year"],
                "trailing_pv_ratio_6y": yf_data.info["trailingPVRatio6Year"],
                "trailing_pv_ratio_7y": yf_data.info["trailingPVRatio7Year"],
                "trailing_pv_ratio_8y": yf_data.info["trailingPVRatio8Year"],
                "trailing_pv_ratio_9y": yf_data.info["trailingPVRatio9Year"],
                "trailing_pv_ratio_10y": yf_data.info["trailingPVRatio10Year"],
                "trailing_pv_ratio_11y": yf_data.info["trailingPVRatio11Year"],
                "trailing_pv_ratio_12y": yf_data.info["trailingPVRatio12Year"],
                "trailing_pv_ratio_13y": yf_data.info["trailingPVRatio13Year"],
                "trailing_pv_ratio_14y": yf_data.info["trailingPVRatio14Year"],
                "trailing_pv_ratio_15y": yf_data.info["trailingPVRatio15Year"],
                "trailing_pv_ratio_16y": yf_data.info["trailingPVRatio16Year"],
                "trailing_pv_ratio_17y": yf_data.info["trailingPVRatio17Year"],
                "trailing_pv_ratio_18y": yf_data.info["trailingPVRatio18Year"],
                "trailing_pv_ratio_19y": yf_data.info["trailingPVRatio19Year"],
                "trailing_pv_ratio_20y": yf_data.info["trailingPVRatio20Year"],
                "trailing_pv_ratio_21y": yf_data.info["trailingPVRatio21Year"]
            }
            
            return data
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return None

# Initialize services
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')
    fd = FundamentalData(key=ALPHA_VANTAGE_KEY, output_format='pandas')
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
    redis_client = None

# Request/Response models
class MarketDataRequest(BaseModel):
    query: str
    symbols: Optional[List[str]] = []
    type: str = "portfolio_exposure"
    timeframe: str = "1d"
    include_fundamentals: bool = False

class PortfolioExposureRequest(BaseModel):
    portfolio_id: Optional[str] = None
    region: Optional[str] = None
    sector: Optional[str] = None

class MarketDataResponse(BaseModel):
    data: Dict[str, Any]
    confidence: float
    sources: List[str]
    timestamp: datetime
    cache_hit: bool = False

# Asia Tech Stock symbols for demo
ASIA_TECH_SYMBOLS = {
    'TSMC': 'TSM',      # Taiwan Semiconductor
    'Samsung': '005930.KS',  # Samsung Electronics
    'ASML': 'ASML',     # ASML Holding
    'Tencent': '0700.HK',    # Tencent
    'Alibaba': 'BABA',  # Alibaba
    'NVDA': 'NVDA',     # NVIDIA (has Asia exposure)
    'Sony': 'SONY',     # Sony
    'Nintendo': 'NTDOY' # Nintendo
}

class MarketDataService:
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes cache
        
    async def get_portfolio_exposure(self, request: MarketDataRequest) -> Dict[str, Any]:
        """Get portfolio exposure data for Asia tech stocks"""
        cache_key = f"portfolio_exposure:{request.query}"
        
        # Try cache first
        if redis_client:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info("Cache hit for portfolio exposure")
                return {**json.loads(cached_data), "cache_hit": True}
        
        try:
            # Simulate portfolio data - in production, connect to portfolio management system
            portfolio_data = await self._fetch_asia_tech_exposure()
            
            # Get current market data
            market_data = await self._fetch_current_prices(list(ASIA_TECH_SYMBOLS.values()))
            
            # Calculate exposure metrics
            exposure_metrics = self._calculate_exposure_metrics(portfolio_data, market_data)
            
            result = {
                "portfolio_exposure": exposure_metrics,
                "current_prices": market_data,
                "timestamp": datetime.now().isoformat(),
                "cache_hit": False
            }
            
            # Cache result
            if redis_client:
                redis_client.setex(cache_key, self.cache_ttl, json.dumps(result, default=str))
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching portfolio exposure: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _fetch_asia_tech_exposure(self) -> Dict[str, Any]:
        """Simulate fetching portfolio exposure data"""
        # In production, this would connect to your portfolio management system
        return {
            "total_aum": 10000000,  # $10M AUM
            "asia_tech_allocation": {
                "current_percentage": 22.0,
                "previous_percentage": 18.0,
                "absolute_value": 2200000,
                "change_24h": 4.0
            },
            "holdings": {
                "TSM": {"shares": 1000, "weight": 8.5},
                "005930.KS": {"shares": 500, "weight": 6.2},
                "BABA": {"shares": 800, "weight": 4.1},
                "0700.HK": {"shares": 1200, "weight": 3.2}
            }
        }
    
    async def _fetch_current_prices(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch current market prices using yfinance"""
        prices = {}
        
        try:
            # Use yfinance for real-time data
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    hist = ticker.history(period="2d")  # Get last 2 days for comparison
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        previous_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change_pct = ((current_price - previous_price) / previous_price) * 100
                        
                        prices[symbol] = {
                            "current_price": float(current_price),
                            "previous_close": float(previous_price),
                            "change_percent": float(change_pct),
                            "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                            "market_cap": info.get('marketCap', 0),
                            "currency": info.get('currency', 'USD')
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    # Provide fallback data
                    prices[symbol] = {
                        "current_price": 0.0,
                        "previous_close": 0.0,
                        "change_percent": 0.0,
                        "volume": 0,
                        "market_cap": 0,
                        "currency": "USD",
                        "error": str(e)
                    }
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching market prices: {e}")
            raise
    
    def _calculate_exposure_metrics(self, portfolio_data: Dict, market_data: Dict) -> Dict[str, Any]:
        """Calculate portfolio exposure metrics"""
        asia_tech = portfolio_data["asia_tech_allocation"]
        
        # Calculate weighted performance
        total_weighted_return = 0.0
        total_weight = 0.0
        
        for symbol, holding in portfolio_data["holdings"].items():
            if symbol in market_data:
                weight = holding["weight"] / 100  # Convert to decimal
                return_pct = market_data[symbol]["change_percent"]
                total_weighted_return += weight * return_pct
                total_weight += weight
        
        avg_return = total_weighted_return / total_weight if total_weight > 0 else 0.0
        
        return {
            "percentage": asia_tech["current_percentage"],
            "previous_percentage": asia_tech["previous_percentage"],
            "change_percentage": asia_tech["current_percentage"] - asia_tech["previous_percentage"],
            "absolute_value": asia_tech["absolute_value"],
            "weighted_return_24h": avg_return,
            "top_holdings": [
                {
                    "symbol": symbol,
                    "weight": holding["weight"],
                    "current_return": market_data.get(symbol, {}).get("change_percent", 0.0)
                }
                for symbol, holding in sorted(
                    portfolio_data["holdings"].items(),
                    key=lambda x: x[1]["weight"],
                    reverse=True
                )[:5]
            ]
        }
    
    async def get_fundamental_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get fundamental data for stocks"""
        fundamentals = {}
        
        for symbol in symbols:
            cache_key = f"fundamentals:{symbol}"
            
            # Check cache
            if redis_client:
                cached = redis_client.get(cache_key)
                if cached:
                    fundamentals[symbol] = json.loads(cached)
                    continue
            
            try:
                # Use Alpha Vantage for fundamentals
                overview, _ = fd.get_company_overview(symbol)
                
                if not overview.empty:
                    fundamental_data = {
                        "pe_ratio": float(overview.get('PERatio', 0) or 0),
                        "market_cap": int(overview.get('MarketCapitalization', 0) or 0),
                        "dividend_yield": float(overview.get('DividendYield', 0) or 0),
                        "eps": float(overview.get('EPS', 0) or 0),
                        "revenue_ttm": int(overview.get('RevenueTTM', 0) or 0),
                        "profit_margin": float(overview.get('ProfitMargin', 0) or 0)
                    }
                    
                    fundamentals[symbol] = fundamental_data
                    
                    # Cache for longer period (1 hour)
                    if redis_client:
                        redis_client.setex(cache_key, 3600, json.dumps(fundamental_data))
                        
            except Exception as e:
                logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
                fundamentals[symbol] = {"error": str(e)}
        
        return fundamentals
    
    async def get_earnings_calendar(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get upcoming earnings calendar"""
        if not symbols:
            symbols = list(ASIA_TECH_SYMBOLS.values())
        
        earnings_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                calendar = ticker.calendar
                
                if calendar is not None and not calendar.empty:
                    next_earnings = calendar.index[0] if len(calendar.index) > 0 else None
                    
                    earnings_data[symbol] = {
                        "next_earnings_date": next_earnings.isoformat() if next_earnings else None,
                        "estimate": float(calendar.iloc[0]['Earnings Estimate']) if not calendar.empty else 0.0
                    }
                else:
                    earnings_data[symbol] = {
                        "next_earnings_date": None,
                        "estimate": 0.0
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to get earnings calendar for {symbol}: {e}")
                earnings_data[symbol] = {"error": str(e)}
        
        return earnings_data

# Initialize service
market_service = MarketDataService()

# API Endpoints
@app.post("/market-data", response_model=MarketDataResponse)
async def get_market_data(request: MarketDataRequest):
    """Main market data endpoint"""
    try:
        if request.type == "portfolio_exposure":
            data = await market_service.get_portfolio_exposure(request)
            confidence = 0.9
            sources = ["yfinance", "portfolio_system"]
            
        elif request.type == "fundamentals":
            symbols = request.symbols or list(ASIA_TECH_SYMBOLS.values())
            data = await market_service.get_fundamental_data(symbols)
            confidence = 0.85
            sources = ["alpha_vantage", "yfinance"]
            
        elif request.type == "earnings_calendar":
            data = await market_service.get_earnings_calendar(request.symbols)
            confidence = 0.8
            sources = ["yfinance"]
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown data type: {request.type}")
        
        return MarketDataResponse(
            data=data,
            confidence=confidence,
            sources=sources,
            timestamp=datetime.now(),
            cache_hit=data.get("cache_hit", False)
        )
        
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-overview")
async def get_market_overview():
    """Get general market overview"""
    try:
        # Fetch major indices
        indices = {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "^HSI": "Hang Seng",
            "^N225": "Nikkei 225"
        }
        
        overview_data = {}
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change_pct = ((current - previous) / previous) * 100
                    
                    overview_data[symbol] = {
                        "name": name,
                        "value": float(current),
                        "change_percent": float(change_pct)
                    }
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        
        return {
            "indices": overview_data,
            "timestamp": datetime.now(),
            "market_status": "OPEN"  # Simplified - you'd check actual market hours
        }
        
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "redis": "connected" if redis_client else "disconnected",
            "yfinance": "available",
            "alpha_vantage": "configured" if ALPHA_VANTAGE_KEY != "demo" else "demo_mode"
        }
    }
    
    # Test Redis connection
    if redis_client:
        try:
            redis_client.ping()
            health_status["services"]["redis"] = "connected"
        except:
            health_status["services"]["redis"] = "error"
    
    return health_status

@app.get("/symbols")
async def get_supported_symbols():
    """Get list of supported stock symbols"""
    return {
        "asia_tech_symbols": ASIA_TECH_SYMBOLS,
        "total_symbols": len(ASIA_TECH_SYMBOLS),
        "regions": ["US", "Asia", "Global"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
