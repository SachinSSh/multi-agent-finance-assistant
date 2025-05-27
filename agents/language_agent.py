
# finance-assistant/agents/language_agent.py
"""
Language Agent - LLM Synthesis
Handles natural language processing, report generation, and AI-powered insights
"""

import openai
from typing import Dict, List, Optional, Union
import json
import re
from dataclasses import dataclass
import logging
from datetime import datetime
import pandas as pd

@dataclass
class GeneratedReport:
    title: str
    content: str
    summary: str
    key_insights: List[str]
    recommendations: List[str]
    confidence_score: float
    sources: List[str]

class LanguageAgent:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        if api_key:
            openai.api_key = api_key
    
    def generate_market_analysis(self, market_data: Dict, 
                               technical_analysis: Dict,
                               context: str = "") -> GeneratedReport:
        """Generate comprehensive market analysis report"""
        try:
            # Prepare data summary
            data_summary = self._prepare_data_summary(market_data, technical_analysis)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(data_summary, context)
            
            # Generate report using LLM
            response = self._call_llm(prompt)
            
            # Parse and structure the response
            report = self._parse_analysis_response(response)
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating market analysis: {e}")
            return self._create_fallback_report("Market Analysis", str(e))
    
    def generate_portfolio_report(self, portfolio_metrics: Dict,
                                risk_analysis: Dict,
                                recommendations: List[str] = None) -> GeneratedReport:
        """Generate portfolio performance and risk report"""
        try:
            # Prepare portfolio summary
            portfolio_summary = self._prepare_portfolio_summary(portfolio_metrics, risk_analysis)
            
            # Create portfolio analysis prompt
            prompt = f"""
            As a financial advisor, analyze the following portfolio performance and provide insights:

            Portfolio Metrics:
            {json.dumps(portfolio_summary, indent=2)}

            Please provide:
            1. Executive Summary (2-3 sentences)
            2. Performance Analysis (strengths and weaknesses)
            3. Risk Assessment
            4. Specific Recommendations for improvement
            5. Overall portfolio health score (1-10)

            Format the response as a professional investment report.
            """
            
            # Generate report
            response = self._call_llm(prompt)
            report = self._parse_portfolio_response(response, portfolio_summary)
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating portfolio report: {e}")
            return self._create_fallback_report("Portfolio Report", str(e))
    
    def explain_financial_concept(self, concept: str, context: str = "",
                                user_level: str = "intermediate") -> str:
        """Explain financial concepts in user-friendly language"""
        try:
            prompt = f"""
            Explain the financial concept "{concept}" to someone with {user_level} knowledge of finance.

            Context: {context}

            Requirements:
            - Use clear, accessible language
            - Provide practical examples
            - Include why this concept matters for investors
            - Keep explanation concise but comprehensive
            - Use analogies if helpful for understanding
            """
            
            explanation = self._call_llm(prompt, max_tokens=800)
            return explanation
        except Exception as e:
            self.logger.error(f"Error explaining concept {concept}: {e}")
            return f"Unable to explain {concept} at this time."
    
    def generate_investment_thesis(self, symbol: str, analysis_data: Dict,
                                 market_context: str = "") -> GeneratedReport:
        """Generate investment thesis for a specific stock"""
        try:
            # Prepare investment data
            investment_data = {
                'symbol': symbol,
                'current_price': analysis_data.get('current_price', 'N/A'),
                'technical_indicators': analysis_data.get('technical_indicators', {}),
                'fundamental_data': analysis_data.get('fundamental_data', {}),
                'market_context': market_context
            }
            
            prompt = f"""
            Create a comprehensive investment thesis for {symbol} based on the following analysis:

            {json.dumps(investment_data, indent=2)}

            Provide:
            1. Investment Thesis Summary (Bull/Bear/Neutral)
            2. Key Strengths and Opportunities
            3. Key Risks and Concerns
            4. Technical Analysis Insights
            5. Price Target and Timeline
            6. Risk-Reward Assessment
            7. Recommended Position Size
            8. Key Catalysts to Watch

            Write in a professional, analytical tone suitable for investment decisions.
            """
            
            response = self._call_llm(prompt, max_tokens=1200)
            
            # Parse investment thesis
            thesis = self._parse_investment_thesis(response, symbol)
            
            return thesis
        except Exception as e:
            self.logger.error(f"Error generating investment thesis for {symbol}: {e}")
            return self._create_fallback_report(f"Investment Thesis - {symbol}", str(e))
    
    def summarize_earnings_call(self, transcript: str) -> Dict[str, str]:
        """Summarize earnings call transcript and extract key insights"""
        try:
            prompt = f"""
            Analyze the following earnings call transcript and provide:

            1. Executive Summary (3-4 sentences)
            2. Key Financial Highlights
            3. Management Guidance and Outlook
            4. Key Business Updates
            5. Analyst Questions Summary
            6. Market Sentiment Analysis
            7. Key Takeaways for Investors

            Transcript:
            {transcript[:4000]}...  # Truncate if too long

            Keep each section concise and focused on actionable insights.
            """
            
            summary = self._call_llm(prompt, max_tokens=1000)
            
            # Parse the summary into structured format
            sections = self._parse_earnings_summary(summary)
            
            return sections
        except Exception as e:
            self.logger.error(f"Error summarizing earnings call: {e}")
            return {"error": str(e)}
    
    def generate_market_commentary(self, market_data: Dict, 
                                 news_sentiment: Dict,
                                 technical_levels: Dict) -> str:
        """Generate daily market commentary"""
        try:
            commentary_data = {
                'market_performance': market_data,
                'sentiment_analysis': news_sentiment,
                'technical_levels': technical_levels,
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            
            prompt = f"""
            Write a concise daily market commentary based on:

            {json.dumps(commentary_data, indent=2)}

            Include:
            - Market overview and key movements
            - Notable sector performances  
            - Technical analysis insights
            - Sentiment and news impact
            - Key levels to watch
            - Tomorrow's key events/catalysts

            Write in a professional but accessible tone, 200-300 words.
            """
            
            commentary = self._call_llm(prompt, max_tokens=500)
            return commentary
        except Exception as e:
            self.logger.error(f"Error generating market commentary: {e}")
            return "Market commentary unavailable at this time."
    
    def _prepare_data_summary(self, market_data: Dict, technical_analysis: Dict) -> Dict:
        """Prepare structured data summary for analysis"""
        return {
            'price_data': {
                'current_price': market_data.get('current_price'),
                'price_change': market_data.get('price_change'),
                'volume': market_data.get('volume')
            },
            'technical_indicators': technical_analysis,
            'timestamp': datetime.now().isoformat()
        }
    
    def _prepare_portfolio_summary(self, metrics: Dict, risk_analysis: Dict) -> Dict:
        """Prepare portfolio data for analysis"""
        return {
            'performance_metrics': metrics,
            'risk_metrics': risk_analysis,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _create_analysis_prompt(self, data_summary: Dict, context: str) -> str:
        """Create structured prompt for market analysis"""
        return f"""
        As a senior financial analyst, provide a comprehensive analysis of the following market data:

        Market Data:
        {json.dumps(data_summary, indent=2)}

        Additional Context:
        {context}

        Please provide:
        1. Current Market Assessment
        2. Technical Analysis Interpretation
        3. Key Risk Factors
        4. Trading Opportunities
        5. Market Outlook (Short/Medium term)
        6. Specific Recommendations

        Format as a professional research report with clear sections and actionable insights.
        """
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """Call LLM API with error handling"""
        try:
            if not self.api_key:
                # Fallback to template-based generation
                return self._generate_template_response(prompt)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst with expertise in market analysis, portfolio management, and investment research."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {e}")
            return self._generate_template_response(prompt)
    
    def _generate_template_response(self, prompt: str) -> str:
        """Generate template response when LLM is unavailable"""
        if "portfolio" in prompt.lower():
            return """
            Portfolio Analysis Report
            
            Executive Summary:
            Based on the provided metrics, your portfolio shows mixed performance with areas for improvement.
            
            Performance Analysis:
            - Portfolio returns require detailed analysis against benchmarks
            - Risk metrics indicate need for diversification review
            - Technical indicators suggest monitoring key support/resistance levels
            
            Recommendations:
            - Review asset allocation for optimal risk-return balance
            - Consider rebalancing based on target allocations
            - Monitor correlation between holdings
            - Implement risk management strategies
            
            Overall Assessment: Requires active monitoring and potential adjustments.
            """
        
        return """
        Market Analysis Report
        
        Current Market Assessment:
        The market data provided indicates mixed signals requiring careful analysis.
        
        Technical Analysis:
        Key technical indicators show varying momentum and trend strength.
        
        Recommendations:
        - Monitor key support and resistance levels
        - Consider risk management strategies
        - Stay informed on market developments
        - Maintain disciplined approach to position sizing
        """
    
    def _parse_analysis_response(self, response: str) -> GeneratedReport:
        """Parse LLM response into structured report"""
        try:
            # Extract key sections using regex
            summary_match = re.search(r'(Executive Summary|Summary):(.*?)(?=\n[A-Z]|\n\d\.|\Z)', response, re.DOTALL | re.IGNORECASE)
            summary = summary_match.group(2).strip() if summary_match else response[:200]
            
            # Extract recommendations
            rec_pattern = r'(Recommendations?|Suggestions?):(.*?)(?=\n[A-Z]|\n\d\.|\Z)'
            rec_match = re.search(rec_pattern, response, re.DOTALL | re.IGNORECASE)
            
            recommendations = []
            if rec_match:
                rec_text = rec_match.group(2).strip()
                recommendations = [r.strip('- ').strip() for r in rec_text.split('\n') if r.strip()]
            
            # Extract key insights
            insights = self._extract_key_insights(response)
            
            return GeneratedReport(
                title="Market Analysis Report",
                content=response,
                summary=summary,
                key_insights=insights,
                recommendations=recommendations,
                confidence_score=0.8,
                sources=["Market Data", "Technical Analysis"]
            )
        except Exception as e:
            self.logger.error(f"Error parsing analysis response: {e}")
            return self._create_fallback_report("Market Analysis", response)
    
    def _parse_portfolio_response(self, response: str, data: Dict) -> GeneratedReport:
        """Parse portfolio analysis response"""
        try:
            return GeneratedReport(
                title="Portfolio Performance Report",
                content=response,
                summary=response[:300] + "..." if len(response) > 300 else response,
                key_insights=self._extract_key_insights(response),
                recommendations=self._extract_recommendations(response),
                confidence_score=0.85,
                sources=["Portfolio Metrics", "Risk Analysis"]
            )
        except Exception as e:
            return self._create_fallback_report("Portfolio Report", response)
    
    def _parse_investment_thesis(self, response: str, symbol: str) -> GeneratedReport:
        """Parse investment thesis response"""
        try:
            return GeneratedReport(
                title=f"Investment Thesis - {symbol}",
                content=response,
                summary=self._extract_thesis_summary(response),
                key_insights=self._extract_key_insights(response),
                recommendations=self._extract_recommendations(response),
                confidence_score=0.9,
                sources=["Technical Analysis", "Fundamental Analysis", "Market Context"]
            )
        except Exception as e:
            return self._create_fallback_report(f"Investment Thesis - {symbol}", response)
    
    def _parse_earnings_summary(self, summary: str) -> Dict[str, str]:
        """Parse earnings call summary into sections"""
        sections = {}
        
        section_headers = [
            "Executive Summary", "Financial Highlights", "Management Guidance",
            "Business Updates", "Analyst Questions", "Market Sentiment", "Key Takeaways"
        ]
        
        for header in section_headers:
            pattern = f"{header}:?(.*?)(?={'|'.join(section_headers[section_headers.index(header)+1:])}:|$)"
            match = re.search(pattern, summary, re.DOTALL | re.IGNORECASE)
            if match:
                sections[header.lower().replace(' ', '_')] = match.group(1).strip()
        
        return sections
    
    def _extract_key_insights(self, text: str) -> List[str]:
        """Extract key insights from analysis text"""
        insights = []
        
        # Look for bullet points or numbered lists
        bullet_pattern = r'[•\-\*]\s*(.+?)(?=\n|$)'
        numbered_pattern = r'\d+\.\s*(.+?)(?=\n\d+\.|\n[A-Z]|\Z)'
        
        bullet_matches = re.findall(bullet_pattern, text)
        numbered_matches = re.findall(numbered_pattern, text, re.DOTALL)
        
        insights.extend([match.strip() for match in bullet_matches[:5]])
        insights.extend([match.strip()[:200] for match in numbered_matches[:3]])
        
        return insights[:8] if insights else ["Analysis completed successfully"]
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from text"""
        # Look for recommendation sections
        rec_section = re.search(r'(Recommendations?|Suggestions?):(.*?)(?=\n[A-Z]|\Z)', 
                               text, re.DOTALL | re.IGNORECASE)
        
        if rec_section:
            rec_text = rec_section.group(2)
            recommendations = re.findall(r'[•\-\*]\s*(.+?)(?=\n Using yfinance for simplicity (replace with paid API for production)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            latest = hist.iloc[-1]
            previous_close = info.get('previousClose', latest['Close'])
            
            change = latest['Close'] - previous_close
            change_percent = (change / previous_close) * 100
            
            return MarketData(
                symbol=symbol,
                price=latest['Close'],
                change=change,
                change_percent=change_percent,
                volume=int(latest['Volume']),
                timestamp=datetime.now(),
                additional_data={
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE')
                }
            )
        except Exception as e:
            self.logger.error(f"Error fetching quote for {symbol}: {e}")
            raise
    
    async def get_historical_data(self, symbol: str, period: str = "1y", 
                                interval: str = "1d") -> pd.DataFrame:
        """Get historical data for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                raise ValueError(f"No historical data found for {symbol}")
            
            # Add technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['MACD'], hist['MACD_Signal'] = self._calculate_macd(hist['Close'])
            
            return hist
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
    
    async def get_batch_quotes(self, symbols: List[str]) -> List[MarketData]:
        """Get quotes for multiple symbols concurrently"""
        tasks = [self.get_real_time_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, MarketData):
                valid_results.append(result)
            else:
                self.logger.error(f"Error in batch quote: {result}")
        
        return valid_results
    
    async def get_market_news(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Fetch market news"""
        try:
            if symbol:
                ticker = yf.Ticker(symbol)
                news = ticker.news[:limit]
            else:
                # General market news - would need news API in production
                news = []
            
            return news
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
