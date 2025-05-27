# data_ingestion/document_loaders.py
"""
SEC Filing and Document Loaders
Handles loading and processing of regulatory filings and financial documents
"""

import requests
import json
import re
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pandas as pd
from urllib.parse import urljoin
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SECFilingLoader:
    """SEC EDGAR database filing loader"""
    
    def __init__(self, user_agent: str = "Financial Analysis Tool 1.0"):
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.search_url = "https://efts.sec.gov/LATEST/search-index"
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        self.rate_limit_delay = 0.1  # SEC requires 10 requests per second max
    
    def search_filings(self, 
                      ticker: str, 
                      form_types: List[str] = ['10-K', '10-Q', '8-K'],
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      limit: int = 100) -> List[Dict]:
        """
        Search for SEC filings by ticker symbol
        
        Args:
            ticker: Company ticker symbol
            form_types: List of form types to search for
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of results
        
        Returns:
            List of filing information dictionaries
        """
        try:
            filings = []
            
            for form_type in form_types:
                params = {
                    'dateRange': 'custom' if start_date and end_date else 'all',
                    'entityName': ticker,
                    'forms': form_type,
                    'from': start_date or '',
                    'to': end_date or '',
                    'startdt': start_date or '',
                    'enddt': end_date or ''
                }
                
                response = requests.get(self.search_url, params=params, headers=self.headers)
                response.raise_for_status()
                
                # Parse search results (this is simplified - actual SEC search may require different parsing)
                data = response.json() if response.content else {}
                
                if 'hits' in data and 'hits' in data['hits']:
                    for hit in data['hits']['hits'][:limit]:
                        source = hit.get('_source', {})
                        filing_info = {
                            'ticker': ticker,
                            'form_type': source.get('form'),
                            'filing_date': source.get('file_date'),
                            'company_name': source.get('display_names', [None])[0],
                            'cik': source.get('ciks', [None])[0],
                            'accession_number': source.get('accession_number'),
                            'document_url': self._build_document_url(source.get('ciks', [None])[0], 
                                                                   source.get('accession_number'))
                        }
                        filings.append(filing_info)
                
                time.sleep(self.rate_limit_delay)
            
            logger.info(f"Found {len(filings)} filings for {ticker}")
            return sorted(filings, key=lambda x: x['filing_date'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error searching filings for {ticker}: {str(e)}")
            raise
    
    def _build_document_url(self, cik: str, accession_number: str) -> str:
        """Build URL for SEC document"""
        if not cik or not accession_number:
            return ""
        
        # Format CIK (remove leading zeros, pad to 10 digits)
        cik_formatted = str(int(cik)).zfill(10)
        accession_clean = accession_number.replace('-', '')
        
        return f"{self.base_url}/{cik_formatted}/{accession_clean}/{accession_number}.txt"
    
    def download_filing(self, filing_info: Dict) -> str:
        """
        Download the content of a specific filing
        
        Args:
            filing_info: Filing information dictionary from search_filings
        
        Returns:
            Raw filing content as string
        """
        try:
            url = filing_info.get('document_url')
            if not url:
                raise ValueError("No document URL provided")
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            time.sleep(self.rate_limit_delay)
            
            logger.info(f"Downloaded filing: {filing_info.get('form_type')} for {filing_info.get('ticker')}")
            return response.text
            
        except Exception as e:
            logger.error(f"Error downloading filing: {str(e)}")
            raise
    
    def get_latest_filings(self, ticker: str, form_type: str = '10-K', count: int = 1) -> List[Dict]:
        """Get the most recent filings of a specific type"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')  # 3 years back
            
            filings = self.search_filings(
                ticker=ticker,
                form_types=[form_type],
                start_date=start_date,
                end_date=end_date,
                limit=count
            )
            
            return filings[:count]
            
        except Exception as e:
            logger.error(f"Error getting latest {form_type} filings for {ticker}: {str(e)}")
            raise


class DocumentProcessor:
    """Process and extract information from financial documents"""
    
    def __init__(self):
        self.financial_keywords = [
            'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities', 
            'equity', 'cash flow', 'earnings', 'ebitda', 'debt', 'margin'
        ]
    
    def extract_text_from_filing(self, filing_content: str) -> str:
        """
        Extract clean text from SEC filing HTML/SGML content
        
        Args:
            filing_content: Raw filing content
        
        Returns:
            Cleaned text content
        """
        try:
            # Remove SGML tags and HTML
            soup = BeautifulSoup(filing_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from filing: {str(e)}")
            return filing_content  # Return original if parsing fails
    
    def extract_financial_sections(self, filing_text: str) -> Dict[str, str]:
        """
        Extract key financial sections from filing text
        
        Args:
            filing_text: Cleaned filing text
        
        Returns:
            Dictionary with extracted sections
        """
        sections = {}
        
        try:
            # Common section patterns in SEC filings
            section_patterns = {
                'business_overview': r'(?i)(item\s*1[\.\s]*business|business\s*overview|company\s*overview)',
                'risk_factors': r'(?i)(item\s*1a[\.\s]*risk\s*factors|risk\s*factors)',
                'financial_position': r'(?i)(financial\s*position|balance\s*sheet)',
                'results_operations': r'(?i)(results\s*of\s*operations|operating\s*results)',
                'cash_flows': r'(?i)(cash\s*flows|liquidity)',
                'management_discussion': r'(?i)(management.s\s*discussion|md&a)'
            }
            
            text_lower = filing_text.lower()
            
            for section_name, pattern in section_patterns.items():
                matches = list(re.finditer(pattern, text_lower))
                if matches:
                    start_pos = matches[0].start()
                    # Find next section or take next 5000 characters
                    end_pos = start_pos + 5000
                    
                    # Try to find natural end point
                    for other_pattern in section_patterns.values():
                        if other_pattern != pattern:
                            other_matches = list(re.finditer(other_pattern, text_lower[start_pos + 100:]))
                            if other_matches:
                                potential_end = start_pos + 100 + other_matches[0].start()
                                if potential_end < end_pos:
                                    end_pos = potential_end
                    
                    sections[section_name] = filing_text[start_pos:end_pos].strip()
            
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting financial sections: {str(e)}")
            return {}
    
    def extract_financial_metrics(self, filing_text: str) -> Dict[str, float]:
        """
        Extract numerical financial metrics from filing text
        
        Args:
            filing_text: Filing text content
        
        Returns:
            Dictionary of extracted financial metrics
        """
        metrics = {}
        
        try:
            # Patterns for common financial metrics (simplified)
            metric_patterns = {
                'total_revenue': r'(?i)total\s+revenue[:\s]+\$?\s*([0-9,]+\.?[0-9]*)',
                'net_income': r'(?i)net\s+income[:\s]+\$?\s*([0-9,]+\.?[0-9]*)',
                'total_assets': r'(?i)total\s+assets[:\s]+\$?\s*([0-9,]+\.?[0-9]*)',
                'total_debt': r'(?i)total\s+debt[:\s]+\$?\s*([0-9,]+\.?[0-9]*)',
                'cash_equivalents': r'(?i)cash\s+and\s+cash\s+equivalents[:\s]+\$?\s*([0-9,]+\.?[0-9]*)'
            }
            
            for metric_name, pattern in metric_patterns.items():
                matches = re.findall(pattern, filing_text)
                if matches:
                    # Take the first match and convert to float
                    value_str = matches[0].replace(',', '')
                    try:
                        metrics[metric_name] = float(value_str)
                    except ValueError:
                        continue
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {str(e)}")
            return {}
    
    def summarize_document(self, filing_text: str, max_length: int = 1000) -> str:
        """
        Create a summary of the document focusing on key financial information
        
        Args:
            filing_text: Full filing text
            max_length: Maximum length of summary
        
        Returns:
            Document summary
        """
        try:
            # Extract key sentences containing financial keywords
            sentences = filing_text.split('.')
            key_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                # Check if sentence contains financial keywords
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in self.financial_keywords):
                    key_sentences.append(sentence)
                
                # Stop if we have enough content
                if len(' '.join(key_sentences)) > max_length:
                    break
            
            summary = '. '.join(key_sentences[:10])  # Take first 10 relevant sentences
            
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating document summary: {str(e)}")
            return filing_text[:max_length] + "..." if len(filing_text) > max_length else filing_text


