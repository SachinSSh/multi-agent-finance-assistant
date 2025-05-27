# Financial Agent API Documentation

## Overview

The Financial Agent system exposes a set of RESTful APIs that allow clients to interact with the various agent services. This document provides detailed information about the available endpoints, request/response formats, and authentication requirements.

## Base URLs

- **Orchestrator**: `http://localhost:8000`
- **API Agent**: `http://localhost:8001`
- **Scraper Agent**: `http://localhost:8002`
- **Retriever Agent**: `http://localhost:8003`
- **Analysis Agent**: `http://localhost:8004`
- **Language Agent**: `http://localhost:8005`
- **Voice Agent**: `http://localhost:8006`

## Authentication

All API endpoints require authentication using JWT (JSON Web Tokens). To authenticate:

1. Obtain a token by calling the `/auth/token` endpoint with valid credentials
2. Include the token in the `Authorization` header of all subsequent requests:
   ```
   Authorization: Bearer <your_token>
   ```

## Common Response Format

All API responses follow a standard format:

```json
{
  "data": {},        // Response data (varies by endpoint)
  "confidence": 0.95, // Confidence score (0.0-1.0)
  "sources": [],     // List of data sources used
  "metrics": {       // Performance metrics
    "total_latency": 0.45,
    "agent_latencies": {},
    "agents_used": []
  },
  "timestamp": "2023-11-01T12:34:56Z"
}
```

## Error Handling

Errors are returned with appropriate HTTP status codes and a JSON body:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "code": "ERROR_CODE",
  "timestamp": "2023-11-01T12:34:56Z"
}
```

## Orchestrator Endpoints

### Market Brief

```
POST /market-brief
```

Generates a comprehensive market brief based on the provided query.

**Request Body:**

```json
{
  "query": "What is the current market outlook for tech stocks?",
  "voice_input": "base64_encoded_audio", // Optional
  "portfolio_id": "portfolio_123",      // Optional
  "preferences": {                       // Optional
    "voice_output": true,
    "detail_level": "high",
    "focus_sectors": ["technology", "healthcare"]
  }
}
```

**Response:**

```json
{
  "text_response": "The technology sector is showing mixed signals...",
  "audio_response": "base64_encoded_audio", // Only if voice_output is true
  "confidence": 0.92,
  "sources": [
    {
      "name": "Yahoo Finance",
      "url": "https://finance.yahoo.com/news/...",
      "timestamp": "2023-11-01T10:30:00Z"
    }
  ],
  "metrics": {
    "total_latency": 1.25,
    "agent_latencies": {
      "api_agent": 0.45,
      "analysis_agent": 0.65,
      "language_agent": 0.35
    },
    "agents_used": ["api_agent", "analysis_agent", "language_agent"]
  },
  "timestamp": "2023-11-01T12:34:56Z"
}
```

## API Agent Endpoints

### Market Data

```
POST /market-data
```

Retrieves market data for specified securities.

**Request Body:**

```json
{
  "query": "AAPL stock price",
  "type": "stock_price", // Options: stock_price, historical_data, fundamentals, portfolio_exposure
  "portfolio_id": "portfolio_123", // Optional
  "preferences": {} // Optional
}
```

**Response:**

```json
{
  "data": {
    "symbol": "AAPL",
    "price": 175.25,
    "change": 2.34,
    "change_percent": 1.35,
    "volume": 45678923,
    "market_cap": 2850000000000,
    "timestamp": "2023-11-01T16:00:00Z"
  },
  "confidence": 1.0,
  "sources": [
    {
      "name": "Yahoo Finance",
      "url": "https://finance.yahoo.com/quote/AAPL",
      "timestamp": "2023-11-01T16:00:00Z"
    }
  ],
  "metrics": {
    "total_latency": 0.35,
    "agent_latencies": {},
    "agents_used": ["api_agent"]
  },
  "timestamp": "2023-11-01T16:01:23Z"
}
```

## Analysis Agent Endpoints

### Risk Analysis

```
POST /analyze-risk
```

Performs risk analysis on a portfolio or security.

**Request Body:**

```json
{
  "query": "What is the risk profile of my portfolio?",
  "portfolio_id": "portfolio_123", // Optional
  "ticker": "AAPL", // Optional, for single security analysis
  "risk_metrics": ["volatility", "var", "beta"], // Optional
  "preferences": {} // Optional
}
```

**Response:**

```json
{
  "data": {
    "volatility": 0.15,
    "var_95": 0.025,
    "beta": 1.2,
    "sharpe_ratio": 1.8,
    "max_drawdown": 0.12,
    "risk_rating": "moderate",
    "analysis": "The portfolio shows moderate risk with a beta of 1.2..."
  },
  "confidence": 0.95,
  "sources": [],
  "metrics": {
    "total_latency": 0.85,
    "agent_latencies": {},
    "agents_used": ["analysis_agent"]
  },
  "timestamp": "2023-11-01T12:34:56Z"
}
```

## Scraper Agent Endpoints

### Earnings Reports

```
POST /scrape-earnings
```

Retrieves recent earnings reports and related news.

**Request Body:**

```json
{
  "query": "Recent earnings for AAPL",
  "ticker": "AAPL", // Optional
  "date_range": { // Optional
    "start": "2023-10-01",
    "end": "2023-11-01"
  },
  "preferences": {} // Optional
}
```

**Response:**

```json
{
  "data": {
    "earnings": {
      "AAPL": {
        "report_date": "2023-10-26",
        "eps": 1.46,
        "eps_estimate": 1.39,
        "surprise": 0.07,
        "surprise_percent": 5.04,
        "revenue": 89700000000,
        "revenue_estimate": 88500000000
      }
    },
    "news": [
      {
        "title": "Apple Beats Earnings Expectations",
        "url": "https://example.com/news/apple-earnings",
        "source": "Financial Times",
        "published_at": "2023-10-26T18:30:00Z",
        "sentiment": "positive"
      }
    ]
  },
  "confidence": 0.9,
  "sources": [
    {
      "name": "Company Earnings Report",
      "url": "https://investor.apple.com/earnings/",
      "timestamp": "2023-10-26T16:30:00Z"
    }
  ],
  "metrics": {
    "total_latency": 1.45,
    "agent_latencies": {},
    "agents_used": ["scraper_agent"]
  },
  "timestamp": "2023-11-01T12:34:56Z"
}
```

## Retriever Agent Endpoints

### Information Retrieval

```
POST /retrieve
```

Retrieves relevant financial information based on the query.

**Request Body:**

```json
{
  "query": "What are the key factors affecting inflation?",
  "top_k": 5, // Number of results to return
  "preferences": {} // Optional
}
```

**Response:**

```json
{
  "data": {
    "results": [
      {
        "content": "Inflation is primarily affected by...",
        "source": "Economic Research Paper",
        "relevance_score": 0.92,
        "timestamp": "2023-06-15T00:00:00Z"
      },
      // Additional results...
    ],
    "summary": "The key factors affecting inflation include..."
  },
  "confidence": 0.88,
  "sources": [],
  "metrics": {
    "total_latency": 0.65,
    "agent_latencies": {},
    "agents_used": ["retriever_agent"]
  },
  "timestamp": "2023-11-01T12:34:56Z"
}
```

## Language Agent Endpoints

### Response Synthesis

```
POST /synthesize
```

Synthesizes information from multiple sources into a coherent response.

**Request Body:**

```json
{
  "query": "What is the outlook for tech stocks?",
  "agent_data": {
    "api_agent": { /* API agent response data */ },
    "analysis_agent": { /* Analysis agent response data */ },
    "retriever_agent": { /* Retriever agent response data */ }
  },
  "preferences": {} // Optional
}
```

**Response:**

```json
{
  "data": {
    "narrative": "Based on recent market data and analysis, the outlook for tech stocks is...",
    "key_points": [
      "Tech sector valuations remain elevated",
      "Interest rate concerns are creating headwinds",
      "Earnings growth remains strong for leading companies"
    ],
    "confidence_factors": {
      "market_data": 0.95,
      "analysis": 0.85,
      "historical_context": 0.75
    }
  },
  "confidence": 0.9,
  "sources": [],
  "metrics": {
    "total_latency": 0.55,
    "agent_latencies": {},
    "agents_used": ["language_agent"]
  },
  "timestamp": "2023-11-01T12:34:56Z"
}
```

## Voice Agent Endpoints

### Speech-to-Text

```
POST /stt
```

Converts audio input to text.

**Request Body:**

```json
{
  "audio_data": "base64_encoded_audio",
  "preferences": {
    "language": "en-US", // Optional
    "model": "standard" // Optional: standard or enhanced
  }
}
```

**Response:**

```json
{
  "data": {
    "transcription": "What is the current price of Apple stock?",
    "confidence": 0.95,
    "language_detected": "en-US"
  },
  "confidence": 0.95,
  "sources": [],
  "metrics": {
    "total_latency": 0.75,
    "agent_latencies": {},
    "agents_used": ["voice_agent"]
  },
  "timestamp": "2023-11-01T12:34:56Z"
}
```

### Text-to-Speech

```
POST /tts
```

Converts text to speech audio.

**Request Body:**

```json
{
  "text": "The current price of Apple stock is $175.25, up 1.35% today.",
  "preferences": {
    "voice": "female", // Optional: male, female, neutral
    "speed": 1.0,      // Optional: 0.5-2.0
    "format": "mp3"    // Optional: mp3, wav, ogg
  }
}
```

**Response:**

```json
{
  "data": {
    "audio_data": "base64_encoded_audio",
    "format": "mp3",
    "duration": 3.5 // seconds
  },
  "confidence": 1.0,
  "sources": [],
  "metrics": {
    "total_latency": 0.65,
    "agent_latencies": {},
    "agents_used": ["voice_agent"]
  },
  "timestamp": "2023-11-01T12:34:56Z"
}
```
