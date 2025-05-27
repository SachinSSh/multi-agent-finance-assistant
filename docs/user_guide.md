# Financial Agent User Guide

## Introduction

The Financial Agent is an advanced multi-agent system designed to provide comprehensive financial analysis, market data, and portfolio insights. This guide will help you set up and use the system effectively.

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL (for data storage)
- Redis (for caching and message queuing)

### Setup Instructions

#### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/financial-agent.git
   cd financial-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy the `.env.example` file to `.env`
   - Update the variables with your API keys and configuration settings

5. Start the services:
   ```bash
   # Start the orchestrator
   python -m orchestrator.main
   
   # Start individual agents (in separate terminals)
   python -m agents.api_agent
   python -m agents.analysis_agent
   python -m agents.scraper_agent
   python -m agents.retriever_agent
   python -m agents.language_agent
   python -m agents.voice_agent
   ```

#### Docker Deployment

1. Build and start the containers:
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

2. Verify that all services are running:
   ```bash
   docker-compose -f docker/docker-compose.yml ps
   ```

## Using the Financial Agent

### Streamlit Web Interface

The Financial Agent provides a user-friendly web interface built with Streamlit.

1. Start the Streamlit app:
   ```bash
   streamlit run streamlit_app/app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the sidebar to navigate between different sections:
   - Dashboard: Overview of market data and portfolio performance
   - Data Analysis: Detailed analysis tools and visualizations
   - Settings: Configure your preferences and API connections
   - Contact: Get support or provide feedback

### Making API Requests

You can also interact with the Financial Agent through its API endpoints.

#### Example: Getting a Market Brief

```python
import requests
import json

# Get authentication token
auth_response = requests.post(
    "http://localhost:8000/auth/token",
    json={"username": "your_username", "password": "your_password"}
)
token = auth_response.json()["token"]

# Make a request to the market brief endpoint
response = requests.post(
    "http://localhost:8000/market-brief",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "query": "What is the current market outlook for tech stocks?",
        "preferences": {
            "detail_level": "high",
            "focus_sectors": ["technology", "healthcare"]
        }
    }
)

# Print the response
print(json.dumps(response.json(), indent=2))
```

#### Example: Analyzing Portfolio Risk

```python
import requests
import json

# Assuming you have already obtained an authentication token

# Make a request to the risk analysis endpoint
response = requests.post(
    "http://localhost:8004/analyze-risk",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "query": "What is the risk profile of my portfolio?",
        "portfolio_id": "portfolio_123",
        "risk_metrics": ["volatility", "var", "beta", "sharpe_ratio"]
    }
)

# Print the response
print(json.dumps(response.json(), indent=2))
```

## Common Use Cases

### Market Research

Use the Financial Agent to research market trends and get insights on specific sectors or companies:

1. Navigate to the Dashboard in the Streamlit app
2. Enter your query in the search box, e.g., "What is the outlook for renewable energy stocks?"
3. Review the comprehensive analysis provided by the system

### Portfolio Analysis

Analyze your investment portfolio's performance and risk profile:

1. Navigate to the Data Analysis section
2. Select your portfolio from the dropdown menu
3. Choose the analysis type (Performance, Risk, Allocation)
4. Review the detailed metrics and visualizations

### Earnings Reports

Get summaries and insights from recent earnings reports:

1. Enter a query like "Recent earnings for AAPL" in the search box
2. Review the earnings data, including EPS, revenue, and analyst expectations
3. Read the AI-generated summary of the earnings report and its market implications

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Ensure all services are running
   - Check that the correct ports are open and not blocked by a firewall
   - Verify that the base URLs in your configuration match the actual service locations

2. **Authentication Issues**
   - Ensure your API keys are correctly set in the `.env` file
   - Check that your JWT token is valid and not expired
   - Verify that you're including the token in the Authorization header

3. **Data Quality Issues**
   - Some financial data may be delayed or unavailable due to API limitations
   - Consider upgrading to premium data sources for real-time data
   - Check the `sources` field in the API response for information about data origins

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs for error messages:
   ```bash
   docker-compose -f docker/docker-compose.yml logs
   ```

2. Open an issue on the project's GitHub repository

3. Contact the support team through the Contact form in the Streamlit app
