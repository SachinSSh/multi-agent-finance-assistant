# orchestrator/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import httpx
import logging
from typing import Dict, List, Optional, Any
import time
import json
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Finance Assistant Orchestrator",
    description="Multi-agent orchestration service for finance assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent service URLs
AGENT_SERVICES = {
    "api_agent": "http://localhost:8001",
    "scraper_agent": "http://localhost:8002", 
    "retriever_agent": "http://localhost:8003",
    "analysis_agent": "http://localhost:8004",
    "language_agent": "http://localhost:8005",
    "voice_agent": "http://localhost:8006"
}

# Request/Response models
class MarketBriefRequest(BaseModel):
    query: str
    voice_input: Optional[str] = None
    portfolio_id: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = {}

class MarketBriefResponse(BaseModel):
    text_response: str
    audio_response: Optional[str] = None
    confidence: float
    sources: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    timestamp: datetime

class AgentResponse(BaseModel):
    agent_id: str
    data: Dict[str, Any]
    confidence: float
    latency: float
    error: Optional[str] = None

# Agent orchestration logic
class AgentOrchestrator:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.confidence_threshold = 0.7
        
    async def route_query(self, request: MarketBriefRequest) -> MarketBriefResponse:
        """Main orchestration logic for processing queries"""
        start_time = time.time()
        
        try:
            # Step 1: Process voice input if provided
            query = request.query
            if request.voice_input:
                query = await self.process_voice_input(request.voice_input)
            
            # Step 2: Analyze query intent and route to appropriate agents
            agent_tasks = await self.determine_agent_tasks(query, request.preferences)
            
            # Step 3: Execute agent tasks in parallel
            agent_responses = await self.execute_agent_tasks(agent_tasks)
            
            # Step 4: Synthesize responses
            synthesis_result = await self.synthesize_responses(
                query, agent_responses, request.preferences
            )
            
            # Step 5: Generate voice output if requested
            audio_response = None
            if request.preferences.get("voice_output", False):
                audio_response = await self.generate_voice_output(
                    synthesis_result["text_response"]
                )
            
            # Step 6: Compile final response
            total_latency = time.time() - start_time
            
            return MarketBriefResponse(
                text_response=synthesis_result["text_response"],
                audio_response=audio_response,
                confidence=synthesis_result["confidence"],
                sources=synthesis_result["sources"],
                metrics={
                    "total_latency": total_latency,
                    "agent_latencies": {r.agent_id: r.latency for r in agent_responses},
                    "agents_used": [r.agent_id for r in agent_responses if not r.error]
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Orchestration error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_voice_input(self, audio_data: str) -> str:
        """Process voice input through voice agent"""
        try:
            response = await self.client.post(
                f"{AGENT_SERVICES['voice_agent']}/stt",
                json={"audio_data": audio_data}
            )
            response.raise_for_status()
            return response.json()["transcription"]
        except Exception as e:
            logger.error(f"Voice processing error: {e}")
            raise HTTPException(status_code=500, detail="Voice processing failed")
    
    async def determine_agent_tasks(self, query: str, preferences: Dict) -> Dict[str, Dict]:
        """Determine which agents to call based on query analysis"""
        # Simple intent classification - in production, use NLP model
        tasks = {}
        
        query_lower = query.lower()
        
        # Always need market data for financial queries
        if any(term in query_lower for term in ["stock", "market", "price", "portfolio", "exposure"]):
            tasks["api_agent"] = {
                "endpoint": "/market-data",
                "params": {"query": query, "type": "portfolio_exposure"}
            }
            
        # Risk analysis keywords
        if any(term in query_lower for term in ["risk", "exposure", "volatility", "beta"]):
            tasks["analysis_agent"] = {
                "endpoint": "/analyze-risk", 
                "params": {"query": query}
            }
            
        # Earnings and news keywords
        if any(term in query_lower for term in ["earnings", "surprise", "report", "news"]):
            tasks["scraper_agent"] = {
                "endpoint": "/scrape-earnings",
                "params": {"query": query}
            }
            
        # Always use retriever for context
        tasks["retriever_agent"] = {
            "endpoint": "/retrieve",
            "params": {"query": query, "top_k": 5}
        }
        
        # Language agent for synthesis
        tasks["language_agent"] = {
            "endpoint": "/synthesize",
            "params": {"query": query}
        }
        
        return tasks
    
    async def execute_agent_tasks(self, tasks: Dict[str, Dict]) -> List[AgentResponse]:
        """Execute agent tasks in parallel"""
        agent_responses = []
        
        async def call_agent(agent_id: str, task_config: Dict):
            start_time = time.time()
            try:
                url = f"{AGENT_SERVICES[agent_id]}{task_config['endpoint']}"
                response = await self.client.post(url, json=task_config['params'])
                response.raise_for_status()
                
                latency = time.time() - start_time
                data = response.json()
                
                return AgentResponse(
                    agent_id=agent_id,
                    data=data,
                    confidence=data.get("confidence", 1.0),
                    latency=latency
                )
                
            except Exception as e:
                logger.error(f"Agent {agent_id} error: {e}")
                return AgentResponse(
                    agent_id=agent_id,
                    data={},
                    confidence=0.0,
                    latency=time.time() - start_time,
                    error=str(e)
                )
        
        # Execute all agent calls concurrently
        tasks_list = [call_agent(agent_id, config) for agent_id, config in tasks.items()]
        agent_responses = await asyncio.gather(*tasks_list)
        
        return agent_responses
    
    async def synthesize_responses(self, query: str, agent_responses: List[AgentResponse], preferences: Dict) -> Dict[str, Any]:
        """Synthesize agent responses into final answer"""
        # Compile all successful responses
        successful_responses = [r for r in agent_responses if not r.error]
        
        if not successful_responses:
            raise HTTPException(status_code=500, detail="All agents failed")
        
        # Get language agent response for synthesis
        language_response = next(
            (r for r in successful_responses if r.agent_id == "language_agent"), 
            None
        )
        
        if not language_response:
            # Fallback synthesis
            return self.fallback_synthesis(query, successful_responses)
        
        # Extract data from each agent
        synthesis_data = {
            "query": query,
            "agent_data": {r.agent_id: r.data for r in successful_responses},
            "preferences": preferences
        }
        
        try:
            # Call language agent for final synthesis
            response = await self.client.post(
                f"{AGENT_SERVICES['language_agent']}/synthesize",
                json=synthesis_data
            )
            response.raise_for_status()
            synthesis_result = response.json()
            
            # Calculate overall confidence
            avg_confidence = sum(r.confidence for r in successful_responses) / len(successful_responses)
            
            return {
                "text_response": synthesis_result["narrative"],
                "confidence": avg_confidence,
                "sources": synthesis_result.get("sources", [])
            }
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return self.fallback_synthesis(query, successful_responses)
    
    def fallback_synthesis(self, query: str, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Fallback synthesis when language agent fails"""
        # Simple template-based response
        response_parts = []
        sources = []
        
        for response in responses:
            if response.agent_id == "api_agent" and "portfolio_exposure" in response.data:
                exposure_data = response.data["portfolio_exposure"]
                response_parts.append(f"Your portfolio exposure is {exposure_data.get('percentage', 'N/A')}%")
                
            elif response.agent_id == "scraper_agent" and "earnings" in response.data:
                earnings_data = response.data["earnings"]
                for company, data in earnings_data.items():
                    surprise = data.get("surprise", 0)
                    direction = "beat" if surprise > 0 else "missed"
                    response_parts.append(f"{company} {direction} estimates by {abs(surprise)}%")
                    
            elif response.agent_id == "analysis_agent" and "sentiment" in response.data:
                sentiment = response.data["sentiment"]
                response_parts.append(f"Market sentiment is {sentiment}")
                
            # Collect sources
            if "sources" in response.data:
                sources.extend(response.data["sources"])
        
        text_response = ". ".join(response_parts) if response_parts else "Unable to generate market brief at this time."
        
        return {
            "text_response": text_response,
            "confidence": 0.6,  # Lower confidence for fallback
            "sources": sources
        }
    
    async def generate_voice_output(self, text: str) -> str:
        """Generate voice output through voice agent"""
        try:
            response = await self.client.post(
                f"{AGENT_SERVICES['voice_agent']}/tts",
                json={"text": text}
            )
            response.raise_for_status()
            return response.json()["audio_data"]
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

# Initialize orchestrator
orchestrator = AgentOrchestrator()

# API endpoints
@app.post("/api/market-brief", response_model=MarketBriefResponse)
async def market_brief(request: MarketBriefRequest):
    """Main endpoint for market brief generation"""
    return await orchestrator.route_query(request)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    # Check all agent services
    agent_status = {}
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for agent_id, url in AGENT_SERVICES.items():
            try:
                response = await client.get(f"{url}/health")
                agent_status[agent_id] = "healthy" if response.status_code == 200 else "unhealthy"
            except:
                agent_status[agent_id] = "unreachable"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "agents": agent_status
    }

@app.get("/api/agents")
async def list_agents():
    """List all available agents and their endpoints"""
    return {
        "agents": AGENT_SERVICES,
        "total_agents": len(AGENT_SERVICES)
    }

# Voice-specific endpoints
@app.post("/api/voice-chat")
async def voice_chat(audio_data: dict):
    """Direct voice-to-voice chat interface"""
    request = MarketBriefRequest(
        query="",
        voice_input=audio_data.get("audio"),
        preferences={"voice_output": True}
    )
    return await orchestrator.route_query(request)

# Fallback handling
@app.post("/api/fallback-clarification")
async def fallback_clarification(request: dict):
    """Handle low-confidence responses with clarification"""
    query = request.get("query", "")
    confidence = request.get("confidence", 0.0)
    
    if confidence < orchestrator.confidence_threshold:
        clarification = f"I'm not entirely confident about your request: '{query}'. Could you please clarify what specific information you need about your portfolio or the market?"
        
        # Generate voice clarification if requested
        audio_clarification = None
        if request.get("voice_output", False):
            audio_clarification = await orchestrator.generate_voice_output(clarification)
        
        return {
            "clarification": clarification,
            "audio_clarification": audio_clarification,
            "confidence": confidence,
            "suggestions": [
                "Ask about specific stock symbols or sectors",
                "Request portfolio risk analysis",
                "Inquire about recent earnings reports"
            ]
        }
    
    return {"message": "Confidence level acceptable, no clarification needed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
