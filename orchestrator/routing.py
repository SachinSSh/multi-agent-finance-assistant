
# orchestrator/routing.py
"""
Agent routing logic for the orchestrator.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import aiohttp
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class AgentType(Enum):
    NLP = "nlp"
    VISION = "vision"
    CODE = "code"
    GENERAL = "general"
    SPECIALIZED = "specialized"

@dataclass
class Agent:
    id: str
    name: str
    type: AgentType
    endpoint: str
    capabilities: List[str]
    priority: int = 1
    max_concurrent: int = 5
    timeout: int = 30
    active_requests: int = 0
    health_status: str = "unknown"

class AgentRouter:
    """Routes requests to appropriate agents based on task type and agent availability."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.task_mappings: Dict[str, List[str]] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialize_default_agents()
        
    def _initialize_default_agents(self):
        """Initialize default agent configurations."""
        default_agents = [
            Agent(
                id="nlp-001",
                name="NLP Processor",
                type=AgentType.NLP,
                endpoint="http://nlp-service:8001",
                capabilities=["text_analysis", "sentiment", "summarization", "translation"]
            ),
            Agent(
                id="vision-001",
                name="Vision Processor",
                type=AgentType.VISION,
                endpoint="http://vision-service:8002",
                capabilities=["image_analysis", "ocr", "object_detection", "classification"]
            ),
            Agent(
                id="code-001",
                name="Code Assistant",
                type=AgentType.CODE,
                endpoint="http://code-service:8003",
                capabilities=["code_generation", "debugging", "review", "testing"]
            ),
            Agent(
                id="general-001",
                name="General Assistant",
                type=AgentType.GENERAL,
                endpoint="http://general-service:8004",
                capabilities=["qa", "reasoning", "planning", "general_tasks"]
            )
        ]
        
        for agent in default_agents:
            self.register_agent(agent)
    
    def register_agent(self, agent: Agent):
        """Register a new agent with the router."""
        self.agents[agent.id] = agent
        
        # Update task mappings
        for capability in agent.capabilities:
            if capability not in self.task_mappings:
                self.task_mappings[capability] = []
            self.task_mappings[capability].append(agent.id)
        
        logger.info(f"Registered agent: {agent.id} with capabilities: {agent.capabilities}")
    
    def unregister_agent(self, agent_id: str):
        """Remove an agent from the router."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Remove from task mappings
            for capability in agent.capabilities:
                if capability in self.task_mappings:
                    self.task_mappings[capability] = [
                        aid for aid in self.task_mappings[capability] if aid != agent_id
                    ]
            
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    async def route_request(self, task_type: str, content: str, 
                          context: Dict[str, Any] = None, 
                          timeout: int = 30) -> Tuple[str, Any]:
        """
        Route a request to the most appropriate agent.
        Returns tuple of (agent_id, response).
        """
        context = context or {}
        
        # Find candidate agents
        candidate_agents = self._find_candidate_agents(task_type, context)
        
        if not candidate_agents:
            raise ValueError(f"No agents available for task type: {task_type}")
        
        # Select best agent
        selected_agent = self._select_best_agent(candidate_agents)
        
        # Execute request
        response = await self._execute_request(
            agent=selected_agent,
            content=content,
            context=context,
            timeout=timeout
        )
        
        return selected_agent.id, response
    
    def _find_candidate_agents(self, task_type: str, 
                             context: Dict[str, Any]) -> List[Agent]:
        """Find agents capable of handling the task."""
        candidates = []
        
        # Direct capability match
        if task_type in self.task_mappings:
            for agent_id in self.task_mappings[task_type]:
                agent = self.agents[agent_id]
                if self._is_agent_available(agent):
                    candidates.append(agent)
        
        # Fallback to general agents if no specific match
        if not candidates:
            for agent in self.agents.values():
                if (agent.type == AgentType.GENERAL and 
                    self._is_agent_available(agent)):
                    candidates.append(agent)
        
        return candidates
    
    def _is_agent_available(self, agent: Agent) -> bool:
        """Check if agent is available to handle requests."""
        return (agent.health_status in ["healthy", "unknown"] and
                agent.active_requests < agent.max_concurrent)
    
    def _select_best_agent(self, candidates: List[Agent]) -> Agent:
        """Select the best agent from candidates based on load and priority."""
        if not candidates:
            raise ValueError("No candidate agents available")
        
        # Sort by priority (higher is better) and load (lower is better)
        best_agent = min(candidates, key=lambda a: (
            -a.priority,  # Higher priority first
            a.active_requests / a.max_concurrent  # Lower load percentage
        ))
        
        return best_agent
    
    async def _execute_request(self, agent: Agent, content: str,
                             context: Dict[str, Any], timeout: int) -> Any:
        """Execute request on the selected agent."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        agent.active_requests += 1
        
        try:
            payload = {
                "content": content,
                "context": context
            }
            
            async with self.session.post(
                f"{agent.endpoint}/process",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    agent.health_status = "healthy"
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Agent returned {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            agent.health_status = "timeout"
            raise Exception(f"Agent {agent.id} timed out")
        except Exception as e:
            agent.health_status = "error"
            raise Exception(f"Agent {agent.id} error: {str(e)}")
        finally:
            agent.active_requests = max(0, agent.active_requests - 1)
    
    async def call_agent_directly(self, agent_id: str, content: str,
                                context: Dict[str, Any] = None,
                                timeout: int = 30) -> Any:
        """Call a specific agent directly, bypassing routing logic."""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        return await self._execute_request(agent, content, context or {}, timeout)
    
    async def get_agent_status(self, agent_id: str = None) -> Dict[str, Any]:
        """Get status of agents."""
        if agent_id:
            if agent_id not in self.agents:
                return None
            agent = self.agents[agent_id]
            return {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type.value,
                "health_status": agent.health_status,
                "active_requests": agent.active_requests,
                "max_concurrent": agent.max_concurrent,
                "capabilities": agent.capabilities
            }
        
        # Return all agents
        return {
            agent_id: {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type.value,
                "health_status": agent.health_status,
                "active_requests": agent.active_requests,
                "max_concurrent": agent.max_concurrent,
                "capabilities": agent.capabilities
            }
            for agent_id, agent in self.agents.items()
        }
    
    async def get_agent_info(self) -> Dict[str, Any]:
        """Get detailed information about all agents."""
        return {
            "agents": await self.get_agent_status(),
            "task_mappings": self.task_mappings,
            "total_agents": len(self.agents)
        }
    
    async def health_check_agents(self):
        """Perform health checks on all agents."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        tasks = []
        for agent in self.agents.values():
            tasks.append(self._check_agent_health(agent))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_agent_health(self, agent: Agent):
        """Check health of a single agent."""
        try:
            async with self.session.get(
                f"{agent.endpoint}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    agent.health_status = "healthy"
                else:
                    agent.health_status = "unhealthy"
        except:
            agent.health_status = "unreachable"
    
    async def close(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()


