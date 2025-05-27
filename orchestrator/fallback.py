# orchestrator/fallback.py
"""
Error handling and fallback mechanisms for the orchestrator.
"""

import logging
import asyncio
from typing import Any, Dict, Optional
from enum import Enum
import time

from .main import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    TIMEOUT = "timeout"
    AGENT_UNAVAILABLE = "agent_unavailable"
    INVALID_REQUEST = "invalid_request"
    SYSTEM_ERROR = "system_error"
    RATE_LIMIT = "rate_limit"

class FallbackStrategy(Enum):
    RETRY = "retry"
    REDIRECT = "redirect"
    CACHE = "cache"
    MOCK = "mock"
    FAIL = "fail"

class FallbackHandler:
    """Handles errors and provides fallback mechanisms for failed requests."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.cache: Dict[str, Any] = {}
        self.retry_delays = [1, 2, 4, 8]  # Exponential backoff
        self.max_retries = 3
    
    async def handle_error(self, error: Exception, request: AgentRequest,
                          execution_time: float) -> Optional[AgentResponse]:
        """
        Main error handling entry point. Returns a fallback response if possible.
        """
        error_type = self._classify_error(error)
        strategy = self._determine_strategy(error_type, request)
        
        logger.info(f"Handling {error_type.value} with strategy {strategy.value}")
        
        try:
            if strategy == FallbackStrategy.RETRY:
                return await self._handle_retry(error, request)
            elif strategy == FallbackStrategy.REDIRECT:
                return await self._handle_redirect(error, request)
            elif strategy == FallbackStrategy.CACHE:
                return await self._handle_cache(error, request)
            elif strategy == FallbackStrategy.MOCK:
                return await self._handle_mock(error, request)
            else:
                return await self._handle_fail(error, request, execution_time)
                
        except Exception as fallback_error:
            logger.error(f"Fallback handling failed: {str(fallback_error)}")
            return await self._handle_fail(error, request, execution_time)
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify the type of error to determine appropriate handling."""
        error_msg = str(error).lower()
        
        if "timeout" in error_msg:
            return ErrorType.TIMEOUT
        elif "unavailable" in error_msg or "connection" in error_msg:
            return ErrorType.AGENT_UNAVAILABLE
        elif "invalid" in error_msg or "bad request" in error_msg:
            return ErrorType.INVALID_REQUEST
        elif "rate limit" in error_msg or "too many requests" in error_msg:
            return ErrorType.RATE_LIMIT
        else:
            return ErrorType.SYSTEM_ERROR
    
    def _determine_strategy(self, error_type: ErrorType, 
                          request: AgentRequest) -> FallbackStrategy:
        """Determine the best fallback strategy for the error type."""
        strategies = {
            ErrorType.TIMEOUT: FallbackStrategy.RETRY,
            ErrorType.AGENT_UNAVAILABLE: FallbackStrategy.REDIRECT,
            ErrorType.INVALID_REQUEST: FallbackStrategy.FAIL,
            ErrorType.RATE_LIMIT: FallbackStrategy.RETRY,
            ErrorType.SYSTEM_ERROR: FallbackStrategy.CACHE
        }
        
        base_strategy = strategies.get(error_type, FallbackStrategy.FAIL)
        
        # Modify strategy based on request priority
        if request.priority and request.priority > 5:
            if base_strategy == FallbackStrategy.FAIL:
                return FallbackStrategy.MOCK
        
        return base_strategy
    
    async def _handle_retry(self, error: Exception, 
                          request: AgentRequest) -> Optional[AgentResponse]:
        """Handle retry strategy with exponential backoff."""
        from .routing import AgentRouter
        
        router = AgentRouter()
        
        for attempt in range(self.max_retries):
            try:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                await asyncio.sleep(delay)
                
                logger.info(f"Retry attempt {attempt + 1} after {delay}s delay")
                
                agent_id, response = await router.route_request(
                    task_type=request.task_type,
                    content=request.content,
                    context=request.context,
                    timeout=request.timeout
                )
                
                return AgentResponse(
                    success=True,
                    agent_id=agent_id,
                    response=response,
                    execution_time=0,
                    metadata={
                        "fallback": True,
                        "strategy": "retry",
                        "attempt": attempt + 1
                    }
                )
                
            except Exception as retry_error:
                logger.warning(f"Retry attempt {attempt + 1} failed: {str(retry_error)}")
                if attempt == self.max_retries - 1:
                    break
        
        return None
    
    async def _handle_redirect(self, error: Exception,
                             request: AgentRequest) -> Optional[AgentResponse]:
        """Handle redirect to alternative agents."""
        from .routing import AgentRouter
        
        router = AgentRouter()
        
        try:
            # Try to find an alternative agent
            agent_id, response = await router.route_request(
                task_type="general_tasks",  # Use general task as fallback
                content=request.content,
                context=request.context,
                timeout=request.timeout
            )
            
            return AgentResponse(
                success=True,
                agent_id=agent_id,
                response=response,
                execution_time=0,
                metadata={
                    "fallback": True,
                    "strategy": "redirect",
                    "original_error": str(error)
                }
            )
            
        except Exception as redirect_error:
            logger.error(f"Redirect failed: {str(redirect_error)}")
            return None
    
    async def _handle_cache(self, error: Exception,
                          request: AgentRequest) -> Optional[AgentResponse]:
        """Handle cache-based fallback for similar requests."""
        # Create cache key from request
        cache_key = self._create_cache_key(request)
        
        if cache_key in self.cache:
            cached_response = self.cache[cache_key]
            logger.info(f"Returning cached response for key: {cache_key}")
            
            return AgentResponse(
                success=True,
                agent_id="cache",
                response=cached_response,
                execution_time=0,
                metadata={
                    "fallback": True,
                    "strategy": "cache",
                    "cached": True
                }
            )
        
        return None
    
    async def _handle_mock(self, error: Exception,
                         request: AgentRequest) -> Optional[AgentResponse]:
        """Provide mock responses for high-priority requests."""
        mock_responses = {
            "text_analysis": {
                "sentiment": "neutral",
                "confidence": 0.5,
                "message": "Mock response due to service unavailability"
            },
            "summarization": {
                "summary": "Service temporarily unavailable. Please try again later.",
                "length": "short"
            },
            "qa": {
                "answer": "I'm currently experiencing technical difficulties. Please try your request again in a few moments.",
                "confidence": 0.1
            }
        }
        
        mock_response = mock_responses.get(
            request.task_type,
            {"message": "Service temporarily unavailable", "status": "fallback"}
        )
        
        return AgentResponse(
            success=True,
            agent_id="mock",
            response=mock_response,
            execution_time=0,
            metadata={
                "fallback": True,
                "strategy": "mock",
                "warning": "This is a fallback response due to service issues"
            }
        )
    
    async def _handle_fail(self, error: Exception, request: AgentRequest,
                         execution_time: float) -> AgentResponse:
        """Handle graceful failure with detailed error information."""
        error_id = f"error_{int(time.time())}"
        
        # Track error for monitoring
        error_key = f"{type(error).__name__}:{request.task_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        return AgentResponse(
            success=False,
            agent_id="error_handler",
            response={
                "error": str(error),
                "error_type": type(error).__name__,
                "error_id": error_id,
                "task_type": request.task_type,
                "suggestion": self._get_error_suggestion(error)
            },
            execution_time=execution_time,
            metadata={
                "fallback": True,
                "strategy": "fail",
                "error_count": self.error_counts[error_key]
            }
        )
    
    def _create_cache_key(self, request: AgentRequest) -> str:
        """Create a cache key from request parameters."""
        import hashlib
        
        key_data = f"{request.task_type}:{request.content}:{str(request.context)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_error_suggestion(self, error: Exception) -> str:
        """Provide helpful suggestions based on error type."""
        error_msg = str(error).lower()
        
        if "timeout" in error_msg:
            return "The request took too long to process. Try reducing the complexity or increasing the timeout."
        elif "unavailable" in error_msg:
            return "The service is temporarily unavailable. Please try again in a few moments."
        elif "invalid" in error_msg:
            return "Please check your request format and try again."
        elif "rate limit" in error_msg:
            return "Too many requests. Please wait a moment before trying again."
        else:
            return "An unexpected error occurred. Please contact support if the issue persists."
    
    def cache_response(self, request: AgentRequest, response: Any):
        """Cache a successful response for future fallback use."""
        cache_key = self._create_cache_key(request)
        self.cache[cache_key] = response
        
        # Limit cache size
        if len(self.cache) > 1000:
            # Remove oldest entries (simplified LRU)
            oldest_keys = list(self.cache.keys())[:100]
            for key in oldest_keys:
                del self.cache[key]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "error_counts": self.error_counts,
            "cache_size": len(self.cache),
            "total_errors": sum(self.error_counts.values())
        }
