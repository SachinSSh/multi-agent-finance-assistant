# orchestrator/__init__.py
"""
Orchestrator module for agent routing and request handling.
"""

from .main import app
from .routing import AgentRouter
from .fallback import FallbackHandler

__version__ = "1.0.0"
__all__ = ["app", "AgentRouter", "FallbackHandler"]

