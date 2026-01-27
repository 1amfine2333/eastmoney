"""
AI Assistant-related Pydantic models.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class AssistantChatRequest(BaseModel):
    """Request model for assistant chat."""
    message: str
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None


class AssistantSource(BaseModel):
    """Source item in assistant response."""
    title: str
    url: Optional[str] = None
    source: Optional[str] = None
    type: Optional[str] = None


class AssistantChatResponse(BaseModel):
    """Response model for assistant chat."""
    response: str
    sources: List[AssistantSource] = []
    context_used: Dict[str, Any] = {}
    suggested_questions: Optional[List[str]] = None
