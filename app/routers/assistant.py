"""
AI Assistant endpoints.
"""
import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from app.models.assistant import AssistantChatRequest, AssistantChatResponse, AssistantSource
from app.models.auth import User
from app.core.dependencies import get_current_user
from src.services.assistant_service import assistant_service

router = APIRouter(prefix="/api/assistant", tags=["Assistant"])


@router.post("/chat", response_model=AssistantChatResponse)
async def assistant_chat(
    request: AssistantChatRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Chat with AI assistant.

    The assistant is context-aware and can:
    - Understand current page context (stocks, funds, news)
    - Search for relevant news and information
    - Provide RAG-enhanced responses
    """
    try:
        context = request.context or {}
        history = request.history or []

        # Call assistant service
        result = await asyncio.to_thread(
            assistant_service.chat,
            message=request.message,
            context=context,
            history=history,
            user_id=current_user.id
        )

        # Get suggested questions for follow-up
        suggestions = assistant_service.get_suggested_questions(context)

        return AssistantChatResponse(
            response=result.get("response", ""),
            sources=[AssistantSource(**s) for s in result.get("sources", [])],
            context_used=result.get("context_used", {}),
            suggested_questions=suggestions
        )
    except Exception as e:
        print(f"Assistant chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions")
async def get_assistant_suggestions(
    page: Optional[str] = None,
    stock_code: Optional[str] = None,
    stock_name: Optional[str] = None,
    fund_code: Optional[str] = None,
    fund_name: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get suggested questions based on current context."""
    try:
        context = {"page": page or "dashboard"}
        if stock_code and stock_name:
            context["stock"] = {"code": stock_code, "name": stock_name}
        if fund_code and fund_name:
            context["fund"] = {"code": fund_code, "name": fund_name}

        suggestions = assistant_service.get_suggested_questions(context)
        return {"suggestions": suggestions}
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
