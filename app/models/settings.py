"""
Settings-related Pydantic models.
"""
from typing import Optional
from pydantic import BaseModel


class SettingsUpdate(BaseModel):
    """Update application settings."""
    llm_provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_model: Optional[str] = None
    tavily_api_key: Optional[str] = None


class ModelListRequest(BaseModel):
    """Request to list available LLM models."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    provider: str = "openai"


class GenerateRequest(BaseModel):
    """Request to generate a report."""
    fund_code: Optional[str] = None


class CommodityAnalyzeRequest(BaseModel):
    """Request to analyze a commodity."""
    asset: str  # "gold" or "silver"
