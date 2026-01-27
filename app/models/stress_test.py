"""
Stress test-related Pydantic models.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class StressTestRequest(BaseModel):
    """Request model for stress test."""
    scenario: Optional[Dict[str, Any]] = None
    scenario_type: Optional[str] = None


class AIScenarioRequest(BaseModel):
    """Request model for AI scenario generation."""
    category: str  # monetary_policy, currency, market, sector, commodity


class StressTestChatRequest(BaseModel):
    """Request model for stress test chat."""
    message: str
    history: Optional[List[Dict[str, str]]] = None


class CorrelationExplainRequest(BaseModel):
    """Request for AI explanation of correlation data."""
    correlation_data: dict
