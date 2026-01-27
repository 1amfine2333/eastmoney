"""
Portfolio-related Pydantic models.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class PortfolioCreate(BaseModel):
    """Create a new portfolio."""
    name: str
    description: Optional[str] = None
    benchmark_code: Optional[str] = "000300.SH"
    is_default: Optional[bool] = False


class PortfolioUpdate(BaseModel):
    """Update an existing portfolio."""
    name: Optional[str] = None
    description: Optional[str] = None
    benchmark_code: Optional[str] = None
    is_default: Optional[bool] = None


class PositionCreate(BaseModel):
    """Create a legacy fund position (for backwards compatibility)."""
    fund_code: str
    fund_name: Optional[str] = None
    shares: float
    cost_basis: float
    purchase_date: str
    notes: Optional[str] = None


class PositionUpdate(BaseModel):
    """Update a legacy fund position."""
    fund_name: Optional[str] = None
    shares: Optional[float] = None
    cost_basis: Optional[float] = None
    purchase_date: Optional[str] = None
    notes: Optional[str] = None


class UnifiedPositionCreate(BaseModel):
    """Create a unified position (stock or fund)."""
    asset_type: str  # 'stock' or 'fund'
    asset_code: str
    asset_name: Optional[str] = None
    total_shares: float
    average_cost: float
    sector: Optional[str] = None
    notes: Optional[str] = None


class UnifiedPositionUpdate(BaseModel):
    """Update a unified position."""
    asset_name: Optional[str] = None
    total_shares: Optional[float] = None
    average_cost: Optional[float] = None
    sector: Optional[str] = None
    notes: Optional[str] = None


class TransactionCreate(BaseModel):
    """Create a transaction record."""
    asset_type: str  # 'stock' or 'fund'
    asset_code: str
    asset_name: Optional[str] = None
    transaction_type: str  # 'buy', 'sell', 'dividend', 'split', 'transfer_in', 'transfer_out'
    shares: float
    price: float
    total_amount: Optional[float] = None
    fees: Optional[float] = 0
    transaction_date: str
    notes: Optional[str] = None


class DIPPlanCreate(BaseModel):
    """Create a DIP (定投) plan."""
    asset_type: str  # 'stock' or 'fund'
    asset_code: str
    asset_name: Optional[str] = None
    amount_per_period: float
    frequency: str  # 'daily', 'weekly', 'biweekly', 'monthly'
    execution_day: Optional[int] = None
    start_date: str
    end_date: Optional[str] = None
    is_active: Optional[bool] = True
    notes: Optional[str] = None


class DIPPlanUpdate(BaseModel):
    """Update a DIP plan."""
    asset_name: Optional[str] = None
    amount_per_period: Optional[float] = None
    frequency: Optional[str] = None
    execution_day: Optional[int] = None
    end_date: Optional[str] = None
    is_active: Optional[bool] = None
    notes: Optional[str] = None


class AlertMarkReadRequest(BaseModel):
    """Request to mark an alert as read."""
    alert_id: int


class AIRebalanceRequest(BaseModel):
    """Request for AI rebalancing suggestions."""
    target_allocation: Optional[Dict[str, float]] = None
    risk_preference: Optional[str] = "moderate"  # conservative, moderate, aggressive


class PortfolioAIChatRequest(BaseModel):
    """Request for portfolio AI chat."""
    message: str
    context: Optional[Dict[str, Any]] = None
