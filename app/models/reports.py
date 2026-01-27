"""
Report-related Pydantic models.
"""
from typing import Optional
from pydantic import BaseModel


class ReportSummary(BaseModel):
    """Summary of a generated report."""
    filename: str
    date: str
    mode: str  # 'pre' or 'post'
    fund_code: Optional[str] = None
    fund_name: Optional[str] = None
    is_summary: bool = True  # True if it's a run_all report or Summary
