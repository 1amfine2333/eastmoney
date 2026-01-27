"""
Fund management endpoints.
"""
import json
from typing import List
from fastapi import APIRouter, HTTPException, Depends

from app.models.funds import FundItem, FundCompareRequest
from app.models.auth import User
from app.core.dependencies import get_current_user
from app.core.utils import sanitize_for_json
from app.core.helpers import get_fund_nav_history, get_fund_basic_info, get_fund_holdings_list
from src.storage.db import (
    get_all_funds, upsert_fund, delete_fund, get_diagnosis_cache, save_diagnosis_cache
)
from src.scheduler.manager import scheduler_manager
from src.analysis.fund import FundDiagnosis, RiskMetricsCalculator, DrawdownAnalyzer, FundComparison

import asyncio

router = APIRouter(prefix="/api/funds", tags=["Funds"])


@router.get("", response_model=List[FundItem])
async def get_funds_endpoint(current_user: User = Depends(get_current_user)):
    """Get all funds for current user."""
    try:
        funds = get_all_funds(user_id=current_user.id)
        result = []
        for f in funds:
            item = dict(f)
            if isinstance(item.get('focus'), str):
                try:
                    item['focus'] = json.loads(item['focus'])
                except:
                    item['focus'] = []

            result.append(FundItem(
                code=item['code'],
                name=item['name'],
                style=item.get('style'),
                focus=item['focus'],
                pre_market_time=item.get('pre_market_time'),
                post_market_time=item.get('post_market_time'),
                is_active=bool(item.get('is_active', True))
            ))
        return result
    except Exception as e:
        print(f"Error reading funds: {e}")
        return []


@router.post("")
async def save_funds(funds: List[FundItem], current_user: User = Depends(get_current_user)):
    """Save multiple funds."""
    try:
        for fund in funds:
            fund_dict = fund.model_dump()
            upsert_fund(fund_dict, user_id=current_user.id)
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{code}")
async def upsert_fund_endpoint(code: str, fund: FundItem, current_user: User = Depends(get_current_user)):
    """Create or update a fund."""
    try:
        fund_dict = fund.model_dump()
        upsert_fund(fund_dict, user_id=current_user.id)
        scheduler_manager.add_fund_jobs(fund_dict)
        return {"status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{code}")
async def delete_fund_endpoint(code: str, current_user: User = Depends(get_current_user)):
    """Delete a fund."""
    try:
        delete_fund(code, user_id=current_user.id)
        scheduler_manager.remove_fund_jobs(code)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{code}/diagnosis")
async def get_fund_diagnosis(
    code: str,
    force_refresh: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Get fund diagnosis with five-dimension scoring and radar chart data."""
    try:
        # Check cache first
        if not force_refresh:
            cached = get_diagnosis_cache(code)
            if cached and cached.get('diagnosis'):
                return cached['diagnosis']

        # Fetch NAV history
        loop = asyncio.get_running_loop()
        nav_history = await loop.run_in_executor(None, get_fund_nav_history, code, 500)

        if not nav_history:
            raise HTTPException(status_code=404, detail=f"No NAV history found for fund {code}")

        # Calculate diagnosis
        diagnoser = FundDiagnosis()
        diagnosis = diagnoser.diagnose(code, nav_history)

        # Cache result (6 hours TTL)
        if diagnosis.get('score', 0) > 0:
            save_diagnosis_cache(code, diagnosis, int(diagnosis['score']), ttl_hours=6)

        return sanitize_for_json(diagnosis)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating fund diagnosis for {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{code}/risk-metrics")
async def get_fund_risk_metrics(
    code: str,
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive risk metrics for a fund."""
    try:
        loop = asyncio.get_running_loop()
        nav_history = await loop.run_in_executor(None, get_fund_nav_history, code, 500)

        if not nav_history:
            raise HTTPException(status_code=404, detail=f"No NAV history found for fund {code}")

        calculator = RiskMetricsCalculator()
        metrics = calculator.calculate_all_metrics(nav_history)

        return sanitize_for_json(metrics)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating risk metrics for {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{code}/drawdown-history")
async def get_fund_drawdown_history(
    code: str,
    threshold: float = 0.05,
    current_user: User = Depends(get_current_user)
):
    """Get detailed drawdown history analysis for a fund."""
    try:
        loop = asyncio.get_running_loop()
        nav_history = await loop.run_in_executor(None, get_fund_nav_history, code, 500)

        if not nav_history:
            raise HTTPException(status_code=404, detail=f"No NAV history found for fund {code}")

        analyzer = DrawdownAnalyzer(threshold=threshold)
        analysis = analyzer.analyze_drawdowns(nav_history)

        return sanitize_for_json(analysis)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error analyzing drawdowns for {code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_funds_advanced(
    request: FundCompareRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Compare multiple funds (up to 10) with comprehensive analysis.
    Includes NAV curves, returns, risk metrics, and holdings overlap.
    """
    try:
        codes = request.codes

        if len(codes) < 2:
            raise HTTPException(status_code=400, detail="Please select at least 2 funds to compare")
        if len(codes) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 funds allowed for comparison")

        loop = asyncio.get_running_loop()

        async def fetch_fund_data(code: str):
            nav_history = await loop.run_in_executor(None, get_fund_nav_history, code, 500)
            fund_info = await loop.run_in_executor(None, get_fund_basic_info, code)
            holdings = await loop.run_in_executor(None, get_fund_holdings_list, code)
            return {
                'code': code,
                'name': fund_info.get('name', code) if fund_info else code,
                'nav_history': nav_history,
                'holdings': holdings,
            }

        tasks = [fetch_fund_data(code) for code in codes]
        funds_data = await asyncio.gather(*tasks)

        # Filter out funds with insufficient data
        valid_funds = [f for f in funds_data if f.get('nav_history') and len(f['nav_history']) >= 20]

        if len(valid_funds) < 2:
            raise HTTPException(status_code=400, detail="Not enough funds with valid data for comparison")

        comparator = FundComparison()
        result = comparator.compare(valid_funds)

        return sanitize_for_json(result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error comparing funds: {e}")
        raise HTTPException(status_code=500, detail=str(e))
