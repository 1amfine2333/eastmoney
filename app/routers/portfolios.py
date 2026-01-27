"""
Portfolio management endpoints - the largest router with CRUD, analysis, AI, stress testing, and DIP plans.
"""
import asyncio
from datetime import datetime
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Depends, Body

from app.models.portfolios import (
    PortfolioCreate, PortfolioUpdate, PositionCreate, PositionUpdate,
    UnifiedPositionCreate, UnifiedPositionUpdate, TransactionCreate,
    DIPPlanCreate, DIPPlanUpdate, AIRebalanceRequest, PortfolioAIChatRequest
)
from app.models.stress_test import StressTestRequest, AIScenarioRequest, StressTestChatRequest, CorrelationExplainRequest
from app.models.auth import User
from app.core.dependencies import get_current_user
from app.core.utils import sanitize_for_json
from app.core.helpers import (
    get_fund_nav_history, get_stock_price_history, get_index_history,
    enrich_positions_with_prices
)
from src.storage.db import (
    # Portfolio CRUD
    get_user_portfolios, get_portfolio_by_id, get_default_portfolio,
    create_portfolio, update_portfolio, delete_portfolio,
    set_default_portfolio as db_set_default_portfolio,
    # Legacy positions (kept for backwards compatibility)
    get_user_positions, create_position, update_position, delete_position,
    # Unified positions
    get_portfolio_positions, upsert_position, delete_unified_position,
    get_unified_position_by_id, recalculate_position,
    # Transactions
    get_portfolio_transactions, create_transaction, delete_transaction,
    # DIP Plans
    get_portfolio_dip_plans, get_dip_plan_by_id, create_dip_plan,
    update_dip_plan, delete_dip_plan, execute_dip_plan,
    # Alerts
    get_portfolio_alerts, get_unread_alert_count, mark_alert_read, dismiss_alert,
    # Snapshots & Migration
    get_portfolio_snapshots, migrate_fund_positions_to_positions
)
from src.data_sources.akshare_api import get_stock_realtime_quote
from src.analysis.fund import PortfolioAnalyzer
from src.analysis.portfolio import (
    RiskMetricsCalculator as PortfolioRiskMetrics,
    CorrelationAnalyzer,
    StressTestEngine,
    SignalGenerator
)
from src.analysis.portfolio.stress_test import StressScenario, ScenarioType
from src.llm.client import get_llm_client
from src.services.assistant_service import assistant_service

router = APIRouter(tags=["Portfolios"])


# ====================================================================
# Legacy Portfolio Positions (backwards compatibility)
# These endpoints are kept for migration purposes
# ====================================================================

@router.get("/api/portfolio/positions")
async def get_legacy_positions(current_user: User = Depends(get_current_user)):
    """Get all fund positions for current user (legacy endpoint)."""
    try:
        positions = get_user_positions(current_user.id)

        enriched = []
        for pos in positions:
            fund_code = pos['fund_code']

            current_nav = None
            try:
                loop = asyncio.get_running_loop()
                nav_history = await loop.run_in_executor(None, get_fund_nav_history, fund_code, 5)
                if nav_history:
                    current_nav = float(nav_history[-1]['value'])
            except:
                pass

            shares = float(pos.get('shares', 0))
            cost_basis = float(pos.get('cost_basis', 0))
            position_cost = shares * cost_basis
            position_value = shares * (current_nav or cost_basis)
            pnl = position_value - position_cost
            pnl_pct = (current_nav / cost_basis - 1) * 100 if cost_basis > 0 and current_nav else 0

            enriched.append({
                **pos,
                'current_nav': round(current_nav, 4) if current_nav else None,
                'position_cost': round(position_cost, 2),
                'position_value': round(position_value, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
            })

        return {"positions": enriched}
    except Exception as e:
        print(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolio/positions")
async def create_legacy_position(position: PositionCreate, current_user: User = Depends(get_current_user)):
    """Create a new fund position (legacy endpoint)."""
    try:
        position_id = create_position(position.dict(), current_user.id)
        return {"id": position_id, "message": "Position created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error creating position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/portfolio/positions/{position_id}")
async def update_legacy_position(
    position_id: int,
    updates: PositionUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update an existing position (legacy endpoint)."""
    try:
        update_dict = {k: v for k, v in updates.dict().items() if v is not None}
        if not update_dict:
            raise HTTPException(status_code=400, detail="No updates provided")

        success = update_position(position_id, current_user.id, update_dict)
        if not success:
            raise HTTPException(status_code=404, detail="Position not found")

        return {"message": "Position updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/portfolio/positions/{position_id}")
async def delete_legacy_position(position_id: int, current_user: User = Depends(get_current_user)):
    """Delete a position (legacy endpoint)."""
    try:
        success = delete_position(position_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Position not found")

        return {"message": "Position deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolio/summary")
async def get_legacy_portfolio_summary(current_user: User = Depends(get_current_user)):
    """Get portfolio summary (legacy endpoint)."""
    try:
        positions = get_user_positions(current_user.id)

        if not positions:
            return {
                "total_value": 0,
                "total_cost": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "positions": [],
                "allocation": [],
            }

        fund_nav_map = {}
        loop = asyncio.get_running_loop()

        for pos in positions:
            fund_code = pos['fund_code']
            if fund_code not in fund_nav_map:
                try:
                    nav_history = await loop.run_in_executor(None, get_fund_nav_history, fund_code, 5)
                    if nav_history:
                        fund_nav_map[fund_code] = float(nav_history[-1]['value'])
                except:
                    pass

        analyzer = PortfolioAnalyzer()
        summary = analyzer.calculate_portfolio_summary(positions, fund_nav_map)

        return sanitize_for_json(summary)
    except Exception as e:
        print(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolio/overlap")
async def get_legacy_portfolio_overlap(current_user: User = Depends(get_current_user)):
    """Analyze holdings overlap (legacy endpoint)."""
    try:
        from app.core.helpers import get_fund_holdings_list

        positions = get_user_positions(current_user.id)

        if not positions:
            return {"message": "No positions in portfolio"}

        fund_holdings = {}
        position_weights = {}
        loop = asyncio.get_running_loop()

        total_value = 0
        fund_values = {}
        fund_nav_map = {}

        for pos in positions:
            fund_code = pos['fund_code']
            shares = float(pos.get('shares', 0))
            cost_basis = float(pos.get('cost_basis', 1))

            try:
                nav_history = await loop.run_in_executor(None, get_fund_nav_history, fund_code, 5)
                if nav_history:
                    current_nav = float(nav_history[-1]['value'])
                    fund_nav_map[fund_code] = current_nav
                else:
                    current_nav = cost_basis
            except:
                current_nav = cost_basis

            position_value = shares * current_nav
            fund_values[fund_code] = fund_values.get(fund_code, 0) + position_value
            total_value += position_value

        for fund_code, value in fund_values.items():
            position_weights[fund_code] = value / total_value if total_value > 0 else 0

            try:
                holdings = await loop.run_in_executor(None, get_fund_holdings_list, fund_code)
                if holdings:
                    fund_holdings[fund_code] = holdings
            except:
                pass

        if not fund_holdings:
            return {"message": "No holdings data available for portfolio funds"}

        analyzer = PortfolioAnalyzer()
        overlap = analyzer.analyze_holdings_overlap(fund_holdings, position_weights)

        return sanitize_for_json(overlap)
    except Exception as e:
        print(f"Error analyzing portfolio overlap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# New Multi-Portfolio Management API
# ====================================================================

@router.get("/api/portfolios")
async def list_portfolios(current_user: User = Depends(get_current_user)):
    """Get all portfolios for the current user."""
    try:
        portfolios = get_user_portfolios(current_user.id)
        return {"portfolios": portfolios}
    except Exception as e:
        print(f"Error listing portfolios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios")
async def create_new_portfolio(portfolio: PortfolioCreate, current_user: User = Depends(get_current_user)):
    """Create a new portfolio."""
    try:
        portfolio_id = create_portfolio(portfolio.dict(), current_user.id)
        return {"id": portfolio_id, "message": "Portfolio created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error creating portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/default")
async def get_user_default_portfolio(current_user: User = Depends(get_current_user)):
    """Get the default portfolio for the current user (creates one if needed)."""
    try:
        portfolio = get_default_portfolio(current_user.id)
        return portfolio
    except Exception as e:
        print(f"Error getting default portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}")
async def get_portfolio(portfolio_id: int, current_user: User = Depends(get_current_user)):
    """Get a specific portfolio by ID."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        return portfolio
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/portfolios/{portfolio_id}")
async def update_existing_portfolio(
    portfolio_id: int,
    updates: PortfolioUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update an existing portfolio."""
    try:
        update_dict = {k: v for k, v in updates.dict().items() if v is not None}
        if not update_dict:
            raise HTTPException(status_code=400, detail="No updates provided")

        success = update_portfolio(portfolio_id, current_user.id, update_dict)
        if not success:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        return {"message": "Portfolio updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/portfolios/{portfolio_id}")
async def delete_existing_portfolio(portfolio_id: int, current_user: User = Depends(get_current_user)):
    """Delete a portfolio."""
    try:
        success = delete_portfolio(portfolio_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        return {"message": "Portfolio deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/set-default")
async def set_portfolio_as_default(portfolio_id: int, current_user: User = Depends(get_current_user)):
    """Set a portfolio as the default."""
    try:
        success = db_set_default_portfolio(current_user.id, portfolio_id)
        if not success:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        return {"message": "Portfolio set as default"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error setting default portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# Unified Positions API (Stocks + Funds)
# ====================================================================

@router.get("/api/portfolios/{portfolio_id}/positions")
async def get_portfolio_positions_api(
    portfolio_id: int,
    asset_type: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get all positions for a portfolio."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id, asset_type)
        enriched = await enrich_positions_with_prices(positions)

        return {"positions": enriched, "portfolio": portfolio}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting portfolio positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/positions")
async def create_portfolio_position(
    portfolio_id: int,
    position: UnifiedPositionCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new position directly (without transaction)."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        position_data = position.dict()
        position_data['total_cost'] = position_data['total_shares'] * position_data['average_cost']

        position_id = upsert_position(position_data, portfolio_id, current_user.id)
        return {"id": position_id, "message": "Position created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/portfolios/{portfolio_id}/positions/{position_id}")
async def delete_portfolio_position(
    portfolio_id: int,
    position_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete a position."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        success = delete_unified_position(position_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Position not found")

        return {"message": "Position deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# Transactions API
# ====================================================================

@router.get("/api/portfolios/{portfolio_id}/transactions")
async def get_portfolio_transactions_api(
    portfolio_id: int,
    asset_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """Get all transactions for a portfolio."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        transactions = get_portfolio_transactions(portfolio_id, current_user.id, asset_type, limit, offset)
        return {"transactions": transactions, "portfolio": portfolio}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting transactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/transactions")
async def create_portfolio_transaction(
    portfolio_id: int,
    transaction: TransactionCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new transaction and update position."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        transaction_id = create_transaction(transaction.dict(), portfolio_id, current_user.id)
        return {"id": transaction_id, "message": "Transaction created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/portfolios/{portfolio_id}/transactions/{transaction_id}")
async def delete_portfolio_transaction(
    portfolio_id: int,
    transaction_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete a transaction (does not reverse position changes)."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        success = delete_transaction(transaction_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Transaction not found")

        return {"message": "Transaction deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/positions/{position_id}/recalculate")
async def recalculate_portfolio_position(
    portfolio_id: int,
    position_id: int,
    current_user: User = Depends(get_current_user)
):
    """Recalculate a position from all its transactions."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        position = get_unified_position_by_id(position_id, current_user.id)
        if not position:
            raise HTTPException(status_code=404, detail="Position not found")

        updated = recalculate_position(
            portfolio_id, position['asset_type'], position['asset_code'], current_user.id
        )
        return {"position": updated, "message": "Position recalculated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error recalculating position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# Portfolio Analysis API
# ====================================================================

@router.get("/api/portfolios/{portfolio_id}/summary")
async def get_portfolio_summary_new(portfolio_id: int, current_user: User = Depends(get_current_user)):
    """Get comprehensive portfolio summary with total value, P&L, and allocation."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)

        if not positions:
            return {
                "portfolio": portfolio,
                "total_value": 0,
                "total_cost": 0,
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "positions_count": 0,
                "positions": [],
                "allocation": {"by_type": {}, "by_sector": {}},
            }

        enriched_positions = await enrich_positions_with_prices(positions)

        total_cost = sum(float(p.get('total_shares', 0) * p.get('average_cost', 0)) for p in enriched_positions)
        total_value = sum(float(p.get('current_value') or p.get('total_shares', 0) * p.get('average_cost', 0)) for p in enriched_positions)
        total_pnl = total_value - total_cost
        total_pnl_pct = ((total_value / total_cost) - 1) * 100 if total_cost > 0 else 0

        allocation_by_type = {'stock': 0, 'fund': 0}
        allocation_by_sector = {}

        for pos in enriched_positions:
            pos_value = pos.get('current_value') or (pos.get('total_shares', 0) * pos.get('average_cost', 0))
            asset_type = pos.get('asset_type', 'fund')
            sector = pos.get('sector', '未分类')

            allocation_by_type[asset_type] = allocation_by_type.get(asset_type, 0) + pos_value
            allocation_by_sector[sector] = allocation_by_sector.get(sector, 0) + pos_value

        if total_value > 0:
            allocation_by_type = {k: round(v / total_value * 100, 2) for k, v in allocation_by_type.items()}
            allocation_by_sector = {k: round(v / total_value * 100, 2) for k, v in allocation_by_sector.items()}

        return sanitize_for_json({
            "portfolio": portfolio,
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "positions_count": len(enriched_positions),
            "positions": enriched_positions,
            "allocation": {
                "by_type": allocation_by_type,
                "by_sector": allocation_by_sector,
            },
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get portfolio performance history (snapshots)."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        snapshots = get_portfolio_snapshots(portfolio_id, start_date, end_date)

        return sanitize_for_json({
            "portfolio": portfolio,
            "snapshots": snapshots,
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/risk-metrics")
async def get_portfolio_risk_metrics_api(portfolio_id: int, current_user: User = Depends(get_current_user)):
    """Get portfolio risk metrics including concentration, volatility, etc."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        if not enriched:
            return {"message": "No positions to analyze"}

        total_value = sum(float(p.get('current_value') or p.get('total_shares', 0) * p.get('average_cost', 0)) for p in enriched)

        max_position_pct = 0
        position_values = []
        for pos in enriched:
            pos_value = pos.get('current_value') or (pos.get('total_shares', 0) * pos.get('average_cost', 0))
            pos_pct = (pos_value / total_value * 100) if total_value > 0 else 0
            position_values.append(pos_pct)
            max_position_pct = max(max_position_pct, pos_pct)

        hhi = sum(pct ** 2 for pct in position_values)
        concentration_level = "低" if hhi < 1500 else ("中" if hhi < 2500 else "高")
        diversification_score = max(0, min(100, 100 - (hhi / 100)))

        type_concentration = {}
        for pos in enriched:
            pos_value = pos.get('current_value') or (pos.get('total_shares', 0) * pos.get('average_cost', 0))
            asset_type = pos.get('asset_type', 'fund')
            type_concentration[asset_type] = type_concentration.get(asset_type, 0) + pos_value

        type_concentration = {k: round(v / total_value * 100, 2) for k, v in type_concentration.items()} if total_value > 0 else {}

        return sanitize_for_json({
            "portfolio": portfolio,
            "risk_metrics": {
                "max_single_position_pct": round(max_position_pct, 2),
                "hhi": round(hhi, 2),
                "concentration_level": concentration_level,
                "diversification_score": round(diversification_score, 2),
                "type_concentration": type_concentration,
                "positions_count": len(enriched),
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting portfolio risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/benchmark")
async def get_portfolio_benchmark_comparison(
    portfolio_id: int,
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Compare portfolio performance against benchmark index."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        benchmark_code = portfolio.get('benchmark_code', '000300.SH')

        loop = asyncio.get_running_loop()
        try:
            benchmark_history = await loop.run_in_executor(None, get_index_history, benchmark_code, days)
        except:
            benchmark_history = []

        snapshots = get_portfolio_snapshots(portfolio_id, limit=days)

        return sanitize_for_json({
            "portfolio": portfolio,
            "benchmark_code": benchmark_code,
            "benchmark_history": benchmark_history,
            "portfolio_history": snapshots,
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting benchmark comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# Portfolio Alerts API
# ====================================================================

@router.get("/api/portfolios/{portfolio_id}/alerts")
async def get_portfolio_alerts_api(
    portfolio_id: int,
    unread_only: bool = False,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get alerts for a portfolio."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        alerts = get_portfolio_alerts(portfolio_id, current_user.id, unread_only, limit)
        unread_count = get_unread_alert_count(current_user.id)

        return {
            "alerts": alerts,
            "unread_count": unread_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# DIP (定投) Plans API
# ====================================================================

@router.get("/api/portfolios/{portfolio_id}/dip-plans")
async def get_dip_plans_api(
    portfolio_id: int,
    active_only: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Get all DIP plans for a portfolio."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        plans = get_portfolio_dip_plans(portfolio_id, current_user.id, active_only)
        return {"dip_plans": plans, "portfolio": portfolio}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting DIP plans: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/dip-plans")
async def create_dip_plan_api(
    portfolio_id: int,
    plan: DIPPlanCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new DIP plan."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        plan_id = create_dip_plan(plan.dict(), portfolio_id, current_user.id)
        return {"id": plan_id, "message": "DIP plan created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating DIP plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/dip-plans/{plan_id}")
async def get_dip_plan_api(
    portfolio_id: int,
    plan_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get a specific DIP plan."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        plan = get_dip_plan_by_id(plan_id, current_user.id)
        if not plan:
            raise HTTPException(status_code=404, detail="DIP plan not found")

        return plan
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting DIP plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/portfolios/{portfolio_id}/dip-plans/{plan_id}")
async def update_dip_plan_api(
    portfolio_id: int,
    plan_id: int,
    updates: DIPPlanUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update a DIP plan."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        update_dict = {k: v for k, v in updates.dict().items() if v is not None}
        if not update_dict:
            raise HTTPException(status_code=400, detail="No updates provided")

        success = update_dip_plan(plan_id, current_user.id, update_dict)
        if not success:
            raise HTTPException(status_code=404, detail="DIP plan not found")

        return {"message": "DIP plan updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating DIP plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/portfolios/{portfolio_id}/dip-plans/{plan_id}")
async def delete_dip_plan_api(
    portfolio_id: int,
    plan_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete a DIP plan."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        success = delete_dip_plan(plan_id, current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="DIP plan not found")

        return {"message": "DIP plan deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting DIP plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/dip-plans/{plan_id}/execute")
async def execute_dip_plan_api(
    portfolio_id: int,
    plan_id: int,
    price: Optional[float] = None,
    current_user: User = Depends(get_current_user)
):
    """Manually execute a DIP plan."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        plan = get_dip_plan_by_id(plan_id, current_user.id)
        if not plan:
            raise HTTPException(status_code=404, detail="DIP plan not found")

        if not price:
            loop = asyncio.get_running_loop()
            if plan['asset_type'] == 'fund':
                nav_history = await loop.run_in_executor(None, get_fund_nav_history, plan['asset_code'], 5)
                if nav_history:
                    price = float(nav_history[-1]['value'])
            else:
                quote = await loop.run_in_executor(None, get_stock_realtime_quote, plan['asset_code'])
                if quote and quote.get('price'):
                    price = float(quote['price'])

        if not price:
            raise HTTPException(status_code=400, detail="Could not determine current price")

        tx_data = execute_dip_plan(plan_id, current_user.id, price)
        if not tx_data:
            raise HTTPException(status_code=400, detail="Failed to execute DIP plan")

        transaction_id = create_transaction(tx_data, portfolio_id, current_user.id)

        return {
            "message": "DIP plan executed successfully",
            "transaction_id": transaction_id,
            "shares": tx_data['shares'],
            "price": price,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error executing DIP plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# AI Portfolio Features
# ====================================================================

@router.get("/api/portfolios/{portfolio_id}/ai-diagnosis")
async def get_ai_portfolio_diagnosis(portfolio_id: int, current_user: User = Depends(get_current_user)):
    """Get AI-powered portfolio diagnosis with 5-dimension scoring."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        if not enriched:
            return {"message": "No positions to diagnose"}

        total_value = sum(float(p.get('current_value') or p.get('total_shares', 0) * p.get('average_cost', 0)) for p in enriched)

        # Calculate 5-dimension scores
        position_weights = []
        for pos in enriched:
            pos_value = pos.get('current_value') or (pos.get('total_shares', 0) * pos.get('average_cost', 0))
            weight = pos_value / total_value if total_value > 0 else 0
            position_weights.append(weight)

        hhi = sum(w ** 2 for w in position_weights) * 10000
        diversification_score = max(0, min(20, 20 - (hhi / 500)))

        pnl_pcts = [p.get('unrealized_pnl_pct', 0) or 0 for p in enriched]
        avg_pnl = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0
        risk_efficiency_score = max(0, min(20, 10 + (avg_pnl / 10)))

        type_counts = {'stock': 0, 'fund': 0}
        for pos in enriched:
            pos_value = pos.get('current_value') or (pos.get('total_shares', 0) * pos.get('average_cost', 0))
            type_counts[pos.get('asset_type', 'fund')] += pos_value

        total = sum(type_counts.values())
        if total > 0:
            balance_ratio = min(type_counts.values()) / max(type_counts.values()) if max(type_counts.values()) > 0 else 0
            allocation_score = 10 + (balance_ratio * 10)
        else:
            allocation_score = 10

        positive_positions = sum(1 for p in pnl_pcts if p > 0)
        momentum_score = (positive_positions / len(pnl_pcts)) * 20 if pnl_pcts else 10

        valuation_ratios = []
        for pos in enriched:
            avg_cost = pos.get('average_cost', 1)
            current_price = pos.get('current_price', avg_cost)
            if avg_cost > 0 and current_price:
                ratio = current_price / avg_cost
                valuation_ratios.append(ratio)

        avg_ratio = sum(valuation_ratios) / len(valuation_ratios) if valuation_ratios else 1
        if avg_ratio > 1.5:
            valuation_score = max(5, 15 - (avg_ratio - 1.5) * 10)
        elif avg_ratio < 0.8:
            valuation_score = max(5, 15 - (0.8 - avg_ratio) * 10)
        else:
            valuation_score = 15 + abs(1 - avg_ratio) * 10

        valuation_score = max(0, min(20, valuation_score))

        total_score = diversification_score + risk_efficiency_score + allocation_score + momentum_score + valuation_score

        recommendations = []
        if diversification_score < 12:
            recommendations.append("建议增加持仓多样性，当前集中度较高")
        if risk_efficiency_score < 10:
            recommendations.append("组合整体收益偏低，考虑调整持仓结构")
        if allocation_score < 12:
            recommendations.append("资产配置不够均衡，建议增加股票/基金比例")
        if momentum_score < 10:
            recommendations.append("多数持仓处于亏损状态，建议审视持仓策略")
        if valuation_score < 12:
            recommendations.append("部分持仓估值偏离合理区间")

        return sanitize_for_json({
            "portfolio": portfolio,
            "total_score": round(total_score, 1),
            "max_score": 100,
            "grade": "A" if total_score >= 80 else ("B" if total_score >= 60 else ("C" if total_score >= 40 else "D")),
            "dimensions": [
                {"name": "分散化", "score": round(diversification_score, 1), "max": 20},
                {"name": "风险效率", "score": round(risk_efficiency_score, 1), "max": 20},
                {"name": "配置质量", "score": round(allocation_score, 1), "max": 20},
                {"name": "动量", "score": round(momentum_score, 1), "max": 20},
                {"name": "估值", "score": round(valuation_score, 1), "max": 20},
            ],
            "recommendations": recommendations,
            "analyzed_at": datetime.now().isoformat(),
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating AI diagnosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/ai-rebalance")
async def get_ai_rebalance_suggestions(
    portfolio_id: int,
    request: AIRebalanceRequest = Body(default=AIRebalanceRequest()),
    current_user: User = Depends(get_current_user)
):
    """Get AI-powered rebalancing suggestions."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        if not enriched:
            return {"message": "No positions to analyze"}

        total_value = sum(float(p.get('current_value') or p.get('total_shares', 0) * p.get('average_cost', 0)) for p in enriched)

        suggestions = []

        for pos in enriched:
            pos_value = pos.get('current_value') or (pos.get('total_shares', 0) * pos.get('average_cost', 0))
            weight = (pos_value / total_value * 100) if total_value > 0 else 0
            pnl_pct = pos.get('unrealized_pnl_pct', 0) or 0

            if weight > 30:
                suggestions.append({
                    "asset_code": pos['asset_code'],
                    "asset_name": pos.get('asset_name', pos['asset_code']),
                    "action": "reduce",
                    "reason": f"仓位占比{weight:.1f}%过高，建议减仓至30%以下",
                    "priority": "high",
                })

            if pnl_pct < -20:
                suggestions.append({
                    "asset_code": pos['asset_code'],
                    "asset_name": pos.get('asset_name', pos['asset_code']),
                    "action": "review",
                    "reason": f"亏损{abs(pnl_pct):.1f}%，建议审视是否止损",
                    "priority": "medium",
                })

            if pnl_pct > 50:
                suggestions.append({
                    "asset_code": pos['asset_code'],
                    "asset_name": pos.get('asset_name', pos['asset_code']),
                    "action": "consider_reduce",
                    "reason": f"盈利{pnl_pct:.1f}%，可考虑部分止盈",
                    "priority": "low",
                })

        type_values = {'stock': 0, 'fund': 0}
        for pos in enriched:
            pos_value = pos.get('current_value') or (pos.get('total_shares', 0) * pos.get('average_cost', 0))
            type_values[pos.get('asset_type', 'fund')] += pos_value

        stock_pct = (type_values['stock'] / total_value * 100) if total_value > 0 else 0
        fund_pct = (type_values['fund'] / total_value * 100) if total_value > 0 else 0

        if request.risk_preference == "conservative":
            if stock_pct > 40:
                suggestions.append({
                    "asset_code": None,
                    "asset_name": "整体配置",
                    "action": "adjust",
                    "reason": f"股票占比{stock_pct:.1f}%偏高，保守型投资者建议控制在40%以下",
                    "priority": "medium",
                })
        elif request.risk_preference == "aggressive":
            if fund_pct > 60:
                suggestions.append({
                    "asset_code": None,
                    "asset_name": "整体配置",
                    "action": "adjust",
                    "reason": f"基金占比{fund_pct:.1f}%偏高，进取型投资者可增加股票比例",
                    "priority": "low",
                })

        return sanitize_for_json({
            "portfolio": portfolio,
            "current_allocation": {
                "stock": round(stock_pct, 2),
                "fund": round(fund_pct, 2),
            },
            "suggestions": suggestions,
            "generated_at": datetime.now().isoformat(),
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating rebalance suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/ai-chat")
async def portfolio_ai_chat(
    portfolio_id: int,
    request: PortfolioAIChatRequest,
    current_user: User = Depends(get_current_user)
):
    """AI chat specifically about the portfolio."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        total_value = sum(float(p.get('current_value') or p.get('total_shares', 0) * p.get('average_cost', 0)) for p in enriched)
        total_cost = sum(float(p.get('total_shares', 0) * p.get('average_cost', 0)) for p in enriched)
        total_pnl = total_value - total_cost

        portfolio_context = {
            "portfolio_name": portfolio.get('name', '我的组合'),
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round((total_pnl / total_cost * 100) if total_cost > 0 else 0, 2),
            "positions": [
                {
                    "name": p.get('asset_name', p['asset_code']),
                    "type": p['asset_type'],
                    "value": p.get('current_value'),
                    "pnl_pct": p.get('unrealized_pnl_pct'),
                }
                for p in enriched
            ],
        }

        context = {
            "page": "portfolio",
            "portfolio": portfolio_context,
            **(request.context or {}),
        }

        response = await assistant_service.chat(
            message=request.message,
            context=context,
            history=[]
        )

        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in portfolio AI chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# Portfolio Stress Testing & Advanced Analytics API
# ====================================================================

@router.post("/api/portfolios/{portfolio_id}/stress-test")
async def run_portfolio_stress_test(
    portfolio_id: int,
    request: StressTestRequest,
    current_user: User = Depends(get_current_user)
):
    """Run stress test on portfolio with macro factor scenarios."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        if not enriched:
            return {"message": "No positions to analyze"}

        current_prices = {}
        for pos in enriched:
            code = pos.get('asset_code')
            price = pos.get('current_price') or pos.get('average_cost', 0)
            current_prices[code] = float(price)

        engine = StressTestEngine()

        if request.scenario_type:
            try:
                scenario_enum = ScenarioType(request.scenario_type)
                result = engine.run_predefined_scenario(enriched, scenario_enum, current_prices)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown scenario type: {request.scenario_type}")
        elif request.scenario:
            scenario = StressScenario(
                interest_rate_change_bp=request.scenario.get('interest_rate_change_bp', 0),
                fx_change_pct=request.scenario.get('fx_change_pct', 0),
                index_change_pct=request.scenario.get('index_change_pct', 0),
                oil_change_pct=request.scenario.get('oil_change_pct', 0),
                sector_shocks=request.scenario.get('sector_shocks')
            )
            result = engine.run_stress_test(enriched, scenario, current_prices)
        else:
            scenario = StressScenario(index_change_pct=-5.0)
            result = engine.run_stress_test(enriched, scenario, current_prices)

        return sanitize_for_json(result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error running stress test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/stress-test/scenarios")
async def get_stress_test_scenarios(current_user: User = Depends(get_current_user)):
    """Get available predefined stress test scenarios and factor sliders."""
    return {
        "scenarios": StressTestEngine.get_available_scenarios(),
        "sliders": StressTestEngine.get_factor_sliders()
    }


@router.post("/api/portfolios/{portfolio_id}/stress-test/ai-scenarios")
async def generate_ai_stress_scenario(
    portfolio_id: int,
    request: AIScenarioRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate AI-powered stress test scenario based on current market conditions."""
    try:
        from src.services.ai_scenario_service import ai_scenario_service

        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        result = await ai_scenario_service.generate_scenario(request.category)
        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating AI scenario: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/stress-test/chat")
async def stress_test_chat(
    portfolio_id: int,
    request: StressTestChatRequest,
    current_user: User = Depends(get_current_user)
):
    """Conversational stress testing interface."""
    try:
        from src.services.ai_scenario_service import stress_test_chat_service

        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        total_value = sum(float(p.get('current_value') or 0) for p in enriched)
        total_cost = sum(float(p.get('total_cost') or 0) for p in enriched)
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        portfolio_summary = {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "position_count": len(enriched)
        }

        chat_result = await stress_test_chat_service.chat(
            message=request.message,
            portfolio_id=portfolio_id,
            portfolio_summary=portfolio_summary,
            history=request.history
        )

        stress_result = None
        if chat_result.get("should_run_stress_test") and chat_result.get("scenario_params"):
            try:
                current_prices = {}
                for pos in enriched:
                    code = pos.get('asset_code')
                    price = pos.get('current_price') or pos.get('average_cost', 0)
                    current_prices[code] = float(price)

                params = chat_result["scenario_params"]
                scenario = StressScenario(
                    interest_rate_change_bp=params.get('interest_rate_change_bp', 0),
                    fx_change_pct=params.get('fx_change_pct', 0),
                    index_change_pct=params.get('index_change_pct', 0),
                    oil_change_pct=params.get('oil_change_pct', 0)
                )

                engine = StressTestEngine()
                stress_result = engine.run_stress_test(enriched, scenario, current_prices)
                stress_result = sanitize_for_json(stress_result)
            except Exception as e:
                print(f"Stress test execution failed: {e}")

        return {
            "response": chat_result.get("response", ""),
            "stress_result": stress_result,
            "scenario_used": chat_result.get("scenario_params"),
            "suggested_followups": chat_result.get("suggested_followups", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in stress test chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/correlation")
async def get_portfolio_correlation(
    portfolio_id: int,
    days: int = 90,
    current_user: User = Depends(get_current_user)
):
    """Get correlation matrix for portfolio positions."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        print(f"[Correlation] Portfolio {portfolio_id}: {len(enriched)} positions")

        if not enriched or len(enriched) < 2:
            return {"message": "Need at least 2 positions for correlation analysis"}

        loop = asyncio.get_running_loop()
        price_histories = {}

        for pos in enriched:
            code = pos.get('asset_code')
            asset_type = pos.get('asset_type')

            try:
                if asset_type == 'fund':
                    history = await loop.run_in_executor(None, get_fund_nav_history, code, days)
                    if history:
                        price_histories[code] = [{'date': h['date'], 'price': h['value']} for h in history]
                        print(f"[Correlation] Fund {code}: {len(history)} data points")
                    else:
                        print(f"[Correlation] Fund {code}: No history data")
                else:
                    history = await loop.run_in_executor(None, get_stock_price_history, code, days)
                    if history:
                        price_histories[code] = history
                        print(f"[Correlation] Stock {code}: {len(history)} data points")
                    else:
                        print(f"[Correlation] Stock {code}: No history data")
            except Exception as e:
                print(f"[Correlation] Error fetching history for {code}: {e}")

        print(f"[Correlation] Total assets with history: {len(price_histories)}")

        analyzer = CorrelationAnalyzer(lookback_days=days)
        result = analyzer.calculate_correlation_matrix(enriched, price_histories)

        print(f"[Correlation] Result: size={result.get('size', 0)}, message={result.get('message', 'OK')}")

        return sanitize_for_json(result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/correlation/explain")
async def explain_portfolio_correlation(
    portfolio_id: int,
    request: CorrelationExplainRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate AI explanation for portfolio correlation matrix."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        correlation_data = request.correlation_data

        labels = correlation_data.get('labels', [])
        high_correlations = correlation_data.get('high_correlations', [])
        diversification_score = correlation_data.get('diversification_score', 0)
        diversification_status = correlation_data.get('diversification_status', 'unknown')

        high_corr_text = ""
        if high_correlations:
            pairs = []
            for hc in high_correlations[:5]:
                pairs.append(f"- {hc.get('name_a', '')} 与 {hc.get('name_b', '')}: {hc.get('correlation', 0):.2f}")
            high_corr_text = "\n".join(pairs)
        else:
            high_corr_text = "无显著高相关性持仓对"

        prompt = f"""你是一位专业的投资组合分析师。请根据以下持仓相关性数据，用简洁易懂的语言向普通投资者解释：

## 持仓列表
{', '.join(labels) if labels else '暂无持仓'}

## 分散化评分
- 得分: {diversification_score:.0f}/100
- 状态: {diversification_status}

## 高相关性持仓对
{high_corr_text}

请用2-3句话解释：
1. 这个组合的分散化程度如何？
2. 如果有高相关性的持仓，意味着什么风险？
3. 给出一个简短的建议

要求：
- 使用简单易懂的语言，避免专业术语
- 直接给出分析结论，不要重复数据
- 控制在100字以内
- 使用中文回答"""

        loop = asyncio.get_running_loop()
        llm_client = get_llm_client()
        explanation = await loop.run_in_executor(None, llm_client.generate_content, prompt)

        return {"explanation": explanation}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating correlation explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/signals")
async def get_portfolio_signals(portfolio_id: int, current_user: User = Depends(get_current_user)):
    """Get AI smart signals for all positions in portfolio."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        if not enriched:
            return {"signals": [], "message": "No positions to analyze"}

        loop = asyncio.get_running_loop()
        price_histories = {}

        for pos in enriched:
            code = pos.get('asset_code')
            asset_type = pos.get('asset_type')

            try:
                if asset_type == 'fund':
                    history = await loop.run_in_executor(None, get_fund_nav_history, code, 60)
                    if history:
                        price_histories[code] = [{'date': h['date'], 'price': h['value']} for h in history]
                else:
                    history = await loop.run_in_executor(None, get_stock_price_history, code, 60)
                    if history:
                        price_histories[code] = history
            except Exception as e:
                print(f"Error fetching history for {code}: {e}")

        generator = SignalGenerator()
        signals = generator.generate_signals(
            positions=enriched,
            price_histories=price_histories,
            fund_flows=None,
            sentiments=None,
            correlations=None,
            news_events=None
        )

        signal_counts = {
            "opportunity": sum(1 for s in signals if s['signal_type'] == 'opportunity'),
            "risk": sum(1 for s in signals if s['signal_type'] == 'risk'),
            "neutral": sum(1 for s in signals if s['signal_type'] == 'neutral')
        }

        return sanitize_for_json({
            "signals": signals,
            "counts": signal_counts,
            "total": len(signals),
            "generated_at": datetime.now().isoformat()
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/signals/{asset_code}")
async def get_signal_detail(
    portfolio_id: int,
    asset_code: str,
    current_user: User = Depends(get_current_user)
):
    """Get detailed signal analysis for a specific position."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        position = None
        for pos in enriched:
            if pos.get('asset_code') == asset_code:
                position = pos
                break

        if not position:
            raise HTTPException(status_code=404, detail="Position not found")

        loop = asyncio.get_running_loop()
        asset_type = position.get('asset_type')
        price_history = []

        try:
            if asset_type == 'fund':
                history = await loop.run_in_executor(None, get_fund_nav_history, asset_code, 60)
                if history:
                    price_history = [{'date': h['date'], 'price': h['value']} for h in history]
            else:
                history = await loop.run_in_executor(None, get_stock_price_history, asset_code, 60)
                if history:
                    price_history = history
        except Exception as e:
            print(f"Error fetching history for {asset_code}: {e}")

        generator = SignalGenerator()
        detail = generator.get_signal_detail(
            position=position,
            price_history=price_history,
            fund_flow=None,
            sentiment=None,
            correlation_data=None,
            news_events=None,
            all_positions=enriched
        )

        return sanitize_for_json(detail)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting signal detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/risk-summary")
async def get_portfolio_risk_summary(portfolio_id: int, current_user: User = Depends(get_current_user)):
    """Get comprehensive risk summary including Beta, Sharpe, VaR, and Health Score."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        if not enriched:
            return {
                "var_95": None,
                "message": "No positions to analyze"
            }

        loop = asyncio.get_running_loop()
        price_histories = {}
        current_prices = {}

        for pos in enriched:
            code = pos.get('asset_code')
            asset_type = pos.get('asset_type')
            current_prices[code] = float(pos.get('current_price') or pos.get('average_cost', 0))

            try:
                if asset_type == 'fund':
                    history = await loop.run_in_executor(None, get_fund_nav_history, code, 90)
                    if history:
                        price_histories[code] = [{'date': h['date'], 'price': h['value']} for h in history]
                else:
                    history = await loop.run_in_executor(None, get_stock_price_history, code, 90)
                    if history:
                        price_histories[code] = history
            except Exception as e:
                print(f"Error fetching history for {code}: {e}")

        benchmark_code = portfolio.get('benchmark_code', '000300.SH')
        try:
            benchmark_history = await loop.run_in_executor(None, get_index_history, benchmark_code, 90)
            benchmark_history = [{'date': h['date'], 'price': h['close']} for h in benchmark_history] if benchmark_history else []
        except:
            benchmark_history = []

        calculator = PortfolioRiskMetrics()
        result = calculator.calculate_risk_summary(
            positions=enriched,
            price_histories=price_histories,
            benchmark_history=benchmark_history,
            current_prices=current_prices
        )

        return sanitize_for_json(result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calculating risk summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/sparkline")
async def get_portfolio_sparkline(
    portfolio_id: int,
    days: int = 7,
    current_user: User = Depends(get_current_user)
):
    """Get sparkline data (mini chart) for portfolio value over time."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        snapshots = get_portfolio_snapshots(portfolio_id, limit=days + 5)

        if not snapshots:
            positions = get_portfolio_positions(portfolio_id, current_user.id)
            enriched = await enrich_positions_with_prices(positions)
            total_value = sum(
                float(p.get('current_value') or p.get('total_shares', 0) * p.get('average_cost', 0))
                for p in enriched
            )

            return {
                "portfolio_id": portfolio_id,
                "values": [round(total_value, 2)],
                "dates": [datetime.now().strftime('%Y-%m-%d')],
                "change": 0,
                "change_pct": 0,
                "trend": "flat",
                "days": 1
            }

        calculator = PortfolioRiskMetrics()
        result = calculator.calculate_sparkline_data(portfolio_id, snapshots, days)

        return sanitize_for_json(result)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting sparkline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# Returns Analysis API
# ====================================================================

@router.get("/api/portfolios/{portfolio_id}/returns/summary")
async def get_returns_summary(
    portfolio_id: int,
    current_user: User = Depends(get_current_user)
):
    """Get returns summary with key metrics: total return, annualized, today/week/month, max drawdown, win rate."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        snapshots = get_portfolio_snapshots(portfolio_id, limit=365)

        if not snapshots:
            positions = get_portfolio_positions(portfolio_id, current_user.id)
            enriched = await enrich_positions_with_prices(positions)
            total_value = sum(float(p.get('current_value') or 0) for p in enriched)
            total_cost = sum(float(p.get('total_cost') or 0) for p in enriched)
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

            return {
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 2),
                "annualized_return": 0,
                "today_pnl": 0,
                "today_pnl_pct": 0,
                "week_pnl": 0,
                "week_pnl_pct": 0,
                "month_pnl": 0,
                "month_pnl_pct": 0,
                "max_drawdown": 0,
                "max_drawdown_pct": 0,
                "profitable_days": 0,
                "total_trading_days": 0,
                "best_day": None,
                "worst_day": None,
            }

        # Sort snapshots by date
        snapshots = sorted(snapshots, key=lambda x: x['snapshot_date'])

        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(snapshots)):
            prev_value = float(snapshots[i-1].get('total_value', 0))
            curr_value = float(snapshots[i].get('total_value', 0))
            if prev_value > 0:
                daily_pnl = curr_value - prev_value
                daily_pnl_pct = (daily_pnl / prev_value) * 100
                daily_returns.append({
                    'date': snapshots[i]['snapshot_date'],
                    'pnl': daily_pnl,
                    'pnl_pct': daily_pnl_pct,
                    'value': curr_value,
                })

        # Get current values
        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)
        current_value = sum(float(p.get('current_value') or 0) for p in enriched)
        total_cost = sum(float(p.get('total_cost') or 0) for p in enriched)

        # Total P&L
        total_pnl = current_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        # Annualized return
        if len(snapshots) >= 2:
            first_date = datetime.strptime(snapshots[0]['snapshot_date'], '%Y-%m-%d')
            last_date = datetime.strptime(snapshots[-1]['snapshot_date'], '%Y-%m-%d')
            days = (last_date - first_date).days
            if days > 0:
                first_value = float(snapshots[0].get('total_value', 0))
                total_return = (current_value / first_value) - 1 if first_value > 0 else 0
                annualized_return = ((1 + total_return) ** (365 / days) - 1) * 100
            else:
                annualized_return = 0
        else:
            annualized_return = 0

        # Today's P&L
        today = datetime.now().strftime('%Y-%m-%d')
        today_data = next((r for r in daily_returns if r['date'] == today), None)
        today_pnl = today_data['pnl'] if today_data else 0
        today_pnl_pct = today_data['pnl_pct'] if today_data else 0

        # Week P&L (last 5 trading days)
        week_returns = daily_returns[-5:] if len(daily_returns) >= 5 else daily_returns
        week_pnl = sum(r['pnl'] for r in week_returns)
        if week_returns and len(daily_returns) > len(week_returns):
            week_start_value = daily_returns[-len(week_returns)-1]['value'] if len(daily_returns) > len(week_returns) else total_cost
            week_pnl_pct = (week_pnl / week_start_value * 100) if week_start_value > 0 else 0
        else:
            week_pnl_pct = 0

        # Month P&L (last 22 trading days)
        month_returns = daily_returns[-22:] if len(daily_returns) >= 22 else daily_returns
        month_pnl = sum(r['pnl'] for r in month_returns)
        if month_returns and len(daily_returns) > len(month_returns):
            month_start_value = daily_returns[-len(month_returns)-1]['value'] if len(daily_returns) > len(month_returns) else total_cost
            month_pnl_pct = (month_pnl / month_start_value * 100) if month_start_value > 0 else 0
        else:
            month_pnl_pct = 0

        # Max drawdown
        max_drawdown = 0
        max_drawdown_pct = 0
        peak = float(snapshots[0].get('total_value', 0)) if snapshots else 0
        for snap in snapshots:
            value = float(snap.get('total_value', 0))
            if value > peak:
                peak = value
            drawdown = peak - value
            drawdown_pct = (drawdown / peak * 100) if peak > 0 else 0
            if drawdown_pct > max_drawdown_pct:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        # Win rate
        profitable_days = sum(1 for r in daily_returns if r['pnl'] > 0)
        total_trading_days = len(daily_returns)

        # Best and worst day
        best_day = max(daily_returns, key=lambda x: x['pnl_pct']) if daily_returns else None
        worst_day = min(daily_returns, key=lambda x: x['pnl_pct']) if daily_returns else None

        return sanitize_for_json({
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "annualized_return": round(annualized_return, 2),
            "today_pnl": round(today_pnl, 2),
            "today_pnl_pct": round(today_pnl_pct, 2),
            "week_pnl": round(week_pnl, 2),
            "week_pnl_pct": round(week_pnl_pct, 2),
            "month_pnl": round(month_pnl, 2),
            "month_pnl_pct": round(month_pnl_pct, 2),
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "profitable_days": profitable_days,
            "total_trading_days": total_trading_days,
            "best_day": {
                "date": best_day['date'],
                "pnl": round(best_day['pnl'], 2),
                "pnl_pct": round(best_day['pnl_pct'], 2),
            } if best_day else None,
            "worst_day": {
                "date": worst_day['date'],
                "pnl": round(worst_day['pnl'], 2),
                "pnl_pct": round(worst_day['pnl_pct'], 2),
            } if worst_day else None,
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting returns summary: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/returns/calendar")
async def get_returns_calendar(
    portfolio_id: int,
    view: str = 'day',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get returns calendar data for heatmap (day/month/year view)."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        # Get all snapshots for the period
        limit = 365 if view == 'day' else 365 * 5
        snapshots = get_portfolio_snapshots(portfolio_id, limit=limit)

        if not snapshots:
            return {
                "view": view,
                "data": [],
                "stats": {
                    "total_periods": 0,
                    "profitable_periods": 0,
                    "loss_periods": 0,
                    "best_period": None,
                    "worst_period": None,
                }
            }

        snapshots = sorted(snapshots, key=lambda x: x['snapshot_date'])

        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(snapshots)):
            prev_value = float(snapshots[i-1].get('total_value', 0))
            curr_value = float(snapshots[i].get('total_value', 0))
            if prev_value > 0:
                pnl = curr_value - prev_value
                pnl_pct = (pnl / prev_value) * 100
                daily_returns.append({
                    'date': snapshots[i]['snapshot_date'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'is_trading_day': True,
                })

        if view == 'day':
            data = daily_returns
        elif view == 'month':
            # Aggregate by month
            monthly = {}
            for r in daily_returns:
                month_key = r['date'][:7]  # YYYY-MM
                if month_key not in monthly:
                    monthly[month_key] = {'pnl': 0, 'start_value': None}
                monthly[month_key]['pnl'] += r['pnl']

            # Calculate monthly percentages
            data = []
            for month, vals in sorted(monthly.items()):
                # Find the start value for the month
                month_start = f"{month}-01"
                start_snap = next((s for s in snapshots if s['snapshot_date'] >= month_start), None)
                start_value = float(start_snap.get('total_value', 0)) if start_snap else 0
                pnl_pct = (vals['pnl'] / start_value * 100) if start_value > 0 else 0
                data.append({
                    'date': month,
                    'pnl': round(vals['pnl'], 2),
                    'pnl_pct': round(pnl_pct, 2),
                })
        else:  # year
            # Aggregate by year
            yearly = {}
            for r in daily_returns:
                year_key = r['date'][:4]  # YYYY
                if year_key not in yearly:
                    yearly[year_key] = {'pnl': 0}
                yearly[year_key]['pnl'] += r['pnl']

            # Calculate yearly percentages
            data = []
            for year, vals in sorted(yearly.items()):
                year_start = f"{year}-01-01"
                start_snap = next((s for s in snapshots if s['snapshot_date'] >= year_start), None)
                start_value = float(start_snap.get('total_value', 0)) if start_snap else 0
                pnl_pct = (vals['pnl'] / start_value * 100) if start_value > 0 else 0
                data.append({
                    'date': year,
                    'pnl': round(vals['pnl'], 2),
                    'pnl_pct': round(pnl_pct, 2),
                })

        # Calculate stats
        profitable = [d for d in data if d['pnl'] > 0]
        losses = [d for d in data if d['pnl'] < 0]
        best = max(data, key=lambda x: x['pnl_pct']) if data else None
        worst = min(data, key=lambda x: x['pnl_pct']) if data else None

        return sanitize_for_json({
            "view": view,
            "data": data,
            "stats": {
                "total_periods": len(data),
                "profitable_periods": len(profitable),
                "loss_periods": len(losses),
                "best_period": {"date": best['date'], "pnl_pct": round(best['pnl_pct'], 2)} if best else None,
                "worst_period": {"date": worst['date'], "pnl_pct": round(worst['pnl_pct'], 2)} if worst else None,
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting returns calendar: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/portfolios/{portfolio_id}/returns/daily-detail")
async def get_daily_returns_detail(
    portfolio_id: int,
    date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get detailed daily returns for each position."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        positions = get_portfolio_positions(portfolio_id, current_user.id)
        enriched = await enrich_positions_with_prices(positions)

        if not enriched:
            return {
                "date": date or datetime.now().strftime('%Y-%m-%d'),
                "total_pnl": 0,
                "total_pnl_pct": 0,
                "positions": [],
                "top_contributors": [],
                "top_detractors": [],
            }

        loop = asyncio.get_running_loop()
        position_returns = []
        total_value = sum(float(p.get('current_value') or 0) for p in enriched)
        total_daily_pnl = 0

        for pos in enriched:
            code = pos.get('asset_code')
            asset_type = pos.get('asset_type')
            shares = float(pos.get('total_shares') or 0)
            current_price = float(pos.get('current_price') or pos.get('average_cost', 0))
            current_value = float(pos.get('current_value') or 0)
            weight_pct = (current_value / total_value * 100) if total_value > 0 else 0

            # Get yesterday's price
            yesterday_price = current_price
            try:
                if asset_type == 'fund':
                    history = await loop.run_in_executor(None, get_fund_nav_history, code, 5)
                    if history and len(history) >= 2:
                        yesterday_price = float(history[-2]['value'])
                else:
                    history = await loop.run_in_executor(None, get_stock_price_history, code, 5)
                    if history and len(history) >= 2:
                        yesterday_price = float(history[-2]['price'])
            except:
                pass

            # Calculate daily P&L
            nav_change = current_price - yesterday_price
            nav_change_pct = (nav_change / yesterday_price * 100) if yesterday_price > 0 else 0
            position_pnl = shares * nav_change
            position_pnl_pct = nav_change_pct  # Same as price change percentage

            total_daily_pnl += position_pnl

            position_returns.append({
                'position_id': pos.get('id'),
                'asset_code': code,
                'asset_name': pos.get('asset_name', code),
                'asset_type': asset_type,
                'shares': shares,
                'yesterday_nav': round(yesterday_price, 4),
                'today_nav': round(current_price, 4),
                'nav_change': round(nav_change, 4),
                'nav_change_pct': round(nav_change_pct, 2),
                'position_pnl': round(position_pnl, 2),
                'position_pnl_pct': round(position_pnl_pct, 2),
                'contribution_pct': 0,  # Will be calculated after
                'market_value': round(current_value, 2),
                'weight_pct': round(weight_pct, 2),
            })

        # Calculate contribution percentages
        for pr in position_returns:
            if total_daily_pnl != 0:
                pr['contribution_pct'] = round((pr['position_pnl'] / abs(total_daily_pnl)) * 100, 2)

        # Sort by contribution
        sorted_returns = sorted(position_returns, key=lambda x: x['position_pnl'], reverse=True)
        top_contributors = [p for p in sorted_returns if p['position_pnl'] > 0][:3]
        top_detractors = [p for p in sorted_returns if p['position_pnl'] < 0][:3]

        # Total P&L percentage
        yesterday_total = sum(float(p.get('total_shares', 0)) * p.get('yesterday_nav', 0) for p in position_returns)
        total_pnl_pct = (total_daily_pnl / yesterday_total * 100) if yesterday_total > 0 else 0

        return sanitize_for_json({
            "date": date or datetime.now().strftime('%Y-%m-%d'),
            "total_pnl": round(total_daily_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "positions": sorted_returns,
            "top_contributors": top_contributors,
            "top_detractors": top_detractors,
        })
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting daily returns detail: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/portfolios/{portfolio_id}/returns/explain")
async def explain_daily_returns(
    portfolio_id: int,
    date: Optional[str] = Body(None),
    include_market_context: bool = Body(True),
    current_user: User = Depends(get_current_user)
):
    """Generate AI explanation for daily returns."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        # Get daily detail
        detail = await get_daily_returns_detail(portfolio_id, date, current_user)

        if not detail['positions']:
            return {
                "date": detail['date'],
                "explanation": "暂无持仓数据，无法生成收益解读。",
                "generated_at": datetime.now().isoformat(),
            }

        # Build prompt
        contributors_text = "\n".join([
            f"- {p['asset_name']} ({p['asset_code']}): +{p['position_pnl']:.2f}元 (+{p['nav_change_pct']:.2f}%)"
            for p in detail['top_contributors']
        ]) or "无"

        detractors_text = "\n".join([
            f"- {p['asset_name']} ({p['asset_code']}): {p['position_pnl']:.2f}元 ({p['nav_change_pct']:.2f}%)"
            for p in detail['top_detractors']
        ]) or "无"

        total_value = sum(p['market_value'] for p in detail['positions'])

        # Get market context if requested
        market_context = ""
        if include_market_context:
            try:
                from src.data_sources.akshare_api import get_market_indices
                loop = asyncio.get_running_loop()
                indices = await loop.run_in_executor(None, get_market_indices)
                if indices:
                    sh_idx = next((i for i in indices if '上证' in i.get('name', '')), None)
                    sz_idx = next((i for i in indices if '深证' in i.get('name', '')), None)
                    market_context = f"""
## 市场背景
- 上证指数: {sh_idx['change_pct']:.2f}%
- 深证成指: {sz_idx['change_pct']:.2f}%
""" if sh_idx and sz_idx else ""
            except:
                pass

        prompt = f"""你是一位专业的投资组合分析师。请根据以下今日收益数据，用2-4句话解读今日收益表现：

## 今日收益概况
- 日期: {detail['date']}
- 总收益: ¥{detail['total_pnl']:.2f} ({detail['total_pnl_pct']:.2f}%)
- 组合总市值: ¥{total_value:.2f}

## 主要贡献者（涨幅最大）
{contributors_text}

## 主要拖累者（跌幅最大）
{detractors_text}
{market_context}
请分析：
1. 今日收益的主要驱动因素
2. 表现突出或异常的持仓
3. 是否跑赢/跑输大盘（如有市场数据）
4. 简短的操作建议（如有）

请用简洁专业的语言回答，控制在100字以内。"""

        llm = get_llm_client()
        explanation = await asyncio.get_event_loop().run_in_executor(
            None, llm.generate_content, prompt
        )

        return {
            "date": detail['date'],
            "explanation": explanation.strip(),
            "generated_at": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating returns explanation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================================
# Data Migration API
# ====================================================================

@router.post("/api/portfolios/{portfolio_id}/migrate-positions")
async def migrate_old_positions(portfolio_id: int, current_user: User = Depends(get_current_user)):
    """Migrate old fund_positions to the new positions table."""
    try:
        portfolio = get_portfolio_by_id(portfolio_id, current_user.id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        migrated_count = migrate_fund_positions_to_positions(current_user.id, portfolio_id)
        return {
            "message": f"Successfully migrated {migrated_count} positions",
            "migrated_count": migrated_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error migrating positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
