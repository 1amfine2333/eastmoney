"""
Commodity analysis endpoints.
"""
import os
import glob
import asyncio
from typing import List
from fastapi import APIRouter, HTTPException, Depends

from app.models.settings import CommodityAnalyzeRequest
from app.models.reports import ReportSummary
from app.models.auth import User
from app.core.dependencies import get_current_user, get_user_report_dir
from src.analysis.commodities.gold_silver import GoldSilverAnalyst

router = APIRouter(prefix="/api/commodities", tags=["Commodities"])


@router.post("/analyze")
async def analyze_commodity(request: CommodityAnalyzeRequest, current_user: User = Depends(get_current_user)):
    """Analyze a commodity (gold or silver)."""
    try:
        analyst = GoldSilverAnalyst()
        report = await asyncio.to_thread(analyst.analyze, request.asset, current_user.id)
        return {"status": "success", "message": f"{request.asset} analysis complete"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports", response_model=List[ReportSummary])
async def list_commodity_reports(current_user: User = Depends(get_current_user)):
    """List all commodity analysis reports."""
    user_report_dir = get_user_report_dir(current_user.id)
    commodities_dir = os.path.join(user_report_dir, "commodities")

    if not os.path.exists(commodities_dir):
        return []

    reports = []
    files = glob.glob(os.path.join(commodities_dir, "*.md"))
    files.sort(key=os.path.getmtime, reverse=True)

    for f in files:
        filename = os.path.basename(f)
        try:
            name_no_ext = os.path.splitext(filename)[0]
            parts = name_no_ext.split("_")

            if len(parts) >= 5 and parts[2] == 'commodities':
                date_str = parts[0]
                time_str = parts[1]
                code = parts[3]
                name = "_".join(parts[4:])
                formatted_date = f"{date_str} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"

                reports.append(ReportSummary(
                    filename=filename,
                    date=formatted_date,
                    mode="commodities",
                    fund_code=code,
                    fund_name=name,
                    is_summary=False
                ))
            elif len(parts) >= 4 and parts[1] == 'commodities':
                date_str = parts[0]
                code = parts[2]
                name = "_".join(parts[3:])

                reports.append(ReportSummary(
                    filename=filename,
                    date=date_str,
                    mode="commodities",
                    fund_code=code,
                    fund_name=name,
                    is_summary=False
                ))
        except Exception as e:
            print(f"Error parsing commodity report {filename}: {e}")
            continue

    return reports


@router.delete("/reports/{filename}")
async def delete_commodity_report(filename: str, current_user: User = Depends(get_current_user)):
    """Delete a commodity report."""
    try:
        if not filename.endswith(".md") or ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        user_report_dir = get_user_report_dir(current_user.id)
        commodities_dir = os.path.join(user_report_dir, "commodities")
        file_path = os.path.join(commodities_dir, filename)

        if os.path.exists(file_path):
            os.remove(file_path)
            return {"status": "success", "message": f"Deleted {filename}"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting commodity report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
