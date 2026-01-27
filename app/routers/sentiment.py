"""
Sentiment analysis endpoints.
"""
import os
import glob
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends

from app.models.auth import User
from app.core.dependencies import get_current_user, get_user_report_dir
from src.analysis.sentiment.dashboard import SentimentDashboard

router = APIRouter(prefix="/api/sentiment", tags=["Sentiment"])


@router.post("/analyze")
async def analyze_sentiment(current_user: User = Depends(get_current_user)):
    """Run sentiment analysis and generate report."""
    try:
        dashboard = SentimentDashboard()
        report = await asyncio.to_thread(dashboard.run_analysis)

        user_report_dir = get_user_report_dir(current_user.id)
        sentiment_dir = os.path.join(user_report_dir, "sentiment")
        if not os.path.exists(sentiment_dir):
            os.makedirs(sentiment_dir)

        filename = f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(sentiment_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        return {"report": report, "filename": filename}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports")
async def list_sentiment_reports(current_user: User = Depends(get_current_user)):
    """List all sentiment analysis reports."""
    user_report_dir = get_user_report_dir(current_user.id)
    sentiment_dir = os.path.join(user_report_dir, "sentiment")

    if not os.path.exists(sentiment_dir):
        return []

    reports = []
    files = glob.glob(os.path.join(sentiment_dir, "sentiment_*.md"))
    files.sort(key=os.path.getmtime, reverse=True)

    for f in files:
        filename = os.path.basename(f)
        try:
            parts = filename.replace(".md", "").split("_")
            if len(parts) >= 3:
                date_str = parts[1]
                time_str = parts[2]
                formatted_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}"
            elif len(parts) == 2:
                date_str = parts[1]
                formatted_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            else:
                continue

            reports.append({
                "filename": filename,
                "date": formatted_time
            })
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    return reports


@router.delete("/reports/{filename}")
async def delete_sentiment_report(filename: str, current_user: User = Depends(get_current_user)):
    """Delete a sentiment report."""
    try:
        if not filename.endswith(".md") or ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        user_report_dir = get_user_report_dir(current_user.id)
        sentiment_dir = os.path.join(user_report_dir, "sentiment")
        file_path = os.path.join(sentiment_dir, filename)

        if os.path.exists(file_path):
            os.remove(file_path)
            return {"status": "success", "message": f"Deleted {filename}"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
