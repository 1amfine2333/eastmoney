from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import asyncio
from typing import Dict, Optional
from src.storage.db import get_active_funds, get_fund_by_code
from src.analysis.pre_market import PreMarketAnalyst
from src.analysis.post_market import PostMarketAnalyst
from src.report_gen import save_report

logger = logging.getLogger(__name__)

class SchedulerManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SchedulerManager, cls).__new__(cls)
            cls._instance.scheduler = BackgroundScheduler()
            cls._instance.scheduler.start()
        return cls._instance

    def start(self):
        """Load jobs from DB and start"""
        print("Starting Scheduler Manager...")
        self.refresh_all_jobs()
        
    def refresh_all_jobs(self):
        """Clear all and reload from DB (All users)"""
        self.scheduler.remove_all_jobs()
        # Fetch ALL active funds from ALL users
        funds = get_active_funds(user_id=None) 
        for fund in funds:
            self.add_fund_jobs(fund)

    def add_fund_jobs(self, fund: Dict):
        """Add Pre/Post market jobs for a single fund"""
        code = fund['code']
        # Ensure we have user_id, fallback to None (Admin/Legacy)
        user_id = fund.get('user_id') 
        
        # Pre-market
        if fund.get('pre_market_time'):
            try:
                hour, minute = fund['pre_market_time'].split(':')
                job_id = f"pre_{code}_{user_id}"
                self.scheduler.add_job(
                    self.run_analysis_task,
                    trigger=CronTrigger(hour=hour, minute=minute),
                    id=job_id,
                    args=[code, 'pre', user_id],
                    replace_existing=True
                )
                print(f"Scheduled PRE-market for {code} (User {user_id}) at {hour}:{minute}")
            except Exception as e:
                print(f"Error scheduling PRE task for {code}: {e}")

        # Post-market
        if fund.get('post_market_time'):
            try:
                hour, minute = fund['post_market_time'].split(':')
                job_id = f"post_{code}_{user_id}"
                self.scheduler.add_job(
                    self.run_analysis_task,
                    trigger=CronTrigger(hour=hour, minute=minute),
                    id=job_id,
                    args=[code, 'post', user_id],
                    replace_existing=True
                )
                print(f"Scheduled POST-market for {code} (User {user_id}) at {hour}:{minute}")
            except Exception as e:
                print(f"Error scheduling POST task for {code}: {e}")

    def remove_fund_jobs(self, code: str):
        """
        Remove jobs for a fund. 
        Note: This naive implementation removes jobs matching ID pattern.
        """
        # We need to find jobs starting with pre_{code}_ or post_{code}_
        # Iterate all jobs and match
        for job in self.scheduler.get_jobs():
            if job.id.startswith(f"pre_{code}_") or job.id.startswith(f"post_{code}_"):
                self.scheduler.remove_job(job.id)
                print(f"Removed job {job.id}")

    def run_analysis_task(self, fund_code: str, mode: str, user_id: Optional[int] = None):
        """Worker function"""
        print(f"Executing {mode.upper()}-market task for {fund_code} (User: {user_id})...")
        
        # Re-fetch fund data. Pass user_id if we want to be strict, or None to find by code globally.
        # But wait, code might not be unique globally anymore. We MUST filter by user_id if we have it.
        fund = get_fund_by_code(fund_code, user_id=user_id)
        
        if not fund or not fund.get('is_active'):
            print(f"Fund {fund_code} is inactive or deleted. Skipping.")
            return

        report = ""
        try:
            # Run analysis in thread to avoid blocking scheduler (though scheduler is threaded by default, good practice)
            # Actually apscheduler runs in thread/process pool executor.
            
            if mode == 'pre':
                analyst = PreMarketAnalyst()
                report = analyst.analyze_fund(fund)
            elif mode == 'post':
                analyst = PostMarketAnalyst()
                report = analyst.analyze_fund(fund)
            
            if report:
                save_report(report, mode, fund['name'], fund['code'], user_id=user_id)
                
        except Exception as e:
            logger.error(f"Task failed for {fund_code}: {e}")
            import traceback
            traceback.print_exc()

# Global instance
scheduler_manager = SchedulerManager()