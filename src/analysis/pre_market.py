"""
Pre-Market Analyst - Strategy Driven
====================================
"""

import sys
import os
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.base_analyst import BaseAnalyst
from src.analysis.strategies.factory import StrategyFactory

class PreMarketAnalyst(BaseAnalyst):
    """
    Delegates analysis to specific strategies based on fund type.
    """
    
    SYSTEM_TITLE = "盘前情报系统启动"
    FAILURE_SUFFIX = "分析失败"

    def __init__(self):
        super().__init__()

    def analyze_fund(self, fund: dict) -> str:
        """
        Delegates the analysis to the appropriate strategy.
        """
        fund_name = fund.get("name")
        print(f"\n{'='*60}")
        print(f"🔍 分析基金: {fund_name} ({fund.get('code')})")
        print(f"{'='*60}")
        
        try:
            # 1. Get Strategy
            strategy = StrategyFactory.get_strategy(fund, self.llm, self.web_search)
            
            # 2. Collect Data
            data = strategy.collect_data(mode='pre')
            
            # 3. Generate Report
            report = strategy.generate_report(mode='pre', data=data)
            
            print("  ✅ 分析完成")
            return report
            
        except Exception as e:
            print(f"  ❌ Analysis Failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Analysis Failed: {str(e)}"

if __name__ == "__main__":
    analyst = PreMarketAnalyst()
    print(analyst.run_all())