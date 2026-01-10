from typing import Dict, Any
from .commodity import CommodityStrategy
from .equity import EquityStrategy

class StrategyFactory:
    @staticmethod
    def get_strategy(fund_info: Dict[str, Any], llm_client, web_search):
        """
        Returns the appropriate strategy instance based on fund characteristics.
        """
        name = fund_info.get("name", "")
        # Simple heuristic for now
        if any(k in name for k in ["黄金", "白银", "有色", "油", "石油", "贵金属", "商品"]):
            return CommodityStrategy(fund_info, llm_client, web_search)
        
        # Default to Equity
        return EquityStrategy(fund_info, llm_client, web_search)
