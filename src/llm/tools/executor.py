"""
Tool Executor for AI Assistant Function Calling

Executes tool calls by routing to existing services.
Handles errors gracefully and returns structured results.
"""

import json
import asyncio
from typing import Dict, Any, Callable, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


class ToolExecutor:
    """
    Executes tool calls and returns results.

    Features:
    - Routes tool calls to appropriate services
    - Timeout handling (10 seconds per tool)
    - Error handling with friendly messages
    - Result formatting for LLM consumption
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._timeout = 10  # seconds
        self._executor = ThreadPoolExecutor(max_workers=5)
        self._register_handlers()

    def _register_handlers(self):
        """Register all tool handlers."""
        self._handlers = {
            "get_stock_quote": self._get_stock_quote,
            "get_stock_history": self._get_stock_history,
            "get_market_indices": self._get_market_indices,
            "get_northbound_flow": self._get_northbound_flow,
            "get_industry_flow": self._get_industry_flow,
            "get_main_capital_flow": self._get_main_capital_flow,
            "get_sector_performance": self._get_sector_performance,
            "get_top_list": self._get_top_list,
            "get_forex_rates": self._get_forex_rates,
            "get_stock_news": self._get_stock_news,
            "get_hot_news": self._get_hot_news,
            "search_research_reports": self._search_research_reports,
            "get_fund_info": self._get_fund_info,
            "get_fund_holdings": self._get_fund_holdings,
            "search_funds": self._search_funds,
            "run_portfolio_stress_test": self._run_portfolio_stress_test,
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Dict with success status and data or error message
        """
        if tool_name not in self._handlers:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "data": None
            }

        handler = self._handlers[tool_name]

        try:
            # Execute with timeout
            future = self._executor.submit(handler, arguments)
            result = future.result(timeout=self._timeout)

            return {
                "success": True,
                "data": result,
                "error": None,
                "executed_at": datetime.now().isoformat()
            }

        except FuturesTimeoutError:
            return {
                "success": False,
                "error": f"Tool {tool_name} execution timed out after {self._timeout}s",
                "data": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "data": None
            }

    def execute_multiple(self, tool_calls: list) -> list:
        """
        Execute multiple tool calls.

        Args:
            tool_calls: List of (tool_name, arguments) tuples

        Returns:
            List of execution results
        """
        results = []
        for tool_name, arguments in tool_calls:
            result = self.execute(tool_name, arguments)
            result["tool_name"] = tool_name
            results.append(result)
        return results

    # =========================================================================
    # Tool Handler Implementations
    # =========================================================================

    def _get_stock_quote(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time stock quote."""
        from src.data_sources.akshare_api import get_stock_realtime_quote

        stock_code = args.get("stock_code", "")
        if not stock_code:
            return {"error": "Missing stock_code parameter"}

        # Normalize code (remove prefix if any)
        stock_code = stock_code.replace("SH", "").replace("SZ", "").replace("sh", "").replace("sz", "")

        quote = get_stock_realtime_quote(stock_code)

        if not quote or "error" in quote:
            return {"error": f"Unable to fetch quote for {stock_code}"}

        return {
            "stock_code": stock_code,
            "name": quote.get("name", ""),
            "price": quote.get("close") or quote.get("price"),
            "change": quote.get("change"),
            "change_pct": quote.get("change_pct") or quote.get("pct_change"),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "volume": quote.get("volume"),
            "amount": quote.get("amount"),
            "pe": quote.get("pe"),
            "pb": quote.get("pb"),
        }

    def _get_stock_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get stock historical data."""
        from src.data_sources.akshare_api import get_stock_history

        stock_code = args.get("stock_code", "")
        days = args.get("days", 30)

        if not stock_code:
            return {"error": "Missing stock_code parameter"}

        stock_code = stock_code.replace("SH", "").replace("SZ", "").replace("sh", "").replace("sz", "")

        history = get_stock_history(stock_code, days=days)

        if not history:
            return {"error": f"Unable to fetch history for {stock_code}"}

        # Summarize for LLM
        if len(history) > 0:
            first = history[0]
            last = history[-1]
            prices = [h.get("value") or h.get("close") for h in history if h.get("value") or h.get("close")]

            return {
                "stock_code": stock_code,
                "days": len(history),
                "start_date": first.get("date"),
                "end_date": last.get("date"),
                "start_price": first.get("value") or first.get("close"),
                "end_price": last.get("value") or last.get("close"),
                "high": max(prices) if prices else None,
                "low": min(prices) if prices else None,
                "recent_data": history[-5:] if len(history) >= 5 else history  # Last 5 days
            }

        return {"error": "No historical data available"}

    def _get_market_indices(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get market indices data."""
        from src.data_sources.akshare_api import get_market_indices

        indices = get_market_indices()

        if not indices or "error" in indices:
            return {"error": "Unable to fetch market indices"}

        return {
            "indices": indices,
            "summary": self._format_indices_summary(indices)
        }

    def _format_indices_summary(self, indices: Dict) -> str:
        """Format indices data as readable summary."""
        parts = []
        for key, data in indices.items():
            if isinstance(data, dict) and "close" in data:
                name = data.get("name", key)
                close = data.get("close", 0)
                change_pct = data.get("pct_change", 0)
                direction = "涨" if change_pct >= 0 else "跌"
                parts.append(f"{name}: {close:.2f} ({direction}{abs(change_pct):.2f}%)")
        return "; ".join(parts)

    def _get_northbound_flow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get northbound capital flow data."""
        from src.analysis.widget_service import widget_service

        days = args.get("days", 5)
        flow = widget_service.get_northbound_flow(days=days)

        if "error" in flow and flow.get("error"):
            return {"error": flow["error"]}

        latest = flow.get("latest", {})
        cumulative = flow.get("cumulative_5d", 0)
        history = flow.get("history", [])

        return {
            "today": {
                "date": latest.get("date"),
                "net_flow": latest.get("north_money"),
                "unit": "亿元"
            },
            "cumulative_5d": cumulative,
            "history": history,
            "summary": f"今日北向资金净{'流入' if latest.get('north_money', 0) >= 0 else '流出'}{abs(latest.get('north_money', 0)):.2f}亿元，近5日累计{'流入' if cumulative >= 0 else '流出'}{abs(cumulative):.2f}亿元"
        }

    def _get_industry_flow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get industry capital flow data."""
        from src.analysis.widget_service import widget_service

        limit = args.get("limit", 10)
        flow = widget_service.get_industry_flow(limit=limit)

        if "error" in flow and flow.get("error"):
            return {"error": flow["error"]}

        return {
            "trade_date": flow.get("trade_date"),
            "top_inflow": flow.get("gainers", [])[:5],
            "top_outflow": flow.get("losers", [])[:5],
            "unit": "亿元"
        }

    def _get_main_capital_flow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get main capital flow ranking."""
        from src.analysis.widget_service import widget_service

        limit = args.get("limit", 10)
        flow = widget_service.get_main_capital_flow(limit=limit)

        if "error" in flow and flow.get("error"):
            return {"error": flow["error"]}

        return {
            "top_flows": flow.get("top_flows", []),
            "market_overview": flow.get("market_overview", {}),
            "unit": "亿元"
        }

    def _get_sector_performance(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get sector performance data."""
        from src.analysis.widget_service import widget_service

        limit = args.get("limit", 10)
        perf = widget_service.get_sector_performance(limit=limit)

        if "error" in perf and perf.get("error"):
            return {"error": perf["error"]}

        return {
            "trade_date": perf.get("trade_date"),
            "top_gainers": perf.get("gainers", [])[:5],
            "top_losers": perf.get("losers", [])[:5]
        }

    def _get_top_list(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get dragon tiger list data."""
        from src.analysis.widget_service import widget_service

        limit = args.get("limit", 10)
        top_list = widget_service.get_top_list(limit=limit)

        if "error" in top_list and top_list.get("error"):
            return {"error": top_list["error"]}

        return {
            "trade_date": top_list.get("trade_date"),
            "stocks": top_list.get("data", [])
        }

    def _get_forex_rates(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get forex rates."""
        from src.analysis.widget_service import widget_service

        rates = widget_service.get_forex_rates()

        if "error" in rates and rates.get("error"):
            return {"error": rates["error"]}

        return {
            "rates": rates.get("rates", []),
            "updated_at": rates.get("updated_at")
        }

    def _get_stock_news(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get stock-related news."""
        from src.services.news_service import news_service

        stock_code = args.get("stock_code", "")
        limit = args.get("limit", 10)

        if not stock_code:
            return {"error": "Missing stock_code parameter"}

        stock_code = stock_code.replace("SH", "").replace("SZ", "").replace("sh", "").replace("sz", "")

        news = news_service.get_stock_news([stock_code], limit=limit)

        if not news:
            return {"news": [], "message": f"No news found for stock {stock_code}"}

        return {
            "stock_code": stock_code,
            "news": [
                {
                    "title": item.get("title", ""),
                    "summary": (item.get("content", "") or "")[:200],
                    "source": item.get("source_name", ""),
                    "published_at": item.get("published_at", ""),
                    "url": item.get("url", "")
                }
                for item in news[:limit]
            ]
        }

    def _get_hot_news(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get hot/trending news."""
        from src.services.news_service import news_service

        limit = args.get("limit", 10)
        news = news_service.get_hot_news(limit=limit)

        if not news:
            return {"news": [], "message": "No hot news available"}

        return {
            "news": [
                {
                    "title": item.get("title", ""),
                    "summary": (item.get("content", "") or "")[:200],
                    "source": item.get("source_name", ""),
                    "published_at": item.get("published_at", "")
                }
                for item in news[:limit]
            ]
        }

    def _search_research_reports(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search research reports."""
        from src.services.news_service import news_service

        query = args.get("query", "")
        limit = args.get("limit", 5)

        if not query:
            return {"error": "Missing query parameter"}

        reports = news_service.search_research_reports(query, limit=limit)

        if not reports:
            return {"reports": [], "message": f"No research reports found for '{query}'"}

        return {
            "query": query,
            "reports": [
                {
                    "title": item.get("title", ""),
                    "summary": (item.get("content", "") or "")[:300],
                    "source": item.get("source_name", ""),
                    "url": item.get("url", "")
                }
                for item in reports[:limit]
            ]
        }

    def _get_fund_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get fund information and NAV."""
        import akshare as ak
        import pandas as pd

        fund_code = args.get("fund_code", "")
        if not fund_code:
            return {"error": "Missing fund_code parameter"}

        # Normalize code
        fund_code = fund_code.strip()

        try:
            # Get basic info
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")

            if df is None or df.empty:
                return {"error": f"No NAV data found for fund {fund_code}"}

            # Normalize column names
            cols = list(df.columns)
            if len(cols) >= 3:
                df = df.rename(columns={
                    cols[0]: 'date',
                    cols[1]: 'nav',
                    cols[2]: 'day_change',
                })

            # Sort by date descending
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.sort_values('date', ascending=False).reset_index(drop=True)

            # Get latest NAV
            latest = df.iloc[0] if len(df) > 0 else None

            # Try to get fund name from another API
            fund_name = ""
            try:
                name_df = ak.fund_name_em()
                if name_df is not None and not name_df.empty:
                    match = name_df[name_df['基金代码'] == fund_code]
                    if not match.empty:
                        fund_name = match.iloc[0].get('基金简称', '')
            except:
                pass

            # Get performance data
            perf = {}
            try:
                perf_indicators = ["近1周", "近1月", "近3月", "近6月", "近1年"]
                for indicator in perf_indicators:
                    try:
                        perf_df = ak.fund_open_fund_info_em(symbol=fund_code, indicator=indicator)
                        if perf_df is not None and not perf_df.empty:
                            perf[indicator] = perf_df.iloc[0, 1] if len(perf_df.columns) > 1 else None
                    except:
                        pass
            except:
                pass

            # Build NAV history (last 10 days)
            nav_history = []
            for _, row in df.head(10).iterrows():
                nav_history.append({
                    "date": row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else "",
                    "nav": float(row['nav']) if pd.notna(row['nav']) else None,
                    "day_change": float(row['day_change']) if pd.notna(row.get('day_change')) else None
                })

            return {
                "fund_code": fund_code,
                "name": fund_name,
                "nav": float(latest['nav']) if latest is not None and pd.notna(latest['nav']) else None,
                "nav_date": latest['date'].strftime('%Y-%m-%d') if latest is not None and pd.notna(latest['date']) else None,
                "day_change": float(latest['day_change']) if latest is not None and pd.notna(latest.get('day_change')) else None,
                "week_change": perf.get("近1周"),
                "month_change": perf.get("近1月"),
                "three_month_change": perf.get("近3月"),
                "six_month_change": perf.get("近6月"),
                "year_change": perf.get("近1年"),
                "nav_history": nav_history
            }

        except Exception as e:
            return {"error": f"Failed to fetch fund info: {str(e)}"}

    def _get_fund_holdings(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get fund top holdings."""
        import akshare as ak
        import pandas as pd
        from datetime import datetime

        fund_code = args.get("fund_code", "")
        if not fund_code:
            return {"error": "Missing fund_code parameter"}

        try:
            year = str(datetime.now().year)
            df = ak.fund_portfolio_hold_em(symbol=fund_code, date=year)

            if df is None or df.empty:
                # Try previous year
                year = str(int(year) - 1)
                df = ak.fund_portfolio_hold_em(symbol=fund_code, date=year)

            if df is None or df.empty:
                return {"error": f"No holdings data found for fund {fund_code}"}

            # Convert to list of dicts
            holdings = []
            for _, row in df.head(10).iterrows():
                holdings.append({
                    "stock_code": row.get('股票代码', ''),
                    "stock_name": row.get('股票名称', ''),
                    "weight": row.get('占净值比例', ''),
                    "shares": row.get('持股数', ''),
                    "value": row.get('持仓市值', '')
                })

            return {
                "fund_code": fund_code,
                "report_year": year,
                "holdings": holdings
            }

        except Exception as e:
            return {"error": f"Failed to fetch fund holdings: {str(e)}"}

    def _search_funds(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search funds by name or code."""
        from src.data_sources.akshare_api import search_funds

        query = args.get("query", "")
        limit = args.get("limit", 10)

        if not query:
            return {"error": "Missing query parameter"}

        results = search_funds(query, limit=limit)

        if not results:
            return {"funds": [], "message": f"No funds found for '{query}'"}

        return {
            "query": query,
            "funds": [
                {
                    "code": f.get("code", ""),
                    "name": f.get("name", ""),
                    "type": f.get("type", "")
                }
                for f in results[:limit]
            ]
        }

    def _run_portfolio_stress_test(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run stress test on a portfolio with given scenario parameters."""
        from src.analysis.portfolio.stress_test import StressTestEngine, StressScenario
        from src.storage.db import get_portfolio_positions

        portfolio_id = args.get("portfolio_id")
        if not portfolio_id:
            return {"error": "Missing portfolio_id parameter"}

        try:
            # Get portfolio positions
            # Note: This is a simplified version that works for the tool context
            # In production, user_id would come from the authenticated context
            positions = get_portfolio_positions(portfolio_id, user_id=1)

            if not positions:
                return {"error": "No positions found in portfolio or portfolio not found"}

            # Build scenario from parameters
            scenario = StressScenario(
                interest_rate_change_bp=args.get("interest_rate_change_bp", 0),
                fx_change_pct=args.get("fx_change_pct", 0),
                index_change_pct=args.get("index_change_pct", 0),
                oil_change_pct=args.get("oil_change_pct", 0)
            )

            # Get current prices - simplified approach using average cost as fallback
            current_prices = {}
            for pos in positions:
                code = pos.get('asset_code')
                price = pos.get('average_cost', 100)  # Fallback to average cost
                current_prices[code] = float(price) if price else 100.0

            # Run stress test
            engine = StressTestEngine()
            result = engine.run_stress_test(positions, scenario, current_prices)

            # Format result for LLM consumption
            return {
                "portfolio_id": portfolio_id,
                "scenario": {
                    "interest_rate_change_bp": scenario.interest_rate_change_bp,
                    "fx_change_pct": scenario.fx_change_pct,
                    "index_change_pct": scenario.index_change_pct,
                    "oil_change_pct": scenario.oil_change_pct
                },
                "projected_pnl_pct": result.get("projected_pnl_pct", 0),
                "var_95_pct": result.get("var_95_pct", 0),
                "risk_level": result.get("risk_level", "unknown"),
                "top_losers": result.get("top_losers", [])[:3],
                "top_gainers": result.get("top_gainers", [])[:3],
                "summary": f"预计{'亏损' if result.get('projected_pnl_pct', 0) < 0 else '盈利'}{abs(result.get('projected_pnl_pct', 0)):.2f}%，风险等级：{result.get('risk_level', '未知')}"
            }

        except Exception as e:
            return {"error": f"Stress test failed: {str(e)}"}


# Singleton instance
tool_executor = ToolExecutor()
