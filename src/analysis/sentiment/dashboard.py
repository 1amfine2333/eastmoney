import sys
import os
from datetime import datetime

# Add project root to sys.path if run directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.analysis.sentiment.market_cycle import MarketCycleAnalyst
from src.analysis.sentiment.news_mining import NewsMiner
from src.analysis.sentiment.social_media import SocialSentinel
from src.analysis.sentiment.money_flow import MoneyFlowAnalyst
from src.llm.client import get_llm_client

class SentimentDashboard:
    def __init__(self):
        self.llm = get_llm_client()
        self.cycle_analyst = MarketCycleAnalyst()
        self.news_miner = NewsMiner()
        self.social_sentinel = SocialSentinel()
        self.money_analyst = MoneyFlowAnalyst()

    def run_analysis(self):
        print("1. Analyzing Market Cycle...")
        cycle_data = self.cycle_analyst.get_cycle_metrics()
        cycle_phase = self.cycle_analyst.determine_cycle_phase(cycle_data)
        
        print("2. Mining News...")
        news_items = self.news_miner.fetch_recent_news(limit=15)
        news_report = self.news_miner.analyze_news_sentiment(news_items)
        
        print("3. Checking Social Sentiment...")
        social_data = self.social_sentinel.get_social_sentiment()
        
        print("4. Tracking Money Flow...")
        money_data = self.money_analyst.get_money_flow()
        
        print("5. Generating Final Dashboard Report...")
        final_report = self.generate_final_report(cycle_data, cycle_phase, news_report, social_data, money_data)
        
        return final_report

    def generate_final_report(self, cycle, phase, news_analysis, social, money):
        def _fmt_list(items, limit: int = 5, format_str: str = "{i}. {name}") -> str:
            if not items:
                return "(无数据)"
            lines = []
            for i, item in enumerate(items[:limit], 1):
                if isinstance(item, dict):
                    # Smart format based on available keys
                    name = item.get("name") or item.get("股票名称") or item.get("title") or "N/A"
                    code = item.get("code") or item.get("股票代码")
                    pct = item.get("pct") or item.get("pct_change")
                    net_in = item.get("net_in") # Inflow
                    net_out = item.get("net_out") # Outflow
                    net_buy = item.get("net_buy")
                    turnover = item.get("turnover") # ETF volume
                    
                    parts = [f"{i}. {name}"]
                    if code: parts.append(f"({code})")
                    if pct is not None: parts.append(f"涨跌:{pct}%")
                    if net_in is not None: parts.append(f"主力净流入:{net_in}")
                    if net_out is not None: parts.append(f"主力净流出:{net_out}")
                    if net_buy is not None: parts.append(f"净买:{net_buy}亿")
                    if turnover is not None: 
                        # Format turnover to Yi
                        try:
                             parts.append(f"成交:{round(float(turnover)/1e8, 2)}亿")
                        except:
                             parts.append(f"成交:{turnover}")
                    
                    lines.append(" ".join(parts))
                else:
                    lines.append(f"{i}. {item}")
            return "\n".join(lines)

        report_date = datetime.now().strftime('%Y-%m-%d')
        
        # --- Data Unpacking ---
        # Cycle
        zt_count = cycle.get('zt_count', 0)
        zb_count = cycle.get('zb_count', 0)
        seal_rate = cycle.get('seal_rate', 0)
        market_height = cycle.get('market_height', 0)
        
        # Breadth
        breadth = money.get('market_breadth', {})
        up_count = breadth.get('up', 0)
        down_count = breadth.get('down', 0)
        limit_up_real = breadth.get('limit_up', 0)
        limit_down_real = breadth.get('limit_down', 0)
        
        # Money Flow
        sector_inflow = money.get('sector_inflow', [])
        sector_outflow = money.get('sector_outflow', [])
        etf_active = money.get('etf_active', [])
        north_money = money.get('north_money', 0)
        
        # Social
        top_hot = (social or {}).get('top_hot', [])
        
        prompt = f"""
        【角色设定】
        你是一位在华尔街和陆家嘴都有丰富经验的【首席量化策略师】。你的风格是“用数据说话”，厌恶主观臆测。
        请结合以下多维度的实时量化数据，为专业投资者撰写一份【A股市场全景深度复盘】。

        【核心原则】
        1. **数据驱动**：所有观点必须有下方提供的具体数据支持，禁止编造。
        2. **逻辑闭环**：不仅要说“涨了”，还要分析“是谁买起来的”（游资点火 vs 机构配置）。
        3. **关注机构**：重点解读ETF动向和行业资金流向，这是中期行情的风向标。

        ---
        【📊 市场量化全景】
        1. **市场温度计**:
           - 涨跌家数: {up_count}家上涨 / {down_count}家下跌 (涨跌比: {round(up_count/(down_count+1), 2)})
           - 涨停/跌停: {limit_up_real}家涨停 / {limit_down_real}家跌停
           - 情绪周期阶段: {phase} (涨停{zt_count}家, 炸板率{100-seal_rate if seal_rate else 0}%)
        
        2. **💸 聪明钱去哪了 (Smart Money)**
           - **北向资金**: 净流入 {north_money} 亿元
           - **ETF 战场 (机构风向标)**: 成交最活跃的宽基/行业ETF:
             {_fmt_list(etf_active, 5)}
        
        3. **🌊 板块资金流向 (Real-time Flow)**
           - **🚀 主力加仓榜 (净流入Top5)**:
             {_fmt_list(sector_inflow, 5)}
           - **📉 主力出逃榜 (净流出Top5)**:
             {_fmt_list(sector_outflow, 5)}
             
        4. **🔥 散户情绪 (Counter-Indicator)**
           - 社区人气榜 (警惕高位一致):
             {_fmt_list(top_hot, 5)}
             
        5. **📰 消息面驱动**
           {news_analysis}

        ---
        【写作要求】
        请输出 Markdown 格式，包含以下模块：

        # 📊 A股深度资金复盘 ({report_date})

        ## 1. 核心综述 (Market Pulse)
        - **一句话定性**: (例如：机构进场，指数搭台 / 游资退潮，亏钱效应弥漫)
        - **数据透视**: 引用“涨跌家数比”和“北向/ETF”数据，通过数据对比论证当前是属于“普涨”、“结构性行情”还是“泥沙俱下”。

        ## 2. 资金流向解码 (Follow the Money)
        - **谁在买入?**: 重点分析【主力加仓榜】和【ETF活跃榜】。哪些板块获得了真金白银的流入？这暗示了什么中期逻辑（是防御还是进攻）？
        - **谁在抛售?**: 分析【主力出逃榜】，指出哪些板块正在面临获利了结或机构调仓压力。
        - **风格研判**: 市场风格是偏向“大盘蓝筹”（参考ETF和北向）还是“小盘题材”（参考涨停数和连板高度）？

        ## 3. 情绪与博弈 (Sentiment & Game)
        - **周期位置**: 基于情绪周期阶段（{phase}），判断当前是应该激进做多还是防守。
        - **拥挤度分析**: 结合【社区人气榜】，指出哪些热门股/板块可能过于拥挤，需要警惕冲高回落。

        ## 4. 策略展望 (Action Plan)
        - **下一个交易日剧本**: 预测明天资金可能回流的方向（基于今天的流入逻辑延续或超跌反弹）。
        - **重点关注**: 给出2-3个值得跟踪的**细分方向**（基于资金流入坚决的板块），并提示具体的观察指标（如：成交量是否持续放大）。

        (注意：保持专业、客观、冷静的语调。不要使用夸张的感叹号。)
        """
        
        return self.llm.generate_content(prompt)

if __name__ == "__main__":
    dashboard = SentimentDashboard()
    report = dashboard.run_analysis()
    
    # Ensure reports dir exists
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/sentiment_{datetime.now().strftime('%Y%m%d')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {os.path.abspath(filename)}")
