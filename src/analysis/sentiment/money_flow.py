import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

class MoneyFlowAnalyst:
    def get_money_flow(self):
        """
        获取全方位的资金流向数据
        """
        data = {
            "north_money": 0.0, # 北向资金
            "institution_buy": [], # 机构龙虎榜
            "sector_inflow": [], # 行业净流入Top
            "sector_outflow": [], # 行业净流出Top
            "etf_active": [], # 活跃ETF
            "market_breadth": {}, # 涨跌家数
            "north_date": None,
            "institution_date": None,
        }
        
        # 1. 市场广度 (Market Breadth)
        try:
            # 使用实时行情概览

            legu_df = ak.stock_market_activity_legu()
            if not legu_df.empty:
                # Convert to dict {item: value}
                legu_map = dict(zip(legu_df['item'], legu_df['value']))
                up_count = int(float(legu_map.get("上涨", 0)))
                down_count = int(float(legu_map.get("下跌", 0)))
                data["market_breadth"] = {
                    "up": up_count,
                    "down": down_count,
                    "flat": int(float(legu_map.get("平盘", 0))),
                    "limit_up": int(float(legu_map.get("涨停", 0))),
                    "limit_down": int(float(legu_map.get("跌停", 0))),
                    "ratio": round(up_count / (up_count + down_count + 1) * 100, 1)
                }

            # df_spot = ak.stock_zh_a_spot_em()
            # if df_spot is not None and not df_spot.empty:
            #     up_count = len(df_spot[df_spot['涨跌幅'] > 0])
            #     down_count = len(df_spot[df_spot['涨跌幅'] < 0])
            #     flat_count = len(df_spot[df_spot['涨跌幅'] == 0])
            #     limit_up = len(df_spot[df_spot['涨跌幅'] >= 9.8]) # 粗略统计
            #     limit_down = len(df_spot[df_spot['涨跌幅'] <= -9.8])
                
            #     data["market_breadth"] = {
            #         "up": up_count,
            #         "down": down_count,
            #         "flat": flat_count,
            #         "limit_up": limit_up,
            #         "limit_down": limit_down,
            #         "ratio": round(up_count / (up_count + down_count + 1) * 100, 1)
            #     }
        except Exception as e:
            print(f"Market breadth error: {e}")

        # 2. 行业板块资金流向 (Sector Flow)
        try:
            # 获取今日行业资金流排名
            df_flow = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="行业资金流")
            if df_flow is not None and not df_flow.empty:
                # 东方财富接口返回列名可能包含：名称, 今日涨跌幅, 今日主力净流入-净额, ...
                
                # 动态查找列名
                flow_col = '今日主力净流入' # Default
                pct_col = '今日涨跌幅' # Default
                name_col = '名称'

                for col in df_flow.columns:
                    if "主力净流入" in col and "净额" in col:
                        flow_col = col
                    elif "涨跌幅" in col and "今日" in col:
                        pct_col = col
                
                if flow_col in df_flow.columns:
                    # 转换净流入为数值 (单位通常是元，转换为亿元)
                    def parse_flow(val):
                        try:
                            if pd.isna(val): return 0.0
                            if isinstance(val, (int, float)):
                                return float(val) / 1e8 # 原始单位是元
                            if isinstance(val, str):
                                val = val.replace('亿', '').replace('万', '')
                                return float(val)
                        except:
                            return 0.0
                        return 0.0

                    # Inflow Top 5
                    # Ensure numeric
                    if df_flow[flow_col].dtype == object:
                         # Try to clean if it's string (though recent akshare returns float)
                         pass

                    # Sort just in case API didn't sort by flow
                    df_flow[flow_col] = pd.to_numeric(df_flow[flow_col], errors='coerce').fillna(0)
                    
                    df_sorted = df_flow.sort_values(by=flow_col, ascending=False)
                    
                    # Top 5 Inflow
                    top_in = df_sorted.head(5)
                    inflow_list = []
                    for _, row in top_in.iterrows():
                        inflow_list.append({
                            "name": row.get(name_col),
                            "pct": row.get(pct_col),
                            "net_in": parse_flow(row.get(flow_col))
                        })
                    data["sector_inflow"] = inflow_list

                    # Top 5 Outflow (Bottom 5 of sorted)
                    top_out = df_sorted.tail(5).sort_values(by=flow_col, ascending=True)
                    outflow_list = []
                    for _, row in top_out.iterrows():
                        outflow_list.append({
                            "name": row.get(name_col),
                            "pct": row.get(pct_col),
                            "net_out": parse_flow(row.get(flow_col))
                        })
                    data["sector_outflow"] = outflow_list
                    
        except Exception as e:
            print(f"Sector flow error: {e}")

        # 3. 热门ETF成交 (ETF Activity)
        try:
            # 获取ETF实时行情，按成交额排序
            df_etf = ak.fund_etf_spot_em()
            if df_etf is not None and not df_etf.empty:
                # 按成交额降序
                df_etf = df_etf.sort_values(by='成交额', ascending=False).head(5)
                etf_list = []
                for _, row in df_etf.iterrows():
                    etf_list.append({
                        "code": row.get('代码'),
                        "name": row.get('名称'),
                        "pct": row.get('涨跌幅'),
                        "turnover": row.get('成交额') # 元
                    })
                data["etf_active"] = etf_list
        except Exception as e:
            print(f"ETF data error: {e}")

        # 4. ???? (TuShare, previous trading day)
        try:
            from src.data_sources import tushare_client

            trade_date = tushare_client.get_latest_trade_date(max_days_back=20)
            if trade_date:
                end_date = trade_date
                start_dt = datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=10)
                start_date = start_dt.strftime('%Y%m%d')

                df_north = tushare_client.get_moneyflow_hsgt(
                    start_date=start_date,
                    end_date=end_date
                )

                if df_north is not None and not df_north.empty:
                    df_north = df_north.sort_values('trade_date', ascending=False)
                    latest = df_north.iloc[0]

                    def _to_yi(val) -> float:
                        try:
                            v = pd.to_numeric(val, errors='coerce')
                            if pd.isna(v):
                                return 0.0
                            # TuShare moneyflow_hsgt is in million (??), convert to ?
                            return round(float(v) / 100, 2)
                        except Exception:
                            return 0.0

                    data["north_money"] = _to_yi(latest.get('north_money'))
                    data["north_date"] = str(latest.get('trade_date'))
            else:
                print("North money error: could not determine latest trade date")

        except Exception as e:
            print(f"North money error: {e}")
        # 5. 机构龙虎榜 (现有逻辑，保留)
        try:
            df_jg = ak.stock_lhb_jgmmtj_em()
            if df_jg is not None and not df_jg.empty:
                date_col = "上榜日期" if "上榜日期" in df_jg.columns else None
                if date_col:
                    df_jg[date_col] = pd.to_datetime(df_jg[date_col], errors="coerce")
                    latest_date = df_jg[date_col].max()
                    data["institution_date"] = latest_date.strftime("%Y-%m-%d")
                    df_jg = df_jg[df_jg[date_col] == latest_date]

                net_buy_col = "净买入额" # Standardize
                for col in df_jg.columns:
                    if "净" in col and "额" in col:
                        net_buy_col = col
                        break
                
                name_col = "名称" if "名称" in df_jg.columns else "股票名称"
                
                if net_buy_col and name_col in df_jg.columns:
                     # Sort by abs value of net buy to see big moves (buy or sell)
                     # Or just top buys
                     df_jg[net_buy_col] = pd.to_numeric(df_jg[net_buy_col], errors="coerce")
                     top_buy = df_jg.sort_values(by=net_buy_col, ascending=False).head(5)
                     
                     res = []
                     for _, row in top_buy.iterrows():
                         val = row.get(net_buy_col)
                         res.append({
                             "name": row.get(name_col),
                             "net_buy": round(val / 1e4, 0) if val else 0 # Show in Wan
                         })
                     data["institution_buy"] = res
        except Exception as e:
            print(f"Institution error: {e}")
            
        return data

if __name__ == "__main__":
    analyst = MoneyFlowAnalyst()
    print(analyst.get_money_flow())
