import akshare as ak                                                                                              
import pandas as pd                                                                                               
                                                                                                          
try:                                                                                                              
    # This usually returns the minute-level flow for the current day                                              
    #df1 = ak.stock_hsgt_fund_min_em(symbol="北向资金")  
    #df = ak.fund_etf_hist_min_em(symbol="000690",start_date="2026-01-30 14:32:00", end_date="2026-01-30 15:00:00", period="1",adjust="hfq")
    df = ak.fund_individual_detail_hold_xq(symbol="000690",start_date="2026-01-30 14:32:00", end_date="2026-01-30 15:00:00", period="1",adjust="hfq")

    if df is not None and not df.empty:
        print("Columns:", df.columns.tolist())
        print("First 10 rows:")
        print(df.head(100))
        print("Last 10 rows:")
        print(df.tail(100))

        # Determine the latest net inflow                                                                         
        # Usually looking for something like '时间', '沪股通', '深股通', '北向资金'                               
        # Or maybe it has '净流入'                                                                                
                                                                                                                  
        # Calculate latest accumulated flow                                                                       
        last_row = df.iloc[-1]                                                                                    
        print("\nLast row data:", last_row.to_dict())                                                             
                                                                                                                  
    else:                                                                                                         
        print("DataFrame is empty or None")                                                                       
                                                                                                                  
except Exception as e:                                                                                            
    print(f"Error: {e}")