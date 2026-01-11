import os
from datetime import datetime
from typing import Optional

def save_report(content: str, mode: str, fund_name: str = "Summary", fund_code: str = "", user_id: Optional[int] = None):
    """
    Save the report to a markdown file.
    If user_id is provided: reports/{user_id}/YYYY-MM-DD_{mode}_{fund_code}_{fund_name}.md
    Else: reports/YYYY-MM-DD_{mode}_{fund_code}_{fund_name}.md
    """
    today = datetime.now().strftime("%Y-%m-%d")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Determine directory
    if user_id:
        report_dir = os.path.join(base_dir, "reports", str(user_id))
    else:
        report_dir = os.path.join(base_dir, "reports")
    
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Sanitize filenames (Simple regex replacement for robustness)
    import re
    safe_name = re.sub(r'[^\w\s-]', '', fund_name).strip().replace(' ', '_')
    
    if fund_code:
        filename = f"{today}_{mode}_{fund_code}_{safe_name}.md"
    else:
        filename = f"{today}_{mode}_SUMMARY.md"
        
    filepath = os.path.join(report_dir, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Report saved to: {filepath}")
    return filepath