import sqlite3
import json
import os
from typing import List, Dict, Optional

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "funds.db")
FUNDS_JSON_PATH = os.path.join(BASE_DIR, "config", "funds.json")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS funds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            style TEXT,
            focus TEXT,
            pre_market_time TEXT,  -- Format "HH:MM", e.g., "08:30"
            post_market_time TEXT, -- Format "HH:MM", e.g., "15:30"
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    
    migrate_from_json_if_needed()

def migrate_from_json_if_needed():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT count(*) FROM funds')
    count = cursor.fetchone()[0]
    
    if count == 0 and os.path.exists(FUNDS_JSON_PATH):
        print("Migrating funds.json to SQLite...")
        try:
            with open(FUNDS_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for fund in data:
                    # Default times: Pre 08:30, Post 15:30
                    cursor.execute('''
                        INSERT OR IGNORE INTO funds (code, name, style, focus, pre_market_time, post_market_time)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        fund.get('code'),
                        fund.get('name'),
                        fund.get('style', ''),
                        json.dumps(fund.get('focus', []), ensure_ascii=False),
                        "08:30",
                        "15:30"
                    ))
            conn.commit()
            print("Migration complete.")
        except Exception as e:
            print(f"Migration failed: {e}")
    
    conn.close()

# CRUD Operations

def _parse_focus(fund_row: Dict) -> Dict:
    """Helper to parse JSON focus field."""
    d = dict(fund_row)
    if d.get('focus') and isinstance(d['focus'], str):
        try:
            d['focus'] = json.loads(d['focus'])
        except:
            d['focus'] = []
    return d

def get_all_funds() -> List[Dict]:
    conn = get_db_connection()
    funds = conn.execute('SELECT * FROM funds').fetchall()
    conn.close()
    return [_parse_focus(f) for f in funds]

def get_active_funds() -> List[Dict]:
    conn = get_db_connection()
    funds = conn.execute('SELECT * FROM funds WHERE is_active = 1').fetchall()
    conn.close()
    return [_parse_focus(f) for f in funds]

def get_fund_by_code(code: str) -> Optional[Dict]:
    conn = get_db_connection()
    fund = conn.execute('SELECT * FROM funds WHERE code = ?', (code,)).fetchone()
    conn.close()
    return _parse_focus(fund) if fund else None

def upsert_fund(fund_data: Dict):
    """
    Insert or Update a fund.
    fund_data should include: code, name, style, focus (list), pre_market_time, post_market_time, is_active
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    focus_json = json.dumps(fund_data.get('focus', []), ensure_ascii=False)
    
    # Check if exists
    exists = c.execute('SELECT 1 FROM funds WHERE code = ?', (fund_data['code'],)).fetchone()
    
    if exists:
        c.execute('''
            UPDATE funds 
            SET name=?, style=?, focus=?, pre_market_time=?, post_market_time=?, is_active=?
            WHERE code=?
        ''', (
            fund_data['name'],
            fund_data.get('style', ''),
            focus_json,
            fund_data.get('pre_market_time'),
            fund_data.get('post_market_time'),
            fund_data.get('is_active', 1),
            fund_data['code']
        ))
    else:
        c.execute('''
            INSERT INTO funds (code, name, style, focus, pre_market_time, post_market_time, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            fund_data['code'],
            fund_data['name'],
            fund_data.get('style', ''),
            focus_json,
            fund_data.get('pre_market_time'),
            fund_data.get('post_market_time'),
            fund_data.get('is_active', 1)
        ))
    
    conn.commit()
    conn.close()

def delete_fund(code: str):
    conn = get_db_connection()
    conn.execute('DELETE FROM funds WHERE code = ?', (code,))
    conn.commit()
    conn.close()
