import sqlite3
import json
import os
from typing import List, Dict, Optional

# Define paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Allow overriding via environment variable for Docker volumes
DB_PATH = os.environ.get("DB_FILE_PATH", os.path.join(BASE_DIR, "funds.db"))
FUNDS_JSON_PATH = os.path.join(BASE_DIR, "config", "funds.json")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Create Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            hashed_password TEXT,
            provider TEXT DEFAULT 'local', -- local, google, github
            provider_id TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Create Funds Table (Original)
    c.execute('''
        CREATE TABLE IF NOT EXISTS funds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            style TEXT,
            focus TEXT,
            pre_market_time TEXT,
            post_market_time TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 3. Migration: Add user_id to funds if not exists
    # We want (user_id, code) to be unique, not just code globally. 
    # But SQLite limitations on ALTER TABLE are tricky. 
    # For now, we add the column if missing.
    try:
        c.execute('ALTER TABLE funds ADD COLUMN user_id INTEGER REFERENCES users(id)')
    except sqlite3.OperationalError:
        # Column likely exists
        pass
        
    conn.commit()
    conn.close()
    
    migrate_from_json_if_needed()

def migrate_from_json_if_needed():
    # Only runs if funds table is empty
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT count(*) FROM funds')
    count = cursor.fetchone()[0]
    
    if count == 0 and os.path.exists(FUNDS_JSON_PATH):
        print("Migrating funds.json to SQLite (Assigning to Admin/Null User)...")
        try:
            with open(FUNDS_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for fund in data:
                    cursor.execute('''
                        INSERT INTO funds (code, name, style, focus, pre_market_time, post_market_time, user_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        fund.get('code'),
                        fund.get('name'),
                        fund.get('style', ''),
                        json.dumps(fund.get('focus', []), ensure_ascii=False),
                        "08:30",
                        "15:30",
                        1 # Default to user 1 if migrating
                    ))
            conn.commit()
            print("Migration complete.")
        except Exception as e:
            print(f"Migration failed: {e}")
    
    conn.close()

# --- User Operations ---

def create_user(user_data: Dict) -> int:
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (username, email, hashed_password, provider, provider_id)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_data['username'],
            user_data.get('email'),
            user_data.get('hashed_password'),
            user_data.get('provider', 'local'),
            user_data.get('provider_id')
        ))
        user_id = c.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        raise ValueError("Username already exists")
    finally:
        conn.close()

def get_user_by_username(username: str) -> Optional[Dict]:
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id: int) -> Optional[Dict]:
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

# --- Fund Operations (Multi-tenant) ---

def _parse_focus(fund_row: Dict) -> Dict:
    d = dict(fund_row)
    if d.get('focus') and isinstance(d['focus'], str):
        try:
            d['focus'] = json.loads(d['focus'])
        except:
            d['focus'] = []
    return d

def get_all_funds(user_id: int = None) -> List[Dict]:
    conn = get_db_connection()
    if user_id:
        funds = conn.execute('SELECT * FROM funds WHERE user_id = ?', (user_id,)).fetchall()
    else:
        # Admin or Scheduler context: fetch all
        funds = conn.execute('SELECT * FROM funds').fetchall()
    conn.close()
    return [_parse_focus(f) for f in funds]

def get_active_funds(user_id: int = None) -> List[Dict]:
    conn = get_db_connection()
    sql = 'SELECT * FROM funds WHERE is_active = 1'
    params = []
    if user_id:
        sql += ' AND user_id = ?'
        params.append(user_id)
        
    funds = conn.execute(sql, tuple(params)).fetchall()
    conn.close()
    return [_parse_focus(f) for f in funds]

def get_fund_by_code(code: str, user_id: int = None) -> Optional[Dict]:
    # Note: Code might not be unique globally anymore if different users can watch same fund?
    # For now, let's assume users can have same funds. So we MUST filter by user_id if provided.
    conn = get_db_connection()
    sql = 'SELECT * FROM funds WHERE code = ?'
    params = [code]
    if user_id:
        sql += ' AND user_id = ?'
        params.append(user_id)
        
    fund = conn.execute(sql, tuple(params)).fetchone()
    conn.close()
    return _parse_focus(fund) if fund else None

def upsert_fund(fund_data: Dict, user_id: int):
    """
    Insert or Update a fund for a specific user.
    """
    if not user_id:
        raise ValueError("user_id is required for upserting funds")
        
    conn = get_db_connection()
    c = conn.cursor()
    
    focus_json = json.dumps(fund_data.get('focus', []), ensure_ascii=False)
    
    # Check if exists for THIS user
    exists = c.execute('SELECT 1 FROM funds WHERE code = ? AND user_id = ?', 
                      (fund_data['code'], user_id)).fetchone()
    
    if exists:
        c.execute('''
            UPDATE funds 
            SET name=?, style=?, focus=?, pre_market_time=?, post_market_time=?, is_active=?
            WHERE code=? AND user_id=?
        ''', (
            fund_data['name'],
            fund_data.get('style', ''),
            focus_json,
            fund_data.get('pre_market_time'),
            fund_data.get('post_market_time'),
            fund_data.get('is_active', 1),
            fund_data['code'],
            user_id
        ))
    else:
        c.execute('''
            INSERT INTO funds (code, name, style, focus, pre_market_time, post_market_time, is_active, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fund_data['code'],
            fund_data['name'],
            fund_data.get('style', ''),
            focus_json,
            fund_data.get('pre_market_time'),
            fund_data.get('post_market_time'),
            fund_data.get('is_active', 1),
            user_id
        ))
    
    conn.commit()
    conn.close()

def delete_fund(code: str, user_id: int):
    if not user_id:
        raise ValueError("user_id required")
    conn = get_db_connection()
    conn.execute('DELETE FROM funds WHERE code = ? AND user_id = ?', (code, user_id))
    conn.commit()
    conn.close()