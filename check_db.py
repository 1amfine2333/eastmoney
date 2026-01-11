import sqlite3
import os

DB_PATH = "funds.db"
if os.path.exists("data/funds.db"):
    DB_PATH = "data/funds.db"

print(f"Checking DB at: {DB_PATH}")

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", [t[0] for t in tables])
    
    if 'users' in [t[0] for t in tables]:
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        print("Users columns:", [c[1] for c in columns])
    else:
        print("ERROR: 'users' table missing!")
        
    conn.close()
except Exception as e:
    print(e)
