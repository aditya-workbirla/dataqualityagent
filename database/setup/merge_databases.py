import sqlite3
import os
import shutil

def merge():
    print("Beginning Database Merge sequence...")
    
    source_funcs = "database/functions.db"
    source_kb = "database/knowledge.db"
    source_cp = "database/checkpoints.db"
    target_db = "database/app.db"
    
    # Check if target exists, delete if it does to start fresh
    if os.path.exists(target_db):
        os.remove(target_db)
        
    print(f"Transferring primary engine logic from {source_cp} to {target_db}")
    if os.path.exists(source_cp):
        shutil.copyfile(source_cp, target_db)
    else:
        print("Warning: checkpoints.db didn't exist. LangGraph will auto-create tables later.")
        
    # Connect to new master DB
    conn_master = sqlite3.connect(target_db)
    cur_master = conn_master.cursor()
    
    # 1. Merge Functions Table
    if os.path.exists(source_funcs):
        print("Merging `data_quality_functions` table...")
        cur_master.execute('''
            CREATE TABLE IF NOT EXISTS data_quality_functions (
                function_id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT UNIQUE NOT NULL,
                function_code TEXT NOT NULL,
                function_description TEXT,
                approved_by_team BOOLEAN DEFAULT 0,
                function_group INTEGER DEFAULT 4,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        ''')
        
        conn_funcs = sqlite3.connect(source_funcs)
        cur_funcs = conn_funcs.cursor()
        
        # We need to make sure the table exists in the source
        try:
            cur_funcs.execute("SELECT function_name, function_code, function_description, approved_by_team, function_group, created_at, updated_at FROM data_quality_functions")
            rows = cur_funcs.fetchall()
            for r in rows:
                try:
                    cur_master.execute("INSERT INTO data_quality_functions (function_name, function_code, function_description, approved_by_team, function_group, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)", r)
                except sqlite3.IntegrityError:
                    pass # unique constraint hit
            print(f"Successfully copied {len(rows)} operational functions.")
        except Exception as e:
            print(f"Error copying functions: {e}")
            
        conn_funcs.close()
        
    # 2. Merge Knowledge Base Table
    if os.path.exists(source_kb):
        print("Merging `domain_knowledge` table...")
        cur_master.execute('''
        CREATE TABLE IF NOT EXISTS domain_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            category TEXT NOT NULL CHECK(category IN ('Process', 'Physics/Chemistry', 'Equipment', 'OEM')),
            topic TEXT NOT NULL,
            knowledge_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn_kb = sqlite3.connect(source_kb)
        cur_kb = conn_kb.cursor()
        
        try:
            cur_kb.execute("SELECT thread_id, category, topic, knowledge_text, created_at, updated_at FROM domain_knowledge")
            rows = cur_kb.fetchall()
            for r in rows:
                cur_master.execute("INSERT INTO domain_knowledge (thread_id, category, topic, knowledge_text, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)", r)
            print(f"Successfully copied {len(rows)} domain constraint entries.")
        except Exception as e:
            print(f"Error copying knowledge table: {e}")
            
        conn_kb.close()

    conn_master.commit()
    conn_master.close()
    print("Database Unification Complete! Master DB isolated to -> database/app.db")

if __name__ == "__main__":
    merge()
