import sqlite3
import datetime

DB_PATH = "database/app.db"

def init_knowledge_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing table if any
    cursor.execute('DROP TABLE IF EXISTS domain_knowledge')
    
    # Create the new knowledge table
    # Category will be one of: 'Process', 'Physics/Chemistry', 'Equipment', 'OEM'
    cursor.execute('''
    CREATE TABLE domain_knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT NOT NULL,
        category TEXT NOT NULL,
        topic TEXT NOT NULL,
        knowledge_text TEXT NOT NULL,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL
    )
    ''')
    
    now = datetime.datetime.now().isoformat()
    
    # Seed with some initial examples for the agent to reference
    initial_kb = [
        ("Process", "Pulp Processing Constraints", "Standard pulp processing flow requires consistent pressure levels above 10 PSI to prevent fiber degradation."),
        ("Physics/Chemistry", "Temperature Limits", "Absolute Zero in Celsius is -273.15C. Process temperatures cannot physically drop below environmental ambient unless active cooling is engaged."),
        ("Physics/Chemistry", "pH Levels", "pH values must strictly fall between 0.0 and 14.0. Any pH outside this range is mathematically impossible for aqueous solutions."),
        ("Equipment", "Blower Inlet", "The primary blower inlet operates with variable frequency drives. Zero RPM indicates a stopped state, but negative RPM is impossible."),
        ("OEM", "Sensor Tolerance", "Standard OEM temperature sensors (Type K Thermocouples) have an operational error margin of +/- 2.2C.")
    ]
    
    for kb in initial_kb:
        cursor.execute('''
        INSERT INTO domain_knowledge
        (thread_id, category, topic, knowledge_text, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', ("global", kb[0], kb[1], kb[2], now, now))
        
    conn.commit()
    conn.close()
    print("Database `knowledge.db` created and seeded with initial 4-part categories (Process, Physics/Chemistry, Equipment, OEM).")

if __name__ == "__main__":
    init_knowledge_db()
