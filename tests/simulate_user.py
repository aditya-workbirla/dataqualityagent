import sys
sys.path.append(r"c:\Users\aditya.sareen-st\Downloads\Data agent")

import os
from dotenv import load_dotenv
import uuid
import pandas as pd
from agents.agent import build_data_quality_graph
from langgraph.checkpoint.sqlite import SqliteSaver

def simulate():
    load_dotenv()
    
    # We will simulate the exact call from app.py
    import sqlite3
    conn = sqlite3.connect("database/app.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = build_data_quality_graph(checkpointer=checkpointer)
    
    # Create fake dataframe
    df = pd.DataFrame({"temperature": [1, 2], "flow_rate": [3, 4], "ph": [5, 6], "rpm": [7, 8]})
    df_json = df.to_json(orient="records")
    
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "df_json": df_json, 
        "user_context_prompt": "Pulp and Fiber Data",
        "function_results_summary": {}, 
        "messages": [],
        "issues": [], 
        "bad_indices_per_column": {}, 
        "report": ""
    }
    
    print(f"Running simulation for thread: {thread_id}")
    final_state = app.invoke(initial_state, config)
    print("Graph execution complete.")
    
    # Peek at knowledge.db
    import sqlite3
    with sqlite3.connect("knowledge.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM domain_knowledge WHERE thread_id = ?", (thread_id,))
        rows = cursor.fetchall()
        print(f"Found {len(rows)} rows in knowledge.db for this thread:")
        for r in rows:
            print(r)

if __name__ == "__main__":
    simulate()
