import pandas as pd
import numpy as np
from typing import Dict, Any

import sqlite3

def run_all_verified_functions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Orchestrator to run all verified functions independently on each column.
    Fetches functions from the SQLite database where approved_by_team = True.
    Returns a unified JSON-serializable dictionary summary.
    """
    import inspect
    
    summary = {"dataset_checks": {}}
    db_path = "functions.db"
    
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            # Only fetch GROUP 1 functions that have been approved by the team
            cursor.execute("SELECT function_name, function_code FROM data_quality_functions WHERE approved_by_team = 1 AND function_group = 1")
            approved_functions = cursor.fetchall()
    except Exception as e:
        return {"error": f"Failed to connect to database: {e}"}
        
    # Execute Group 1 functions on the full dataset
    for func_name, func_code in approved_functions:
        try:
            local_scope = {}
            exec_globals = {"pd": pd, "np": np, "Dict": Dict, "Any": Any}
            exec(func_code, exec_globals, local_scope)
            
            if func_name in local_scope:
                func = local_scope[func_name]
                # All Group 1 functions take df and optional params
                res = func(df)
                
                # Check if it passed
                passed = res.get("passed", False)
                summary["dataset_checks"][func_name] = res
            else:
                summary["dataset_checks"][func_name] = {"error": "Function name not found in code body."}
                
        except Exception as e:
            summary["dataset_checks"][func_name] = {"error": str(e)}
            
    return summary
