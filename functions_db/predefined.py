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
    summary = {}
    db_path = "functions.db"
    
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            # Only fetch functions that have been approved by the team
            cursor.execute("SELECT function_name, function_code FROM data_quality_functions WHERE approved_by_team = 1")
            approved_functions = cursor.fetchall()
    except Exception as e:
        return {"error": f"Failed to connect to database: {e}"}
        
    for col in df.columns:
        col_series = df[col]
        col_summary = {
            "dtype": str(col_series.dtype),
            "checks": {}
        }
        
        for func_name, func_code in approved_functions:
            try:
                # Sandboxed local scope for executing the function string
                local_scope = {}
                # Provide necessary globals to the exec environment
                exec_globals = {"pd": pd, "np": np, "Dict": Dict, "Any": Any}
                exec(func_code, exec_globals, local_scope)
                
                if func_name in local_scope:
                    func = local_scope[func_name]
                    res = func(col_series)
                    
                    # Ensure the function name is recorded
                    stored_func_name = res.pop("function", func_name)
                    col_summary["checks"][stored_func_name] = res
                else:
                    col_summary["checks"][func_name] = {"error": "Function name not found in code body."}
                    
            except Exception as e:
                col_summary["checks"][func_name] = {"error": str(e)}
                
        summary[col] = col_summary
        
    return summary
