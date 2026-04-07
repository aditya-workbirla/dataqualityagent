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
    db_path = "database/app.db"
    
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            cursor = conn.cursor()
            # Fetch ALL functions that have been approved by the team (Groups 1, 2, and 3)
            cursor.execute("SELECT function_name, function_code, function_group FROM data_quality_functions WHERE approved_by_team = 1 AND function_group IN (1, 2, 3)")
            approved_functions = cursor.fetchall()
    except Exception as e:
        return {"error": f"Failed to connect to database: {e}"}
        
    # Auto-detect a timestamp column
    ts_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            ts_col = col
            break
    if not ts_col:
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower() or 'ts' in col.lower():
                ts_col = col
                break

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Execute functions on the full dataset
    for func_name, func_code, func_group in approved_functions:
        try:
            local_scope = {}
            exec_globals = {"pd": pd, "np": np, "Dict": Dict, "Any": Any}
            exec(func_code, exec_globals, local_scope)
            
            if func_name in local_scope:
                func = local_scope[func_name]
                
                # Check if it's a grouped function requiring auto-looping
                if func_group in (2, 3):
                    # Default parameters
                    base_params = {"timestamp_col": ts_col, "columns": numeric_cols}
                    
                    if 'params.get("column")' in func_code:
                        col_results = {}
                        any_failed = False
                        valid_executions = 0
                        
                        for col in numeric_cols:
                            p = dict(base_params)
                            p["column"] = col
                            res = func(df, p)
                            
                            if res.get("passed") is False and "Missing required parameter" in str(res.get("summary", "")):
                                # Skip if we can't run it (e.g. saturation requires min_limit)
                                continue
                                
                            col_results[col] = res
                            valid_executions += 1
                            if not res.get("passed"):
                                any_failed = True
                                
                        if valid_executions > 0:
                            summary["dataset_checks"][func_name] = {
                                "passed": not any_failed,
                                "summary": f"Executed across {valid_executions} numeric columns.",
                                "column_results": col_results
                            }
                    else:
                        res = func(df, base_params)
                        if not (res.get("passed") is False and "Missing required parameter" in str(res.get("summary", ""))):
                            summary["dataset_checks"][func_name] = res
                else:
                    # Group 1 (or default) - standard execution
                    res = func(df)
                    summary["dataset_checks"][func_name] = res
            else:
                summary["dataset_checks"][func_name] = {"error": "Function name not found in code body."}
                
        except Exception as e:
            summary["dataset_checks"][func_name] = {"error": str(e)}
            
    return summary
