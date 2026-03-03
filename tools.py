import pandas as pd
from typing import Any
from langchain_core.tools import tool
import json

# Global reference to the current dataframe for tools to use
# (In a production app, we would pass this dataframe via LangChain Runnable binding 
# or custom tool creation, but a global variable is simple for this script)
_CURRENT_DF = None

def set_current_df(df: pd.DataFrame):
    """Stores the active dataframe so the `pandas_query_tool` can query against it."""
    global _CURRENT_DF
    _CURRENT_DF = df

@tool
def pandas_query_tool(query: str) -> str:
    """
    Executes a pandas query on the dataset to check for logical consistency and constraints.
    Provide a valid pandas `.query()` string. 
    Returns a JSON string containing the count of rows matching the condition and a sample of matching rows.
    
    Examples:
    - query: "flow_rate < 0" -> Checks if there are any rows with negative flow rate.
    - query: "start_date > end_date" -> Checks if start_date is after end_date.
    - query: "dosing_rate < 0" -> Checks if dosing rate is negative.
    """
    global _CURRENT_DF
    if _CURRENT_DF is None:
        return "Error: No dataframe loaded."
    
    try:
        # Use pandas `.query()` to execute the string condition the LLM generated
        result_df = _CURRENT_DF.query(query)
        match_count = len(result_df)
        
        # We cap the returned sample to 5 rows to avoid blowing up the LLM's context window
        sample = result_df.head(5).to_dict(orient="records")
        
        # Get the indices of the matching rows
        matched_indices = result_df.index.tolist()
        
        return json.dumps({
            "success": True,
            "query": query,
            "match_count": match_count,
            "matched_indices": matched_indices,
            "sample_matches": sample
        }, default=str)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })
