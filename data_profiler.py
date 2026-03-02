import pandas as pd
from typing import Dict, Any

def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes a DataFrame and returns a dictionary with data quality profile stats.
    This function generates a quick overview of missing and repeating values per column.
    """
    total_rows = len(df)
    
    # Store all calculated profile statistics in this dictionary
    profile = {
        "total_rows": total_rows,
        "columns": {}
    }
    
    # Loop over every column in the dataset to calculate individual statistics
    for col in df.columns:
        col_data = df[col]
        
        # Extract the Pandas data type of the column (e.g., 'object', 'int64', 'float64')
        dtype_str = str(col_data.dtype)
        
        # Calculate how many rows are missing (NaN)
        missing_count = col_data.isna().sum()
        # Convert the count into a percentage out of 100 for easier reading
        missing_percentage = (missing_count / total_rows * 100) if total_rows > 0 else 0
        
        # Check for consecutive repeating values (val == previous_val)
        if total_rows > 1:
            consecutive_repeating_count = (col_data == col_data.shift(1)).sum()
            repeating_percentage = (consecutive_repeating_count / total_rows * 100)
        else:
            repeating_percentage = 0
            
        # Basic summary stats for numeric columns only
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        min_val, max_val, std_val = None, None, None
        
        # If numeric, calculate minimum, maximum, and standard deviation to give the LLM context
        if is_numeric and total_rows > 0:
            min_val = float(col_data.min()) if not pd.isna(col_data.min()) else None
            max_val = float(col_data.max()) if not pd.isna(col_data.max()) else None
            std_val = float(col_data.std()) if not pd.isna(col_data.std()) else None
            
        profile["columns"][col] = {
            "dtype": dtype_str,
            "missing_percentage": round(float(missing_percentage), 2),
            "consecutive_repeating_percentage": round(float(repeating_percentage), 2),
            "is_numeric": is_numeric,
        }
        
        if is_numeric:
            profile["columns"][col].update({
                "min": min_val,
                "max": max_val,
                "std": std_val
            })
            
    return profile
