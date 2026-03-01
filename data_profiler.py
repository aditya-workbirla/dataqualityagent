import pandas as pd
from typing import Dict, Any

def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes a DataFrame and returns a dictionary with data quality profile stats.
    """
    total_rows = len(df)
    profile = {
        "total_rows": total_rows,
        "columns": {}
    }
    
    for col in df.columns:
        col_data = df[col]
        
        # Data type
        dtype_str = str(col_data.dtype)
        
        # Missing values
        missing_count = col_data.isna().sum()
        missing_percentage = (missing_count / total_rows * 100) if total_rows > 0 else 0
        
        # Repeating values (most common value)
        if not col_data.mode().empty:
            most_common_val = col_data.mode().iloc[0]
            repeating_count = (col_data == most_common_val).sum()
            repeating_percentage = (repeating_count / total_rows * 100) if total_rows > 0 else 0
        else:
            most_common_val = None
            repeating_percentage = 0
            
        # Summary stats for numeric columns
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        min_val, max_val, std_val = None, None, None
        if is_numeric and total_rows > 0:
            min_val = float(col_data.min()) if not pd.isna(col_data.min()) else None
            max_val = float(col_data.max()) if not pd.isna(col_data.max()) else None
            std_val = float(col_data.std()) if not pd.isna(col_data.std()) else None
            
        profile["columns"][col] = {
            "dtype": dtype_str,
            "missing_percentage": round(float(missing_percentage), 2),
            "most_common_value_percentage": round(float(repeating_percentage), 2),
            "is_numeric": is_numeric,
        }
        
        if is_numeric:
            profile["columns"][col].update({
                "min": min_val,
                "max": max_val,
                "std": std_val
            })
            
    return profile
