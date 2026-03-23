import sqlite3
import datetime

DB_PATH = "database/app.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the table matching the user's schema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS data_quality_functions (
        function_id INTEGER PRIMARY KEY AUTOINCREMENT,
        function_name TEXT UNIQUE NOT NULL,
        function_code TEXT NOT NULL,
        function_description TEXT NOT NULL,
        approved_by_team BOOLEAN NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL,
        approved_by TEXT
    )
    ''')
    
    # Seed predefined functions
    now = datetime.datetime.now().isoformat()
    
    predefined_funcs = [
        {
            "name": "check_missing_values",
            "desc": "Checks for missing values in a column. Expects a pandas Series and returns a dict with missing count and percentage.",
            "code": '''def check_missing_values(series: pd.Series) -> Dict[str, Any]:
    """Checks for missing values in a column."""
    missing_count = int(series.isna().sum())
    total_count = len(series)
    missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
    
    return {
        "function": "check_missing_values",
        "missing_count": missing_count,
        "missing_percentage": round(missing_pct, 2)
    }'''
        },
        {
            "name": "check_repeating_values",
            "desc": "Checks for consecutive repeating values in a column. Expects a pandas Series and returns a dict with repeating count and percentage.",
            "code": '''def check_repeating_values(series: pd.Series) -> Dict[str, Any]:
    """Checks for consecutive repeating values in a column."""
    total_rows = len(series)
    if total_rows > 1:
        # Avoid counting NaNs as repeating
        valid_series = series.dropna()
        if len(valid_series) > 1:
            repeating_count = int((valid_series == valid_series.shift(1)).sum())
            repeating_pct = (repeating_count / total_rows * 100)
        else:
            repeating_count = 0
            repeating_pct = 0
    else:
        repeating_count = 0
        repeating_pct = 0
        
    return {
        "function": "check_repeating_values",
        "consecutive_repeating_count": repeating_count,
        "consecutive_repeating_percentage": round(repeating_pct, 2)
    }'''
        },
        {
            "name": "check_central_tendency_and_outliers",
            "desc": "Calculates min, max, mean, std, and identifies potential outliers using 3 IQR rule. Expects a numeric pandas Series.",
            "code": '''def check_central_tendency_and_outliers(series: pd.Series) -> Dict[str, Any]:
    """Calculates min, max, mean, std, and identifies potential outliers using 3 IQR rule."""
    if not pd.api.types.is_numeric_dtype(series):
        return {"function": "check_central_tendency_and_outliers", "status": "skipped (non-numeric)"}
        
    valid_data = series.dropna()
    if len(valid_data) == 0:
        return {"function": "check_central_tendency_and_outliers", "status": "skipped (no data)"}
        
    min_val = float(valid_data.min())
    max_val = float(valid_data.max())
    mean_val = float(valid_data.mean())
    std_val = float(valid_data.std()) if len(valid_data) > 1 else 0.0
    
    # Outlier detection (basic IQR)
    q1 = valid_data.quantile(0.25)
    q3 = valid_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
    outlier_count = len(outliers)
    
    return {
        "function": "check_central_tendency_and_outliers",
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "std": std_val,
        "outlier_count": outlier_count,
        "negative_values_present": min_val < 0
    }'''
        }
    ]
    
    for func in predefined_funcs:
        # Check if it already exists
        cursor.execute("SELECT function_id FROM data_quality_functions WHERE function_name = ?", (func["name"],))
        result = cursor.fetchone()
        
        if not result:
            cursor.execute('''
            INSERT INTO data_quality_functions
            (function_name, function_code, function_description, approved_by_team, created_at, updated_at, approved_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                func["name"], 
                func["code"], 
                func["desc"], 
                True,  # approved_by_team
                now, 
                now, 
                "System_Initialization"
            ))
            
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH} with {len(predefined_funcs)} predefined functions.")

if __name__ == "__main__":
    init_db()
