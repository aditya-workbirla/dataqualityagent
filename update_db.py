import sqlite3

db_path = "database/app.db"

functions = [
    (
        "check_missing_values",
        "Runs on all columns, returns null count and percentage per column sorted by worst first. Agent uses this to display the top 5.",
        """import pandas as pd

def check_missing_values(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    total_rows = len(df)
    results = {}

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round(null_count / total_rows * 100, 2) if total_rows > 0 else 0.0
        results[col] = {
            "null_count": null_count,
            "null_pct": null_pct,
        }

    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["null_count"], reverse=True)
    )
    columns_with_missing = {k: v for k, v in sorted_results.items() if v["null_count"] > 0}

    return {
        "passed": len(columns_with_missing) == 0,
        "total_columns_checked": len(df.columns),
        "columns_with_missing": len(columns_with_missing),
        "results_sorted": sorted_results,
        "summary": (
            f"{len(columns_with_missing)} column(s) have missing values."
            if columns_with_missing else
            "No missing values found across all columns."
        ),
    }""",
        1,
        1
    ),
    (
        "check_duplicate_values",
        "Runs on all columns independently, returns duplicate count and percentage per column sorted by worst first. Agent uses this to display columns with high internal redundancy.",
        """import pandas as pd

def check_duplicate_values(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    results = {}
    
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
            
        duplicate_count = int(series.duplicated(keep="first").sum())
        duplicate_pct = round((duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
        
        results[col] = {
            "duplicate_count": duplicate_count,
            "duplicate_pct": duplicate_pct
        }

    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["duplicate_count"], reverse=True)
    )
    # filter for ones with > 0 duplicates to display
    columns_with_duplicates = {k: v for k, v in sorted_results.items() if v["duplicate_count"] > 0}

    return {
        "passed": len(columns_with_duplicates) == 0,
        "total_columns_checked": len(df.columns),
        "columns_with_duplicates": len(columns_with_duplicates),
        "results_sorted": sorted_results,
        "summary": (
            f"{len(columns_with_duplicates)} column(s) contain duplicate intra-column values."
            if columns_with_duplicates else
            "No duplicate values detected within any column."
        ),
    }""",
        1,
        1
    ),
    (
        "check_flatline",
        "Runs on all numeric columns, detects any column where a value repeats for 5 or more consecutive rows. Returns per-column flatline count and percentage of total rows that are flatlined. Agent uses this to list all affected columns.",
        """import pandas as pd

def check_flatline(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    consecutive_threshold = params.get("consecutive_threshold", 5)
    total_rows = len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    results = {}

    for col in numeric_cols:
        series = df[col].reset_index(drop=True)
        # Count consecutive runs of the same value
        groups = (series != series.shift()).cumsum()
        run_lengths = series.groupby(groups).transform("count")
        flatline_mask = run_lengths >= consecutive_threshold
        flatline_count = int(flatline_mask.sum())
        flatline_pct = round(flatline_count / total_rows * 100, 2) if total_rows > 0 else 0.0

        if flatline_count > 0:
            results[col] = {
                "flatline_row_count": flatline_count,
                "flatline_pct": flatline_pct,
            }

    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["flatline_row_count"], reverse=True)
    )

    return {
        "passed": len(sorted_results) == 0,
        "total_columns_checked": len(numeric_cols),
        "columns_with_flatlines": len(sorted_results),
        "consecutive_threshold": consecutive_threshold,
        "results_sorted": sorted_results,
        "summary": (
            f"{len(sorted_results)} column(s) have flatline runs of {consecutive_threshold}+ consecutive identical values."
            if sorted_results else
            f"No flatlines detected across {len(numeric_cols)} numeric column(s)."
        ),
    }""",
        1,
        1
    ),
    (
        "detect_outliers_iqr",
        "Runs on all numeric columns using IQR method, returns outlier count and percentage per column sorted by worst first. Agent uses this to display the top 3.",
        """import pandas as pd

def detect_outliers_iqr(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    multiplier = params.get("multiplier", 1.5)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    results = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_count = int(mask.sum())
        outlier_pct = round(outlier_count / len(df) * 100, 2)

        results[col] = {
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
        }

    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["outlier_count"], reverse=True)
    )
    columns_with_outliers = {k: v for k, v in sorted_results.items() if v["outlier_count"] > 0}

    return {
        "passed": len(columns_with_outliers) == 0,
        "total_columns_checked": len(numeric_cols),
        "columns_with_outliers": len(columns_with_outliers),
        "results_sorted": sorted_results,
        "summary": (
            f"{len(columns_with_outliers)} column(s) contain outliers using IQR method."
            if columns_with_outliers else
            "No outliers detected across all numeric columns."
        ),
    }""",
        2,
        1
    ),
    (
        "check_column_variance",
        "Runs on all numeric columns, returns variance per column and flags those below a threshold. Agent uses the flagged list to highlight suspiciously low variance columns.",
        """import pandas as pd

def check_column_variance(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    threshold = params.get("threshold", 0.01)  # flag if variance < 1% of column mean
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    results = {}
    flagged = []

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        var = float(series.var())
        mean = float(series.mean())
        # Use coefficient of variation to normalise across different scales
        cv = (var ** 0.5) / (abs(mean) + 1e-9)
        is_flagged = cv < threshold

        results[col] = {
            "variance": round(var, 6),
            "mean": round(mean, 4),
            "coefficient_of_variation": round(cv, 6),
            "flagged": is_flagged,
        }
        if is_flagged:
            flagged.append(col)

    return {
        "passed": len(flagged) == 0,
        "total_columns_checked": len(numeric_cols),
        "low_variance_columns": flagged,
        "cv_threshold": threshold,
        "results": results,
        "summary": (
            f"{len(flagged)} column(s) show suspiciously low variance (CV < {threshold})."
            if flagged else
            "All numeric columns show acceptable variance."
        ),
    }""",
        2,
        1
    ),
    (
        "check_spike_index",
        "Runs on all numeric columns, detects consecutive value jumps exceeding a dynamic threshold (3x the column's rolling standard deviation). Returns spike count per column sorted by worst first. Agent uses this to display the top 3.",
        """import pandas as pd

def check_spike_index(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    # Dynamic threshold: flag if consecutive diff > 3x the column's std
    std_multiplier = params.get("std_multiplier", 3.0)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    results = {}

    for col in numeric_cols:
        series = df[col].dropna().reset_index(drop=True)
        if len(series) < 2:
            continue
        diffs = series.diff().abs().dropna()
        threshold = std_multiplier * series.std()
        if threshold == 0:
            continue
        spike_mask = diffs > threshold
        spike_count = int(spike_mask.sum())
        spike_pct = round(spike_count / len(df) * 100, 2)

        results[col] = {
            "spike_count": spike_count,
            "spike_pct": spike_pct,
            "dynamic_threshold": round(float(threshold), 4),
        }

    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["spike_count"], reverse=True)
    )
    columns_with_spikes = {k: v for k, v in sorted_results.items() if v["spike_count"] > 0}

    return {
        "passed": len(columns_with_spikes) == 0,
        "total_columns_checked": len(numeric_cols),
        "columns_with_spikes": len(columns_with_spikes),
        "std_multiplier_used": std_multiplier,
        "results_sorted": sorted_results,
        "summary": (
            f"{len(columns_with_spikes)} column(s) contain spike values using a {std_multiplier}x std threshold."
            if columns_with_spikes else
            "No spikes detected across all numeric columns."
        ),
    }""",
        2,
        1
    ),
    (
        "check_negative_values",
        "Runs on all numeric columns, returns every column that contains any negative value along with the count, percentage, min value, and a sample of flagged row indices. Agent passes this full result to the domain-reasoning agent which decides whether the negatives are realistic or not.",
        """import pandas as pd

def check_negative_values(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    results = {}
    flagged = []

    for col in numeric_cols:
        series = df[col].dropna()
        neg_mask = series < 0
        neg_count = int(neg_mask.sum())

        if neg_count > 0:
            neg_pct = round(neg_count / len(df) * 100, 2)
            min_value = round(float(series.min()), 4)
            sample_indices = list(df.index[df[col] < 0])[:10]

            results[col] = {
                "negative_count": neg_count,
                "negative_pct": neg_pct,
                "min_value": min_value,
                "sample_indices": sample_indices,
            }
            flagged.append(col)

    return {
        "passed": len(flagged) == 0,
        "total_columns_checked": len(numeric_cols),
        "columns_with_negatives": flagged,
        "results": results,
        "summary": (
            f"{len(flagged)} column(s) contain negative values. "
            f"Passing to domain agent for realism check."
            if flagged else
            "No negative values found across all numeric columns."
        ),
    }""",
        3,
        1
    )
]

import datetime
now = datetime.datetime.now().isoformat()

formatted_functions = []
for name, desc, code, grp, appr in functions:
    formatted_functions.append((name, code, desc, grp, appr, now, now, "SystemUpdate"))

try:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Ensure schema handles the group field
        cursor.execute("PRAGMA table_info(data_quality_functions)")
        cols = [c[1] for c in cursor.fetchall()]
        if 'function_group' not in cols:
            cursor.execute("ALTER TABLE data_quality_functions ADD COLUMN function_group INTEGER DEFAULT 1")
            
        # Clear existing functions
        cursor.execute("DELETE FROM data_quality_functions")
        
        # Insert new functions
        cursor.executemany(
            "INSERT INTO data_quality_functions (function_name, function_code, function_description, function_group, approved_by_team, created_at, updated_at, approved_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            formatted_functions
        )
        conn.commit()
        print("Successfully updated database with new functions.")
except Exception as e:
    print(f"Error: {e}")
