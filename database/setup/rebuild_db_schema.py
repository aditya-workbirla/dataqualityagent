import sqlite3
import datetime

DB_PATH = "database/app.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Drop existing table
    cursor.execute('DROP TABLE IF EXISTS data_quality_functions')
    
    # Create the new table matching the updated schema
    cursor.execute('''
    CREATE TABLE data_quality_functions (
        function_id INTEGER PRIMARY KEY AUTOINCREMENT,
        function_name TEXT UNIQUE NOT NULL,
        function_code TEXT NOT NULL,
        function_description TEXT NOT NULL,
        function_group INTEGER NOT NULL,
        approved_by_team BOOLEAN NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL,
        approved_by TEXT
    )
    ''')
    
    now = datetime.datetime.now().isoformat()
    
    funcs = [
        # GROUP 1
        (1, "check_null_values", "Counts null and missing values per column. Flags any column exceeding a configurable threshold.", '''import pandas as pd
 
def check_null_values(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    threshold = params.get("threshold", 0.05)
    total_rows = len(df)
    results, flagged = {}, []
 
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = null_count / total_rows if total_rows > 0 else 0.0
        results[col] = {
            "null_count": null_count,
            "null_pct": round(null_pct * 100, 2)
        }
        if null_pct > threshold:
            flagged.append(col)
 
    return {
        "passed": len(flagged) == 0,
        "flagged_columns": flagged,
        "column_results": results,
        "summary": (
            f"{len(flagged)} column(s) exceed the {threshold * 100}% null threshold."
            if flagged else
            "All columns are within the acceptable null threshold."
        ),
    }'''),
        (1, "check_duplicate_rows", "Detects exact duplicate rows or partial-key duplicates.", '''import pandas as pd
 
def check_duplicate_rows(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    subset = params.get("subset", None)
    keep = params.get("keep", "first")
 
    mask = df.duplicated(subset=subset, keep=keep)
    duplicate_count = int(mask.sum())
    duplicate_pct = duplicate_count / len(df) * 100 if len(df) > 0 else 0.0
 
    return {
        "passed": duplicate_count == 0,
        "duplicate_count": duplicate_count,
        "duplicate_pct": round(duplicate_pct, 2),
        "duplicate_indices": list(df.index[mask])[:100],
        "summary": (
            f"{duplicate_count} duplicate row(s) found ({round(duplicate_pct, 2)}%)."
            if duplicate_count > 0 else
            "No duplicate rows detected."
        ),
    }'''),
        (1, "check_row_count", "Validates that the dataset row count falls within an expected range.", '''import pandas as pd
 
def check_row_count(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    min_rows = params.get("min_rows", None)
    max_rows = params.get("max_rows", None)
    actual = len(df)
    issues = []
 
    if min_rows is not None and actual < min_rows:
        issues.append(f"Row count {actual} is below minimum {min_rows}.")
    if max_rows is not None and actual > max_rows:
        issues.append(f"Row count {actual} exceeds maximum {max_rows}.")
 
    return {
        "passed": len(issues) == 0,
        "row_count": actual,
        "issues": issues,
        "summary": " ".join(issues) if issues else f"Row count {actual} is within expected range.",
    }'''),
        (1, "detect_outliers_iqr", "Detects outliers using the IQR (Interquartile Range) method with Tukey fences.", '''import pandas as pd
 
def detect_outliers_iqr(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    multiplier = params.get("multiplier", 1.5)
    columns = params.get("columns", df.select_dtypes(include="number").columns.tolist())
    results = {}
    all_flagged = set()
 
    for col in columns:
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found."}
            continue
 
        series = df[col].dropna()
        if series.empty:
            results[col] = {"outlier_count": 0, "note": "No non-null values."}
            continue
 
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        indices = list(df.index[mask])
        all_flagged.update(indices)
 
        results[col] = {
            "outlier_count": len(indices),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "flagged_indices": indices[:50],
        }
 
    return {
        "passed": len(all_flagged) == 0,
        "total_flagged_rows": len(all_flagged),
        "column_results": results,
        "summary": f"{len(all_flagged)} outlier row(s) across {len(columns)} column(s).",
    }'''),
        (1, "detect_outliers_zscore", "Detects outliers using the Z-score method. Values more than 3 standard deviations from the mean are flagged.", '''import pandas as pd
 
def detect_outliers_zscore(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    threshold = params.get("threshold", 3.0)
    columns = params.get("columns", df.select_dtypes(include="number").columns.tolist())
    results, all_flagged = {}, set()
 
    for col in columns:
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found."}
            continue
 
        series = df[col].dropna()
        if series.empty:
            results[col] = {"outlier_count": 0, "note": "No non-null values."}
            continue
 
        mean, std = series.mean(), series.std()
        if pd.isna(std) or std == 0:
            results[col] = {"outlier_count": 0, "note": "Zero variance or insufficient data - skipped."}
            continue
 
        z_scores = ((df[col] - mean) / std).abs()
        mask = z_scores > threshold
        indices = list(df.index[mask.fillna(False)])
        all_flagged.update(indices)
 
        results[col] = {
            "outlier_count": len(indices),
            "mean": round(mean, 4),
            "std": round(std, 4),
            "flagged_indices": indices[:50],
        }
 
    return {
        "passed": len(all_flagged) == 0,
        "total_flagged_rows": len(all_flagged),
        "column_results": results,
        "summary": f"{len(all_flagged)} outlier row(s) using Z-score threshold {threshold}.",
    }'''),
        (1, "check_column_variance", "Flags numeric columns with zero or near-zero variance.", '''import pandas as pd
 
def check_column_variance(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    threshold = params.get("threshold", 1e-6)
    columns = params.get("columns", df.select_dtypes(include="number").columns.tolist())
    constant_cols = []
    near_zero_cols = []
    results = {}
 
    for col in columns:
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found."}
            continue
 
        var = df[col].var()
        results[col] = {"variance": round(float(var), 8) if not pd.isna(var) else None}
 
        if pd.isna(var):
            continue
        if var == 0:
            constant_cols.append(col)
        elif var < threshold:
            near_zero_cols.append(col)
 
    return {
        "passed": len(constant_cols) == 0,
        "constant_columns": constant_cols,
        "near_zero_columns": near_zero_cols,
        "column_results": results,
        "summary": (
            f"{len(constant_cols)} constant and {len(near_zero_cols)} near-zero variance column(s)."
            if constant_cols or near_zero_cols else
            "All columns have acceptable variance."
        ),
    }'''),
        (1, "check_distribution_shift", "Compares the statistical distribution of numeric columns against a stored baseline.", '''import pandas as pd
 
def check_distribution_shift(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    baseline = params.get("baseline", {})
    mean_tol = params.get("mean_tol", 0.10)
    std_tol = params.get("std_tol", 0.20)
 
    if not baseline:
        return {"passed": True, "summary": "No baseline provided - skipped."}
 
    results, flagged = {}, []
 
    for col, ref in baseline.items():
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found."}
            continue
 
        curr_mean = df[col].mean()
        curr_std = df[col].std()
 
        ref_mean = ref.get("mean")
        ref_std = ref.get("std")
 
        if ref_mean is None or ref_std is None:
            results[col] = {"error": "Baseline mean/std missing."}
            continue
 
        mean_delta = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-9)
        std_delta = abs(curr_std - ref_std) / (abs(ref_std) + 1e-9)
        col_flagged = mean_delta > mean_tol or std_delta > std_tol
 
        if col_flagged:
            flagged.append(col)
 
        results[col] = {
            "current_mean": round(curr_mean, 4),
            "mean_delta_pct": round(mean_delta * 100, 2),
            "current_std": round(curr_std, 4),
            "std_delta_pct": round(std_delta * 100, 2),
            "flagged": col_flagged,
        }
 
    return {
        "passed": len(flagged) == 0,
        "flagged_columns": flagged,
        "column_results": results,
        "summary": (
            f"{len(flagged)} column(s) show significant distribution shift."
            if flagged else
            "No significant distribution shift detected."
        ),
    }'''),
    
        # GROUP 2
        (2, "check_timestamp_gaps", "Detects gaps in a datetime column larger than expected sampling interval. (Requires 'timestamp_col')", '''import pandas as pd
from datetime import datetime, timezone
 
def check_timestamp_gaps(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    ts_col = params.get("timestamp_col")
    interval = params.get("expected_interval", "1T")
    tol = params.get("tolerance_factor", 2.0)
 
    if not ts_col:
        return {"passed": False, "summary": "Missing required parameter: 'timestamp_col'."}
    if ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    ts = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if len(ts) < 2:
        return {"passed": True, "gap_count": 0, "gaps": [], "summary": "Not enough valid timestamps to evaluate gaps."}
 
    expected_delta = pd.tseries.frequencies.to_offset(interval).nanos / 1e9
    diffs = ts.diff().dropna().dt.total_seconds()
    gap_positions = diffs[diffs > expected_delta * tol].index
 
    gaps = [
        {
            "gap_start": str(ts.iloc[i - 1]),
            "gap_end": str(ts.iloc[i]),
            "gap_seconds": round(float(diffs.loc[i]), 1),
        }
        for i in gap_positions
    ]
 
    return {
        "passed": len(gaps) == 0,
        "gap_count": len(gaps),
        "gaps": gaps[:20],
        "summary": (
            f"{len(gaps)} timestamp gap(s) exceed the expected interval of {interval}."
            if gaps else
            f"No timestamp gaps exceed the expected interval of {interval}."
        ),
    }'''),
        (2, "check_timestamp_duplicates", "Identifies duplicate timestamps in a datetime column. (Requires 'timestamp_col')", '''import pandas as pd
 
def check_timestamp_duplicates(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    ts_col = params.get("timestamp_col")
    subset = params.get("subset", None)
 
    if not ts_col:
        return {"passed": False, "summary": "Missing required parameter: 'timestamp_col'."}
    if ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    check_cols = [ts_col] + subset if subset else [ts_col]
    missing = [c for c in check_cols if c not in df.columns]
    if missing:
        return {"passed": False, "summary": f"Column(s) not found: {missing}"}
 
    dup_mask = df.duplicated(subset=check_cols, keep=False)
    dup_count = int(dup_mask.sum())
 
    return {
        "passed": dup_count == 0,
        "duplicate_count": dup_count,
        "flagged_indices": list(df.index[dup_mask])[:50],
        "summary": (
            f"{dup_count} duplicate timestamp(s) found."
            if dup_count > 0 else
            f"No duplicate timestamps in '{ts_col}'."
        ),
    }'''),
        (2, "check_stale_data", "Validates that the most recent timestamp is not older than max age. (Requires 'timestamp_col')", '''import pandas as pd
from datetime import datetime, timezone
 
def check_stale_data(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    ts_col = params.get("timestamp_col")
    max_age_h = params.get("max_age_hours", 24.0)
    ref_str = params.get("reference_time", None)
 
    if not ts_col:
        return {"passed": False, "summary": "Missing required parameter: 'timestamp_col'."}
    if ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
    if ts.empty:
        return {"passed": False, "summary": "No valid timestamps found."}
 
    ref_time = pd.to_datetime(ref_str) if ref_str else datetime.now(timezone.utc)
    latest = ts.max()
 
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    if ref_time.tzinfo is None:
        ref_time = ref_time.tz_localize("UTC")
 
    age_hours = (ref_time - latest).total_seconds() / 3600
    passed = age_hours <= max_age_h
 
    return {
        "passed": passed,
        "latest_record": str(latest),
        "age_hours": round(age_hours, 2),
        "summary": (
            f"Latest record is {round(age_hours, 1)}h old - exceeds max of {max_age_h}h."
            if not passed else
            f"Data is fresh - latest record is {round(age_hours, 1)}h old."
        ),
    }'''),
        (2, "check_monotonic_sequence", "Validates columns are monotonically increasing or decreasing. (Requires 'rules': [{'column': 'x'}])", '''import pandas as pd
 
def check_monotonic_sequence(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    rules = params.get("rules", [])
    results, all_flagged = {}, set()
 
    for rule in rules:
        col = rule.get("column")
        direction = rule.get("direction", "increasing")
        strict = rule.get("strict", False)
 
        if col not in df.columns:
            results[col or "unknown"] = {"error": "Column not found."}
            continue
 
        series = df[col].reset_index(drop=True)
        diffs = series.diff().dropna()
 
        if direction == "increasing":
            mask = diffs < 0 if not strict else diffs <= 0
        else:
            mask = diffs > 0 if not strict else diffs >= 0
 
        positions = list(mask[mask].index)
        orig_indices = [df.index[i] for i in positions if i < len(df)]
        all_flagged.update(orig_indices)
 
        results[col] = {
            "direction": direction,
            "violation_count": len(orig_indices),
            "flagged_indices": orig_indices[:50],
        }
 
    return {
        "passed": len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "column_results": results,
        "summary": f"{len(all_flagged)} monotonicity violation(s) found.",
    }'''),

        # GROUP 3
        (3, "check_flatline", "Detects periods where a sensor reports the exact same value consequently. (Requires 'column' and 'timestamp_col')", '''import pandas as pd
 
def check_flatline(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    col = params.get("column")
    ts_col = params.get("timestamp_col")
    window_mins = params.get("window_minutes", 15)
 
    if not col:
        return {"passed": False, "summary": "Missing required parameter: 'column'."}
    if not ts_col:
        return {"passed": False, "summary": "Missing required parameter: 'timestamp_col'."}
    if col not in df.columns:
        return {"passed": False, "summary": f"Column '{col}' not found."}
    if ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    df_s = df[[ts_col, col]].dropna().copy()
    if df_s.empty:
        return {"passed": True, "flatline_count": 0, "flatline_events": [], "summary": f"No valid data to evaluate '{col}'."}
 
    df_s[ts_col] = pd.to_datetime(df_s[ts_col], errors="coerce")
    df_s = df_s.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
 
    if df_s.empty:
        return {"passed": True, "flatline_count": 0, "flatline_events": [], "summary": f"No valid timestamps to evaluate '{col}'."}
 
    df_s["_grp"] = (df_s[col] != df_s[col].shift()).cumsum()
    grp_info = df_s.groupby("_grp").agg(
        start=(ts_col, "first"),
        end=(ts_col, "last"),
        value=(col, "first"),
        count=(col, "size"),
    )
    grp_info["duration_mins"] = (grp_info["end"] - grp_info["start"]).dt.total_seconds() / 60
    flatlines = grp_info[grp_info["duration_mins"] >= window_mins]
    events = flatlines[["start", "end", "value", "duration_mins"]].to_dict("records")
 
    return {
        "passed": len(events) == 0,
        "flatline_count": len(events),
        "flatline_events": events[:20],
        "summary": (
            f"{len(events)} flatline period(s) >= {window_mins} min detected in '{col}'."
            if events else
            f"No flatlines detected in '{col}'."
        ),
    }'''),
        (3, "check_saturation", "Detects when a sensor reading remains at known limit. (Requires limits, column, timestamp_col)", '''import pandas as pd
 
def check_saturation(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    col = params.get("column")
    ts_col = params.get("timestamp_col")
    min_limit = params.get("min_limit")
    max_limit = params.get("max_limit")
    tolerance = params.get("tolerance", 0.01)
    window_mins = params.get("window_minutes", 10)
 
    missing = []
    if col is None: missing.append("column")
    if ts_col is None: missing.append("timestamp_col")
    if min_limit is None: missing.append("min_limit")
    if max_limit is None: missing.append("max_limit")
 
    if missing:
        return {"passed": False, "summary": f"Missing required parameter(s): {missing}"}
    if col not in df.columns:
        return {"passed": False, "summary": f"Column '{col}' not found."}
    if ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
    if max_limit <= min_limit:
        return {"passed": False, "summary": "Invalid limits: 'max_limit' must be greater than 'min_limit'."}
 
    df_s = df[[ts_col, col]].dropna().copy()
    if df_s.empty:
        return {"passed": True, "saturation_events": [], "summary": f"No valid data to evaluate '{col}'."}
 
    df_s[ts_col] = pd.to_datetime(df_s[ts_col], errors="coerce")
    df_s = df_s.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    if df_s.empty:
        return {"passed": True, "saturation_events": [], "summary": f"No valid timestamps to evaluate '{col}'."}
 
    rng = max_limit - min_limit
    tol = rng * tolerance
 
    saturated = (df_s[col] <= min_limit + tol) | (df_s[col] >= max_limit - tol)
    df_s["_saturated"] = saturated
    df_s["_sat_grp"] = (df_s["_saturated"] != df_s["_saturated"].shift()).cumsum()
 
    sat_grps = df_s[df_s["_saturated"]].groupby("_sat_grp").agg(
        start=(ts_col, "first"),
        end=(ts_col, "last"),
        value=(col, "first"),
    )
 
    if sat_grps.empty:
        return {"passed": True, "saturation_events": [], "summary": f"No saturation detected in '{col}'."}
 
    sat_grps["duration_mins"] = (sat_grps["end"] - sat_grps["start"]).dt.total_seconds() / 60
    flagged = sat_grps[sat_grps["duration_mins"] >= window_mins]
    events = flagged[["start", "end", "value", "duration_mins"]].to_dict("records")
 
    return {
        "passed": len(events) == 0,
        "saturation_events": events[:20],
        "summary": (
            f"{len(events)} saturation period(s) >= {window_mins} min in '{col}'."
            if events else
            f"No saturation detected in '{col}'."
        ),
    }'''),
        (3, "check_stagnation", "Flags periods where a sensor shows near-zero variation over a rolling window.", '''import pandas as pd
 
def check_stagnation(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    col = params.get("column")
    ts_col = params.get("timestamp_col")
    window_mins = params.get("window_minutes", 15)
    var_threshold = params.get("variance_threshold", 0.01)
 
    if not col:
        return {"passed": False, "summary": "Missing required parameter: 'column'."}
    if not ts_col:
        return {"passed": False, "summary": "Missing required parameter: 'timestamp_col'."}
    if col not in df.columns:
        return {"passed": False, "summary": f"Column '{col}' not found."}
    if ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    df_s = df[[ts_col, col]].dropna().copy()
    if df_s.empty:
        return {"passed": True, "stagnant_points": 0, "example_timestamps": [], "summary": f"No valid data to evaluate '{col}'."}
 
    df_s[ts_col] = pd.to_datetime(df_s[ts_col], errors="coerce")
    df_s = df_s.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
 
    if df_s.empty:
        return {"passed": True, "stagnant_points": 0, "example_timestamps": [], "summary": f"No valid timestamps to evaluate '{col}'."}
 
    window = f"{window_mins}min"
    rolling_mean = df_s[col].rolling(window).mean()
    rolling_std = df_s[col].rolling(window).std()
    cv = rolling_std / (rolling_mean.abs() + 1e-9)
 
    stagnant_mask = (cv < var_threshold).fillna(False)
    stagnant_indices = df_s.index[stagnant_mask].tolist()
 
    return {
        "passed": len(stagnant_indices) == 0,
        "stagnant_points": len(stagnant_indices),
        "example_timestamps": [str(t) for t in stagnant_indices[:10]],
        "summary": (
            f"{len(stagnant_indices)} point(s) show stagnation (CV < {var_threshold})."
            if stagnant_indices else
            f"No stagnation detected in '{col}'."
        ),
    }'''),
        (3, "check_spike_index", "Detects sudden, unrealistic jumps between consecutive sensor readings.", '''import pandas as pd
 
def check_spike_index(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    col = params.get("column")
    ts_col = params.get("timestamp_col")
    threshold = params.get("spike_threshold", 20)
 
    if not col:
        return {"passed": False, "summary": "Missing required parameter: 'column'."}
    if not ts_col:
        return {"passed": False, "summary": "Missing required parameter: 'timestamp_col'."}
    if col not in df.columns:
        return {"passed": False, "summary": f"Column '{col}' not found."}
    if ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    df_s = df[[ts_col, col]].dropna().copy()
    if df_s.empty:
        return {"passed": True, "spike_count": 0, "spike_events": [], "summary": f"No valid data to evaluate '{col}'."}
 
    df_s[ts_col] = pd.to_datetime(df_s[ts_col], errors="coerce")
    df_s = df_s.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
 
    if df_s.empty:
        return {"passed": True, "spike_count": 0, "spike_events": [], "summary": f"No valid timestamps to evaluate '{col}'."}
 
    df_s["_delta"] = df_s[col].diff().abs()
    spikes = df_s[df_s["_delta"] > threshold]
 
    events = [
        {"timestamp": str(r[ts_col]), "value": r[col], "delta": round(float(r["_delta"]), 4)}
        for _, r in spikes.head(20).iterrows()
    ]
 
    return {
        "passed": len(spikes) == 0,
        "spike_count": len(spikes),
        "spike_events": events,
        "summary": (
            f"{len(spikes)} spike(s) exceeding {threshold} units in '{col}'."
            if len(spikes) > 0 else
            f"No spikes detected in '{col}'."
        ),
    }'''),
        (3, "check_non_negative", "Flags negative values in columns where negative numbers are physically impossible.", '''import pandas as pd
 
def check_non_negative(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    columns = params.get("columns", [])
    strict = params.get("strict", False)
    results = {}
    all_flagged = set()
 
    if not columns:
        return {"passed": False, "summary": "Missing required parameter: 'columns'."}
 
    for col in columns:
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found."}
            continue
 
        mask = (df[col] < 0) if not strict else (df[col] <= 0)
        mask = mask & df[col].notna()
        indices = list(df.index[mask])
        all_flagged.update(indices)
 
        min_value = df[col].min()
        results[col] = {
            "violation_count": len(indices),
            "min_value_found": float(min_value) if pd.notna(min_value) else None,
            "flagged_indices": indices[:50],
        }
 
    return {
        "passed": len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "column_results": results,
        "summary": (
            f"{len(all_flagged)} negative value(s) in non-negative columns."
            if all_flagged else
            "All specified columns contain only non-negative values."
        ),
    }'''),
        (3, "check_conditional_notnull", "Enforces conditional completeness rules: 'if column A meets condition X, then B must not be null'.", '''import pandas as pd
 
def check_conditional_notnull(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    rules = params.get("rules", [])
    results, all_flagged = {}, set()
 
    ops = {
        "==": lambda s, v: s == v,
        "!=": lambda s, v: s != v,
        ">": lambda s, v: s > v,
        "<": lambda s, v: s < v,
        ">=": lambda s, v: s >= v,
        "<=": lambda s, v: s <= v,
    }
 
    for rule in rules:
        cond_col = rule.get("condition_col")
        cond_val = rule.get("condition_val")
        cond_op = rule.get("condition_op", "==")
        req_col = rule.get("required_col")
        label = rule.get("label", f"{req_col} required when {cond_col} {cond_op} {cond_val}")
 
        if cond_col not in df.columns or req_col not in df.columns:
            results[label] = {"error": "Column(s) not found."}
            continue
 
        op_fn = ops.get(cond_op, ops["=="])
        violate_mask = op_fn(df[cond_col], cond_val) & df[req_col].isnull()
        indices = list(df.index[violate_mask])
        all_flagged.update(indices)
 
        results[label] = {
            "violation_count": len(indices),
            "flagged_indices": indices[:50],
        }
 
    return {
        "passed": len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "rule_results": results,
        "summary": f"{len(all_flagged)} row(s) violate conditional not-null rules.",
    }'''),
        (3, "check_column_correlation_bounds", "Validates that Pearson correlation between pairs of numeric columns falls within an expected range.", '''import pandas as pd
 
def check_column_correlation_bounds(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    pairs = params.get("pairs", [])
    results, flagged = {}, []
 
    for pair in pairs:
        col_a = pair.get("col_a")
        col_b = pair.get("col_b")
        min_corr = pair.get("min_corr", -1.0)
        max_corr = pair.get("max_corr", 1.0)
        label = pair.get("label", f"{col_a} vs {col_b}")
 
        if col_a not in df.columns or col_b not in df.columns:
            results[label] = {"error": "Column(s) not found."}
            continue
 
        valid = df[[col_a, col_b]].dropna()
        actual = valid[col_a].corr(valid[col_b]) if len(valid) > 1 else None
        ok = actual is not None and min_corr <= actual <= max_corr
 
        if not ok:
            flagged.append(label)
 
        results[label] = {
            "actual_correlation": round(actual, 4) if actual is not None else None,
            "expected_min": min_corr,
            "expected_max": max_corr,
            "passed": ok,
        }
 
    return {
        "passed": len(flagged) == 0,
        "flagged_pairs": flagged,
        "pair_results": results,
        "summary": (
            f"{len(flagged)} column pair(s) have unexpected correlations."
            if flagged else
            "All column correlations are within expected bounds."
        ),
    }'''),
        (3, "check_whitespace_in_strings", "Finds strings with leading, trailing, or multiple consecutive internal spaces.", '''import pandas as pd
 
def check_whitespace_in_strings(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    columns = params.get("columns", df.select_dtypes(include="object").columns.tolist())
    fix = params.get("fix", False)
    results = {}
    all_flagged = set()
    df_out = df.copy() if fix else None
 
    for col in columns:
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found."}
            continue
 
        mask = df[col].notna() & df[col].astype(str).str.contains(
            r"^\s|\s$|\s{2,}", regex=True, na=False
        )
        indices = list(df.index[mask])
        all_flagged.update(indices)
 
        results[col] = {
            "whitespace_count": len(indices),
            "flagged_indices": indices[:50],
        }
 
        if fix and df_out is not None:
            original_non_null = df_out[col].notna()
            df_out.loc[original_non_null, col] = (
                df_out.loc[original_non_null, col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
 
    return {
        "passed": len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "column_results": results,
        "cleaned_df": df_out if fix else None,
        "summary": (
            f"{len(all_flagged)} string(s) with whitespace issues found."
            if all_flagged else
            "No whitespace issues detected."
        ),
    }'''),
        (3, "check_case_consistency", "Detects mixed-case inconsistencies in categorical columns.", '''import pandas as pd
 
def check_case_consistency(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    columns = params.get("columns", df.select_dtypes(include="object").columns.tolist())
    expected_case = params.get("expected_case", None)
    results, all_flagged = {}, set()
 
    def detect_dominant(series):
        styles = {"upper": 0, "lower": 0, "title": 0}
        for v in series:
            s = str(v)
            if s.isupper():
                styles["upper"] += 1
            elif s.islower():
                styles["lower"] += 1
            elif s.istitle():
                styles["title"] += 1
        return max(styles, key=styles.get)
 
    case_checks = {"upper": str.isupper, "lower": str.islower, "title": str.istitle}
 
    for col in columns:
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found."}
            continue
 
        series = df[col].dropna().astype(str)
        if series.empty:
            results[col] = {"dominant_case": None, "violation_count": 0, "flagged_indices": []}
            continue
 
        dom = expected_case or detect_dominant(series)
        check_fn = case_checks.get(dom)
        if not check_fn:
            results[col] = {"error": f"Unsupported case style '{dom}'."}
            continue
 
        mask = df[col].notna() & ~df[col].astype(str).map(check_fn)
        indices = list(df.index[mask])
        all_flagged.update(indices)
 
        results[col] = {
            "dominant_case": dom,
            "violation_count": len(indices),
            "flagged_indices": indices[:50],
        }
 
    return {
        "passed": len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "column_results": results,
        "summary": f"{len(all_flagged)} value(s) deviate from the expected case style.",
    }'''),
        (3, "check_regex_pattern", "Validates string values against a required regex pattern.", '''import pandas as pd
import re
 
def check_regex_pattern(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    rules = params.get("rules", [])
    results, all_flagged = {}, set()
 
    for rule in rules:
        col = rule.get("column")
        pattern = rule.get("pattern", ".*")
        label = rule.get("label", f"{col} pattern check")
 
        if col not in df.columns:
            results[label] = {"error": f"Column '{col}' not found."}
            continue
 
        compiled = re.compile(str(pattern))
        mask = df[col].notna() & ~df[col].astype(str).apply(lambda v: bool(compiled.fullmatch(v)))
        indices = list(df.index[mask])
        all_flagged.update(indices)
 
        results[label] = {
            "pattern": pattern,
            "violation_count": len(indices),
            "non_matching": df.loc[indices, col].value_counts().head(10).to_dict(),
            "flagged_indices": indices[:50],
        }
 
    return {
        "passed": len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "rule_results": results,
        "summary": f"{len(all_flagged)} value(s) fail regex format validation.",
    }'''),
        (3, "check_column_sum_bounds", "Validates that the sum of a numeric column falls within an expected range.", '''import pandas as pd
 
def check_column_sum_bounds(df: pd.DataFrame, params: dict = None) -> dict:
    params = params or {}
    rules = params.get("rules", [])
    results, flagged = {}, []
 
    for rule in rules:
        col = rule.get("column")
        min_sum = rule.get("min_sum", None)
        max_sum = rule.get("max_sum", None)
        grp_col = rule.get("group_by", None)
        label = rule.get("label", f"sum({col})")
 
        if col not in df.columns:
            results[label] = {"error": f"Column '{col}' not found."}
            continue
 
        if grp_col and grp_col in df.columns:
            sums = df.groupby(grp_col)[col].sum()
            issues = {
                str(g): round(float(t), 4)
                for g, t in sums.items()
                if (min_sum is not None and t < min_sum) or
                   (max_sum is not None and t > max_sum)
            }
            results[label] = {"group_violations": issues}
            if issues:
                flagged.append(label)
        else:
            total = df[col].sum()
            ok = True
            if min_sum is not None and total < min_sum:
                ok = False
            if max_sum is not None and total > max_sum:
                ok = False
 
            results[label] = {"actual_sum": round(float(total), 4), "passed": ok}
            if not ok:
                flagged.append(label)
 
    return {
        "passed": len(flagged) == 0,
        "flagged_rules": flagged,
        "rule_results": results,
        "summary": f"{len(flagged)} sum rule(s) violated.",
    }''')
    ]

    for fx in funcs:
        cursor.execute('''
        INSERT INTO data_quality_functions
        (function_group, function_name, function_description, function_code, approved_by_team, created_at, updated_at, approved_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fx[0], fx[1], fx[2], fx[3], 
            True,  # approved_by_team
            now, 
            now, 
            "System_Initialization"
        ))
        
    conn.commit()
    conn.close()
    print("Database recreated and seeded with the 22 user functions matching group schemas (1, 2, 3).")

if __name__ == "__main__":
    init_db()
