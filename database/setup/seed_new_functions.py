import sqlite3
import datetime

DB_PATH = "database/app.db"

NEW_FUNCTIONS = [
    {
        "name": "check_null_values",
        "desc": "Counts null and missing values per column and flags any column exceeding a threshold.",
        "code": '''def check_null_values(df: pd.DataFrame, params: dict = {}) -> dict:
    threshold = params.get("threshold", 0.05)
    total_rows = len(df)
    results, flagged = {}, []
 
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct   = null_count / total_rows if total_rows > 0 else 0.0
        results[col] = {"null_count": null_count, "null_pct": round(null_pct * 100, 2)}
        if null_pct > threshold:
            flagged.append(col)
 
    return {
        "passed": len(flagged) == 0,
        "flagged_columns": flagged,
        "column_results": results,
        "summary": (f"{len(flagged)} column(s) exceed the {threshold*100}% null threshold."
                    if flagged else "All columns are within the acceptable null threshold."),
    }'''
    },
    {
        "name": "check_duplicate_rows",
        "desc": "Detects exact duplicate rows or partial-key duplicates.",
        "code": '''def check_duplicate_rows(df: pd.DataFrame, params: dict = {}) -> dict:
    subset = params.get("subset", None)
    keep   = params.get("keep", "first")
 
    mask            = df.duplicated(subset=subset, keep=keep)
    duplicate_count = int(mask.sum())
    duplicate_pct   = duplicate_count / len(df) * 100 if len(df) > 0 else 0.0
 
    return {
        "passed":            duplicate_count == 0,
        "duplicate_count":   duplicate_count,
        "duplicate_pct":     round(duplicate_pct, 2),
        "duplicate_indices": list(df.index[mask])[:100],
        "summary": (f"{duplicate_count} duplicate row(s) found ({round(duplicate_pct,2)}%)."
                    if duplicate_count > 0 else "No duplicate rows detected."),
    }'''
    },
    {
        "name": "check_row_count",
        "desc": "Validates that the dataset row count falls within an expected range.",
        "code": '''def check_row_count(df: pd.DataFrame, params: dict = {}) -> dict:
    min_rows = params.get("min_rows", None)
    max_rows = params.get("max_rows", None)
    actual   = len(df)
    issues   = []
 
    if min_rows is not None and actual < min_rows:
        issues.append(f"Row count {actual} is below minimum {min_rows}.")
    if max_rows is not None and actual > max_rows:
        issues.append(f"Row count {actual} exceeds maximum {max_rows}.")
 
    return {
        "passed":    len(issues) == 0,
        "row_count": actual,
        "issues":    issues,
        "summary":   " ".join(issues) if issues else f"Row count {actual} is within expected range.",
    }'''
    },
    {
        "name": "check_flatline",
        "desc": "Detects periods where a sensor reports the exact same value consecutively for longer than a configurable window.",
        "code": '''def check_flatline(df: pd.DataFrame, params: dict = {}) -> dict:
    col         = params.get("column")
    ts_col      = params.get("timestamp_col")
    window_mins = params.get("window_minutes", 15)
 
    if not col or col not in df.columns:
        return {"passed": False, "summary": f"Column '{col}' not found."}
    if not ts_col or ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    df_s = df[[ts_col, col]].dropna().copy()
    if df_s.empty:
        return {"passed": True, "summary": f"Not enough valid data in '{col}'."}
    df_s[ts_col] = pd.to_datetime(df_s[ts_col])
    df_s = df_s.sort_values(ts_col).reset_index(drop=True)
 
    df_s["_grp"] = (df_s[col] != df_s[col].shift()).cumsum()
    grp_info = df_s.groupby("_grp").agg(
        start=(ts_col, "first"), end=(ts_col, "last"),
        value=(col, "first"), count=(col, "size")
    )
    grp_info["duration_mins"] = (grp_info["end"] - grp_info["start"]).dt.total_seconds() / 60
    flatlines = grp_info[grp_info["duration_mins"] >= window_mins]
    events    = flatlines[["start","end","value","duration_mins"]].to_dict("records")
 
    return {
        "passed":          len(events) == 0,
        "flatline_count":  len(events),
        "flatline_events": events[:20],
        "summary": (f"{len(events)} flatline period(s) >= {window_mins} min detected in '{col}'."
                    if events else f"No flatlines detected in '{col}'."),
    }'''
    },
    {
        "name": "check_saturation",
        "desc": "Detects when a sensor reading remains at or near its known physical minimum or maximum limit for an extended period.",
        "code": '''def check_saturation(df: pd.DataFrame, params: dict = {}) -> dict:
    col         = params.get("column")
    ts_col      = params.get("timestamp_col")
    min_limit   = params.get("min_limit")
    max_limit   = params.get("max_limit")
    tolerance   = params.get("tolerance", 0.01)
    window_mins = params.get("window_minutes", 10)
 
    if None in [col, min_limit, max_limit]:
        return {"passed": False, "summary": "column, min_limit, and max_limit are required."}
    if not ts_col or ts_col not in df.columns:
        return {"passed": False, "summary": "timestamp column required but missing."}
 
    rng  = max_limit - min_limit
    tol  = rng * tolerance
    df_s = df[[ts_col, col]].dropna().copy()
    if df_s.empty:
        return {"passed": True, "summary": f"Not enough valid data in '{col}'."}
    df_s[ts_col] = pd.to_datetime(df_s[ts_col])
    df_s = df_s.sort_values(ts_col).reset_index(drop=True)
 
    saturated        = (df_s[col] <= min_limit + tol) | (df_s[col] >= max_limit - tol)
    df_s["_sat_grp"] = (saturated != saturated.shift()).cumsum()
    sat_grps = df_s[saturated].groupby("_sat_grp").agg(
        start=(ts_col, "first"), end=(ts_col, "last"), value=(col, "first")
    )
    sat_grps["duration_mins"] = (sat_grps["end"] - sat_grps["start"]).dt.total_seconds() / 60
    flagged = sat_grps[sat_grps["duration_mins"] >= window_mins]
    events  = flagged[["start","end","value","duration_mins"]].to_dict("records")
 
    return {
        "passed":            len(events) == 0,
        "saturation_events": events[:20],
        "summary": (f"{len(events)} saturation period(s) >= {window_mins} min in '{col}'."
                    if events else f"No saturation detected in '{col}'."),
    }'''
    },
    {
        "name": "check_stagnation",
        "desc": "Flags periods where a sensor shows near-zero variation over a rolling time window.",
        "code": '''def check_stagnation(df: pd.DataFrame, params: dict = {}) -> dict:
    col           = params.get("column")
    ts_col        = params.get("timestamp_col")
    window_mins   = params.get("window_minutes", 15)
    var_threshold = params.get("variance_threshold", 0.01)
 
    if not col or col not in df.columns:
        return {"passed": False, "summary": f"Column '{col}' not found."}
    if not ts_col or ts_col not in df.columns:
        return {"passed": False, "summary": "timestamp column required but missing."}
 
    df_s = df[[ts_col, col]].dropna().copy()
    if df_s.empty:
        return {"passed": True, "summary": f"Not enough valid data in '{col}'."}
    df_s[ts_col] = pd.to_datetime(df_s[ts_col])
    df_s = df_s.set_index(ts_col).sort_index()
 
    window       = f"{window_mins}min"
    rolling_mean = df_s[col].rolling(window).mean()
    rolling_std  = df_s[col].rolling(window).std()
    cv           = rolling_std / (rolling_mean.abs() + 1e-9)
 
    stagnant_mask    = cv < var_threshold
    # Fill NaN from rolling so we don't crash
    stagnant_mask = stagnant_mask.fillna(False)
    stagnant_indices = df_s.index[stagnant_mask].tolist()
 
    return {
        "passed":             not bool(stagnant_mask.any()),
        "stagnant_points":    len(stagnant_indices),
        "example_timestamps": [str(t) for t in stagnant_indices[:10]],
        "summary": (f"{len(stagnant_indices)} point(s) show stagnation (CV < {var_threshold})."
                    if stagnant_indices else f"No stagnation detected in '{col}'."),
    }'''
    },
    {
        "name": "check_spike_index",
        "desc": "Detects sudden, unrealistic jumps between consecutive sensor readings.",
        "code": '''def check_spike_index(df: pd.DataFrame, params: dict = {}) -> dict:
    col       = params.get("column")
    ts_col    = params.get("timestamp_col")
    threshold = params.get("spike_threshold", 20)
 
    if not col or col not in df.columns:
        return {"passed": False, "summary": f"Column '{col}' not found."}
    if not ts_col or ts_col not in df.columns:
        return {"passed": False, "summary": "timestamp column required but missing."}
 
    df_s = df[[ts_col, col]].dropna().copy()
    if df_s.empty:
        return {"passed": True, "summary": f"Not enough valid data in '{col}'."}
    df_s[ts_col] = pd.to_datetime(df_s[ts_col])
    df_s = df_s.sort_values(ts_col).reset_index(drop=True)
    df_s["_delta"] = df_s[col].diff().abs()
    spikes = df_s[df_s["_delta"] > threshold]
 
    events = [{"timestamp": str(r[ts_col]), "value": r[col], "delta": round(r["_delta"], 4)}
              for _, r in spikes.head(20).iterrows()]
 
    return {
        "passed":       len(events) == 0,
        "spike_count":  len(spikes),
        "spike_events": events,
        "summary": (f"{len(spikes)} spike(s) exceeding {threshold} units in '{col}'."
                    if spikes.shape[0] > 0 else f"No spikes detected in '{col}'."),
    }'''
    },
    {
        "name": "detect_outliers_iqr",
        "desc": "Detects outliers using the IQR (Interquartile Range) method with Tukey fences.",
        "code": '''def detect_outliers_iqr(df: pd.DataFrame, params: dict = {}) -> dict:
    multiplier  = params.get("multiplier", 1.5)
    columns     = params.get("columns", df.select_dtypes(include="number").columns.tolist())
    results     = {}
    all_flagged = set()
 
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty: continue
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR    = Q3 - Q1
        lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
        mask    = (df[col] < lower) | (df[col] > upper)
        indices = list(df.index[mask])
        all_flagged.update(indices)
        results[col] = {"outlier_count": len(indices), "lower_bound": round(lower, 4),
                        "upper_bound": round(upper, 4), "flagged_indices": indices[:50]}
 
    return {
        "passed":             len(all_flagged) == 0,
        "total_flagged_rows": len(all_flagged),
        "column_results":     results,
        "summary":            f"{len(all_flagged)} outlier row(s) across {len(columns)} column(s).",
    }'''
    },
    {
        "name": "detect_outliers_zscore",
        "desc": "Detects outliers using the Z-score method.",
        "code": '''def detect_outliers_zscore(df: pd.DataFrame, params: dict = {}) -> dict:
    threshold   = params.get("threshold", 3.0)
    columns     = params.get("columns", df.select_dtypes(include="number").columns.tolist())
    results, all_flagged = {}, set()
 
    for col in columns:
        if col not in df.columns:
            continue
        series    = df[col].dropna()
        if series.empty: continue
        mean, std = series.mean(), series.std()
        if std == 0 or pd.isna(std):
            results[col] = {"outlier_count": 0, "note": "Zero or unknown variance - skipped."}
            continue
        z_scores = ((df[col] - mean) / std).abs()
        mask     = z_scores > threshold
        indices  = list(df.index[mask])
        all_flagged.update(indices)
        results[col] = {"outlier_count": len(indices), "mean": round(mean, 4),
                        "std": round(std, 4), "flagged_indices": indices[:50]}
 
    return {
        "passed":             len(all_flagged) == 0,
        "total_flagged_rows": len(all_flagged),
        "column_results":     results,
        "summary":            f"{len(all_flagged)} outlier row(s) using Z-score threshold {threshold}.",
    }'''
    },
    {
        "name": "check_column_variance",
        "desc": "Flags numeric columns with zero or near-zero variance.",
        "code": '''def check_column_variance(df: pd.DataFrame, params: dict = {}) -> dict:
    threshold      = params.get("threshold", 1e-6)
    columns        = params.get("columns", df.select_dtypes(include="number").columns.tolist())
    constant_cols  = []
    near_zero_cols = []
    results        = {}
 
    for col in columns:
        if col not in df.columns: continue
        var = df[col].var()
        results[col] = {"variance": round(float(var), 8) if not pd.isna(var) else None}
        if pd.isna(var): continue
        if var == 0:
            constant_cols.append(col)
        elif var < threshold:
            near_zero_cols.append(col)
 
    return {
        "passed":            len(constant_cols) == 0,
        "constant_columns":  constant_cols,
        "near_zero_columns": near_zero_cols,
        "column_results":    results,
        "summary": (f"{len(constant_cols)} constant and {len(near_zero_cols)} near-zero variance column(s)."
                    if constant_cols or near_zero_cols else "All columns have acceptable variance."),
    }'''
    },
    {
        "name": "check_distribution_shift",
        "desc": "Compares the statistical distribution of numeric columns against a stored baseline.",
        "code": '''def check_distribution_shift(df: pd.DataFrame, params: dict = {}) -> dict:
    baseline = params.get("baseline", {})
    mean_tol = params.get("mean_tol", 0.10)
    std_tol  = params.get("std_tol", 0.20)
 
    if not baseline:
        return {"passed": True, "summary": "No baseline provided - skipped."}
 
    results, flagged = {}, []
    for col, ref in baseline.items():
        if col not in df.columns:
            continue
        curr_mean  = df[col].mean()
        curr_std   = df[col].std()
        mean_delta = abs(curr_mean - ref["mean"]) / (abs(ref["mean"]) + 1e-9)
        std_delta  = abs(curr_std  - ref["std"])  / (abs(ref["std"])  + 1e-9)
        col_flagged= mean_delta > mean_tol or std_delta > std_tol
        if col_flagged:
            flagged.append(col)
        results[col] = {"current_mean": round(curr_mean, 4), "mean_delta_pct": round(mean_delta*100, 2),
                        "current_std":  round(curr_std,  4), "std_delta_pct":  round(std_delta*100, 2),
                        "flagged": col_flagged}
 
    return {
        "passed":          len(flagged) == 0,
        "flagged_columns": flagged,
        "column_results":  results,
        "summary": (f"{len(flagged)} column(s) show significant distribution shift."
                    if flagged else "No significant distribution shift detected."),
    }'''
    },
    {
        "name": "check_non_negative",
        "desc": "Flags negative values in columns where negative numbers are physically impossible.",
        "code": '''def check_non_negative(df: pd.DataFrame, params: dict = {}) -> dict:
    columns     = params.get("columns", df.select_dtypes(include="number").columns.tolist())
    strict      = params.get("strict", False)
    results     = {}
    all_flagged = set()
 
    for col in columns:
        if col not in df.columns:
            results[col] = {"error": f"Column '{col}' not found."}
            continue
        mask    = df[col] < 0 if not strict else df[col] <= 0
        mask    = mask & df[col].notna()
        indices = list(df.index[mask])
        all_flagged.update(indices)
        if len(indices) > 0:
            results[col] = {"violation_count": len(indices),
                            "min_value_found": float(df[col].min()),
                            "flagged_indices": indices[:50]}
 
    return {
        "passed":           len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "column_results":   results,
        "summary": (f"{len(all_flagged)} negative value(s) in non-negative columns."
                    if all_flagged else "All specified columns contain only non-negative values."),
    }'''
    },
    {
        "name": "check_conditional_notnull",
        "desc": "Enforces conditional completeness rules: 'if column A meets condition X, then column B must not be null'.",
        "code": '''def check_conditional_notnull(df: pd.DataFrame, params: dict = {}) -> dict:
    rules, results, all_flagged = params.get("rules", []), {}, set()
    ops = {"==": lambda s,v: s==v, "!=": lambda s,v: s!=v,
           ">":  lambda s,v: s>v,  "<":  lambda s,v: s<v,
           ">=": lambda s,v: s>=v, "<=": lambda s,v: s<=v}
 
    for rule in rules:
        cond_col = rule.get("condition_col")
        cond_val = rule.get("condition_val")
        cond_op  = rule.get("condition_op", "==")
        req_col  = rule.get("required_col")
        label    = rule.get("label", f"{req_col} required when {cond_col} {cond_op} {cond_val}")
        if cond_col not in df.columns or req_col not in df.columns:
            results[label] = {"error": "Column(s) not found."}
            continue
        op_fn        = ops.get(cond_op, lambda s,v: s==v)
        violate_mask = op_fn(df[cond_col], cond_val) & df[req_col].isnull()
        indices      = list(df.index[violate_mask])
        all_flagged.update(indices)
        results[label] = {"violation_count": len(indices), "flagged_indices": indices[:50]}
 
    return {
        "passed":           len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "rule_results":     results,
        "summary":          f"{len(all_flagged)} row(s) violate conditional not-null rules.",
    }'''
    },
    {
        "name": "check_column_correlation_bounds",
        "desc": "Validates that the Pearson correlation between pairs of numeric columns falls within an expected range.",
        "code": '''def check_column_correlation_bounds(df: pd.DataFrame, params: dict = {}) -> dict:
    pairs, results, flagged = params.get("pairs", []), {}, []
 
    for pair in pairs:
        col_a    = pair.get("col_a")
        col_b    = pair.get("col_b")
        min_corr = pair.get("min_corr", -1.0)
        max_corr = pair.get("max_corr",  1.0)
        label    = pair.get("label", f"{col_a} vs {col_b}")
        if col_a not in df.columns or col_b not in df.columns:
            results[label] = {"error": "Column(s) not found."}
            continue
        valid  = df[[col_a, col_b]].dropna()
        actual = valid[col_a].corr(valid[col_b]) if len(valid) > 1 else None
        ok     = actual is not None and min_corr <= actual <= max_corr
        if not ok:
            flagged.append(label)
        results[label] = {"actual_correlation": round(actual, 4) if actual else None,
                          "expected_min": min_corr, "expected_max": max_corr, "passed": ok}
 
    return {
        "passed":        len(flagged) == 0,
        "flagged_pairs": flagged,
        "pair_results":  results,
        "summary": (f"{len(flagged)} column pair(s) have unexpected correlations."
                    if flagged else "All column correlations are within expected bounds."),
    }'''
    },
    {
        "name": "check_monotonic_sequence",
        "desc": "Validates that specified columns are monotonically increasing or decreasing. Applied to timestamp columns, batch sequence numbers, or cumulative counters.",
        "code": '''def check_monotonic_sequence(df: pd.DataFrame, params: dict = {}) -> dict:
    rules, results, all_flagged = params.get("rules", []), {}, set()
 
    for rule in rules:
        col       = rule.get("column")
        direction = rule.get("direction", "increasing")
        strict    = rule.get("strict", False)
        if col not in df.columns:
            results[col] = {"error": "Column not found."}
            continue
        series = df[col].reset_index(drop=True)
        diffs  = series.diff().dropna()
        if direction == "increasing":
            mask = diffs < 0 if not strict else diffs <= 0
        else:
            mask = diffs > 0 if not strict else diffs >= 0
        positions    = list(mask[mask].index)
        orig_indices = [df.index[i] for i in positions if i < len(df)]
        all_flagged.update(orig_indices)
        results[col] = {"direction": direction, "violation_count": len(orig_indices),
                        "flagged_indices": orig_indices[:50]}
 
    return {
        "passed":           len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "column_results":   results,
        "summary":          f"{len(all_flagged)} monotonicity violation(s) found.",
    }'''
    },
    {
        "name": "check_timestamp_gaps",
        "desc": "Detects gaps in a datetime column larger than the expected sampling interval.",
        "code": '''def check_timestamp_gaps(df: pd.DataFrame, params: dict = {}) -> dict:
    ts_col   = params.get("timestamp_col")
    interval = params.get("expected_interval", "1T")
    tol      = params.get("tolerance_factor", 2.0)
 
    if not ts_col or ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    ts             = pd.to_datetime(df[ts_col]).sort_values().reset_index(drop=True)
    try:
        expected_delta = pd.tseries.frequencies.to_offset(interval).nanos / 1e9
    except Exception:
        expected_delta = pd.Timedelta(interval).total_seconds()
    diffs          = ts.diff().dropna().dt.total_seconds()
    gap_mask       = diffs > expected_delta * tol
    gaps = [{"gap_start": str(ts.iloc[i-1]), "gap_end": str(ts.iloc[i]),
              "gap_seconds": round(diffs.iloc[i], 1)}
             for i in gap_mask[gap_mask].index]
 
    return {
        "passed":    len(gaps) == 0,
        "gap_count": len(gaps),
        "gaps":      gaps[:20],
        "summary":   f"{len(gaps)} timestamp gap(s) exceed the expected interval of {interval}.",
    }'''
    },
    {
        "name": "check_timestamp_duplicates",
        "desc": "Identifies duplicate timestamps in a datetime column.",
        "code": '''def check_timestamp_duplicates(df: pd.DataFrame, params: dict = {}) -> dict:
    ts_col = params.get("timestamp_col")
    subset = params.get("subset", None)
 
    if not ts_col or ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    check_cols = ([ts_col] + subset) if subset else [ts_col]
    dup_mask   = df.duplicated(subset=check_cols, keep=False)
    dup_count  = int(dup_mask.sum())
 
    return {
        "passed":           dup_count == 0,
        "duplicate_count":  dup_count,
        "flagged_indices":  list(df.index[dup_mask])[:50],
        "summary": (f"{dup_count} duplicate timestamp(s) found."
                    if dup_count > 0 else f"No duplicate timestamps in '{ts_col}'."),
    }'''
    },
    {
        "name": "check_stale_data",
        "desc": "Validates that the most recent timestamp in a datetime column is not older than a configurable maximum age.",
        "code": '''def check_stale_data(df: pd.DataFrame, params: dict = {}) -> dict:
    ts_col    = params.get("timestamp_col")
    max_age_h = params.get("max_age_hours", 24.0)
    ref_str   = params.get("reference_time", None)
 
    if not ts_col or ts_col not in df.columns:
        return {"passed": False, "summary": f"Timestamp column '{ts_col}' not found."}
 
    import pandas as pd
    from datetime import datetime, timezone
    ref_time = pd.to_datetime(ref_str) if ref_str else datetime.now(timezone.utc)
    try:
        latest = pd.to_datetime(df[ts_col]).dropna().max()
    except Exception:
        return {"passed": False, "summary": "Failed to parse dates."}
        
    if pd.isnull(latest):
        return {"passed": False, "summary": "No valid timestamps found."}
    if str(latest.tzinfo) == "None":
        latest = latest.tz_localize("UTC")
    if hasattr(ref_time, "tzinfo") and str(ref_time.tzinfo) == "None":
        ref_time = ref_time.replace(tzinfo=timezone.utc)
 
    age_hours = (ref_time - latest).total_seconds() / 3600
    passed    = age_hours <= max_age_h
 
    return {
        "passed":        passed,
        "latest_record": str(latest),
        "age_hours":     round(age_hours, 2),
        "summary": (f"Latest record is {round(age_hours,1)}h old - exceeds max of {max_age_h}h."
                    if not passed else f"Data is fresh - latest record is {round(age_hours,1)}h old."),
    }'''
    },
    {
        "name": "check_whitespace_in_strings",
        "desc": "Finds strings with leading, trailing, or multiple consecutive internal spaces.",
        "code": '''def check_whitespace_in_strings(df: pd.DataFrame, params: dict = {}) -> dict:
    columns     = params.get("columns", df.select_dtypes(include="object").columns.tolist())
    fix         = params.get("fix", False)
    results     = {}
    all_flagged = set()
    df_out      = df.copy() if fix else None
 
    for col in columns:
        if col not in df.columns:
            continue
        mask = df[col].notna() & df[col].astype(str).str.contains(
            r"^\s|\s$|\s{2,}", regex=True, na=False)
        indices = list(df.index[mask])
        all_flagged.update(indices)
        if len(indices) > 0:
            results[col] = {"whitespace_count": len(indices), "flagged_indices": indices[:50]}
        if fix and df_out is not None:
            df_out[col] = df_out[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
 
    return {
        "passed":           len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "column_results":   results,
        "cleaned_df":       "Use fix directly if needed",
        "summary": (f"{len(all_flagged)} string(s) with whitespace issues found."
                    if all_flagged else "No whitespace issues detected."),
    }'''
    },
    {
        "name": "check_case_consistency",
        "desc": "Detects mixed-case inconsistencies in categorical columns.",
        "code": '''def check_case_consistency(df: pd.DataFrame, params: dict = {}) -> dict:
    columns       = params.get("columns", df.select_dtypes(include="object").columns.tolist())
    expected_case = params.get("expected_case", None)
    results, all_flagged = {}, set()
 
    def detect_dominant(series):
        styles = {"upper": 0, "lower": 0, "title": 0}
        for v in series:
            s = str(v)
            if s.isupper(): styles["upper"] += 1
            elif s.islower(): styles["lower"] += 1
            elif s.istitle(): styles["title"] += 1
        return max(styles, key=styles.get)
 
    case_checks = {"upper": str.isupper, "lower": str.islower, "title": str.istitle}
 
    for col in columns:
        if col not in df.columns:
            continue
        series   = df[col].dropna().astype(str)
        if series.empty: continue
        dom      = expected_case or detect_dominant(series)
        check_fn = case_checks.get(dom)
        if not check_fn:
            continue
        mask    = df[col].notna() & ~df[col].astype(str).map(check_fn)
        indices = list(df.index[mask])
        all_flagged.update(indices)
        if len(indices) > 0:
            results[col] = {"dominant_case": dom, "violation_count": len(indices),
                            "flagged_indices": indices[:50]}
 
    return {
        "passed":           len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "column_results":   results,
        "summary":          f"{len(all_flagged)} value(s) deviate from the expected case style.",
    }'''
    },
    {
        "name": "check_regex_pattern",
        "desc": "Validates string values against a required regex pattern.",
        "code": '''def check_regex_pattern(df: pd.DataFrame, params: dict = {}) -> dict:
    import re
    rules, results, all_flagged = params.get("rules", []), {}, set()
 
    for rule in rules:
        col     = rule.get("column")
        pattern = rule.get("pattern", ".*")
        label   = rule.get("label", f"{col} pattern check")
        if col not in df.columns:
            results[label] = {"error": f"Column '{col}' not found."}
            continue
        compiled = re.compile(str(pattern))
        mask     = df[col].notna() & ~df[col].astype(str).apply(
            lambda v: bool(compiled.fullmatch(v)))
        indices  = list(df.index[mask])
        all_flagged.update(indices)
        results[label] = {"pattern": pattern, "violation_count": len(indices),
                          "non_matching": df.loc[indices, col].value_counts().head(10).to_dict(),
                          "flagged_indices": indices[:50]}
 
    return {
        "passed":           len(all_flagged) == 0,
        "total_violations": len(all_flagged),
        "rule_results":     results,
        "summary":          f"{len(all_flagged)} value(s) fail regex format validation.",
    }'''
    },
    {
        "name": "check_column_sum_bounds",
        "desc": "Validates that the sum of a numeric column falls within an expected range.",
        "code": '''def check_column_sum_bounds(df: pd.DataFrame, params: dict = {}) -> dict:
    rules, results, flagged = params.get("rules", []), {}, []
 
    for rule in rules:
        col     = rule.get("column")
        min_sum = rule.get("min_sum", None)
        max_sum = rule.get("max_sum", None)
        grp_col = rule.get("group_by", None)
        label   = rule.get("label", f"sum({col})")
        if col not in df.columns:
            results[label] = {"error": f"Column '{col}' not found."}
            continue
        if grp_col and grp_col in df.columns:
            sums   = df.groupby(grp_col)[col].sum()
            issues = {str(g): round(t, 4) for g, t in sums.items()
                      if (min_sum is not None and t < min_sum) or
                         (max_sum is not None and t > max_sum)}
            results[label] = {"group_violations": issues}
            if issues:
                flagged.append(label)
        else:
            total = df[col].sum()
            ok    = True
            if min_sum is not None and total < min_sum: ok = False
            if max_sum is not None and total > max_sum: ok = False
            results[label] = {"actual_sum": round(total, 4), "passed": ok}
            if not ok:
                flagged.append(label)
 
    return {
        "passed":        len(flagged) == 0,
        "flagged_rules": flagged,
        "rule_results":  results,
        "summary":       f"{len(flagged)} sum rule(s) violated.",
    }'''
    },
    {
        "name": "check_batch_completeness",
        "desc": "Validates that each unique batch ID contains the expected number of records.",
        "code": '''def check_batch_completeness(df: pd.DataFrame, params: dict = {}) -> dict:
    batch_col      = params.get("batch_col")
    expected_count = params.get("expected_count", None)
    min_count      = params.get("min_count", None)
    max_count      = params.get("max_count", None)
 
    if not batch_col or batch_col not in df.columns:
        return {"passed": False, "summary": f"Batch column '{batch_col}' not found."}
 
    counts, flagged = df.groupby(batch_col).size(), {}
    for batch, count in counts.items():
        issues = []
        if expected_count is not None and count != expected_count:
            issues.append(f"expected {expected_count}, got {count}")
        if min_count is not None and count < min_count:
            issues.append(f"below minimum {min_count}")
        if max_count is not None and count > max_count:
            issues.append(f"exceeds maximum {max_count}")
        if issues:
            flagged[str(batch)] = {"count": int(count), "issues": issues}
 
    return {
        "passed":          len(flagged) == 0,
        "total_batches":   int(counts.shape[0]),
        "flagged_batches": flagged,
        "summary": (f"{len(flagged)} batch(es) do not meet completeness requirement."
                    if flagged else f"All {counts.shape[0]} batches are complete."),
    }'''
    }
]

def seed_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.datetime.now().isoformat()
    
    for func in NEW_FUNCTIONS:
        try:
            cursor.execute('''
            INSERT INTO data_quality_functions
            (function_name, function_code, function_description, approved_by_team, created_at, updated_at, approved_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                func["name"], 
                func["code"], 
                func["desc"], 
                True,  # Set to approved directly
                now, 
                now, 
                "System_Advanced_Seeder"
            ))
            print(f"Added {func['name']} to DB.")
        except sqlite3.IntegrityError:
            print(f"Function {func['name']} already exists. Updating.")
            cursor.execute('''
            UPDATE data_quality_functions
            SET function_code = ?, function_description = ?, updated_at = ?
            WHERE function_name = ?
            ''', (func["code"], func["desc"], now, func["name"]))
            
    conn.commit()
    conn.close()

if __name__ == "__main__":
    seed_db()
