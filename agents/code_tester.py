"""
Code Testing Agent
==================
Two testing modes:

1. test_and_fix_function(function_name, code, target_column, df)
   Legacy single-column mode — kept for backward compatibility.

2. test_and_fix_script(script, df, question)  ← NEW
   Full-dataframe script mode.
   The LLM writes a complete Python script that:
     - receives the full DataFrame as `df`
     - may use any/all columns
     - may chain multiple operations
     - must assign its final answer to a variable called `RESULT`
       (a JSON-serialisable dict or list)
   The tester executes the script, captures RESULT, and auto-corrects
   via the LLM up to MAX_FIX_ATTEMPTS times on failure.
"""

from __future__ import annotations

import ast
import io
import sys
import traceback
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CodeTestResult:
    success: bool
    final_code: str
    result: Any = None
    error: str = ""
    attempts: int = 0
    test_log: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MAX_FIX_ATTEMPTS = 3

_EXEC_GLOBALS_BASE = lambda: {
    "pd": pd,
    "np": __import__("numpy"),
    "Dict": Dict,
    "Any": Any,
    "List": List,
    "Optional": Optional,
}


# ===========================================================================
#  MODE 1 — Single-column function  (legacy, kept for compatibility)
# ===========================================================================

def _static_check(code: str, function_name: str) -> Optional[str]:
    try:
        tree = compile(code, "<generated>", "exec", flags=ast.PyCF_ONLY_AST)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}\n  → {exc.text}"
    defined_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if function_name not in defined_names:
        return (
            f"Function `{function_name}` is not defined in the provided code.\n"
            f"Defined names: {sorted(defined_names) or '(none)'}"
        )
    return None


def _runtime_test(code: str, function_name: str, series: pd.Series) -> tuple[bool, Any, str]:
    local_scope: dict = {}
    g = _EXEC_GLOBALS_BASE()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exec(code, g, local_scope)
        func = local_scope[function_name]
        result = func(series)
        return True, result, ""
    except Exception:
        tb = traceback.format_exc()
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
        combined = tb + (f"\n[stdout]\n{out}" if out else "") + (f"\n[stderr]\n{err}" if err else "")
        return False, None, combined
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _llm_fix_function(function_name: str, code: str, error: str, series_info: str, attempt: int) -> str:
    import re
    from agents.agent import get_llm
    llm = get_llm(timeout_seconds=60.0)
    prompt = textwrap.dedent(f"""
        You are an expert Python bug-fixer for data quality functions.

        ## Task
        Fix the function `{function_name}` (attempt {attempt}).
        Return ONLY the corrected Python function definition — no prose, no markdown fences.

        ## Requirements
        - Signature: `def {function_name}(series: pd.Series) -> dict:`
        - Return a plain Python dict (JSON-serialisable). Use int()/float() not numpy scalars.
        - Only imports available: pd, np, Dict, Any, List, Optional.

        ## Column info
        {series_info}

        ## Error
        ```
        {error}
        ```

        ## Broken code
        ```python
        {code}
        ```

        ## Fixed code:
    """).strip()
    from langchain_core.messages import HumanMessage
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        return code
    raw = re.sub(r"^```(?:python)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def test_and_fix_function(
    function_name: str,
    code: str,
    target_column: str,
    df: pd.DataFrame,
    verbose: bool = True,
) -> CodeTestResult:
    """Legacy single-column mode."""
    if target_column not in df.columns:
        return CodeTestResult(
            success=False, final_code=code,
            error=f"Column '{target_column}' not found. Available: {list(df.columns)}",
        )
    series = df[target_column]
    series_info = (
        f"Column: '{target_column}'  dtype: {series.dtype}  "
        f"n={len(series)}  sample: {series.dropna().head(5).tolist()}"
    )
    current_code = code
    test_log: List[Dict[str, Any]] = []
    for attempt in range(1, MAX_FIX_ATTEMPTS + 2):
        if verbose:
            print(f"\n{'─'*60}\n🧪  Code Tester (fn) — attempt {attempt}  [{function_name}]")
        static_err = _static_check(current_code, function_name)
        if static_err:
            test_log.append({"attempt": attempt, "phase": "static", "error": static_err})
            if attempt > MAX_FIX_ATTEMPTS:
                break
            current_code = _llm_fix_function(function_name, current_code, static_err, series_info, attempt)
            continue
        ok, result, runtime_err = _runtime_test(current_code, function_name, series)
        if ok:
            test_log.append({"attempt": attempt, "phase": "runtime", "success": True, "result": result})
            if verbose:
                print(f"  ✅ Passed on attempt {attempt}. Result: {str(result)[:200]}")
            return CodeTestResult(success=True, final_code=current_code, result=result,
                                  attempts=attempt, test_log=test_log)
        else:
            test_log.append({"attempt": attempt, "phase": "runtime", "success": False, "error": runtime_err})
            if verbose:
                print(f"  ❌ Runtime error:\n{textwrap.indent(runtime_err[:400], '     ')}")
            if attempt > MAX_FIX_ATTEMPTS:
                break
            current_code = _llm_fix_function(function_name, current_code, runtime_err, series_info, attempt)
    last_error = test_log[-1].get("error", "Unknown error")
    if verbose:
        print(f"  ⚠️  Gave up after {MAX_FIX_ATTEMPTS} attempts.")
    return CodeTestResult(success=False, final_code=current_code, error=last_error,
                          attempts=MAX_FIX_ATTEMPTS, test_log=test_log)


# ===========================================================================
#  MODE 2 — Full-dataframe script  (NEW)
# ===========================================================================

def _script_syntax_check(script: str) -> Optional[str]:
    """Check the script compiles and assigns RESULT."""
    try:
        compile(script, "<script>", "exec", flags=ast.PyCF_ONLY_AST)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}\n  → {exc.text}"
    # Warn (not block) if RESULT not found — runtime will catch it
    if "RESULT" not in script:
        return "Script must assign its answer to a variable named `RESULT`."
    return None


def _script_runtime_test(script: str, df: pd.DataFrame) -> tuple[bool, Any, str]:
    """
    Execute `script` with `df` in scope.
    Expects the script to set `RESULT = <dict or list>`.
    Returns (ok, result, error_text).
    """
    g = _EXEC_GLOBALS_BASE()
    g["df"] = df
    local_scope: dict = {}
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exec(script, g, local_scope)
        # RESULT may be in locals or globals
        result = local_scope.get("RESULT", g.get("RESULT"))
        if result is None:
            raise ValueError("Script completed but `RESULT` was not set or is None.")
        captured_out = sys.stdout.getvalue()
        return True, result, captured_out  # return stdout as extra info
    except Exception:
        tb = traceback.format_exc()
        out = sys.stdout.getvalue()
        err = sys.stderr.getvalue()
        combined = tb + (f"\n[stdout]\n{out}" if out else "") + (f"\n[stderr]\n{err}" if err else "")
        return False, None, combined
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _llm_fix_script(
    script: str,
    error: str,
    df_info: str,
    question: str,
    attempt: int,
) -> str:
    """Ask the LLM to fix a broken analysis script."""
    import re
    from agents.agent import get_llm
    llm = get_llm(timeout_seconds=90.0)
    prompt = textwrap.dedent(f"""
        You are an expert Python data analyst and bug-fixer.

        ## Task
        Fix the analysis script below (attempt {attempt}).
        Return ONLY the corrected Python script — no prose, no markdown fences.

        ## Original question being answered
        {question}

        ## Rules
        - The script receives the full DataFrame as `df` (already loaded, do NOT reload it).
        - Available imports: pd, np, json, os (all pre-imported — do NOT import them again).
        - The script MUST assign its final answer to `RESULT` — a JSON-serialisable dict or list.
        - Use float()/int() to convert numpy scalars. No DataFrames in RESULT (use .to_dict()).
        - You may define helper functions and chain them.
        - Do NOT use plt.show() or any GUI calls.

        ## Dataset info
        {df_info}

        ## Error / Traceback
        ```
        {error}
        ```

        ## Broken script
        ```python
        {script}
        ```

        ## Fixed script:
    """).strip()
    from langchain_core.messages import HumanMessage
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    except Exception:
        return script
    raw = re.sub(r"^```(?:python)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def _df_info(df: pd.DataFrame) -> str:
    """Compact dataset description for the LLM."""
    col_dtypes = ", ".join(f"{c}:{str(t)}" for c, t in df.dtypes.items())
    return (
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Columns+dtypes: {col_dtypes}\n"
        f"Numeric columns: {list(df.select_dtypes(include='number').columns)}\n"
        f"Sample (first 3 rows): {df.head(3).to_dict(orient='records')}"
    )


def test_and_fix_script(
    script: str,
    df: pd.DataFrame,
    question: str = "",
    verbose: bool = True,
) -> CodeTestResult:
    """
    Test and auto-correct a full-dataframe analysis script.

    The script must:
      - Use `df` (the full DataFrame, already in scope)
      - Assign its final answer to `RESULT` (a JSON-serialisable dict or list)

    Returns a CodeTestResult with:
      - success: bool
      - final_code: the corrected script
      - result: the value of RESULT after execution
      - error: last error if all attempts failed
    """
    df_info_str = _df_info(df)
    current_script = script
    test_log: List[Dict[str, Any]] = []

    for attempt in range(1, MAX_FIX_ATTEMPTS + 2):
        if verbose:
            print(f"\n{'─'*60}\n🧪  Script Tester — attempt {attempt}/{MAX_FIX_ATTEMPTS + 1}")

        # Static check
        static_err = _script_syntax_check(current_script)
        if static_err:
            test_log.append({"attempt": attempt, "phase": "static", "error": static_err})
            if verbose:
                print(f"  ❌ Syntax error: {static_err}")
            if attempt > MAX_FIX_ATTEMPTS:
                break
            current_script = _llm_fix_script(current_script, static_err, df_info_str, question, attempt)
            continue

        # Runtime test
        ok, result, err_or_out = _script_runtime_test(current_script, df)
        if ok:
            test_log.append({"attempt": attempt, "phase": "runtime", "success": True})
            if verbose:
                print(f"  ✅ Script passed on attempt {attempt}")
                print(f"     RESULT preview: {str(result)[:300]}")
            return CodeTestResult(success=True, final_code=current_script, result=result,
                                  attempts=attempt, test_log=test_log)
        else:
            test_log.append({"attempt": attempt, "phase": "runtime", "success": False, "error": err_or_out})
            if verbose:
                print(f"  ❌ Runtime error:\n{textwrap.indent(err_or_out[:500], '     ')}")
            if attempt > MAX_FIX_ATTEMPTS:
                break
            current_script = _llm_fix_script(current_script, err_or_out, df_info_str, question, attempt)

    last_error = test_log[-1].get("error", "Unknown error after all attempts.")
    if verbose:
        print(f"  ⚠️  Script tester gave up after {MAX_FIX_ATTEMPTS} fix attempts.")
    return CodeTestResult(success=False, final_code=current_script, error=last_error,
                          attempts=MAX_FIX_ATTEMPTS, test_log=test_log)


# ---------------------------------------------------------------------------
# LangGraph tool wrapper (legacy)
# ---------------------------------------------------------------------------

def test_generated_function_in_pipeline(
    function_name: str,
    code: str,
    target_column: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    result = test_and_fix_function(function_name, code, target_column, df, verbose=True)
    return {
        "tested": True,
        "success": result.success,
        "final_code": result.final_code,
        "original_code_changed": result.final_code != code,
        "fix_attempts": result.attempts,
        "result": result.result,
        "error": result.error,
        "test_log_summary": [
            {"attempt": e["attempt"], "phase": e["phase"], "ok": e.get("success", False)}
            for e in result.test_log
        ],
    }
