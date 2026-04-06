"""
Code Testing Agent
==================
A self-contained agent that validates, tests, and auto-corrects Python
data-quality functions before they are saved to the database.

Flow
----
1. Receive (function_name, code, target_column, df)
2. Static analysis  — syntax check (compile), scope check
3. Runtime test     — exec + call on actual data, capture stdout/stderr/exceptions
4. If failure: LLM bugfix loop (up to MAX_FIX_ATTEMPTS)
   - Sends code + error + traceback to the LLM with a strict repair prompt
   - Re-tests the patched code
5. Return CodeTestResult dataclass with:
   - success: bool
   - final_code: str        (corrected code, or original if all attempts failed)
   - result: Any            (function return value on success)
   - error: str             (last error if still failing)
   - attempts: int          (how many fix rounds happened)
   - test_log: list[dict]   (per-attempt record)
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
# Internal helpers
# ---------------------------------------------------------------------------

MAX_FIX_ATTEMPTS = 3


def _static_check(code: str, function_name: str) -> Optional[str]:
    """
    Compile + AST-walk the code.
    Returns an error string on failure, None on success.
    """
    try:
        tree = compile(code, "<generated>", "exec", flags=ast.PyCF_ONLY_AST)
    except SyntaxError as exc:
        return f"SyntaxError at line {exc.lineno}: {exc.msg}\n  → {exc.text}"

    # Confirm the function is actually defined
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


def _runtime_test(
    code: str,
    function_name: str,
    series: pd.Series,
) -> tuple[bool, Any, str]:
    """
    Execute `code`, call `function_name(series)`, capture result.
    Returns (ok, result, error_text).
    """
    local_scope: dict = {}
    exec_globals: dict = {
        "pd": pd,
        "np": __import__("numpy"),
        "Dict": Dict,
        "Any": Any,
        "List": List,
        "Optional": Optional,
    }

    # Redirect stdout/stderr to catch any print statements
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exec(code, exec_globals, local_scope)
        func = local_scope[function_name]
        result = func(series)
        captured_out = sys.stdout.getvalue()
        captured_err = sys.stderr.getvalue()
    except Exception:
        tb = traceback.format_exc()
        captured_out = sys.stdout.getvalue()
        captured_err = sys.stderr.getvalue()
        sys.stdout, sys.stderr = old_stdout, old_stderr
        combined = tb
        if captured_out:
            combined += f"\n[stdout]\n{captured_out}"
        if captured_err:
            combined += f"\n[stderr]\n{captured_err}"
        return False, None, combined
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    return True, result, ""


def _llm_fix(
    function_name: str,
    code: str,
    error: str,
    series_info: str,
    attempt: int,
) -> str:
    """
    Call the LLM with a tight repair prompt.
    Returns corrected code (just the function definition, no prose).
    """
    import re
    from agents.agent import get_llm  # avoid circular at module level

    llm = get_llm(timeout_seconds=60.0)

    prompt = textwrap.dedent(f"""
        You are an expert Python bug-fixer for data quality functions.

        ## Task
        The function `{function_name}` failed during testing (attempt {attempt}).
        Return ONLY the corrected Python function definition — no prose, no markdown fences.

        ## Requirements
        - Function signature: `def {function_name}(series: pd.Series) -> dict:`
        - Must return a plain Python `dict` (JSON-serialisable — no numpy scalars, no DataFrames).
        - Use `.item()` to convert numpy scalars: e.g. `int(val)`, `float(val)`.
        - No external imports beyond `pd`, `np`, `Dict`, `Any`, `List`, `Optional`
          (all pre-imported in exec scope).

        ## Dataset column info
        {series_info}

        ## Error / Traceback
        ```
        {error}
        ```

        ## Broken code
        ```python
        {code}
        ```

        ## Fixed code (output ONLY the function definition):
    """).strip()

    from langchain_core.messages import HumanMessage
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
    except Exception as exc:
        return code  # keep original if LLM fails

    # Strip any markdown fences the LLM might add despite instructions
    raw = re.sub(r"^```(?:python)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_and_fix_function(
    function_name: str,
    code: str,
    target_column: str,
    df: pd.DataFrame,
    verbose: bool = True,
) -> CodeTestResult:
    """
    Main entry point.  Tests `code` against `df[target_column]` and
    auto-corrects up to MAX_FIX_ATTEMPTS times via the LLM.

    Parameters
    ----------
    function_name:  Name of the Python function inside `code`.
    code:           Full Python source (must define `function_name`).
    target_column:  Column name in `df` to run the function on.
    df:             The live dataset DataFrame.
    verbose:        If True, prints progress to the server terminal.

    Returns
    -------
    CodeTestResult
    """
    if target_column not in df.columns:
        return CodeTestResult(
            success=False,
            final_code=code,
            error=f"Column '{target_column}' not found in dataset.  "
                  f"Available: {list(df.columns)}",
        )

    series = df[target_column]
    series_info = (
        f"Column: '{target_column}'  |  dtype: {series.dtype}  "
        f"|  n={len(series)}  |  sample: {series.dropna().head(5).tolist()}"
    )

    current_code = code
    test_log: List[Dict[str, Any]] = []

    for attempt in range(1, MAX_FIX_ATTEMPTS + 2):  # +1 so we test after final fix
        if verbose:
            sep = "─" * 60
            print(f"\n{sep}")
            print(f"🧪  Code Tester — attempt {attempt}/{MAX_FIX_ATTEMPTS + 1}  [{function_name}]")
            print(sep)

        # ── 1. Static check ──────────────────────────────────────────────
        static_err = _static_check(current_code, function_name)
        if static_err:
            entry = {"attempt": attempt, "phase": "static", "error": static_err}
            test_log.append(entry)
            if verbose:
                print(f"  ❌ Static check failed:\n{textwrap.indent(static_err, '     ')}")
            if attempt > MAX_FIX_ATTEMPTS:
                break
            current_code = _llm_fix(function_name, current_code, static_err, series_info, attempt)
            continue

        # ── 2. Runtime test ──────────────────────────────────────────────
        ok, result, runtime_err = _runtime_test(current_code, function_name, series)
        entry: Dict[str, Any] = {"attempt": attempt, "phase": "runtime"}

        if ok:
            entry["success"] = True
            entry["result"] = result
            test_log.append(entry)
            if verbose:
                print(f"  ✅ All tests passed on attempt {attempt}")
                print(f"     Result preview: {str(result)[:200]}")
            return CodeTestResult(
                success=True,
                final_code=current_code,
                result=result,
                attempts=attempt,
                test_log=test_log,
            )
        else:
            entry["success"] = False
            entry["error"] = runtime_err
            test_log.append(entry)
            if verbose:
                print(f"  ❌ Runtime error on attempt {attempt}:")
                print(textwrap.indent(runtime_err[:600], "     "))
            if attempt > MAX_FIX_ATTEMPTS:
                break
            current_code = _llm_fix(function_name, current_code, runtime_err, series_info, attempt)

    # All attempts exhausted
    last_error = test_log[-1].get("error", "Unknown error after all fix attempts.")
    if verbose:
        print(f"\n  ⚠️  Code Tester gave up after {MAX_FIX_ATTEMPTS} fix attempts.")
    return CodeTestResult(
        success=False,
        final_code=current_code,
        error=last_error,
        attempts=MAX_FIX_ATTEMPTS,
        test_log=test_log,
    )


# ---------------------------------------------------------------------------
# LangGraph tool wrapper  (used by tool_execution_node)
# ---------------------------------------------------------------------------

def test_generated_function_in_pipeline(
    function_name: str,
    code: str,
    target_column: str,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Thin wrapper around `test_and_fix_function` designed for use inside
    the LangGraph tool_execution_node.

    Returns a JSON-serialisable dict compatible with what `generate_and_test_custom_function`
    returns, so the analyst can read it identically.
    """
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
