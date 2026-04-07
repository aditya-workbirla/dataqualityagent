import os
import pandas as pd
from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import operator
from operations.predefined import run_all_verified_functions
from agents.tools import generate_and_test_custom_function, set_current_df, execute_existing_function_with_params, run_analysis_script
from prompts.knowledge_agent_prompts import (
    get_process_prompt,
    get_physics_prompt,
    get_equipment_prompt,
    get_oem_prompt,
    get_critique_injection,
    get_critique_prompt,
)
from prompts.analyst_prompts import get_analyst_system_prompt
from prompts.report_prompts import get_report_prompt
from prompts.planner_prompts import get_planner_prompt
from agents.rag_kb import (
    build_kb_embeddings,
    build_kb_embeddings_from_db,
    embeddings_exist,
    retrieve_relevant_chunks,
)

# Define the LangGraph State type
# This state dictionary is passed between every node in the graph.
class AgentState(TypedDict):
    df_json: str
    user_context_prompt: str
    function_results_summary: Dict[str, Any]
    retrieved_knowledge: str
    messages: Annotated[List[Any], operator.add]   # Use operator.add so messages append instead of overwrite
    issues: List[str]
    bad_indices_per_column: dict
    report: str
    
    # Follow-up chat planner
    chat_plan: str          # Structured execution plan produced by the planner agent
    chat_mode: str          # KB_ONLY | DATA_ONLY | BOTH | CONVERSATIONAL
    rag_chunks: str         # Top-k retrieved KB chunks for the current follow-up query

    # Multi-Agent KB Validation Loop
    kb_candidate: List[Dict[str, Any]]
    kb_critique: Dict[str, Any]
    kb_retry_count: int
    kb_history: List[Dict[str, Any]]

def format_messages(messages):
    """
    Cleans up the LangGraph message state to strictly comply with LLM API's
    roles and turn-taking requirements:

    Rules enforced:
    1. Every AIMessage with tool_calls MUST be followed by one ToolMessage per
       tool_call_id before the next AIMessage.  If any tool_call_id is missing
       a response we synthesise a placeholder ToolMessage so the API never
       sees an un-answered tool_call.
    2. Orphaned ToolMessages (no preceding AIMessage with tool_calls) are dropped.
    3. AIMessage with tool_calls but empty content gets a filler string so the
       API doesn't reject it.
    """
    cleaned = []
    # Set of tool_call_ids that have been answered by a ToolMessage
    answered_ids: set = set()
    # tool_call_ids that are still outstanding (from the last AIMessage with tool_calls)
    outstanding_ids: set = set()

    def _flush_outstanding():
        """Insert synthetic ToolMessages for any un-answered tool_call_ids."""
        for tid in list(outstanding_ids):
            cleaned.append(
                ToolMessage(
                    content="(no result — tool call was not executed)",
                    tool_call_id=tid,
                )
            )
            answered_ids.add(tid)
        outstanding_ids.clear()

    for m in messages:
        if isinstance(m, AIMessage):
            # Before appending a new AIMessage, ensure previous tool_calls are answered
            if outstanding_ids:
                _flush_outstanding()
            # Fix empty content for tool-calling AIMessages
            if getattr(m, "tool_calls", None) and not m.content:
                m = m.copy(update={"content": "Calling tool"})
            cleaned.append(m)
            # Track which tool_call_ids this message expects responses for
            for tc in getattr(m, "tool_calls", []) or []:
                tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tid:
                    outstanding_ids.add(tid)

        elif isinstance(m, ToolMessage):
            tid = getattr(m, "tool_call_id", None)
            if tid and tid in outstanding_ids:
                cleaned.append(m)
                outstanding_ids.discard(tid)
                answered_ids.add(tid)
            # Orphaned ToolMessage (no matching outstanding tool_call_id) — drop it

        else:
            # SystemMessage / HumanMessage
            if outstanding_ids:
                _flush_outstanding()
            cleaned.append(m)

    # Final flush in case the list ends with un-answered tool calls
    if outstanding_ids:
        _flush_outstanding()

    return cleaned

def get_llm(timeout_seconds: float = 180.0):
    import httpx
    use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

    # connect=15s — enough to detect a dead firewall quickly.
    # read/write/pool = timeout_seconds — window for large LLM completions.
    connect_timeout = 15.0

    if use_azure:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        host_header = os.getenv("AZURE_OPENAI_HOST_HEADER", "")
        _is_private_ip = any(
            endpoint.replace("https://","").replace("http://","").startswith(pfx)
            for pfx in ("10.", "172.", "192.168.")
        )

        # Build default headers — when hitting a private IP endpoint, Azure requires
        # the real hostname in the Host header so it can route the request correctly.
        default_headers = {}
        if _is_private_ip and host_header:
            default_headers["Host"] = host_header

        # For private IP endpoints: bypass Zscaler/corporate proxy entirely.
        # For public endpoints:     use system proxy (trust_env=True).
        if _is_private_ip:
            async_client = httpx.AsyncClient(
                verify=False, trust_env=False,
                timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout),
                headers=default_headers,
            )
            sync_client = httpx.Client(
                verify=False, trust_env=False,
                timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout),
                headers=default_headers,
            )
        else:
            # Public Azure endpoint — let system proxy (Zscaler) handle routing
            async_client = httpx.AsyncClient(
                verify=False, trust_env=True,
                timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout),
            )
            sync_client = httpx.Client(
                verify=False, trust_env=True,
                timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout),
            )

        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0,
            max_retries=0,
            timeout=timeout_seconds,
            http_client=sync_client,
            http_async_client=async_client,
        )

    # OpenAI public endpoint
    async_client = httpx.AsyncClient(
        verify=False, trust_env=True,
        timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout),
    )
    sync_client = httpx.Client(
        verify=False, trust_env=True,
        timeout=httpx.Timeout(timeout_seconds, connect=connect_timeout),
    )
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_retries=0,
        timeout=timeout_seconds,
        http_client=sync_client,
        http_async_client=async_client,
    )

# ==============================================================================
#                 NODE 1: THE FUNCTION ORCHESTRATOR
# ==============================================================================
def sanitize_for_msgpack(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: sanitize_for_msgpack(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_msgpack(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_msgpack(obj.tolist())
    else:
        return obj


def _rehydrate_df_for_tools(state: AgentState) -> None:
    """
    Ensures _CURRENT_DF is populated from state["df_json"] so that tools
    (generate_and_test_custom_function, execute_existing_function_with_params)
    can access the dataframe even during follow-up chat turns where
    collect_function_results_node is NOT re-executed.

    Also re-hydrates execute_writer's df metadata so execute.py stays accurate.
    """
    import io
    df_json = state.get("df_json", "")
    if not df_json:
        return
    try:
        df = pd.read_json(io.StringIO(df_json))
        set_current_df(df)
        # Keep execute_writer in sync so it knows column names / shape
        try:
            from agents.execute_writer import _capture_df_meta
            _capture_df_meta(df)
        except Exception:
            pass
    except Exception as e:
        print(f"⚠️  _rehydrate_df_for_tools failed: {e}")

def collect_function_results_node(state: AgentState) -> AgentState:
    """
    Runs all predefined statistical and quality checks on the raw dataset locally.
    Also resets execute.py so it starts fresh for this analysis run.
    """
    import pandas as pd
    import io
    
    # Reconstruct DF from string to bypass msgpack serialization errors in SQL state
    df_json = state.get("df_json", "{}")
    if getattr(df_json, "startswith", lambda x: False)("{"):
         # In case they parse it directly as dict, handle list of dicts or normal JSON
         pass
    
    try:
        df = pd.read_json(io.StringIO(df_json))
    except Exception:
        df = pd.DataFrame() # Fallback
        
    summary = run_all_verified_functions(df)
    summary = sanitize_for_msgpack(summary)
    
    # Store df globally so the custom function generator can test it tools.py
    set_current_df(df)

    # ── Reset execute.py for this new analysis run ─────────────────────────
    try:
        from agents.execute_writer import reset_execute_file
        reset_execute_file(dataset_path="", df=df)
    except Exception as _ew_err:
        print(f"⚠️  execute.py reset failed (non-fatal): {_ew_err}")
    
    return {"function_results_summary": summary}

from langchain_core.runnables import RunnableConfig

# ==============================================================================
#                 NODE 1.5: THE KNOWLEDGE AGENT
# ==============================================================================
def check_existing_kb_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Checks if a finalized Knowledge Base already exists in the database for this chat.
    If so, it loads it and skips generation.
    """
    import sqlite3
    db_path = "database/app.db"
    thread_id = config.get("configurable", {}).get("thread_id", "global")
    retrieved_text = ""
    
    # We don't overwrite if it's already in state
    if state.get("retrieved_knowledge"):
        return state
        
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, topic, knowledge_text FROM domain_knowledge WHERE thread_id = ?", (thread_id,))
            results = cursor.fetchall()
            if results:
                for r in results:
                    retrieved_text += f"- [{r[0]}] {r[1]}: {r[2]}\\n"
    except Exception:
        pass
        
    if retrieved_text:
        return {"retrieved_knowledge": retrieved_text}
    return {}

def knowledge_agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Builds the KB using online search. If it receives critique feedback from a 
    previous iteration, it aggressively incorporates it to produce a stronger candidate.
    """
    import sqlite3
    import json
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_core.messages import SystemMessage
    
    # If we already have retrieved_knowledge, skip generation completely
    if state.get("retrieved_knowledge"):
        return state
        
    user_context = state.get("user_context_prompt", "")
    if not user_context.strip():
        return state
        
    retry_count = state.get("kb_retry_count", 0)
    critique = state.get("kb_critique", {})
    
    try:
        raw_data = json.loads(state.get("df_json", "[]"))
        columns = list(raw_data[0].keys()) if len(raw_data) > 0 else []
        col_list = ", ".join(columns)
    except Exception:
        col_list = "Unknown Columns"
        
    llm = get_llm()
    search = DuckDuckGoSearchRun()
    llm_with_tools = llm.bind_tools([search])
    
    # --- 1. SET UP THE 4 MODULAR PROMPTS (loaded from prompts/ package) ---
    import os

    def load_sample(filename: str) -> str:
        filepath = os.path.join("docs", filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return "No reference sample provided."

    process_sample  = load_sample("Process_sample.txt")
    physics_sample  = load_sample("Physics_Chemistry_sample.txt")
    equipment_sample = load_sample("Equipment_sample.txt")
    oem_sample      = load_sample("OEM_sample.txt")

    critique_injection = get_critique_injection(critique)

    # Bundle prompts — each builder function lives in prompts/knowledge_agent_prompts.py
    split_prompts = [
        ("Process",          get_process_prompt(user_context, col_list, process_sample)   + critique_injection),
        ("Physics/Chemistry", get_physics_prompt(user_context, col_list, physics_sample)  + critique_injection),
        ("Equipment",        get_equipment_prompt(user_context, col_list, equipment_sample) + critique_injection),
        ("OEM",              get_oem_prompt(user_context, col_list, oem_sample)           + critique_injection),
    ]

    import concurrent.futures
    import re
    
    def process_single_section(name: str, section_prompt: str) -> dict:
        sys_msg = SystemMessage(content=section_prompt)
        agent_msgs = [sys_msg]
        
        for _ in range(5):  # Max 5 turns for tool loop
            response = llm_with_tools.invoke(agent_msgs)
            agent_msgs.append(response)
            
            if not getattr(response, "tool_calls", None):
                break
                
            for tc in response.tool_calls:
                if tc["name"] in ["duckduckgo_search", "duckduckgo_results_json"]: 
                    try:
                        # Direct invoke since we are already inside a threaded executor
                        res = search.invoke(tc["args"])
                    except Exception as e:
                        res = f"Search failed: {{e}}. Generate constraints manually."
                    
                    agent_msgs.append({"role": "tool", "name": tc["name"], "content": str(res), "tool_call_id": tc["id"]})
                    
        # Extract individual JSON
        final_text = agent_msgs[-1].content
        match = re.search(r'\{.*\}', final_text, re.DOTALL)
        if match:
            final_text = match.group(0)
            
        try:
            return json.loads(final_text)
        except Exception:
            return {"category": name, "topic": "Format Error", "knowledge_text": str(final_text)}

    # Parallel Execution for Mega-Prompt Splitting (Massive performance speedup)
    final_kb = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_name = {executor.submit(process_single_section, name, p): name for name, p in split_prompts}
        for future in concurrent.futures.as_completed(future_to_name):
            try:
                res_json = future.result()
                final_kb.append(res_json)
            except Exception as e:
                name = future_to_name[future]
                final_kb.append({"category": name, "topic": "Execution Error", "knowledge_text": str(e)})

    # Enforce correct category sorting so it stays 1. Process 2. Physics 3. Equipment 4. OEM
    order_map = {"Process": 1, "Physics/Chemistry": 2, "Equipment": 3, "OEM": 4}
    final_kb.sort(key=lambda x: order_map.get(x.get("category", ""), 99))

    # Log and Return
    import os
    os.makedirs("data", exist_ok=True)
    thread_id = config.get("configurable", {}).get("thread_id", "global")
    mode = "a" if retry_count > 0 else "w"
    try:
        with open("data/knowledge_critique_log.md", mode, encoding="utf-8") as f:
            if retry_count == 0:
                f.write(f"# Knowledge Creation Log (Thread: {thread_id})\n")
            f.write(f"\n## Knowledge Agent Attempt {retry_count + 1} (Multi-Prompt Threaded)\n")
            f.write(f"```json\n{json.dumps(final_kb, indent=2)}\n```\n")
    except Exception:
        pass
        
    return {"kb_candidate": final_kb, "kb_retry_count": retry_count + 1}

def critique_agent_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Rigorous quality-gate that evaluates the kb_candidate section by section.
    """
    import json
    from langchain_core.messages import SystemMessage, HumanMessage
    
    # Bypass if already retrieved
    if state.get("retrieved_knowledge"):
        return state
        
    candidate = state.get("kb_candidate", [])
    if not candidate:
        return state
        
    user_context = state.get("user_context_prompt", "")
    critique_prompt = get_critique_prompt(user_context, candidate)

    llm = get_llm()
    # Use JSON mode if supported to force strict schema, but we'll manually strip just in case
    response = llm.invoke([SystemMessage(content=critique_prompt)])
    
    import re
    resp_text = response.content
    match = re.search(r'\{.*\}', resp_text, re.DOTALL)
    if match:
        resp_text = match.group(0)
        
    try:
        critique = json.loads(resp_text)
    except Exception as e:
        critique = {
            "status": "rejected",
            "scores": {"process_understanding": 0, "physics_chemistry": 0, "oem_based": 0, "equipment_based": 0},
            "hard_fail_reasons": [f"Critique Agent failed to output valid JSON: {str(e)}"],
            "improvement_instructions": "Format your generation strictly as JSON."
        }
    
    # Calculate average score to help select best fallback
    scores = critique.get("scores", {})
    avg_score = sum(scores.values()) / 4.0 if scores else 0.0
    
    # Append to history
    history = state.get("kb_history", [])
    history.append({
        "candidate": candidate,
        "critique": critique,
        "avg_score": avg_score
    })
    
    try:
        import os
        thread_id = config.get("configurable", {}).get("thread_id", "global")
        with open("data/knowledge_critique_log.md", "a", encoding="utf-8") as f:
            f.write(f"\n### Critique Agent Feedback\n")
            f.write(f"**Score Average:** {avg_score}/10\n")
            f.write(f"```json\n{json.dumps(critique, indent=2)}\n```\n")
    except Exception:
        pass
    
    return {"kb_critique": critique, "kb_history": history}


def finalize_kb_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Saves the approved (or best) candidate to the database permanently.
    """
    import sqlite3
    import datetime
    
    # Bypass if already retrieved
    if state.get("retrieved_knowledge"):
        return state
        
    db_path = "database/app.db"
    thread_id = config.get("configurable", {}).get("thread_id", "global")
    critique = state.get("kb_critique", {})
    
    # Determine the best candidate
    if critique.get("status") == "approved":
        final_candidate = state.get("kb_candidate", [])
    else:
        # Loop maxed out, find the highest scoring attempt
        history = state.get("kb_history", [])
        if history:
            best_attempt = max(history, key=lambda x: x.get("avg_score", 0))
            final_candidate = best_attempt.get("candidate", [])
        else:
            final_candidate = state.get("kb_candidate", [])
            
    # Save to SQLite
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            # Clear any older attempts just in case
            cursor.execute("DELETE FROM domain_knowledge WHERE thread_id = ?", (thread_id,))
            now = datetime.datetime.now().isoformat()
            
            for rule in final_candidate:
                cat = rule.get("category", "Unknown")
                top = rule.get("topic", "Unknown")
                text = rule.get("knowledge_text", "")
                
                # Make sure the category falls into our 4 accepted buckets if the LLM skewed it
                cat_lower = cat.lower()
                clean_cat = "Process"
                if "phys" in cat_lower or "chem" in cat_lower: clean_cat = "Physics/Chemistry"
                elif "equip" in cat_lower: clean_cat = "Equipment"
                elif "oem" in cat_lower: clean_cat = "OEM"
                
                cursor.execute('''
                INSERT INTO domain_knowledge (thread_id, category, topic, knowledge_text, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (thread_id, clean_cat, top, text, now, now))
            conn.commit()
    except Exception as e:
        print(f"Failed to finalize knowledge DB: {e}")

    # ── Build RAG embeddings for this thread ──────────────────────────────
    # Normalise final_candidate to use the cleaned category names
    kb_for_rag = []
    for rule in final_candidate:
        cat_lower = rule.get("category", "").lower()
        clean_cat = "Process"
        if "phys" in cat_lower or "chem" in cat_lower: clean_cat = "Physics/Chemistry"
        elif "equip" in cat_lower: clean_cat = "Equipment"
        elif "oem" in cat_lower: clean_cat = "OEM"
        kb_for_rag.append({
            "category": clean_cat,
            "topic": rule.get("topic", ""),
            "knowledge_text": rule.get("knowledge_text", ""),
        })
    try:
        build_kb_embeddings(thread_id, kb_for_rag)
    except Exception as e:
        print(f"⚠️  RAG embedding step failed (non-fatal): {e}")

    # Read it back instantly to hydrate the analyst retrieval slot
    retrieved_text = ""
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT category, topic, knowledge_text FROM domain_knowledge WHERE thread_id = ?", (thread_id,))
            results = cursor.fetchall()
            for r in results:
                retrieved_text += f"- [{r[0]}] {r[1]}: {r[2]}\\n"
    except Exception as e:
        retrieved_text = f"Could not load finalized knowledge base: {e}"

    return {"retrieved_knowledge": retrieved_text}


# ==============================================================================
#                 NODE 1.7: RAG RETRIEVAL
# ==============================================================================
def rag_retrieval_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Retrieves the most relevant KB chunks for the current follow-up question
    using vector similarity search.

    - Runs only during follow-up turns (state["report"] exists).
    - If embeddings haven't been built yet for this thread (e.g. session resume),
      triggers a build from the domain_knowledge table.
    - Stores the top-k chunk texts in state["rag_chunks"] for the analyst.
    """
    # Only run during follow-up turns
    if not state.get("report"):
        return {"rag_chunks": ""}

    # Skip RAG entirely when the planner determined no KB reasoning is needed
    chat_mode = state.get("chat_mode", "BOTH").upper()
    if chat_mode in ("DATA_ONLY", "CONVERSATIONAL"):
        print(f"⏭️  RAG retrieval skipped — mode is {chat_mode}")
        return {"rag_chunks": ""}

    thread_id = config.get("configurable", {}).get("thread_id", "global")

    # If embeddings don't exist yet (session resume), build them now
    if not embeddings_exist(thread_id):
        try:
            n = build_kb_embeddings_from_db(thread_id)
            if n == 0:
                return {"rag_chunks": ""}
        except Exception as e:
            print(f"⚠️  RAG build on resume failed: {e}")
            return {"rag_chunks": ""}

    # Find the latest user question
    messages = state.get("messages", [])
    query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            query = m.content
            break

    if not query:
        return {"rag_chunks": ""}

    # Also incorporate the planner's plan as additional query context
    plan = state.get("chat_plan", "")
    if plan:
        query = f"{query}\n\nExecution plan context: {plan[:500]}"

    try:
        chunks_text = retrieve_relevant_chunks(query, thread_id, top_k=6)
    except Exception as e:
        print(f"⚠️  RAG retrieval failed: {e}")
        chunks_text = ""

    return {"rag_chunks": chunks_text}


# ==============================================================================
#                 NODE 2: THE REASONING AGENT
# ==============================================================================
def quality_analyst_node(state: AgentState) -> AgentState:
    """
    The core AI node. It reviews the JSON summary of predefined checks, applying 
    the user's specific context/domain prompt to identify anomalies.
    """
    # ── Always ensure tools have access to the live dataframe ─────────────
    # collect_function_results_node is skipped on follow-up turns, so we must
    # re-hydrate _CURRENT_DF from the serialised df_json in state.
    _rehydrate_df_for_tools(state)

    summary = state["function_results_summary"]
    user_context = state.get("user_context_prompt", "No specific context provided.")
    messages = state.get("messages", [])
    
    llm = get_llm(timeout_seconds=180.0)

    # Gate tool binding on chat mode — skip function tools when only KB reasoning is needed
    chat_mode = state.get("chat_mode", "").upper()
    if chat_mode in ("KB_ONLY", "CONVERSATIONAL"):
        llm_with_tools = llm   # No tools — pure reasoning/KB answer
    elif state.get("report"):
        # Follow-up chat turn: use script-based full-df tool so LLM can write
        # multi-column, multi-step analysis scripts (RESULT = ... contract)
        llm_with_tools = llm.bind_tools([run_analysis_script, execute_existing_function_with_params])
    else:
        # Initial analysis run: use single-column function generator
        llm_with_tools = llm.bind_tools([generate_and_test_custom_function, execute_existing_function_with_params])
    
    # Fetch parameters for Group 2 and 3 functions so LLM knows what to call
    import sqlite3
    db_path = "database/app.db"
    advanced_funcs_desc = ""
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT function_name, function_description FROM data_quality_functions WHERE approved_by_team = 1 AND function_group IN (2, 3)")
            funcs = cursor.fetchall()
            if funcs:
                advanced_funcs_desc = "\n".join([f"        - {f[0]}: {f[1]}" for f in funcs])
    except Exception as e:
        advanced_funcs_desc = f"        - (Could not load from DB: {e})"
        
    knowledge = state.get("retrieved_knowledge", "No domain knowledge retrieved.")
    # For follow-up turns use RAG-retrieved chunks (more focused, higher accuracy).
    # For the initial analysis run, fall back to the full KB.
    rag_chunks = state.get("rag_chunks", "")
    if rag_chunks:
        knowledge = (
            "## Relevant Knowledge Base Sections (retrieved via RAG)\n\n"
            + rag_chunks
            + "\n\n---\n*(Only the most relevant sections are shown above. "
            "Full KB is available in the Domain Knowledge Base tab.)*"
        )
    
    import json
    try:
        raw_data = json.loads(state.get("df_json", "[]"))
        num_rows = len(raw_data)
        columns = list(raw_data[0].keys()) if num_rows > 0 else []
        dataset_metadata = f"The dataset has {num_rows} rows and {len(columns)} columns.\nAvailable Columns: {', '.join(columns)}"
    except Exception:
        dataset_metadata = "Dataset dimensions could not be parsed."
    
    system_prompt = get_analyst_system_prompt(
        dataset_metadata=dataset_metadata,
        user_context=user_context,
        advanced_funcs_desc=advanced_funcs_desc,
        knowledge=knowledge,
        summary=summary,
    )

    if not messages:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content="Please analyze the predefined function results summary using the provided context. If a custom test is needed, generate and run it. List all identified data quality issues.")]
    else:
        # Follow-up chat turn: inject both the planner's execution plan AND the
        # RAG-retrieved KB chunks as a single system context block.
        chat_plan  = state.get("chat_plan", "")
        rag_chunks = state.get("rag_chunks", "")

        injection_parts = []
        if chat_plan:
            injection_parts.append(
                "## 📋 Execution Plan (from Planner Agent)\n\n"
                "You MUST follow this plan step by step to answer the question.\n\n"
                + chat_plan
            )

        # ── Extract FUNCTION GAPS from plan and build a mandatory tool checklist ──
        import re as _re_plan
        gaps_section = ""
        gaps_match = _re_plan.search(
            r"###\s*FUNCTION GAPS\s*\n(.*?)(?=\n###|\Z)",
            chat_plan, _re_plan.DOTALL | _re_plan.IGNORECASE,
        )
        if gaps_match:
            gaps_raw = gaps_match.group(1).strip()
            # Only proceed if there are real gaps (not "None")
            if gaps_raw and "none" not in gaps_raw.lower():
                gaps_section = gaps_raw

        if gaps_section:
            # During follow-up turns (report exists) the only script tool available is
            # run_analysis_script.  Build a single-call instruction that folds ALL gaps
            # into ONE script (helper functions + combined RESULT dict).
            is_followup = bool(state.get("report"))

            gap_blocks = _re_plan.split(r"\n(?=- Gap\s+\d+)", gaps_section)

            if is_followup:
                # ── Script-based mandatory call (follow-up only) ──────────────
                fn_specs = []
                for idx, block in enumerate(gap_blocks, 1):
                    block = block.strip()
                    if not block:
                        continue
                    fn_match  = _re_plan.search(r"`([^`]+)`", block)
                    fn_name   = fn_match.group(1) if fn_match else f"gap_fn_{idx}"
                    col_match = _re_plan.search(r"Target column[:\s]+`([^`]+)`", block, _re_plan.IGNORECASE)
                    target_col = col_match.group(1) if col_match else "df.columns[0]"
                    ret_match  = _re_plan.search(r"Returns[:\s]+(.+)", block, _re_plan.IGNORECASE)
                    returns_hint = ret_match.group(1).strip() if ret_match else "dict of stats"
                    desc_match = _re_plan.search(r"`[^`]+`\s*[—–-]+\s*(.+)", block)
                    description = desc_match.group(1).strip() if desc_match else block.splitlines()[0]
                    fn_specs.append({
                        "name": fn_name,
                        "col": target_col,
                        "desc": description,
                        "returns": returns_hint,
                    })

                helper_lines = []
                result_keys  = []
                for spec in fn_specs:
                    helper_lines.append(
                        f'# {spec["desc"]}\n'
                        f'def {spec["name"]}(df):\n'
                        f'    # TODO: implement — target column: {spec["col"]}\n'
                        f'    # Must return: {spec["returns"]}\n'
                        f'    pass'
                    )
                    result_keys.append(f'    "{spec["name"]}": {spec["name"]}(df)')

                example_script = (
                    "\n\n".join(helper_lines)
                    + "\n\nRESULT = {\n"
                    + ",\n".join(result_keys)
                    + "\n}"
                )

                checklist_lines = [
                    "## 🔴 MANDATORY TOOL CALL — run_analysis_script",
                    "",
                    "The Planner identified computations that must be performed. "
                    "You MUST make exactly ONE call to `run_analysis_script` that "
                    "implements ALL of the logic below in a single script. "
                    "Do NOT call `generate_and_test_custom_function` — it is NOT available.",
                    "",
                    f"**Script must cover {len(fn_specs)} computation(s):**",
                ]
                for spec in fn_specs:
                    checklist_lines.append(f"  - `{spec['name']}`: {spec['desc']} (column: `{spec['col']}`)")
                checklist_lines += [
                    "",
                    "**Contract**: The script must end with `RESULT = {{...}}` "
                    "containing JSON-serialisable values for all computations.",
                    "",
                    "**Skeleton** (fill in the actual logic):",
                    "```python",
                    example_script,
                    "```",
                ]
                injection_parts.append("\n".join(checklist_lines))

            else:
                # ── Legacy single-column checklist (initial analysis only) ───
                checklist_lines = [
                    "## 🔴 MANDATORY TOOL CALLS — generate_new functions",
                    "",
                    "The Planner identified functions that DO NOT EXIST yet. "
                    "You MUST call `generate_and_test_custom_function` for EACH one below "
                    "BEFORE answering the question. Do NOT skip any.",
                    "",
                ]
                for idx, block in enumerate(gap_blocks, 1):
                    block = block.strip()
                    if not block:
                        continue
                    fn_match = _re_plan.search(r"`([^`]+)`", block)
                    fn_name  = fn_match.group(1) if fn_match else f"gap_function_{idx}"
                    col_match = _re_plan.search(r"Target column[:\s]+`([^`]+)`", block, _re_plan.IGNORECASE)
                    target_col = col_match.group(1) if col_match else "(see plan)"
                    ret_match  = _re_plan.search(r"Returns[:\s]+(.+)", block, _re_plan.IGNORECASE)
                    returns_hint = ret_match.group(1).strip() if ret_match else "dict of stats"
                    desc_match = _re_plan.search(r"`[^`]+`\s*[—–-]+\s*(.+)", block)
                    description = desc_match.group(1).strip() if desc_match else block.splitlines()[0]

                    checklist_lines += [
                        f"**Call {idx}: `{fn_name}`**",
                        f"- target_column: `{target_col}`",
                        f"- function_description: \"{description}\"",
                        f"- code: write `def {fn_name}(series: pd.Series) -> dict:` "
                          f"that returns `{{{returns_hint}}}`",
                        "",
                    ]
                injection_parts.append("\n".join(checklist_lines))

        if rag_chunks:
            injection_parts.append(
                "## 📚 Retrieved Domain Knowledge (RAG — most relevant sections)\n\n"
                "Use these sections as your authoritative reference for domain reasoning. "
                "Do NOT contradict them.\n\n"
                + rag_chunks
            )

        # Mode-specific instructions to keep the analyst focused
        mode_instruction = ""
        if chat_mode == "KB_ONLY":
            mode_instruction = (
                "## ⚠️ MODE: KB_ONLY\n"
                "Answer using ONLY the domain knowledge base sections above. "
                "Do NOT call any tools or run any functions. "
                "Do NOT compute statistics from data."
            )
        elif chat_mode == "DATA_ONLY":
            mode_instruction = (
                "## ⚠️ MODE: DATA_ONLY\n"
                "Answer using ONLY data computations (run functions / generate custom checks). "
                "Do NOT reference the knowledge base — focus purely on what the data shows."
            )
        elif chat_mode == "CONVERSATIONAL":
            mode_instruction = (
                "## ⚠️ MODE: CONVERSATIONAL\n"
                "Answer conversationally from the existing analysis context. "
                "Do NOT call any tools or retrieve any KB sections."
            )
        if mode_instruction:
            injection_parts.insert(0, mode_instruction)

        if injection_parts:
            injection_msg = SystemMessage(content="\n\n---\n\n".join(injection_parts))
            messages = list(messages)
            # Insert right before the last HumanMessage
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    messages.insert(i, injection_msg)
                    break
        
    cleaned_messages = format_messages(messages)
    try:
        response = llm_with_tools.invoke(cleaned_messages)
    except Exception as e:
        # Surface a clear, actionable error message instead of a cryptic LangGraph crash
        error_msg = str(e)
        if "timed out" in error_msg.lower() or "timeout" in error_msg.lower() or "connect" in error_msg.lower():
            user_msg = (
                f"⚠️ **LLM Connection Error**: The request to the AI endpoint timed out or was refused. "
                f"Please check that `AZURE_OPENAI_ENDPOINT` ({os.getenv('AZURE_OPENAI_ENDPOINT', 'not set')}) "
                f"is reachable from this server and that the API key is valid.\n\nTechnical detail: `{error_msg}`"
            )
        else:
            user_msg = f"⚠️ **LLM Error**: {error_msg}"
        from langchain_core.messages import AIMessage as _AIMessage
        return {"messages": [_AIMessage(content=user_msg)]}
    
    return {"messages": [response]}

def tool_execution_node(state: AgentState) -> AgentState:
    """
    Node 3: Execute tools requested by the LLM.
    """
    # Safety net: rehydrate df in case this worker process lost it
    _rehydrate_df_for_tools(state)

    messages = state["messages"]
    last_message = messages[-1]
    
    # Initialize bad_indices_per_column if not present
    bad_indices_per_column = state.get("bad_indices_per_column", {})
    
    new_messages = []
    for tool_call in last_message.tool_calls:
        try:
            if tool_call["name"] == "generate_and_test_custom_function":
                result = generate_and_test_custom_function.invoke(tool_call)
                new_messages.append({
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": result,
                    "tool_call_id": tool_call["id"]
                })
            elif tool_call["name"] == "execute_existing_function_with_params":
                result = execute_existing_function_with_params.invoke(tool_call)
                new_messages.append({
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": result,
                    "tool_call_id": tool_call["id"]
                })
            elif tool_call["name"] == "run_analysis_script":
                result = run_analysis_script.invoke(tool_call)
                new_messages.append({
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": result,
                    "tool_call_id": tool_call["id"]
                })
        except Exception as e:
                new_messages.append({
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": f"Error: {str(e)}",
                    "tool_call_id": tool_call["id"]
                })
                
    return {"messages": new_messages, "bad_indices_per_column": bad_indices_per_column}

def should_continue(state: AgentState) -> str:
    """
    Edge condition to determine if we should loop back to the analyst or generate the report.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Anti-loop measure
    tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage) or (isinstance(m, dict) and m.get("role") == "tool"))
    if tool_message_count > 10:
        return "chat_end" if state.get("report") else "generate_report"
        
    if getattr(last_message, "tool_calls", None):
        return "continue"
        
    # If a report already exists in state, this is a follow-up chat. Bypass report generation layer.
    if state.get("report"):
        return "chat_end"
        
    return "generate_report"

def generate_report_node(state: AgentState) -> AgentState:
    """
    Node 4: Final Summarization
    This node takes all the raw findings from the AI analyst (tool matches, missing stats, etc.) 
    and synthesizes them into a polished markdown report for the user.
    """
    messages = state["messages"]
    import json
    summary = state.get("function_results_summary", {})
    function_outputs = json.dumps(summary, indent=2)
    
    try:
        raw_data = json.loads(state.get("df_json", "[]"))
        cols = list(raw_data[0].keys()) if len(raw_data) > 0 else []
        col_list = "\\n".join([f"- {c}" for c in cols])
    except Exception:
        col_list = "Could not extract columns."
        
    knowledge_base = state.get("retrieved_knowledge", "No knowledge base retrieved.")
    final_analysis = messages[-1].content if messages else ""
    
    # Re-initialize the LLM for the final generation task
    llm = get_llm(timeout_seconds=180.0)
    prompt = get_report_prompt(
        col_list=col_list,
        function_outputs=function_outputs,
        knowledge_base=knowledge_base,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"report": response.content}
    except Exception as e:
        error_msg = str(e)
        fallback_report = (
            f"## ⚠️ Report Generation Error\n\n"
            f"The AI report generator could not complete the request due to an error:\n\n"
            f"**Error:** `{error_msg}`\n\n"
            f"**Raw Analysis Findings (pre-report):**\n\n{final_analysis}"
        )
        return {"report": fallback_report}

def chat_planner_node(state: AgentState) -> AgentState:
    """
    Follow-up Chat Planner Agent.

    Runs ONLY during follow-up chat turns (when state["report"] already exists).
    Reads the latest HumanMessage, inspects available DB functions, and produces
    a structured Execution Plan that is:
      - Printed to the server terminal (for developer visibility)
      - Stored in state["chat_plan"] so the quality_analyst_node can follow it
    """
    import json
    import sqlite3

    messages = state.get("messages", [])

    # Identify the latest human question
    user_question = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_question = m.content
            break
    if not user_question:
        return {"chat_plan": ""}

    # Build dataset metadata
    dataset_metadata = "Dataset dimensions unavailable."
    try:
        raw_data = json.loads(state.get("df_json", "[]"))
        num_rows = len(raw_data)
        columns = list(raw_data[0].keys()) if num_rows > 0 else []
        dataset_metadata = f"{num_rows} rows × {len(columns)} columns\nColumns: {', '.join(columns)}"
    except Exception:
        pass

    user_context = state.get("user_context_prompt", "No context provided.")

    # For the planner we pass a short KB summary (not the full text) —
    # the full retrieval happens in rag_retrieval_node which runs AFTER the planner.
    # We give the planner enough context to reason about what to look up.
    retrieved_knowledge = state.get("retrieved_knowledge", "")
    kb_summary_lines = []
    for line in retrieved_knowledge.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [") and "]" in stripped:
            # Extract just the "[Category] Topic:" part (first 120 chars)
            kb_summary_lines.append(stripped[:120])
    knowledge_summary = "\n".join(kb_summary_lines[:30]) if kb_summary_lines else "Knowledge base not yet loaded."

    # Fetch ALL functions from DB (all groups) for the planner to reason about
    db_path = "database/app.db"
    available_functions = "  (Could not load functions from database)"
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT function_name, function_description, function_group, approved_by_team "
                "FROM data_quality_functions ORDER BY function_group, function_name"
            )
            rows = cursor.fetchall()
            if rows:
                group_labels = {1: "Group 1 — Dataset-Level", 2: "Group 2 — Metadata-Dependent",
                                3: "Group 3 — Domain Logic", 4: "Group 4 — AI Generated"}
                lines = []
                current_group = None
                for fn_name, fn_desc, fn_group, approved in rows:
                    if fn_group != current_group:
                        current_group = fn_group
                        lines.append(f"\n**{group_labels.get(fn_group, f'Group {fn_group}')}**")
                    status = "✅ approved" if approved else "⚠️ quarantined"
                    lines.append(f"  - `{fn_name}` [{status}]: {fn_desc}")
                available_functions = "\n".join(lines)
    except Exception as e:
        available_functions = f"  (DB error: {e})"

    # Call the LLM planner
    llm = get_llm(timeout_seconds=60.0)
    planner_prompt = get_planner_prompt(
        user_question=user_question,
        dataset_metadata=dataset_metadata,
        user_context=user_context,
        available_functions=available_functions,
        knowledge_base=knowledge_summary,
    )

    try:
        response = llm.invoke([SystemMessage(content=planner_prompt)])
        plan = response.content
    except Exception as e:
        plan = f"⚠️ Planner agent failed: {e}\nProceeding without a structured plan."

    # ── Print plan to terminal ─────────────────────────────────────────────
    separator = "=" * 72
    print(f"\n{separator}")
    print(f"📋  CHAT PLANNER — execution plan for follow-up question")
    print(separator)
    print(f"❓  Question : {user_question}")
    print(separator)
    print(plan)
    print(f"{separator}\n")
    # ───────────────────────────────────────────────────────────────────────

    # ── Extract MODE from plan ─────────────────────────────────────────────
    import re as _re
    mode = "BOTH"  # safe default
    mode_match = _re.search(r"MODE\s*:\s*(KB_ONLY|DATA_ONLY|BOTH|CONVERSATIONAL)", plan, _re.IGNORECASE)
    if mode_match:
        mode = mode_match.group(1).upper()
    print(f"🎯  Planner MODE = {mode}\n")

    return {"chat_plan": plan, "chat_mode": mode}


def route_to_start(state: AgentState) -> str:
    """
    Conditional entry point:
    - First analysis run    → collect_function_results
    - Follow-up chat turn   → chat_planner  (new), then check_existing_kb → quality_analyst
    """
    summary = state.get("function_results_summary", {})
    if summary:
        # Follow-up: if there's already a report AND the last message is a HumanMessage,
        # run the planner first.
        if state.get("report"):
            return "chat_planner"
        return "check_existing_kb"
    return "collect_function_results"

def route_after_kb_check(state: AgentState) -> str:
    """
    If the KB was successfully loaded from the DB, skip the generation loop.
    """
    if state.get("retrieved_knowledge"):
        return "quality_analyst"
    return "knowledge_agent"

def route_critique(state: AgentState) -> str:
    """
    Evaluates the critique status. Routes to finalize_kb if approved or if retries maxed out.
    """
    if state.get("retrieved_knowledge"):
        # Edge case: already grabbed from DB before the loop
        return "quality_analyst"
        
    critique = state.get("kb_critique", {})
    if critique.get("status") == "approved":
        return "finalize_kb"
        
    retry_count = state.get("kb_retry_count", 0)
    if retry_count >= 3:
        return "finalize_kb"
        
    return "knowledge_agent"

def build_data_quality_graph(checkpointer=None) -> StateGraph:
    """
    Assembles and compiles the LangGraph workflow.
    """
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("collect_function_results", collect_function_results_node)
    workflow.add_node("check_existing_kb", check_existing_kb_node)
    workflow.add_node("knowledge_agent", knowledge_agent_node)
    workflow.add_node("critique_agent", critique_agent_node)
    workflow.add_node("finalize_kb", finalize_kb_node)
    workflow.add_node("chat_planner", chat_planner_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("quality_analyst", quality_analyst_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # Define execution order using conditional entry point
    workflow.set_conditional_entry_point(
        route_to_start,
        {
            "collect_function_results": "collect_function_results",
            "check_existing_kb": "check_existing_kb",
            "chat_planner": "chat_planner",
        }
    )
    workflow.add_edge("collect_function_results", "check_existing_kb")
    
    # After the planner finishes → RAG retrieval → quality analyst
    workflow.add_edge("chat_planner", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "quality_analyst")

    workflow.add_conditional_edges(
        "check_existing_kb",
        route_after_kb_check,
        {
            "quality_analyst": "quality_analyst",
            "knowledge_agent": "knowledge_agent"
        }
    )
    
    # Knowledge Generation Loop
    workflow.add_edge("knowledge_agent", "critique_agent")
    workflow.add_conditional_edges(
        "critique_agent",
        route_critique,
        {
            "knowledge_agent": "knowledge_agent",
            "finalize_kb": "finalize_kb",
            "quality_analyst": "quality_analyst"  # failsafe bypass
        }
    )
    workflow.add_edge("finalize_kb", "quality_analyst")
    
    # Conditional edge: after Analyst, either run a tool or generate the final report
    workflow.add_conditional_edges(
        "quality_analyst",
        should_continue,
        {
            "continue": "tool_execution",
            "generate_report": "generate_report",
            "chat_end": END
        }
    )
    # After a tool runs, ALWAYS return to the Analyst so it can read the result
    workflow.add_edge("tool_execution", "quality_analyst")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile(checkpointer=checkpointer)
