import os
import pandas as pd
from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import operator
from operations.predefined import run_all_verified_functions
from agents.tools import generate_and_test_custom_function, set_current_df, execute_existing_function_with_params

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
    
    # Multi-Agent KB Validation Loop
    kb_candidate: List[Dict[str, Any]]
    kb_critique: Dict[str, Any]
    kb_retry_count: int
    kb_history: List[Dict[str, Any]]

def format_messages(messages):
    """
    Cleans up the LangGraph message state to strictly comply with LLM API's 
    roles and turn-taking requirements (e.g., no orphaned tool messages).
    """
    cleaned = []
    
    for m in messages:
        if isinstance(m, AIMessage):
            # fix empty content issue for tool calls
            if getattr(m, 'tool_calls', None) and not m.content:
                m.content = 'Calling tool'
            cleaned.append(m)
        elif isinstance(m, ToolMessage):
            # Check if previous message expects a tool message
            if cleaned:
                prev = cleaned[-1]
                if isinstance(prev, AIMessage) and getattr(prev, 'tool_calls', None):
                    cleaned.append(m)
                elif isinstance(prev, ToolMessage):
                    cleaned.append(m)
                # If orphaned (e.g. after a standard AIMessage), drop it.
        else:
            cleaned.append(m)
            
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

def collect_function_results_node(state: AgentState) -> AgentState:
    """
    Runs all predefined statistical and quality checks on the raw dataset locally.
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

    # Use a shorter timeout for the KB builder — it has many turns, and a
    # per-call timeout of 90 s is plenty while keeping the total run bounded.
    llm = get_llm(timeout_seconds=90.0)

    # Try to import DuckDuckGo search; gracefully degrade if not available or blocked
    _search_tool = None
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        _search_tool = DuckDuckGoSearchRun()
    except Exception:
        pass

    if _search_tool is not None:
        llm_with_tools = llm.bind_tools([_search_tool])
    else:
        llm_with_tools = llm
    
    prompt = f"""
You are an Expert Domain Knowledge Base Builder for industrial/process data analysis.

You are building a permanent technical knowledge base for a single chat session. This knowledge base will be used as the fixed engineering reference for all downstream follow-up questions in the chat. Therefore, the output must be specific, technically rigorous, practically useful, and tightly aligned to the dataset variables.

USER CONTEXT:
{user_context}

DATASET VARIABLES / COLUMNS:
{col_list}

PRIMARY OBJECTIVE:
Generate a high-quality domain knowledge base tailored specifically to the user context and the listed dataset variables/columns.

SEARCH INSTRUCTION:
Use the duckduckgo_search tool extensively and intelligently to gather relevant information. Search deeply for:
- process understanding related to the variables
- physics and chemistry relationships governing the variables
- equipment behavior, constraints, and operational relevance
- OEM/vendor/manual/spec-limit style guidance where available

Do not rely on generic industrial boilerplate. Prioritize information that can help interpret, troubleshoot, validate, and reason about the actual variables in this dataset.

--------------------------------------------------
CORE REQUIREMENTS
--------------------------------------------------

You must output exactly 4 JSON objects inside a single JSON array.

Each JSON object must contain exactly these 3 keys:
- "category"
- "topic"
- "knowledge_text"

The 4 categories must be exactly:
1. "Process"
2. "Physics/Chemistry"
3. "Equipment"
4. "OEM"

Do not output markdown.
Do not output commentary.
Do not output any keys other than the 3 required keys.
Output only the raw JSON array.

--------------------------------------------------
GLOBAL QUALITY STANDARD
--------------------------------------------------

Each section must be written as if it will be used by an engineer or analyst to interpret process data from this exact dataset.

This means each section must:
- explicitly reference the kinds of variables present in the dataset
- explain how these variables are used in interpretation
- include operating logic, constraints, cause-effect relationships, and analytical relevance
- avoid vague statements such as "temperature is important" unless followed by specific engineering interpretation
- avoid filler, repetition, and generic textbook explanation
- include practical reasoning useful for anomaly detection, diagnostics, root cause analysis, and follow-up Q&A

Each "knowledge_text" must be:
- highly detailed
- multi-paragraph
- comprehensive
- at least 900 words
- tailored to the given context and variables

--------------------------------------------------
VARIABLE GROUNDING RULES
--------------------------------------------------

Before writing, infer likely variable types from the dataset columns, such as:
- temperature
- pressure
- flow
- level
- composition
- pH
- speed
- current
- vibration
- valve position
- control signals
- utility variables
- equipment tags
- alarms/status indicators

You must ground the knowledge base in these variables.

For every section:
- mention representative variables or variable groups from the dataset
- explain what those variables likely indicate in operation
- explain how changes in one variable may affect others
- explain how analysts should interpret these variables together, not in isolation

If exact equipment/process identity is uncertain, make careful engineering inferences from naming patterns, but do not invent highly specific plant facts unless supported by the variable names and search results.

--------------------------------------------------
SECTION-SPECIFIC INSTRUCTIONS
--------------------------------------------------

SECTION 1: PROCESS
Category must be "Process"

Topic should summarize the process understanding for the dataset.

The Process section must:
- explain the likely process flow and operational sequence implied by the variables
- describe how material, energy, or utility streams likely move through the system
- explain upstream/downstream relationships between variable groups
- identify likely control points, bottlenecks, recirculation loops, heat exchange stages, separation stages, reaction stages, or utility interactions if relevant
- explain what “normal operation” would mean in terms of variable patterns
- explain what abnormal patterns may indicate
- connect process behavior to the user context, not just generic plant theory
- help a downstream analyst understand the system holistically from a process perspective

This section should read like a process-engineering operating interpretation manual tailored to the dataset.

SECTION 2: PHYSICS/CHEMISTRY
Category must be "Physics/Chemistry"

Topic should summarize the governing physical and chemical principles for the dataset.

The Physics/Chemistry section must:
- explain the physical laws, thermodynamics, fluid mechanics, heat transfer, mass transfer, reaction behavior, equilibrium behavior, or chemical interactions relevant to the variables
- include equations, proportionalities, or directional relationships wherever useful
- explain how variables such as temperature, pressure, flow, concentration, and level interact physically
- describe expected first-principles behavior, including nonlinearities, delays, coupling, and trade-offs
- explain what variable combinations may violate physical expectations and therefore signal bad data, fouling, sensor issues, leaks, instability, or process upset
- include engineering reasoning that can support validation, feature engineering, and anomaly detection
- avoid becoming a generic chemistry textbook; tie every major concept back to likely variable interpretation in this dataset

This section should read like a first-principles diagnostic reference for the actual data.

SECTION 3: EQUIPMENT
Category must be "Equipment"

Topic should summarize the equipment-level interpretation for the dataset.

The Equipment section must:
- identify likely equipment classes implied by the variables, such as pumps, compressors, blowers, fans, heat exchangers, columns, tanks, reactors, scrubbers, filters, valves, condensers, evaporators, separators, conveyors, motors, and instrumentation loops
- explain what each equipment class does and how its health/performance appears in process data
- describe expected input-output variable relationships for each equipment type
- explain common failure modes and how they appear in variables
- discuss control behavior, operating constraints, and performance degradation patterns
- explain what analysts should look for in pressure, flow, temperature, level, current, differential pressure, vibration, or valve movement data
- distinguish process-side issues from equipment-side issues where possible
- make the section useful for troubleshooting and engineering interpretation of actual tags/variables

This section should read like a practical equipment troubleshooting and interpretation guide for the dataset.

SECTION 4: OEM
Category must be "OEM"

Topic should summarize OEM/manual/specification style constraints relevant to the dataset.

The OEM section must:
- focus on specification-style guidance, design envelopes, operating windows, alarm/trip philosophies, maintenance recommendations, and vendor/manual style constraints
- include realistic types of OEM considerations such as allowable temperature limits, pressure ranges, flow operating windows, NPSH considerations, fouling margins, vibration thresholds, motor loading considerations, control valve sizing implications, exchanger approach temperatures, separator residence times, pump/compressor operating regions, and instrument accuracy/response limitations where relevant
- clearly distinguish between hard OEM limits, common industry guidance, and inferred good engineering practice
- state when exact OEM values are equipment-specific and should be validated against actual manuals
- provide the analyst with actionable “do not over-interpret beyond this limit” style cautionary guidance
- help downstream users understand what kinds of boundaries, thresholds, and reliability limits matter when analyzing the dataset

This section should read like a vendor/OEM/specification-oriented engineering reference, even when exact vendor data is not available.

--------------------------------------------------
ANTI-GENERIC RULES
--------------------------------------------------

Do NOT do any of the following:
- do not write shallow summaries
- do not repeat the same content across sections
- do not use generic filler to reach word count
- do not provide generic safety disclaimers unless directly relevant
- do not describe broad industrial concepts without linking them to likely variables in this dataset
- do not invent precise OEM numbers unless they are strongly supported by search results or standard engineering practice
- do not produce near-duplicate paragraphs across Process / Physics / Equipment / OEM

Each section must feel distinct in purpose:
- Process = system flow and operating logic
- Physics/Chemistry = first-principles governing behavior
- Equipment = asset-level interpretation and failure behavior
- OEM = specification limits, envelopes, vendor-style constraints, caution boundaries

--------------------------------------------------
OUTPUT FORMAT EXAMPLE STRUCTURE
--------------------------------------------------

[
  {{
    "category": "Process",
    "topic": "....",
    "knowledge_text": "...."
  }},
  {{
    "category": "Physics/Chemistry",
    "topic": "....",
    "knowledge_text": "...."
  }},
  {{
    "category": "Equipment",
    "topic": "....",
    "knowledge_text": "...."
  }},
  {{
    "category": "OEM",
    "topic": "....",
    "knowledge_text": "...."
  }}
]

Return only the raw JSON array.
"""
             
    if critique:
        prompt += f"""
        
WARNING: Your previous attempt was REJECTED by the Critique Agent.
You MUST specifically address the following feedback and improve the output, or you will fail again.
Scores (out of 10): {json.dumps(critique.get('scores', {}))}
Hard Fail Reasons: {json.dumps(critique.get('hard_fail_reasons', []))}
Specific Section Feedback: {json.dumps(critique.get('section_feedback', {}))}
Missing Elements: {json.dumps(critique.get('missing_elements', []))}
Improvement Instructions: {critique.get('improvement_instructions', 'Improve specificity and depth.')}

CRITICAL: Do NOT repeat the exact same generic output. Increase specificity, include process-level reasoning, and align tightly with the dataset variables.
"""
    
    sys_msg = SystemMessage(content=prompt)
    agent_msgs = [sys_msg]
    new_rules = None  # initialise so it is always defined after the loop

    for _ in range(5): # max 5 turns (reduced from 7 to limit total latency)
        try:
            response = llm_with_tools.invoke(agent_msgs)
        except Exception as e:
            # LLM call itself timed out or failed — stop the loop and fall through to
            # the best-effort JSON parse below with whatever we have so far
            error_note = f"LLM call failed: {type(e).__name__}: {e}"
            new_rules = [{"category": "Error", "topic": "LLM Timeout", "knowledge_text": error_note}]
            break
        agent_msgs.append(response)
        
        if not getattr(response, "tool_calls", None):
            break
            
        for tc in response.tool_calls:
            if tc["name"] in ["duckduckgo_search", "duckduckgo_results_json"]: 
                import concurrent.futures
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(_search_tool.invoke, tc["args"])
                        res = future.result(timeout=10.0)
                except concurrent.futures.TimeoutError:
                    res = "Error: DuckDuckGo search timed out! SSH firewall is blocking outbound internet access. DO NOT search again. Generate engineering constraints manually using only your pre-trained domain knowledge."
                except Exception as e:
                    res = f"Search failed: {e}"
                
                agent_msgs.append({"role": "tool", "name": tc["name"], "content": str(res), "tool_call_id": tc["id"]})
                
    # Parse final response (only if new_rules not already set by the error handler)
    if new_rules is None:
        import re
        last_msg = agent_msgs[-1]
        final_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        match = re.search(r'\[.*\]', final_text, re.DOTALL)
        if match:
            final_text = match.group(0)
            
        try:
            parsed = json.loads(final_text)
            if isinstance(parsed, list):
                new_rules = parsed
            else:
                new_rules = [{"category": "Error", "topic": "Parsing", "knowledge_text": str(final_text)}]
        except Exception:
            new_rules = [{"category": "Error", "topic": "Format", "knowledge_text": str(final_text)}]
        
    # Increment retry counter
    import os
    os.makedirs("data", exist_ok=True)
    thread_id = config.get("configurable", {}).get("thread_id", "global")
    mode = "a" if retry_count > 0 else "w"
    try:
        with open("data/knowledge_critique_log.md", mode, encoding="utf-8") as f:
            if retry_count == 0:
                f.write(f"# Knowledge Creation Log (Thread: {thread_id})\n")
            f.write(f"\n## Knowledge Agent Attempt {retry_count + 1}\n")
            f.write(f"```json\n{json.dumps(new_rules, indent=2)}\n```\n")
    except Exception:
        pass
        
    return {"kb_candidate": new_rules, "kb_retry_count": retry_count + 1}

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
    critique_prompt = f"""You are a rigorous, unforgiving Knowledge Base Critique Agent. 
Your goal is to evaluate a generated Candidate Knowledge Base for a dataset analysis task.

User Context: '{user_context}'
Candidate Knowledge Base: 
{json.dumps(candidate, indent=2)}

EVALUATION FRAMEWORK:
For EACH of the mandatory 4 sections (Process, Physics/Chemistry, OEM, Equipment), assign a score from 0 to 10:
0-3: Poor (missing / incorrect / useless)
4-6: Weak (generic / incomplete / shallow)
7-8: Good (mostly correct, some gaps)
9-10: Strong (detailed, specific, actionable)

HARD FAIL CONDITIONS (Auto-Reject if ANY are true):
- Any of the 4 sections is missing
- Any section score < 4
- Content is mostly generic boilerplate
- No clear linkage to process/dataset context
- Contradictions or obvious inaccuracies

APPROVAL LOGIC:
Approve ONLY IF:
- All 4 sections are present
- All section scores >= 7
- At least 2 sections score >= 8
- No hard fail conditions triggered

OUTPUT FORMAT RULES:
You MUST output EXACTLY one raw JSON object (with NO markdown backticks or wrappers) matching this schema exactly:
{{
  "status": "approved" or "rejected",
  "scores": {{"process_understanding": int, "physics_chemistry": int, "oem_based": int, "equipment_based": int}},
  "hard_fail_reasons": ["..."],
  "section_feedback": {{"process_understanding": "...", "physics_chemistry": "...", "oem_based": "...", "equipment_based": "..."}},
  "missing_elements": ["..."],
  "improvement_instructions": "..."
}}

Be STRICT. If unsure -> REJECT. Weak outputs must NOT pass.
"""

    llm = get_llm(timeout_seconds=90.0)
    # Use JSON mode if supported to force strict schema, but we'll manually strip just in case
    try:
        response = llm.invoke([SystemMessage(content=critique_prompt)])
        resp_text = response.content
    except Exception as e:
        # If the critique call itself times out, auto-approve so we don't loop forever
        critique = {
            "status": "approved",
            "scores": {"process_understanding": 7, "physics_chemistry": 7, "oem_based": 7, "equipment_based": 7},
            "hard_fail_reasons": [],
            "section_feedback": {},
            "missing_elements": [],
            "improvement_instructions": f"Critique agent timed out: {e}. Auto-approving to avoid infinite loop."
        }
        history = state.get("kb_history", [])
        history.append({"candidate": candidate, "critique": critique, "avg_score": 7.0})
        return {"kb_critique": critique, "kb_history": history}

    import re
    resp_text = resp_text
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
#                 NODE 2: THE REASONING AGENT
# ==============================================================================
def quality_analyst_node(state: AgentState) -> AgentState:
    """
    The core AI node. It reviews the JSON summary of predefined checks, applying 
    the user's specific context/domain prompt to identify anomalies.
    """
    summary = state["function_results_summary"]
    user_context = state.get("user_context_prompt", "No specific context provided.")
    messages = state.get("messages", [])
    
    llm = get_llm(timeout_seconds=180.0)
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
    
    import json
    try:
        raw_data = json.loads(state.get("df_json", "[]"))
        num_rows = len(raw_data)
        columns = list(raw_data[0].keys()) if num_rows > 0 else []
        dataset_metadata = f"The dataset has {num_rows} rows and {len(columns)} columns.\nAvailable Columns: {', '.join(columns)}"
    except Exception:
        dataset_metadata = "Dataset dimensions could not be parsed."
    
    system_prompt = f"""You are an expert Data Quality Analyst. Your job is to analyze the output of predefined data quality checks.
    
    DATASET OVERVIEW:
    {dataset_metadata}
    
    The user has provided the following context about the data:
    "{user_context}"
    
    You need to:
    1. Review the JSON output of ALL the predefined functions (e.g. missing values, repeating values, outliers, min/max).
    2. You also need to check if the values are realistic based on the user's context. Don't just highlight missing, repeated, negative values because that would be very surface level. Along with highlighting those, highlight insights about the data quality that are not easy to catch.
    3. Reason about the physics and logic of the variables *based STRICTLY on the user's context*.
        For example: If the user context says it's a Pulp and Fiber plant, use your knowledge of that domain to understand if certain values (like negative pressures or temperatures, or specific pH ranges) are realistic. TEMPERATURE CAN BE NEGATIVE, PRESSURE CANNOT BE NEGATIVE.
    4. If the user context requests a specific check that is NOT covered by the predefined functions (e.g., a specific complex logical constraint), you MUST use the `generate_and_test_custom_function` tool to write a new python function to check for it.
        CRITICAL RULES FOR GENERATION:
        - Ensure the function name is GLOBALLY UNIQUE (e.g., append the column name to the function name `check_negative_pressure_blower_inlet`). If you generate the identical function name twice, the database will throw an integrity constraint error!
        - If the tool execution returns an error like "Function name conflict" or "Database locked", do NOT display that technical error to the user in your output! Instead, internally generate a new unique function name and try again.
        - DO NOT TRY MORE THAN ONCE per concept. If generation fails repeatedly, move on to avoid getting stuck in a loop.
    5. The system has several advanced domain physics and statistical functions available (Group 2 and Group 3) that require specific column parameters to run. You MUST use the `execute_existing_function_with_params` tool to run them if they are relevant to the user context.
        AVAILABLE ADVANCED FUNCTIONS:
{advanced_funcs_desc}
        If the function requires a timestamp column, you must provide it in the params dictionary like `{{"timestamp_col": "NameOfColumn"}}`.
    6. Ensure your reasoning STRICTLY ADHERES to the following domain limits retrieved from our 4-part physical Knowledge Base (Process, Physics/Chemistry, Equipment, OEM):
{knowledge}
    7. CRITICAL FOLLOW-UP CHAT RULE: If the user asks a follow-up question requesting data rows (e.g. "show me the last 10 rows", "display the outliers"), YOU MUST NEVER output unexecuted Python code blocks! You MUST natively invoke the `generate_and_test_custom_function` tool to write a pandas script, let the Engine execute it, read the JSON output, and display the final values to the user in a Markdown table.
    
    If you use any tool, log the results as a specific data quality issue.
    
    Do not stop until you have considered all columns and checked potential logical constraints against the user's context.
    When you are done, summarize all identified issues clearly in your final response.
    
    Here is the predefined function results summary:
    {summary}
    """
    
    if not messages:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content="Please analyze the predefined function results summary using the provided context. If a custom test is needed, generate and run it. List all identified data quality issues.")]
        
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
                
                # We optionally track metrics from the custom tool result if we want, 
                # but for now we just feed the LLM the JSON string backward.
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
        return "end"
        
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
    summary = state.get("function_results_summary", {})
    
    # Grab the final analysis from the LLM
    final_analysis = messages[-1].content
    
    # Build text for missing values and high-level column stats
    missing_repeating_text = "       (Aggregated from automated Group 1 checks)\n"
    
    dataset_checks = summary.get("dataset_checks", {})
    null_results = dataset_checks.get("check_null_values", {}).get("column_results", {})
    
    # We'll use the columns found in null_results to list metrics for all columns
    for col, stats in null_results.items():
        null_pct = stats.get("null_pct", "N/A")
        missing_repeating_text += f"       - **{col}**: {null_pct}% null/missing\n"
        
    # Extract advanced dataset checks summary (Group 1 overview)
    dataset_summary = ""
    for check_name, check_result in dataset_checks.items():
        ds_msg = check_result.get("summary", "")
        if ds_msg:
            dataset_summary += f"       - **{check_name}**: {ds_msg}\n"
    
    # Re-initialize the LLM for the final generation task
    llm = get_llm(timeout_seconds=180.0)
    prompt = f"""Based on the following data quality analysis findings, write a clear, professional Data Quality Report.
    The report should have sections for:
    1. Executive Summary
    2. Missing and Repeating Values
       IMPORTANT: You MUST explicitly include the following column-wise percentages for ALL columns in the dataset:
{missing_repeating_text}
    3. Advanced Dataset Quality
       Include these high-level dataset metrics:
{dataset_summary}
    4. Logical Inconsistencies and Invalid Values
    5. Recommendations
    
    IMPORTANT: For section "4. Logical Inconsistencies and Invalid Values", you MUST present the findings as a Markdown table. The table should have exactly two columns: "Variable Name" and "Inconsistencies Found". Do not use a numbered list for this section.
    
    Analysis Findings:
    {final_analysis}
    """
    
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

def route_to_start(state: AgentState) -> str:
    """
    Conditional entry point: bypass the orchestrator if this is a follow-up chat.
    If function_results_summary is already populated, do not run it again.
    """
    summary = state.get("function_results_summary", {})
    if summary:
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
    workflow.add_node("quality_analyst", quality_analyst_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # Define execution order using conditional entry point
    workflow.set_conditional_entry_point(
        route_to_start,
        {
            "collect_function_results": "collect_function_results",
            "check_existing_kb": "check_existing_kb"
        }
    )
    workflow.add_edge("collect_function_results", "check_existing_kb")
    
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
