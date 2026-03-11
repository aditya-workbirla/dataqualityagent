import os
import pandas as pd
from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import operator
from functions_db.predefined import run_all_verified_functions
from tools import generate_and_test_custom_function, set_current_df, execute_existing_function_with_params

# Define the LangGraph State type
# This state dictionary is passed between every node in the graph.
class AgentState(TypedDict):
    df: pd.DataFrame
    user_context_prompt: str
    function_results_summary: Dict[str, Any]
    messages: Annotated[List[Any], operator.add]   # Use operator.add so messages append instead of overwrite
    issues: List[str]
    bad_indices_per_column: dict
    report: str

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

def get_llm():
    import httpx
    use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
    
    # Bypass corporate SSL MITM inspection and broken Windows Registry proxies
    custom_client = httpx.Client(verify=False, trust_env=False)
    
    if use_azure:
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0,
            http_client=custom_client
        )
    return ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        http_client=custom_client
    )

def collect_function_results_node(state: AgentState) -> AgentState:
    """
    Node 1: Predefined Function Execution
    Runs all predefined statistical and quality checks on the raw dataset locally.
    """
    df = state["df"]
    summary = run_all_verified_functions(df)
    
    # Store df globally so the custom function generator can test it
    set_current_df(df)
    
    return {"function_results_summary": summary}

def quality_analyst_node(state: AgentState) -> AgentState:
    """
    Node 2: LLM Contextual Analysis
    The core AI node. It reviews the JSON summary of predefined checks, applying 
    the user's specific context/domain prompt to identify anomalies.
    """
    summary = state["function_results_summary"]
    user_context = state.get("user_context_prompt", "No specific context provided.")
    messages = state.get("messages", [])
    
    llm = get_llm()
    llm_with_tools = llm.bind_tools([generate_and_test_custom_function, execute_existing_function_with_params])
    
    # Fetch parameters for Group 2 and 3 functions so LLM knows what to call
    import sqlite3
    db_path = "functions.db"
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
    
    system_prompt = f"""You are an expert Data Quality Analyst. Your job is to analyze the output of predefined data quality checks.
    The user has provided the following context about the data:
    "{user_context}"
    
    You need to:
    1. Review the JSON output of the predefined functions (e.g. missing values, repeating values, outliers, min/max).
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
    
    If you use any tool, log the results as a specific data quality issue.
    
    Do not stop until you have considered all columns and checked potential logical constraints against the user's context.
    When you are done, summarize all identified issues clearly in your final response.
    
    Here is the predefined function results summary:
    {summary}
    """
    
    if not messages:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content="Please analyze the predefined function results summary using the provided context. If a custom test is needed, generate and run it. List all identified data quality issues.")]
        
    cleaned_messages = format_messages(messages)
    response = llm_with_tools.invoke(cleaned_messages)
    
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
    return "end"

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
    
    # Build text for missing and repeating percentages
    missing_repeating_text = ""
    for col, col_data in summary.items():
        if col == "dataset_checks":
            continue
        checks = col_data.get("checks", {})
        missing_pct = checks.get("check_missing_values", {}).get("missing_percentage", "N/A")
        repeating_pct = checks.get("check_repeating_values", {}).get("consecutive_repeating_percentage", "N/A")
        missing_repeating_text += f"       - **{col}**: {missing_pct}% missing, {repeating_pct}% repeating\n"
        
    # Extract advanced dataset checks summary
    dataset_summary = ""
    dataset_checks = summary.get("dataset_checks", {})
    for check_name, check_result in dataset_checks.items():
        ds_msg = check_result.get("summary", "")
        if ds_msg:
            dataset_summary += f"       - **{check_name}**: {ds_msg}\n"
    
    # Re-initialize the LLM for the final generation task
    llm = get_llm()
    prompt = f"""Based on the following data quality analysis findings, write a clear, professional Data Quality Report.
    The report should have sections for:
    1. Executive Summary
    2. Missing and Repeating Values
       IMPORTANT: You MUST explicitly include the following column-wise percentages for ALL columns in this section:
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
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"report": response.content}

def build_data_quality_graph() -> StateGraph:
    """
    Assembles and compiles the LangGraph workflow.
    """
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("collect_function_results", collect_function_results_node)
    workflow.add_node("quality_analyst", quality_analyst_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # Define execution order
    workflow.set_entry_point("collect_function_results")
    workflow.add_edge("collect_function_results", "quality_analyst")
    
    # Conditional edge: after Analyst, either run a tool or generate the final report
    workflow.add_conditional_edges(
        "quality_analyst",
        should_continue,
        {
            "continue": "tool_execution",
            "end": "generate_report"
        }
    )
    # After a tool runs, ALWAYS return to the Analyst so it can read the result
    workflow.add_edge("tool_execution", "quality_analyst")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()
