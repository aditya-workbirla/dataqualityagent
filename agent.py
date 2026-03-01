import os
import pandas as pd
from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import operator
from data_profiler import profile_dataframe
from tools import pandas_query_tool, set_current_df

# Define the LangGraph State type
# This state dictionary is passed between every node in the graph.
class AgentState(TypedDict):
    df: pd.DataFrame
    profile: Dict[str, Any]
    messages: Annotated[List[Any], operator.add]   # Use operator.add so messages append instead of overwrite
    issues: List[str]
    report: str

def format_gemini_messages(messages):
    """
    Cleans up the LangGraph message state to strictly comply with Gemini API's 
    roles and turn-taking requirements (e.g., no orphaned tool messages).
    """
    cleaned = []
    
    for m in messages:
        if isinstance(m, AIMessage):
            # fix empty content issue for gemini tool calls
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

def profile_data_node(state: AgentState) -> AgentState:
    """
    Node 1: Data Profiling
    This node takes the raw dataframe, runs the basic statistical profiling function, 
    and saves the profile to the state to pass to the LLM later.
    """
    df = state["df"]
    profile = profile_dataframe(df)
    
    # Store df globally so the pandas query tool can access it when the LLM calls it
    set_current_df(df)
    
    return {"profile": profile}

def quality_analyst_node(state: AgentState) -> AgentState:
    """
    Node 2: LLM Analysis
    The core AI node. It reviews the data profile, actively reasons about logical constraints based 
    on column names, and identifies specific issues using the pandas query tool.
    """
    profile = state["profile"]
    messages = state.get("messages", [])
    df = state["df"]
    
    # Initialize the Google Gemini model and bind our custom Pandas query tool to it
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools([pandas_query_tool])
    
    system_prompt = f"""You are an expert Data Quality Analyst API. Your job is to analyze the profile of a dataset and find issues.
    You will be provided with a profile summary containing missing value percentages, repeating value percentages, and basic stats per column.
    
    You need to:
    1. Identify columns with high percentages (>20%) of missing values.
    2. Identify columns with high percentages (>50%) of repeating values.
    3. Reason about the physics and logic of the variables based on their names. 
        For example: 'Flow Rate' or 'Dosing Rate' cannot be negative. 'Age' must be between 0 and 120, etc.
    4. You MUST use the `pandas_query_tool` to check the dataset for rows violating these logical constraints.
        Provide a pandas query string to the tool representing the error case (e.g. "flow_rate < 0" or "Dosing_Rate < 0"). Notice if the columns have spaces, pandas usually expects backticks for spaces like "`Dosing Rate` < 0".
    
    If the tool returns matches, log them as a specific data quality issue.
    
    Do not stop until you have considered all columns and checked potential logical constraints.
    When you are done, summarize all issues clearly in your final response.
    
    Here is the data profile:
    {profile}
    """
    
    if not messages:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content="Please analyze the data profile and use your tools to find any logical inconsistencies. When you are done, list all identified data quality issues.")]
        
    cleaned_messages = format_gemini_messages(messages)
    response = llm_with_tools.invoke(cleaned_messages)
    
    return {"messages": [response]}

def tool_execution_node(state: AgentState) -> AgentState:
    """
    Node 3: Execute tools requested by the LLM.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    new_messages = []
    for tool_call in last_message.tool_calls:
        try:
            if tool_call["name"] == "pandas_query_tool":
                result = pandas_query_tool.invoke(tool_call)
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
                
    return {"messages": new_messages}

def should_continue(state: AgentState) -> str:
    """
    Edge condition to determine if we should loop back to the analyst or generate the report.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
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
    
    # Grab the final analysis from the LLM
    final_analysis = messages[-1].content
    
    # Re-initialize the LLM for the final generation task
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = f"""Based on the following data quality analysis findings, write a clear, professional Data Quality Report.
    The report should have sections for:
    1. Executive Summary
    2. Missing and Repeating Values
    3. Logical Inconsistencies and Invalid Values
    4. Recommendations
    
    Make it easy to read for a team member who just uploaded their file.

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
    workflow.add_node("profile_data", profile_data_node)
    workflow.add_node("quality_analyst", quality_analyst_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("generate_report", generate_report_node)
    
    # Define execution order
    workflow.set_entry_point("profile_data")
    workflow.add_edge("profile_data", "quality_analyst")
    
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
