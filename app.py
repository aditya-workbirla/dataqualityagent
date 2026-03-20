import streamlit as st
import pandas as pd
import io
import os
from dotenv import load_dotenv

from agent import build_data_quality_graph, AgentState
import ui_components
import kg_builder
import streamlit.components.v1 as components
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import uuid

# Load environment variables
load_dotenv(override=True)

st.set_page_config(
    page_title="DataQA — Agentic Data Quality Platform",
    page_icon="🔍",
    layout="wide"
)

# Inject the custom CSS and glowing variables
st.markdown(ui_components.load_css("style.css"), unsafe_allow_html=True)

# Top Navigation Bar mimicking HTML
st.markdown(ui_components.get_nav_html(), unsafe_allow_html=True)

# Ensure API keys are set up
use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
api_key_set = False

if use_azure:
    if "AZURE_OPENAI_API_KEY" in os.environ:
        api_key_set = True
    else:
        st.error("Warning: `AZURE_OPENAI_API_KEY` is not set in your environment.")
else:
    if "OPENAI_API_KEY" in os.environ:
        api_key_set = True
    else:
        st.error("Warning: `OPENAI_API_KEY` is not set in your environment.")

if not api_key_set:
    st.info("Please set your API key in the `.env` file and restart the application.")
    st.stop()
    
# Initialize session state variables if they don't exist
if "report" not in st.session_state:
    st.session_state["report"] = None
if "annotated_df" not in st.session_state:
    st.session_state["annotated_df"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = None
if "kg_html" not in st.session_state:
    st.session_state["kg_html"] = None
if "checkpointer" not in st.session_state:
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # Ensures the checkpoints tables are created
    st.session_state["checkpointer"] = saver
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# Sidebar Session History
with st.sidebar:
    st.header("🗄️ Session History")
    # Fetch threads from sqlite checkpointer
    try:
        with sqlite3.connect("checkpoints.db", timeout=5.0) as cp_conn:
            cp_cursor = cp_conn.cursor()
            cp_cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            threads = [row[0] for row in cp_cursor.fetchall() if row[0] != "main_thread"]
    except Exception:
        threads = []
        
    if threads:
        thread_names = {"-- New Session --": "-- New Session --"}
        try:
            app_for_history = build_data_quality_graph(checkpointer=st.session_state["checkpointer"])
            for t in threads:
                config = {"configurable": {"thread_id": t}}
                state = app_for_history.get_state(config).values
                if state and "user_context_prompt" in state:
                    prompt = state["user_context_prompt"].strip().replace('\n', ' ')
                    name = (prompt[:35] + "...") if len(prompt) > 35 else prompt
                    if not name:
                         name = "Empty Prompt"
                    # Add short UUID tag to differentiate identical prompts
                    thread_names[t] = f"{name} ({t[:4]})"
                else:
                    thread_names[t] = t
        except Exception:
            for t in threads: thread_names[t] = t
            
        selected_thread = st.selectbox(
            "Resume Previous Session", 
            options=["-- New Session --"] + threads,
            format_func=lambda x: thread_names.get(x, x)
        )
        if selected_thread != "-- New Session --" and selected_thread != st.session_state.get("thread_id"):
            st.session_state["thread_id"] = selected_thread
            config = {"configurable": {"thread_id": selected_thread}}
            app = build_data_quality_graph(checkpointer=st.session_state["checkpointer"])
            saved_state = app.get_state(config).values
            
            if saved_state:
                st.session_state["report"] = saved_state.get("report", "No report generated.")
                try:
                    df_json_str = saved_state.get("df_json", "[]")
                    st.session_state["annotated_df"] = pd.read_json(io.StringIO(df_json_str))
                except Exception:
                    st.session_state["annotated_df"] = pd.DataFrame()
                st.session_state["analysis_done"] = True
                st.session_state["agent_state"] = saved_state
                
                chat_hist = []
                if "messages" in saved_state:
                    from langchain_core.messages import HumanMessage, AIMessage
                    for m in saved_state["messages"]:
                        if isinstance(m, HumanMessage):
                            chat_hist.append({"role": "user", "content": m.content})
                        elif isinstance(m, AIMessage) and m.content:
                            chat_hist.append({"role": "assistant", "content": m.content})
                st.session_state["chat_history"] = chat_hist
                
                # Rebuild Knowledge Graph since it's not natively in state
                if st.session_state["annotated_df"] is not None:
                    try:
                        st.session_state["kg_html"] = kg_builder.build_knowledge_graph(st.session_state["annotated_df"])
                    except Exception:
                        pass
                
                st.rerun()

tab1, tab2, tab3, tab4 = st.tabs(["New Analysis", "Functions DB", "Knowledge Graph", "Domain Knowledge Base"])

with tab1:
    st.markdown(ui_components.get_header_html(), unsafe_allow_html=True)
    
    st.markdown('<div class="card"><div class="card-label">Dataset Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop dataset here or click to browse", type=["csv", "xlsx", "xls", "parquet"], label_visibility="collapsed")
    st.markdown('</div><div class="spacer-20"></div>', unsafe_allow_html=True)
    
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.markdown(f"""
            <div class="file-preview visible" style="margin-top:-10px; margin-bottom: 20px;">
              <div class="file-icon">📊</div>
              <div class="file-info">
                <div class="file-name">{uploaded_file.name}</div>
                <div class="file-size">{len(df):,} rows · {len(df.columns)} columns</div>
              </div>
              <div class="badge ok">Ready</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
    
    st.markdown('<div class="card"><div class="card-label">Dataset Context + Requirements</div>', unsafe_allow_html=True)
    user_context = st.text_area(
        "Prompt", 
        value="General dataset.",
        placeholder="Describe your dataset and what it represents...\ne.g. 'This is a production log from a Kraft pulp and fiber processing plant...'",
        label_visibility="collapsed",
        height=120
    )
    st.markdown('</div><div class="spacer-20"></div>', unsafe_allow_html=True)
    
    run_button = st.button("▶ Run Quality Analysis", type="primary", use_container_width=False)

    if run_button and df is not None:
        with st.spinner("Agent is analyzing the data (this may take a minute)..."):
            try:
                # Generate a new unique thread for this analysis and wipe old chat history
                st.session_state["thread_id"] = str(uuid.uuid4())
                st.session_state["chat_history"] = []
                
                # Initialize Graph explicitly wrapping it in the memory saver checkpointer
                app = build_data_quality_graph(checkpointer=st.session_state["checkpointer"])
                config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
                
                initial_state = {
                    "df_json": df.to_json(orient="records"), "user_context_prompt": user_context,
                    "function_results_summary": {}, "messages": [],
                    "issues": [], "bad_indices_per_column": {}, "report": ""
                }
                
                # Run the Agent with thread config
                final_state = app.invoke(initial_state, config)
                st.session_state["report"] = final_state.get("report", "No report generated.")
                st.session_state["annotated_df"] = df
                st.session_state["agent_state"] = final_state
                
                # Build Knowledge Graph JSON and HTML
                try:
                    st.session_state["kg_html"] = kg_builder.build_knowledge_graph(df)
                except Exception as kg_err:
                    st.error(f"Error building knowledge graph: {kg_err}")
                    
                st.session_state["analysis_done"] = True
                st.rerun() # Refresh to show results
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

    # If analysis is complete, show the report
    if st.session_state.get("analysis_done", False) and st.session_state["agent_state"]:
        state = st.session_state["agent_state"]
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(ui_components.get_header_html("Analysis", "Report", "", "step 02 — quality report", ""), unsafe_allow_html=True)
        
        # Get actual issues count
        st.markdown("<div class='spacer-20'></div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card"><div class="card-label">Executive AI Report</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="analysis-box">{st.session_state["report"]}</div></div>', unsafe_allow_html=True)
        
        st.markdown("<div class='spacer-20'></div>", unsafe_allow_html=True)
        st.download_button(
            label="⬇ Export Cleaned Dataset (CSV)",
            data=convert_df_to_csv(st.session_state["annotated_df"]),
            file_name="analyzed_data.csv",
            mime="text/csv"
        )
        
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown(ui_components.get_header_html("Follow-Up", "Chat Interface", "", "step 03 — conversational query", ""), unsafe_allow_html=True)
        st.markdown("<div class='spacer-20'></div>", unsafe_allow_html=True)
        
        # Render the chat dialog
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Handle new prompts
        if prompt := st.chat_input("Ask a follow-up question about this dataset..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.spinner("Analyzing follow-up..."):
                from langchain_core.messages import HumanMessage
                app = build_data_quality_graph(checkpointer=st.session_state["checkpointer"])
                config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
                
                # Thanks to the conditional entry point, it bypasses the orchestrator node!
                new_state = app.invoke({"messages": [HumanMessage(content=prompt)]}, config)
                
                ai_response = new_state["messages"][-1].content
                st.session_state["chat_history"].append({"role": "assistant", "content": ai_response})
                st.rerun()

with tab2:
    st.header("Internal Functions Database")
    st.markdown("View all pre-defined and agent-generated data quality functions securely stored in `functions.db`.")
    
    try:
        import sqlite3
        with sqlite3.connect("functions.db", timeout=5.0) as conn:
            db_df = pd.read_sql_query("SELECT * FROM data_quality_functions ORDER BY approved_by_team DESC, created_at DESC", conn)
            
            # Convert boolean column to string for better display
            if 'approved_by_team' in db_df.columns:
                db_df['approved_by_team'] = db_df['approved_by_team'].apply(lambda x: "Yes" if x == 1 else "No (Quarantined)")
            
            # Map function groups to human-readable labels
            if 'function_group' in db_df.columns:
                group_map = {
                    1: "Group 1 (Dataset-Level)", 
                    2: "Group 2 (Metadata-Dependent)", 
                    3: "Group 3 (Domain Logic)", 
                    4: "Group 4 (AI Generated)"
                }
                db_df['function_group'] = db_df['function_group'].map(group_map).fillna("Unknown")
            
            # Display stats
            st.metric("Total Functions inside DB", len(db_df))
            
            # Full table display
            st.dataframe(
                db_df,
                use_container_width=True,
                hide_index=True
            )
    except Exception as e:
        st.error(f"Could not load database: {e}")

with tab3:
    st.header("Dataset Knowledge Graph")
    st.markdown("This interactive graph visualizes the strongest mathematical correlations (top 5 per feature) found across all numeric columns in your dataset. Hover over nodes to see their exact Min and Max values.")
    
    if st.session_state.get("kg_html"):
        components.html(st.session_state["kg_html"], height=650)
        
        try:
            with open("knowledge_graph.json", "rb") as f:
                kg_json = f.read()
            st.download_button(
                label="⬇ Download Knowledge Graph (JSON)", 
                data=kg_json, 
                file_name="knowledge_graph.json", 
                mime="application/json"
            )
        except Exception:
            pass
    else:
        st.info("No Knowledge Graph available yet. Please upload a dataset and run the Quality Analysis on the New Analysis tab.")

with tab4:
    st.header("Domain Knowledge Base")
    st.markdown("View all the Process, Physics, Equipment, and OEM knowledge constraints automatically injected into the Reasoning Agent prior to analysis.")
    
    try:
        import sqlite3
        with sqlite3.connect("knowledge.db", timeout=5.0) as conn:
            current_thread = st.session_state.get("thread_id", "")
            query = "SELECT category, topic, knowledge_text, updated_at FROM domain_knowledge WHERE thread_id = ? ORDER BY category, topic"
            kb_df = pd.read_sql_query(query, conn, params=(current_thread,))
            
            st.metric("Total Knowledge Entries", len(kb_df))
            st.dataframe(
                kb_df,
                use_container_width=True,
                hide_index=True
            )
    except Exception as e:
        st.error(f"Could not load knowledge database: {e}")
