import streamlit as st
import pandas as pd
import io
import os
from dotenv import load_dotenv

from agent import build_data_quality_graph, AgentState

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Data Quality Agent",
    page_icon="🔍",
    layout="wide"
)

st.title("Data Quality Agent 🔍")
st.markdown("Upload your dataset to run an AI-powered data quality check. The agent will profile your data, check for logical inconsistencies, and provide a detailed report.")

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

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Helper to convert dataframe to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Load the data
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"File '{uploaded_file.name}' loaded successfully. Shape: {df.shape}")
        
        # Display a preview of the raw input data
        with st.expander("Preview Raw Data"):
            st.dataframe(df.head(10))
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    if st.button("Run Data Quality Analysis"):
        with st.spinner("Agent is analyzing the data (this may take a minute)..."):
            try:
                # Initialize Graph
                app = build_data_quality_graph()
                
                initial_state = AgentState(
                    df=df,
                    profile={},
                    messages=[],
                    issues=[],
                    bad_indices_per_column={},
                    report=""
                )
                
                # Run the Agent
                final_state = app.invoke(initial_state)
                report = final_state.get("report", "No report generated.")
                
                st.session_state["report"] = report
                st.session_state["annotated_df"] = df
                st.session_state["analysis_done"] = True
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

# If analysis is complete, show the report and download button
if st.session_state.get("analysis_done", False):
    st.header("Data Quality Report")
    st.markdown(st.session_state["report"])
    
    st.header("Analyzed Dataset Preview")
    st.dataframe(st.session_state["annotated_df"])
    
    csv_bytes = convert_df_to_csv(st.session_state["annotated_df"])
    st.download_button(
        label="Download Dataset (CSV)",
        data=csv_bytes,
        file_name="analyzed_data.csv",
        mime="text/csv"
    )
