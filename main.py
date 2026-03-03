import sys
import os
import pandas as pd
from dotenv import load_dotenv
from agent import build_data_quality_graph, AgentState

# Load environment variables from .env file
load_dotenv()

def load_data(file_path: str) -> pd.DataFrame:
    """Loads CSV or Excel data."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_data_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
        
    print(f"Loading data from {file_path}...")
    try:
        df = load_data(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
        
    print("Initializing Data Quality Agent...")
    
    # We need an API key setup to run langgraph
    use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
    if use_azure:
        if "AZURE_OPENAI_API_KEY" not in os.environ:
            print("Warning: AZURE_OPENAI_API_KEY environment variable not found.")
            print("Please set it in your .env to run the analysis with Azure OpenAI.")
            sys.exit(1)
    else:
        if "OPENAI_API_KEY" not in os.environ:
            print("Warning: OPENAI_API_KEY environment variable not found.")
            print("Please set it in your .env to run the analysis.")
            sys.exit(1)
        
    app = build_data_quality_graph()
    
    initial_state = AgentState(
        df=df,
        profile={},
        messages=[],
        issues=[],
        bad_indices_per_column={},
        report=""
    )
    
    print("Agent is analyzing the data (this may take a minute)...\n")
    
    try:
        # Run graph
        final_state = app.invoke(initial_state)
        report = final_state.get("report")
        
        print("===================================\n")
        print("DATA QUALITY FINAL REPORT")
        print("===================================\n")
        print(report)
        print("\n===================================")
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
