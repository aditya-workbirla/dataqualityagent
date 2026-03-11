import pandas as pd
import json
import networkx as nx
from pyvis.network import Network
import os

def build_knowledge_graph(df: pd.DataFrame, json_path="knowledge_graph.json"):
    """
    Analyzes the dataframe, generates a Knowledge Graph JSON containing 
    min, max, and top 5 correlated variables for each column.
    Then builds an interactive PyVis network HTML string.
    """
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        return None
        
    kg_dict = {}
    corr_matrix = numeric_df.corr().abs()
    
    for col in numeric_df.columns:
        if numeric_df[col].isna().all():
            continue
            
        col_min = float(numeric_df[col].min())
        col_max = float(numeric_df[col].max())
        
        # Get top 5 correlated
        if col in corr_matrix.columns:
            corrs = corr_matrix[col].drop(index=[col]).dropna()
            top_5 = corrs.sort_values(ascending=False).head(5).index.tolist()
        else:
            top_5 = []
            
        kg_dict[col] = {
            "min": col_min,
            "max": col_max,
            "top_5_correlated": top_5
        }
        
    # Write the JSON as requested
    with open(json_path, "w") as f:
        json.dump(kg_dict, f, indent=4)
        
    # Build Network Graph
    net = Network(height="600px", width="100%", bgcolor="#080c10", font_color="#c9d8e8", directed=False)
    net.force_atlas_2based()
    
    # Add Nodes
    for col, info in kg_dict.items():
        title = f"Min: {info['min']}<br>Max: {info['max']}"
        net.add_node(
            n_id=col,
            label=col,
            title=title,
            color="#1a9fff",      # Streamlit accent blue
            size=20,
            borderWidth=2,
            borderWidthSelected=4
        )
        
    # Add Edges
    for col, info in kg_dict.items():
        for target in info["top_5_correlated"]:
            # Make sure target node actually exists before adding edge
            if target in kg_dict:
                net.add_edge(col, target, color="#4f6f8f", value=1, physics=True)
                
    # Enhancing physics and visual clarity
    net.set_options("""
    var options = {
      "edges": {
        "color": {
          "inherit": false
        },
        "smooth": {
          "type": "continuous",
          "forceDirection": "none"
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
                
    # Save physical html file and read it back as string (PyVis template handling)
    tmp_html = "tmp_kg.html"
    net.save_graph(tmp_html)
    with open(tmp_html, "r", encoding="utf-8") as f:
        html_str = f.read()
    
    # Optionally clean up the temp file
    if os.path.exists(tmp_html):
        os.remove(tmp_html)
        
    return html_str
