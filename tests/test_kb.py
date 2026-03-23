import json
import re
import sys
import os

sys.path.append(r"c:\Users\aditya.sareen-st\Downloads\Data agent")

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage
from agents.agent import get_llm
from dotenv import load_dotenv

def test():
    load_dotenv()
    try:
        llm = get_llm()
        search = DuckDuckGoSearchRun()
        llm_with_tools = llm.bind_tools([search])
        
        user_context = "test pulp and fiber plant data"
        col_list = "temperature, flow_rate, ph, rpm"
        
        prompt = f"You are an Expert Domain Knowledge Base Builder. The user is analyzing a dataset with this context: '{user_context}'. " \
                 f"The dataset contains the following specific variables/columns: {col_list}. " \
                 "1. Use the duckduckgo_search tool extensively to find deep process, physics, equipment, and OEM limits tailored EXACTLY to these specific variables for this exact industry. " \
                 "2. Ask specific queries to the search engine. Once you have comprehensive facts, output exactly 4 JSON objects. " \
                 "The JSON array must contain exactly 4 objects. Categories MUST be exactly: 'Process', 'Physics/Chemistry', 'Equipment', 'OEM'. " \
                 "CRITICAL INSTRUCTION: The 'knowledge_text' for EACH of the 4 sections MUST be a highly detailed, comprehensive multi-paragraph document (at least 150 words per section). It must act as a definitive engineering referencing manual containing all relevant constraints, standard operating limits, formulas, and physics rules for that category. " \
                 "DO NOT WRAP in markdown blocks. Output only the raw JSON array."
        
        sys_msg = SystemMessage(content=prompt)
        agent_msgs = [sys_msg]
        for i in range(7):
            print(f"Turn {i}")
            response = llm_with_tools.invoke(agent_msgs)
            agent_msgs.append(response)
            
            if not getattr(response, "tool_calls", None):
                print("No more tool calls, exiting loop.")
                break
                
            for tc in response.tool_calls:
                print("Tool call:", tc["name"])
                if tc["name"] in ["duckduckgo_search", "duckduckgo_results_json"]: 
                    res = search.invoke(tc["args"])
                    agent_msgs.append({"role": "tool", "name": tc["name"], "content": str(res), "tool_call_id": tc["id"]})
                    
        final_text = agent_msgs[-1].content
        print("Final text length:", len(final_text))
        match = re.search(r'\[.*\]', final_text, re.DOTALL)
        if match:
            final_text = match.group(0)
        try:
            new_rules = json.loads(final_text)
            print("Success, rules:", len(new_rules))
        except Exception as e:
            print("Error parsing JSON:", e)
            print("Text was:\n", final_text)
    except Exception as overall_err:
        print("Crash before end:", overall_err)

if __name__ == "__main__":
    test()
