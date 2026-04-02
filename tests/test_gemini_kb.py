"""
Standalone execution of Gemini Deep Research Knowledge Builder.
Run this directly from your IDE to test KB generation while Azure is offline.
"""
import os
import json
import requests
from dotenv import load_dotenv

def run_gemini_deep_search_kb():
    # Load Environment Variables
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY is not set in the .env file. Please add your key first.")
        return

    print("⏳ Loading dataset variables from VIL_P2_agentdata.xlsx...")
    try:
        import pandas as pd
        df = pd.read_excel("data/VIL_P2_agentdata.xlsx")
        col_list = ", ".join(df.columns.astype(str).tolist())
    except Exception as e:
        print(f"⚠️ Warning: Dataset not found: {e}")
        col_list = "Unknown Columns (Dataset not found)"
    
    print("⚙️ Constructing Expert Knowledge Builder Prompt...")
    
    # Exact Prompt Extracted from `agents/agent.py`
    prompt = f"""
I am providing you with some sample data of the manufacturing process of fibre from plant pulp(cellulose) that is followed in our plant for you to see which variables we are working with (column names):
{col_list}

I need you to prepare a knowledge base consisting of 4 sections -

1. Process understanding - The Process section must:
- explain the likely process flow and operational sequence implied by the variables
- describe how material, energy, or utility streams likely move through the system
- explain upstream/downstream relationships between variable groups
- identify likely control points, bottlenecks, recirculation loops, heat exchange stages, separation stages, reaction stages, or utility interactions if relevant
- explain what “normal operation” would mean in terms of variable patterns
- explain what abnormal patterns may indicate
- connect process behavior to the user context, not just generic plant theory
- help a downstream analyst understand the system holistically from a process perspective

2. Physics/Chemistry - should summarize the governing physical and chemical principles for the dataset.

The Physics/Chemistry section must:
- explain the physical laws, thermodynamics, fluid mechanics, heat transfer, mass transfer, reaction behavior, equilibrium behavior, or chemical interactions relevant to the variables
- include equations, proportionalities, or directional relationships wherever useful
- explain how variables such as temperature, pressure, flow, concentration, and level interact physically
- describe expected first-principles behavior, including nonlinearities, delays, coupling, and trade-offs
- explain what variable combinations may violate physical expectations and therefore signal bad data, fouling, sensor issues, leaks, instability, or process upset
- include engineering reasoning that can support validation, feature engineering, and anomaly detection
- avoid becoming a generic chemistry textbook; tie every major concept back to likely variable interpretation in this dataset

This section should read like a first-principles diagnostic reference for the actual data.

3. Equipment - should summarize the equipment-level interpretation for the dataset.

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

4. OEM - should summarize OEM/manual/specification style constraints relevant to the dataset.

The OEM section must:
- focus on specification-style guidance, design envelopes, operating windows, alarm/trip philosophies, maintenance recommendations, and vendor/manual style constraints
- include realistic types of OEM considerations such as allowable temperature limits, pressure ranges, flow operating windows, NPSH considerations, fouling margins, vibration thresholds, motor loading considerations, control valve sizing implications, exchanger approach temperatures, separator residence times, pump/compressor operating regions, and instrument accuracy/response limitations where relevant
- clearly distinguish between hard OEM limits, common industry guidance, and inferred good engineering practice
- state when exact OEM values are equipment-specific and should be validated against actual manuals
- provide the analyst with actionable “do not over-interpret beyond this limit” style cautionary guidance
- help downstream users understand what kinds of boundaries, thresholds, and reliability limits matter when analyzing the dataset

This section should read like a vendor/OEM/specification-oriented engineering reference, even when exact vendor data is not available.

Make sure whatever you prepare is also relevant to the variable names in the excel since thats the data we are working with.

--------------------------------------------------
OUTPUT FORMAT EXAMPLE STRUCTURE
--------------------------------------------------
You must output exactly 4 JSON objects inside a single JSON array matching this structure:
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

    print("🚀 Firing Request to Gemini Deep Research API (gemini-2.5-flash - Free Tier Bypass)...")
    print("   (This will take 1-3 minutes because of the massive 900+ word requirement per section!)")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Natively enabling the API Tool parameter required to activate Gemini Deep Research ('googleSearch' is the JSON key required by the backend)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"googleSearch": {}}],
        "generationConfig": {
            "temperature": 0.2
        }
    }
    
    try:
        # Increase timeout drastically for heavy generation loops
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        
        if response.status_code == 200:
            resp_data = response.json()
            try:
                # Extract the text content from the Gemini response structure
                generated_text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
                print("\n✅ MISSION ACCOMPLISHED! Gemini generated the following Knowledge Base:")
                print("="*80)
                print(generated_text)
                print("="*80)
                
                # Check for Search Grounding Attribution (Deep Research Evidence)
                if "groundingMetadata" in resp_data["candidates"][0]:
                    print("\n🔍 Deep Research Sources Utilized by Gemini:")
                    sources = resp_data["candidates"][0]["groundingMetadata"].get("groundingChunks", [])
                    for s in sources:
                        if "web" in s:
                            print(f"- {s['web'].get('title', 'Unknown Title')} ({s['web'].get('uri', '')})")
                
                # Optionally dump directly to a file for easy reading
                with open("tests/gemini_output.json", "w", encoding="utf-8") as out_f:
                    out_f.write(generated_text)
                print("\n💾 Output also securely saved to 'tests/gemini_output.json'!")
                
            except KeyError as e:
                print(f"❌ Could not parse the JSON response properly. Missing Key: {e}")
                print("Raw Response:", resp_data)
        else:
            print(f"❌ API Error {response.status_code}: {response.text}")
    except requests.exceptions.Timeout:
        print("❌ Request Timed Out. Building a massive 4-part JSON with Deep Research takes several minutes. Please increase timeout!")
    except Exception as e:
        print(f"❌ Local Execution Error: {str(e)}")

if __name__ == "__main__":
    run_gemini_deep_search_kb()
