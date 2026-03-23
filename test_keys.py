import os
import requests
import json
from dotenv import load_dotenv

load_dotenv(override=True)

azure_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

# Create custom session to bypass proxy intercepts
session = requests.Session()
session.trust_env = False
session.verify = False

print("--- AZURE OPENAI TEST ---")
azure_public = "https://dna-dca-km-dev-oai-gpt4o.openai.azure.com/openai/models?api-version=2023-05-15"
try:
    resp = session.get(azure_public, headers={"api-key": azure_key}, timeout=5)
    print(f"Public Endpoint (dns-dca-km) Status: {resp.status_code}")
    if resp.status_code != 200:
        try:
           print(resp.json())
        except:
           print("Response not JSON, likely Zscaler block page.")
except Exception as e:
    print(f"Public Endpoint Error: {e}")

azure_internal = "http://10.18.30.20/openai/models?api-version=2023-05-15"
try:
    resp = session.get(azure_internal, headers={"api-key": azure_key}, timeout=5)
    print(f"Internal IP Endpoint (10.18.30.20) Status: {resp.status_code}")
    if resp.status_code != 200:
        try:
           print(resp.json())
        except:
           print("Response not JSON, likely Zscaler block page.")
except Exception as e:
    print(f"Internal IP Endpoint Error: {e}")

print("\n--- NATIVE OPENAI TEST ---")
openai_url = "https://api.openai.com/v1/models"
try:
    resp = session.get(openai_url, headers={"Authorization": f"Bearer {openai_key}"}, timeout=5)
    print(f"api.openai.com Status: {resp.status_code}")
    if resp.status_code != 200:
        try:
           print(resp.json())
        except:
           print("Response not JSON.")
except Exception as e:
    print(f"Native OpenAI Error: {e}")
