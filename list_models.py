import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print(f"Testing with API Key: {api_key if api_key else 'NONE'}")
try:
    models = genai.list_models()
    for m in models:
        print(f"{m.name} - {m.supported_generation_methods}")
except Exception as e:
    print(f"Error during list_models: {e}")
