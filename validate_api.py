import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv

def validate_key():
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    
    print(f"DEBUG: Key found in .env starts with: '{api_key[:5]}...' and ends with '...{api_key[-5:]}'")
    print(f"DEBUG: Key length: {len(api_key)}")
    
    genai.configure(api_key=api_key)
    
    try:
        print("DEBUG: Attempting to list models as a validity check...")
        genai.list_models()
        print("SUCCESS: API Key is valid and functional!")
    except Exception as e:
        print(f"FAILURE: API Key validation failed.")
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    validate_key()
