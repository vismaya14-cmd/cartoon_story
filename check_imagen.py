import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("Checking available models for Image Generation...")
try:
    for m in genai.list_models():
        if 'generateContent' not in m.supported_generation_methods and 'predict' not in m.supported_generation_methods:
             print(f"Other model: {m.name} - Methods: {m.supported_generation_methods}")
        if 'imagen' in m.name.lower() or 'image' in m.name.lower():
             print(f"Potential Image Model: {m.name} - Methods: {m.supported_generation_methods}")
            
except Exception as e:
    print(f"Error: {e}")
