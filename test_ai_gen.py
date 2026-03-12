import requests
import os
from dotenv import load_dotenv
import time

load_dotenv(override=True)

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

def test_hf(prompt):
    print(f"Testing HF with prompt: {prompt}")
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=90)
        print(f"HF Status: {response.status_code}")
        if response.status_code == 200:
            print("HF SUCCESS")
            return True
        else:
            print(f"HF Error: {response.text}")
    except Exception as e:
        print(f"HF Exception: {e}")
    return False

def test_pollinations(prompt):
    print(f"Testing Pollinations with prompt: {prompt}")
    from urllib.parse import quote
    encoded = quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?nologo=true&seed={int(time.time())}"
    try:
        response = requests.get(url, timeout=30)
        print(f"Pollinations Status: {response.status_code}")
        if response.status_code == 200:
            print("Pollinations SUCCESS")
            return True
    except Exception as e:
        print(f"Pollinations Exception: {e}")
    return False

if __name__ == "__main__":
    prompt_safe = "3d animation style, cute character, magical background, cinematic lighting"
    print("--- Testing Safe Prompt ---")
    test_hf(prompt_safe)
    test_pollinations(prompt_safe)
