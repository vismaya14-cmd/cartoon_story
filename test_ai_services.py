import requests
import os
from dotenv import load_dotenv
import urllib.parse

load_dotenv(override=True)

def test_pollinations():
    prompt = "Pixar style 3D animation, colorful cinematic lighting, expressive cartoon character, detailed animated movie style, masterpiece"
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?seed=42&nologo=true"
    print(f"Testing Pollinations: {url}")
    try:
        r = requests.get(url, timeout=30)
        print(f"Status: {r.status_code}")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
        if r.status_code == 200 and 'image' in r.headers.get('Content-Type', ''):
            with open('test_pollinations.jpg', 'wb') as f:
                f.write(r.content)
            print("Saved test_pollinations.jpg")
        else:
            print(f"Failed: {r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

def test_hf():
    key = os.getenv("HF_API_KEY")
    url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"inputs": "Pixar style 3D animation, colorful cinematic lighting, expressive cartoon character"}
    print(f"Testing HF: {url}")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pollinations()
    test_hf()
