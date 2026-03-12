import requests
import urllib.parse
import time
import os

def test_pollinations(prompt):
    print(f"Testing Prompt: {prompt}")
    
    # Method 1: Current logic (with underscores)
    simple_prompt = prompt[:200].replace(" ", "_")
    encoded_prompt = urllib.parse.quote(simple_prompt)
    url_1 = f"https://image.pollinations.ai/prompt/{encoded_prompt}?nologo=true&width=1024&height=1024"
    
    # Method 2: Standard encoding (no underscores)
    encoded_prompt_2 = urllib.parse.quote(prompt[:200])
    url_2 = f"https://image.pollinations.ai/prompt/{encoded_prompt_2}?nologo=true"
    
    # Method 3: /p/ endpoint
    url_3 = f"https://pollinations.ai/p/{encoded_prompt_2}?width=1024&height=1024&model=flux"

    urls = [url_1, url_2, url_3]
    
    for i, url in enumerate(urls, 1):
        print(f"\nTrying URL {i}: {url}")
        try:
            r = requests.get(url, timeout=30)
            print(f"Status: {r.status_code}")
            print(f"Content-Type: {r.headers.get('Content-Type')}")
            print(f"Size: {len(r.content)}")
            if r.status_code == 200 and 'image' in r.headers.get('Content-Type', ''):
                out = f"debug_out_{i}.jpg"
                with open(out, "wb") as f:
                    f.write(r.content)
                print(f"SUCCESS! Saved to {out}")
            else:
                print(f"FAILED. First 100 bytes: {r.content[:100]}")
        except Exception as e:
            print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    test_pollinations("Disney Pixar style 3D render, a girl with glossy hair, magical village background, cinematic lighting")
