import requests
import sys

def test_pollinations():
    url = "https://pollinations.ai/p/Pixar%20style%20background%20magical%20village?width=1024&height=1024&seed=123"
    print(f"Testing URL: {url}")
    try:
        r = requests.get(url, timeout=30)
        print(f"Status: {r.status_code}")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
        if r.status_code == 200 and 'image' in r.headers.get('Content-Type', ''):
            print("SUCCESS: Received an image.")
        else:
            print(f"FAILURE: Status {r.status_code}, Body: {r.text[:200]}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_pollinations()
