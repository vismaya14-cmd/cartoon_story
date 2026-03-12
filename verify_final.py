import os
import sys
import requests
import urllib.parse
import time

# Project path
sys.path.append(r"c:\Users\Vismaya H M\Desktop\U01MI23S0018\cartoon_story")
from story_app.views import pollinations_generate, huggingface_generate
from story_app.cartoon_filter import cartoonize

def test_full_pipeline():
    media_root = r"c:\Users\Vismaya H M\Desktop\U01MI23S0018\cartoon_story\media"
    output_dir = os.path.join(media_root, "final_test")
    os.makedirs(output_dir, exist_ok=True)
    
    prompt = "Disney Pixar style 3D render, a girl in a magenta saree, magical forest background, cinematic lighting"
    out_path = os.path.join(output_dir, "test_ai.jpg")
    
    print("Testing Pollinations...")
    success = pollinations_generate(prompt, out_path)
    if success:
        print(f"Pollinations Success: {out_path} (Size: {os.path.getsize(out_path)})")
    else:
        print("Pollinations Failed. Testing Fallback...")
        source = os.path.join(media_root, "images", "vishu.jpeg")
        cartoonize(source, out_path)
        print(f"Fallback Success: {out_path} (Size: {os.path.getsize(out_path)})")

if __name__ == "__main__":
    test_full_pipeline()
