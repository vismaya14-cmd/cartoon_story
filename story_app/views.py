import os
import time
import json
import base64
import urllib.parse
from django.shortcuts import render
from django.conf import settings
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    ImageContentItem,
    ImageUrl,
    TextContentItem,
)
from azure.core.credentials import AzureKeyCredential
from .cartoon_filter import cartoonize
import requests
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv(override=True)

# Configure GitHub Models API
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ENDPOINT = "https://models.github.ai/inference"
MODEL = "openai/gpt-4o"

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in environment. Please check your .env file.")

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# Configure Hugging Face API
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

def pollinations_generate(prompt, output_path, seed=None):
    """
    Generate an image using Pollinations.ai (Free, high-speed fallback).
    """
    try:
        if seed is None:
            seed = int(time.time())
            
        # Pollinations is VERY sensitive to length and special chars.
        # We simplify to the core 120 chars to ensure it doesn't time out.
        simple_prompt = prompt[:120]
        encoded_prompt = urllib.parse.quote(simple_prompt)
        
        # Simplest possible URL works best
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&nologo=true"
        print(f"DEBUG: Calling Pollinations (Simple) for: {simple_prompt[:40]}...")
        
        response = requests.get(url, timeout=35)
        
        # Check if it's an image
        content_type = response.headers.get("Content-Type", "").lower()
        if response.status_code == 200 and "image" in content_type:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"DEBUG: Pollinations Failure ({response.status_code}) - Type: {content_type}")
    except Exception as e:
        print(f"DEBUG: Pollinations Exception: {str(e)}")
    return False


def huggingface_generate(prompt, output_path, retries=1):
    """
    Generate an image using Hugging Face SDXL.
    """
    if not HF_API_KEY:
        return False
    
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "photorealistic, photo, real life, gritty, messy, blurry, lowres, text, watermark",
            "guidance_scale": 9.0,
            "num_inference_steps": 30
        }
    }

    for attempt in range(retries + 1):
        try:
            print(f"DEBUG: Calling Hugging Face for prompt: {prompt[:50]}... (Attempt {attempt + 1})")
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=90)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return True
            else:
                print(f"DEBUG: HF Error {response.status_code}: {response.text[:100]}")
                # Don't retry on auth errors or 404
                if response.status_code in [401, 403, 404]:
                    break
                time.sleep(2)
        except Exception as e:
            print(f"DEBUG: HF Exception: {str(e)}")
            time.sleep(1)
    return False


def home(request):

    if request.method == "POST":

        image_file = request.FILES.get("image")
        hero_name = request.POST.get("hero", "Hero")
        language = request.POST.get("language", "English")

        if not image_file:
            return render(request, "index.html", {"error": "Please upload an image."})

        # Create folders
        images_folder = os.path.join(settings.MEDIA_ROOT, "images")
        cartoon_folder = os.path.join(settings.MEDIA_ROOT, "cartoon")
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(cartoon_folder, exist_ok=True)

        # Save uploaded image
        timestamp = int(time.time())
        image_name = f"{timestamp}_{image_file.name}"
        image_path = os.path.join(images_folder, image_name)

        with open(image_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        original_image_url = settings.MEDIA_URL + "images/" + image_name

        # Read image for AI description
        image_file.seek(0)
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        content_type = image_file.content_type or "image/jpeg"
        data_uri = f"data:{content_type};base64,{base64_image}"

        # 1. Ask GPT-4o to describe the character and write the story
        # This ensures character consistency across prompts
        prompt = f"""
        You are a Master Director at Pixar Animation Studios. 
        Analyze the uploaded photo of {hero_name} and create a professional children's animated movie.

        Task 1: Signature Pixar Character. 
        Describe {hero_name} as a 3D animated character with a STRIKING and CONSISTENT visual signature.
        Details to include: Unique hairstyle/color, eye color, a specific signature outfit (e.g., "wearing a glowing blue tunic and silver boots"), and expressive facial style.
        This "Character Signature" MUST be the centerpiece of every scene prompt.

        Task 2: Cinematic 5-Scene Script. 
        Write a heartwarming 5-scene story in {language} about {hero_name}. 
        MANDATORY PROGRESSION: 
        1. Intro: {hero_name} in a vibrant, detailed village.
        2. Adventure: Exploring a magical forest with floating lights.
        3. Mystery: Discovering a hidden ancient secret.
        4. Climax: Solving a challenge with a clever idea.
        5. Celebration: A grand happy ending in a festival setting.

        Task 3: Pixar-Style Image Prompts. 
        Write a unique, highly-detailed image prompt for EACH scene.
        FORMAT: "Pixar style 3D animation, [CHARACTER SIGNATURE], [SPECIFIC SCENE ACTION], [VIBRANT CINEMATIC BACKGROUND], colorful lighting, masterpiece, 8k render, children's animated movie style"
        CRITICAL: Each background MUST be different and highly detailed.

        Return ONLY valid JSON:
        {{
            "character_signature": "detailed 50-word visual description",
            "story_title": "Grand Title",
            "panels": [
                {{
                    "text": "story narrative for this page",
                    "prompt": "the full cinematic image prompt"
                }}
            ]
        }}
        """

        try:
            print("DEBUG: Generating premium 3D AI story...")
            response = client.complete(
                messages=[
                    SystemMessage("You are a Disney-Pixar Storyboard Artist. You create concise, high-impact 3D animation prompts and whimsical stories. Keep prompts under 200 chars. Output valid JSON only."),
                    UserMessage(content=[
                        TextContentItem(text=prompt),
                        ImageContentItem(image_url=ImageUrl(url=data_uri))
                    ])
                ],
                model=MODEL
            )

            result_text = response.choices[0].message.content.strip()
            if result_text.startswith("```"): result_text = result_text.strip("`").strip("json")
            
            story_data = json.loads(result_text)
            char_sig = story_data.get("character_signature", "a friendly child with expressive eyes")
            story_title = story_data.get("story_title", f"{hero_name}'s Adventure")
            panels = story_data.get("panels", [])

            # Generate Hero Image (Hugging Face -> Pollinations Fallback)
            # Use user's EXACT required keywords for maximum Pixar quality
            hero_prompt = f"Pixar style 3D animation, {char_sig}, standing in a vibrant, detailed village, colorful cinematic lighting, expressive cartoon character, detailed animated movie style, masterpiece, 8k render."
            
            cartoon_name = f"cartoon_{timestamp}.jpg"
            cartoon_path = os.path.join(cartoon_folder, cartoon_name)
            print(f"DEBUG: Generating Pixar-style 3D hero for {hero_name}...")
            
            # Try HF first, then Pollinations
            hf_success = huggingface_generate(hero_prompt, cartoon_path)
            if not hf_success:
                print("DEBUG: HF Hero failed. Trying Pollinations fallback...")
                hf_success = pollinations_generate(hero_prompt, cartoon_path, seed=timestamp)
            
            # FINAL FALLBACK: If AI completely fails, use OpenCV cartoonize
            if not hf_success:
                print("DEBUG: AI Hero failed. Falling back to OpenCV CARTOONIZE...")
                cartoon_success = cartoonize(image_path, cartoon_path)
                cartoon_image_url = (settings.MEDIA_URL + "cartoon/" + cartoon_name) if cartoon_success else original_image_url
            else:
                cartoon_image_url = settings.MEDIA_URL + "cartoon/" + cartoon_name

            # Generate all Scene Images (Parallel for speed)
            pages = []
            
            def generate_scene_image(idx, panel_prompt):
                scene_name = f"scene_{timestamp}_{idx}.jpg"
                scene_path = os.path.join(cartoon_folder, scene_name)
                
                # 1. Try Primary High-Quality Generation (HF SDXL)
                success = huggingface_generate(panel_prompt, scene_path)
                
                # 2. Try Pollinations with the full prompt
                if not success:
                    print(f"DEBUG: Scene {idx} HF failed. Trying Pollinations Full Prompt...")
                    success = pollinations_generate(panel_prompt, scene_path, seed=timestamp + idx)
                
                # 3. New Hybrid Fallback: If full prompt fails, try a SIMPLE BACKGROUND ONLY prompt
                # This ensures the user at least gets a beautiful Pixar background.
                if not success:
                    # Extract last 40 chars of prompt (usually where background is described)
                    bg_keyword = panel_prompt.split(",")[-2].strip() if "," in panel_prompt else "magical 3d world"
                    bg_prompt = f"Pixar style 3D animation background, {bg_keyword}, colorful cinematic lighting, masterpiece, high-quality 3D render"
                    print(f"DEBUG: Scene {idx} full prompt failed. Trying simplified Background-Only: {bg_keyword}")
                    success = pollinations_generate(bg_prompt, scene_path, seed=timestamp + idx)

                # 4. FINAL LOCAL FALLBACK: If AI completely fails, use Sharp 3D Synth 3.0
                if success:
                    return settings.MEDIA_URL + "cartoon/" + scene_name
                else:
                    print(f"DEBUG: Scene {idx} AI completely failed. Using Sharp 3D Synth Variation {idx}")
                    # Variation shifts camera framing and color mood
                    cartoon_success = cartoonize(image_path, scene_path, variation=idx)
                    if cartoon_success:
                        return settings.MEDIA_URL + "cartoon/" + scene_name
                    return cartoon_image_url

            print(f"DEBUG: Generating {len(panels)} Pixar scenes in parallel...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Kick off all generations
                futures = [executor.submit(generate_scene_image, i, p["prompt"]) for i, p in enumerate(panels)]
                # Wait and collect results
                scene_urls = [f.result() for f in futures]

            for i, panel in enumerate(panels):
                pages.append({
                    "narrative_text": panel["text"],
                    "scene_image_url": scene_urls[i],
                    "panel_number": i + 1
                })

            return render(request, "storybook.html", {
                "hero": hero_name,
                "language": language,
                "hero_description": char_sig,
                "story_title": story_title,
                "cartoon_image_url": cartoon_image_url,
                "pages": pages,
            })

        except Exception as e:
            print(f"DEBUG: Critical Error: {str(e)}")
            return render(request, "index.html", {"error": f"Failed to generate story: {str(e)}"})

    return render(request, "index.html")