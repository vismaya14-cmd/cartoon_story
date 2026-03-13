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
HF_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "prompthero/openjourney" 
]

def pollinations_generate(prompt, output_path, seed=None):
    """
    Generate an image using Pollinations.ai (Free, high-speed fallback).
    """
    try:
        if seed is None:
            seed = int(time.time())
            
        # Pollinations is VERY sensitive to length and special chars.
        # We simplify to ensure it doesn't time out or return HTML error pages.
        # Clean special characters that break URLs
        clean_prompt = prompt.replace("\n", " ").replace('"', '').replace("'", "")
        simple_prompt = clean_prompt[:180] 
        encoded_prompt = urllib.parse.quote(simple_prompt)
        
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?seed={seed}&nologo=true&width=1024&height=1024"
        print(f"DEBUG: Calling Pollinations for: {simple_prompt[:50]}...")
        
        response = requests.get(url, timeout=40)
        
        content_type = response.headers.get("Content-Type", "").lower()
        # CRITICAL: Must be an image and MUST be a reasonable size (HTML errors are usually < 5KB)
        if response.status_code == 200 and "image" in content_type and len(response.content) > 10000:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"DEBUG: Pollinations Success! Size: {len(response.content)} bytes")
            return True
        else:
            print(f"DEBUG: Pollinations Invalid Response - Status: {response.status_code}, Type: {content_type}, Size: {len(response.content)}")
    except Exception as e:
        print(f"DEBUG: Pollinations Exception: {str(e)}")
    return False


def huggingface_generate(prompt, output_path, model_id):
    """
    Generate an image using a specific Hugging Face model with strict validation.
    """
    if not HF_API_KEY:
        return False
    
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "photorealistic, photo, real life, gritty, messy, blurry, lowres, text, watermark, deformed, ugly",
            "guidance_scale": 7.5,
            "num_inference_steps": 25
        }
    }

    try:
        print(f"DEBUG: Calling Hugging Face [{model_id}]...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=65)
        
        content_type = response.headers.get("Content-Type", "").lower()
        if response.status_code == 200 and "image" in content_type and len(response.content) > 10000:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"DEBUG: HF Success with {model_id}! Size: {len(response.content)} bytes")
            return True
        else:
            msg = response.text[:50] if "json" in content_type else "Non-image response"
            print(f"DEBUG: HF {model_id} failed. Status: {response.status_code}, Msg: {msg}")
            return False
    except Exception as e:
        print(f"DEBUG: HF Exception with {model_id}: {str(e)}")
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

        Task 1: **Absolute Visual Identity Signature**. 
        Analyze the person in the photo and capture their specific character traits (Exact hair style/color, eye color/shape, facial structure like jawline and cheekbones, and their specific outfit colors).
        TRANSFORMATION: Describe them as a masterpiece 3D Pixar character with LARGE GLISSENING EXPRESSIVE EYES, smooth subsurface scattering skin, and a friendly smile.
        The description MUST be a detailed 60-word visual "Identity Lock" string that will be used to keep them looking the same in every scene.

        Task 2: Cinematic 5-Scene Script. 
        Write a heartwarming 5-scene story in {language} about {hero_name}. 
        MANDATORY PROGRESSION: 
        1. Intro: {hero_name} in a vibrant, detailed village.
        2. Adventure: Exploring a magical forest with floating lights.
        3. Mystery: Discovering a hidden ancient secret.
        4. Climax: Solving a challenge with a clever idea.
        5. Celebration: A grand happy ending in a festival setting.

        Task 3: **Cinematic Scene Prompts**. 
        Write a unique, highly-detailed image prompt for EACH of the 5 scenes.
        MANDATORY ELEMENTS:
        - Must start with "Disney Pixar 3D animation style,"
        - Must include the full [IMAGE SIGNATURE] from Task 1.
        - Must include a specific environment (e.g., "colorful glowing village," "dark emerald forest," "ancient library").
        - Must include the character's action and a unique camera angle.
        - Finish with keywords: "octane render, Unreal Engine 5, ray-traced lighting, masterpiece, 8k, vibrant colors, stylized 3D".

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
            print(f"DEBUG: GPT-4o Result: {result_text}")
            if result_text.startswith("```"): result_text = result_text.strip("`").strip("json")
            
            story_data = json.loads(result_text)
            char_sig = story_data.get("character_signature", "a friendly child with expressive eyes")
            story_title = story_data.get("story_title", f"{hero_name}'s Adventure")
            panels = story_data.get("panels", [])

            for i, p in enumerate(panels):
                print(f"DEBUG: Panel {i} Prompt: {p.get('prompt', 'MISSING')}")

            # Generate Hero Image (Hugging Face -> Pollinations Fallback)
            hero_prompt = f"Disney Pixar 3D animation style, {char_sig}, standing heroically in a vibrant village square, cinematic lighting, stylized 3D, high-quality render, 8k."
            
            cartoon_name = f"cartoon_{timestamp}.jpg"
            cartoon_path = os.path.join(cartoon_folder, cartoon_name)
            print(f"DEBUG: Sculpting Pixar-style 3D Hero for {hero_name}...")
            
            # --- Hero Tiered Strategy ---
            hf_success = False
            for model_id in HF_MODELS:
                if huggingface_generate(hero_prompt, cartoon_path, model_id):
                    hf_success = True; break
            
            if not hf_success:
                print("DEBUG: HF Hero failed. Trying Pollinations (Turbo Mode)...")
                # Turbo mode is significantly faster and more available
                hf_success = pollinations_generate(hero_prompt + " --turbo", cartoon_path, seed=timestamp)

            if hf_success:
                cartoon_image_url = settings.MEDIA_URL + f"cartoon/{cartoon_name}"
            else:
                print("DEBUG: All Hero AI failed. Using Local 3D Stylization Fallback...")
                if cartoonize(image_path, cartoon_path, variation=0):
                    cartoon_image_url = settings.MEDIA_URL + f"cartoon/{cartoon_name}"
                else:
                    cartoon_image_url = settings.STATIC_URL + "images/placeholder.png"

            # Generate all Scene Images (Parallel)
            pages = []
            
            def generate_scene_image(idx, p):
                p_prompt = p.get("prompt", "")
                p_text = p.get("text", "magical adventure")
                s_name = f"scene_{timestamp}_{idx}.jpg"
                s_path = os.path.join(cartoon_folder, s_name)
                
                # 1. HF Tiers
                success = False
                for m_id in HF_MODELS:
                    if huggingface_generate(p_prompt, s_path, m_id):
                        success = True; break
                
                # 2. Pollinations Precise
                if not success:
                    print(f"DEBUG: Scene {idx} HF failed. Trying Pollinations Precisely...")
                    success = pollinations_generate(p_prompt, s_path, seed=timestamp + idx + 10)
                
                # 3. Pollinations Simplified Identity Lock
                if not success:
                    print(f"DEBUG: Scene {idx} simplifying for identity lock...")
                    # Extract roughly what's happening
                    action_summary = p_text[:60].split(".")[0]
                    simple_p = f"Disney Pixar 3D animation, {char_sig}, {action_summary}, cinematic background, masterpiece, 8k"
                    success = pollinations_generate(simple_p, s_path, seed=timestamp + idx + 20)

                if success:
                    return settings.MEDIA_URL + f"cartoon/{s_name}"
                else:
                    print(f"DEBUG: Scene {idx} AI failed. Using Local Stylized Hero + Narrative Overlay...")
                    # Even if pure AI fails, we use their face stylized locally
                    if cartoonize(image_path, s_path, variation=idx+1):
                        return settings.MEDIA_URL + f"cartoon/{s_name}"
                    return settings.STATIC_URL + "images/placeholder.png"

            print(f"DEBUG: Generating {len(panels)} unique Pixar scenes...")
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(generate_scene_image, i, p) for i, p in enumerate(panels)]
                scene_urls = [f.result() for f in futures]

            for i, p_data in enumerate(panels):
                pages.append({
                    "narrative_text": p_data["text"],
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