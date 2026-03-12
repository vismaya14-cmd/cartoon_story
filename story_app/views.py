import os
import time
import json
import urllib.parse
from django.shortcuts import render
from django.conf import settings
from gtts import gTTS
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

# Load environment variables (override allows updating if file changes)
load_dotenv(override=True)

# Configure Gemini with API key from environment
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    # This should not happen if .env is correctly loaded
    raise ValueError("GEMINI_API_KEY not found in environment. Please check your .env file.")


def home(request):

    if request.method == "POST":

        image_file = request.FILES.get("image")
        hero_name = request.POST.get("hero", "Hero")
        language = request.POST.get("language", "English")

        if not image_file:
            return render(request, "index.html", {"error": "Please upload an image."})

        # create folders automatically
        images_folder = os.path.join(settings.MEDIA_ROOT, "images")
        audio_folder = os.path.join(settings.MEDIA_ROOT, "audio")

        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)

        # save uploaded image
        image_name = f"{int(time.time())}_{image_file.name}"
        image_path = os.path.join(images_folder, image_name)

        with open(image_path, "wb+") as f:
            for chunk in image_file.chunks():
                f.write(chunk)

        original_image_url = settings.MEDIA_URL + "images/" + image_name

        # AI STORY GENERATION - Upgraded to 2.0 Flash for better reliability
        model = genai.GenerativeModel("gemini-2.0-flash")

        try:
            image_file.seek(0)
            image_bytes = image_file.read()

            print(f"DEBUG: Generating {language} story for {hero_name} using 2.0 Flash model...")
            
            prompt = f"""
            You are a creative children's comic book author and a visual artist.
            The user has uploaded a photo of a person. The character's name is {hero_name}.
            
            Task 1: Describe the person in the photo physically in English (age, hair style/color, eye color, clothing style/colors, defining features). Be very specific so the character looks consistent in all panels.
            
            Task 2: Write a magical, child-friendly 7-panel comic book story starring {hero_name}. 
            The story MUST be written in {language}. 
            It should be fun, magical, and have a clear beginning, middle, and satisfying conclusion. 
            DO NOT use markdown formatting like ** in the story text.
            
            Task 3: For each of the 7 panels, write a highly detailed English image generation prompt. 
            The prompt must start with: "Pixar 3d style cartoon illustration of a..." 
            and include the EXACT physical description from Task 1, plus what is happening in the scene, the background, and the magical environment.
            
            You MUST return the output ONLY as a raw JSON string with the following structure:
            {{
                "hero_description": "English description of the hero",
                "panels": [
                    {{
                        "narrative_text": "Story text for panel 1 in {language}",
                        "image_prompt": "Pixar 3d style cartoon illustration of a... [detailed hero description] [action] [magical background]"
                    }},
                    ... (7 panels total)
                ]
            }}
            Do not include ```json markdown blocks, just the raw JSON object.
            """

            content_parts = [
                {"text": prompt},
                {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                }
            ]

            # Attempt API Call
            print(f"DEBUG: Generating content using Gemini 2.0 Flash... (Waiting for Google API)")
            response = model.generate_content(content_parts)
            print(f"DEBUG: Google API responded successfully.")
            
            if response.text:
                # Clean up markdown if the AI includes it by accident
                json_text = response.text.strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]
                
                story_data = json.loads(json_text.strip())
                hero_desc = story_data.get("hero_description", "")
                panels = story_data.get("panels", [])

                pages = []

                for panel in panels:
                    img_prompt = panel.get("image_prompt", "")
                    story = panel.get("narrative_text", "")

                    # Proper URL encoding for image prompts
                    encoded_prompt = urllib.parse.quote(img_prompt)
                    image_gen_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={int(time.time())}"

                    pages.append({
                        "narrative_text": story,
                        "image_url": image_gen_url
                    })
            else:
                raise Exception("AI returned empty response")
            
            if not pages:
                raise Exception("AI failed to generate story panels.")

        except exceptions.ResourceExhausted:
            print("DEBUG: Quota Exhausted")
            return render(request, "index.html", {
                "error": "The shared API quota is exhausted. Please try again later or use your own free key (see instructions below)."
            })
        except exceptions.DeadlineExceeded:
            print("DEBUG: Deadline Exceeded")
            return render(request, "index.html", {
                "error": "Google servers took too long. Please try again."
            })
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON Parse Error: {str(e)}")
            return render(request, "index.html", {
                 "error": "AI generated an invalid story format. Please try again."
            })
        except Exception as e:
            print(f"DEBUG: General Error: {str(e)}")
            return render(request, "index.html", {
                "error": f"AI Generation error: {str(e)}"
            })

        print("DEBUG: Rendering final storybook page!")
        return render(request, "storybook.html", {
            "hero": hero_name,
            "language": language,
            "hero_description": hero_desc,
            "original_image_url": original_image_url,
            "pages": pages
        })

    return render(request, "index.html")