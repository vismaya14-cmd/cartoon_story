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

# Load environment variables (override allows updating if file changes)
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

        # Read image bytes and encode to base64 for the API
        image_file.seek(0)
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Determine MIME type
        content_type = image_file.content_type or "image/jpeg"

        # Build the data URI for the image
        data_uri = f"data:{content_type};base64,{base64_image}"

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
            "cover_image_prompt": "Pixar 3d style cartoon portrait of a... [detailed hero description from Task 1] standing in a magical sparkling environment, vibrant colors, heroic pose, cinematic lighting",
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

        try:
            print(f"DEBUG: Generating {language} story for {hero_name} using GitHub Models API ({MODEL})...")

            response = client.complete(
                messages=[
                    SystemMessage("You are a creative children's story writer. Always respond with valid JSON only."),
                    UserMessage(
                        content=[
                            TextContentItem(text=prompt),
                            ImageContentItem(image_url=ImageUrl(url=data_uri)),
                        ]
                    ),
                ],
                model=MODEL,
            )

            print(f"DEBUG: GitHub Models API responded successfully.")

            result_text = response.choices[0].message.content
            if result_text:
                # Clean up markdown if the AI includes it by accident
                json_text = result_text.strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:]
                if json_text.startswith("```"):
                    json_text = json_text[3:]
                if json_text.endswith("```"):
                    json_text = json_text[:-3]

                story_data = json.loads(json_text.strip())
                hero_desc = story_data.get("hero_description", "")
                panels = story_data.get("panels", [])

                # Generate a cartoon cover image from the hero description
                cover_prompt = story_data.get("cover_image_prompt", "")
                if not cover_prompt:
                    cover_prompt = f"Pixar 3d style cartoon portrait of a {hero_desc}, standing in a magical sparkling environment, vibrant colors, heroic pose, cinematic lighting"

                encoded_cover = urllib.parse.quote(cover_prompt)
                cover_image_url = f"https://image.pollinations.ai/prompt/{encoded_cover}?width=1024&height=1024&nologo=true&seed={int(time.time())}"

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
            "cover_image_url": cover_image_url,
            "pages": pages
        })

    return render(request, "index.html")