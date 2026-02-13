import json
import os

from comfyui import generate_image

PROMPTS_FILE = os.path.join(os.path.dirname(__file__), ".personas", "emily-prompt", "prompts.json")
PROMPT_IMAGES_FILE = os.path.join(os.path.dirname(__file__), ".personas", "emily-prompt", "prompt_images.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), ".personas", "emily-prompt", "prompt_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(10):
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if not prompts:
        break
    # Take and remove the first prompt each time so we process in order and don't skip
    user_input = prompts.pop(0)
    save_path = os.path.join(OUTPUT_DIR, f"out_{i}.png")
    generate_image(user_input, save_path)
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2)

    # Append this prompt and image path to prompt_images.json (load existing array, add one, write back)
    if os.path.isfile(PROMPT_IMAGES_FILE):
        with open(PROMPT_IMAGES_FILE, "r", encoding="utf-8") as f:
            prompt_images_list = json.load(f)
    else:
        prompt_images_list = []
    prompt_images_list.append({"prompt": user_input, "image_path": save_path})
    with open(PROMPT_IMAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(prompt_images_list, f, indent=2)