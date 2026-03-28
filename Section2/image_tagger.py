import os
import json
import base64
from pathlib import Path
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# --- Setup & Configuration ---
# Load environment variables (assumes .env is in the project root)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize the OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Define the folders
IMAGE_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_FILE = Path(__file__).resolve().parent / "tags_output.json"

# Supported image formats for vision models
SUPPORTED_FORMATS = {".png", ".jpeg", ".jpg", ".webp"}

# --- Helper Functions ---
def encode_image_to_base64(image_path: Path) -> str:
    """Reads an image file and returns a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image_with_llm(base64_image: str, image_ext: str) -> dict:
    """Sends the base64 image to GPT-4o and returns the structured JSON analysis."""
    
    # Remove the dot from the extension for the MIME type (e.g., .png -> png)
    mime_type = image_ext.replace(".", "")
    if mime_type == "jpg":
        mime_type = "jpeg"

    prompt_instructions = (
        "You are an expert advertising analyst and content moderator. "
        "Analyze the provided image and generate metadata for an agency asset library. "
        "You MUST respond in valid JSON format with the following keys EXACTLY: \n"
        "- 'alt_text': A descriptive alt text for accessibility.\n"
        "- 'tags': An array of 5-8 descriptive content tags (strings).\n"
        "- 'brand_safety_score': An integer from 1 to 10 (10 being completely safe for all brands, 1 being highly controversial or unsafe).\n"
        "- 'use_cases': An array of 2-3 suggested advertising use cases (e.g., 'Social Media Story', 'Hero Website Banner')."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_instructions},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000,
        temperature=0.3 # Keep temperature low for analytical accuracy
    )
    
    # Parse the returned JSON string into a Python dictionary
    return json.loads(response.choices[0].message.content)

# --- Main Execution ---
def main():
    # Ensure the images directory exists
    if not IMAGE_DIR.exists():
        print(f"Error: Directory '{IMAGE_DIR}' not found. Creating it now...")
        IMAGE_DIR.mkdir()
        print("Please place your 5 test images in the 'images' folder and run the script again.")
        return

    results = []
    image_files = [f for f in IMAGE_DIR.iterdir() if f.is_file()]

    if not image_files:
        print(f"No files found in '{IMAGE_DIR}'. Please add test images and try again.")
        return

    print(f"Found {len(image_files)} file(s). Starting batch processing...\n")

    for img_path in image_files:
        if img_path.suffix.lower() not in SUPPORTED_FORMATS:
            print(f"⏭️ Skipping {img_path.name} (Unsupported format: {img_path.suffix})")
            continue
        
        print(f"⏳ Processing {img_path.name}...")
        try:
            base64_img = encode_image_to_base64(img_path)
            analysis_data = analyze_image_with_llm(base64_img, img_path.suffix.lower())
            
            # Append the filename to the resulting dictionary as required by the brief
            analysis_data["filename"] = img_path.name
            results.append(analysis_data)
            print(f"✅ Successfully processed {img_path.name}")
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")

    # Write all results to the output JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n🎉 Batch processing complete! Results saved to {OUTPUT_FILE.name}")

if __name__ == "__main__":
    main()