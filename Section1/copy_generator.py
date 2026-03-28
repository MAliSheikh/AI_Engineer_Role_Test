import os
import json
import time
from openai import OpenAI, OpenAIError, RateLimitError, APIError
from dotenv import load_dotenv
from pathlib import Path

# Get root folder (one level up)
env_path = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path=env_path)


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)



def generate_ad_copy(brief: str, max_retries: int = 3) -> dict:
    
    # SYSTEM PROMPT: Defines the persona, instructions, and strict output constraints.
    system_prompt = (
        "You are an expert advertising copywriter at a top-tier creative agency. "
        "Your task is to generate compelling, high-converting ad copy based on the user's product brief.\n\n"
        "You must output EXACTLY 3 distinct variations."
        "You must respond in valid JSON format. The JSON structure MUST exactly match this schema without any markdown wrapping or extra text:\n"
        "{"
        '  "variation_1": { "headline": "...", "tagline": "...", "body": "...", "cta": "..." },\n'
        '  "variation_2": { "headline": "...", "tagline": "...", "body": "...", "cta": "..." },\n'
        '  "variation_3": { "headline": "...", "tagline": "...", "body": "...", "cta": "..." }\n'
        "}"
    )

    # USER PROMPT: Simply passes the dynamic input data.
    user_prompt = f"Product Brief: {brief}\n\nPlease generate the 3 copy variations."

    # Ad copy requires a high degree of creativity, emotive language, and varied vocabulary. 
    # A low temperature would result in generic, robotic-sounding text. 
    temperature = 0.8

    # Help avoid hitting rate limits
    base_wait_time = 2  # Starting wait time for exponential backoff (in seconds)
    

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o", # gpt-3.5-turbo 
                response_format={ "type": "json_object" }, # JSON output
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=1000, # OpenRouter will only check balance against the cost of 1,000 tokens
            )

            # Extract the raw string from the model's response and parse it
            raw_content = response.choices[0].message.content
            return json.loads(raw_content)

        except RateLimitError as e:
            wait = base_wait_time * (2 ** attempt)
            print(f"[Attempt {attempt + 1}/{max_retries}] Rate limit hit. Retrying in {wait} seconds...")
            time.sleep(wait)
            
        except (APIError, OpenAIError) as e:
            wait = base_wait_time * (2 ** attempt)
            print(f"[Attempt {attempt + 1}/{max_retries}] API Error: {e}. Retrying in {wait} seconds...")
            time.sleep(wait)
            
        except json.JSONDecodeError as e:
            # Fallback in the rare event the model outputs malformed JSON
            print(f"[Attempt {attempt + 1}/{max_retries}] Failed to parse JSON output: {e}. Retrying...")
            # We don't sleep for formatting errors, just immediately retry
            pass

    raise RuntimeError("Max retries exceeded. Failed to generate ad copy.")

if __name__ == "__main__":
    test_brief = "New luxury perfume for men, brand name: Noir, target: 30-45 year old professionals"
    print(f"Generating ad copy for brief: '{test_brief}'...\n")

    try:
        # Call the tool
        result = generate_ad_copy(test_brief)
        
        # Output the parsed JSON object neatly
        print("Successfully generated copy:\n")
        print(json.dumps(result, indent=2))
        
    except Exception as err:
        print(f"Process failed: {err}")
