import os
import time
import anthropic
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the root .env file
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialized Anthropic client
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def generate_anthropic_response(prompt: str, max_retries: int = 3) -> str:
    base_wait_time = 2  # Starting wait time for exponential backoff (in seconds)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307", # You can change this to claude-3-opus or claude-3-sonnet
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # print(response)
            return response.content[0].text

        except anthropic.RateLimitError as e:
            wait = base_wait_time * (2 ** attempt)
            print(f"[Attempt {attempt + 1}/{max_retries}] Rate limit hit: {e}. Retrying in {wait} seconds...")
            time.sleep(wait)
            
        except anthropic.APIError as e:
            # Catch other API errors and raise them immediately so we don't blindly retry on bad requests
            raise RuntimeError(f"Anthropic API Error: {e}")

    raise RuntimeError("Max retries exceeded. Failed to get response from Anthropic API due to rate limits.")

if __name__ == "__main__":
    test_prompt = "Write a simple haiku about what it feels like to write clean, reliable Python code—something that captures the calm, focus, or satisfaction that comes with it."
    print("Sending request to Anthropic API...\n")
    
    result = generate_anthropic_response(test_prompt)
    print(f"Response:\n{result}")