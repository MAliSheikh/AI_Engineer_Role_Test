from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import os
import uvicorn
import fitz  # PyMuPDF
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from pathlib import Path

from models import AnalysisResponse
from prompt import SYSTEM_PROMPT

# Load environment variables (assumes .env in project root)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

app = FastAPI(
    title="AI Campaign Brief Analyzer",
    description="An API that uses an LLM to analyze a campaign brief and return a structured analysis.",
    version="1.0.0"
)

# Function to extract text from file upload or plain text
async def extract_text_from_upload(brief_text: str = None, file: UploadFile = None) -> str:
    extracted_text = ""

    if file:
        if file.content_type == "application/pdf" or file.filename.endswith(".pdf"):
            try:
                pdf_bytes = await file.read()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                for page in doc:
                    extracted_text += page.get_text()
                doc.close()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read PDF file: {e}")
        else:
            try:
                extracted_text = (await file.read()).decode("utf-8")
            except Exception:
                raise HTTPException(status_code=400, detail="Unsupported file type. Upload PDF or plain text.")
    elif brief_text:
        extracted_text = brief_text
    else:
        raise HTTPException(status_code=400, detail="You must provide either 'brief_text' or a 'file'.")

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="The provided brief is empty.")

    return extracted_text

# Reusable AI response function (works for normal + streaming)
def generate_ai_response(extracted_text: str, stream: bool = False, model: str = "gpt-4o", temperature: float = 0.5):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": extracted_text}
        ],
        temperature=temperature,
        max_tokens=1500,
        response_format={"type": "json_object"} if not stream else None,
        stream=stream
    )

# Normal API endpoint
@app.post("/analyze-brief", response_model=AnalysisResponse)
async def analyze_brief(
    brief_text: str = Form(None, description="Plain text campaign brief"),
    file: UploadFile = File(None, description="PDF or text file containing the campaign brief")
):
    extracted_text = await extract_text_from_upload(brief_text, file)

    try:
        response = generate_ai_response(extracted_text, stream=False)
        analysis_data = response.choices[0].message.content
        return AnalysisResponse.parse_raw(analysis_data)
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process or parse AI response: {e}")

# Streaming API endpoint (SSE)
@app.post("/analyze-brief-stream")
async def analyze_brief_stream(
    brief_text: str = Form(None, description="Plain text campaign brief"),
    file: UploadFile = File(None, description="PDF or text file containing the campaign brief")
):
    extracted_text = await extract_text_from_upload(brief_text, file)

    def event_stream():
        try:
            response = generate_ai_response(extracted_text, stream=True)
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {content}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)