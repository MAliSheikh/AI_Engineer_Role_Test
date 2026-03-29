# AI Engineer Skills Assessment

Welcome to my submission! This project contains several AI scripts, including copy generators, RAG pipelines, and AI image description tools.

## 🚀 Setup & Installation

This project uses `uv` for extremely fast Python package management.

1. **Install `uv`**:
   ```bash
   pip install uv
   ```
2. **Sync the dependencies**:
   ```bash
   uv sync
   ```
3. **Activate the virtual environment** (Windows):
   ```bash
   .venv\Scripts\activate
   ```

## 🔑 Environment Variables

I have used the OpenRouter API key through the OpenAI library. You will need to set this in a `.env` file at the root of the project:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
```

## 🛠️ Usage

### Section 1: Prompt Engineering
```bash
uv run Section1\task_1_1_copy_generator.py
```

### Section 2: Building AI Tools
```bash
Task: 1
uv run Section2\task_2_1_ai_campaign_analyzer.py
Go to http://127.0.0.1:8000/docs

Prompt File for task 1:
test_prompt_2_1.txt
test_prompt_2_1.pdf

Task: 2
uv run Section2\task_2_2_ai_image_description.py

Task: 3
streamlit run Section2/task_2_3_rag.py
```

### Section 3: Debugging & Resiliency
> **Note:** I encountered rate limit errors with the Anthropic API in the first task. The second task includes my bug fixes for the broken LangChain pipeline.
```bash

Task: 1
uv run Section3\task_3_1_anthropic_retry.py 

Task: 2
uv run Section3\S3_Q2_broken_rag_pipeline.py 


Task: 3
task_3_3_ai_prompt.txt

Task: 4
task_3_4_image_scoring.pdf


Task: 5
Architecture Diagram Link: https://drive.google.com/file/d/1omkYwoeOd36duP_uCSUltNOzxxs4vQWJ/view?usp=sharing

and file name 
task3_5_architecture_diagram.pdf
```