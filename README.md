pip install uv

uv sync

I have used OPENROUTER API KEY THROUGH OPENAI LIBRARIRY
ITS VARIABLE NAME IS OPENROUTER_API_KEY

<!-- Section 1 -->
uv run Section1\task_1_1_copy_generator.py

<!-- Section 2 -->
uv run Section2\task_2_1_ai_campaign_analyzer.py
uv run Section2\task_2_2_ai_image_description.py
streamlit run Section2/task_2_3_rag.py

<!-- Section 3 -->
<!-- IN this i face rate limits errors -->
uv run Section3\task_3_1_anthropic_retry.py 

<!-- I fix bugs -->
uv run Section3\S3_Q2_broken_rag_pipeline.py 