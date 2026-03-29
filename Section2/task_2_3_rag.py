import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from dotenv import load_dotenv

# 1. UI Configuration
st.set_page_config(page_title="Campaign Knowledge Bot", page_icon="🤖")
st.title("Campaign Knowledge Bot")
st.write("Ask questions about our agency case studies and brand guidelines.")

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Check for API key
if not OPENROUTER_API_KEY:
    st.error("Please set your OPENROUTER_API_KEY environment variable to run this app.")
    st.stop()

# 2. Document Loading & Vector Store Initialization
# Using st.cache_resource so we don't re-embed the PDFs on every UI interaction
@st.cache_resource
def load_and_prepare_vector_store():
    data_dir = Path(__file__).parent / "data"

    if not data_dir.is_dir():
        st.error(f"Data directory not found. Please create a folder named 'data' at '{data_dir}' and add your PDF files.")
        st.stop()

    docs = []
    # Load all PDFs from the data directory
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        st.error(f"No PDF files found in '{data_dir}'. Please add your documents.")
        st.stop()

    for file_path in pdf_files:
        loader = PyPDFLoader(str(file_path))
        docs.extend(loader.load())
            
    # Split the documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents(docs)
    
    # Create the vector store
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        ),
        persist_directory="./chroma_db" # Persists the DB locally
    )
    return vectorstore

vectorstore = load_and_prepare_vector_store()

# Retrieve the top 4 most relevant chunks
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

# 3. Prompt Engineering (Enforcing the brief's rules)
system_prompt = """You are a campaign knowledge assistant.

Answer ONLY using the provided context.

If the answer is partially available, you MUST still answer by combining relevant information from the context.

Only say "I cannot answer this question based on the provided documents." if absolutely no relevant information exists.

Format:

Answer: [Concise answer]
Source: [Document name]
Quote: "[Exact supporting quote(s)]"

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# 4. Helper function to pass document metadata (source) to the LLM
def format_docs(docs):
    formatted = []
    for doc in docs:
        # Extract just the filename from the source path
        source = os.path.basename(doc.metadata.get("source", "Unknown Document"))
        formatted.append(f"Source Document: {source}\nContent: {doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# 5. Build the LangChain Expression Language (LCEL) Chain
# We use temperature=0 to prevent hallucinations and keep the LLM strictly factual

llm = ChatOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-3.5-turbo", # OpenRouter requires the provider prefix
    temperature=0
) 

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Streamlit Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input handling
if user_query := st.chat_input("Ask a question about the documents..."):
    docs = retriever.invoke(user_query)
    for d in docs:
        print(d.page_content[:300])
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response = rag_chain.invoke(user_query)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
