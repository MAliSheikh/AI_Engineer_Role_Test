"""
AI ENGINEER SKILLS ASSESSMENT — Section 3, Question 2
=====================================================
BROKEN LANGCHAIN RAG PIPELINE — DEBUG TASK

INSTRUCTIONS:
    This file contains a LangChain RAG (Retrieval-Augmented Generation) 
    pipeline with EXACTLY 4 deliberate bugs hidden in the code.

    Your task:
    1. Identify all 4 bugs
    2. Fix them
    3. Write a comment above each fix explaining:
       - What the bug was
       - Why it breaks the pipeline
       - What the correct fix is

    The pipeline should:
    - Load and chunk 3 PDF documents
    - Create vector embeddings and store in ChromaDB
    - Accept a user question
    - Retrieve relevant chunks
    - Generate a grounded answer using an LLM
    - Return the answer WITH the source document name

    DO NOT change the overall architecture — only fix the bugs.
    DO NOT add new features — just make it work correctly.

SCORING:
    - 2.5 points per bug correctly identified + fixed with explanation
    - Total: 10 points

SETUP (before running):
    pip install langchain langchain-openai langchain-community chromadb pypdf
    export OPENAI_API_KEY="your-key-here"
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ─── Configuration ────────────────────────────────────────────────────────────
PDF_PATHS = [
    "RAG_Doc1_Lumiere_Case_Study.pdf",
    "RAG_Doc2_Velocity_Case_Study.pdf",
    "RAG_Doc3_Noir_Brand_Guidelines.pdf",
]

CHROMA_PERSIST_DIR = "./chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ─── Step 1: Load Documents ───────────────────────────────────────────────────
def load_documents(pdf_paths: list) -> list:
    """Load all PDF documents and return a flat list of pages."""
    all_docs = []
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"WARNING: File not found: {path}")
            continue
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Loaded {len(docs)} pages from: {path}")
    return all_docs


# ─── Step 2: Split into Chunks ────────────────────────────────────────────────
def split_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks for better retrieval.
    
    BUG IS SOMEWHERE IN THIS FUNCTION.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=500,   # <-- examine this value carefully
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


# ─── Step 3: Create Vector Store ─────────────────────────────────────────────
def create_vector_store(chunks: list) -> Chroma:
    """
    Embed chunks and store in ChromaDB.
    
    BUG IS SOMEWHERE IN THIS FUNCTION.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=OPENAI_API_KEY
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # Persist the vector store to disk
    vector_store.persist()
    print(f"Vector store created and persisted to: {CHROMA_PERSIST_DIR}")
    return vector_store


# ─── Step 4: Build RAG Chain ──────────────────────────────────────────────────
def build_rag_chain(vector_store: Chroma) -> RetrievalQA:
    """
    Build the RetrievalQA chain with a custom prompt.
    The chain must:
    - Only answer from the provided documents
    - Return the source document name alongside the answer
    - Refuse to answer if the information is not in the documents
    
    TWO BUGS ARE SOMEWHERE IN THIS FUNCTION.
    """
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.9,     # <-- examine this value for a RAG use case
        openai_api_key=OPENAI_API_KEY,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # Custom prompt to enforce grounding
    prompt_template = """You are a knowledgeable assistant for an advertising agency.
Use ONLY the following context to answer the question.
If the answer is not in the context, say exactly: "I cannot find this information in the provided documents."
Do NOT use any outside knowledge.

Context:
{context}

Question: {question}

Answer (include the source document name if relevant):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,   # <-- examine this for the task requirement
        chain_type_kwargs={"prompt": PROMPT},
    )

    print("RAG chain built successfully")
    return chain


# ─── Step 5: Query the Chain ─────────────────────────────────────────────────
def query_rag(chain: RetrievalQA, question: str) -> dict:
    """Query the RAG chain and return the answer with source information."""
    print(f"\nQuestion: {question}")
    print("-" * 60)

    result = chain.invoke({"query": question})

    answer = result.get("result", "No answer returned")
    sources = result.get("source_documents", [])

    # Extract unique source file names
    source_names = list(set(
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in sources
    )) if sources else ["No sources returned"]

    return {
        "question": question,
        "answer": answer,
        "sources": source_names,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ADVERTISING AGENCY — RAG KNOWLEDGE BOT")
    print("=" * 60)

    # Pipeline
    docs = load_documents(PDF_PATHS)
    if not docs:
        print("ERROR: No documents loaded. Check PDF paths.")
        return

    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)
    chain = build_rag_chain(vector_store)

    # Test questions — the bot should answer ALL from the documents
    test_questions = [
        "What was the total budget for the Lumiere 'Glow With Confidence' campaign?",
        "What ROAS did the Velocity 'No Limits' campaign achieve?",
        "According to the Noir brand guidelines, what words should we NEVER use in copy?",
        "Which channel had the highest ROAS in the Lumiere campaign?",
        "What is Noir Espresso's policy on using emoji in Instagram captions?",
        "Who was the primary target audience for the Velocity sports drink campaign?",
        # This should trigger refusal — not in any document
        "What is the capital city of Brazil?",
    ]

    print("\n" + "=" * 60)
    print("RUNNING TEST QUERIES")
    print("=" * 60)

    for question in test_questions:
        result = query_rag(chain, question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
        print("-" * 60)


if __name__ == "__main__":
    main()


# ─── ANSWER KEY (For Evaluator Only — Remove Before Distributing) ─────────────
"""
BUG 1 — split_documents() — Line ~55:
    chunk_overlap=500 is equal to chunk_size=500.
    Overlap must always be LESS than chunk_size.
    When overlap >= chunk_size, the splitter creates infinite or nonsensical 
    chunks because each chunk would fully overlap the previous one.
    FIX: chunk_overlap=100 (or any value < 500, typically 10-20% of chunk_size)

BUG 2 — create_vector_store() — The .persist() call:
    In newer versions of LangChain/Chroma (chromadb >= 0.4.0), 
    calling vector_store.persist() is deprecated and raises an error.
    The data is auto-persisted when persist_directory is set.
    FIX: Remove the vector_store.persist() line entirely.
    (Candidate must recognise version-aware API deprecation)

BUG 3 — build_rag_chain() — temperature=0.9:
    For a RAG system that must return factual, grounded answers from documents,
    a high temperature (0.9) causes the model to be too creative/random,
    leading to hallucinated answers that drift from the source material.
    FIX: temperature=0.0 (or at most 0.1 for minimal stylistic variation)

BUG 4 — build_rag_chain() — return_source_documents=False:
    The task requirement is to return the answer WITH the source document name.
    With return_source_documents=False, the 'source_documents' key is absent 
    from the result dict, so query_rag() always returns "No sources returned".
    FIX: return_source_documents=True
"""
