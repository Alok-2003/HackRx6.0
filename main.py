# main.py
# FINAL, COMPETITION-GRADE High-Accuracy API Server for HackRx 6.0
# Architecture: Hybrid Parsing (unstructured w/ fitz fallback), Hybrid Search, and Re-ranking

import os
import asyncio
from typing import List, Dict, Any
import time

# --- Core FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl

# --- AI and ML Imports ---
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from rank_bm25 import BM25Okapi  # For keyword search

# --- Utility Imports ---
import httpx  # For async HTTP requests
import fitz  # PyMuPDF for RELIABLE PDF reading (our fallback)
from unstructured.partition.pdf import partition_pdf # ADVANCED: For intelligent PDF parsing
from unstructured.chunking.title import chunk_by_title # ADVANCED: For smart chunking
from dotenv import load_dotenv

# --- Environment and Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_AUTH_TOKEN = "e31e480650ef213ef618fe685acb61ff925d2780a2853e489b73eec846a6a0a7"

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Global State and Caching ---
app_state: Dict[str, Any] = {
    "embedding_model": None, # For semantic search
    "reranker_model": None,  # For re-ranking search results
    "document_cache": {},
}

# --- FastAPI Application Setup ---
app = FastAPI(
    title="PolicyGuard AI for HackRx 6.0 (Advanced)",
    description="An advanced API with Hybrid Search and Re-ranking for high-accuracy document analysis.",
    version="4.0.0"
)
auth_scheme = HTTPBearer()

def check_auth_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials != HACKRX_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials.credentials

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Pre-loads all AI models into memory on server start."""
    print("Server starting up...")
    if not GEMINI_API_KEY:
        print("CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")
    
    print("Loading embedding model (for semantic search)...")
    app_state["embedding_model"] = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    
    print("Loading reranker model (for accuracy boost)...")
    app_state["reranker_model"] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    print("All models loaded successfully.")

# --- Core Logic Functions ---

async def process_document_hybrid(document_url: str) -> List[str]:
    """
    HYBRID PARSING: Tries the advanced 'unstructured' parser first for tables and layout.
    If it fails, it falls back to the reliable 'PyMuPDF' parser.
    """
    print(f"Processing document from URL: {document_url}")
    temp_filename = "temp_doc.pdf"
    try:
        # Download the file once
        async with httpx.AsyncClient() as client:
            response = await client.get(document_url, timeout=60.0)
            response.raise_for_status()
            pdf_data = response.content
            with open(temp_filename, "wb") as f:
                f.write(pdf_data)
        
        # --- Attempt 1: Advanced Parsing with 'unstructured' ---
        try:
            print("Attempting advanced parsing with 'unstructured'...")
            elements = partition_pdf(
                filename=temp_filename, 
                strategy="hi_res",
                infer_table_structure=True,
                extract_images_in_pdf=False
            )
            chunks = chunk_by_title(elements, max_characters=1500, combine_under_n_chars=500)
            chunk_texts = [chunk.text for chunk in chunks if chunk.text.strip()]
            if not chunk_texts:
                raise ValueError("Unstructured parsing yielded no text chunks.")
            print(f"Advanced parsing successful. Document chunked into {len(chunk_texts)} pieces.")
            return chunk_texts
        except Exception as e:
            print(f"Advanced parsing failed: {e}. Falling back to reliable parser.")

            # --- Attempt 2: Reliable Fallback with 'PyMuPDF' ---
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + "\n"
            doc.close()

            if not full_text.strip():
                raise ValueError("Both parsing methods failed to extract text.")
            
            # Simple but robust chunking for the fallback text
            paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 100]
            print(f"Fallback parsing successful. Document chunked into {len(paragraphs)} pieces.")
            return paragraphs

    except Exception as e:
        print(f"Fatal error during document processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF document: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


async def get_answer_for_question(question: str, doc_chunks: List[str], doc_embeddings: np.ndarray, bm25_index) -> str:
    """
    Performs hybrid search (semantic + keyword), re-ranks results, and gets an answer from Gemini.
    """
    model = app_state["embedding_model"]
    reranker = app_state["reranker_model"]
    
    # --- 1. Hybrid Search ---
    tokenized_query = question.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-20:]

    question_embedding = model.encode(question)
    semantic_scores = np.dot(doc_embeddings, question_embedding)
    top_semantic_indices = np.argsort(semantic_scores)[-20:]

    combined_indices = list(set(top_bm25_indices) | set(top_semantic_indices))
    
    # --- 2. Re-ranking ---
    rerank_pairs = [[question, doc_chunks[i]] for i in combined_indices]
    if not rerank_pairs:
        return "The answer to this question is not available in the provided document."

    rerank_scores = reranker.predict(rerank_pairs)
    sorted_indices = [idx for _, idx in sorted(zip(rerank_scores, combined_indices), reverse=True)]
    
    # --- 3. Context Assembly & Generation ---
    top_k = 7 # Use a slightly larger context for complex questions
    final_indices = sorted_indices[:top_k]
    relevant_chunks = [doc_chunks[i] for i in final_indices]
    context = "\n\n---\n\n".join(relevant_chunks)
    
    print(f"Top {top_k} re-ranked chunks selected for question: '{question}'.")

    # Precision-focused "Chain-of-Thought" Prompt
    prompt = f"""
    You are a world-class AI system for policy analysis. Your task is to answer the user's question with extreme precision, based ONLY on the provided context.

    Follow these steps to generate your answer:
    1.  First, carefully read the user's **QUESTION** and the **CONTEXT** provided.
    2.  Identify the exact sentences, table rows, or data points within the **CONTEXT** that directly answer the **QUESTION**.
    3.  Synthesize these key pieces of information into a concise and clear answer. Quote numbers and specific terms exactly as they appear.
    4.  **CRITICAL RULE:** If the context does not contain the information needed to answer the question, you MUST respond with the single phrase: "The answer to this question is not available in the provided document." Do not apologize or explain further.

    **CONTEXT FROM DOCUMENT:**
    {context}

    **QUESTION:**
    {question}

    **PRECISE ANSWER:**
    """

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=45.0
            )
            response.raise_for_status()
            result = response.json()
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            return answer.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Error: The AI model failed to generate a response."

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_submission(request: HackRxRequest, token: str = Security(check_auth_token)):
    start_time = time.time()
    doc_url = str(request.documents)
    model = app_state["embedding_model"]
    cache = app_state["document_cache"]

    if doc_url in cache:
        print("Cache hit! Using pre-processed document.")
        cached_data = cache[doc_url]
    else:
        print("Cache miss. Processing new document.")
        doc_chunks = await process_document_hybrid(doc_url)
        
        if not doc_chunks:
            answers = ["Could not process the document to find answers."] * len(request.questions)
            return HackRxResponse(answers=answers)

        doc_embeddings = model.encode(doc_chunks, show_progress_bar=True)
        tokenized_chunks = [chunk.lower().split() for chunk in doc_chunks]
        bm25_index = BM25Okapi(tokenized_chunks)
        
        cached_data = {
            "chunks": doc_chunks, 
            "embeddings": doc_embeddings, 
            "bm25_index": bm25_index
        }
        cache[doc_url] = cached_data

    tasks = [
        get_answer_for_question(
            q, 
            cached_data["chunks"], 
            cached_data["embeddings"], 
            cached_data["bm25_index"]
        ) for q in request.questions
    ]
    
    print(f"Starting concurrent processing for {len(tasks)} questions...")
    answers = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"All questions processed in {end_time - start_time:.2f} seconds.")
    
# --- Home Route ---
@app.get("/")
def home():
    """Simple home route that returns a welcome message."""
    return {"message": "Welcome to HackRx 6.0 API Server", "status": "running"}

    return HackRxResponse(answers=answers)