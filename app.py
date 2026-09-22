"""
PDF RAG chatbot — Streamlit frontend + Groq LLM.

How it works
------------
1. You upload a PDF.
2. The app extracts the text, splits it into overlapping chunks, and embeds
   each chunk locally with sentence-transformers (no extra API key needed —
   this runs on the server, not through Groq).
3. When you ask a question, the app finds the most similar chunks (cosine
   similarity) and sends them to a Groq-hosted LLM along with your question,
   so the answer is grounded in the document instead of hallucinated.

Deployment
----------
Push this repo (app.py + requirements.txt) to GitHub, then deploy on
Streamlit Community Cloud. Add your Groq key under
Settings -> Secrets as:

    GROQ_API_KEY = "gsk_..."

Get a key at https://console.groq.com/keys
"""

import os
from typing import List, Tuple

import numpy as np
import streamlit as st
from pypdf import PdfReader
from groq import Groq

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs locally (CPU is fine)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")

# --------------------------------------------------------------------------
# Cached resources (loaded once per server process)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_api_key() -> str:
    """Groq key from Streamlit secrets > env var > manual sidebar entry."""
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not key:
        key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        key = st.session_state.get("manual_api_key", "")
    return key


def get_groq_client():
    api_key = get_api_key()
    return Groq(api_key=api_key) if api_key else None


# --------------------------------------------------------------------------
# RAG helpers
# --------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def embed_texts(embedder, texts: List[str]) -> np.ndarray:
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def retrieve_relevant_chunks(
    embedder, question: str, chunks: List[str], chunk_embeddings: np.ndarray, top_k: int = 4
) -> List[Tuple[str, float]]:
    query_emb = embed_texts(embedder, [question])[0]
    scores = chunk_embeddings @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


def ask_groq(client, model_name, question, context_chunks, temperature=0.2) -> str:
    context = "\n\n".join(f"[Excerpt {i+1}]\n{chunk}" for i, (chunk, _) in enumerate(context_chunks))
    system_prompt = (
        "You are a helpful assistant answering questions about a document. "
        "Use ONLY the excerpts provided below to answer. If the answer is not "
        "contained in the excerpts, say you don't know based on the document — "
        "do not make anything up."
    )
    user_prompt = f"Document excerpts:\n\n{context}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
DEFAULTS = {
    "chunks": [],
    "chunk_embeddings": None,
    "doc_name": None,
    "chat_history": [],
    "manual_api_key": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if get_api_key():
        st.success("Groq API key detected.")
    else:
        st.warning("No Groq API key found in Streamlit secrets.")
        st.session_state.manual_api_key = st.text_input(
            "Groq API key (local testing only)", type="password"
        )
        st.caption("For deployment, add GROQ_API_KEY under Settings → Secrets instead.")

    model_name = st.selectbox("Groq model", GROQ_MODELS, index=0)
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=8, value=4)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    if st.button("🗑️ Clear document & chat"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    st.divider()
    st.caption(
        "Text is extracted from your PDF, chunked, and embedded locally "
        "(sentence-transformers). The most relevant chunks are retrieved for "
        "each question and sent to Groq to generate a grounded answer."
    )

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF, then ask questions about it — answers are grounded in the document (RAG).")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.doc_name:
    with st.spinner("Reading and indexing your PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        if not text:
            st.error(
                "No extractable text found in this PDF. It might be a scanned "
                "document without OCR — try a different file."
            )
        else:
            embedder = load_embedder()
            chunks = chunk_text(text)
            st.session_state.chunks = chunks
            st.session_state.chunk_embeddings = embed_texts(embedder, chunks)
            st.session_state.doc_name = uploaded_file.name
            st.session_state.chat_history = []
            st.success(f"Indexed **{uploaded_file.name}** into {len(chunks)} chunks. Ask away!")

if st.session_state.chunks:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources used"):
                    for i, (chunk, score) in enumerate(msg["sources"]):
                        st.markdown(f"**Excerpt {i+1}** (similarity {score:.2f})")
                        st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))

    question = st.chat_input("Ask a question about the document...")
    if question:
        client = get_groq_client()
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if client is None:
                answer = (
                    "⚠️ No Groq API key configured. Add it under Settings → Secrets "
                    "(or the sidebar for local testing)."
                )
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            else:
                with st.spinner("Thinking..."):
                    embedder = load_embedder()
                    sources = retrieve_relevant_chunks(
                        embedder, question, st.session_state.chunks,
                        st.session_state.chunk_embeddings, top_k=top_k,
                    )
                    try:
                        answer = ask_groq(client, model_name, question, sources, temperature)
                    except Exception as e:
                        answer = f"⚠️ Error calling Groq API: {e}"
                st.markdown(answer)
                with st.expander("Sources used"):
                    for i, (chunk, score) in enumerate(sources):
                        st.markdown(f"**Excerpt {i+1}** (similarity {score:.2f})")
                        st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
else:
    st.info("👆 Upload a PDF to get started.")
