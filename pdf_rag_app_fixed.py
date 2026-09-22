import streamlit as st
import fitz
import re
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄")

st.title("📄 PDF RAG Assistant")
st.write("Upload a PDF and ask questions about its contents.")

# Get the Groq key from Streamlit Secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = ""

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it under Manage app → Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(uploaded_file):
    data = uploaded_file.getvalue()
    pdf = fitz.open(stream=data, filetype="pdf")

    pages = []

    for page_number, page in enumerate(pdf, start=1):
        text = clean_text(page.get_text("text"))

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    pdf.close()
    return pages


def create_chunks(pages, chunk_size=1800, overlap=300):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()

            if piece:
                chunks.append({
                    "page": page["page"],
                    "text": piece
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks


def retrieve_chunks(question, chunks, k=6):
    documents = [x["text"] for x in chunks]

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(documents)
        question_vector = vectorizer.transform([question])
        scores = cosine_similarity(question_vector, matrix).flatten()

        ranked = scores.argsort()[::-1]

        selected = [
            chunks[i]
            for i in ranked[:k]
            if scores[i] > 0
        ]

        # Prevent a useful PDF from returning "not found" just because
        # the wording of the question differs from the PDF wording.
        if not selected:
            selected = chunks[:min(k, len(chunks))]

        return selected

    except ValueError:
        return chunks[:min(k, len(chunks))]


def ask_groq(question, chunks):
    context = "\n\n---\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in chunks
    )

    prompt = f"""
You are a PDF question answering assistant.

Answer the question using the PDF context below.

Rules:
- Give a direct and useful answer.
- Use the PDF as the main source.
- Do not invent facts.
- If the answer is supported by the context, answer it even if the
  wording of the question is different from the wording in the PDF.
- If useful, mention the relevant page number.
- Only say the information is unavailable when the context genuinely
  does not contain enough information.

PDF CONTEXT:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "Answer questions about the uploaded PDF accurately and clearly."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_completion_tokens=1200,
        include_reasoning=False
    )

    return response.choices[0].message.content.strip()


if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []


uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:

    if uploaded_file.name != st.session_state.file_name:

        with st.spinner("Reading and indexing the PDF..."):
            pages = extract_pdf_text(uploaded_file)

            if not pages:
                st.error(
                    "No readable text was found in this PDF. "
                    "If it is a scanned/image-only PDF, OCR is required."
                )
                st.stop()

            chunks = create_chunks(pages)

            st.session_state.file_name = uploaded_file.name
            st.session_state.chunks = chunks
            st.session_state.messages = []

        st.success(
            f"PDF ready — {len(pages)} pages and {len(chunks)} chunks indexed."
        )

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF..."):
                try:
                    relevant_chunks = retrieve_chunks(
                        question,
                        st.session_state.chunks
                    )

                    answer = ask_groq(question, relevant_chunks)
                    st.markdown(answer)

                except Exception as error:
                    answer = "Something went wrong while generating the answer."
                    st.error(answer)
                    st.caption(f"Technical details: {error}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("Upload a PDF to get started.")
