import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="PDF Chatbot (RAG)", page_icon="📄", layout="wide")

# API Key Validation
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Missing `GROQ_API_KEY`. Add it to your Streamlit secrets (`.streamlit/secrets.toml`).")
    st.stop()

# Cache HuggingFace Embeddings
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing document..."):
            vectorstore = process_pdf(uploaded_file)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            # 1. Prompt to rephrase question considering history
            rephrase_prompt = ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the user's latest question, formulate a standalone question. Do NOT answer it, just rephrase it if necessary or return it as is."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            rephrase_chain = rephrase_prompt | llm | StrOutputParser()

            # 2. Main Q&A Prompt
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant for question-answering tasks. Use the retrieved context to answer the question. If the context does not contain the answer, say you don't know. Keep it clear and concise.\n\nContext:\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            # 3. Context retrieval logic with history
            def retrieve_context(inputs):
                if inputs.get("chat_history"):
                    standalone_query = rephrase_chain.invoke(inputs)
                else:
                    standalone_query = inputs["input"]
                matched_docs = retriever.invoke(standalone_query)
                return format_docs(matched_docs)

            # 4. Final RAG Chain
            rag_chain = (
                RunnablePassthrough.assign(context=retrieve_context)
                | qa_prompt
                | llm
                | StrOutputParser()
            )

            st.session_state.rag_chain = rag_chain
            st.session_state.messages = []
            st.success("Ready! You can now chat with your PDF.")

# Chat Interface
st.title("📚 Chat with your PDF")

if not st.session_state.rag_chain:
    st.info("👈 Upload and process a PDF from the sidebar to start chatting.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        chat_history = [
            (msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})