import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(page_title="PDF Chatbot (RAG)", page_icon="📄", layout="wide")

# 1. API Key Handling
groq_api_key = st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Missing `GROQ_API_KEY`. Please add it to your Streamlit secrets (`.streamlit/secrets.toml`).")
    st.stop()

# 2. Embedding Model Caching (Loads lightweight, high-performance open-source embedding model)
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# 3. Helper function to process PDF and create vectorstore
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

# 4. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 5. Sidebar for PDF Upload
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Parsing and indexing document..."):
            vectorstore = process_pdf(uploaded_file)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Setup LLM via Groq
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            # Contextualize Question Prompt (Rephrasing based on history)
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

            # Answering Prompt
            qa_system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer or it's not in the context, say that you don't know. "
                "Keep the answer concise and clear.\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            # Save state
            st.session_state.rag_chain = rag_chain
            st.session_state.messages = []
            st.success("PDF processed successfully! You can now start asking questions.")

# 6. Chat Interface
st.title("📚 Chat with your PDF")

if not st.session_state.rag_chain:
    st.info("👈 Please upload and process a PDF from the sidebar to start chatting.")
else:
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if user_input := st.chat_input("Ask a question about the document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare chat history for LangChain
        chat_history = [
            (msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]
        ]

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                answer = response.get("answer", "Sorry, I could not generate an answer.")
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})