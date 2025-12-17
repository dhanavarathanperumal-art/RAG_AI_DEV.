# RAGstreamlit.py
import os
import tempfile
from typing import List

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 🔄 CHANGED: embeddings (OpenAI → HuggingFace)
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔄 CHANGED: LLM still ChatOpenAI but used for DeepSeek
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

st.set_page_config(
    page_title="Resume RAG Bot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# Sidebar / API key
# -------------------------
st.sidebar.header("🔑 DeepSeek API Key")
deepseek_key = st.sidebar.text_input("Enter DeepSeek API key", type="password")

if deepseek_key:
    os.environ["OPENAI_API_KEY"] = deepseek_key  # required by langchain_openai


# -------------------------
# Helper: save uploaded file
# -------------------------
def save_uploaded_file(uploaded_file) -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tf.write(uploaded_file.getbuffer())
    tf.close()
    return tf.name


# -------------------------
# Build vectorstore
# -------------------------
def build_vectorstore_from_pdf_paths(pdf_paths: List[str]):
    all_docs: List[Document] = []

    for p in pdf_paths:
        loader = PyPDFLoader(p)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = os.path.basename(p)

        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No documents loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    # 🔄 CHANGED: FREE embeddings (NO API, NO BILLING)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# -------------------------
# UI
# -------------------------
st.title("📄 Resume RAG Chatbot")
st.markdown(
    "Upload bulk **PDF resumes** and ask questions like "
    "**Who has 5+ years of Python experience?**"
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# Index button
if uploaded_files and st.button("Index uploaded resumes ✅"):
    with st.spinner("Indexing resumes..."):
        try:
            paths = [save_uploaded_file(f) for f in uploaded_files]
            st.session_state.vector_db = build_vectorstore_from_pdf_paths(paths)
            st.success(f"Indexed {len(paths)} resumes successfully!")
        except Exception as e:
            st.error(f"Indexing failed: {e}")


# -------------------------
# Search (RAG)
# -------------------------
if st.session_state.vector_db:
    query = st.text_input("Ask a question about candidates")

    if st.button("Search 🔎"):
        if not deepseek_key:
            st.warning("Please enter your DeepSeek API key")
        elif not query.strip():
            st.warning("Please enter a query")
        else:
            with st.spinner("Searching..."):
                retriever = st.session_state.vector_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )

                # 🔄 CHANGED: DeepSeek LLM
                llm = ChatOpenAI(
                    model="deepseek-chat",
                    base_url="https://api.deepseek.com",
                    api_key=deepseek_key,
                    temperature=0
                )

                prompt = ChatPromptTemplate.from_template(
                    """
                    Use the following resume context to answer the question.
                    If the answer is not found, say so clearly.

                    Context:
                    {context}

                    Question:
                    {question}
                    """
                )

                chain = (
                    {"context": retriever, "question": lambda x: x}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                answer = chain.invoke(query)

                st.subheader("✅ Answer")
                st.write(answer)
