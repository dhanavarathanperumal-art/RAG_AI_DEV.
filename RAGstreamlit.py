# RAGstreamlit.py
import os
import tempfile
from typing import List, Tuple

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Offline Resume RAG (No Internet)",
    page_icon="📄",
    layout="wide"
)

# -------------------------
# Helper: save uploaded file with ORIGINAL NAME
# -------------------------
def save_uploaded_file(uploaded_file) -> Tuple[str, str]:
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path, uploaded_file.name


# -------------------------
# Build vectorstore (OFFLINE)
# -------------------------
def build_vectorstore_from_files(files: List[Tuple[str, str]]):
    all_docs: List[Document] = []

    for file_path, original_name in files:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = original_name  # ✅ REAL filename

        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No documents loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# -------------------------
# UI
# -------------------------
st.title("📄 Offline Resume RAG (No Internet)")
st.markdown(
    """
    ✅ Works fully offline  
    ✅ No API keys  
    ✅ No LLMs  
    🔍 Finds most relevant resume sections
    """
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# -------------------------
# Index button
# -------------------------
if uploaded_files and st.button("Index resumes"):
    with st.spinner("Indexing resumes..."):
        try:
            files = [save_uploaded_file(f) for f in uploaded_files]
            st.session_state.vector_db = build_vectorstore_from_files(files)
            st.success(f"Indexed {len(files)} resumes successfully!")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

# -------------------------
# Search (Semantic search only)
# -------------------------
if st.session_state.vector_db:
    query = st.text_input("Search resumes (e.g. Python, Django, 5 years)")

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching resumes..."):
                results = st.session_state.vector_db.similarity_search(
                    query,
                    k=5
                )

                st.subheader("🔍 Top Matching Resume Sections")

                for i, doc in enumerate(results, 1):
                    st.markdown(f"### {i}. 📄 {doc.metadata.get('source')}")
                    st.code(doc.page_content)
