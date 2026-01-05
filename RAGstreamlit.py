# RAGstreamlit_offline_top15.py
import os
import tempfile
from typing import List, Tuple, Dict

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
    page_title="Offline Resume Search (Top 15)",
    page_icon="📄",
    layout="wide"
)


# -------------------------
# Save uploaded PDF
# -------------------------
def save_uploaded_file(uploaded_file) -> Tuple[str, str]:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path, uploaded_file.name


# -------------------------
# Build FAISS (offline)
# -------------------------
def build_vectorstore(files: List[Tuple[str, str]]):
    all_docs: List[Document] = []

    for file_path, filename in files:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = filename  # resume name

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# -------------------------
# Group chunks → resumes
# -------------------------
def top_resumes(
    docs_with_scores: List[Tuple[Document, float]],
    limit: int = 15
):
    """
    FAISS gives chunk-level results.
    This groups them by resume (source) and
    keeps ONLY the best chunk per resume.
    """

    best_per_resume: Dict[str, Tuple[Document, float]] = {}

    for doc, score in docs_with_scores:
        src = doc.metadata["source"]

        # smaller score = better similarity
        if src not in best_per_resume or score < best_per_resume[src][1]:
            best_per_resume[src] = (doc, score)

    # sort resumes by best score
    ranked = sorted(
        best_per_resume.values(),
        key=lambda x: x[1]
    )

    return ranked[:limit]


# -------------------------
# UI
# -------------------------
st.title("📄 Offline Resume Search (Vector Only)")
st.markdown(
    """
    ✅ Fully offline  
    ✅ No API / No LLM  
    🔍 Query and resumes are compared using vectors  
    🏆 Top **15 resumes** shown (no duplicates)
    """
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


# -------------------------
# Index resumes
# -------------------------
if uploaded_files and st.button("Index resumes"):
    with st.spinner("Indexing resumes..."):
        files = [save_uploaded_file(f) for f in uploaded_files]
        st.session_state.vector_db = build_vectorstore(files)
        st.success(f"Indexed {len(files)} resumes successfully!")


# -------------------------
# Search
# -------------------------
if st.session_state.vector_db:
    query = st.text_input(
        "Search resumes (e.g. Python, Django, Linux, 5 years)"
    )

    if st.button("Search"):
        if not query.strip():
            st.warning("Enter a search query")
        else:
            with st.spinner("Searching..."):
                # STEP 1: query vector vs resume vectors
                raw_results = st.session_state.vector_db.similarity_search_with_score(
                    query,
                    k=60   # large pool
                )

                # STEP 2: group by resume
                top_results = top_resumes(raw_results, limit=15)

                if not top_results:
                    st.warning("No matching resumes found.")
                else:
                    st.subheader("🏆 Top 15 Matching Resumes")

                    for idx, (doc, score) in enumerate(top_results, 1):
                        st.markdown(
                            f"### {idx}. 📄 {doc.metadata['source']}  |  score: `{score:.4f}`"
                        )

                        text = doc.page_content.strip()
                        if len(text) > 2000:
                            text = text[:2000] + "\n\n[...truncated...]"

                        st.code(text)
