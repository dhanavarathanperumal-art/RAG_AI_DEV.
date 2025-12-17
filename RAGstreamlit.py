# app.py
import os
import tempfile
from typing import List

import streamlit as st

# LangChain imports (community integrations where needed)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

st.set_page_config(page_title="Resume RAG Bot", page_icon="🤖", layout="wide")


# -------------------------
# Sidebar / API key
# -------------------------
st.sidebar.header("🔑 OpenAI API Key")
openai_key = st.sidebar.text_input("Enter OpenAI API key", type="password")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key


# -------------------------
# Helper: save uploaded file to disk (PyPDFLoader expects a filepath)
# -------------------------
def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a real temporary file and return its path."""
    suffix = "" if uploaded_file.name.lower().endswith(".pdf") else ".pdf"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded_file.getbuffer())
    tf.flush()
    tf.close()
    return tf.name


# -------------------------
# Process resumes -> vector DB
# -------------------------
@st.cache_data(show_spinner=False)
def build_vectorstore_from_pdf_paths(pdf_paths: List[str]):
    """
    Load PDFs, split text into chunks, embed and create FAISS vectorstore.
    Returns: FAISS object (in-memory)
    """
    # 1) load documents (page-level)
    all_docs: List[Document] = []
    for p in pdf_paths:
        loader = PyPDFLoader(p)
        # loader.load() returns list of Document objects; for many pdfs use load() or load_and_split()
        docs = loader.load()
        # tag metadata with source filename for easier traceability
        for d in docs:
            d.metadata["source"] = os.path.basename(p)
        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No documents loaded from provided PDFs.")

    # 2) split into smaller chunks for embeddings & retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)

    # 3) embeddings
    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env
    # 4) build FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# -------------------------
# Streamlit UI
# -------------------------
st.title("📄 Resume RAG Chatbot")
st.markdown(
    "Upload bulk **PDF resumes** (multiple files). The bot will index them and let you ask recruiter-style queries "
    "like **'Who has 5+ years in Python?'** or **'Show me John Doe's experience'**."
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes (multiple files allowed)",
    type=["pdf"],
    accept_multiple_files=True,
)

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    st.session_state.indexed_files = []


# Button to (re)index files
if uploaded_files:
    if st.button("Index uploaded resumes ✅"):
        with st.spinner("Processing PDFs and building vector index..."):
            try:
                saved_paths = []
                for uf in uploaded_files:
                    path = save_uploaded_file(uf)
                    saved_paths.append(path)

                vs = build_vectorstore_from_pdf_paths(saved_paths)
                st.session_state.vector_db = vs
                st.session_state.indexed_files = [os.path.basename(p) for p in saved_paths]
                st.success(f"Indexed {len(saved_paths)} PDFs. Ready to search!")
            except Exception as e:
                st.error(f"Failed to index resumes: {e}")


# If already indexed, show indexed files
if st.session_state.vector_db:
    st.info(f"Indexed files: {', '.join(st.session_state.indexed_files)}")
    query = st.text_input("Ask about candidates (e.g. 'Who knows Django?', 'Show resume of Rahul')")

    # Retrieval settings
    k = st.sidebar.number_input("Retriever: top k results", min_value=1, max_value=10, value=4)
    model_name = st.sidebar.selectbox("LLM model", options=["gpt-4o-mini", "gpt-4o-mini-rc", "gpt-4o"], index=0)

    if st.button("Search 🔎"):
        if not query.strip():
            st.warning("Please enter a question or query.")
        elif not openai_key:
            st.warning("Please provide your OpenAI API key in the sidebar.")
        else:
            with st.spinner("Running retrieval + LLM..."):
                try:
                    retriever = st.session_state.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k})

                    llm = ChatOpenAI(model=model_name, temperature=0.0, max_tokens=1024)

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                    )

                    result = qa_chain({"query": query})
                    answer = result.get("result") or result.get("answer") or result
                    source_docs = result.get("source_documents", [])

                    st.markdown("### 🔍 Answer")
                    st.write(answer)

                    if source_docs:
                        st.markdown("#### 🧾 Source chunks")
                        for i, sd in enumerate(source_docs, start=1):
                            src = sd.metadata.get("source", "unknown")
                            # show first ~500 chars of the chunk
                            snippet = sd.page_content[:500].strip()
                            st.markdown(f"**{i}.** from `{src}` — _snippet:_")
                            st.code(snippet + ("..." if len(sd.page_content) > 500 else ""))

                except Exception as e:
                    st.error(f"Search failed: {e}")


# Helpful tips / footer
st.markdown("---")
st.markdown(
    "Tips: 1) Upload clean text PDFs when possible (not scanned images).  "
    "2) Use queries like `Show me candidate named <Name>` or `Who has experience with <technology>`."
)

