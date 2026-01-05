"""
🚀 ENTERPRISE RESUME SEARCH ENGINE
Features:
- Multi-database backend (SQLite + Chroma/FAISS)
- Advanced skill & entity extraction
- Resume ranking & scoring system
- Batch processing & persistence
- Export capabilities
- Admin dashboard
"""

import os
import sys
import json
import sqlite3
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import re

import streamlit as st
# Removed: from streamlit_ace import st_ace
import plotly.express as px
import plotly.graph_objects as go

# PDF Processing
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

# Vector Databases
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# NLP & ML
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================== CONFIGURATION & CONSTANTS ====================
@dataclass
class AppConfig:
    """Application configuration"""
    # Database paths
    SQLITE_DB = "resume_database.db"
    CHROMA_PATH = "chroma_db"
    FAISS_PATH = "faiss_index"
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SPACY_MODEL = "en_core_web_sm"
    
    # Processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    MAX_FILE_SIZE_MB = 50
    
    # Search
    DEFAULT_K = 50
    SCORE_THRESHOLD = 0.3
    
    # UI
    PAGE_TITLE = "🏢 Enterprise Resume Intelligence"
    THEME = "dark"

# Initialize config
config = AppConfig()

# ==================== DATABASE LAYER ====================
class ResumeDatabase:
    """SQLite database for structured resume data"""
    
    def __init__(self, db_path: str = config.SQLITE_DB):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main resumes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE,
                filename TEXT,
                file_path TEXT,
                file_size INTEGER,
                upload_date TIMESTAMP,
                processed_date TIMESTAMP,
                candidate_name TEXT,
                candidate_email TEXT,
                candidate_phone TEXT,
                total_experience_years REAL,
                current_company TEXT,
                current_title TEXT,
                education_level TEXT,
                university TEXT,
                skills_json TEXT,
                extracted_text TEXT,
                metadata_json TEXT,
                status TEXT DEFAULT 'pending',
                version INTEGER DEFAULT 1
            )
        ''')
        
        # Skills mapping table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resume_skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER,
                skill_name TEXT,
                skill_category TEXT,
                confidence_score REAL,
                mention_count INTEGER,
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
        ''')
        
        # Experience table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id INTEGER,
                company TEXT,
                title TEXT,
                start_date TEXT,
                end_date TEXT,
                duration_months INTEGER,
                description TEXT,
                technologies_json TEXT,
                FOREIGN KEY (resume_id) REFERENCES resumes (id)
            )
        ''')
        
        # Search history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                filters_json TEXT,
                results_count INTEGER,
                search_date TIMESTAMP,
                user_id TEXT
            )
        ''')
        
        # Analytics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                category TEXT,
                timestamp TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_resume(self, resume_data: Dict) -> int:
        """Save resume to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate file hash
        content_hash = hashlib.md5(
            (resume_data['filename'] + str(resume_data.get('extracted_text', ''))).encode()
        ).hexdigest()
        
        # Check if resume already exists
        cursor.execute('SELECT id FROM resumes WHERE file_hash = ?', (content_hash,))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return existing[0]
        
        # Insert new resume
        cursor.execute('''
            INSERT INTO resumes (
                file_hash, filename, file_path, file_size, upload_date,
                candidate_name, candidate_email, candidate_phone,
                total_experience_years, current_company, current_title,
                education_level, university, skills_json, extracted_text,
                metadata_json, processed_date, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            content_hash,
            resume_data['filename'],
            resume_data.get('file_path', ''),
            resume_data.get('file_size', 0),
            datetime.now().isoformat(),
            resume_data.get('candidate_name', ''),
            resume_data.get('candidate_email', ''),
            resume_data.get('candidate_phone', ''),
            resume_data.get('total_experience_years', 0),
            resume_data.get('current_company', ''),
            resume_data.get('current_title', ''),
            resume_data.get('education_level', ''),
            resume_data.get('university', ''),
            json.dumps(resume_data.get('skills', {})),
            resume_data.get('extracted_text', ''),
            json.dumps(resume_data.get('metadata', {})),
            datetime.now().isoformat(),
            'processed'
        ))
        
        resume_id = cursor.lastrowid
        
        # Save skills
        skills = resume_data.get('skills', {})
        for category, skill_list in skills.items():
            for skill in skill_list:
                cursor.execute('''
                    INSERT INTO resume_skills (resume_id, skill_name, skill_category)
                    VALUES (?, ?, ?)
                ''', (resume_id, skill, category))
        
        # Save experiences
        experiences = resume_data.get('experiences', [])
        for exp in experiences:
            cursor.execute('''
                INSERT INTO experiences (
                    resume_id, company, title, start_date, end_date,
                    duration_months, description, technologies_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                resume_id,
                exp.get('company', ''),
                exp.get('title', ''),
                exp.get('start_date', ''),
                exp.get('end_date', ''),
                exp.get('duration_months', 0),
                exp.get('description', ''),
                json.dumps(exp.get('technologies', []))
            ))
        
        conn.commit()
        conn.close()
        return resume_id
    
    def search_resumes(self, query: str = None, filters: Dict = None, limit: int = 50) -> List[Dict]:
        """Search resumes with SQL queries"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        sql = "SELECT * FROM resumes WHERE status = 'processed'"
        params = []
        
        # Add filters
        if filters:
            conditions = []
            if filters.get('skills'):
                # For skills, we need to join with resume_skills
                skill_list = filters['skills']
                placeholders = ','.join(['?'] * len(skill_list))
                sql = f'''
                    SELECT DISTINCT r.* FROM resumes r
                    JOIN resume_skills rs ON r.id = rs.resume_id
                    WHERE rs.skill_name IN ({placeholders})
                '''
                params.extend(skill_list)
            
            if filters.get('min_experience'):
                sql += " AND total_experience_years >= ?"
                params.append(filters['min_experience'])
            
            if filters.get('education'):
                sql += " AND education_level LIKE ?"
                params.append(f'%{filters["education"]}%')
        
        # Add text search
        if query:
            sql += " AND extracted_text LIKE ?"
            params.append(f'%{query}%')
        
        sql += f" ORDER BY id DESC LIMIT {limit}"
        
        cursor.execute(sql, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        # Parse JSON fields
        for result in results:
            if result.get('skills_json'):
                result['skills'] = json.loads(result['skills_json'])
            if result.get('metadata_json'):
                result['metadata'] = json.loads(result['metadata_json'])
        
        conn.close()
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM resumes")
        stats['total_resumes'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT skill_name) FROM resume_skills")
        stats['unique_skills'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(total_experience_years) FROM resumes")
        stats['avg_experience'] = cursor.fetchone()[0] or 0
        
        # Skill distribution
        cursor.execute('''
            SELECT skill_category, COUNT(*) as count 
            FROM resume_skills 
            GROUP BY skill_category 
            ORDER BY count DESC
        ''')
        stats['skill_distribution'] = dict(cursor.fetchall())
        
        conn.close()
        return stats

# ==================== NLP PROCESSING ENGINE ====================
class NLPProcessor:
    """Advanced NLP processing for resumes"""
    
    def __init__(self):
        # Initialize spaCy
        try:
            self.nlp = spacy.load(config.SPACy_MODEL)
        except:
            # If model not found, download it
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", config.SPACy_MODEL])
            self.nlp = spacy.load(config.SPACy_MODEL)
        
        # Skill databases
        self.skill_db = self._load_skill_database()
        
        # Education patterns
        self.education_patterns = [
            r'\b(?:B\.?S\.?|B\.?A\.?|B\.?Tech|Bachelor).*?(?:Computer|Engineering|Science)',
            r'\b(?:M\.?S\.?|M\.?A\.?|M\.?Tech|Master).*?(?:Computer|Engineering|Science)',
            r'\b(?:Ph\.?D\.?|Doctorate).*?(?:Computer|Engineering)',
            r'\bMBA\b'
        ]
    
    def _load_skill_database(self) -> Dict:
        """Load comprehensive skill database"""
        return {
            "programming": ["python", "java", "javascript", "c++", "c#", "go", "rust", 
                          "typescript", "swift", "kotlin", "scala", "r", "matlab"],
            "web_dev": ["react", "angular", "vue", "node.js", "django", "flask", "spring",
                       "express", "laravel", "asp.net", "html5", "css3", "sass", "less"],
            "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", 
                     "ansible", "jenkins", "helm", "openshift", "serverless"],
            "databases": ["sql", "mysql", "postgresql", "mongodb", "redis", "oracle",
                         "cassandra", "dynamodb", "elasticsearch", "snowflake"],
            "data_science": ["pandas", "numpy", "pytorch", "tensorflow", "scikit-learn",
                           "spark", "hadoop", "tableau", "powerbi", "apache airflow"],
            "devops": ["git", "linux", "bash", "shell", "ci/cd", "github actions",
                      "gitlab", "jira", "confluence", "prometheus", "grafana"],
            "mobile": ["android", "ios", "flutter", "react native", "xamarin"],
            "soft_skills": ["leadership", "communication", "project management", "agile",
                           "scrum", "problem solving", "teamwork", "presentation"]
        }
    
    def extract_entities(self, text: str) -> Dict:
        """Extract named entities from resume text"""
        doc = self.nlp(text)
        
        entities = {
            "person": [],
            "organization": [],
            "date": [],
            "email": [],
            "phone": [],
            "url": [],
            "skills": set(),
            "education": [],
            "experience": []
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["person"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organization"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["date"].append(ent.text)
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["email"] = re.findall(email_pattern, text)
        
        # Extract phone numbers
        phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        entities["phone"] = re.findall(phone_pattern, text)
        
        # Extract skills
        text_lower = text.lower()
        for category, skills in self.skill_db.items():
            for skill in skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                    entities["skills"].add(skill)
        
        # Extract education
        for pattern in self.education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["education"].extend(matches)
        
        # Convert set to list
        entities["skills"] = list(entities["skills"])
        
        return entities
    
    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience with duration"""
        experiences = []
        
        # Look for experience patterns
        experience_patterns = [
            r'(?P<company>[A-Z][A-Za-z\s&]+)\s*(?:\||-|–)?\s*(?P<title>[^,\n]+?)\s*,\s*(?P<duration>\d{4}\s*-\s*(?:\d{4}|Present))',
            r'(?P<title>[^,\n]+?)\s*at\s*(?P<company>[A-Z][A-Za-z\s&]+)\s*\((?P<duration>\d{4}\s*-\s*(?:\d{4}|Present))\)'
        ]
        
        for pattern in experience_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                experience = {
                    "company": match.group("company").strip(),
                    "title": match.group("title").strip(),
                    "duration": match.group("duration").strip(),
                    "technologies": self._extract_technologies_from_context(text, match.start(), match.end())
                }
                experiences.append(experience)
        
        return experiences
    
    def _extract_technologies_from_context(self, text: str, start: int, end: int) -> List[str]:
        """Extract technologies mentioned near the experience"""
        context = text[max(0, start-500):min(len(text), end+500)]
        technologies = []
        
        for category, skills in self.skill_db.items():
            if category not in ["soft_skills"]:
                for skill in skills:
                    if re.search(r'\b' + re.escape(skill) + r'\b', context.lower()):
                        technologies.append(skill)
        
        return list(set(technologies))
    
    def calculate_experience_years(self, text: str) -> float:
        """Calculate total years of experience"""
        # Look for year patterns
        year_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)(?:\s*of?\s*experience)?',
            r'experience\s*:\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*(?:years?|yrs?)\s*(?:in|of)\s*(?:industry|field)'
        ]
        
        total_years = 0
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the highest number found
                years = [int(match) for match in matches if match.isdigit()]
                if years:
                    total_years = max(years, total_years)
        
        return total_years
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate TF-IDF vector for text"""
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # For single document, we need to fit_transform with at least 2 documents
        # So we add a dummy document
        documents = [text, "dummy document"]
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Return the first document's vector
        return tfidf_matrix[0].toarray()

# ==================== VECTOR STORE MANAGER ====================
class VectorStoreManager:
    """Manager for multiple vector store backends"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.active_stores = {}
    
    def create_chroma_store(self, documents, collection_name="resumes"):
        """Create Chroma vector store"""
        chroma_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=config.CHROMA_PATH,
            collection_name=collection_name
        )
        chroma_store.persist()
        self.active_stores["chroma"] = chroma_store
        return chroma_store
    
    def create_faiss_store(self, documents, index_path=config.FAISS_PATH):
        """Create FAISS vector store"""
        faiss_store = FAISS.from_documents(documents, self.embeddings)
        faiss_store.save_local(index_path)
        self.active_stores["faiss"] = faiss_store
        return faiss_store
    
    def load_store(self, store_type="chroma", collection_name="resumes"):
        """Load existing vector store"""
        if store_type == "chroma":
            store = Chroma(
                persist_directory=config.CHROMA_PATH,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            self.active_stores["chroma"] = store
            return store
        elif store_type == "faiss":
            store = FAISS.load_local(
                config.FAISS_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.active_stores["faiss"] = store
            return store
    
    def hybrid_search(self, query: str, store_type="chroma", k=20, 
                     filters: Dict = None) -> List[Tuple[Any, float]]:
        """Perform hybrid search across vector stores"""
        if store_type not in self.active_stores:
            self.load_store(store_type)
        
        store = self.active_stores[store_type]
        
        if hasattr(store, 'similarity_search_with_score'):
            results = store.similarity_search_with_score(query, k=k)
            
            # Apply filters if provided
            if filters:
                filtered_results = []
                for doc, score in results:
                    if self._matches_filters(doc, filters):
                        filtered_results.append((doc, score))
                return filtered_results
            
            return results
        
        return []
    
    def _matches_filters(self, document, filters: Dict) -> bool:
        """Check if document matches given filters"""
        metadata = document.metadata
        
        if filters.get('min_experience'):
            exp = metadata.get('experience_years', 0)
            if exp < filters['min_experience']:
                return False
        
        if filters.get('required_skills'):
            doc_skills = metadata.get('skills', [])
            if not all(skill in doc_skills for skill in filters['required_skills']):
                return False
        
        if filters.get('excluded_skills'):
            doc_skills = metadata.get('skills', [])
            if any(skill in doc_skills for skill in filters['excluded_skills']):
                return False
        
        return True

# ==================== RESUME PROCESSING PIPELINE ====================
class ResumePipeline:
    """End-to-end resume processing pipeline"""
    
    def __init__(self):
        self.db = ResumeDatabase()
        self.nlp = NLPProcessor()
        self.vector_manager = VectorStoreManager()
        
    def process_resume(self, file_path: str, filename: str) -> Dict:
        """Process a single resume through the pipeline"""
        # Extract text from PDF
        text = self._extract_text_from_pdf(file_path)
        
        # Extract entities and metadata
        entities = self.nlp.extract_entities(text)
        experiences = self.nlp.extract_experience(text)
        total_experience = self.nlp.calculate_experience_years(text)
        
        # Prepare resume data
        resume_data = {
            "filename": filename,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "extracted_text": text[:10000],  # Store first 10k chars
            "candidate_name": entities["person"][0] if entities["person"] else "",
            "candidate_email": entities["email"][0] if entities["email"] else "",
            "candidate_phone": entities["phone"][0] if entities["phone"] else "",
            "total_experience_years": total_experience,
            "current_company": entities["organization"][0] if entities["organization"] else "",
            "skills": {
                "technical": entities["skills"],
                "soft": list(set(entities["skills"]) & set(self.nlp.skill_db["soft_skills"]))
            },
            "education": entities["education"],
            "experiences": experiences,
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "text_length": len(text),
                "entity_counts": {k: len(v) for k, v in entities.items()}
            }
        }
        
        # Save to SQLite
        resume_id = self.db.save_resume(resume_data)
        resume_data["database_id"] = resume_id
        
        # Prepare document for vector store
        document_text = f"""
        Candidate: {resume_data['candidate_name']}
        Experience: {total_experience} years
        Skills: {', '.join(resume_data['skills']['technical'])}
        Education: {', '.join(resume_data['education'])}
        
        Summary:
        {text[:2000]}
        """
        
        from langchain_core.documents import Document
        doc = Document(
            page_content=document_text,
            metadata={
                "id": resume_id,
                "filename": filename,
                "candidate_name": resume_data['candidate_name'],
                "experience_years": total_experience,
                "skills": resume_data['skills']['technical'],
                "education": resume_data['education'][0] if resume_data['education'] else "",
                "source": "resume_pipeline"
            }
        )
        
        return resume_data, doc
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        try:
            # Method 1: pdfplumber (better for structured text)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            pass
        
        # If pdfplumber failed, try PyPDFLoader
        if len(text.strip()) < 100:
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                text = "\n".join([page.page_content for page in pages])
            except:
                pass
        
        return text.strip()
    
    def batch_process(self, file_list: List[Tuple[str, str]]) -> Dict:
        """Process multiple resumes in batch"""
        results = {
            "successful": 0,
            "failed": 0,
            "total": len(file_list),
            "documents": [],
            "resume_data": []
        }
        
        documents = []
        
        for file_path, filename in file_list:
            try:
                resume_data, doc = self.process_resume(file_path, filename)
                results["resume_data"].append(resume_data)
                documents.append(doc)
                results["successful"] += 1
            except Exception as e:
                st.error(f"Failed to process {filename}: {str(e)}")
                results["failed"] += 1
        
        # Create vector stores
        if documents:
            # Create Chroma store
            chroma_store = self.vector_manager.create_chroma_store(documents)
            
            # Create FAISS store
            faiss_store = self.vector_manager.create_faiss_store(documents)
            
            results["chroma_collection"] = chroma_store._collection.name
            results["faiss_index"] = "created"
        
        return results

# ==================== STREAMLIT UI ====================
class ResumeSearchApp:
    """Main Streamlit application"""
    
    def __init__(self):
        self.pipeline = ResumePipeline()
        self.setup_page()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=config.PAGE_TITLE,
            page_icon="🏢",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .main-header {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            margin: 0.5rem;
        }
        .resume-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            margin: 1rem 0;
            border-left: 5px solid #667eea;
        }
        .skill-tag {
            display: inline-block;
            background: #e0e7ff;
            color: #3730a3;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            margin: 0.2rem;
            font-size: 0.9rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.title("⚙️ Controls")
            
            # Database selector
            db_option = st.selectbox(
                "Database Backend",
                ["Chroma", "FAISS", "SQLite", "All"],
                help="Choose vector database for search"
            )
            
            # Search mode
            search_mode = st.radio(
                "Search Mode",
                ["🔍 Semantic", "🔤 Keyword", "⚡ Hybrid", "🎯 Advanced"],
                index=3
            )
            
            # Advanced filters
            with st.expander("🎯 Advanced Filters", expanded=True):
                min_exp = st.slider("Minimum Experience (years)", 0, 20, 0)
                
                # Skill selection
                all_skills = list(self.pipeline.nlp.skill_db.keys())
                selected_categories = st.multiselect(
                    "Skill Categories",
                    all_skills,
                    default=["programming", "cloud"]
                )
                
                # Get skills from selected categories
                selected_skills = []
                for category in selected_categories:
                    selected_skills.extend(self.pipeline.nlp.skill_db[category])
                
                required_skills = st.multiselect(
                    "Required Skills",
                    sorted(selected_skills)[:50],  # Limit for performance
                    default=["python", "linux"]
                )
                
                excluded_skills = st.multiselect(
                    "Excluded Skills",
                    sorted(selected_skills)[:50]
                )
                
                education_level = st.selectbox(
                    "Education Level",
                    ["Any", "Bachelor", "Master", "PhD", "MBA"]
                )
            
            # Search parameters
            with st.expander("⚙️ Search Parameters"):
                k_value = st.slider("Search Depth (k)", 10, 200, 50)
                score_threshold = st.slider("Score Threshold", 0.0, 2.0, 0.3, 0.05)
                enable_reranking = st.checkbox("Enable Re-ranking", True)
            
            # Batch operations
            with st.expander("📦 Batch Operations"):
                if st.button("🔄 Refresh All Indexes", use_container_width=True):
                    st.session_state.refresh_indexes = True
                
                if st.button("📊 Update Statistics", use_container_width=True):
                    st.session_state.update_stats = True
            
            # Export options
            with st.expander("💾 Export Data"):
                export_format = st.selectbox("Format", ["CSV", "JSON", "Excel"])
                if st.button(f"Export Results as {export_format}", use_container_width=True):
                    st.session_state.export_data = export_format
            
            # System info
            st.divider()
            st.caption(f"📊 Database: {self.pipeline.db.db_path}")
            st.caption(f"🧠 Embeddings: {config.EMBEDDING_MODEL}")
            st.caption("🏢 Enterprise Resume Search v2.0")
    
    def render_dashboard(self):
        """Render main dashboard"""
        # Header
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown('<div class="main-header">', unsafe_allow_html=True)
            st.title("🏢 Enterprise Resume Intelligence")
            st.markdown("""
            **Advanced AI-powered resume search with multi-database backend, 
            entity extraction, and intelligent ranking.**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.metric("Total Resumes", self.pipeline.db.get_stats()['total_resumes'])
        
        with col3:
            with st.container():
                st.metric("Unique Skills", self.pipeline.db.get_stats()['unique_skills'])
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📤 Upload & Process", 
            "🔍 Search", 
            "📊 Analytics", 
            "👥 Candidates", 
            "⚙️ Admin"
        ])
        
        # Tab 1: Upload & Process
        with tab1:
            self.render_upload_section()
        
        # Tab 2: Search
        with tab2:
            self.render_search_section()
        
        # Tab 3: Analytics
        with tab3:
            self.render_analytics_section()
        
        # Tab 4: Candidates
        with tab4:
            self.render_candidates_section()
        
        # Tab 5: Admin
        with tab5:
            self.render_admin_section()
    
    def render_upload_section(self):
        """Render resume upload and processing section"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📤 Upload Resumes")
            
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload multiple resumes for processing"
            )
            
            if uploaded_files:
                with st.expander("📋 Uploaded Files", expanded=True):
                    for file in uploaded_files:
                        st.write(f"📄 {file.name} ({file.size:,} bytes)")
                
                # Processing options
                col_a, col_b = st.columns(2)
                with col_a:
                    process_now = st.button("🚀 Process Now", type="primary", use_container_width=True)
                
                with col_b:
                    schedule = st.button("⏰ Schedule Batch", use_container_width=True)
                
                if process_now and uploaded_files:
                    with st.spinner("🔬 Processing resumes..."):
                        # Save files temporarily
                        temp_files = []
                        for uploaded_file in uploaded_files:
                            temp_dir = tempfile.mkdtemp()
                            temp_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            temp_files.append((temp_path, uploaded_file.name))
                        
                        # Process batch
                        results = self.pipeline.batch_process(temp_files)
                        
                        # Show results
                        st.success(f"✅ Processed {results['successful']} of {results['total']} resumes")
                        
                        if results['successful'] > 0:
                            st.info(f"""
                            **Processing Summary:**
                            - Created Chroma collection: `{results.get('chroma_collection', 'N/A')}`
                            - Created FAISS index
                            - Added to SQLite database
                            """)
            
            # Batch processing history
            with st.expander("📜 Processing History", expanded=False):
                if st.button("View History"):
                    history = self.pipeline.db.search_resumes(limit=10)
                    for resume in history:
                        st.write(f"**{resume['filename']}** - {resume['upload_date'][:10]}")
        
        with col2:
            st.subheader("⚙️ Processing Settings")
            
            # Chunking settings
            chunk_size = st.slider("Chunk Size", 200, 2000, config.CHUNK_SIZE, 50)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, config.CHUNK_OVERLAP, 50)
            
            # Extraction settings
            extract_entities = st.checkbox("Extract Named Entities", True)
            extract_experience = st.checkbox("Extract Work Experience", True)
            extract_skills = st.checkbox("Extract Skills", True)
            
            # Vector store options
            st.subheader("🧠 Vector Stores")
            create_chroma = st.checkbox("Create Chroma Store", True)
            create_faiss = st.checkbox("Create FAISS Store", True)
            store_embeddings = st.checkbox("Store Embeddings", True)
            
            # Advanced options
            with
