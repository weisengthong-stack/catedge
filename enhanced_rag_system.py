import os
import re
import uuid
import json
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Compatible with your existing dependencies - handle LangChain compatibility
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available, using fallback implementation")

# PDF processing imports (using your existing dependencies)
import pypdf
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Simple document class for fallback when LangChain not available
class SimpleDocument:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Metadata extraction patterns and classifiers
class PDFMetadataExtractor:
    """Extract and classify PDF content into specific metadata categories"""
    
    def __init__(self):
        self.metadata_patterns = {
            'rating_action_basis': [
                r'rating\s+action\s+basis',
                r'basis\s+for\s+rating\s+action',
                r'rationale\s+for\s+rating',
                r'rating\s+rationale',
                r'action\s+rationale'
            ],
            'rating_drivers_positive': [
                r'positive\s+rating\s+drivers?',
                r'rating\s+drivers?\s+positive',
                r'strengths?',
                r'positive\s+factors?',
                r'credit\s+strengths?',
                r'favorable\s+factors?'
            ],
            'rating_drivers_negative': [
                r'negative\s+rating\s+drivers?',
                r'rating\s+drivers?\s+negative',
                r'weaknesses?',
                r'negative\s+factors?',
                r'credit\s+weaknesses?',
                r'concerns?',
                r'challenges?'
            ],
            'rating_triggers_upgrade': [
                r'upgrade\s+triggers?',
                r'triggers?\s+for\s+upgrade',
                r'positive\s+triggers?',
                r'upward\s+rating\s+pressure',
                r'factors?\s+that\s+could\s+lead\s+to\s+upgrade'
            ],
            'rating_triggers_downgrade': [
                r'downgrade\s+triggers?',
                r'triggers?\s+for\s+downgrade',
                r'negative\s+triggers?',
                r'downward\s+rating\s+pressure',
                r'factors?\s+that\s+could\s+lead\s+to\s+downgrade'
            ],
            'esg_descriptor': [
                r'esg\s+considerations?',
                r'environmental\s+social\s+governance',
                r'sustainability\s+factors?',
                r'esg\s+factors?',
                r'environmental\s+factors?',
                r'social\s+factors?',
                r'governance\s+factors?'
            ],
            'related_criteria': [
                r'related\s+criteria',
                r'applicable\s+criteria',
                r'rating\s+criteria',
                r'methodology',
                r'related\s+research',
                r'reference\s+criteria'
            ],
            'specific_ratings': [
                r'rating\s+history',
                r'current\s+ratings?',
                r'rating\s+details?',
                r'specific\s+ratings?',
                r'rating\s+scale',
                r'rating\s+definitions?'
            ],
            'introductory_pages': [
                r'executive\s+summary',
                r'introduction',
                r'overview',
                r'key\s+highlights?',
                r'summary',
                r'table\s+of\s+contents?'
            ],
            'financial_statement_pages': [
                r'financial\s+statements?',
                r'balance\s+sheet',
                r'income\s+statement',
                r'cash\s+flow',
                r'financial\s+data',
                r'financial\s+metrics?',
                r'financial\s+ratios?',
                r'consolidated\s+statements?'
            ]
        }
        
        # Initialize sentence transformer for semantic similarity
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create semantic embeddings for each category
        self.category_embeddings = self._create_category_embeddings()
    
    def _create_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Create semantic embeddings for each metadata category"""
        category_descriptions = {
            'rating_action_basis': "The fundamental reasoning and justification behind a credit rating decision or action",
            'rating_drivers_positive': "Positive factors, strengths, and favorable aspects that support the credit rating",
            'rating_drivers_negative': "Negative factors, weaknesses, concerns, and challenges that constrain the rating",
            'rating_triggers_upgrade': "Specific conditions or factors that could lead to a rating upgrade",
            'rating_triggers_downgrade': "Specific conditions or factors that could lead to a rating downgrade",
            'esg_descriptor': "Environmental, social, and governance factors affecting the credit rating",
            'related_criteria': "Applicable rating methodologies, criteria, and related research documents",
            'specific_ratings': "Current ratings, rating history, and specific rating details",
            'introductory_pages': "Executive summary, introduction, overview, and key highlights of the document",
            'financial_statement_pages': "Financial statements, balance sheets, income statements, and financial data"
        }
        
        embeddings = {}
        for category, description in category_descriptions.items():
            embeddings[category] = self.embedder.encode(description)
        
        return embeddings
    
    def extract_page_metadata(self, page_text: str, page_number: int) -> Dict[str, Any]:
        """Extract metadata for a single page"""
        metadata = {
            'page_number': page_number,
            'categories': [],
            'confidence_scores': {},
            'matched_patterns': {},
            'semantic_scores': {}
        }
        
        page_text_lower = page_text.lower()
        
        # Pattern-based classification
        for category, patterns in self.metadata_patterns.items():
            matched_patterns = []
            for pattern in patterns:
                if re.search(pattern, page_text_lower):
                    matched_patterns.append(pattern)
            
            if matched_patterns:
                metadata['categories'].append(category)
                metadata['matched_patterns'][category] = matched_patterns
                metadata['confidence_scores'][category] = len(matched_patterns) / len(patterns)
        
        # Semantic similarity classification
        if page_text.strip():
            page_embedding = self.embedder.encode(page_text)
            
            for category, category_embedding in self.category_embeddings.items():
                similarity = np.dot(page_embedding, category_embedding) / (
                    np.linalg.norm(page_embedding) * np.linalg.norm(category_embedding)
                )
                metadata['semantic_scores'][category] = float(similarity)
                
                # Add category if semantic similarity is high enough
                if similarity > 0.6 and category not in metadata['categories']:
                    metadata['categories'].append(category)
                    if category not in metadata['confidence_scores']:
                        metadata['confidence_scores'][category] = similarity
        
        return metadata
    
    def classify_document_structure(self, pages_text: List[str]) -> Dict[str, Any]:
        """Classify the entire document structure and identify section boundaries"""
        document_metadata = {
            'total_pages': len(pages_text),
            'page_classifications': [],
            'section_boundaries': {},
            'category_page_ranges': {}
        }
        
        # Process each page
        for i, page_text in enumerate(pages_text):
            page_metadata = self.extract_page_metadata(page_text, i + 1)
            document_metadata['page_classifications'].append(page_metadata)
        
        # Identify section boundaries and page ranges for each category
        for category in self.metadata_patterns.keys():
            pages_with_category = []
            for page_meta in document_metadata['page_classifications']:
                if category in page_meta['categories']:
                    pages_with_category.append(page_meta['page_number'])
            
            if pages_with_category:
                document_metadata['category_page_ranges'][category] = {
                    'pages': pages_with_category,
                    'start_page': min(pages_with_category),
                    'end_page': max(pages_with_category),
                    'total_pages': len(pages_with_category)
                }
        
        return document_metadata

# Enhanced RAG System compatible with your existing dependencies
class EnhancedRAGSystem:
    """Enhanced RAG system with metadata extraction - compatible with existing dependencies"""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        # Initialize embeddings - try LangChain first, fallback to sentence-transformers
        if LANGCHAIN_AVAILABLE:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs={'device': 'cpu'}
                )
                self.use_langchain = True
            except Exception as e:
                print(f"LangChain embeddings failed, using sentence-transformers: {e}")
                self.embeddings = SentenceTransformer(embedding_model_name)
                self.use_langchain = False
        else:
            self.embeddings = SentenceTransformer(embedding_model_name)
            self.use_langchain = False
        
        # Initialize text splitter
        if LANGCHAIN_AVAILABLE:
            try:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
            except Exception:
                self.text_splitter = None
        else:
            self.text_splitter = None
        
        # Initialize metadata extractor
        self.metadata_extractor = PDFMetadataExtractor()
        
        # Storage
        self.documents = {}  # Store document metadata
        self.chunks = []     # Store text chunks
        self.chunk_metadata = []  # Store metadata for each chunk
        self.index = None    # FAISS index
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        self.vectorstore = None  # LangChain vectorstore if available
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks - fallback if LangChain not available"""
        if self.text_splitter and LANGCHAIN_AVAILABLE:
            try:
                return self.text_splitter.split_text(text)
            except Exception:
                pass
        
        # Fallback implementation
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > chunk_size * 0.7:
                    chunk = chunk[:boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
        
    def process_pdf_with_metadata(self, file_path: str, filename: str) -> Tuple[str, Dict]:
        """Process PDF file and extract both content and metadata"""
        
        # Extract text using multiple methods for better coverage
        pages_text = []
        
        try:
            # Method 1: PyPDF for basic text extraction
            pdf_reader = pypdf.PdfReader(file_path)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ''
                pages_text.append(page_text)
            
            # Method 2: pdfplumber for better table extraction (if available)
            if PDFPLUMBER_AVAILABLE:
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for i, page in enumerate(pdf.pages):
                            if i < len(pages_text):
                                plumber_text = page.extract_text() or ''
                                if len(plumber_text) > len(pages_text[i]):
                                    pages_text[i] = plumber_text
                            else:
                                pages_text.append(page.extract_text() or '')
                except Exception as e:
                    print(f"PDFPlumber extraction failed: {e}")
            
        except Exception as e:
            print(f"PDF extraction failed: {e}")
            return None, None
        
        # Extract metadata for the entire document
        document_metadata = self.metadata_extractor.classify_document_structure(pages_text)
        
        # Combine all pages into full text
        full_text = "\n\n".join([f"=== PAGE {i+1} ===\n{text}" for i, text in enumerate(pages_text)])
        
        return full_text, document_metadata
    
    def add_document(self, doc_id: str, filename: str, file_path: str = None, content: str = None):
        """Add a document to the RAG system with metadata extraction"""
        
        document_metadata = None
        
        if file_path and filename.lower().endswith('.pdf'):
            # Process PDF with metadata
            result = self.process_pdf_with_metadata(file_path, filename)
            if result[0] is None:
                return 0
            
            content, document_metadata = result
        else:
            # Handle text files
            document_metadata = {
                'total_pages': 1,
                'page_classifications': [],
                'section_boundaries': {},
                'category_page_ranges': {}
            }
        
        # Store document metadata
        self.documents[doc_id] = {
            'filename': filename,
            'upload_time': datetime.now().isoformat(),
            'content_length': len(content),
            'metadata': document_metadata
        }
        
        # Create chunks with metadata
        chunks = self.chunk_text(content)
        
        # If using LangChain, try to create vectorstore
        if LANGCHAIN_AVAILABLE and self.use_langchain:
            try:
                # Create LangChain documents
                langchain_docs = []
                for i, chunk in enumerate(chunks):
                    chunk_categories = self._classify_chunk_categories(chunk, document_metadata)
                    
                    doc_class = Document if LANGCHAIN_AVAILABLE else SimpleDocument
                    doc = doc_class(
                        page_content=chunk,
                        metadata={
                            'doc_id': doc_id,
                            'filename': filename,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'categories': chunk_categories
                        }
                    )
                    langchain_docs.append(doc)
                
                # Create or update vectorstore
                if self.vectorstore is None:
                    self.vectorstore = FAISS.from_documents(langchain_docs, self.embeddings)
                else:
                    self.vectorstore.add_documents(langchain_docs)
                    
            except Exception as e:
                print(f"LangChain vectorstore creation failed: {e}")
                self.use_langchain = False
        
        # Fallback to manual chunk storage
        for i, chunk in enumerate(chunks):
            chunk_categories = self._classify_chunk_categories(chunk, document_metadata)
            
            self.chunks.append(chunk)
            self.chunk_metadata.append({
                'doc_id': doc_id,
                'filename': filename,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'categories': chunk_categories
            })
        
        # Update FAISS index
        self._update_index()
        
        return len(chunks)
    
    def _classify_chunk_categories(self, chunk: str, document_metadata: Dict) -> List[str]:
        """Classify which metadata categories a chunk belongs to"""
        if not document_metadata:
            return []
        
        chunk_metadata = self.metadata_extractor.extract_page_metadata(chunk, 0)
        return chunk_metadata['categories']
    
    def _update_index(self):
        """Update the FAISS index with all chunks"""
        if not self.chunks:
            return
        
        # Use sentence transformers directly for compatibility
        if hasattr(self.embeddings, 'encode'):
            embeddings = self.embeddings.encode(self.chunks)
        else:
            embeddings = self.embeddings.embed_documents(self.chunks)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index.reset()
        
        self.index.add(np.array(embeddings).astype('float32'))
    
    def search_by_category(self, query: str, categories: List[str] = None, k: int = 10) -> List[Dict]:
        """Search documents with optional category filtering"""
        if not self.chunks or (self.index is None and self.vectorstore is None):
            return []
        
        # Try LangChain vectorstore first
        if self.use_langchain and self.vectorstore is not None:
            try:
                # Use LangChain search
                docs = self.vectorstore.similarity_search(query, k=k*2)
                
                results = []
                for doc in docs:
                    doc_metadata = doc.metadata
                    
                    # Filter by categories if specified
                    if categories:
                        doc_categories = doc_metadata.get('categories', [])
                        if not any(cat in doc_categories for cat in categories):
                            continue
                    
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc_metadata,
                        'distance': 0  # LangChain doesn't return distance by default
                    })
                
                return results[:k]
                
            except Exception as e:
                print(f"LangChain search failed, using fallback: {e}")
        
        # Fallback to manual FAISS search
        if self.index is None:
            return []
            
        # Get query embedding
        if hasattr(self.embeddings, 'encode'):
            query_embedding = self.embeddings.encode([query])
        else:
            query_embedding = [self.embeddings.embed_query(query)]
            
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            min(k * 2, len(self.chunks))
        )
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk_meta = self.chunk_metadata[idx]
                
                # Filter by categories if specified
                if categories:
                    chunk_categories = chunk_meta.get('categories', [])
                    if not any(cat in chunk_categories for cat in categories):
                        continue
                
                candidates.append({
                    'content': self.chunks[idx],
                    'metadata': chunk_meta,
                    'distance': float(distances[0][i])
                })
        
        # Sort by distance and return top k
        candidates.sort(key=lambda x: x['distance'])
        return candidates[:k]
    
    def get_document_summary(self, doc_id: str) -> Dict:
        """Get a summary of document metadata and categories"""
        if doc_id not in self.documents:
            return None
        
        doc_info = self.documents[doc_id]
        metadata = doc_info['metadata']
        
        summary = {
            'filename': doc_info['filename'],
            'upload_time': doc_info['upload_time'],
            'total_pages': metadata.get('total_pages', 0),
            'categories_found': list(metadata.get('category_page_ranges', {}).keys()),
            'category_details': {}
        }
        
        for category, details in metadata.get('category_page_ranges', {}).items():
            summary['category_details'][category] = {
                'pages': details['pages'],
                'total_pages': details['total_pages'],
                'page_range': f"{details['start_page']}-{details['end_page']}"
            }
        
        return summary
    
    def save_to_disk(self, path: str):
        """Save the enhanced RAG system to disk"""
        # Save document metadata and chunks
        data = {
            'documents': self.documents,
            'chunks': self.chunks,
            'chunk_metadata': self.chunk_metadata
        }
        with open(f"{path}_enhanced_data.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, f"{path}_enhanced_index.faiss")
            
        # Save LangChain vectorstore if available
        if self.vectorstore is not None:
            try:
                self.vectorstore.save_local(f"{path}_vectorstore")
            except Exception as e:
                print(f"Could not save vectorstore: {e}")
    
    def load_from_disk(self, path: str):
        """Load the enhanced RAG system from disk"""
        try:
            # Load document metadata and chunks
            with open(f"{path}_enhanced_data.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.chunks = data['chunks']
            self.chunk_metadata = data['chunk_metadata']
            
            # Load FAISS index
            if os.path.exists(f"{path}_enhanced_index.faiss"):
                self.index = faiss.read_index(f"{path}_enhanced_index.faiss")
            
            # Try to load LangChain vectorstore
            if LANGCHAIN_AVAILABLE and os.path.exists(f"{path}_vectorstore"):
                try:
                    self.vectorstore = FAISS.load_local(f"{path}_vectorstore", self.embeddings)
                    self.use_langchain = True
                except Exception as e:
                    print(f"Could not load vectorstore: {e}")
                    self.use_langchain = False
                
        except Exception as e:
            print(f"Error loading enhanced RAG system: {e}")

# Helper functions for Flask integration
def process_pdf_file_enhanced(file, enhanced_rag_system):
    """Process PDF file with enhanced metadata extraction"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            file.seek(0)
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Process with enhanced system
            result = enhanced_rag_system.process_pdf_with_metadata(tmp_file_path, file.filename)
            if result[0] is None:
                return None
            
            content, metadata = result
            return content
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        print(f"Enhanced PDF processing error: {e}")
        return None