from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import requests
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Any
import logging
from datetime import datetime
import hashlib
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class SimpleEmbedding:
    """Simple embedding using TF-IDF as a lightweight alternative"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=512,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and transform texts"""
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return self.tfidf_matrix.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using fitted vectorizer"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted yet")
        return self.vectorizer.transform(texts).toarray()

class DocumentProcessor:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.embedding_model = SimpleEmbedding()
        self.file_metadata = {}  # Store file metadata
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                current_chunk = ' '.join(overlap_words) + ' ' + sentence
            else:
                current_chunk += ' ' + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def process_single_file(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process a single PDF file and return metadata"""
        logger.info(f"Processing {filename}")
        
        # Get file stats
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        upload_time = datetime.now().isoformat()
        
        # Extract text and create chunks
        text = self.extract_text_from_pdf(file_path)
        chunks = self.chunk_text(text)
        
        # Generate file ID
        file_id = hashlib.md5(f"{filename}_{upload_time}".encode()).hexdigest()[:8]
        
        # Store file metadata
        file_metadata = {
            'id': file_id,
            'filename': filename,
            'size': file_size,
            'chunk_count': len(chunks),
            'upload_time': upload_time,
            'country': self.extract_country_from_filename(filename),
            'path': file_path
        }
        
        self.file_metadata[file_id] = file_metadata
        
        # Add chunks to documents with enhanced metadata
        chunk_start_index = len(self.documents)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_chunk_{i}"
            self.documents.append({
                'id': chunk_id,
                'text': chunk,
                'source': filename,
                'file_id': file_id,
                'chunk_index': i,
                'country': file_metadata['country'],
                'upload_time': upload_time
            })
        
        logger.info(f"Processed {filename}: {len(chunks)} chunks created")
        return file_metadata
    
    def process_documents(self, pdf_files: List[str]) -> List[Dict[str, Any]]:
        """Process multiple PDF documents"""
        processed_files = []
        
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            file_metadata = self.process_single_file(pdf_file, filename)
            processed_files.append(file_metadata)
        
        # Generate embeddings using TF-IDF
        if self.documents:
            logger.info("Generating embeddings...")
            texts = [doc['text'] for doc in self.documents]
            self.embeddings = self.embedding_model.fit_transform(texts)
            logger.info(f"Generated embeddings for {len(self.documents)} document chunks")
        
        return processed_files
    
    def extract_country_from_filename(self, filename: str) -> str:
        """Extract country from filename"""
        filename = os.path.basename(filename).upper()
        country_map = {
            'SGP': 'Singapore',
            'MYS': 'Malaysia', 
            'IDN': 'Indonesia',
            'THA': 'Thailand'
        }
        
        for code, country in country_map.items():
            if code in filename:
                return country
        return 'Unknown'
    
    def search_similar(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """Search for similar documents using TF-IDF similarity"""
        if not self.documents or self.embeddings is None:
            return []
        
        # Transform query
        query_embedding = self.embedding_model.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                doc = self.documents[idx]
                results.append({
                    'chunk_id': doc['id'],
                    'text': doc['text'],
                    'source': doc['source'],
                    'file_id': doc['file_id'],
                    'chunk_index': doc['chunk_index'],
                    'country': doc['country'],
                    'score': float(similarities[idx]),
                    'rank': i + 1,
                    'upload_time': doc['upload_time']
                })
        
        return results
    
    def get_file_list(self) -> List[Dict[str, Any]]:
        """Get list of all uploaded files with metadata"""
        return list(self.file_metadata.values())
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file and its chunks"""
        if file_id not in self.file_metadata:
            return False
        
        # Remove file metadata
        file_metadata = self.file_metadata.pop(file_id)
        
        # Remove chunks from documents
        self.documents = [doc for doc in self.documents if doc['file_id'] != file_id]
        
        # Remove file from disk
        try:
            if os.path.exists(file_metadata['path']):
                os.remove(file_metadata['path'])
        except Exception as e:
            logger.error(f"Error deleting file {file_metadata['path']}: {e}")
        
        # Regenerate embeddings if documents remain
        if self.documents:
            texts = [doc['text'] for doc in self.documents]
            self.embeddings = self.embedding_model.fit_transform(texts)
        else:
            self.embeddings = None
            self.embedding_model = SimpleEmbedding()
        
        return True

class EnhancedRAGSystem:
    def __init__(self):
        self.processor = DocumentProcessor()
        self.is_initialized = False
        self.ollama_url = 'http://localhost:11434/api/generate'
    
    def initialize_with_existing_files(self) -> Dict[str, Any]:
        """Initialize with existing PDF files in the directory"""
        pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
        
        if not pdf_files:
            return {'error': 'No PDF files found'}
        
        processed_files = self.processor.process_documents(pdf_files)
        self.is_initialized = True
        
        return {
            'status': 'success',
            'message': f'System initialized with {len(pdf_files)} PDF files',
            'files': processed_files,
            'document_count': len(self.processor.documents)
        }
    
    def upload_file(self, file) -> Dict[str, Any]:
        """Upload and process a new PDF file"""
        if not file or file.filename == '':
            return {'error': 'No file selected'}
        
        if not self.allowed_file(file.filename):
            return {'error': 'Invalid file type. Only PDF files are allowed.'}
        
        # Secure filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return {'error': f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB'}
        
        # Save file
        file.save(file_path)
        
        # Process file
        try:
            file_metadata = self.processor.process_single_file(file_path, filename)
            
            # Regenerate embeddings
            if self.processor.documents:
                texts = [doc['text'] for doc in self.processor.documents]
                self.processor.embeddings = self.processor.embedding_model.fit_transform(texts)
            
            self.is_initialized = True
            
            return {
                'status': 'success',
                'message': f'File {filename} uploaded and processed successfully',
                'file': file_metadata
            }
        except Exception as e:
            # Clean up file if processing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            return {'error': f'Error processing file: {str(e)}'}
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def generate_answer_with_citations(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using local Ollama model with detailed citations"""
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context with detailed source information
        context_parts = []
        for i, doc in enumerate(context_docs):
            citation = f"[Source {i+1}: {doc['source']}, Chunk {doc['chunk_index']+1}]"
            context_parts.append(f"{citation}: {doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create focused prompt for business data analysis with citation requirements
        prompt = f"""You are an expert business analyst. Based on the following business regulatory documents, answer the question accurately and comprehensively.

CONTEXT FROM BUSINESS DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a clear, accurate answer based ONLY on the information in the context
- For each piece of information you use, cite the specific source using the format [Source X]
- For comparison questions, analyze data from all mentioned countries
- Include specific numbers, percentages, rankings, or procedures when available
- If comparing countries, clearly state which country performs better and why
- Cite the country name and source when referencing specific data
- If information is insufficient, state what's missing
- Be concise but comprehensive
- Always include citations in your response

ANSWER:"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    'model': 'gemma3:2b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'top_p': 0.9,
                        'num_predict': 1000
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self.generate_fallback_answer_with_citations(question, context_docs)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return self.generate_fallback_answer_with_citations(question, context_docs)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return self.generate_fallback_answer_with_citations(question, context_docs)
    
    def generate_fallback_answer_with_citations(self, question: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate a fallback answer with citations when Ollama is not available"""
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Create a structured response based on retrieved documents with citations
        response_parts = []
        response_parts.append("Based on the retrieved documents, here's what I found:\n")
        
        # Group by country for better organization
        country_info = {}
        for doc in context_docs[:5]:  # Limit to top 5 results
            country = doc['country']
            if country not in country_info:
                country_info[country] = []
            country_info[country].append({
                'text': doc['text'][:300] + "...",
                'source': doc['source'],
                'chunk': doc['chunk_index'] + 1,
                'score': doc['score']
            })
        
        for country, docs in country_info.items():
            response_parts.append(f"\n**{country}:**")
            for i, doc in enumerate(docs[:2]):  # Limit to 2 excerpts per country
                citation = f"[{doc['source']}, Chunk {doc['chunk']}]"
                response_parts.append(f"- {doc['text']} {citation}")
        
        response_parts.append(f"\n\nNote: This is a document retrieval result with {len(context_docs)} sources found. For AI-generated analysis with detailed citations, please ensure your local Ollama service is running and accessible.")
        
        return "\n".join(response_parts)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query and return answer with detailed sources"""
        if not self.is_initialized:
            return {
                'answer': 'System not initialized. Please upload some PDF files first.',
                'sources': [],
                'error': 'System not initialized'
            }
        
        # Retrieve relevant documents
        relevant_docs = self.processor.search_similar(question, k=8)
        
        # Generate answer with citations
        answer = self.generate_answer_with_citations(question, relevant_docs)
        
        return {
            'answer': answer,
            'sources': relevant_docs,
            'question': question,
            'num_sources': len(relevant_docs)
        }

# Initialize RAG system
rag_system = EnhancedRAGSystem()

@app.route('/')
def index():
    return "Enhanced RAG System Backend"

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload a new PDF file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        result = rag_system.upload_file(file)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/files', methods=['GET'])
def get_files():
    """Get list of uploaded files"""
    try:
        files = rag_system.processor.get_file_list()
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Error getting file list: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/files/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Delete a file and its chunks"""
    try:
        success = rag_system.processor.delete_file(file_id)
        if success:
            return jsonify({'message': 'File deleted successfully'})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/initialize', methods=['POST'])
def initialize_system():
    """Initialize the RAG system with existing PDF files"""
    try:
        result = rag_system.initialize_with_existing_files()
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_system():
    """Process a query through the RAG system"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        result = rag_system.query(question)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'initialized': rag_system.is_initialized,
        'document_count': len(rag_system.processor.documents),
        'file_count': len(rag_system.processor.file_metadata),
        'model': 'Local Ollama (with fallback)',
        'embedding': 'TF-IDF'
    })

@app.route('/test_questions', methods=['GET'])
def get_test_questions():
    """Get predefined test questions"""
    questions = [
        {
            'level': 'Easy',
            'question': 'How many procedures are required to start a business in Singapore?',
            'description': 'Tests basic factual retrieval from a single document.'
        },
        {
            'level': 'Medium', 
            'question': 'Summarize the key steps to get a construction permit in Malaysia.',
            'description': 'Requires multi-step summarization from a procedural list.'
        },
        {
            'level': 'Hard',
            'question': 'Which country has the lowest cost to register property as a percentage of property value?',
            'description': 'Tests numerical comparison across all four countries.'
        },
        {
            'level': 'Very Hard',
            'question': 'Compare the building quality control index across Malaysia, Indonesia, Thailand, and Singapore and explain which country has the most robust system.',
            'description': 'Requires comparison + reasoned judgment based on structured scoring criteria.'
        }
    ]
    return jsonify({'questions': questions})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003, debug=False)

