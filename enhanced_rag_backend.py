from flask import Flask, request, render_template, jsonify, session
import requests
import os
import pypdf
import re
import tempfile
import hashlib
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
from typing import List, Dict, Tuple
import uuid
import json

# New imports for table detection
import tabula
import pdfplumber

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure maximum file size (32MB to handle larger batches)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# Fireworks.ai API configuration
FIREWORKS_API_KEY = "fw_3Zb8uQDPQeUtX9kzmqnfrL6P"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p1-70b-instruct"  # You can change this to your preferred model

# Add CORS support
try:
    from flask_cors import CORS
    CORS(app)
    print("DEBUG: CORS enabled")
except ImportError:
    print("DEBUG: flask-cors not installed, adding manual CORS headers")
    
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

# Initialize models
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loading reranker model...")
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
print("Models loaded successfully!")

# Enhanced RAG System Class with Reranker
class RAGSystem:
    def __init__(self, embedding_model, reranker_model=None):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.documents = {}  # Store document metadata
        self.chunks = []     # Store text chunks
        self.chunk_metadata = []  # Store metadata for each chunk
        self.index = None    # FAISS index
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks - optimized for JSON extraction"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary for better context preservation
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_colon = chunk.rfind(':')
                boundary = max(last_period, last_newline, last_colon)
                
                if boundary > chunk_size * 0.7:  # More flexible boundary
                    chunk = chunk[:boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def add_document(self, doc_id: str, filename: str, content: str):
        """Add a document to the RAG system"""
        # Store document metadata
        self.documents[doc_id] = {
            'filename': filename,
            'upload_time': datetime.now().isoformat(),
            'content_length': len(content)
        }
        
        # Chunk the document with larger chunks for better JSON extraction
        chunks = self.chunk_text(content)
        
        # Store chunks with metadata
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_metadata.append({
                'doc_id': doc_id,
                'filename': filename,
                'chunk_index': i,
                'total_chunks': len(chunks)
            })
        
        # Update the index
        self._update_index()
        
        return len(chunks)
    
    def _update_index(self):
        """Update the FAISS index with all chunks"""
        if not self.chunks:
            return
        
        embeddings = self.embedding_model.encode(self.chunks)
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index.reset()
        
        self.index.add(np.array(embeddings).astype('float32'))
    
    def search_with_reranker(self, query: str, k: int = 20, final_k: int = 10) -> List[Dict]:
        """Enhanced search with reranker for better JSON extraction"""
        if not self.chunks or self.index is None:
            return []
        
        # Step 1: Get initial candidates using semantic search
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            min(k, len(self.chunks))
        )
        
        candidates = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                candidates.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'distance': float(distances[0][i]),
                    'idx': idx
                })
        
        # Step 2: Use reranker to improve relevance
        if self.reranker_model and len(candidates) > 1:
            query_chunk_pairs = [(query, candidate['chunk']) for candidate in candidates]
            rerank_scores = self.reranker_model.predict(query_chunk_pairs)
            
            # Update candidates with rerank scores
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(rerank_scores[i])
            
            # Sort by rerank score (higher is better)
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        else:
            # Fallback to distance-based sorting
            candidates.sort(key=lambda x: x['distance'])
        
        # Step 3: Apply diversity and return final results
        diversified_results = self._diversify_results(candidates, final_k)
        
        return diversified_results
    
    def _diversify_results(self, candidates: List[Dict], final_k: int) -> List[Dict]:
        """Ensure diversity while maintaining relevance for JSON extraction"""
        if not candidates:
            return []
        
        # Group candidates by document
        doc_groups = {}
        for candidate in candidates:
            doc_id = candidate['metadata']['doc_id']
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(candidate)
        
        # Select results ensuring diversity across documents
        selected = []
        doc_ids = list(doc_groups.keys())
        max_per_doc = max(2, final_k // len(doc_ids)) if doc_ids else final_k
        
        # First pass: get the best chunks from each document
        for doc_id in doc_ids:
            chunks_from_doc = doc_groups[doc_id][:max_per_doc]
            selected.extend(chunks_from_doc)
            if len(selected) >= final_k:
                break
        
        # Second pass: fill remaining slots with best remaining chunks
        if len(selected) < final_k:
            remaining_candidates = []
            for doc_id in doc_ids:
                remaining_candidates.extend(doc_groups[doc_id][max_per_doc:])
            
            # Sort by rerank_score if available, otherwise by distance
            if remaining_candidates and 'rerank_score' in remaining_candidates[0]:
                remaining_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            else:
                remaining_candidates.sort(key=lambda x: x['distance'])
            
            selected.extend(remaining_candidates[:final_k - len(selected)])
        
        # Final sort by relevance
        if selected and 'rerank_score' in selected[0]:
            selected.sort(key=lambda x: x['rerank_score'], reverse=True)
        else:
            selected.sort(key=lambda x: x['distance'])
            
        return selected[:final_k]
    
    def remove_document(self, doc_id: str):
        """Remove a document from the RAG system"""
        if doc_id not in self.documents:
            return False
        
        del self.documents[doc_id]
        
        new_chunks = []
        new_metadata = []
        for i, metadata in enumerate(self.chunk_metadata):
            if metadata['doc_id'] != doc_id:
                new_chunks.append(self.chunks[i])
                new_metadata.append(metadata)
        
        self.chunks = new_chunks
        self.chunk_metadata = new_metadata
        
        if self.chunks:
            self._update_index()
        else:
            self.index = None
        
        return True
    
    def get_all_documents(self):
        """Get list of all documents"""
        return [
            {
                'id': doc_id,
                'filename': doc_info['filename'],
                'upload_time': doc_info['upload_time'],
                'size': doc_info['content_length']
            }
            for doc_id, doc_info in self.documents.items()
        ]
    
    def save_to_disk(self, path: str):
        """Save RAG system to disk"""
        data = {
            'documents': self.documents,
            'chunks': self.chunks,
            'chunk_metadata': self.chunk_metadata
        }
        with open(f"{path}_data.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        if self.index is not None:
            faiss.write_index(self.index, f"{path}_index.faiss")
    
    def load_from_disk(self, path: str):
        """Load RAG system from disk"""
        try:
            with open(f"{path}_data.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.chunks = data['chunks']
            self.chunk_metadata = data['chunk_metadata']
            
            if os.path.exists(f"{path}_index.faiss"):
                self.index = faiss.read_index(f"{path}_index.faiss")
        except Exception as e:
            print(f"Error loading RAG system: {e}")

# Initialize RAG system with reranker
rag_system = RAGSystem(model, reranker)

# Try to load existing data
try:
    rag_system.load_from_disk("rag_data")
    print("DEBUG: Loaded existing RAG data")
except:
    print("DEBUG: Starting with empty RAG system")

def clean_text(text):
    """Clean and normalize extracted text for JSON extraction"""
    # Preserve structure that might be important for JSON extraction
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Reduce multiple newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces but preserve newlines
    return text.strip()

def generate_customer_extraction_response(prompt, rag_results):
    """Generate natural language response for customer data extraction using Fireworks.ai API"""
    
    # Format RAG results as context
    context = "DOCUMENT CONTENT:\n\n"
    
    for i, result in enumerate(rag_results):
        context += f"=== SECTION {i+1} (from {result['metadata']['filename']}) ===\n"
        context += f"{result['chunk']}\n\n"

    # Enhanced prompt for customer information extraction
    system_prompt = """You are a professional business analyst specialized in extracting and presenting customer information from business documents.

INSTRUCTIONS:
1. Analyze the document content to extract relevant customer/company information
2. Present the information in a clear, organized, and professional manner
3. Focus on key business details such as:
   - Company name and legal information
   - Contact details (address, phone, email)
   - Business type and industry
   - Incorporation details
   - Financial information
   - Key personnel
   - Business relationships and activities
4. If specific information is not available, clearly state what is missing
5. Organize your response with clear headings and bullet points for readability
6. Provide context and explain any complex business relationships or structures
7. Be thorough but concise, focusing on actionable business intelligence

Your goal is to help with customer onboarding and due diligence by presenting all relevant information in an easy-to-understand format."""

    user_prompt = f"""DOCUMENT CONTENT:
{context}

USER REQUEST: {prompt}

Please analyze the documents and provide a comprehensive overview of the customer information found:"""

    # Prepare the request for Fireworks.ai API
    headers = {
        'Authorization': f'Bearer {FIREWORKS_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': FIREWORKS_MODEL,
        'messages': [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ],
        'max_tokens': 4096,
        'temperature': 0.3,  # Slightly higher for more natural responses
        'top_p': 0.9,
        'stream': False
    }

    print(f"DEBUG: Sending customer extraction request to Fireworks.ai")
    
    try:
        response = requests.post(FIREWORKS_BASE_URL, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
                
        else:
            error_msg = response.text if response.text else f"Status code: {response.status_code}"
            return f"I apologize, but I'm having trouble analyzing the documents due to an API error: {error_msg}"
            
    except Exception as e:
        print(f"DEBUG: Request failed: {e}")
        return f"I apologize, but I'm having trouble processing your request: {str(e)}"

def generate_chat_response(prompt, rag_results):
    """Generate chat response using Fireworks.ai API"""
    
    # Format RAG results as context
    context = "RELEVANT DOCUMENT CONTENT:\n\n"
    
    for i, result in enumerate(rag_results):
        context += f"=== SECTION {i+1} (from {result['metadata']['filename']}) ===\n"
        context += f"{result['chunk']}\n\n"

    system_prompt = """You are a helpful AI assistant that answers questions based on the provided document content. 

INSTRUCTIONS:
1. Use the document content to answer the user's question accurately
2. Be specific and reference information from the documents when possible
3. If the information is not in the documents, clearly state that
4. Provide clear, well-structured responses
5. Maintain a professional and helpful tone"""

    user_prompt = f"""DOCUMENT CONTENT:
{context}

USER QUESTION: {prompt}

Please answer the user's question based on the document content above:"""

    # Prepare the request for Fireworks.ai API
    headers = {
        'Authorization': f'Bearer {FIREWORKS_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': FIREWORKS_MODEL,
        'messages': [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ],
        'max_tokens': 2048,
        'temperature': 0.3,
        'top_p': 0.9,
        'stream': False
    }

    print(f"DEBUG: Sending chat request to Fireworks.ai")
    
    try:
        response = requests.post(FIREWORKS_BASE_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            error_msg = response.text if response.text else f"Status code: {response.status_code}"
            return f"I apologize, but I'm having trouble processing your request due to an API error: {error_msg}"
            
    except Exception as e:
        print(f"DEBUG: Chat request failed: {e}")
        return f"I apologize, but I'm having trouble processing your request: {str(e)}"

# Remove the customer schema since we're not using JSON format anymore
# CUSTOMER_SCHEMA has been removed

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_multiple', methods=['POST'])
def upload_multiple():
    """Handle multiple file uploads optimized for customer data extraction"""
    print("DEBUG: Upload multiple endpoint called")
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files part'}), 400
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        print(f"DEBUG: Processing {len(files)} files for customer data extraction")
        uploaded_files = []
        errors = []

        for file in files:
            if file.filename == '':
                continue
            if not allowed_file(file.filename):
                errors.append(f"{file.filename}: File type not allowed")
                continue
                
            try:
                print(f"DEBUG: Processing file: {file.filename}")
                
                if file.filename.lower().endswith('.pdf'):
                    content = process_pdf_file(file)
                else:
                    content = process_text_file(file)
                
                if content is None:
                    errors.append(f"{file.filename}: Could not extract content")
                    continue

                # Add to RAG system
                doc_id = str(uuid.uuid4())
                num_chunks = rag_system.add_document(doc_id, file.filename, content)
                
                uploaded_files.append({
                    'id': doc_id,
                    'filename': file.filename,
                    'size': len(content),
                    'chunks': num_chunks
                })
                
                print(f"DEBUG: Successfully added {file.filename} with {num_chunks} chunks")
                
            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")
                print(f"DEBUG: Error processing {file.filename}: {str(e)}")

        # Save progress
        rag_system.save_to_disk("rag_data")
        session['conversation_history'] = []
        
        print(f"DEBUG: Upload complete. Successfully processed {len(uploaded_files)} files")
        
        return jsonify({
            'uploaded': uploaded_files,
            'errors': errors,
            'total_documents': len(rag_system.documents),
            'total_chunks': len(rag_system.chunks)
        }), 200
        
    except Exception as e:
        print(f"DEBUG: General exception in upload_multiple: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

def process_pdf_file(file):
    """Process PDF file optimized for customer data extraction"""
    try:
        print(f"DEBUG: Processing PDF file: {file.filename}")
        file.seek(0)
        pdf_reader = pypdf.PdfReader(file)
        content = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text() or ''
                if page_text.strip():
                    # Preserve structure for better data extraction
                    content += f"\n=== PAGE {page_num + 1} ===\n{page_text}\n"
            except Exception as page_error:
                print(f"DEBUG: Error extracting text from page {page_num + 1}: {page_error}")
                continue

        if not content.strip():
            return None
            
        return clean_text(content)
        
    except Exception as e:
        print(f"DEBUG: Error processing PDF {file.filename}: {e}")
        return None

def process_text_file(file):
    """Process text file and return content"""
    try:
        file.seek(0)
        raw_content = file.read()
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                content = raw_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            content = raw_content.decode('utf-8', errors='ignore')
        
        content = clean_text(content)
        if not content.strip():
            return None
            
        return content
        
    except Exception as e:
        print(f"DEBUG: Error processing text file {file.filename}: {e}")
        return None

@app.route('/extract_customer_json', methods=['POST'])
def extract_customer_json():
    """Specialized endpoint for customer data extraction with natural language response"""
    data = request.json
    user_input = data.get('message', 'Extract all customer information and provide a comprehensive overview')
    
    if len(rag_system.chunks) == 0:
        return jsonify({'error': 'No documents uploaded for extraction'}), 400

    # Use enhanced search with reranker for better extraction
    search_query = "customer information company details incorporation address phone email business"
    rag_results = rag_system.search_with_reranker(search_query, k=30, final_k=15)
    
    print(f"DEBUG: Found {len(rag_results)} relevant chunks for customer extraction")

    if rag_results:
        response_text = generate_customer_extraction_response(user_input, rag_results)
    else:
        response_text = "I couldn't find any relevant customer information in the uploaded documents. Please ensure you've uploaded the appropriate customer onboarding documents, such as incorporation certificates, business registration forms, or other relevant business documentation."

    sources = []
    if rag_results:
        seen_files = set()
        for result in rag_results:
            filename = result['metadata']['filename']
            if filename not in seen_files:
                seen_files.add(filename)
                sources.append(filename)

    return jsonify({
        'response': response_text,
        'sources': sources,
        'chunks_used': len(rag_results),
        'extraction_type': 'customer_analysis'
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with Fireworks.ai API"""
    data = request.json
    user_input = data.get('message', '')
    if not user_input.strip():
        return jsonify({'error': 'Please enter a message'}), 400
    
    search_mode = data.get('search_mode', 'enhanced')
    
    # Determine chunk count based on search mode
    if search_mode == 'comprehensive':
        candidate_k, final_k = 40, 20
    elif search_mode == 'enhanced':
        candidate_k, final_k = 25, 15
    else:  # basic
        candidate_k, final_k = 15, 8
    
    # Use enhanced search with reranker
    rag_results = rag_system.search_with_reranker(user_input, k=candidate_k, final_k=final_k)
    
    print(f"DEBUG: Found {len(rag_results)} relevant chunks with reranker")

    if rag_results:
        # For customer extraction requests, use specialized response generation
        if any(keyword in user_input.lower() for keyword in ['extract', 'customer data', 'customer information', 'company details', 'business information']):
            ai_response = generate_customer_extraction_response(user_input, rag_results)
        else:
            # Regular chat response using Fireworks.ai
            ai_response = generate_chat_response(user_input, rag_results)
    else:
        ai_response = "I couldn't find any relevant information in the uploaded documents for your query. Please make sure you've uploaded the relevant documents or try rephrasing your question."

    sources = []
    if rag_results:
        seen_files = set()
        for result in rag_results:
            filename = result['metadata']['filename']
            if filename not in seen_files:
                seen_files.add(filename)
                sources.append(filename)

    return jsonify({
        'response': ai_response,
        'sources': sources,
        'chunks_used': len(rag_results),
        'search_mode': search_mode
    })

@app.route('/documents', methods=['GET'])
def get_documents():
    """Get list of all uploaded documents"""
    documents = rag_system.get_all_documents()
    return jsonify({
        'documents': documents,
        'total_chunks': len(rag_system.chunks)
    })

@app.route('/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a specific document"""
    success = rag_system.remove_document(doc_id)
    if success:
        rag_system.save_to_disk("rag_data")
        return jsonify({
            'status': 'success',
            'message': 'Document deleted successfully',
            'remaining_documents': len(rag_system.documents)
        })
    else:
        return jsonify({'error': 'Document not found'}), 404

@app.route('/clear_all', methods=['POST'])
def clear_all():
    """Clear all documents and conversation history"""
    rag_system.documents.clear()
    rag_system.chunks.clear()
    rag_system.chunk_metadata.clear()
    rag_system.index = None
    rag_system.save_to_disk("rag_data")
    session.clear()
    return jsonify({'status': 'success', 'message': 'All data cleared'})

if __name__ == "__main__":
    app.run(debug=True)