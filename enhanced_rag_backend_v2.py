from flask import Flask, request, render_template, jsonify, session
import requests
import os
import pypdf
import re
import tempfile
import hashlib
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import List, Dict, Tuple
import uuid
import json

# New imports for table detection
import tabula
import pdfplumber

# Import our enhanced financial detection module
from enhanced_financial_detection import FinancialDataDetector

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure maximum file size (32MB to handle larger batches)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

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

# Initialize sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize financial data detector
financial_detector = FinancialDataDetector()

# RAG System Class (enhanced)
class EnhancedRAGSystem:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.documents = {}  # Store document metadata
        self.chunks = []     # Store text chunks
        self.chunk_metadata = []  # Store metadata for each chunk
        self.index = None    # FAISS index
        self.dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > chunk_size * 0.8:
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
        
        # Chunk the document
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
    
    def search_with_reranking(self, query: str, k: int = 25, final_k: int = 15) -> List[Dict]:
        """Enhanced search with re-ranking and diversity"""
        if not self.chunks or self.index is None:
            return []
        
        # Step 1: Get more candidates than needed
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
        
        # Step 2: Diversify results - ensure we get chunks from different documents
        diversified_results = self._diversify_results(candidates, final_k)
        
        return diversified_results
    
    def _diversify_results(self, candidates: List[Dict], final_k: int) -> List[Dict]:
        """Ensure diversity in results by including chunks from different documents"""
        if not candidates:
            return []
        
        # Group candidates by document
        doc_groups = {}
        for candidate in candidates:
            doc_id = candidate['metadata']['doc_id']
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(candidate)
        
        # Sort each group by relevance (distance)
        for doc_id in doc_groups:
            doc_groups[doc_id].sort(key=lambda x: x['distance'])
        
        # Select results using round-robin to ensure diversity
        selected = []
        doc_ids = list(doc_groups.keys())
        max_per_doc = max(1, final_k // len(doc_ids)) if doc_ids else 1
        
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
            
            remaining_candidates.sort(key=lambda x: x['distance'])
            selected.extend(remaining_candidates[:final_k - len(selected)])
        
        # Sort final results by relevance
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

# Initialize enhanced RAG system
rag_system = EnhancedRAGSystem(model)

def clean_text(text):
    """Clean and normalize extracted text"""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def enhanced_format_response_for_display(response_text, context_chunks):
    """Enhanced response formatting with financial table detection"""
    
    # Use financial detector to enhance response with tables
    enhanced_response = financial_detector.enhance_response_with_tables(response_text, context_chunks)
    
    # Apply additional formatting for better display
    sections = re.split(r'\n\s*\n', enhanced_response.strip())
    formatted_sections = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Check if this is already a markdown table
        if '|' in section and ('---' in section or section.count('|') > 4):
            formatted_sections.append(section)
            continue

        # Look for financial data patterns
        financial_pattern = r'([A-Za-z\s]+?):\s*((?:RM|USD|\$)?\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|mil|billion|bil|thousand|k|times|%))?)'
        financial_matches = re.findall(financial_pattern, section, re.IGNORECASE)

        if len(financial_matches) >= 3:
            # Build a Markdown table for financial data
            table_md = "| Financial Metric | Value |\n"
            table_md += "|------------------|-------|\n"
            for metric, value in financial_matches:
                table_md += f"| {metric.strip()} | {value.strip()} |\n"
            formatted_sections.append(table_md)
        else:
            # Regular paragraph
            formatted_sections.append(section)

    return "\n\n".join(formatted_sections)

def generate_enhanced_response_with_gemma(prompt, conversation_history, rag_results, search_mode="enhanced"):
    """Generate response using RAG results with enhanced financial formatting via local Gemma API"""
    
    # Detect query type for appropriate response formatting
    query_type = financial_detector.detect_financial_query_type(prompt)
    
    # Calculate context window based on number of results
    total_context_length = sum(len(result['chunk']) for result in rag_results)
    print(f"DEBUG: Total context length: {total_context_length} characters, Query type: {query_type}")
    
    # Format RAG results as context
    context = "RELEVANT CONTEXT FROM DOCUMENTS:\n\n"
    
    # Group results by document for better organization
    doc_groups = {}
    for result in rag_results:
        filename = result['metadata']['filename']
        if filename not in doc_groups:
            doc_groups[filename] = []
        doc_groups[filename].append(result)
    
    # Add context from each document
    for filename, results in doc_groups.items():
        context += f"=== FROM DOCUMENT: {filename} ===\n"
        for i, result in enumerate(results):
            chunk_info = f"[Chunk {result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}]"
            context += f"{chunk_info}\n{result['chunk']}\n\n"
        context += "\n"

    # Format conversation history
    recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
    formatted_history = "\n".join(recent_history)

    # Adjust instructions and context window based on query type
    if query_type == 'simple':
        response_guidance = """
CRITICAL: This is a simple identification question. Provide a direct, concise answer.
- Answer in 1-3 sentences maximum
- Do not provide additional information unless specifically requested
- Focus only on answering exactly what was asked"""
        num_ctx = 2048
    elif query_type == 'financial':
        response_guidance = """
FINANCIAL DATA RESPONSE:
- Present financial data in structured tables when possible
- Highlight key financial metrics and ratios
- Include year-over-year comparisons when available
- Use clear formatting for currency amounts and percentages
- Provide context for financial performance"""
        num_ctx = 6144
    else:
        response_guidance = """
- Provide comprehensive analysis using available information
- Structure response with clear headings and sections
- Include relevant data in table format when appropriate
- Cross-reference information between documents when relevant"""
        num_ctx = 4096

    # Prepare the full prompt for Gemma
    full_prompt = f"""You are a financial analysis AI assistant. {response_guidance}

CONTEXT INFORMATION:
{context}

CONVERSATION HISTORY:
{formatted_history}

USER QUESTION: {prompt}

Please provide a comprehensive answer based on the context provided. When presenting financial data, use markdown tables for better readability. Highlight important financial metrics and provide clear explanations."""

    # Determine if this is a simple question for context optimization
    is_simple_question = query_type == 'simple'
    
    print(f"DEBUG: Using {len(rag_results)} chunks from {len(doc_groups)} documents")
    print(f"DEBUG: Context window size: {num_ctx}, Simple question: {is_simple_question}")

    try:
        # Make API call to local Gemma model
        url = 'http://localhost:11434/api/generate'
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': 'gemma3:4b',
            'prompt': full_prompt,
            'stream': False,
            'options': {
                'temperature': 0.1,
                'num_ctx': num_ctx,  # Dynamic context window
                'top_p': 0.9,
                'repeat_penalty': 0.8
            }
        }
        
        print(f"DEBUG: Making request to Gemma API at {url}")
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            response_data = response.json()
            ai_response = response_data.get('response', '')
            
            print(f"DEBUG: Received response from Gemma API: {len(ai_response)} characters")
            
            # Extract context chunks for table enhancement
            context_chunks = [result['chunk'] for result in rag_results]
            
            # Apply enhanced formatting
            formatted_response = enhanced_format_response_for_display(ai_response, context_chunks)
            
            return formatted_response
        else:
            print(f"DEBUG: Gemma API error: {response.status_code} - {response.text}")
            return f"Error generating response: {response.status_code} - {response.text}"
            
    except requests.exceptions.ConnectionError:
        print("DEBUG: Connection error - Gemma API server may not be running")
        return "Error: Cannot connect to Gemma API. Please ensure the Gemma server is running on localhost:11434"
    except requests.exceptions.Timeout:
        print("DEBUG: Timeout error - Gemma API request took too long")
        return "Error: Gemma API request timed out. Please try again."
    except Exception as e:
        print(f"DEBUG: Exception in generate_response_with_gemma: {e}")
        return f"Error generating response: {str(e)}"

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        total_chunks = 0
        
        for file in files:
            if file.filename == '':
                continue
                
            filename = file.filename
            print(f"DEBUG: Processing file: {filename}")
            
            try:
                # Generate unique document ID
                doc_id = str(uuid.uuid4())
                
                # Read file content based on type
                if filename.lower().endswith('.pdf'):
                    # Handle PDF files
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        file.save(temp_file.name)
                        
                        # Try pdfplumber first for better text extraction
                        try:
                            with pdfplumber.open(temp_file.name) as pdf:
                                text_content = []
                                for page in pdf.pages:
                                    page_text = page.extract_text()
                                    if page_text:
                                        text_content.append(page_text)
                                content = "\n".join(text_content)
                        except Exception as e:
                            print(f"DEBUG: pdfplumber failed, trying pypdf: {e}")
                            # Fallback to pypdf
                            with open(temp_file.name, 'rb') as pdf_file:
                                pdf_reader = pypdf.PdfReader(pdf_file)
                                content = ""
                                for page in pdf_reader.pages:
                                    content += page.extract_text() + "\n"
                        
                        os.unlink(temp_file.name)
                else:
                    # Handle text files
                    content = file.read().decode('utf-8', errors='ignore')
                
                # Clean the content
                content = clean_text(content)
                
                if not content.strip():
                    results.append({
                        'filename': filename,
                        'status': 'error',
                        'message': 'No readable content found'
                    })
                    continue
                
                # Add to RAG system
                chunks_added = rag_system.add_document(doc_id, filename, content)
                total_chunks += chunks_added
                
                results.append({
                    'filename': filename,
                    'status': 'success',
                    'chunks': chunks_added,
                    'doc_id': doc_id
                })
                
                print(f"DEBUG: Successfully processed {filename}: {chunks_added} chunks")
                
            except Exception as e:
                print(f"DEBUG: Error processing {filename}: {e}")
                results.append({
                    'filename': filename,
                    'status': 'error',
                    'message': str(e)
                })
        
        # Save RAG system state
        try:
            rag_system.save_to_disk("rag_data")
            print("DEBUG: RAG system saved to disk")
        except Exception as e:
            print(f"DEBUG: Error saving RAG system: {e}")
        
        return jsonify({
            'results': results,
            'total_chunks': total_chunks,
            'total_documents': len(rag_system.documents)
        })
        
    except Exception as e:
        print(f"DEBUG: Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        search_mode = data.get('search_mode', 'enhanced')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        print(f"DEBUG: Chat request - Message: {message[:100]}..., Mode: {search_mode}")
        
        # Get conversation history from session
        if 'conversation' not in session:
            session['conversation'] = []
        
        conversation_history = session['conversation']
        
        # Determine search parameters based on mode
        search_params = {
            'basic': {'k': 25, 'final_k': 5},
            'enhanced': {'k': 35, 'final_k': 15},
            'comprehensive': {'k': 50, 'final_k': 30}
        }
        
        params = search_params.get(search_mode, search_params['enhanced'])
        
        # Search for relevant chunks
        rag_results = rag_system.search_with_reranking(
            message, 
            k=params['k'], 
            final_k=params['final_k']
        )
        
        print(f"DEBUG: Found {len(rag_results)} relevant chunks")
        
        if not rag_results:
            response_text = "I don't have any relevant information in the uploaded documents to answer your question. Please upload some documents first or ask about the content that's available."
        else:
            # Generate response using enhanced method with Gemma
            response_text = generate_enhanced_response_with_gemma(
                message, 
                conversation_history, 
                rag_results, 
                search_mode
            )
        
        # Update conversation history
        conversation_history.append(f"User: {message}")
        conversation_history.append(f"Assistant: {response_text}")
        
        # Keep only last 20 exchanges
        if len(conversation_history) > 40:
            conversation_history = conversation_history[-40:]
        
        session['conversation'] = conversation_history
        
        # Prepare sources information
        sources = []
        for result in rag_results[:5]:  # Show top 5 sources
            sources.append({
                'filename': result['metadata']['filename'],
                'chunk_index': result['metadata']['chunk_index'],
                'relevance': 1 - result['distance']  # Convert distance to relevance score
            })
        
        return jsonify({
            'response': response_text,
            'sources': sources,
            'chunks_used': len(rag_results)
        })
        
    except Exception as e:
        print(f"DEBUG: Chat error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    try:
        documents = rag_system.get_all_documents()
        return jsonify({
            'documents': documents,
            'total_chunks': len(rag_system.chunks)
        })
    except Exception as e:
        print(f"DEBUG: Get documents error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    try:
        success = rag_system.remove_document(doc_id)
        if success:
            # Save updated state
            rag_system.save_to_disk("rag_data")
            return jsonify({'message': 'Document deleted successfully'})
        else:
            return jsonify({'error': 'Document not found'}), 404
    except Exception as e:
        print(f"DEBUG: Delete document error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_all():
    try:
        global rag_system
        rag_system = EnhancedRAGSystem(model)
        
        # Clear saved data
        for file_path in ["rag_data_data.pkl", "rag_data_index.faiss"]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clear session
        session.clear()
        
        return jsonify({'message': 'All documents cleared successfully'})
    except Exception as e:
        print(f"DEBUG: Clear all error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify Gemma API connectivity"""
    try:
        # Test connection to Gemma API
        url = 'http://localhost:11434/api/generate'
        test_data = {
            'model': 'gemma3:4b',
            'prompt': 'Hello',
            'stream': False,
            'options': {'temperature': 0.1, 'num_ctx': 512}
        }
        
        response = requests.post(url, json=test_data, timeout=10)
        
        if response.status_code == 200:
            return jsonify({
                'status': 'healthy',
                'gemma_api': 'connected',
                'documents': len(rag_system.documents),
                'chunks': len(rag_system.chunks)
            })
        else:
            return jsonify({
                'status': 'degraded',
                'gemma_api': 'error',
                'error': f"Gemma API returned {response.status_code}",
                'documents': len(rag_system.documents),
                'chunks': len(rag_system.chunks)
            }), 503
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'status': 'degraded',
            'gemma_api': 'disconnected',
            'error': 'Cannot connect to Gemma API on localhost:11434',
            'documents': len(rag_system.documents),
            'chunks': len(rag_system.chunks)
        }), 503
    except Exception as e:
        return jsonify({
            'status': 'error',
            'gemma_api': 'unknown',
            'error': str(e),
            'documents': len(rag_system.documents),
            'chunks': len(rag_system.chunks)
        }), 500

if __name__ == '__main__':
    # Try to load existing data
    try:
        rag_system.load_from_disk("rag_data")
        print("DEBUG: Loaded existing RAG data")
    except:
        print("DEBUG: Starting with empty RAG system")
    
    print("DEBUG: Starting enhanced RAG backend server with Gemma API integration...")
    print("DEBUG: Gemma API expected at http://localhost:11434/api/generate")
    app.run(host='0.0.0.0', port=5000, debug=True)

