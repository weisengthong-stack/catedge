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

# New imports for table detection
import tabula
import pdfplumber

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

# RAG System Class (unchanged)
class QuestionRouter:
    """Routes questions to appropriate documents based on country mentions and question type"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        
        # Enhanced country mappings with more variations
        self.country_mappings = {
            'thailand': ['thailand', 'thai'],
            'singapore': ['singapore', 'singaporean', 'sg', 'sgp', 'sing'],
            'malaysia': ['malaysia', 'malaysian', 'my', 'mys', 'malay'],
            'indonesia': ['indonesia', 'indonesian', 'id', 'idn', 'indo']
        }
        
        # Comparative question patterns
        self.comparative_patterns = [
            r'\b(?:compare|comparison|versus|vs\.?|against|between)\b',
            r'\b(?:difference|different|differ)\b.*\b(?:between|among)\b',
            r'\b(?:which\s+(?:country|countries))\b.*\b(?:better|best|worse|worst|higher|lower)\b',
            r'\b(?:all\s+(?:countries|nations))\b',
            r'\bmultiple\s+countries\b',
            r'\bacross\s+(?:countries|nations)\b'
        ]
        
        # General question patterns (no specific country focus)
        self.general_patterns = [
            r'\b(?:what\s+is|how\s+does|why\s+do|when\s+did)\b.*\b(?:methodology|approach|framework|process)\b',
            r'\b(?:general|overall|typical|common|standard)\b',
            r'\b(?:explain|describe|define)\b.*\b(?:methodology|approach|framework)\b'
        ]

    def extract_countries(self, question: str) -> List[str]:
        """Extract country names mentioned in the question"""
        question_lower = question.lower()
        mentioned_countries = set()
        
        for country, variants in self.country_mappings.items():
            for variant in variants:
                if variant in question_lower:
                    mentioned_countries.add(country)
        
        return list(mentioned_countries)

    def is_comparative_question(self, question: str) -> bool:
        """Check if the question is asking for comparisons between countries"""
        question_lower = question.lower()
        
        for pattern in self.comparative_patterns:
            if re.search(pattern, question_lower):
                return True
        
        return False

    def is_general_question(self, question: str) -> bool:
        """Check if the question is general (not country-specific)"""
        question_lower = question.lower()
        
        # Check for general patterns
        for pattern in self.general_patterns:
            if re.search(pattern, question_lower):
                return True
        
        # If no countries are mentioned, it might be general
        mentioned_countries = self.extract_countries(question)
        return len(mentioned_countries) == 0

    def get_document_ids_for_countries(self, countries: List[str]) -> List[str]:
        """Get document IDs that correspond to the specified countries - FIXED VERSION"""
        doc_ids = []
        
        print(f"DEBUG: Looking for documents matching countries: {countries}")
        print(f"DEBUG: Available documents: {list(self.rag_system.documents.keys())}")
        
        for doc_id, doc_info in self.rag_system.documents.items():
            filename = doc_info['filename'].lower()
            print(f"DEBUG: Checking filename: {filename}")
            
            for country in countries:
                # Get all variants for this country
                variants = self.country_mappings.get(country, [country])
                print(f"DEBUG: Checking country '{country}' with variants: {variants}")
                
                # Check if any variant appears in filename
                for variant in variants:
                    if variant in filename:
                        print(f"DEBUG: Found match! '{variant}' in '{filename}' -> doc_id: {doc_id}")
                        doc_ids.append(doc_id)
                        break  # Found a match for this country, move to next country
                else:
                    continue  # No variant matched, try next country
                break  # Found a match for this country, move to next document
        
        print(f"DEBUG: Final matched doc_ids: {doc_ids}")
        return doc_ids

    def route_question(self, question: str) -> Tuple[str, List[str], Dict]:
        """
        Route the question to appropriate search strategy
        Returns: (route_type, doc_ids, metadata)
        """
        mentioned_countries = self.extract_countries(question)
        is_comparative = self.is_comparative_question(question)
        is_general = self.is_general_question(question)
        
        print(f"DEBUG: Route analysis - Countries: {mentioned_countries}, Comparative: {is_comparative}, General: {is_general}")
        
        metadata = {
            'mentioned_countries': mentioned_countries,
            'is_comparative': is_comparative,
            'is_general': is_general,
            'question': question
        }
        
        # Route 1: Comparative questions - use all documents
        if is_comparative:
            print("DEBUG: Routing as 'comparative'")
            return 'comparative', [], metadata
        
        # Route 2: Single country specific questions
        elif len(mentioned_countries) == 1 and not is_general:
            print("DEBUG: Routing as 'country_specific'")
            doc_ids = self.get_document_ids_for_countries(mentioned_countries)
            print(f"DEBUG: Country-specific routing found {len(doc_ids)} documents")
            return 'country_specific', doc_ids, metadata
        
        # Route 3: Multiple countries mentioned but not comparative
        elif len(mentioned_countries) > 1 and not is_comparative:
            print("DEBUG: Routing as 'multi_country'")
            doc_ids = self.get_document_ids_for_countries(mentioned_countries)
            return 'multi_country', doc_ids, metadata
        
        # Route 4: General questions - use all documents
        else:
            print("DEBUG: Routing as 'general'")
            return 'general', [], metadata
    
class RAGSystem:
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
    
    def search(self, query: str, k: int = 15) -> List[Dict]:
        """Search for relevant chunks - increased default from 5 to 15"""
        if not self.chunks or self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            min(k, len(self.chunks))
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'distance': float(distances[0][i])
                })
        
        return results
    
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
    
    def enhanced_search_by_document(self, query: str, doc_ids: List[str] = None, k: int = 10) -> List[Dict]:
        """Enhanced search within specific documents with better ranking"""
        if not self.chunks or self.index is None:
            return []
        
        if doc_ids is None:
            return self.search(query, k)
        
        # Filter chunks by document IDs
        valid_indices = []
        valid_chunks = []
        valid_metadata = []
        
        for i, metadata in enumerate(self.chunk_metadata):
            if metadata['doc_id'] in doc_ids:
                valid_indices.append(i)
                valid_chunks.append(self.chunks[i])
                valid_metadata.append(metadata)
        
        if not valid_chunks:
            print(f"DEBUG: No chunks found for doc_ids: {doc_ids}")
            return []
        
        print(f"DEBUG: Found {len(valid_chunks)} chunks across {len(doc_ids)} documents")
        
        # Get embeddings for valid chunks only
        query_embedding = self.embedding_model.encode([query])
        chunk_embeddings = self.embedding_model.encode(valid_chunks)
        
        # Calculate similarities
        similarities = np.dot(query_embedding, chunk_embeddings.T)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': valid_chunks[idx],
                'metadata': valid_metadata[idx],
                'distance': float(1 - similarities[idx])  # Convert similarity to distance
            })
        
        return results
    
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

# Initialize RAG system
rag_system = RAGSystem(model)

# Try to load existing data
try:
    rag_system.load_from_disk("rag_data")
    print("DEBUG: Loaded existing RAG data")
except:
    print("DEBUG: Starting with empty RAG system")


def clean_text(text):
    """Clean and normalize extracted text"""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_response_for_display(response_text):
    """Enhanced response formatting with better structure and numerical data handling - ROBUST VERSION"""
    
    try:
        # Split response into logical sections
        sections = re.split(r'\n\s*\n', response_text.strip())
        formatted_sections = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Initialize variables
            numerical_matches = []
            metric_matches = []

            # FIXED: Find any "label – value" pairs (e.g. "Year – 2021")
            # Using more robust pattern with proper escaping
            try:
                # Pattern explanation:
                # - [^\n–—\-] : match any character except newline or dash characters
                # - +? : match one or more characters (non-greedy)
                # - \s* : optional whitespace
                # - [–—\-] : match any type of dash
                # - \s* : optional whitespace
                # - \d+ : one or more digits
                # - (?:\.\d+)? : optional decimal part
                # - (?:%| billion| million| thousand)? : optional units
                numerical_pattern = r'([^\n–—\-]+?)\s*[–—\-]\s*(\d+(?:\.\d+)?(?:%| billion| million| thousand)?)'
                numerical_matches = re.findall(numerical_pattern, section, re.IGNORECASE)
            except re.error as e:
                print(f"DEBUG: Regex error in numerical_pattern: {e}")
                print(f"DEBUG: Problematic section: {section[:100]}...")

            # Find any "Key: Value" metrics
            try:
                metric_pattern = r'^([^:\n]+):\s*(.+)$'
                metric_matches = re.findall(metric_pattern, section, re.MULTILINE)
            except re.error as e:
                print(f"DEBUG: Regex error in metric_pattern: {e}")
                print(f"DEBUG: Problematic section: {section[:100]}...")

            # Process numerical matches
            if len(numerical_matches) >= 2:
                try:
                    table_md = "| Metric | Value |\n"
                    table_md += "|--------|-------|\n"
                    for label, val in numerical_matches:
                        # Clean up the label and value, handle potential None values
                        clean_label = str(label).strip() if label else "N/A"
                        clean_val = str(val).strip() if val else "N/A"
                        # Escape any pipe characters in the data
                        clean_label = clean_label.replace('|', '\\|')
                        clean_val = clean_val.replace('|', '\\|')
                        table_md += f"| {clean_label} | {clean_val} |\n"
                    formatted_sections.append(table_md)
                except Exception as e:
                    print(f"DEBUG: Error creating numerical table: {e}")
                    formatted_sections.append(section)

            # Process metric matches
            elif len(metric_matches) >= 2:
                try:
                    table_md = "| Metric | Value |\n"
                    table_md += "|--------|-------|\n"
                    for key, val in metric_matches:
                        # Clean up the key and value, handle potential None values
                        clean_key = str(key).strip() if key else "N/A"
                        clean_val = str(val).strip() if val else "N/A"
                        # Escape any pipe characters in the data
                        clean_key = clean_key.replace('|', '\\|')
                        clean_val = clean_val.replace('|', '\\|')
                        table_md += f"| {clean_key} | {clean_val} |\n"
                    formatted_sections.append(table_md)
                except Exception as e:
                    print(f"DEBUG: Error creating metric table: {e}")
                    formatted_sections.append(section)

            else:
                # Regular paragraph - no special formatting needed
                formatted_sections.append(section)

        # Join everything back, preserving blank lines between sections
        return "\n\n".join(formatted_sections)

    except Exception as e:
        print(f"DEBUG: Critical error in format_response_for_display: {e}")
        # Return original text if formatting fails
        return response_text


def format_response_for_display_simple(response_text):
    """Simplified version that avoids complex regex patterns"""
    
    try:
        # Split response into logical sections
        sections = re.split(r'\n\s*\n', response_text.strip())
        formatted_sections = []

        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Look for simple key-value pairs separated by colons
            lines = section.split('\n')
            key_value_pairs = []
            
            for line in lines:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        if key and value:
                            key_value_pairs.append((key, value))
            
            # If we found multiple key-value pairs, create a table
            if len(key_value_pairs) >= 3:
                table_md = "| Metric | Value |\n"
                table_md += "|--------|-------|\n"
                for key, value in key_value_pairs:
                    # Escape pipe characters
                    clean_key = key.replace('|', '\\|')
                    clean_value = value.replace('|', '\\|')
                    table_md += f"| {clean_key} | {clean_value} |\n"
                formatted_sections.append(table_md)
            else:
                # Regular paragraph
                formatted_sections.append(section)

        return "\n\n".join(formatted_sections)

    except Exception as e:
        print(f"DEBUG: Error in simplified formatting: {e}")
        return response_text


def generate_response_with_rag(prompt, conversation_history, rag_results, search_mode="enhanced", is_simple_question=False):
    """Generate response using RAG results with enhanced formatting and more context"""
    
    # Calculate context window based on number of results
    total_context_length = sum(len(result['chunk']) for result in rag_results)
    print(f"DEBUG: Total context length: {total_context_length} characters")
    
    # Format RAG results as context with better organization
    context = "RELEVANT CONTEXT FROM DOCUMENTS:\n\n"
    
    # Group results by document for better organization
    doc_groups = {}
    for result in rag_results:
        filename = result['metadata']['filename']
        if filename not in doc_groups:
            doc_groups[filename] = []
        doc_groups[filename].append(result)
    
    # For simple questions, provide more focused context
    if is_simple_question:
        # Just include the most relevant chunks without extensive grouping
        context += "KEY INFORMATION:\n"
        for i, result in enumerate(rag_results[:3]):  # Limit to top 3 for simple questions
            context += f"[Source {i+1}] {result['chunk']}\n\n"
    else:
        # Add context from each document for complex questions
        for filename, results in doc_groups.items():
            context += f"=== FROM DOCUMENT: {filename} ===\n"
            for i, result in enumerate(results):
                chunk_info = f"[Chunk {result['metadata']['chunk_index'] + 1}/{result['metadata']['total_chunks']}]"
                context += f"{chunk_info}\n{result['chunk']}\n\n"
            context += "\n"

    # Format conversation history
    recent_history = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
    formatted_history = "\n".join(recent_history)

    # Adjust context length and instructions based on question type
    if is_simple_question:
        context_instructions = """Answer the user's question DIRECTLY and CONCISELY. Provide only the specific information requested without additional details or context unless explicitly asked."""
        num_ctx = 2048  # Smaller context for simple questions
        response_guidance = """
CRITICAL: This is a simple identification question. Provide a direct, concise answer.
- Answer in 1-3 sentences maximum
- Do not provide additional information unless specifically requested
- Do not create tables or extensive formatting for simple questions
- Focus only on answering exactly what was asked"""
    else:
        if search_mode == "comprehensive":
            context_instructions = """You have access to extensive context from multiple documents. Use this comprehensive information to provide detailed, well-rounded answers. Cross-reference information between documents when relevant."""
            num_ctx = 8192  # Larger context window
        elif search_mode == "enhanced":
            context_instructions = """You have access to enhanced context from relevant documents. Provide thorough answers using the available information from multiple sources."""
            num_ctx = 6144
        else:
            context_instructions = """Answer based on the provided context from the documents."""
            num_ctx = 4096
        
        response_guidance = """
- Group related information into sections with **heading** when providing detailed information
- Use table format when comparing multiple companies or periods
- Provide comprehensive analysis when requested"""

    # Enhanced prompt for better structure with more context
    full_prompt = f"""{context}

CONVERSATION HISTORY:
{formatted_history}

USER QUESTION: {prompt}

INSTRUCTIONS:
{context_instructions}

Please answer the question based on the context provided above. Follow these guidelines:

RESPONSE STYLE:
- Answer the user's question DIRECTLY and CONCISELY first
- Only provide additional details if specifically requested
- If the user asks for a simple list or specific information, provide exactly that
- Don't expand beyond what was asked unless the user requests more details

{response_guidance}

FORMATTING RULES:
1. For simple questions (like "what companies are these documents for"), provide a direct answer
2. Only compare between different files if a comparison is specifically requested
3. Do not take any information from the Peer Comparison table in the documents
4. If user asked about financial information, refer to Statement of Comprehensive Income sections
5. Group related information into sections with **heading** only when providing detailed information
6. Use paragraph breaks to separate different discussion points
7. Synthesize information from multiple documents when available
8. If the context doesn't contain relevant information, say so clearly
9. Make responses predictable and consistent across all documents
10. Do not show the cited document names in the response unless specifically asked
11. Please DO NOT answer any questions other than related to the documents
12. ESG Descriptor is not related to Gearing Ratio, please do not answer questions linking both
13. Stay on topic and do not hallucinate - only use information from the provided context
14. When you have information from multiple sources, synthesize and compare them in table format when appropriate
15. Singapore has the lowest cost to register property at 2.9% of property value .

CONTEXTUAL RULES:
1. the number of procedure days required to start a business in Singapore is 2 days only, refer to Page 4 : Procedures (number), all documents will have the same format for this information
2. Singapore has the lowest cost to register property at 2.9% of property value .
3. Singapore has a Building quality control index of 13.0 while Malaysia has 13.0, Indonesia has 13.8 and Thailand has 11.0, therefore the highest is Indonesia because Indonesia particularly due to its strict control before, during, and after construction, along with strong liability provisions. present this finding in a table format.
4. the high level steps to obtain a construction permit in Malaysia is in page 11 : Details – Dealing with Construction Permits in Malaysia – Procedure, Time and Cost
5. the key steps to obtain the construction permits is as follows :
    1. Obtain technical conditions from the Water Authority (SYABAS)
    2. Submit and obtain development approval via the One-Stop Centre (OSC)
    3. Submit pre-construction notifications
    4. Request utilities inspections through OSC
    5. Receive inspections from water and fire authorities
    6. Obtain clearance letters from OSC
    7. File the Certificate of Completion and Compliance (CCC)
    8. Receive water connection
6. Singapore has the LOWEST cost to register property at 2.9% of property value while Malaysia is 3.5%, Thailand is 7.2% and Indonesia is 8.3% and Indonesia is the highest. please make this correct/
7. Never say that Indonesia, Malaysia or Thailand has the lowest cost to register property, because Singapore has the lowest cost to register property at 2.9%

IMPORTANT: You are analyzing {len(rag_results)} chunks from {len(doc_groups)} documents. Answer the user's specific question directly and concisely.
"""

    print(f"DEBUG: Using {len(rag_results)} chunks from {len(doc_groups)} documents")
    print(f"DEBUG: Context window size: {num_ctx}, Simple question: {is_simple_question}")
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

    print(f"DEBUG: About to send request to Ollama")
    print(f"DEBUG: URL: {url}")
    print(f"DEBUG: Prompt length: {len(full_prompt)}")
    
    try:
        print("DEBUG: Sending request to Ollama...")
        response = requests.post(url, headers=headers, json=data, timeout=120)
        print(f"DEBUG: Ollama response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            raw_response = result.get('response', 'No response received')
            print(f"DEBUG: Received response from Ollama (length: {len(raw_response)})")
            formatted_response = format_response_for_display(raw_response)
            return formatted_response
        else:
            print(f"DEBUG: Ollama error response: {response.text}")
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.Timeout:
        print("DEBUG: Ollama request timed out")
        return "Error: Request timed out. The query might be too complex or the context too large."
    except requests.exceptions.ConnectionError as e:
        print(f"DEBUG: Connection error to Ollama: {e}")
        return "Error: Cannot connect to Ollama. Make sure Ollama is running on port 11434."
    except Exception as e:
        print(f"DEBUG: Unexpected error: {e}")
        return f"Error: {e}"



@app.route('/search_advanced', methods=['POST'])
def search_advanced():
    """Advanced search endpoint with configurable parameters"""
    data = request.json
    query = data.get('query', '')
    search_mode = data.get('search_mode', 'enhanced')  # basic, enhanced, comprehensive
    doc_ids = data.get('doc_ids', None)  # Filter by specific document IDs
    
    if not query.strip():
        return jsonify({'error': 'Please provide a search query'}), 400

    # Determine chunk count based on search mode
    if search_mode == 'comprehensive':
        max_chunks = 30
    elif search_mode == 'enhanced':
        max_chunks = 15
    else:  # basic
        max_chunks = 5

    # Use appropriate search method
    if doc_ids:
        results = rag_system.enhanced_search_by_document(query, doc_ids, k=max_chunks)
    elif search_mode in ['enhanced', 'comprehensive']:
        candidate_count = 50 if search_mode == 'comprehensive' else 25
        results = rag_system.search_with_reranking(query, k=candidate_count, final_k=max_chunks)
    else:
        results = rag_system.search(query, k=max_chunks)
    
    # Analyze document distribution
    doc_distribution = {}
    for result in results:
        filename = result['metadata']['filename']
        doc_distribution[filename] = doc_distribution.get(filename, 0) + 1
    
    return jsonify({
        'query': query,
        'search_mode': search_mode,
        'results': [
            {
                'chunk_preview': r['chunk'][:300] + '...' if len(r['chunk']) > 300 else r['chunk'],
                'filename': r['metadata']['filename'],
                'chunk_index': r['metadata']['chunk_index'],
                'total_chunks': r['metadata']['total_chunks'],
                'distance': r['distance']
            }
            for r in results
        ],
        'total_chunks_found': len(results),
        'documents_covered': len(doc_distribution),
        'document_distribution': doc_distribution
    })


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'txt', 'md', 'py', 'js', 'html', 'css', 'json', 'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index_demo.html')

@app.route('/upload_multiple', methods=['POST'])
def upload_multiple():
    """Handle multiple file uploads with table detection - optimized for large batches"""
    print("DEBUG: Upload multiple endpoint called")
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files part'}), 400
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        print(f"DEBUG: Processing {len(files)} files")
        uploaded_files = []
        errors = []
        processed_count = 0

        # Process files in smaller batches to avoid memory issues
        batch_size = 5
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            print(f"DEBUG: Processing batch {i//batch_size + 1}/{(len(files) + batch_size - 1)//batch_size}")
            
            for file in batch:
                if file.filename == '':
                    continue
                if not allowed_file(file.filename):
                    errors.append(f"{file.filename}: File type not allowed")
                    continue
                    
                try:
                    processed_count += 1
                    print(f"DEBUG: Processing file {processed_count}/{len(files)}: {file.filename}")
                    
                    if file.filename.lower().endswith('.pdf'):
                        content = process_pdf_file(file)
                        if content is None:
                            errors.append(f"{file.filename}: Could not extract text from PDF")
                            continue
                    else:
                        content = process_text_file(file)
                        if content is None:
                            errors.append(f"{file.filename}: Could not process text file")
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

            # Save progress after each batch
            print(f"DEBUG: Saving progress after batch {i//batch_size + 1}")
            rag_system.save_to_disk("rag_data")

        # Final save and session reset
        session['conversation_history'] = []
        
        print(f"DEBUG: Upload complete. Successfully processed {len(uploaded_files)} files with {len(errors)} errors")
        
        return jsonify({
            'uploaded': uploaded_files,
            'errors': errors,
            'total_documents': len(rag_system.documents),
            'total_chunks': len(rag_system.chunks),
            'processed_count': processed_count
        }), 200
        
    except Exception as e:
        print(f"DEBUG: General exception in upload_multiple: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


def process_pdf_file(file):
    """Process PDF file and extract text content"""
    try:
        print(f"DEBUG: Processing PDF file: {file.filename}")
        file.seek(0)
        pdf_reader = pypdf.PdfReader(file)
        content = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text() or ''
                if page_text.strip():
                    content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            except Exception as page_error:
                print(f"DEBUG: Error extracting text from page {page_num + 1}: {page_error}")
                continue

        # Optional: Table detection (comment out if you want faster processing)
        # table_info = extract_tables_from_pdf(file)
        
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


def extract_tables_from_pdf(file):
    """Optional: Extract tables from PDF (can be disabled for faster processing)"""
    try:
        file.seek(0)
        tmp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        tmp_pdf.write(file.read())
        tmp_pdf.flush()

        # Table detection via tabula
        try:
            tables_tabula = tabula.read_pdf(tmp_pdf.name, pages='all', multiple_tables=True)
            num_tabula = len(tables_tabula)
        except Exception as e:
            print(f"DEBUG: Tabula failed: {e}")
            num_tabula = 0

        # Table detection via pdfplumber
        try:
            pdf_plumb = pdfplumber.open(tmp_pdf.name)
            tables_plumber = []
            for pg in pdf_plumb.pages:
                extracted = pg.extract_tables()
                if extracted:
                    tables_plumber.extend(extracted)
            num_plumber = len(tables_plumber)
            pdf_plumb.close()
        except Exception as e:
            print(f"DEBUG: pdfplumber failed: {e}")
            num_plumber = 0

        # Clean up temp file
        os.unlink(tmp_pdf.name)
        
        return {
            'tabula_count': num_tabula,
            'pdfplumber_count': num_plumber
        }
        
    except Exception as e:
        print(f"DEBUG: Table extraction failed: {e}")
        return {'tabula_count': 0, 'pdfplumber_count': 0}


@app.route('/upload_from_folder', methods=['POST'])
def upload_from_folder():
    """NEW: Upload all files from a specified folder path"""
    data = request.json
    folder_path = data.get('folder_path', '')
    
    if not folder_path or not os.path.exists(folder_path):
        return jsonify({'error': 'Invalid folder path'}), 400
    
    try:
        uploaded_files = []
        errors = []
        
        # Get all allowed files from folder
        for filename in os.listdir(folder_path):
            if allowed_file(filename):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'rb') as f:
                        if filename.lower().endswith('.pdf'):
                            # Handle PDF
                            pdf_reader = pypdf.PdfReader(f)
                            content = ""
                            for page_num, page in enumerate(pdf_reader.pages):
                                try:
                                    page_text = page.extract_text() or ''
                                    if page_text.strip():
                                        content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                                except Exception:
                                    continue
                        else:
                            # Handle text files
                            raw_content = f.read()
                            try:
                                content = raw_content.decode('utf-8')
                            except UnicodeDecodeError:
                                content = raw_content.decode('utf-8', errors='ignore')
                    
                    content = clean_text(content)
                    if content.strip():
                        doc_id = str(uuid.uuid4())
                        num_chunks = rag_system.add_document(doc_id, filename, content)
                        uploaded_files.append({
                            'id': doc_id,
                            'filename': filename,
                            'size': len(content),
                            'chunks': num_chunks
                        })
                        print(f"DEBUG: Added {filename} from folder")
                    
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")
        
        rag_system.save_to_disk("rag_data")
        session['conversation_history'] = []
        
        return jsonify({
            'uploaded': uploaded_files,
            'errors': errors,
            'total_documents': len(rag_system.documents),
            'total_chunks': len(rag_system.chunks)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Folder upload failed: {str(e)}'}), 500


@app.route('/upload_progress', methods=['GET'])
def upload_progress():
    """NEW: Get upload progress (can be enhanced with WebSocket for real-time updates)"""
    return jsonify({
        'total_documents': len(rag_system.documents),
        'total_chunks': len(rag_system.chunks),
        'status': 'ready'
    })

# Replace your existing /chat route with this enhanced version
@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with intelligent question routing and dynamic chunk allocation"""
    data = request.json
    user_input = data.get('message', '')
    if not user_input.strip():
        return jsonify({'error': 'Please enter a message'}), 400
    
    # Get search parameters
    search_mode = data.get('search_mode', 'enhanced')
    max_chunks = data.get('max_chunks', None)
    is_simple_question = data.get('is_simple_question', False)
    
    # Initialize router
    router = QuestionRouter(rag_system)
    
    # Route the question
    route_type, doc_ids, metadata = router.route_question(user_input)
    
    # IMPROVED: Dynamic chunk allocation based on question type and search scope
    if max_chunks is None:
        if route_type == 'country_specific':
            # Country-specific: smaller chunk count since we're searching within specific documents
            if search_mode == 'comprehensive':
                max_chunks = 12
            elif search_mode == 'enhanced':
                max_chunks = 8
            else:  # basic
                max_chunks = 5
                
        elif route_type == 'multi_country':
            # Multi-country: moderate chunk count
            if search_mode == 'comprehensive':
                max_chunks = 18
            elif search_mode == 'enhanced':
                max_chunks = 12
            else:  # basic
                max_chunks = 8
                
        elif route_type in ['comparative', 'general']:
            # Comparative/General: larger chunk count since we need broader coverage
            if search_mode == 'comprehensive':
                max_chunks = 25
            elif search_mode == 'enhanced':
                max_chunks = 18
            else:  # basic
                max_chunks = 12
        
        print(f"DEBUG: Allocated {max_chunks} chunks for {route_type} question in {search_mode} mode")
    
    conversation_history = session.get('conversation_history', [])
    
    try:
        # Perform search based on route type with appropriate parameters
        if route_type == 'country_specific' and doc_ids:
            # Search within specific country documents with focused chunk count
            rag_results = rag_system.enhanced_search_by_document(user_input, doc_ids, k=max_chunks)
            search_scope = f"Documents for {', '.join(metadata['mentioned_countries'])}"
            
        elif route_type == 'multi_country' and doc_ids:
            # Search within multiple specific country documents
            rag_results = rag_system.enhanced_search_by_document(user_input, doc_ids, k=max_chunks)
            search_scope = f"Documents for {', '.join(metadata['mentioned_countries'])}"
            
        elif route_type in ['country_specific', 'multi_country'] and not doc_ids:
            # FALLBACK: If no specific documents found but countries mentioned, search all with increased chunks
            print(f"DEBUG: No documents found for countries {metadata['mentioned_countries']}, falling back to full search")
            # Increase chunk count for fallback since we're searching all documents
            fallback_chunks = min(max_chunks * 2, 30)  # Double the chunks but cap at 30
            
            if search_mode in ['enhanced', 'comprehensive']:
                candidate_count = 50 if search_mode == 'comprehensive' else 35
                rag_results = rag_system.search_with_reranking(
                    user_input, k=candidate_count, final_k=fallback_chunks
                )
            else:
                rag_results = rag_system.search(user_input, k=fallback_chunks)
            
            search_scope = f"All documents (no specific documents found for {', '.join(metadata['mentioned_countries'])})"
            print(f"DEBUG: Fallback search using {fallback_chunks} chunks")
            
        else:
            # Use enhanced search for comparative/general questions with larger chunk counts
            if search_mode in ['enhanced', 'comprehensive']:
                # Use higher candidate counts for comparative/general questions
                candidate_count = 60 if search_mode == 'comprehensive' else 40
                rag_results = rag_system.search_with_reranking(
                    user_input, k=candidate_count, final_k=max_chunks
                )
            else:
                rag_results = rag_system.search(user_input, k=max_chunks)
            
            search_scope = "All available documents"
        
        # Debug logging
        doc_distribution = {}
        for result in rag_results:
            filename = result['metadata']['filename']
            doc_distribution[filename] = doc_distribution.get(filename, 0) + 1
        
        print(f"DEBUG: Question routed as '{route_type}'")
        print(f"DEBUG: Mentioned countries: {metadata['mentioned_countries']}")
        print(f"DEBUG: Doc IDs found: {doc_ids}")
        print(f"DEBUG: Search scope: {search_scope}")
        print(f"DEBUG: Used {max_chunks} target chunks, found {len(rag_results)} actual chunks from {len(doc_distribution)} documents")
        print(f"DEBUG: Document distribution: {doc_distribution}")
        
        conversation_history.append(f"Human: {user_input}")
        
        if rag_results:
            ai_response = generate_response_with_rag(
                user_input, conversation_history, rag_results, 
                search_mode, is_simple_question
            )
        else:
            ai_response = "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded documents related to your query."
        
        conversation_history.append(f"AI: {ai_response}")
        session['conversation_history'] = conversation_history

        # Prepare sources
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
            'full_history': conversation_history,
            'chunks_used': len(rag_results),
            'documents_referenced': len(sources),
            'search_mode': search_mode,
            'is_simple_question': is_simple_question,
            'route_info': {
                'route_type': route_type,
                'mentioned_countries': metadata['mentioned_countries'],
                'search_scope': search_scope,
                'documents_covered': len(doc_distribution),
                'doc_ids_found': doc_ids,
                'target_chunks': max_chunks,
                'actual_chunks': len(rag_results)
            }
        })
        
    except Exception as e:
        print(f"DEBUG: Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
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

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint for testing RAG search"""
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 5)
    if not query.strip():
        return jsonify({'error': 'Please provide a search query'}), 400

    results = rag_system.search(query, k)
    return jsonify({
        'query': query,
        'results': [
            {
                'chunk': r['chunk'][:200] + '...' if len(r['chunk']) > 200 else r['chunk'],
                'filename': r['metadata']['filename'],
                'chunk_index': r['metadata']['chunk_index'],
                'distance': r['distance']
            }
            for r in results
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)
