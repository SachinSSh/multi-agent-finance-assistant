# finance-assistant/agents/retriever_agent.py
"""
Retriever Agent - RAG & Embeddings
Handles document indexing, similarity search, and retrieval augmented generation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from dataclasses import dataclass
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict
    embedding: np.ndarray = None
    source_document: str = ""
    chunk_id: str = ""

@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: Dict
    source: str

class RetrieverAgent:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 512):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the knowledge base"""
        try:
            for doc in documents:
                # Chunk the document
                chunks = self._chunk_document(doc['content'], doc.get('metadata', {}), doc.get('source', ''))
                self.chunks.extend(chunks)
            
            # Generate embeddings
            self._generate_embeddings()
            
            # Build FAISS index
            self._build_faiss_index()
            
            # Build TF-IDF matrix for hybrid search
            self._build_tfidf_matrix()
            
            self.logger.info(f"Added {len(documents)} documents, created {len(self.chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
    
    def _chunk_document(self, content: str, metadata: Dict, source: str) -> List[DocumentChunk]:
        """Split document into chunks for better retrieval"""
        # Clean content
        content = self._clean_text(content)
        
        # Split into sentences
        sentences = sent_tokenize(content)
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    metadata={**metadata, 'sentence_count': len(current_sentences)},
                    source_document=source,
                    chunk_id=f"{source}_{len(chunks)}"
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = sentence
                current_sentences = [sentence]
            else:
                current_chunk += " " + sentence
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata={**metadata, 'sentence_count': len(current_sentences)},
                source_document=source,
                chunk_id=f"{source}_{len(chunks)}"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        return text.strip()
    
    def _generate_embeddings(self) -> None:
        """Generate embeddings for all chunks"""
        try:
            contents = [chunk.content for chunk in self.chunks]
            embeddings = self.model.encode(contents, show_progress_bar=True)
            
            for i, chunk in enumerate(self.chunks):
                chunk.embedding = embeddings[i]
            
            self.embeddings = np.array(embeddings)
            self.logger.info(f"Generated embeddings for {len(self.chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index for fast similarity search"""
        try:
            if self.embeddings is None:
                raise ValueError("Embeddings not generated yet")
            
            # Create FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            
            # Add embeddings to index
            self.index.add(self.embeddings)
            
            self.logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            self.logger.error(f"Error building FAISS index: {e}")
            raise
    
    def _build_tfidf_matrix(self) -> None:
        """Build TF-IDF matrix for keyword-based search"""
        try:
            contents = [chunk.content for chunk in self.chunks]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
            self.logger.info("Built TF-IDF matrix for hybrid search")
        except Exception as e:
            self.logger.error(f"Error building TF-IDF matrix: {e}")
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Perform semantic search using embeddings"""
        try:
            if self.index is None:
                raise ValueError("Index not built yet. Add documents first.")
            
            # Generate query embedding
            query_embedding = self.model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    result = RetrievalResult(
                        content=chunk.content,
                        score=float(score),
                        metadata=chunk.metadata,
                        source=chunk.source_document
                    )
                    results.append(result)
            
            return results
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Perform keyword-based search using TF-IDF"""
        try:
            if self.tfidf_matrix is None:
                raise ValueError("TF-IDF matrix not built yet")
            
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only return results with positive similarity
                    chunk = self.chunks[idx]
                    result = RetrievalResult(
                        content=chunk.content,
                        score=float(similarities[idx]),
                        metadata=chunk.metadata,
                        source=chunk.source_document
                    )
                    results.append(result)
            
            return results
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 5, 
                     semantic_weight: float = 0.7) -> List[RetrievalResult]:
        """Combine semantic and keyword search results"""
        try:
            # Get results from both methods
            semantic_results = self.semantic_search(query, top_k * 2)
            keyword_results = self.keyword_search(query, top_k * 2)
            
            # Combine and re-rank
            combined_results = {}
            
            # Add semantic results
            for result in semantic_results:
                key = result.source + "_" + result.content[:50]
                combined_results[key] = {
                    'result': result,
                    'semantic_score': result.score,
                    'keyword_score': 0.0
                }
            
            # Add keyword results
            for result in keyword_results:
                key = result.source + "_" + result.content[:50]
                if key in combined_results:
                    combined_results[key]['keyword_score'] = result.score
                else:
                    combined_results[key] = {
                        'result': result,
                        'semantic_score': 0.0,
                        'keyword_score': result.score
                    }
            
            # Calculate hybrid scores
            final_results = []
            for item in combined_results.values():
                hybrid_score = (semantic_weight * item['semantic_score'] + 
                              (1 - semantic_weight) * item['keyword_score'])
                
                result = item['result']
                result.score = hybrid_score
                final_results.append(result)
            
            # Sort by hybrid score and return top_k
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:top_k]
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return []
    
    def get_relevant_context(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context for RAG"""
        try:
            results = self.hybrid_search(query, top_k=10)
            
            context_parts = []
            current_length = 0
            
            for result in results:
                content_length = len(result.content)
                if current_length + content_length <= max_context_length:
                    context_parts.append(f"Source: {result.source}\n{result.content}")
                    current_length += content_length + len(f"Source: {result.source}\n")
                else:
                    break
            
            return "\n\n---\n\n".join(context_parts)
        except Exception as e:
            self.logger.error(f"Error getting relevant context: {e}")
            return ""
    
    def save_index(self, filepath: str) -> None:
        """Save the index and chunks to disk"""
        try:
            data = {
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Save FAISS index separately
            if self.index:
                faiss.write_index(self.index, f"{filepath}.faiss")
            
            self.logger.info(f"Saved index to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    def load_index(self, filepath: str) -> None:
        """Load the index and chunks from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            self.tfidf_vectorizer = data['tfidf_vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            
            # Load FAISS index
            try:
                self.index = faiss.read_index(f"{filepath}.faiss")
            except:
                self.logger.warning("FAISS index file not found, rebuilding...")
                self._build_faiss_index()
            
            self.logger.info(f"Loaded index from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading index: {e}")

