# data_ingestion/embeddings.py
"""
Vector Embeddings and Semantic Search
Handles document vectorization and similarity search for financial documents
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from sentence_transformers import SentenceTransformer
import json
import pickle
from datetime import datetime
import hashlib
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document data structure for vector storage"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    timestamp: Optional[str] = None


class EmbeddingManager:
    """Manages document embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding manager
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model: {model_name} (dim: {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            # Fallback to a mock implementation for demonstration
            self.model = None
            self.model_name = model_name
            self.embedding_dim = 384
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding vector for text
        
        Args:
            text: Input text to embed
        
        Returns:
            Embedding vector as numpy array
        """
        try:
            if self.model:
                embedding = self.model.encode([text])[0]
                return embedding.astype(np.float32)
            else:
                # Mock embedding for demonstration
                np.random.seed(hash(text) % 2**32)
                return np.random.randn(self.embedding_dim).astype(np.float32)
                
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def create_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Array of embeddings
        """
        try:
            if self.model:
                embeddings = self.model.encode(texts)
                return embeddings.astype(np.float32)
            else:
                # Mock embeddings for demonstration
                embeddings = []
                for text in texts:
                    np.random.seed(hash(text) % 2**32)
                    embeddings.append(np.random.randn(self.embedding_dim))
                return np.array(embeddings, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {str(e)}")
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0


class VectorStore:
    """Vector database for storing and searching document embeddings"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize vector store
        
        Args:
            embedding_manager: EmbeddingManager instance
        """
        self.embedding_manager = embedding_manager
        self.documents: Dict[str, Document] = {}
        self.index = None
        self.doc_ids: List[str] = []
        
        # Initialize FAISS index
        try:
            self.index = faiss.IndexFlatIP(embedding_manager.embedding_dim)  # Inner product (cosine similarity)
            logger.info(f"Initialized FAISS index with dimension {embedding_manager.embedding_dim}")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {str(e)}")
            self.index = None
    
    def add_document(self, content: str, metadata: Dict[str, Any], doc_id: Optional[str] = None) -> str:
        """
        Add document to vector store
        
        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Optional document ID (generated if not provided)
        
        Returns:
            Document ID
        """
        try:
            # Generate document ID if not provided
            if doc_id is None:
                doc_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()
            
            # Create embedding
            embedding = self.embedding_manager.create_embedding(content)
            
            # Create document object
            document = Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                embedding=embedding,
                timestamp=datetime.now().isoformat()
            )
            
            # Store document
            self.documents[doc_id] = document
            
            # Add to FAISS index
            if self.index is not None:
                # Normalize embedding for cosine similarity
                normalized_embedding = embedding / np.linalg.norm(embedding)
                self.index.add(normalized_embedding.reshape(1, -1))
                self.doc_ids.append(doc_id)
            
            logger.info(f"Added document {doc_id} to vector store")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            raise
    
    def add_documents_batch(self, documents_data: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """
        Add multiple documents efficiently
        
        Args:
            documents_data: List of (content, metadata) tuples
        
        Returns:
            List of document IDs
        """
        try:
            doc_ids = []
            contents = [doc[0] for doc in documents_data]
            
            # Create embeddings in batch
            embeddings = self.embedding_manager.create_embeddings_batch(contents)
            
            for i, (content, metadata) in enumerate(documents_data):
                doc_id = hashlib.md5(f"{content}{datetime.now()}{i}".encode()).hexdigest()
                
                document = Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    embedding=embeddings[i],
                    timestamp=datetime.now().isoformat()
                )
                
                self.documents[doc_id] = document
                doc_ids.append(doc_id)
            
            # Add to FAISS index
            if self.index is not None and len(embeddings) > 0:
                # Normalize embeddings
                normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                self.index.add(normalized_embeddings)
                self.doc_ids.extend(doc_ids)
            
            logger.info(f"Added {len(doc_ids)} documents to vector store")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding batch documents: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Tuple[Document, float]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            # Create query embedding
            query_embedding = self.embedding_manager.create_embedding(query)
            
            if self.index is not None and len(self.doc_ids) > 0:
                # Use FAISS for efficient search
                normalized_query = query_embedding / np.linalg.norm(query_embedding)
                scores, indices = self.index.search(normalized_query.reshape(1, -1), min(top_k, len(self.doc_ids)))
                
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.doc_ids):
                        doc_id = self.doc_ids[idx]
                        document = self.documents[doc_id]
                        
                        # Apply metadata filtering if specified
                        if filter_metadata:
                            if not all(document.metadata.get(k) == v for k, v in filter_metadata.items()):
                                continue
                        
                        results.append((document, float(score)))
                
                return results[:top_k]
            
            else:
                # Fallback to brute force search
                results = []
                for doc_id, document in self.documents.items():
                    # Apply metadata filtering if specified
                    if filter_metadata:
                        if not all(document.metadata.get(k) == v for k, v in filter_metadata.items()):
                            continue
                    
                    similarity = self.embedding_manager.compute_similarity(query_embedding, document.embedding)
                    results.append((document, similarity))
                
                # Sort by similarity and return top_k
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]
                
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents"""
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document from vector store
        
        Args:
            doc_id: Document ID to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                
                # Remove from FAISS index (requires rebuilding index)
                if doc_id in self.doc_ids:
                    idx = self.doc_ids.index(doc_id)
                    self.doc_ids.pop(idx)
                    
                    # Rebuild FAISS index without the deleted document
                    if len(self.doc_ids) > 0:
                        embeddings = np.array([self.documents[did].embedding for did in self.doc_ids])
                        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                        
                        self.index = faiss.IndexFlatIP(self.embedding_manager.embedding_dim)
                        self.index.add(normalized_embeddings)
                    else:
                        self.index = faiss.IndexFlatIP(self.embedding_manager.embedding_dim)
                
                logger.info(f"Deleted document {doc_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False
    
    def save_to_disk(self, filepath: str):
        """
        Save vector store to disk
        
        Args:
            filepath: Path to save the vector store
        """
        try:
            data = {
                'documents': {doc_id: {
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'embedding': doc.embedding.tolist() if doc.embedding is not None else None,
                    'timestamp': doc.timestamp
                } for doc_id, doc in self.documents.items()},
                'doc_ids': self.doc_ids,
                'model_name': self.embedding_manager.model_name,
                'embedding_dim': self.embedding_manager.embedding_dim
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Save FAISS index separately
            if self.index is not None:
                faiss.write_index(self.index, f"{filepath}.faiss")
            
            logger.info(f"Saved vector store to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_from_disk(self, filepath: str):
        """
        Load vector store from disk
        
        Args:
            filepath: Path to load the vector store from
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Reconstruct documents
            self.documents = {}
            for doc_id, doc_data in data['documents'].items():
                embedding = np.array(doc_data['embedding'], dtype=np.float32) if doc_data['embedding'] else None
                document = Document(
                    id=doc_data['id'],
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    embedding=embedding,
                    timestamp=doc_data['timestamp']
                )
                self.documents[doc_id] = document
            
            self.doc_ids = data['doc_ids']
            
            # Load FAISS index
            try:
                self.index = faiss.read_index(f"{filepath}.faiss")
            except:
                # Rebuild index if loading fails
                if len(self.doc_ids) > 0:
                    embeddings = np.array([self.documents[doc_id].embedding for doc_id in self.doc_ids])
                    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    self.index = faiss.IndexFlatIP(self.embedding_manager.embedding_dim)
                    self.index.add(normalized_embeddings)
            
            logger.info(f"Loaded vector store from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embedding_manager.embedding_dim,
            'model_name': self.embedding_manager.model_name,
            'index_size': self.index.ntotal if self.index else 0,
            'memory_usage_mb': sum(doc.embedding.nbytes for doc in self.documents.values() if doc.embedding is not None) / (1024 * 1024)
        }


# Example usage and integration
if __name__ == "__main__":
    # Example usage of the data ingestion module
    
    # 1. Market Data Example
    print("=== Market Data Example ===")
    
    # Yahoo Finance
    yahoo_loader = YahooFinanceLoader()
    try:
        aapl_data = yahoo_loader.get_stock_data("AAPL", period="1mo")
        print(f"Loaded {len(aapl_data)} AAPL records")
        
        aapl_quote = yahoo_loader.get_real_time_quote("AAPL")
        print(f"AAPL Current Price: ${aapl_quote.get('price', 'N/A')}")
        
        aapl_info = yahoo_loader.get_company_info("AAPL")
        print(f"AAPL Market Cap: ${aapl_info.get('market_cap', 'N/A'):,}")
        
    except Exception as e:
        print(f"Yahoo Finance example failed: {e}")
    
    # Alpha Vantage (requires API key)
    # alpha_loader = AlphaVantageLoader("YOUR_API_KEY")
    # msft_data = alpha_loader.get_stock_data("MSFT")
    
    # 2. SEC Filing Example
    print("\n=== SEC Filing Example ===")
    
    sec_loader = SECFilingLoader()
    doc_processor = DocumentProcessor()
    
    try:
        # Search for recent Apple 10-K filings
        filings = sec_loader.search_filings("AAPL", form_types=["10-K"], limit=2)
        print(f"Found {len(filings)} Apple 10-K filings")
        
        if filings:
            # Download and process the most recent filing
            filing_content = sec_loader.download_filing(filings[0])
            clean_text = doc_processor.extract_text_from_filing(filing_content)
            
            # Extract sections
            sections = doc_processor.extract_financial_sections(clean_text)
            print(f"Extracted {len(sections)} sections from filing")
            
            # Extract metrics
            metrics = doc_processor.extract_financial_metrics(clean_text)
            print(f"Extracted metrics: {list(metrics.keys())}")
            
            # Create summary
            summary = doc_processor.summarize_document(clean_text, max_length=500)
            print(f"Document summary length: {len(summary)} characters")
            
    except Exception as e:
        print(f"SEC filing example failed: {e}")
    
    # 3. Vector Embeddings Example
    print("\n=== Vector Embeddings Example ===")
    
    try:
        # Initialize embedding system
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore(embedding_manager)
        
        # Sample financial documents
        documents = [
            ("Apple Inc. reported strong quarterly earnings with revenue growth of 15% year-over-year", 
             {"company": "AAPL", "type": "earnings", "quarter": "Q1"}),
            ("Microsoft Azure cloud services continue to drive significant revenue growth", 
             {"company": "MSFT", "type": "cloud", "segment": "Azure"}),
            ("Tesla's automotive segment showed improved margins despite supply chain challenges", 
             {"company": "TSLA", "type": "automotive", "topic": "margins"}),
            ("Amazon Web Services maintains market leadership in cloud infrastructure", 
             {"company": "AMZN", "type": "cloud", "segment": "AWS"}),
            ("Google's advertising revenue faced headwinds from economic uncertainty", 
             {"company": "GOOGL", "type": "advertising", "topic": "revenue"})
        ]
        
        # Add documents to vector store
        doc_ids = vector_store.add_documents_batch(documents)
        print(f"Added {len(doc_ids)} documents to vector store")
        
        # Search for similar documents
        query = "cloud computing revenue growth"
        results = vector_store.search(query, top_k=3)
        
        print(f"\nSearch results for '{query}':")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. Score: {score:.3f}")
            print(f"   Content: {doc.content[:100]}...")
            print(f"   Metadata: {doc.metadata}")
        
        # Filter search by metadata
        cloud_results = vector_store.search(query, top_k=5, filter_metadata={"type": "cloud"})
        print(f"\nFiltered search (cloud only): {len(cloud_results)} results")
        
        # Get statistics
        stats = vector_store.get_statistics()
        print(f"\nVector store statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Vector embeddings example failed: {e}")
    
    print("\n=== Data Ingestion Module Ready ===")
    print("Available classes:")
    print("- YahooFinanceLoader: Free market data")
    print("- AlphaVantageLoader: Premium market data (requires API key)")
    print("- SECFilingLoader: SEC EDGAR filings")
    print("- DocumentProcessor: Text processing and extraction")
    print("- EmbeddingManager: Document vectorization")
    print("- VectorStore: Semantic search and storage")
