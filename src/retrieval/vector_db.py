import os
import numpy as np
import pandas as pd
import faiss
import pickle
from tqdm import tqdm

class VectorDatabase:
    def __init__(self, embedding_dim=384):
        """Initialize an empty vector database with specified embedding dimension."""
        self.embedding_dim = embedding_dim
        self.index = None
        self.papers_df = None
        self.paper_ids = None
        
    def build_index(self, embeddings, papers_df, paper_id_column="id"):
        """
        Build a FAISS index from embeddings and store paper information.
        
        Args:
            embeddings: numpy.ndarray of embeddings
            papers_df: DataFrame containing paper information
            paper_id_column: Column name for paper IDs
        """
        if len(embeddings) != len(papers_df):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) doesn't match number of papers ({len(papers_df)})")
        
        embeddings = embeddings.astype(np.float32)
        
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.index.add(embeddings)
        
        self.papers_df = papers_df.copy()
        self.paper_ids = papers_df[paper_id_column].tolist()
        
        print(f"Built index with {self.index.ntotal} vectors of dimension {self.embedding_dim}")
    
    def search(self, query_embedding, k=5):
        """
        Search for papers similar to the query embedding.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            DataFrame with search results and scores
        """
        if self.index is None:
            raise ValueError("Index is not built yet")
        
        query_embedding = query_embedding.reshape(1, self.embedding_dim).astype(np.float32)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            similarity = 1.0 / (1.0 + distance)
            
            paper_info = self.papers_df.iloc[idx].to_dict()
            paper_info['similarity'] = float(similarity)
            paper_info['rank'] = i + 1
            
            results.append(paper_info)
        
        return pd.DataFrame(results)
    
    def save(self, folder_path):
        """
        Save the vector database to disk.
        
        Args:
            folder_path: Folder to save the database
        """
        os.makedirs(folder_path, exist_ok=True)
        
        index_path = os.path.join(folder_path, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        papers_path = os.path.join(folder_path, "papers.csv")
        self.papers_df.to_csv(papers_path, index=False)
        
        metadata = {
            "embedding_dim": self.embedding_dim,
            "num_papers": len(self.papers_df),
            "paper_ids": self.paper_ids
        }
        metadata_path = os.path.join(folder_path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Saved vector database to {folder_path}")
    
    def load(self, folder_path):
        """
        Load the vector database from disk.
        
        Args:
            folder_path: Folder containing the saved database
        """
        index_path = os.path.join(folder_path, "faiss_index.bin")
        self.index = faiss.read_index(index_path)
        
        papers_path = os.path.join(folder_path, "papers.csv")
        self.papers_df = pd.read_csv(papers_path)
        
        metadata_path = os.path.join(folder_path, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.embedding_dim = metadata["embedding_dim"]
        self.paper_ids = metadata["paper_ids"]
        
        print(f"Loaded vector database with {self.index.ntotal} papers")

