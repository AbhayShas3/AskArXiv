import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the embedding generator with a specific model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Loaded {model_name} model with dimension {self.embedding_dim}")
        
        if torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda"))
            print("Using GPU for embedding generation")
        
    def generate_embeddings(self, texts, batch_size=32):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            numpy.ndarray: Matrix of embeddings
        """
        print(f"Generating {len(texts)} embeddings with batch size {batch_size}...")
        
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_papers(self, papers_df, text_column="combined_text", batch_size=32):
        """
        Generate embeddings for papers in a DataFrame.
        
        Args:
            papers_df: DataFrame containing papers
            text_column: Column name containing text to embed
            batch_size: Batch size for embedding generation
            
        Returns:
            numpy.ndarray: Matrix of embeddings
        """
        if text_column not in papers_df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = papers_df[text_column].tolist()
        return self.generate_embeddings(texts, batch_size=batch_size)
