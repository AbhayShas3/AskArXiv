import os
import argparse
import pandas as pd
from src.models.embeddings import EmbeddingGenerator
from src.retrieval.vector_db import VectorDatabase

def build_vector_database(
    papers_path, 
    output_folder, 
    text_column="combined_text",
    model_name="all-MiniLM-L6-v2",
    batch_size=32
):
    """
    Build a vector database from processed papers.
    
    Args:
        papers_path: Path to the CSV file with processed papers
        output_folder: Folder to save the vector database
        text_column: Column containing text to embed
        model_name: Name of the sentence transformer model
        batch_size: Batch size for embedding generation
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Loading papers from {papers_path}...")
    papers_df = pd.read_csv(papers_path)
    print(f"Loaded {len(papers_df)} papers")
    
    embedding_generator = EmbeddingGenerator(model_name=model_name)
    embeddings = embedding_generator.embed_papers(
        papers_df, 
        text_column=text_column,
        batch_size=batch_size
    )
    
    vector_db = VectorDatabase(embedding_dim=embedding_generator.embedding_dim)
    vector_db.build_index(embeddings, papers_df)
    vector_db.save(output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vector database from processed papers")
    parser.add_argument("--input", type=str, required=True, help="Path to processed papers CSV")
    parser.add_argument("--output", type=str, required=True, help="Output folder for vector database")
    parser.add_argument("--text-column", type=str, default="combined_text", help="Column containing text to embed")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    
    args = parser.parse_args()
    
    build_vector_database(
        args.input,
        args.output,
        text_column=args.text_column,
        model_name=args.model,
        batch_size=args.batch_size
    )