import argparse
import pandas as pd
from src.retrieval.rag_system import RAGSystem
from src.evaluation.metrics import RAGEvaluator
from src.models.embeddings import EmbeddingGenerator

def display_metrics(answer, retrieved_papers, query):
    """
    Display evaluation metrics for the current query.
    
    Args:
        answer: Generated answer text
        retrieved_papers: DataFrame of retrieved papers
        query: User query
    """
    evaluator = RAGEvaluator()
    
    metrics = evaluator.evaluate_overall(query, answer, retrieved_papers)
    
    print("\nEvaluation Metrics:")
    print("------------------")
    
    print("Retrieval metrics:")
    print(f"  Retrieved papers: {len(retrieved_papers)}")
    print(f"  Top similarity score: {metrics['retrieval'].get('top_similarity', 0):.4f}")
    print(f"  Average similarity: {metrics['retrieval'].get('avg_similarity', 0):.4f}")
    
    print("\nAnswer quality metrics:")
    
    citations = evaluator.extract_citations(answer)
    print(f"  Number of citations: {metrics['answer'].get('num_citations', 0)}")
    print(f"  Citation ratio: {metrics['answer'].get('citation_ratio', 0):.4f}")
    
    if citations:
        print("\nCitations found:")
        for i, citation in enumerate(citations):
            print(f"  {i+1}. [{citation['authors']}, \"{citation['title']}\", {citation['year']}]")
    
    print(f"\n  Answer length: {metrics['answer'].get('answer_length', 0)} words")
    
    if len(citations) > 0:
        print(f"  Valid citation ratio: {metrics['answer'].get('valid_citation_ratio', 0):.4f}")
    
    print()

def main():
    """Main function for the CLI interface."""
    parser = argparse.ArgumentParser(description="Academic Paper RAG System")
    parser.add_argument("--vector-db", type=str, default="data/vector_db", help="Path to vector database")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--llm-provider", type=str, default="ollama", help="LLM provider (openai, anthropic, huggingface, ollama, local)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for LLM provider")
    parser.add_argument("--ollama-model", type=str, default="llama2", help="Model to use with Ollama")
    parser.add_argument("--k", type=int, default=5, help="Number of papers to retrieve")
    parser.add_argument("--evaluate", action="store_true", help="Enable real-time evaluation")
    
    args = parser.parse_args()
    
    rag_system = RAGSystem(
        vector_db_path=args.vector_db,
        embedding_model_name=args.embedding_model,
        llm_provider=args.llm_provider,
        api_key=args.api_key,
        ollama_model=args.ollama_model
    )
    
    print("Academic Paper RAG System")
    print("------------------------")
    print(f"Using LLM provider: {args.llm_provider}")
    if args.llm_provider == "ollama":
        print(f"Using Ollama model: {args.ollama_model}")
    print("Type 'exit' to quit")
    print()
    
    while True:
        query = input("Enter your question: ")
        
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        print("\nRetrieving relevant papers and generating answer...")
        
        answer, retrieved_papers = rag_system.answer_question(query, k=args.k)
        
        print("\nAnswer:")
        print("-------")
        print(answer)
        print()
        
        if args.evaluate:
            display_metrics(answer, retrieved_papers, query)
        
        show_papers = input("Do you want to see the retrieved papers? (y/n): ")
        
        if show_papers.lower() in ["y", "yes"]:
            print("\nRetrieved Papers:")
            print("----------------")
            
            for i, paper in retrieved_papers.iterrows():
                print(f"[{i+1}] {paper['title']}")
                print(f"    Authors: {paper['authors']}")
                print(f"    Year: {paper.get('date', 'N/A')[:4] if isinstance(paper.get('date'), str) else 'N/A'}")
                print(f"    Similarity: {paper['similarity']:.4f}")
                print()
        
        save_results = input("Do you want to save this result? (y/n): ")
        if save_results.lower() in ["y", "yes"]:
            filename = input("Enter filename to save results (default: results.txt): ") or "results.txt"
            
            with open(filename, 'a') as f:
                f.write(f"Question: {query}\n\n")
                f.write(f"Answer:\n{answer}\n\n")
                
                f.write("Retrieved Papers:\n")
                for i, paper in retrieved_papers.iterrows():
                    f.write(f"[{i+1}] {paper['title']}\n")
                    f.write(f"    Authors: {paper['authors']}\n")
                    f.write(f"    Year: {paper.get('date', 'N/A')[:4] if isinstance(paper.get('date'), str) else 'N/A'}\n")
                    f.write(f"    Similarity: {paper['similarity']:.4f}\n\n")
                
                if args.evaluate:
                    evaluator = RAGEvaluator()
                    metrics = evaluator.evaluate_overall(query, answer, retrieved_papers)
                    
                    f.write("Evaluation Metrics:\n")
                    f.write(f"Retrieved papers: {len(retrieved_papers)}\n")
                    f.write(f"Top similarity score: {metrics['retrieval'].get('top_similarity', 0):.4f}\n")
                    f.write(f"Average similarity: {metrics['retrieval'].get('avg_similarity', 0):.4f}\n")
                    f.write(f"Number of citations: {metrics['answer'].get('num_citations', 0)}\n")
                    f.write(f"Citation ratio: {metrics['answer'].get('citation_ratio', 0):.4f}\n")
                    f.write(f"Answer length: {metrics['answer'].get('answer_length', 0)} words\n")
                    if metrics['answer'].get('num_citations', 0) > 0:
                        f.write(f"Valid citation ratio: {metrics['answer'].get('valid_citation_ratio', 0):.4f}\n")
                
                f.write("\n" + "-"*50 + "\n\n")
            
            print(f"Results saved to {filename}")
        
        print()

if __name__ == "__main__":
    main()