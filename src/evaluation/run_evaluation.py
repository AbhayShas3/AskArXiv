# src/evaluation/run_evaluation.py
"""
Script to run evaluation of the RAG system.
"""

import argparse
import pandas as pd
import json
from src.retrieval.rag_system import RAGSystem
from src.evaluation.metrics import RAGEvaluator, BENCHMARK_QUERIES
from src.models.embeddings import EmbeddingGenerator

def evaluate_on_benchmark(rag_system, evaluator, benchmark_queries, output_file=None):
    """
    Evaluate the RAG system on benchmark queries.
    
    Args:
        rag_system: Initialized RAG system
        evaluator: RAG evaluator
        benchmark_queries: List of benchmark queries
        output_file: Optional file to save results
        
    Returns:
        Dictionary of evaluation results
    """
    results = []
    
    for query_data in benchmark_queries:
        query = query_data['query']
        relevant_paper_ids = query_data.get('relevant_paper_ids')
        ground_truth = query_data.get('ground_truth')
        
        print(f"Evaluating query: {query}")
        
        answer, retrieved_papers = rag_system.answer_question(query, k=10)
        
        metrics = evaluator.evaluate_overall(
            query, 
            answer, 
            retrieved_papers, 
            relevant_paper_ids, 
            ground_truth
        )
        
        result = {
            'query': query,
            'answer': answer,
            'metrics': metrics
        }
        
        results.append(result)
    
    avg_metrics = {
        'retrieval': {},
        'answer': {}
    }
    
    for metric in results[0]['metrics']['retrieval']:
        values = [r['metrics']['retrieval'].get(metric, 0) for r in results]
        avg_metrics['retrieval'][metric] = sum(values) / len(values)
    
    for metric in results[0]['metrics']['answer']:
        values = [r['metrics']['answer'].get(metric, 0) for r in results]
        avg_metrics['answer'][metric] = sum(values) / len(values)
    
    final_results = {
        'queries': results,
        'average_metrics': avg_metrics
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
    
    return final_results

def print_evaluation_summary(results):
    """
    Print a summary of evaluation results.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n=== RAG System Evaluation Summary ===\n")
    
    avg_metrics = results['average_metrics']
    
    print("Retrieval Metrics:")
    print("-----------------")
    for metric, value in avg_metrics['retrieval'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAnswer Metrics:")
    print("--------------")
    for metric, value in avg_metrics['answer'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nIndividual Query Results:")
    print("-------------------------")
    for i, query_result in enumerate(results['queries']):
        print(f"\nQuery {i+1}: {query_result['query']}")
        print(f"Precision@5: {query_result['metrics']['retrieval'].get('precision@5', 0):.4f}")
        print(f"MRR: {query_result['metrics']['retrieval'].get('mrr', 0):.4f}")
        print(f"Valid Citation Ratio: {query_result['metrics']['answer'].get('valid_citation_ratio', 0):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--vector-db", type=str, default="data/vector_db", help="Path to vector database")
    parser.add_argument("--llm-provider", type=str, default="ollama", help="LLM provider")
    parser.add_argument("--ollama-model", type=str, default="llama2", help="Ollama model name")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    embedding_model = EmbeddingGenerator()
    
    rag_system = RAGSystem(
        vector_db_path=args.vector_db,
        llm_provider=args.llm_provider,
        ollama_model=args.ollama_model
    )
    
    evaluator = RAGEvaluator(embedding_model=embedding_model)
    
    results = evaluate_on_benchmark(
        rag_system, 
        evaluator, 
        BENCHMARK_QUERIES,
        args.output
    )
    
    print_evaluation_summary(results)

if __name__ == "__main__":
    main()