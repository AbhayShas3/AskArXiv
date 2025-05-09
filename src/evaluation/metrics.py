# src/evaluation/metrics.py
"""
Evaluation metrics for the RAG system.
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class RAGEvaluator:
    def __init__(self, embedding_model=None):
        """
        Initialize the evaluator.
        
        Args:
            embedding_model: Optional embedding model for semantic similarity calculation
        """
        self.embedding_model = embedding_model
    
    def evaluate_retrieval(self, retrieved_papers, relevant_paper_ids=None, k_values=[1, 3, 5, 10]):
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_papers: DataFrame of retrieved papers
            relevant_paper_ids: List of IDs of known relevant papers (for test queries)
            k_values: List of k values for precision@k, recall@k
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if relevant_paper_ids is None:
            metrics['num_retrieved'] = len(retrieved_papers)
            metrics['top_similarity'] = retrieved_papers['similarity'].iloc[0] if len(retrieved_papers) > 0 else 0
            metrics['avg_similarity'] = retrieved_papers['similarity'].mean() if len(retrieved_papers) > 0 else 0
            return metrics
        
        for k in k_values:
            if k <= len(retrieved_papers):
                top_k_ids = retrieved_papers['id'].iloc[:k].tolist()
                relevant_in_top_k = sum(1 for paper_id in top_k_ids if paper_id in relevant_paper_ids)
                
                precision_k = relevant_in_top_k / k
                recall_k = relevant_in_top_k / len(relevant_paper_ids) if len(relevant_paper_ids) > 0 else 0
                
                metrics[f'precision@{k}'] = precision_k
                metrics[f'recall@{k}'] = recall_k
                
                if precision_k + recall_k > 0:
                    metrics[f'f1@{k}'] = 2 * (precision_k * recall_k) / (precision_k + recall_k)
                else:
                    metrics[f'f1@{k}'] = 0
        
        first_relevant_rank = None
        for i, paper_id in enumerate(retrieved_papers['id']):
            if paper_id in relevant_paper_ids:
                first_relevant_rank = i + 1
                break
        
        metrics['mrr'] = 1.0 / first_relevant_rank if first_relevant_rank else 0
        
        relevance_scores = [1.0 if paper_id in relevant_paper_ids else 0.0 
                            for paper_id in retrieved_papers['id']]
        
        dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
        
        metrics['ndcg'] = dcg / idcg if idcg > 0 else 0
        
        return metrics
    
    def extract_citations(self, answer):
        """
        Extract citations from the answer using proper academic format.
        
        Args:
            answer: Generated answer text
            
        Returns:
            List of citation dictionaries with author, title, year
        """
        citation_pattern = r'\[([^,]+),\s*"([^"]+)",\s*(\d{4})\]'
        raw_citations = re.findall(citation_pattern, answer)
        
        citations = []
        for authors, title, year in raw_citations:
            citations.append({
                "authors": authors.strip(),
                "title": title.strip(),
                "year": year.strip()
            })
        
        return citations
    
    def evaluate_answer(self, answer, query, retrieved_papers, ground_truth=None):
        """
        Evaluate the quality of the generated answer.
        
        Args:
            answer: Generated answer text
            query: User query
            retrieved_papers: DataFrame of retrieved papers used for generation
            ground_truth: Optional ground truth answer for the query
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        citations = self.extract_citations(answer)
        metrics['num_citations'] = len(citations)
        metrics['citation_ratio'] = len(citations) / len(retrieved_papers) if len(retrieved_papers) > 0 else 0
        
        metrics['answer_length'] = len(answer.split())
        
        if len(citations) > 0:
            valid_citations = 0
            for citation in citations:
                for i, paper in retrieved_papers.iterrows():
                    paper_title = paper['title'].lower()
                    citation_title = citation['title'].lower()
                    
                   
                    if (paper_title in citation_title or citation_title in paper_title or 
                        self._similarity(paper_title, citation_title) > 0.8):
                        valid_citations += 1
                        break
            
            metrics['valid_citation_ratio'] = valid_citations / len(citations)
        else:
            metrics['valid_citation_ratio'] = 0
        
        if ground_truth:
            try:
                from nltk.translate.bleu_score import sentence_bleu
                from nltk.tokenize import word_tokenize
                
                reference = [word_tokenize(ground_truth.lower())]
                candidate = word_tokenize(answer.lower())
                metrics['bleu_score'] = sentence_bleu(reference, candidate)
            except ImportError:
                metrics['bleu_score'] = 0
        
        if hasattr(self, 'embedding_model') and self.embedding_model:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                
                query_emb = self.embedding_model.encode([query])[0]
                answer_emb = self.embedding_model.encode([answer])[0]
                metrics['query_answer_similarity'] = cosine_similarity([query_emb], [answer_emb])[0][0]
            except:
                metrics['query_answer_similarity'] = 0
        
        return metrics

    def _similarity(self, str1, str2):
        """Simple string similarity function for citation matching"""
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def evaluate_overall(self, query, answer, retrieved_papers, relevant_paper_ids=None, ground_truth=None):
        """
        Evaluate the overall RAG system performance.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_papers: DataFrame of retrieved papers
            relevant_paper_ids: List of IDs of known relevant papers (optional)
            ground_truth: Ground truth answer (optional)
            
        Returns:
            Dictionary of all metrics
        """
        retrieval_metrics = self.evaluate_retrieval(retrieved_papers, relevant_paper_ids)
        answer_metrics = self.evaluate_answer(answer, query, retrieved_papers, ground_truth)
        
        metrics = {
            'retrieval': retrieval_metrics,
            'answer': answer_metrics
        }
        
        return metrics


BENCHMARK_QUERIES = [
    {
        'query': 'How do neural networks handle ordinal regression?',
        'relevant_paper_ids': ['0704.1028'],  
        'ground_truth': 'Neural networks handle ordinal regression by adapting traditional neural networks to learn ordinal categories. The NNRank method is a generalization of the perceptron method for ordinal regression that outperforms standard neural network classification methods.'
    },
    {
        'query': 'What techniques are used for Arabic speech recognition?',
        'relevant_paper_ids': ['0704.2083', '0704.2201'], 
        'ground_truth': 'Arabic speech recognition can be implemented using the CMU Sphinx-4 system, which is based on discrete Hidden Markov Models (HMMs). The system can be adapted to Arabic language by making specific changes to the acoustic model.'
    }
]