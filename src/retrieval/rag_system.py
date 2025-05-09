import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import requests
from sentence_transformers import SentenceTransformer
from src.retrieval.vector_db import VectorDatabase
from src.models.embeddings import EmbeddingGenerator


LLM_PROVIDER = "ollama"  
class RAGSystem:
    def __init__(
        self, 
        vector_db_path: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_provider: str = LLM_PROVIDER,
        api_key: str = None,
        ollama_model: str = "llama2"  
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_db_path: Path to the vector database
            embedding_model_name: Name of the embedding model
            llm_provider: LLM provider to use
            api_key: API key for the LLM provider
            ollama_model: Model to use with Ollama
        """
        self.vector_db = VectorDatabase()
        self.vector_db.load(vector_db_path)
        
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
        
        self.llm_provider = llm_provider
        self.api_key = api_key or os.environ.get(f"{llm_provider.upper()}_API_KEY")
        self.ollama_model = ollama_model
        
        if self.llm_provider not in ["openai", "anthropic", "huggingface", "ollama", "local"]:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        
        self.setup_prompt_template()
    
    def setup_prompt_template(self):
        base_system_prompt = """
        You are a helpful research assistant with expertise in academic papers. Your task is to answer questions based ONLY on the papers provided to you.

        Important guidelines:
        1. ONLY use information from the provided papers.
        2. For EVERY piece of information you provide, cite the source using the format: [Authors, "Title", Year]
        3. DO NOT reference papers that weren't provided to you.
        4. If the provided papers don't contain enough information to answer the question, say so clearly.
        5. Be precise and accurate in your citations.
        6. Prioritize papers with higher similarity scores in your answer.
        7. Only include relevant information to answer the question.
        """
        
        if self.llm_provider == "ollama":
            self.system_prompt = f"""<s>[INST]<<SYS>>
{base_system_prompt}
<</SYS>>
"""
            self.prompt_suffix = "[/INST]"
        else:
            self.system_prompt = base_system_prompt
            self.prompt_suffix = ""
        
        self.user_prompt_template = """
        Question: {question}
        
        Relevant Papers:
        {paper_context}
        
        Please answer the question using ONLY the information from these papers, with proper citations.{suffix}
        """
    
    def format_papers_for_context(self, papers: pd.DataFrame) -> str:
        """
        Format retrieved papers into context for the prompt.
        
        Args:
            papers: DataFrame of retrieved papers
            
        Returns:
            str: Formatted context string
        """
        context = ""
        for i, paper in papers.iterrows():
            context += f"[{i+1}] Title: {paper['title']}\n"
            context += f"    Authors: {paper['authors']}\n"
            
            year = "N/A"
            if 'date' in paper and paper['date']:
                if isinstance(paper['date'], str):
                    if len(paper['date']) >= 4 and paper['date'][:4].isdigit():
                        year = paper['date'][:4]
                    else:
                        year = paper['date']
            
            context += f"    Year: {year}\n"
            context += f"    Abstract: {paper['abstract']}\n"
            context += f"    Similarity Score: {paper['similarity']:.4f}\n\n"
        return context
    
    def retrieve(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Retrieve relevant papers for a query.
        
        Args:
            query: Query string
            k: Number of papers to retrieve
            
        Returns:
            DataFrame of retrieved papers
        """
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        
        results = self.vector_db.search(query_embedding, k=k)
        
        return results
    
    def generate_answer(self, query: str, papers: pd.DataFrame) -> str:
        """
        Generate an answer based on retrieved papers.
        
        Args:
            query: Query string
            papers: DataFrame of retrieved papers
            
        Returns:
            str: Generated answer
        """
        paper_context = self.format_papers_for_context(papers)
        
        user_prompt = self.user_prompt_template.format(
            question=query,
            paper_context=paper_context,
            suffix=self.prompt_suffix
        )
        
        if self.llm_provider == "openai":
            return self._generate_with_openai(user_prompt)
        elif self.llm_provider == "anthropic":
            return self._generate_with_anthropic(user_prompt)
        elif self.llm_provider == "huggingface":
            return self._generate_with_huggingface(user_prompt)
        elif self.llm_provider == "ollama":
            return self._generate_with_ollama(user_prompt)
        elif self.llm_provider == "local":
            return self._generate_with_local_model(user_prompt)
    
    def _generate_with_openai(self, prompt: str) -> str:
        """Generate answer using OpenAI's API."""
        import openai
        openai.api_key = self.api_key
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return f"Error generating response: {e}"
    
    def _generate_with_anthropic(self, prompt: str) -> str:
        """Generate answer using Anthropic's API."""
        import anthropic
        
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-2",
                max_tokens=1000,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content
        except Exception as e:
            print(f"Error with Anthropic API: {e}")
            return f"Error generating response: {e}"
    
    def _generate_with_huggingface(self, prompt: str) -> str:
        """Generate answer using Hugging Face API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            full_prompt = f"{self.system_prompt}\n\n{prompt}"
            
            response = requests.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers=headers,
                json={"inputs": full_prompt, "parameters": {"max_new_tokens": 1000}}
            )
            
            return response.json()[0]["generated_text"]
        except Exception as e:
            print(f"Error with Hugging Face API: {e}")
            return f"Error generating response: {e}"
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate answer using local Ollama instance."""
        try:
            full_prompt = f"{self.system_prompt}{prompt}"
            
            print("Calling Ollama API...")
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": full_prompt,
                    "options": {"temperature": 0.3}
                }
            )
            
            if response.status_code == 200:
                response_text = ""
                for line in response.text.strip().split('\n'):
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            response_text += data["response"]
                    except json.JSONDecodeError:
                        continue
                
                return response_text if response_text else "No valid response generated"
            else:
                print(f"Error with Ollama API: Status {response.status_code}")
                return f"Error generating response: Status {response.status_code}, {response.text}"
                
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return f"Error generating response: {e}"
    
    def _generate_with_local_model(self, prompt: str) -> str:
        """
        Generate an answer using a local model, or a simulated response.
        This is a fallback when no API key is available.
        """
        try:
            return self._generate_simulated_response(prompt)
        except Exception as e:
            print(f"Error generating local response: {e}")
            return f"Error generating response: {e}"
    
    def _generate_simulated_response(self, prompt: str) -> str:
        """Generate a simulated response for testing purposes."""
        question_line = [line for line in prompt.split("\n") if line.startswith("Question:")]
        question = question_line[0].replace("Question:", "").strip() if question_line else "the topic"
        
        paper_sections = prompt.split("[")
        papers = []
        
        for section in paper_sections[1:]:  
            try:
                lines = section.split("\n")
                paper_num = lines[0].split("]")[0].strip()
                title = lines[0].split("Title:")[1].strip() if "Title:" in lines[0] else "Unknown Title"
                authors = lines[1].split("Authors:")[1].strip() if len(lines) > 1 and "Authors:" in lines[1] else "Unknown Authors"
                year = lines[2].split("Year:")[1].strip() if len(lines) > 2 and "Year:" in lines[2] else "Unknown Year"
                
                papers.append({
                    "number": paper_num,
                    "title": title,
                    "authors": authors,
                    "year": year
                })
            except Exception:
                continue
        
        response = f"Based on the papers provided, I can address your question about {question}.\n\n"
        
        if not papers:
            return "I couldn't find relevant information in the provided papers to answer your question."
        
        for i, paper in enumerate(papers[:min(3, len(papers))]):
            author_last_name = paper["authors"].split(",")[0].split()[-1] if "," in paper["authors"] else paper["authors"].split()[0]
            
            if i == 0:
                response += f"According to {author_last_name} et al. [{paper['authors']}, \"{paper['title']}\", {paper['year']}], "
                response += f"there are several important aspects to consider about {question}. "
                response += f"Their research highlights the significance of understanding this topic in depth.\n\n"
            
            elif i == 1:
                response += f"Additionally, {author_last_name} et al. [{paper['authors']}, \"{paper['title']}\", {paper['year']}] "
                response += f"found complementary results that provide further insights. Their work demonstrates "
                response += f"how these concepts apply in various contexts.\n\n"
            
            elif i == 2:
                response += f"Furthermore, {author_last_name} et al. [{paper['authors']}, \"{paper['title']}\", {paper['year']}] "
                response += f"explored related aspects, showing that there are multiple perspectives on this topic.\n\n"
        
        response += "This information is based solely on the provided papers. For a more comprehensive understanding, additional research may be needed."
        
        return response
    
    def answer_question(self, query: str, k: int = 5) -> Tuple[str, pd.DataFrame]:
        """
        Answer a question using the RAG system.
        
        Args:
            query: Question to answer
            k: Number of papers to retrieve
            
        Returns:
            Tuple[str, pd.DataFrame]: Answer and retrieved papers
        """
        retrieved_papers = self.retrieve(query, k=k)
        
        if len(retrieved_papers) > 0:
            answer = self.generate_answer(query, retrieved_papers)
        else:
            answer = "I couldn't find any relevant papers to answer your question."
        
        return answer, retrieved_papers