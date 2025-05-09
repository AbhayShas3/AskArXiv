import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
import re
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
        """Set up the prompt template for the LLM with stronger citation requirements."""
        
        base_system_prompt = """
        You are a specialized academic research assistant with expertise in citing scholarly papers accurately. Your task is to answer questions based EXCLUSIVELY on the academic papers I provide to you.

        STRICT CITATION RULES:
        1. Every piece of information must include a citation using the format: [Authors, "Title", Year]
        2. Insert citations directly after the sentence or claim they support - NEVER wait until the end to cite
        3. Cite each paper at least once if it contains relevant information
        4. Papers are numbered [1], [2], etc. in the order presented - use these exact paper details in your citations
        5. NEVER refer to "Paper 1" or "Paper 2" in your text - use only proper citations like [Smith et al., "Paper Title", 2020]
        6. Do not invent or include information not found in the provided papers
        7. If the papers don't contain enough information to answer, state this clearly

        Format your answer as a cohesive academic response with integrated citations. This is CRITICAL for the user's research.
        """
        
        if self.llm_provider == "ollama":
            self.system_prompt = f"""<s>[INST]<<SYS>>
{base_system_prompt}

Example of CORRECT citation usage:
Question: What are the benefits of transformers in NLP?
Answer: Transformers have revolutionized natural language processing by enabling parallel computation [Vaswani et al., "Attention Is All You Need", 2017]. Further research has shown they excel at capturing long-range dependencies [Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2018].

Example of INCORRECT citation usage:
Question: What are the benefits of transformers in NLP?
Answer: Transformers have revolutionized natural language processing by enabling parallel computation. Further research has shown they excel at capturing long-range dependencies. As discussed in Paper 1 and Paper 2, these models have changed the field.
<</SYS>>
"""
            self.prompt_suffix = "[/INST]"
        else:
            self.system_prompt = base_system_prompt
            self.prompt_suffix = ""
        
        self.user_prompt_template = """
        Question: {question}
        
        Relevant Academic Papers:
        {paper_context}
        
        Remember: Your answer must use proper citations in the format [Authors, "Title", Year] immediately after each claim or information from a specific paper. Citation is MANDATORY for this task.{suffix}
        """
    
    def format_papers_for_context(self, papers: pd.DataFrame) -> str:
        """
        Format retrieved papers into context for the prompt with clearer numbering.
        
        Args:
            papers: DataFrame of retrieved papers
            
        Returns:
            str: Formatted context string
        """
        context = ""
        for i, paper in papers.iterrows():
            context += f"[{i+1}] -------------------------------------\n"
            context += f"Title: {paper['title']}\n"
            
            authors = paper['authors']
            if isinstance(authors, str) and len(authors) > 30:
                first_author = authors.split(',')[0].strip()
                authors = f"{first_author} et al."
            
            context += f"Authors: {authors}\n"
            
            year = "N/A"
            if 'date' in paper and paper['date']:
                if isinstance(paper['date'], str):
                    if len(paper['date']) >= 4 and paper['date'][:4].isdigit():
                        year = paper['date'][:4]
                    else:
                        year = paper['date']
            
            context += f"Year: {year}\n"
            context += f"Abstract: {paper['abstract']}\n"
            context += f"Similarity Score: {paper['similarity']:.4f}\n\n"
        
        context += "CITATION REMINDER: In your answer, you MUST cite these papers in the format [Authors, \"Title\", Year] each time you use information from them.\n\n"

        if len(papers) > 0:
            paper = papers.iloc[0]
            authors = paper['authors']
            if isinstance(authors, str) and len(authors) > 30:
                first_author = authors.split(',')[0].strip()
                authors = f"{first_author} et al."
            
            year = "N/A"
            if 'date' in paper and isinstance(paper['date'], str) and len(paper['date']) >= 4:
                year = paper['date'][:4]
            
            context += f"""
Format Example:
When citing paper [1] titled "{paper['title']}" by {authors} from {year}, you would write:
"This sentence contains information from the paper [{authors}, "{paper['title']}", {year}]."

NOT: "As discussed in Paper 1..." or "According to the first paper..."
"""
        
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
                f"http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": full_prompt,
                    "temperature": 0.2,  
                    "system": self.system_prompt.strip() if not self.system_prompt.strip().startswith("<s>") else ""
                },
                stream=False
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
                title_line = [l for l in lines if "Title:" in l]
                title = title_line[0].split("Title:")[1].strip() if title_line else "Unknown Title"
                
                authors_line = [l for l in lines if "Authors:" in l]
                authors = authors_line[0].split("Authors:")[1].strip() if authors_line else "Unknown Authors"
                
                year_line = [l for l in lines if "Year:" in l]
                year = year_line[0].split("Year:")[1].strip() if year_line else "Unknown Year"
                
                papers.append({
                    "number": paper_num,
                    "title": title,
                    "authors": authors,
                    "year": year
                })
            except Exception:
                continue
        
        response = f"Based on the retrieved papers, I can provide information about {question}.\n\n"
        
        if not papers:
            return "I couldn't find relevant information in the provided papers to answer your question."
        
        for i, paper in enumerate(papers[:min(3, len(papers))]):
            author_last_name = paper["authors"].split(",")[0].split()[-1] if "," in paper["authors"] else paper["authors"].split()[0]
            
            if i == 0:
                response += f"According to [{paper['authors']}, \"{paper['title']}\", {paper['year']}], "
                response += f"there are several important aspects to consider about {question}. "
                response += f"Their research highlights the significance of understanding this topic in depth.\n\n"
            
            elif i == 1:
                response += f"Additionally, research by [{paper['authors']}, \"{paper['title']}\", {paper['year']}] "
                response += f"found complementary results that provide further insights. Their work demonstrates "
                response += f"how these concepts apply in various contexts.\n\n"
            
            elif i == 2:
                response += f"Furthermore, the study by [{paper['authors']}, \"{paper['title']}\", {paper['year']}] "
                response += f"explored related aspects, showing that there are multiple perspectives on this topic.\n\n"
        
        response += "This information is based solely on the provided papers. For a more comprehensive understanding, additional research may be needed."
        
        return response
    
    def post_process_answer(self, answer: str, papers: pd.DataFrame) -> str:
        """
        Post-process the answer to ensure proper citations.
        
        Args:
            answer: Generated answer
            papers: DataFrame of papers used
            
        Returns:
            Processed answer with citation reminders if needed
        """
        citation_pattern = r'\[([^,]+),\s*"([^"]+)",\s*(\d{4})\]'
        citations = re.findall(citation_pattern, answer)
        
        if not citations:
            reminder = "\n\nNOTE: The information above should have included proper citations to the following papers:\n"
            for i, paper in papers.iterrows():
                authors = paper['authors']
                if isinstance(authors, str) and len(authors) > 30:
                    first_author = authors.split(',')[0].strip()
                    authors = f"{first_author} et al."
                
                year = "N/A"
                if 'date' in paper and isinstance(paper['date'], str) and len(paper['date']) >= 4:
                    year = paper['date'][:4]
                
                reminder += f"- [{authors}, \"{paper['title']}\", {year}]\n"
            
            return answer + reminder
        
        return answer
    
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
            
            answer = self.post_process_answer(answer, retrieved_papers)
        else:
            answer = "I couldn't find any relevant papers to answer your question."
        
        return answer, retrieved_papers