# AskArXiv


## Overview

The system implements a complete RAG pipeline:

- Vector Database: Stores embeddings of academic paper abstracts  
- Retrieval: Finds the most relevant papers based on semantic similarity  
- Generation: Creates answers with proper academic citations  
- Evaluation: Measures the quality of retrieval and answer generation  

## Features

- Semantic search across academic papers using FAISS vector database  
- Proper academic citations in the format `[Authors, "Title", Year]`  
- Evaluation metrics for retrieval accuracy and answer quality  
- Command-line interface for interactive querying  
- Support for multiple LLM providers (OpenAI, Anthropic, Hugging Face, Ollama)  

## Project Structure


```
rag-system/
├── data/                  # Data storage
│   ├── raw/               # Raw dataset files
│   ├── processed/         # Cleaned dataset files
│   └── vector_db/         # Vector database files
├── src/                   # Source code
│   ├── data/              # Data processing code
│   │   └── clean.py       # Dataset cleaning script
│   ├── models/            # Embedding models
│   │   └── embeddings.py  # Embedding generation code
│   ├── retrieval/         # Retrieval system
│   │   ├── vector_db.py   # Vector database operations
│   │   ├── build_db.py    # Database building script
│   │   └── rag_system.py  # RAG system implementation
│   ├── evaluation/        # Evaluation code
│   │   └── metrics.py     # Evaluation metrics
│   └── ui/                # User interfaces
│       └── cli.py         # Command-line interface
└── README.md              # This file
```


## Installation

Clone the repository:

```bash
git clone https://github.com/AbhayShas3/AskArXiv.git
cd AskArXiv
```

Create and activate a virtual environment:

```bash
python -m venv rag_env
source rag_env/bin/activate  
```

Install dependencies:

```bash
pip install langchain sentence-transformers faiss-cpu torch transformers gradio pandas numpy requests nltk
```

If using Ollama (What I used for this project):

- Download from https://ollama.ai
- Run:

```bash
ollama pull llama2
```

## Usage

### Data Preparation

Download an arXiv dataset (or use your own academic paper collection)  
Clean the dataset:

```bash
python -m src.data.clean --input data/raw/arxiv-metadata.json --output data/processed/cleaned_arxiv.csv --categories cs.AI cs.CL
```

### Building the Vector Database

```bash
python -m src.retrieval.build_db --input data/processed/cleaned_arxiv.csv --output data/vector_db
```

### Running the Interactive CLI

```bash
python -m src.ui.cli --vector-db data/vector_db --llm-provider ollama --ollama-model llama2 --evaluate
```

### Example Queries

Try asking:

- "How do neural networks handle ordinal regression?"
- "What techniques are used for Arabic speech recognition?"
- "How can social annotations be exploited for automatic resource discovery?"

## Evaluation

The system includes comprehensive evaluation metrics.

### Retrieval metrics

- Similarity scores
- Precision and recall (when using benchmark queries)

### Answer quality metrics

- Citation count and accuracy
- Answer length and relevance

To run a full evaluation with benchmark queries:

```bash
python -m src.evaluation.run_evaluation --vector-db data/vector_db --output evaluation_results.json
```

## Architecture Decisions

- Sentence Transformers: Used for generating embeddings due to their balance of quality and performance  
- FAISS: Chosen for efficient similarity search with fast query times  
- LLM Prompt Engineering: Designed specifically to encourage proper academic citations  
- Citation Format: Standardized as `[Authors, "Title", Year]` for academic usefulness  
