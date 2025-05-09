

import os
import json
import re
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime

def clean_text(text):
    """Clean text by removing special characters and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[^\w\s.,;:!?()[\]{}"\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_authors(authors_str):
    """Extract author names from the authors string."""
    if not isinstance(authors_str, str):
        return ""
    authors_str = re.sub(r'\([^)]*\)', '', authors_str)
    authors = re.split(r',\s*|\s+and\s+', authors_str)
    authors = [clean_text(author) for author in authors if author.strip()]
    return ", ".join(authors)

def extract_categories(categories_str):
    """Extract categories from the categories string."""
    if not isinstance(categories_str, str):
        return []
    categories = re.split(r'[\s,]+', categories_str)
    return [cat.strip() for cat in categories if cat.strip()]

def parse_date(date_str):
    """Parse date from various formats to YYYY-MM-DD."""
    if not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    formats = [
        '%Y-%m-%d',        
        '%a, %d %b %Y',    
        '%d %b %Y'        
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    year_match = re.search(r'\d{4}', date_str)
    if year_match:
        return year_match.group(0) + "-01-01"
    
    return None

def clean_arxiv_dataset(input_file, output_file, categories=None, min_abstract_length=100, max_papers=None):
    """
    Clean and filter the arXiv dataset.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output CSV file
        categories: List of categories to include (None for all)
        min_abstract_length: Minimum abstract length to include
        max_papers: Maximum number of papers to process (None for all)
    """
    print(f"Loading dataset from {input_file}...")
    
    if input_file.endswith('.json'):
        papers = []
        with open(input_file, 'r') as f:
            for i, line in tqdm(enumerate(f)):
                if max_papers and i >= max_papers:
                    break
                try:
                    paper = json.loads(line)
                    papers.append(paper)
                except json.JSONDecodeError:
                    continue
        df = pd.DataFrame(papers)
    elif input_file.endswith('.csv'):
        if max_papers:
            df = pd.read_csv(input_file, nrows=max_papers)
        else:
            df = pd.read_csv(input_file)
    else:
        raise ValueError("Unsupported file format. Use .json or .csv")
    
    print(f"Loaded {len(df)} papers. Processing...")
    
    processed_df = df.copy()
    
    if 'abstract' not in processed_df.columns and 'summary' in processed_df.columns:
        processed_df['abstract'] = processed_df['summary']
    if 'title' not in processed_df.columns and 'name' in processed_df.columns:
        processed_df['title'] = processed_df['name']
    
    if categories:
        if 'categories' in processed_df.columns:
            processed_df['category_list'] = processed_df['categories'].apply(extract_categories)
            category_mask = processed_df['category_list'].apply(
                lambda cats: any(cat.lower() in [c.lower() for c in cats] for cat in categories)
            )
            processed_df = processed_df[category_mask].copy()
            print(f"After category filtering: {len(processed_df)} papers")
    
    processed_df = processed_df[processed_df['abstract'].notna()].copy()
    processed_df = processed_df[processed_df['abstract'].str.len() >= min_abstract_length].copy()
    print(f"After abstract filtering: {len(processed_df)} papers")
    
    processed_df['title'] = processed_df['title'].apply(clean_text)
    processed_df['abstract'] = processed_df['abstract'].apply(clean_text)
    
    if 'authors' in processed_df.columns:
        processed_df['authors'] = processed_df['authors'].apply(extract_authors)
    elif 'author' in processed_df.columns:
        processed_df['authors'] = processed_df['author'].apply(extract_authors)
    
    if 'update_date' in processed_df.columns:
        processed_df['date'] = processed_df['update_date'].apply(parse_date)
    elif 'published' in processed_df.columns:
        processed_df['date'] = processed_df['published'].apply(parse_date)
    
    processed_df['combined_text'] = processed_df['title'] + ". " + processed_df['abstract']
    
    final_columns = ['id', 'title', 'authors', 'date', 'abstract', 'combined_text']
    final_columns = [col for col in final_columns if col in processed_df.columns]
    
    if 'id' not in processed_df.columns and 'article_id' in processed_df.columns:
        processed_df['id'] = processed_df['article_id']
    elif 'id' not in processed_df.columns:
        processed_df['id'] = [f"paper_{i}" for i in range(len(processed_df))]
        final_columns.insert(0, 'id')
    
    print(f"Saving {len(processed_df)} processed papers to {output_file}")
    processed_df[final_columns].to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and preprocess arXiv dataset")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--categories", type=str, nargs="+", help="List of categories to include")
    parser.add_argument("--min-abstract-length", type=int, default=100, help="Minimum abstract length")
    parser.add_argument("--max-papers", type=int, default=None, help="Maximum number of papers to process")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    clean_arxiv_dataset(
        args.input, 
        args.output, 
        categories=args.categories, 
        min_abstract_length=args.min_abstract_length,
        max_papers=args.max_papers
    )


