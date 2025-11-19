"""
Data loading module for Movie Mixer.
Handles CSV file loading, data cleaning, and keyword merging.
"""

import pandas as pd
from typing import Tuple


def load_movies_data(movies_file: str = "assets/movies.csv", 
                     keywords_file: str = "assets/keywords.csv",
                     min_vote_count: int = 1000) -> pd.DataFrame:
    """
    Load and clean movies and keywords data from CSV files.
    
    Args:
        movies_file: Path to the movies CSV file
        keywords_file: Path to the keywords CSV file
        min_vote_count: Minimum number of votes required for a movie to be included
        
    Returns:
        Merged and cleaned DataFrame containing movies with keywords
    """
    # Define read parameters
    movie_read_kwargs = {
        "dtype": {"popularity": "string"},
        "low_memory": False,
    }
    
    # Load movies CSV with encoding fallback
    try:
        movies = pd.read_csv(movies_file, encoding='utf-8', **movie_read_kwargs)
    except UnicodeDecodeError:
        movies = pd.read_csv(movies_file, encoding='latin1', **movie_read_kwargs)
    
    # Load keywords CSV with encoding fallback
    try:
        keywords_df = pd.read_csv(keywords_file, encoding='utf-8')
    except UnicodeDecodeError:
        keywords_df = pd.read_csv(keywords_file, encoding='latin1')
    
    # Filter movies by vote count
    movies = movies[movies['vote_count'] >= min_vote_count].copy()
    
    # Clean and convert movie IDs
    movies = _clean_ids(movies)
    keywords_df = _clean_ids(keywords_df)
    
    # Fill missing keywords with empty list
    keywords_df['keywords'] = keywords_df['keywords'].fillna('[]')
    
    # Merge keywords into movies DataFrame
    movies = movies.merge(keywords_df[['id', 'keywords']], on='id', how='left')
    movies['keywords'] = movies['keywords'].fillna('[]')
    
    return movies


def _clean_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert ID column to integer type.
    
    Args:
        df: DataFrame with an 'id' column
        
    Returns:
        DataFrame with cleaned integer IDs
    """
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id'])
    df['id'] = df['id'].astype(int)
    df = df.reset_index(drop=True)
    return df
