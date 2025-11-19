"""
Preprocessing module for Movie Mixer.
Handles genre and keyword cleaning, and creates combined feature fields.
"""

import pandas as pd
import ast
from typing import Any


def clean_genres(genres: Any) -> str:
    """
    Extract and clean genre names from JSON-like string format.
    
    Args:
        genres: Genre data in JSON string format or NaN
        
    Returns:
        Space-separated string of genre names
    """
    if pd.isna(genres) or genres == '[]':
        return ''
    if isinstance(genres, str):
        try:
            genre_list = ast.literal_eval(genres)
            return ' '.join([g['name'] for g in genre_list])
        except (ValueError, SyntaxError, KeyError):
            return ''
    return ''


def clean_keywords(keywords: Any) -> str:
    """
    Extract and clean keyword names from JSON-like string format.
    
    Args:
        keywords: Keyword data in JSON string format or NaN
        
    Returns:
        Space-separated string of keyword names
    """
    if pd.isna(keywords) or keywords == '[]':
        return ''
    if isinstance(keywords, str):
        try:
            kw_list = ast.literal_eval(keywords)
            return ' '.join([k['name'] for k in kw_list])
        except (ValueError, SyntaxError, KeyError):
            return ''
    return ''


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess movies DataFrame by cleaning genres and keywords,
    and creating a combined feature field for TF-IDF.
    
    Args:
        movies: Raw movies DataFrame with genres and keywords columns
        
    Returns:
        Preprocessed DataFrame with cleaned text fields and combined features
    """
    # Create cleaned genre and keyword strings
    movies['genres_str'] = movies['genres'].apply(clean_genres)
    movies['keywords_str'] = movies['keywords'].apply(clean_keywords)
    
    # Combine genres and keywords for TF-IDF vectorization
    # Note: overview is commented out but can be added back if needed
    movies['combined'] = (
        # movies['overview'].fillna('') + ' ' + 
        movies['genres_str'].fillna('') + ' ' + 
        movies['keywords_str']
    )
    
    return movies
